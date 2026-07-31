//! x86 / `x86_64` SIMD kernels.
//!
//! Everything `unsafe` in this crate for this architecture lives under here.
//! The safe `pub(crate)` wrappers in the submodules are the boundary: the
//! caller has already selected a matching [`Backend`](crate::kernel::Backend),
//! so each wrapper re-establishes the CPU-feature proof next to the
//! intrinsic call it guards.
//!
//! Two multiply strategies:
//!
//! - **GFNI.** `GF2P8MULB` multiplies bytes in `GF(2)[x] / 0x11B` — exactly
//!   this crate's GF(2^8) — 32 lanes per instruction, no table, no shuffle
//!   port pressure. This is why the field uses the AES polynomial.
//! - **Nibble shuffle.** Without GFNI, `PSHUFB` performs a 16-entry lookup
//!   per lane, so `c * x` becomes two shuffles and an XOR against the
//!   precomputed [`ScaleTable`](crate::kernel::tables::ScaleTable).

#![allow(unsafe_code)]
#![allow(clippy::incompatible_msrv)]

pub mod avx512;
pub mod fan_paar;
pub mod gf16;
pub mod gf32;
pub mod gf64;
pub mod gf8;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar;

/// Smallest destination for which non-temporal stores pay for themselves.
///
/// `mul_into` writes a destination it never reads, so an ordinary store pays
/// a read-for-ownership fetch of every line it is about to overwrite
/// completely. `vmovntdq` skips that fetch and the allocation, at the price
/// of evicting the destination from cache.
///
/// The threshold is set by the workload this pessimizes — encode, then read
/// the destination back — because that is the case where the eviction costs
/// something. Below it the read-back loop loses; at 2 MiB it breaks even while
/// the write-only case is already several times ahead. `mul_add` and
/// `mul_assign` read their destination anyway and keep ordinary stores. The
/// measurements are under "Crossover and dispatch decisions" in
/// BENCHMARKS.md.
pub(super) const NT_STORE_MIN: usize = 2 << 20;

/// Head bytes to store normally so that a non-temporal body starts on a
/// 32-byte boundary, or `None` when this destination should stay temporal.
///
/// `None` covers a destination too small to repay the eviction, and the case
/// where the alignment peel would not be a whole number of `elem_bytes`
/// elements — a kernel may only split a buffer on an element boundary.
#[inline]
pub(super) fn nt_split(dst: &[u8], elem_bytes: usize) -> Option<usize> {
    if dst.len() < NT_STORE_MIN {
        return None;
    }
    let peel = dst.as_ptr().align_offset(32);
    (peel != usize::MAX && peel.is_multiple_of(elem_bytes)).then_some(peel)
}

/// Bytes of lead-in that put a row group's vector accesses on a 32-byte
/// boundary, or `0` when the row is too short to repay it.
///
/// A 32-byte `vmovdqu` at an odd multiple of 32 straddles two cache lines,
/// and a multi-row body issues one load and one store per row per vector: a
/// misaligned destination therefore doubles the line traffic of every access
/// but the shared source load. Peeling at most 31 bytes per row group buys
/// the aligned body for the rest of the pass; the measurements, and the
/// shapes where the same peel was tried and rejected, are in BENCHMARKS.md.
/// Rows of a group sit `row_len` apart, so aligning one aligns them all
/// whenever `row_len` is a multiple of 32 — the usual case. The source is
/// left where it falls: it is one access against many, and only the
/// destination is ours to choose.
///
/// The peel is rounded up to a whole number of `elem_bytes` elements, because
/// a kernel may only split a buffer on an element boundary. For an odd
/// destination pointer in a 2-byte field that is unreachable, and the peel is
/// skipped.
#[inline]
pub(super) fn peel_to_align(ptr: *const u8, len: usize, elem_bytes: usize) -> usize {
    // The peel runs the narrow single-row kernels, which descend to 128-bit
    // lanes and then scalar code. Keep the conservative crossover measured for
    // the former all-scalar GF(2^8) peel: it won from about 2 KiB up and lost
    // badly below 1 KiB. Sixteen turns of the 128-byte body remains the floor.
    const FLOOR: usize = 16 * 128;
    let head = ptr.align_offset(32);
    // `align_offset` reports `usize::MAX` only for an unreachable alignment,
    // which cannot happen for a byte pointer; `head >= len` also covers the
    // degenerate case of a peel longer than the row.
    if len < FLOOR || head >= len || !head.is_multiple_of(elem_bytes) {
        0
    } else {
        head
    }
}

/// Store 32 bytes, non-temporally when `NT`.
///
/// # Safety
/// `ptr` must be writable for 32 bytes, and when `NT` it must additionally be
/// 32-byte aligned. A `NT` caller must `_mm_sfence()` before any other thread
/// observes the stores.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn store256<const NT: bool>(ptr: *mut u8, value: __m256i) {
    // SAFETY: the caller guarantees writability, and alignment when `NT`.
    unsafe {
        if NT {
            _mm256_stream_si256(ptr.cast(), value);
        } else {
            _mm256_storeu_si256(ptr.cast(), value);
        }
    }
}

/// Store 16 bytes, non-temporally when `NT`.
///
/// # Safety
/// As [`store256`], for 16 bytes and 16-byte alignment.
#[inline]
#[target_feature(enable = "sse2")]
pub(super) unsafe fn store128<const NT: bool>(ptr: *mut u8, value: __m128i) {
    // SAFETY: the caller guarantees writability, and alignment when `NT`.
    unsafe {
        if NT {
            _mm_stream_si128(ptr.cast(), value);
        } else {
            _mm_storeu_si128(ptr.cast(), value);
        }
    }
}

/// `dst ^= src` using 32-byte AVX2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_avx2(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected an AVX2-capable backend, and the slices are
    // equal-length and independently borrowed.
    unsafe { xor_avx2_impl(dst, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_avx2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !31;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let s = _mm256_loadu_si256(src_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(d, s));
        }
        offset += 32;
    }
    let mut offset = len;
    if offset + 16 <= dst.len() {
        // SAFETY: `offset + 16 <= dst.len() == src.len()`, and AVX2 implies
        // SSE2 for the single narrow tail lane.
        unsafe {
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let s = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[offset..], &src[offset..]);
}

/// `dst ^= src` using 16-byte SSE2 lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor_sse2(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: SSE2 is baseline on x86_64 and implied by the SSSE3 backend on
    // x86; the slices are equal-length and independently borrowed.
    unsafe { xor_sse2_impl(dst, src) }
}

#[target_feature(enable = "sse2")]
unsafe fn xor_sse2_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let s = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[len..], &src[len..]);
}
