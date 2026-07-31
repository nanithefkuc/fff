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
/// Measured on a Core Ultra 7 258V (12 MiB L3), one core, gf8 `mul_into`
/// stock vs. non-temporal: 1 MiB 32.3 → 61.2 GiB/s, 2 MiB 21.3 → 66.0,
/// 4 MiB 21.5 → 37.8, 16 MiB 16.1 → 34.4, 64 MiB 14.2 → 23.0. The forced
/// `avx2` and `ssse3` arms sit at the same ~14 GiB/s ceiling at 16 MiB, so
/// every backend is store-bound there, not multiply-bound.
///
/// The threshold guards the workload this pessimizes — encode, then read the
/// destination back. Timing both in one loop, non-temporal stores lose only
/// while the destination fits a cache that would have kept it: 1 MiB
/// 21.6 → 17.9 GiB/s, but 2 MiB 16.5 → 16.9, 4 MiB 14.5 → 16.1, 16 MiB
/// 10.7 → 12.9. 2 MiB is where the read-back case stops losing and the
/// write-only case is already 3x.
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
