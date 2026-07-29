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
pub mod gf16;
pub mod gf8;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernel::scalar;

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
