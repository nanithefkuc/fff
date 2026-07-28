//! `AArch64` NEON kernels.
//!
//! Everything `unsafe` in this crate for this architecture lives under here.
//! The safe `pub(crate)` wrappers in the submodules are the boundary.
//!
//! NEON has no byte-wide field multiply, so both fields use the split-nibble
//! strategy: `VTBL`/`vqtbl1q_u8` performs a 16-entry lookup across all 16
//! lanes, and `c * x` is two lookups and an XOR against the precomputed
//! [`ScaleTable`](crate::kernel::tables::ScaleTable).

#![allow(unsafe_code)]

pub(crate) mod gf16;
pub(crate) mod gf8;

use core::arch::aarch64::*;

use crate::kernel::scalar;

/// `dst ^= src` using 16-byte NEON lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn xor_neon(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: NEON is baseline on AArch64; the slices are equal-length and
    // independently borrowed.
    unsafe { xor_neon_impl(dst, src) }
}

#[target_feature(enable = "neon")]
unsafe fn xor_neon_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let d = vld1q_u8(dst_ptr.add(offset));
            let s = vld1q_u8(src_ptr.add(offset));
            vst1q_u8(dst_ptr.add(offset), veorq_u8(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[len..], &src[len..]);
}
