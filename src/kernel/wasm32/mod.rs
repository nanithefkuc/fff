//! WebAssembly `simd128` kernels.
//!
//! Wasm's lane-local `i8x16.swizzle` has the same 16-entry lookup shape as
//! SSSE3 `PSHUFB` and NEON `TBL`, so the existing nibble tables are reused
//! without conversion.

#![allow(unsafe_code)]

pub mod gf16;
pub mod gf8;

use core::arch::wasm32::*;

use crate::kernel::scalar;

/// `dst ^= src` over 16-byte Wasm SIMD lanes.
///
/// # Panics
/// If `dst` and `src` have different lengths.
pub fn xor_simd128(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: this module is reachable only in a `+simd128` build, and the
    // slices are equal-length and independently borrowed.
    unsafe { xor_impl(dst, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn xor_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = v128_load(dst_ptr.add(offset).cast());
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), v128_xor(d, s));
        }
        offset += 16;
    }
    scalar::xor(&mut dst[len..], &src[len..]);
}
