//! GF(2^8) kernels using WebAssembly `simd128` swizzles.

use core::arch::wasm32::*;

use crate::field::gf8::Gf8;
use crate::kernel::tables::ScaleTable;

#[derive(Clone, Copy)]
struct Factors {
    lo: v128,
    hi: v128,
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_factors(table: &ScaleTable) -> Factors {
    // SAFETY: both arrays contain exactly 16 readable bytes.
    unsafe {
        Factors {
            lo: v128_load(table.lo.as_ptr().cast()),
            hi: v128_load(table.hi.as_ptr().cast()),
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn scaled(value: v128, factors: Factors) -> v128 {
    let low = v128_and(value, u8x16_splat(0x0f));
    let high = u8x16_shr(value, 4);
    v128_xor(
        u8x16_swizzle(factors.lo, low),
        u8x16_swizzle(factors.hi, high),
    )
}

/// `dst ^= coeff * src` over 16-byte SIMD lanes.
pub fn mul_add_simd128(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_add_impl(dst, table, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_add_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len()) & !15;
    let factors = load_factors(table);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = v128_load(dst_ptr.add(offset).cast());
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), v128_xor(d, scaled(s, factors)));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_add_nibble(&mut dst[len..], table, &src[len..]);
}

/// `dst = coeff * dst` over 16-byte SIMD lanes.
pub fn mul_assign_simd128(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: the binary requires `simd128`.
    unsafe { mul_assign_impl(dst, table) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_assign_impl(dst: &mut [u8], table: &ScaleTable) {
    let len = dst.len() & !15;
    let factors = load_factors(table);
    let ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in `dst`.
        unsafe {
            let d = v128_load(ptr.add(offset).cast());
            v128_store(ptr.add(offset).cast(), scaled(d, factors));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_assign_nibble(&mut dst[len..], table);
}

/// `dst = coeff * src`, out of place, over 16-byte SIMD lanes.
///
/// Fuses what would otherwise be a copy then an in-place scale: one pass, no `dst` read.
pub fn mul_into_simd128(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_into_impl(dst, table, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_into_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len()) & !15;
    let factors = load_factors(table);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), scaled(s, factors));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_into_nibble(&mut dst[len..], table, &src[len..]);
}

/// Lane-parallel multiply for two varying base-field vectors.
#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn multiply_vectors(mut a: v128, mut b: v128) -> v128 {
    let one = u8x16_splat(1);
    let reduction = u8x16_splat(0x1b);
    let mut product = u8x16_splat(0);
    for _ in 0..8 {
        let selected = u8x16_eq(v128_and(b, one), one);
        product = v128_xor(product, v128_and(a, selected));
        let high = u8x16_eq(u8x16_shr(a, 7), one);
        a = v128_xor(u8x16_shl(a, 1), v128_and(reduction, high));
        b = u8x16_shr(b, 1);
    }
    product
}

/// `dst[i] = a[i] * b[i]` over 16-byte SIMD lanes.
pub fn elementwise_simd128(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the binary requires `simd128`, and all geometry was validated.
    unsafe { elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "simd128")]
unsafe fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = v128_load(a_ptr.add(offset).cast());
            let y = v128_load(b_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), multiply_vectors(x, y));
        }
        offset += 16;
    }
    crate::kernel::scalar::mul_elementwise::<Gf8>(&mut dst[len..], &a[len..], &b[len..]);
}
