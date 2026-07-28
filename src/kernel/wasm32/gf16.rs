//! GF(2^16) tower kernels using WebAssembly `simd128` swizzles.

use core::arch::wasm32::*;

use crate::field::gf16::{Elem, Gf16};
use crate::kernel::tables::TowerTables;
use crate::kernel::wasm32::gf8::multiply_vectors;

#[derive(Clone, Copy)]
struct Factors {
    lo: [v128; 4],
    hi: [v128; 4],
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_factors(tables: &TowerTables) -> Factors {
    let factors = &tables.factors;
    // SAFETY: every table half contains exactly 16 readable bytes.
    unsafe {
        Factors {
            lo: core::array::from_fn(|i| v128_load(factors[i].lo.as_ptr().cast())),
            hi: core::array::from_fn(|i| v128_load(factors[i].hi.as_ptr().cast())),
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn swap_adjacent(value: v128) -> v128 {
    const SWAP: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];
    // SAFETY: `SWAP` contains exactly 16 readable bytes.
    let mask = unsafe { v128_load(SWAP.as_ptr().cast()) };
    u8x16_swizzle(value, mask)
}

#[inline]
#[target_feature(enable = "simd128")]
fn lookup(lo: v128, hi: v128, low: v128, high: v128) -> v128 {
    v128_xor(u8x16_swizzle(lo, low), u8x16_swizzle(hi, high))
}

#[inline]
#[target_feature(enable = "simd128")]
fn scaled(value: v128, factors: Factors) -> v128 {
    let nibble = u8x16_splat(0x0f);
    let swapped = swap_adjacent(value);
    let low = v128_and(value, nibble);
    let high = u8x16_shr(value, 4);
    let swapped_low = v128_and(swapped, nibble);
    let swapped_high = u8x16_shr(swapped, 4);
    let direct_even = lookup(factors.lo[0], factors.hi[0], low, high);
    let direct_odd = lookup(factors.lo[1], factors.hi[1], low, high);
    let cross_even = lookup(factors.lo[2], factors.hi[2], swapped_low, swapped_high);
    let cross_odd = lookup(factors.lo[3], factors.hi[3], swapped_low, swapped_high);
    let even_lanes = u16x8_splat(0x00ff);
    v128_bitselect(
        v128_xor(direct_even, cross_even),
        v128_xor(direct_odd, cross_odd),
        even_lanes,
    )
}

/// `dst ^= coeff * src` over interleaved tower elements.
pub(crate) fn mul_add_simd128(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_add_impl(dst, tables, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_add_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let len = dst.len().min(src.len()) & !15;
    let factors = load_factors(tables);
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
    crate::kernel::gf16::mul_add_scalar(&mut dst[len..], tables.coeff, &src[len..]);
}

/// `dst = coeff * dst` over interleaved tower elements.
pub(crate) fn mul_assign_simd128(dst: &mut [u8], tables: &TowerTables) {
    // SAFETY: the binary requires `simd128`.
    unsafe { mul_assign_impl(dst, tables) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_assign_impl(dst: &mut [u8], tables: &TowerTables) {
    let len = dst.len() & !15;
    let factors = load_factors(tables);
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
    crate::kernel::gf16::mul_assign_scalar(&mut dst[len..], tables.coeff);
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements.
pub(crate) fn elementwise_simd128(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the binary requires `simd128`, and all geometry was validated.
    unsafe { elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "simd128")]
unsafe fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let even = u16x8_splat(0x00ff);
    let delta_even = u16x8_splat(u16::from_le_bytes([crate::field::gf16::DELTA.0, 0]));
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = v128_load(a_ptr.add(offset).cast());
            let y = v128_load(b_ptr.add(offset).cast());
            let direct = multiply_vectors(x, y);
            let crossed = multiply_vectors(x, swap_adjacent(y));
            let delta_bd = multiply_vectors(swap_adjacent(direct), delta_even);
            let constant = v128_xor(direct, delta_bd);
            let extension = v128_xor(v128_xor(crossed, swap_adjacent(crossed)), direct);
            v128_store(
                dst_ptr.add(offset).cast(),
                v128_bitselect(constant, extension, even),
            );
        }
        offset += 16;
    }
    crate::kernel::scalar::mul_elementwise::<Gf16>(&mut dst[len..], &a[len..], &b[len..]);
}

/// One source into many rows using SIMD tower products.
pub(crate) fn scatter_simd128(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        mul_add_simd128(row, &TowerTables::new(coeff), src);
    }
}

/// Many sources into one destination using SIMD tower products.
pub(crate) fn gather_simd128(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    for (&coeff, src) in coeffs.iter().zip(srcs) {
        mul_add_simd128(dst, &TowerTables::new(coeff), src);
    }
}

/// Many sources into many rows using the SIMD single-row primitive.
pub(crate) fn matrix_simd128(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in rows[..nrows * row_len]
            .chunks_exact_mut(row_len)
            .zip(coeffs)
        {
            mul_add_simd128(row, &TowerTables::new(coeff), src);
        }
    }
}
