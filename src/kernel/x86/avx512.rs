//! AVX-512 GFNI kernels for x86 and `x86_64`.
//!
//! The 64-byte lanes double GFNI throughput over the AVX2 backend. The
//! multi-row shapes use the wider register file as well: scatter shares one
//! source load across eight rows, gather holds a 512-byte destination tile,
//! and matrix holds 128 bytes from each of eight rows across all terms.

use crate::field::{gf8, gf16};
use crate::kernel::tables::{TowerCoeff, scale_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const LANE: usize = 64;
const TILE_VECTORS: usize = 8;

/// `dst ^= src` over 64-byte AVX-512 lanes.
pub(crate) fn xor(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch established AVX-512F and the slices are independently
    // borrowed.
    unsafe { xor_impl(dst, src) }
}

#[target_feature(enable = "avx512f")]
unsafe fn xor_impl(dst: &mut [u8], src: &[u8]) {
    let len = dst.len().min(src.len()) & !(LANE - 1);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = _mm512_loadu_si512(dst_ptr.add(offset).cast());
            let s = _mm512_loadu_si512(src_ptr.add(offset).cast());
            _mm512_storeu_si512(dst_ptr.add(offset).cast(), _mm512_xor_si512(d, s));
        }
        offset += LANE;
    }
    crate::kernel::scalar::xor(&mut dst[len..], &src[len..]);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn swap_mask() -> __m512i {
    #[repr(align(64))]
    struct Aligned([u8; LANE]);
    static MASK: Aligned = Aligned([
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7,
        6, 9, 8, 11, 10, 13, 12, 15, 14,
    ]);
    // SAFETY: `MASK` is 64-byte aligned and exactly one vector wide.
    unsafe { _mm512_load_si512(MASK.0.as_ptr().cast()) }
}

#[inline]
fn broadcast_words(coeff: TowerCoeff) -> (i16, i16) {
    (
        i16::from_ne_bytes(coeff.same.to_ne_bytes()),
        i16::from_ne_bytes(coeff.cross.to_ne_bytes()),
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn factors16(coeff: TowerCoeff) -> (__m512i, __m512i) {
    let (same, cross) = broadcast_words(coeff);
    (_mm512_set1_epi16(same), _mm512_set1_epi16(cross))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn scale16(value: __m512i, swap: __m512i, same: __m512i, cross: __m512i) -> __m512i {
    _mm512_xor_si512(
        _mm512_gf2p8mul_epi8(value, same),
        _mm512_gf2p8mul_epi8(_mm512_shuffle_epi8(value, swap), cross),
    )
}

/// `dst ^= coeff * src` over 64 byte-wide GFNI lanes.
pub(crate) fn gf8_mul_add(dst: &mut [u8], coeff: gf8::Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf8_mul_add_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_mul_add_impl(dst: &mut [u8], coeff: gf8::Elem, src: &[u8]) {
    let len = dst.len().min(src.len()) & !(LANE - 1);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let factor = _mm512_set1_epi8(coeff.0.cast_signed());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = _mm512_loadu_si512(dst_ptr.add(offset).cast());
            let s = _mm512_loadu_si512(src_ptr.add(offset).cast());
            _mm512_storeu_si512(
                dst_ptr.add(offset).cast(),
                _mm512_xor_si512(d, _mm512_gf2p8mul_epi8(s, factor)),
            );
        }
        offset += LANE;
    }
    crate::kernel::gf8::mul_add_nibble(&mut dst[len..], scale_table(coeff), &src[len..]);
}

/// `dst = coeff * dst` over 64 byte-wide GFNI lanes.
pub(crate) fn gf8_mul_assign(dst: &mut [u8], coeff: gf8::Elem) {
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf8_mul_assign_impl(dst, coeff) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_mul_assign_impl(dst: &mut [u8], coeff: gf8::Elem) {
    let len = dst.len() & !(LANE - 1);
    let ptr = dst.as_mut_ptr();
    let factor = _mm512_set1_epi8(coeff.0.cast_signed());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in `dst`.
        unsafe {
            let d = _mm512_loadu_si512(ptr.add(offset).cast());
            _mm512_storeu_si512(ptr.add(offset).cast(), _mm512_gf2p8mul_epi8(d, factor));
        }
        offset += LANE;
    }
    crate::kernel::gf8::mul_assign_nibble(&mut dst[len..], scale_table(coeff));
}

/// `dst = coeff * src` over 64 byte-wide GFNI lanes, out of place.
///
/// Fused form of copy-then-scale: one pass, `dst` is never read.
pub(crate) fn gf8_mul_into(dst: &mut [u8], coeff: gf8::Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf8_mul_into_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_mul_into_impl(dst: &mut [u8], coeff: gf8::Elem, src: &[u8]) {
    let len = dst.len().min(src.len()) & !(LANE - 1);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let factor = _mm512_set1_epi8(coeff.0.cast_signed());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let s = _mm512_loadu_si512(src_ptr.add(offset).cast());
            _mm512_storeu_si512(dst_ptr.add(offset).cast(), _mm512_gf2p8mul_epi8(s, factor));
        }
        offset += LANE;
    }
    crate::kernel::gf8::mul_into_nibble(&mut dst[len..], scale_table(coeff), &src[len..]);
}

/// `dst[i] = a[i] * b[i]` over 64 byte-wide GFNI lanes.
pub(crate) fn gf8_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf8_elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !(LANE - 1);
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = _mm512_loadu_si512(a_ptr.add(offset).cast());
            let y = _mm512_loadu_si512(b_ptr.add(offset).cast());
            _mm512_storeu_si512(dst_ptr.add(offset).cast(), _mm512_gf2p8mul_epi8(x, y));
        }
        offset += LANE;
    }
    crate::kernel::scalar::mul_elementwise::<gf8::Gf8>(&mut dst[len..], &a[len..], &b[len..]);
}

/// `dst ^= coeff * src` over 64-byte tower-field lanes.
pub(crate) fn gf16_mul_add(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf16_mul_add_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_mul_add_impl(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let len = dst.len().min(src.len()) & !(LANE - 1);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let swap = swap_mask();
    let (same, cross) = factors16(coeff);
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = _mm512_loadu_si512(dst_ptr.add(offset).cast());
            let s = _mm512_loadu_si512(src_ptr.add(offset).cast());
            _mm512_storeu_si512(
                dst_ptr.add(offset).cast(),
                _mm512_xor_si512(d, scale16(s, swap, same, cross)),
            );
        }
        offset += LANE;
    }
    crate::kernel::gf16::mul_add_scalar(&mut dst[len..], coeff.coeff, &src[len..]);
}

/// `dst = coeff * dst` over 64-byte tower-field lanes.
pub(crate) fn gf16_mul_assign(dst: &mut [u8], coeff: TowerCoeff) {
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf16_mul_assign_impl(dst, coeff) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_mul_assign_impl(dst: &mut [u8], coeff: TowerCoeff) {
    let len = dst.len() & !(LANE - 1);
    let ptr = dst.as_mut_ptr();
    let swap = swap_mask();
    let (same, cross) = factors16(coeff);
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in `dst`.
        unsafe {
            let d = _mm512_loadu_si512(ptr.add(offset).cast());
            _mm512_storeu_si512(ptr.add(offset).cast(), scale16(d, swap, same, cross));
        }
        offset += LANE;
    }
    crate::kernel::gf16::mul_assign_scalar(&mut dst[len..], coeff.coeff);
}

/// `dst = coeff * src` over 64-byte tower-field lanes, out of place.
///
/// Fused form of copy-then-scale: one pass, `dst` is never read.
pub(crate) fn gf16_mul_into(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf16_mul_into_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_mul_into_impl(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let len = dst.len().min(src.len()) & !(LANE - 1);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let swap = swap_mask();
    let (same, cross) = factors16(coeff);
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let s = _mm512_loadu_si512(src_ptr.add(offset).cast());
            _mm512_storeu_si512(dst_ptr.add(offset).cast(), scale16(s, swap, same, cross));
        }
        offset += LANE;
    }
    crate::kernel::gf16::mul_into_scalar(&mut dst[len..], coeff.coeff, &src[len..]);
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements.
pub(crate) fn gf16_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf16_elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !(LANE - 1);
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let swap = swap_mask();
    let even = _mm512_set1_epi16(0x00ff);
    let delta_even = _mm512_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = _mm512_loadu_si512(a_ptr.add(offset).cast());
            let y = _mm512_loadu_si512(b_ptr.add(offset).cast());
            let direct = _mm512_gf2p8mul_epi8(x, y);
            let crossed = _mm512_gf2p8mul_epi8(x, _mm512_shuffle_epi8(y, swap));
            let delta_bd = _mm512_gf2p8mul_epi8(_mm512_shuffle_epi8(direct, swap), delta_even);
            let constant = _mm512_xor_si512(direct, delta_bd);
            let cross_sum = _mm512_xor_si512(crossed, _mm512_shuffle_epi8(crossed, swap));
            let extension = _mm512_xor_si512(cross_sum, direct);
            let product = _mm512_xor_si512(
                _mm512_and_si512(constant, even),
                _mm512_andnot_si512(even, extension),
            );
            _mm512_storeu_si512(dst_ptr.add(offset).cast(), product);
        }
        offset += LANE;
    }
    crate::kernel::scalar::mul_elementwise::<gf16::Gf16>(&mut dst[len..], &a[len..], &b[len..]);
}

/// One GF(2^8) source into many rows, sharing each load across eight rows.
pub(crate) fn gf8_scatter(rows: &mut [u8], row_len: usize, coeffs: &[gf8::Elem], src: &[u8]) {
    debug_assert_eq!(rows.len(), row_len * coeffs.len());
    debug_assert_eq!(src.len(), row_len);
    // SAFETY: dispatch established the features and rows are disjoint by construction.
    unsafe { gf8_scatter_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[gf8::Elem], src: &[u8]) {
    let mut row = 0;
    while row + 8 <= coeffs.len() {
        // SAFETY: this group addresses eight complete, disjoint rows.
        unsafe {
            gf8_scatter_group::<8>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                &coeffs[row..row + 8],
                src,
            );
        };
        row += 8;
    }
    while row < coeffs.len() {
        // SAFETY: this group addresses one complete row.
        unsafe {
            gf8_scatter_group::<1>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                &coeffs[row..=row],
                src,
            );
        };
        row += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_scatter_group<const N: usize>(
    base: *mut u8,
    stride: usize,
    coeffs: &[gf8::Elem],
    src: &[u8],
) {
    let len = src.len().min(stride) & !(LANE - 1);
    let factors: [__m512i; N] =
        core::array::from_fn(|i| _mm512_set1_epi8(coeffs[i].0.cast_signed()));
    let mut offset = 0;
    while offset < len {
        // SAFETY: the source and every disjoint row have a complete vector.
        unsafe {
            let s = _mm512_loadu_si512(src.as_ptr().add(offset).cast());
            for (i, factor) in factors.iter().enumerate() {
                let ptr = base.add(i * stride + offset);
                let d = _mm512_loadu_si512(ptr.cast());
                _mm512_storeu_si512(
                    ptr.cast(),
                    _mm512_xor_si512(d, _mm512_gf2p8mul_epi8(s, *factor)),
                );
            }
        }
        offset += LANE;
    }
    for (i, &coeff) in coeffs.iter().enumerate().take(N) {
        // SAFETY: the tail is contained in row `i`, and groups own disjoint rows.
        let tail =
            unsafe { core::slice::from_raw_parts_mut(base.add(i * stride + len), stride - len) };
        crate::kernel::gf8::mul_add_nibble(tail, scale_table(coeff), &src[len..]);
    }
}

/// Many GF(2^8) sources into one destination, blocked over 512-byte tiles.
pub(crate) fn gf8_gather(dst: &mut [u8], coeffs: &[gf8::Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf8_gather_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_gather_impl(dst: &mut [u8], coeffs: &[gf8::Elem], srcs: &[&[u8]]) {
    let tile = LANE * TILE_VECTORS;
    let len = dst.len() / tile * tile;
    let ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete tile remains in `dst` and every validated source.
        unsafe {
            let mut acc: [__m512i; TILE_VECTORS] =
                core::array::from_fn(|v| _mm512_loadu_si512(ptr.add(offset + v * LANE).cast()));
            for (&coeff, src) in coeffs.iter().zip(srcs) {
                let factor = _mm512_set1_epi8(coeff.0.cast_signed());
                for (v, d) in acc.iter_mut().enumerate() {
                    let s = _mm512_loadu_si512(src.as_ptr().add(offset + v * LANE).cast());
                    *d = _mm512_xor_si512(*d, _mm512_gf2p8mul_epi8(s, factor));
                }
            }
            for (v, d) in acc.iter().enumerate() {
                _mm512_storeu_si512(ptr.add(offset + v * LANE).cast(), *d);
            }
        }
        offset += tile;
    }
    for (&coeff, src) in coeffs.iter().zip(srcs) {
        crate::kernel::gf8::mul_add_nibble(&mut dst[len..], scale_table(coeff), &src[len..]);
    }
}

/// Many sources into many GF(2^8) rows, eight rows by 128 bytes per tile.
pub(crate) fn gf8_matrix(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf8::Elem], &[u8])],
) {
    // SAFETY: public wrappers validated all geometry and dispatch established the features.
    unsafe { gf8_matrix_impl(rows, row_len, nrows, terms) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_matrix_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf8::Elem], &[u8])],
) {
    let mut row = 0;
    while row + 8 <= nrows {
        // SAFETY: the group owns eight complete, disjoint rows.
        unsafe { gf8_matrix_group::<8>(rows.as_mut_ptr().add(row * row_len), row_len, row, terms) };
        row += 8;
    }
    while row < nrows {
        // SAFETY: the group owns one complete row.
        unsafe { gf8_matrix_group::<1>(rows.as_mut_ptr().add(row * row_len), row_len, row, terms) };
        row += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf8_matrix_group<const N: usize>(
    base: *mut u8,
    stride: usize,
    first: usize,
    terms: &[(&[gf8::Elem], &[u8])],
) {
    let tile = 2 * LANE;
    let len = stride / tile * tile;
    let mut offset = 0;
    while offset < len {
        // SAFETY: two complete vectors remain in every source and group row.
        unsafe {
            let mut lo: [__m512i; N] =
                core::array::from_fn(|i| _mm512_loadu_si512(base.add(i * stride + offset).cast()));
            let mut hi: [__m512i; N] = core::array::from_fn(|i| {
                _mm512_loadu_si512(base.add(i * stride + offset + LANE).cast())
            });
            for &(coeffs, src) in terms {
                let s0 = _mm512_loadu_si512(src.as_ptr().add(offset).cast());
                let s1 = _mm512_loadu_si512(src.as_ptr().add(offset + LANE).cast());
                for i in 0..N {
                    let factor = _mm512_set1_epi8(coeffs[first + i].0.cast_signed());
                    lo[i] = _mm512_xor_si512(lo[i], _mm512_gf2p8mul_epi8(s0, factor));
                    hi[i] = _mm512_xor_si512(hi[i], _mm512_gf2p8mul_epi8(s1, factor));
                }
            }
            for i in 0..N {
                _mm512_storeu_si512(base.add(i * stride + offset).cast(), lo[i]);
                _mm512_storeu_si512(base.add(i * stride + offset + LANE).cast(), hi[i]);
            }
        }
        offset += tile;
    }
    for i in 0..N {
        // SAFETY: the tail lies within this group's uniquely owned row.
        let tail =
            unsafe { core::slice::from_raw_parts_mut(base.add(i * stride + len), stride - len) };
        for &(coeffs, src) in terms {
            crate::kernel::gf8::mul_add_nibble(tail, scale_table(coeffs[first + i]), &src[len..]);
        }
    }
}

/// One tower-field source into many rows, sharing each load across eight rows.
pub(crate) fn gf16_scatter(rows: &mut [u8], row_len: usize, coeffs: &[gf16::Elem], src: &[u8]) {
    debug_assert_eq!(rows.len(), row_len * coeffs.len());
    debug_assert_eq!(src.len(), row_len);
    // SAFETY: dispatch established the features and rows are disjoint by construction.
    unsafe { gf16_scatter_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[gf16::Elem], src: &[u8]) {
    let swap = swap_mask();
    let mut row = 0;
    while row + 8 <= coeffs.len() {
        // SAFETY: this group addresses eight complete, disjoint rows.
        unsafe {
            gf16_scatter_group::<8>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                &coeffs[row..row + 8],
                src,
                swap,
            );
        };
        row += 8;
    }
    while row < coeffs.len() {
        // SAFETY: this group addresses one complete row.
        unsafe {
            gf16_scatter_group::<1>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                &coeffs[row..=row],
                src,
                swap,
            );
        };
        row += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_scatter_group<const N: usize>(
    base: *mut u8,
    stride: usize,
    coeffs: &[gf16::Elem],
    src: &[u8],
    swap: __m512i,
) {
    let len = src.len().min(stride) & !(LANE - 1);
    let factors: [(__m512i, __m512i); N] =
        core::array::from_fn(|i| factors16(TowerCoeff::new(coeffs[i])));
    let mut offset = 0;
    while offset < len {
        // SAFETY: the source and every disjoint row have a complete vector.
        unsafe {
            let s = _mm512_loadu_si512(src.as_ptr().add(offset).cast());
            for (i, &(same, cross)) in factors.iter().enumerate() {
                let ptr = base.add(i * stride + offset);
                let d = _mm512_loadu_si512(ptr.cast());
                _mm512_storeu_si512(
                    ptr.cast(),
                    _mm512_xor_si512(d, scale16(s, swap, same, cross)),
                );
            }
        }
        offset += LANE;
    }
    for (i, &coeff) in coeffs.iter().enumerate().take(N) {
        // SAFETY: the tail is contained in row `i`, and groups own disjoint rows.
        let tail =
            unsafe { core::slice::from_raw_parts_mut(base.add(i * stride + len), stride - len) };
        crate::kernel::gf16::mul_add_scalar(tail, coeff, &src[len..]);
    }
}

/// Many tower-field sources into one destination, blocked over 512-byte tiles.
pub(crate) fn gf16_gather(dst: &mut [u8], coeffs: &[gf16::Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: dispatch established AVX-512F, AVX-512BW, and GFNI.
    unsafe { gf16_gather_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_gather_impl(dst: &mut [u8], coeffs: &[gf16::Elem], srcs: &[&[u8]]) {
    let tile = LANE * TILE_VECTORS;
    let len = dst.len() / tile * tile;
    let ptr = dst.as_mut_ptr();
    let swap = swap_mask();
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete tile remains in `dst` and every validated source.
        unsafe {
            let mut acc: [__m512i; TILE_VECTORS] =
                core::array::from_fn(|v| _mm512_loadu_si512(ptr.add(offset + v * LANE).cast()));
            for (&coeff, src) in coeffs.iter().zip(srcs) {
                let (same, cross) = factors16(TowerCoeff::new(coeff));
                for (v, d) in acc.iter_mut().enumerate() {
                    let s = _mm512_loadu_si512(src.as_ptr().add(offset + v * LANE).cast());
                    *d = _mm512_xor_si512(*d, scale16(s, swap, same, cross));
                }
            }
            for (v, d) in acc.iter().enumerate() {
                _mm512_storeu_si512(ptr.add(offset + v * LANE).cast(), *d);
            }
        }
        offset += tile;
    }
    for (&coeff, src) in coeffs.iter().zip(srcs) {
        crate::kernel::gf16::mul_add_scalar(&mut dst[len..], coeff, &src[len..]);
    }
}

/// Many sources into many tower-field rows, eight rows by 128 bytes per tile.
pub(crate) fn gf16_matrix(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf16::Elem], &[u8])],
) {
    // SAFETY: public wrappers validated all geometry and dispatch established the features.
    unsafe { gf16_matrix_impl(rows, row_len, nrows, terms) }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_matrix_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[gf16::Elem], &[u8])],
) {
    let swap = swap_mask();
    let mut row = 0;
    while row + 8 <= nrows {
        // SAFETY: the group owns eight complete, disjoint rows.
        unsafe {
            gf16_matrix_group::<8>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                row,
                terms,
                swap,
            );
        };
        row += 8;
    }
    while row < nrows {
        // SAFETY: the group owns one complete row.
        unsafe {
            gf16_matrix_group::<1>(
                rows.as_mut_ptr().add(row * row_len),
                row_len,
                row,
                terms,
                swap,
            );
        };
        row += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
unsafe fn gf16_matrix_group<const N: usize>(
    base: *mut u8,
    stride: usize,
    first: usize,
    terms: &[(&[gf16::Elem], &[u8])],
    swap: __m512i,
) {
    let tile = 2 * LANE;
    let len = stride / tile * tile;
    let mut offset = 0;
    while offset < len {
        // SAFETY: two complete vectors remain in every source and group row.
        unsafe {
            let mut lo: [__m512i; N] =
                core::array::from_fn(|i| _mm512_loadu_si512(base.add(i * stride + offset).cast()));
            let mut hi: [__m512i; N] = core::array::from_fn(|i| {
                _mm512_loadu_si512(base.add(i * stride + offset + LANE).cast())
            });
            for &(coeffs, src) in terms {
                let s0 = _mm512_loadu_si512(src.as_ptr().add(offset).cast());
                let s1 = _mm512_loadu_si512(src.as_ptr().add(offset + LANE).cast());
                for i in 0..N {
                    let (same, cross) = factors16(TowerCoeff::new(coeffs[first + i]));
                    lo[i] = _mm512_xor_si512(lo[i], scale16(s0, swap, same, cross));
                    hi[i] = _mm512_xor_si512(hi[i], scale16(s1, swap, same, cross));
                }
            }
            for i in 0..N {
                _mm512_storeu_si512(base.add(i * stride + offset).cast(), lo[i]);
                _mm512_storeu_si512(base.add(i * stride + offset + LANE).cast(), hi[i]);
            }
        }
        offset += tile;
    }
    for i in 0..N {
        // SAFETY: the tail lies within this group's uniquely owned row.
        let tail =
            unsafe { core::slice::from_raw_parts_mut(base.add(i * stride + len), stride - len) };
        for &(coeffs, src) in terms {
            crate::kernel::gf16::mul_add_scalar(tail, coeffs[first + i], &src[len..]);
        }
    }
}
