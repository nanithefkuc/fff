//! GF(2^8) kernel dispatch.
//!
//! Owns the runtime backend selection for the field and provides the nibble
//! scalar path that every vector loop uses for its sub-lane tail.
//!
//! The prepared form of a coefficient is simply a borrow of its entry in the
//! shared nibble table bank. Every backend can use it: the shuffle backends
//! index the tables directly, and GFNI reads back the coefficient byte to
//! broadcast. Preparation is therefore a single array index — unlike
//! GF(2^16), this field has nothing worth caching beyond what is already
//! precomputed in rodata.

use crate::field::gf8::{Elem, Gf8};
use crate::kernel::tables::{ScaleTable, scale_table};
// `Backend` is referenced only from the SIMD dispatch arms, which cfg away
// entirely on a scalar-only build.
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, scalar};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::kernel::aarch64;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
use crate::kernel::wasm32;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

/// `dst ^= coeff * src` via split-nibble lookup.
///
/// Two table reads and two XORs per byte, versus a log/antilog pair with a
/// modular reduction. Also the tail handler for every vector kernel, which is
/// why it takes the already-built table rather than a coefficient.
#[cfg(feature = "internals")]
pub fn mul_add_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_add_nibble_impl(dst, table, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_add_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_add_nibble_impl(dst, table, src);
}

fn mul_add_nibble_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

/// `dst *= coeff` via split-nibble lookup.
#[cfg(feature = "internals")]
pub fn mul_assign_nibble(dst: &mut [u8], table: &ScaleTable) {
    mul_assign_nibble_impl(dst, table);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_assign_nibble(dst: &mut [u8], table: &ScaleTable) {
    mul_assign_nibble_impl(dst, table);
}

fn mul_assign_nibble_impl(dst: &mut [u8], table: &ScaleTable) {
    for d in dst.iter_mut() {
        *d = table.lo[(*d & 0x0f) as usize] ^ table.hi[(*d >> 4) as usize];
    }
}

/// `dst = coeff * src` via split-nibble lookup.
///
/// Tail handler for the fused out-of-place kernels, mirroring
/// [`mul_add_nibble`].
#[cfg(feature = "internals")]
pub fn mul_into_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_into_nibble_impl(dst, table, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_into_nibble(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    mul_into_nibble_impl(dst, table, src);
}

fn mul_into_nibble_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = table.lo[(s & 0x0f) as usize] ^ table.hi[(s >> 4) as usize];
    }
}

impl FieldKernels for Gf8 {
    type Prepared = &'static ScaleTable;

    #[inline]
    fn prepare(coeff: Elem) -> Self::Prepared {
        scale_table(coeff)
    }

    #[inline]
    fn prepared_coeff(prepared: &Self::Prepared) -> Elem {
        prepared.coeff
    }
    #[inline]
    fn active_backend() -> Backend {
        backend()
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        matches!(
            backend(),
            Backend::Avx512 | Backend::Gfni | Backend::Neon | Backend::Simd128
        )
    }

    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_mul_add(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::mul_add_gfni(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::mul_add_avx2(dst, coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::mul_add_ssse3(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::mul_add_neon(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::mul_add_simd128(dst, coeff, src),
            _ => mul_add_nibble(dst, coeff, src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_mul_assign(dst, coeff.coeff),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::mul_assign_gfni(dst, coeff.coeff),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::mul_assign_avx2(dst, coeff),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::mul_assign_ssse3(dst, coeff),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::mul_assign_neon(dst, coeff),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::mul_assign_simd128(dst, coeff),
            _ => mul_assign_nibble(dst, coeff),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_mul_into(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::mul_into_gfni(dst, coeff.coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::mul_into_avx2(dst, coeff, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::mul_into_ssse3(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::mul_into_neon(dst, coeff, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::mul_into_simd128(dst, coeff, src),
            _ => mul_into_nibble(dst, coeff, src),
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_scatter(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::scatter_gfni(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::scatter_avx2(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::scatter_ssse3(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::scatter_neon(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::scatter_simd128(rows, row_len, coeffs, src),
            _ => scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src),
        }
    }
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem],
        _coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter(rows, row_len, values, src);
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_gather(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::gather_gfni(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::gather_avx2(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::gather_ssse3(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::gather_neon(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::gather_simd128(dst, coeffs, srcs),
            _ => scalar::mul_add_gather::<Self>(dst, coeffs, srcs),
        }
    }
    fn mul_add_gather_plan(
        dst: &mut [u8],
        values: &[Elem],
        _coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather(dst, values, srcs);
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_matrix(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::matrix_gfni(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf8::matrix_avx2(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf8::matrix_ssse3(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::matrix_neon(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::matrix_simd128(rows, row_len, nrows, terms),
            _ => scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms),
        }
    }
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = values;
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let terms = crate::kernel::FlatMatrix {
                coefficients: values,
                nrows,
                sources: srcs,
            };
            match backend() {
                Backend::Gfni => {
                    return x86::gf8::matrix_gfni_with(rows, row_len, nrows, &terms);
                }
                Backend::Avx2 => {
                    return x86::gf8::matrix_avx2_with(rows, row_len, nrows, &terms);
                }
                Backend::Ssse3 => {
                    return x86::gf8::matrix_ssse3_with(rows, row_len, nrows, &terms);
                }
                _ => {}
            }
        }
        for (term, &src) in srcs.iter().enumerate() {
            let start = term * nrows;
            for (row, coeff) in rows
                .chunks_exact_mut(row_len)
                .take(nrows)
                .zip(&coeffs[start..start + nrows])
            {
                Self::mul_add(row, coeff, src);
            }
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf8_elementwise(dst, a, b),
            // `GF2P8MULB` multiplies two vectors directly: no broadcast, no
            // table, the same one instruction per 32 lanes.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf8::elementwise_gfni(dst, a, b),
            // Prefer PMULL for varying operands when the optional crypto
            // extension is present; baseline NEON keeps the bit-serial path.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon if std::arch::is_aarch64_feature_detected!("aes") => {
                aarch64::gf8::elementwise_pmull(dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf8::elementwise_neon(dst, a, b),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf8::elementwise_simd128(dst, a, b),
            // A nibble table is indexed by one varying operand against one
            // *fixed* coefficient. With both operands varying there is no
            // table to build, so the shuffle backends have nothing to offer.
            _ => scalar::mul_elementwise::<Gf8>(dst, a, b),
        }
    }
}
