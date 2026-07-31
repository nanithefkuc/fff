//! Kernel dispatch for the canonical Fan–Paar fields.
//!
//! [`FanPaar16`] has a hand-written AVX2/SSSE3 kernel over the `fp8`
//! nibble-table tower (the same four-shuffle shape as the polynomial
//! GF(2^16) kernel, with `fp8` arithmetic in the tables). Every other
//! Fan–Paar field and every other backend uses the portable scalar kernel,
//! which is also the differential oracle. As with the polynomial towers, the
//! [`crate::kernel::FieldKernels`] defaults compose every multi-row and
//! prepared operation from `mul_add`, so the win reaches scatter/gather/
//! matrix through a prepared [`crate::ops::Plan`] without a per-shape kernel.

use crate::field::fan_paar::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, fp16, fp32, fp64};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, scalar};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

crate::kernel::scalar::impl_field_kernels!(FanPaar8);

impl FieldKernels for FanPaar32 {
    type Prepared = fp32::Elem;

    #[inline]
    fn prepare(coeff: fp32::Elem) -> Self::Prepared {
        coeff
    }

    #[inline]
    fn prepared_coeff(prepared: &Self::Prepared) -> fp32::Elem {
        *prepared
    }

    #[inline]
    fn active_backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }

    #[inline]
    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_add_fp32_avx2(dst, *coeff, src);
            }
            _ => scalar::mul_add::<FanPaar32>(dst, *coeff, src),
        }
    }

    #[inline]
    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_assign_fp32_avx2(dst, *coeff);
            }
            _ => scalar::mul_assign::<FanPaar32>(dst, *coeff),
        }
    }

    #[inline]
    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_into_fp32_avx2(dst, *coeff, src);
            }
            _ => {
                dst.copy_from_slice(src);
                scalar::mul_assign::<FanPaar32>(dst, *coeff);
            }
        }
    }

    #[inline]
    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[fp32::Elem], src: &[u8]) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &Self::prepare(coeff), src);
        }
    }

    #[inline]
    fn mul_add_gather(dst: &mut [u8], coeffs: &[fp32::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }

    #[inline]
    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp32::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &Self::prepare(coeff), src);
            }
        }
    }

    #[inline]
    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        scalar::mul_elementwise::<FanPaar32>(dst, a, b);
    }
}

impl FieldKernels for FanPaar64 {
    type Prepared = fp64::Elem;

    #[inline]
    fn prepare(coeff: fp64::Elem) -> Self::Prepared {
        coeff
    }

    #[inline]
    fn prepared_coeff(prepared: &Self::Prepared) -> fp64::Elem {
        *prepared
    }

    #[inline]
    fn active_backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }

    #[inline]
    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_add_fp64_avx2(dst, *coeff, src);
            }
            _ => scalar::mul_add::<FanPaar64>(dst, *coeff, src),
        }
    }

    #[inline]
    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_assign_fp64_avx2(dst, *coeff);
            }
            _ => scalar::mul_assign::<FanPaar64>(dst, *coeff),
        }
    }

    #[inline]
    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 => {
                x86::fan_paar::mul_into_fp64_avx2(dst, *coeff, src);
            }
            _ => {
                dst.copy_from_slice(src);
                scalar::mul_assign::<FanPaar64>(dst, *coeff);
            }
        }
    }

    #[inline]
    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[fp64::Elem], srcs: &[u8]) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &Self::prepare(coeff), srcs);
        }
    }

    #[inline]
    fn mul_add_gather(dst: &mut [u8], coeffs: &[fp64::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }

    #[inline]
    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp64::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &Self::prepare(coeff), src);
            }
        }
    }

    #[inline]
    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        scalar::mul_elementwise::<FanPaar64>(dst, a, b);
    }
}

/// A Fan–Paar GF(2^16) coefficient resolved into the form this host's backend
/// wants.
///
/// `Tables` holds the four `fp8` nibble-table factors, built once in
/// [`FieldKernels::prepare`] and reused across the whole buffer; `Plain`
/// hands the element to the portable scalar kernel.
#[derive(Clone, Debug)]
pub enum Fp16Prepared {
    /// AVX2 or SSSE3: the four `fp8` nibble tables, plus the element for the
    /// scalar tail.
    Tables {
        /// The Fan–Paar GF(2^16) coefficient.
        coeff: fp16::Elem,
        /// [`crate::kernel::tables::FpTowerTables`] for `coeff`.
        tables: crate::kernel::tables::FpTowerTables,
    },
    /// No shuffle backend: the element itself.
    Plain(fp16::Elem),
}

impl Fp16Prepared {
    /// The coefficient this was built from.
    #[inline]
    #[must_use]
    pub const fn coeff(&self) -> fp16::Elem {
        match self {
            Self::Plain(coeff) | Self::Tables { coeff, .. } => *coeff,
        }
    }
}

impl FieldKernels for FanPaar16 {
    type Prepared = Fp16Prepared;

    fn prepare(coeff: fp16::Elem) -> Fp16Prepared {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 | Backend::Ssse3 => {
                Fp16Prepared::Tables {
                    coeff,
                    tables: crate::kernel::tables::FpTowerTables::new(coeff),
                }
            }
            _ => Fp16Prepared::Plain(coeff),
        }
    }

    #[inline]
    fn prepared_coeff(prepared: &Fp16Prepared) -> fp16::Elem {
        prepared.coeff()
    }

    #[inline]
    fn active_backend() -> Backend {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni | Backend::Avx2 | Backend::Ssse3 => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }

    fn mul_add(dst: &mut [u8], coeff: &Fp16Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::Ssse3 => x86::fan_paar::mul_add_ssse3(dst, tables, src),
                _ => x86::fan_paar::mul_add_avx2(dst, tables, src),
            },
            other => scalar::mul_add::<FanPaar16>(dst, other.coeff(), src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Fp16Prepared) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::Ssse3 => x86::fan_paar::mul_assign_ssse3(dst, tables),
                _ => x86::fan_paar::mul_assign_avx2(dst, tables),
            },
            other => scalar::mul_assign::<FanPaar16>(dst, other.coeff()),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Fp16Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Fp16Prepared::Tables { tables, .. } => match backend() {
                Backend::Ssse3 => x86::fan_paar::mul_into_ssse3(dst, tables, src),
                _ => x86::fan_paar::mul_into_avx2(dst, tables, src),
            },
            other => {
                dst.copy_from_slice(src);
                Self::mul_assign(dst, other);
            }
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[fp16::Elem], src: &[u8]) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &Self::prepare(coeff), src);
        }
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[fp16::Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }

    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[fp16::Elem], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &Self::prepare(coeff), src);
            }
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        // Both operands vary per lane; no fixed coefficient to broadcast.
        scalar::mul_elementwise::<FanPaar16>(dst, a, b);
    }
}
