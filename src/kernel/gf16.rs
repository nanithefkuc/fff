//! GF(2^16) kernel dispatch.
//!
//! Every backend here exploits the same tower identity: a 16-bit multiply is
//! two byte-wide multiplies, one of the source and one of the source with
//! adjacent bytes swapped, under the alternating coefficient pair in
//! [`TowerCoeff`]. Hardware that can multiply bytes in the field — GFNI —
//! needs no table; hardware that cannot emulates each of the four base-field
//! factors with a nibble shuffle.

use crate::field::gf16::{Elem, Gf16};
// `Backend`, `TowerCoeff` and `TowerTables` are referenced only from the SIMD
// dispatch arms, which cfg away entirely on a scalar-only build or on an
// architecture without the corresponding backend.
#[allow(unused_imports)]
use crate::kernel::tables::{TowerCoeff, TowerTables};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, scalar};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use crate::kernel::aarch64;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
use crate::kernel::wasm32;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

/// A GF(2^16) coefficient resolved into the form this host's backend wants.
///
/// The three variants are not interchangeable representations of the same
/// cost. `Compact` is two base multiplies; `Tables` is four nibble tables and
/// ~140 bytes to copy. Choosing between them at *preparation* time is the
/// point: a GFNI host never pays for tables it will not read, and a shuffle
/// host builds them once instead of on every call.
#[derive(Clone, Debug)]
pub enum Prepared {
    /// Native byte multiply (GFNI): a pair of broadcast words.
    Compact(TowerCoeff),
    /// Shuffle backends (AVX2, SSSE3, NEON): four nibble tables.
    Tables(TowerTables),
    /// No vector unit: the coefficient itself.
    Plain(Elem),
}

impl Prepared {
    /// The coefficient this was built from.
    #[inline]
    #[must_use]
    pub const fn coeff(&self) -> Elem {
        match self {
            Self::Compact(compact) => compact.coeff,
            Self::Tables(tables) => tables.coeff,
            Self::Plain(coeff) => *coeff,
        }
    }
}

/// `dst ^= coeff * src` over interleaved elements, one element at a time.
///
/// The tail handler for the vector kernels.
pub(crate) fn mul_add_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let product = Elem::from_bytes([s[0], s[1]]).mul(coeff);
        let current = Elem::from_bytes([d[0], d[1]]);
        d.copy_from_slice(&current.add(product).to_bytes());
    }
}

/// `dst *= coeff` over interleaved elements, one element at a time.
pub(crate) fn mul_assign_scalar(dst: &mut [u8], coeff: Elem) {
    for d in dst.chunks_exact_mut(2) {
        let value = Elem::from_bytes([d[0], d[1]]).mul(coeff);
        d.copy_from_slice(&value.to_bytes());
    }
}

/// `dst = coeff * src` over interleaved elements, one element at a time.
///
/// Tail handler for the fused out-of-place vector kernels, mirroring
/// [`mul_add_scalar`] without the destination read.
#[allow(dead_code)]
pub(crate) fn mul_into_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let product = Elem::from_bytes([s[0], s[1]]).mul(coeff);
        d.copy_from_slice(&product.to_bytes());
    }
}

/// GFNI's compact factors are cheap to derive, but the generic blocked gather
/// cannot keep a variable number of broadcasts live and measured 1.5–2.7x
/// behind the four-chain single-coefficient kernel. Repeated GFNI AXPY is the
/// measured crossover for this one shape.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gather_gfni_axpy(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        x86::gf16::mul_add_gfni(dst, TowerCoeff::new(coeff), src);
    }
}

/// AVX2 has enough width but not enough registers to retain several
/// GF(2^16) four-table coefficient sets: the blocked gather/matrix kernels
/// spill and measured 3–25% behind repeated AVX2 AXPY. SSSE3's smaller table
/// vectors, by contrast, win 1.4–1.5x. Use the measured crossover rather than
/// forcing the nominally wider blocked kernel.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gather_avx2_axpy(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        x86::gf16::mul_add_avx2(dst, &TowerTables::new(coeff), src);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn matrix_avx2_axpy(rows: &mut [u8], row_len: usize, terms: &[(&[Elem], &[u8])]) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            x86::gf16::mul_add_avx2(row, &TowerTables::new(coeff), src);
        }
    }
}

impl FieldKernels for Gf16 {
    type Prepared = Prepared;

    fn prepare(coeff: Elem) -> Prepared {
        match backend() {
            Backend::Avx512 | Backend::Gfni => Prepared::Compact(TowerCoeff::new(coeff)),
            Backend::Avx2 | Backend::Ssse3 | Backend::Neon | Backend::Simd128 => {
                Prepared::Tables(TowerTables::new(coeff))
            }
            Backend::Scalar => Prepared::Plain(coeff),
        }
    }

    #[inline]
    fn prepared_coeff(prepared: &Prepared) -> Elem {
        prepared.coeff()
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

    fn mul_add(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact(compact) => match backend() {
                Backend::Avx512 => x86::avx512::gf16_mul_add(dst, *compact, src),
                _ => x86::gf16::mul_add_gfni(dst, *compact, src),
            },
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Tables(tables) => match backend() {
                Backend::Ssse3 => x86::gf16::mul_add_ssse3(dst, tables, src),
                _ => x86::gf16::mul_add_avx2(dst, tables, src),
            },
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Prepared::Tables(tables) => aarch64::gf16::mul_add_neon(dst, tables, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Prepared::Tables(tables) => wasm32::gf16::mul_add_simd128(dst, tables, src),
            other => mul_add_scalar(dst, other.coeff(), src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Prepared) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact(compact) => match backend() {
                Backend::Avx512 => x86::avx512::gf16_mul_assign(dst, *compact),
                _ => x86::gf16::mul_assign_gfni(dst, *compact),
            },
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Tables(tables) => match backend() {
                Backend::Ssse3 => x86::gf16::mul_assign_ssse3(dst, tables),
                _ => x86::gf16::mul_assign_avx2(dst, tables),
            },
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Prepared::Tables(tables) => aarch64::gf16::mul_assign_neon(dst, tables),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Prepared::Tables(tables) => wasm32::gf16::mul_assign_simd128(dst, tables),
            other => mul_assign_scalar(dst, other.coeff()),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact(compact) => match backend() {
                Backend::Avx512 => x86::avx512::gf16_mul_into(dst, *compact, src),
                _ => x86::gf16::mul_into_gfni(dst, *compact, src),
            },
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Tables(tables) => match backend() {
                Backend::Ssse3 => x86::gf16::mul_into_ssse3(dst, tables, src),
                _ => x86::gf16::mul_into_avx2(dst, tables, src),
            },
            // aarch64 and wasm32 keep the default copy-then-scale until their
            // GF(2^16) fused kernels are written.
            other => {
                dst.copy_from_slice(src);
                Self::mul_assign(dst, other);
            }
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf16_scatter(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf16::scatter_gfni(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf16::scatter_avx2(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::scatter_ssse3(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf16::scatter_neon(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf16::scatter_simd128(rows, row_len, coeffs, src),
            _ => scalar::mul_add_scatter::<Gf16>(rows, row_len, coeffs, src),
        }
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf16_gather(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => gather_gfni_axpy(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => gather_avx2_axpy(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::gather_ssse3(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf16::gather_neon(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf16::gather_simd128(dst, coeffs, srcs),
            _ => scalar::mul_add_gather::<Gf16>(dst, coeffs, srcs),
        }
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf16_matrix(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf16::matrix_gfni(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => matrix_avx2_axpy(rows, row_len, terms),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::matrix_ssse3(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf16::matrix_neon(rows, row_len, nrows, terms),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf16::matrix_simd128(rows, row_len, nrows, terms),
            _ => scalar::mul_add_matrix::<Gf16>(rows, row_len, nrows, terms),
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf16_elementwise(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf16::elementwise_gfni(dst, a, b),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon if std::arch::is_aarch64_feature_detected!("aes") => {
                aarch64::gf16::elementwise_pmull(dst, a, b);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf16::elementwise_neon(dst, a, b),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf16::elementwise_simd128(dst, a, b),
            // See `Gf8::mul_elementwise`: with both operands varying there is
            // no fixed coefficient for a nibble table to be built from.
            _ => scalar::mul_elementwise::<Gf16>(dst, a, b),
        }
    }
}
