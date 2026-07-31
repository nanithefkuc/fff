//! GF(2^64) kernel dispatch.
//!
//! On GFNI x86 a GF(2^64) multiply is the level-3 tower identity — two
//! GF(2^32) lane multiplies under a period-2 coefficient, each itself the
//! four-`GF2P8MULB` [`crate::kernel::gf32`] scale, so eight `GF2P8MULB` per
//! 32-byte lane. Everywhere else the portable scalar kernel applies. As
//! with GF(2^32), the trait defaults carry every multi-row and prepared
//! operation from [`FieldKernels::mul_add`].

use crate::field::gf64::{Elem, Gf64};
#[allow(unused_imports)]
use crate::kernel::{Backend, FieldKernels, backend, scalar};

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::x86;

/// A GF(2^64) coefficient resolved into the form this host's backend wants.
///
/// `Compact` carries the eight 8-byte GFNI broadcast tiles derived once in
/// [`FieldKernels::prepare`]; `Plain` hands the element to the portable
/// scalar kernel. See [`crate::kernel::gf32::Prepared`] for the rationale.
#[derive(Clone, Debug)]
pub enum Prepared {
    /// GFNI: eight 8-byte broadcast tiles, plus the element for the scalar
    /// tail.
    Compact {
        /// The GF(2^64) coefficient, for the portable tail.
        coeff: Elem,
        /// The eight `[same_a, cross_a, …, cross_d]` tiles from
        /// [`x86::gf64::gf64_tiles`].
        tiles: [u64; 8],
    },
    /// No GFNI backend: the element itself, for the portable scalar kernel.
    Plain(Elem),
}

impl Prepared {
    /// The coefficient this was built from.
    #[inline]
    #[must_use]
    pub const fn coeff(&self) -> Elem {
        match self {
            Self::Plain(coeff) | Self::Compact { coeff, .. } => *coeff,
        }
    }
}

impl FieldKernels for Gf64 {
    type Prepared = Prepared;

    fn prepare(coeff: Elem) -> Prepared {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 | Backend::Gfni => Prepared::Compact {
                coeff,
                tiles: x86::gf64::gf64_tiles(coeff),
            },
            _ => Prepared::Plain(coeff),
        }
    }

    #[inline]
    fn prepared_coeff(prepared: &Prepared) -> Elem {
        prepared.coeff()
    }

    #[inline]
    fn active_backend() -> Backend {
        match backend() {
            Backend::Avx512 | Backend::Gfni => backend(),
            _ => Backend::Scalar,
        }
    }

    #[inline]
    fn has_vector_elementwise() -> bool {
        false
    }

    fn mul_add(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact { coeff, tiles } => x86::gf64::mul_add_gfni(dst, *coeff, *tiles, src),
            other => scalar::mul_add::<Gf64>(dst, other.coeff(), src),
        }
    }

    fn mul_assign(dst: &mut [u8], coeff: &Prepared) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact { coeff, tiles } => x86::gf64::mul_assign_gfni(dst, *coeff, *tiles),
            other => scalar::mul_assign::<Gf64>(dst, other.coeff()),
        }
    }

    fn mul_into(dst: &mut [u8], coeff: &Prepared, src: &[u8]) {
        match coeff {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Prepared::Compact { coeff, tiles } => {
                x86::gf64::mul_into_gfni(dst, *coeff, *tiles, src);
            }
            other => {
                dst.copy_from_slice(src);
                Self::mul_assign(dst, other);
            }
        }
    }

    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, &Self::prepare(coeff), src);
        }
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, &Self::prepare(coeff), src);
        }
    }

    fn mul_add_matrix(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
        for &(coeffs, src) in terms {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, &Self::prepare(coeff), src);
            }
        }
    }

    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
        scalar::mul_elementwise::<Gf64>(dst, a, b);
    }
}
