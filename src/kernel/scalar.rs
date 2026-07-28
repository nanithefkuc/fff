//! Portable reference kernels.
//!
//! These are field-generic and elementwise: correct everywhere, dependent on
//! nothing but [`Field`]. They serve three roles.
//!
//! 1. **Oracle.** Every SIMD backend is differentially tested against these.
//! 2. **Fallback.** Targets with no supported vector unit run these.
//! 3. **Tails.** Vector loops hand their sub-lane remainder here.
//!
//! Hot scalar paths that beat the generic form — the GF(2^8) nibble tail, for
//! instance — live in the per-field dispatch modules, not here. Keeping this
//! module obviously-correct is worth more than making it fast.

use crate::field::{Elem, Field};

/// `dst ^= src`, eight bytes at a time.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "xor: length mismatch");

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mixed =
            u64::from_ne_bytes(d.try_into().unwrap()) ^ u64::from_ne_bytes(s.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= s;
    }
}

/// `dst ^= coeff * src`, elementwise.
pub fn mul_add<F: Field>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());

    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        xor(dst, src);
        return;
    }
    for (d, s) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(src.chunks_exact(F::BYTES))
    {
        let value = F::read(d).add(F::read(s).mul(coeff));
        F::write(d, value);
    }
}

/// `dst *= coeff`, elementwise, in place.
pub fn mul_assign<F: Field>(dst: &mut [u8], coeff: F::Elem) {
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    for d in dst.chunks_exact_mut(F::BYTES) {
        let value = F::read(d).mul(coeff);
        F::write(d, value);
    }
}

/// `rows[j] ^= coeffs[j] * src` for every row `j`.
pub fn mul_add_scatter<F: Field>(rows: &mut [u8], row_len: usize, coeffs: &[F::Elem], src: &[u8]) {
    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
        mul_add::<F>(row, coeff, src);
    }
}

/// Apply every `(coeffs, src)` term to all `nrows` rows.
pub fn mul_add_matrix<F: Field>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Elem], &[u8])],
) {
    for &(coeffs, src) in terms {
        for (row, &coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
            mul_add::<F>(row, coeff, src);
        }
    }
}

/// `dst ^= sum(coeffs[i] * srcs[i])`.
pub fn mul_add_gather<F: Field>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add::<F>(dst, coeff, src);
    }
}

/// `dst[i] = a[i] * b[i]`, elementwise.
pub fn mul_elementwise<F: Field>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    for ((d, x), y) in dst
        .chunks_exact_mut(F::BYTES)
        .zip(a.chunks_exact(F::BYTES))
        .zip(b.chunks_exact(F::BYTES))
    {
        F::write(d, F::read(x).mul(F::read(y)));
    }
}

/// Implement [`crate::kernel::FieldKernels`] with the portable primitives for
/// a supported field that has no hand-written SIMD backend.
macro_rules! impl_field_kernels {
    ($field:ty) => {
        impl crate::kernel::FieldKernels for $field {
            type Prepared = <Self as crate::field::Field>::Elem;

            #[inline]
            fn prepare(coeff: Self::Elem) -> Self::Prepared {
                coeff
            }

            #[inline]
            fn prepared_coeff(prepared: &Self::Prepared) -> Self::Elem {
                *prepared
            }

            fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
                crate::kernel::scalar::mul_add::<Self>(dst, *coeff, src);
            }

            fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared) {
                crate::kernel::scalar::mul_assign::<Self>(dst, *coeff);
            }

            fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Self::Elem], src: &[u8]) {
                crate::kernel::scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
            }

            fn mul_add_gather(dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]) {
                crate::kernel::scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
            }

            fn mul_add_matrix(
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[Self::Elem], &[u8])],
            ) {
                crate::kernel::scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
            }

            // `Prepared` is `Elem` for these fields, so the prepared forms
            // are the same call: there is nothing a backend could have
            // resolved ahead of time.
            fn mul_add_scatter_with(
                rows: &mut [u8],
                row_len: usize,
                coeffs: &[Self::Prepared],
                src: &[u8],
            ) {
                crate::kernel::scalar::mul_add_scatter::<Self>(rows, row_len, coeffs, src);
            }

            fn mul_add_gather_with(dst: &mut [u8], coeffs: &[Self::Prepared], srcs: &[&[u8]]) {
                crate::kernel::scalar::mul_add_gather::<Self>(dst, coeffs, srcs);
            }

            fn mul_add_matrix_with(
                rows: &mut [u8],
                row_len: usize,
                nrows: usize,
                terms: &[(&[Self::Prepared], &[u8])],
            ) {
                crate::kernel::scalar::mul_add_matrix::<Self>(rows, row_len, nrows, terms);
            }

            fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]) {
                crate::kernel::scalar::mul_elementwise::<Self>(dst, a, b);
            }
        }
    };
}

pub(crate) use impl_field_kernels;
