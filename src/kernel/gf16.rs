//! GF(2^16) kernel dispatch.
//!
//! Every backend here exploits the same tower identity: a 16-bit multiply is
//! two byte-wide multiplies, one of the source and one of the source with
//! adjacent bytes swapped, under the alternating coefficient pair in
//! [`TowerCoeff`]. Hardware that can multiply bytes in the field — GFNI —
//! needs no table; hardware that cannot emulates each of the four base-field
//! factors with a nibble shuffle.

use crate::field::gf16::{Elem, Gf16};
use crate::kernel::tables::{ScaleTable, TowerCoeff, scale_table};
// `Backend` and `TowerTables` are referenced only from the SIMD dispatch
// arms, which cfg away entirely on a scalar-only build or on an architecture
// without the corresponding backend.
#[allow(unused_imports)]
use crate::kernel::tables::TowerTables;
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

/// The four base-field nibble tables for `coeff`, borrowed from the shared
/// bank.
///
/// Deliberately not a [`TowerTables`]: that copies 136 bytes of table onto the
/// stack, and the tail handlers below run on as little as a single two-byte
/// element. Borrowing costs the two base multiplies of [`TowerCoeff::new`]
/// plus four rodata indexes, so the blocked backend kernels resolve their
/// coefficients through here too rather than duplicating the derivation.
#[inline]
pub(crate) fn factor_tables(coeff: Elem) -> [&'static ScaleTable; 4] {
    let f = TowerCoeff::new(coeff).factors();
    [
        scale_table(f[0]),
        scale_table(f[1]),
        scale_table(f[2]),
        scale_table(f[3]),
    ]
}

/// `dst ^= coeff * src` over interleaved elements, one element at a time.
///
/// The tail handler for the vector kernels.
#[cfg(feature = "internals")]
pub fn mul_add_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    mul_add_scalar_impl(dst, coeff, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_add_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    mul_add_scalar_impl(dst, coeff, src);
}

/// Per element this is 8 nibble lookups and 7 XORs against the 3 base-field
/// multiplies a Karatsuba [`Elem::mul`] performs — each of those a
/// carry-less product plus reduction, and none of them shared between
/// elements. Since `coeff` is fixed for the whole buffer its four base
/// factors collapse into split-nibble tables once, so the loop body becomes
/// pure loads.
fn mul_add_scalar_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    if coeff == Elem::ZERO {
        return;
    }
    if coeff == Elem::ONE {
        scalar::xor(dst, src);
        return;
    }
    let [f0, f1, f2, f3] = factor_tables(coeff);
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
        let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
        d[0] ^= f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
        d[1] ^= f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
    }
}

/// `dst *= coeff` over interleaved elements, one element at a time.
#[cfg(feature = "internals")]
pub fn mul_assign_scalar(dst: &mut [u8], coeff: Elem) {
    mul_assign_scalar_impl(dst, coeff);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn mul_assign_scalar(dst: &mut [u8], coeff: Elem) {
    mul_assign_scalar_impl(dst, coeff);
}

/// Table-driven for the same reason as [`mul_add_scalar_impl`], reading the
/// destination as its own source.
fn mul_assign_scalar_impl(dst: &mut [u8], coeff: Elem) {
    if coeff == Elem::ONE {
        return;
    }
    if coeff == Elem::ZERO {
        dst.fill(0);
        return;
    }
    let [f0, f1, f2, f3] = factor_tables(coeff);
    for d in dst.chunks_exact_mut(2) {
        let (a_lo, a_hi) = ((d[0] & 0x0f) as usize, (d[0] >> 4) as usize);
        let (b_lo, b_hi) = ((d[1] & 0x0f) as usize, (d[1] >> 4) as usize);
        d[0] = f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
        d[1] = f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
    }
}

/// `dst = coeff * src` over interleaved elements, one element at a time.
///
/// Tail handler for the fused out-of-place vector kernels, mirroring
/// [`mul_add_scalar`] without the destination read.
#[cfg(feature = "internals")]
pub fn mul_into_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    mul_into_scalar_impl(dst, coeff, src);
}

#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) fn mul_into_scalar(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    mul_into_scalar_impl(dst, coeff, src);
}

/// Table-driven for the same reason as [`mul_add_scalar_impl`], storing the
/// product instead of accumulating it.
fn mul_into_scalar_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    if coeff == Elem::ZERO {
        dst.fill(0);
        return;
    }
    if coeff == Elem::ONE {
        dst.copy_from_slice(src);
        return;
    }
    let [f0, f1, f2, f3] = factor_tables(coeff);
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let (a_lo, a_hi) = ((s[0] & 0x0f) as usize, (s[0] >> 4) as usize);
        let (b_lo, b_hi) = ((s[1] & 0x0f) as usize, (s[1] >> 4) as usize);
        d[0] = f0.lo[a_lo] ^ f0.hi[a_hi] ^ f2.lo[b_lo] ^ f2.hi[b_hi];
        d[1] = f1.lo[b_lo] ^ f1.hi[b_hi] ^ f3.lo[a_lo] ^ f3.hi[a_hi];
    }
}

/// AVX2 has enough width but not enough registers to retain several GF(2^16)
/// four-table coefficient sets. Blocking a gather cannot help there in any
/// case: coefficients are one-to-one with sources, so a source's nibble split
/// feeds exactly one coefficient and there is nothing to share. Measured
/// 0.84–1.01x against repeated AVX2 AXPY across 2–16 sources and 4–64 KiB
/// rows, so AXPY stays wired. SSSE3's smaller table vectors, by contrast,
/// block profitably (1.5–1.8x) and are wired to the blocked kernel.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn gather_avx2_axpy(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        x86::gf16::mul_add_avx2(dst, &TowerTables::new(coeff), src);
    }
}

/// The blocked AVX2 matrix folds every term into a register-resident
/// destination tile, which should beat re-reading the destination per term.
/// It does not, quite: measured 0.95–1.20x against repeated AVX2 AXPY, a win
/// only for rows at or below ~8 KiB and parity or slightly behind at 16–64
/// KiB. Not enough to justify a row-length branch in dispatch, so AXPY stays
/// wired; revisit if the tile ever widens past two accumulators.
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
            Backend::Avx512
                | Backend::Gfni
                | Backend::Avx2
                | Backend::Ssse3
                | Backend::Neon
                | Backend::Simd128
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
            // Not `scalar::mul_add_scatter`: that would re-derive a full
            // Karatsuba multiply per element. `mul_add_scalar` amortizes one
            // table resolve over each row.
            _ => {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    mul_add_scalar(row, coeff, src);
                }
            }
        }
    }
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        values: &[Elem],
        coeffs: &[Prepared],
        src: &[u8],
    ) {
        match backend() {
            Backend::Avx512 | Backend::Gfni => {
                Self::mul_add_scatter(rows, row_len, values, src);
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf16::scatter_avx2(rows, row_len, coeffs, src),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::scatter_ssse3(rows, row_len, coeffs, src),
            _ => Self::mul_add_scatter_with(rows, row_len, coeffs, src),
        }
    }

    fn mul_add_gather(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
        match backend() {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx512 => x86::avx512::gf16_gather(dst, coeffs, srcs),
            // Blocked: the four-source group derives its broadcasts once and
            // keeps them live, so it reads the destination once per group
            // instead of once per source. Measured 1.03–1.59x over repeated
            // GFNI AXPY across 2–16 sources and 4–64 KiB rows; before the
            // broadcasts were hoisted out of the byte loop the same kernel
            // ran at 0.29–0.47x, which is why dispatch used to avoid it.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Gfni => x86::gf16::gather_gfni(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => gather_avx2_axpy(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::gather_ssse3(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            Backend::Neon => aarch64::gf16::gather_neon(dst, coeffs, srcs),
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            Backend::Simd128 => wasm32::gf16::gather_simd128(dst, coeffs, srcs),
            // See `mul_add_scatter`: one table resolve per term beats the
            // generic oracle's per-element multiply.
            _ => {
                for (&coeff, &src) in coeffs.iter().zip(srcs) {
                    mul_add_scalar(dst, coeff, src);
                }
            }
        }
    }

    fn mul_add_gather_plan(dst: &mut [u8], values: &[Elem], coeffs: &[Prepared], srcs: &[&[u8]]) {
        match backend() {
            Backend::Avx512 | Backend::Gfni => Self::mul_add_gather(dst, values, srcs),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::gather_ssse3(dst, coeffs, srcs),
            _ => Self::mul_add_gather_with(dst, coeffs, srcs),
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
            // See `mul_add_scatter`: one table resolve per (term, row) beats
            // the generic oracle's per-element multiply.
            _ => {
                for &(coeffs, src) in terms {
                    let blocks = rows.chunks_exact_mut(row_len).take(nrows);
                    for (row, &coeff) in blocks.zip(coeffs) {
                        mul_add_scalar(row, coeff, src);
                    }
                }
            }
        }
    }
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        values: &[Elem],
        coeffs: &[Prepared],
        srcs: &[&[u8]],
    ) {
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        let _ = values;
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        match backend() {
            Backend::Gfni => {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: values,
                    nrows,
                    sources: srcs,
                };
                x86::gf16::matrix_gfni_with(rows, row_len, nrows, &terms);
                return;
            }
            Backend::Ssse3 => {
                let terms = crate::kernel::FlatMatrix {
                    coefficients: coeffs,
                    nrows,
                    sources: srcs,
                };
                x86::gf16::matrix_ssse3_with(rows, row_len, nrows, &terms);
                return;
            }
            _ => {}
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
            // See `Gf8::mul_elementwise`: no fixed coefficient, so the
            // shuffle backends multiply the two varying base-field operands
            // bit-serially and keep a nibble table only for constant `DELTA`.
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Avx2 => x86::gf16::elementwise_avx2(dst, a, b),
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            Backend::Ssse3 => x86::gf16::elementwise_ssse3(dst, a, b),
            _ => scalar::mul_elementwise::<Gf16>(dst, a, b),
        }
    }
}
