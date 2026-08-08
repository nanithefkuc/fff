//! Vector kernels and backend selection.
//!
//! Layout of this module:
//!
//! - [`Backend`] — which instruction set the kernels will use, detected once
//!   per process.
//! - [`FieldKernels`] — the per-field kernel contract. GF(2^8) and GF(2^16)
//!   own hand-written SIMD dispatch; wider and Fan–Paar fields use the
//!   portable `scalar` kernels.
//! - `scalar` — the portable reference and fallback implementation. Every
//!   SIMD backend is differentially tested against it, and vector loops use
//!   it for sub-lane tails.
//! - `x86` / `aarch64` / `wasm32` — architecture-local intrinsics.
//!
//! Callers should use the safe, validated wrappers in [`crate::ops`] rather
//! than this module directly.

#[cfg(feature = "internals")]
pub mod fan_paar;
#[cfg(not(feature = "internals"))]
pub(crate) mod fan_paar;

#[cfg(feature = "internals")]
pub mod gf16;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf16;

#[cfg(feature = "internals")]
pub mod gf32;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf32;

#[cfg(feature = "internals")]
pub mod gf64;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf64;

#[cfg(feature = "internals")]
pub mod gf8;
#[cfg(not(feature = "internals"))]
pub(crate) mod gf8;

#[cfg(feature = "internals")]
pub mod scalar;
#[cfg(not(feature = "internals"))]
pub(crate) mod scalar;

#[cfg(feature = "internals")]
pub mod tables;
#[cfg(not(feature = "internals"))]
pub(crate) mod tables;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[cfg(feature = "internals")]
pub mod aarch64;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[cfg(not(feature = "internals"))]
pub(crate) mod aarch64;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[cfg(feature = "internals")]
pub mod wasm32;
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[cfg(not(feature = "internals"))]
pub(crate) mod wasm32;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
pub mod x86;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) mod x86;

#[cfg(test)]
mod tests;

use crate::field::Field;

// Only the SIMD-enabled resolve path consults the environment; under a
// std-less build `backend()` reports `Scalar` without touching `Selection`.
#[cfg(feature = "simd")]
use simdispatch::Selection;

mod private {
    pub trait Sealed {}
}
impl private::Sealed for crate::field::gf8::Gf8 {}
impl private::Sealed for crate::field::gf16::Gf16 {}
impl private::Sealed for crate::field::gf32::Gf32 {}
impl private::Sealed for crate::field::gf64::Gf64 {}
impl private::Sealed for crate::field::fan_paar::FanPaar8 {}
impl private::Sealed for crate::field::fan_paar::FanPaar16 {}
impl private::Sealed for crate::field::fan_paar::FanPaar32 {}
impl private::Sealed for crate::field::fan_paar::FanPaar64 {}

// The backend ladder is owned by `simdispatch` (the Level 0 single source for
// detection and ordering); FFF re-exports it so downstream consumers keep
// compiling, and builds its own `Selection` over the tiers it implements.
pub use simdispatch::{Backend, ParseBackendError};

/// The tiers FFF implements kernels for, in detection-preference order: the
/// common ladder minus the x86 `V1` floor (FFF's 16-byte multiply kernels
/// require SSSE3, not plain SSE2, so a V1-only host resolves to
/// [`Backend::Scalar`]) and minus the deferred 64-byte AVX-512 tier (V4x,
/// fff's `avx512.rs` kernels are cross-compile-only and not validated).
pub const FFF_TIERS: &[Backend] = &[
    Backend::V3GfniCrypto,
    Backend::V3,
    Backend::V2,
    Backend::NeonAes,
    Backend::Neon,
    Backend::Wasm128,
    Backend::Scalar,
];

/// FFF's kernel-policy questions over the shared ladder.
///
/// `simdispatch::Backend` is capability; these two derive from the kernels
/// this crate actually implements, so they stay here (field-kernel policy,
/// not hardware facts). The ladder is owned upstream, so `Backend` gets these
/// only through this trait.
pub trait KernelBackend {
    /// Whether this backend has a native byte-wide field multiply (GFNI).
    fn has_native_mul(self) -> bool;

    /// Whether this backend implements the register-blocked multi-row
    /// kernels. Others decompose into repeated single-row AXPY.
    fn has_blocked_rows(self) -> bool;
}

impl KernelBackend for Backend {
    #[inline]
    fn has_native_mul(self) -> bool {
        matches!(self, Backend::V3GfniCrypto)
    }

    #[inline]
    fn has_blocked_rows(self) -> bool {
        matches!(
            self,
            Backend::V3GfniCrypto | Backend::Neon | Backend::NeonAes
        )
    }
}

/// The backend these kernels use, resolved once per process.
///
/// Runs [`Selection`] over [`FFF_TIERS`] — every tier this crate implements
/// kernels for, or [`Backend::Scalar`] when SIMD is compiled out — then
/// adjusted by the downgrade-only `SIMD_BACKEND` override. May be downgraded
/// at startup via `SIMD_BACKEND` (`v3_gfni_crypto`, `v3`, `v2`, `neon_aes`,
/// `neon`, `wasm128`, `scalar`); requests for a backend the host cannot run
/// are ignored. Detection itself is `simdispatch`'s `archmage` `summon()`
/// probe — the one probe in the stack.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "simd")]
    {
        *BACKEND
    }
    #[cfg(not(feature = "simd"))]
    {
        Backend::Scalar
    }
}

/// Memoized [`Selection`] over [`FFF_TIERS`], so dispatch never touches the
/// environment per call — a cache of the single-source resolve, not a second
/// resolver.
#[cfg(feature = "simd")]
static BACKEND: std::sync::LazyLock<Backend> =
    std::sync::LazyLock::new(|| Selection::new("SIMD_BACKEND").supports(FFF_TIERS).resolve());
/// The backend used for a particular field.
///
/// Wider polynomial towers and the Fan–Paar fields currently report
/// [`Backend::Scalar`] even when [`backend()`] selected a vector backend for
/// `Gf8` and `Gf16`.
#[inline]
#[must_use]
pub fn backend_for<F: FieldKernels>() -> Backend {
    F::active_backend()
}

/// Whether elementwise multiplication is vectorized for `F` on this host.
#[inline]
#[must_use]
pub fn has_vector_elementwise<F: FieldKernels>() -> bool {
    F::has_vector_elementwise()
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
/// Matrix-like coefficient/source provider for the register-blocked x86 kernels.
#[allow(clippy::len_without_is_empty)]
pub trait Matrix<C> {
    /// Number of terms.
    fn len(&self) -> usize;
    /// The coefficient of `term` for destination row `row`.
    fn coefficient(&self, term: usize, row: usize) -> &C;
    /// The source buffer of `term`.
    fn source(&self, term: usize) -> &[u8];
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) trait Matrix<C> {
    fn len(&self) -> usize;
    fn coefficient(&self, term: usize, row: usize) -> &C;
    fn source(&self, term: usize) -> &[u8];
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<C> Matrix<C> for [(&[C], &[u8])] {
    #[inline]
    fn len(&self) -> usize {
        <[(&[C], &[u8])]>::len(self)
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self[term].0[row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self[term].1
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(feature = "internals")]
/// Flat row-major coefficient matrix over borrowed sources.
pub struct FlatMatrix<'a, C> {
    /// Flat row-major coefficients, `terms * nrows` entries.
    pub coefficients: &'a [C],
    /// Destination row count.
    pub nrows: usize,
    /// Source buffers, one per term.
    pub sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg(not(feature = "internals"))]
pub(crate) struct FlatMatrix<'a, C> {
    pub(crate) coefficients: &'a [C],
    pub(crate) nrows: usize,
    pub(crate) sources: &'a [&'a [u8]],
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<C> Matrix<C> for FlatMatrix<'_, C> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self.coefficients[term * self.nrows + row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}

/// The per-field vector kernel contract.
///
/// Implementations own runtime dispatch for their field. Every method's
/// preconditions are checked by the [`crate::ops`] wrappers, not here.
///
/// # Preconditions
///
/// All slice lengths are in **bytes** and must be whole multiples of
/// `Self::BYTES`. `dst` and `src` must have equal length.
// The seal stays private even under `internals`: implementors are fixed.
#[allow(private_interfaces)]
pub trait FieldKernels: Field + private::Sealed {
    /// The backend-ready form of one coefficient.
    ///
    /// Different backends want different things from a coefficient: GFNI
    /// wants a broadcast word, the shuffle backends want nibble tables, the
    /// scalar path wants the element itself. [`FieldKernels::prepare`]
    /// resolves that once — which is why the single-coefficient kernels below
    /// take a `Prepared` and not an `Elem`. The backend is fixed for the life
    /// of the process, so this moves the backend decision *out* of the hot
    /// call rather than adding one.
    type Prepared: Clone + Send + Sync + core::fmt::Debug;

    /// Resolve a coefficient into the form this host's backend wants.
    fn prepare(coeff: Self::Elem) -> Self::Prepared;

    /// Recover the coefficient a [`FieldKernels::Prepared`] was built from.
    fn prepared_coeff(prepared: &Self::Prepared) -> Self::Elem;

    /// Backend used by this field's kernels.
    #[inline]
    #[must_use]
    fn active_backend() -> Backend {
        Backend::Scalar
    }

    /// Whether [`FieldKernels::mul_elementwise`] uses a vector implementation.
    #[inline]
    #[must_use]
    fn has_vector_elementwise() -> bool {
        false
    }

    /// `dst ^= coeff * src`. The workhorse AXPY.
    fn mul_add(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]);

    /// `dst = coeff * src`, out of place.
    ///
    /// The default copies `src` into `dst` and scales in place — two passes
    /// over the destination. Backends with a fused single-pass kernel
    /// override this. The override is worth roughly 2x on large buffers where
    /// the kernel is bandwidth-bound (x86 GF(2^8)/GF(2^16)); where it is
    /// compute-bound instead, as GF(2^16) is on NEON and wasm, halving
    /// destination traffic buys only a few percent (BENCHMARKS.md).
    fn mul_into(dst: &mut [u8], coeff: &Self::Prepared, src: &[u8]) {
        dst.copy_from_slice(src);
        Self::mul_assign(dst, coeff);
    }

    /// `dst *= coeff`, in place.
    fn mul_assign(dst: &mut [u8], coeff: &Self::Prepared);

    /// One source into many rows: `rows[j] ^= coeffs[j] * src` for each `j`.
    ///
    /// `rows` is a flat buffer of `coeffs.len()` contiguous rows of
    /// `row_len` bytes. This is the systematic-encode shape; blocked
    /// backends load `src` once per tile and update several rows from it.
    ///
    /// The coefficients are raw elements, so every backend has to resolve
    /// each one into its own form before it can use it. Callers holding the
    /// resolved form should call [`FieldKernels::mul_add_scatter_with`].
    fn mul_add_scatter(rows: &mut [u8], row_len: usize, coeffs: &[Self::Elem], src: &[u8]);

    /// Many sources into one row: `dst ^= sum(coeffs[i] * srcs[i])`.
    ///
    /// The transpose of [`FieldKernels::mul_add_scatter`], and the shape that
    /// rebuilds a single lost symbol. Blocked backends hold the destination
    /// tile in registers while every source is folded in, so the destination
    /// is read and written once per tile rather than once per source.
    ///
    /// See [`FieldKernels::mul_add_gather_with`] for the prepared form.
    fn mul_add_gather(dst: &mut [u8], coeffs: &[Self::Elem], srcs: &[&[u8]]);

    /// Many sources into many rows: for each `(coeffs, src)` term,
    /// `rows[j] ^= coeffs[j] * src` for every `j` in `0..nrows`.
    ///
    /// Equivalent to a [`FieldKernels::mul_add_scatter`] per term, but
    /// blocked backends hold a destination tile in registers across all
    /// terms, so destination memory traffic is independent of the term
    /// count. This is the decode/reconstruction shape.
    ///
    /// See [`FieldKernels::mul_add_matrix_with`] for the prepared form.
    fn mul_add_matrix(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Elem], &[u8])],
    );

    /// [`FieldKernels::mul_add_scatter`] over already-prepared coefficients.
    ///
    /// The default keeps preparation out of the row loop by applying the
    /// single-coefficient kernel to each row. Fields may override this when a
    /// backend can retain several prepared coefficients in registers.
    fn mul_add_scatter_with(
        rows: &mut [u8],
        row_len: usize,
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        for (row, coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
            Self::mul_add(row, coeff, src);
        }
    }
    /// Prepared-plan scatter with access to both original and resolved
    /// coefficients.
    ///
    /// The default uses the prepared single-row path. Blocked backends may use
    /// `values` for representations whose preparation is already free.
    fn mul_add_scatter_plan(
        rows: &mut [u8],
        row_len: usize,
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        src: &[u8],
    ) {
        Self::mul_add_scatter_with(rows, row_len, coeffs, src);
    }

    /// [`FieldKernels::mul_add_gather`] over already-prepared coefficients.
    ///
    /// The default applies the single-coefficient kernel once per source.
    fn mul_add_gather_with(dst: &mut [u8], coeffs: &[Self::Prepared], srcs: &[&[u8]]) {
        for (coeff, &src) in coeffs.iter().zip(srcs) {
            Self::mul_add(dst, coeff, src);
        }
    }
    /// Prepared-plan gather with access to both original and resolved
    /// coefficients.
    ///
    /// The default applies prepared AXPY once per source.
    fn mul_add_gather_plan(
        dst: &mut [u8],
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
        Self::mul_add_gather_with(dst, coeffs, srcs);
    }

    /// [`FieldKernels::mul_add_matrix`] over already-prepared coefficients.
    ///
    /// The default applies each term row by row. Fields may override this with
    /// a blocked implementation that retains destination tiles in registers.
    fn mul_add_matrix_with(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        terms: &[(&[Self::Prepared], &[u8])],
    ) {
        for &(coeffs, src) in terms {
            for (row, coeff) in rows.chunks_exact_mut(row_len).take(nrows).zip(coeffs) {
                Self::mul_add(row, coeff, src);
            }
        }
    }
    /// Prepared-plan matrix using flat row-major coefficients and source rows.
    ///
    /// The default is allocation-free repeated prepared AXPY. Register-blocked
    /// backends may override it and consume the same flat geometry directly.
    fn mul_add_matrix_plan(
        rows: &mut [u8],
        row_len: usize,
        nrows: usize,
        _values: &[Self::Elem],
        coeffs: &[Self::Prepared],
        srcs: &[&[u8]],
    ) {
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

    /// `dst[i] = a[i] * b[i]`, elementwise over two full vectors.
    ///
    /// Both operands vary per lane, so there is no coefficient to broadcast
    /// and no table to index. GFNI multiplies vectors directly; `AArch64`,
    /// Wasm, and the shuffle-only x86 backends use a branchless
    /// shift/reduce vector multiply. The wider fields run the reference path.
    fn mul_elementwise(dst: &mut [u8], a: &[u8], b: &[u8]);
}

/// `dst ^= src` over raw bytes.
///
/// Field-independent: addition in every binary field is XOR, and XOR of a
/// packed element array is XOR of its bytes regardless of element width.
///
/// # Panics
/// Panics if the slices differ in length.
#[cfg(feature = "internals")]
pub fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fff::xor: length mismatch");
    xor_impl(dst, src);
}

#[cfg(not(feature = "internals"))]
pub(crate) fn xor(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fff::xor: length mismatch");
    xor_impl(dst, src);
}

fn xor_impl(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "fff::xor: length mismatch");

    match backend() {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V3GfniCrypto | Backend::V3 => x86::xor_avx2(dst, src),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::V2 => x86::xor_sse2(dst, src),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Backend::Neon | Backend::NeonAes => aarch64::xor_neon(dst, src),
        #[cfg(all(feature = "simd", target_arch = "wasm32"))]
        Backend::Wasm128 => wasm32::xor_simd128(dst, src),
        _ => scalar::xor(dst, src),
    }
}
