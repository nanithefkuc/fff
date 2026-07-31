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

/// The instruction set the vector kernels run on.
///
/// Ordered by capability: earlier variants subsume later ones, and detection
/// picks the first available.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Backend {
    /// x86 AVX-512F + AVX-512BW + GFNI. Native field multiply over 64 byte
    /// lanes, with 32 architectural vector registers for blocked kernels.
    Avx512,
    /// x86 AVX2 + GFNI. One `GF2P8MULB` performs 32 field multiplies in the
    /// crate's GF(2^8) polynomial directly, with no table in the loop.
    Gfni,
    /// x86 AVX2 split-nibble shuffle over 32-byte lanes.
    Avx2,
    /// x86 SSSE3 split-nibble shuffle over 16-byte lanes.
    Ssse3,
    /// `AArch64` NEON + PMULL. Everything [`Backend::Neon`] does, plus
    /// `PMULL` for the one shape it wins: a varying operand pair, where the
    /// alternative is eight bit-serial rounds. Fixed coefficients still use
    /// the nibble shuffle, which PMULL's reduction network cannot match.
    /// Detecting the extension once, here, is what keeps
    /// [`FieldKernels::mul_elementwise`] from probing it per call.
    Pmull,
    /// `AArch64` NEON split-nibble shuffle over 16-byte lanes.
    Neon,
    /// WebAssembly `simd128` split-nibble shuffle over 16-byte lanes.
    Simd128,
    /// Portable scalar fallback. Always correct, always available.
    Scalar,
}

impl Backend {
    /// Every backend identifier, in detection-preference order.
    pub const ALL: [Self; 8] = [
        Self::Avx512,
        Self::Gfni,
        Self::Avx2,
        Self::Ssse3,
        Self::Pmull,
        Self::Neon,
        Self::Simd128,
        Self::Scalar,
    ];

    /// Probe the host for the best available backend.
    ///
    /// Prefer [`backend()`], which caches this.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(feature = "simd")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("gfni")
                {
                    return Self::Avx512;
                }
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni") {
                    return Self::Gfni;
                }
                if std::is_x86_feature_detected!("avx2") {
                    return Self::Avx2;
                }
                if std::is_x86_feature_detected!("ssse3") {
                    return Self::Ssse3;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                // NEON is baseline; PMULL rides in the optional crypto
                // extension. Probing it here rather than per call is the whole
                // point of caching a backend.
                if std::arch::is_aarch64_feature_detected!("neon") {
                    if std::arch::is_aarch64_feature_detected!("pmull") {
                        return Self::Pmull;
                    }
                    return Self::Neon;
                }
            }
            #[cfg(target_arch = "wasm32")]
            if cfg!(target_feature = "simd128") {
                return Self::Simd128;
            }
        }
        Self::Scalar
    }

    /// Whether this backend has a native byte-wide field multiply.
    #[inline]
    #[must_use]
    pub const fn has_native_mul(self) -> bool {
        matches!(self, Self::Avx512 | Self::Gfni)
    }

    /// Whether this backend implements the register-blocked multi-row
    /// kernels. Others decompose into repeated single-row AXPY.
    #[inline]
    #[must_use]
    pub const fn has_blocked_rows(self) -> bool {
        matches!(self, Self::Avx512 | Self::Gfni | Self::Neon | Self::Pmull)
    }

    /// Vector width in bytes.
    #[inline]
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        match self {
            Self::Avx512 => 64,
            Self::Gfni | Self::Avx2 => 32,
            Self::Ssse3 | Self::Neon | Self::Pmull | Self::Simd128 => 16,
            Self::Scalar => 8,
        }
    }

    /// Short stable identifier, also the value accepted by `FFF_BACKEND`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Avx512 => "avx512",
            Self::Gfni => "gfni",
            Self::Avx2 => "avx2",
            Self::Ssse3 => "ssse3",
            Self::Pmull => "pmull",
            Self::Neon => "neon",
            Self::Simd128 => "simd128",
            Self::Scalar => "scalar",
        }
    }

    /// Parse a backend name, as accepted by the `FFF_BACKEND` override.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "avx512" => Self::Avx512,
            "gfni" => Self::Gfni,
            "avx2" => Self::Avx2,
            "ssse3" => Self::Ssse3,
            "pmull" => Self::Pmull,
            "neon" => Self::Neon,
            "simd128" => Self::Simd128,
            "scalar" => Self::Scalar,
            _ => return None,
        })
    }

    #[cfg(any(feature = "std", test))]
    #[inline]
    const fn is_for_current_arch(self) -> bool {
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512 | Self::Gfni | Self::Avx2 | Self::Ssse3 => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon | Self::Pmull => true,
            #[cfg(target_arch = "wasm32")]
            Self::Simd128 => true,
            Self::Scalar => true,
            _ => false,
        }
    }
}
impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Error returned when a backend name is not recognized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseBackendError;

impl core::fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("unknown fff backend")
    }
}

impl core::str::FromStr for Backend {
    type Err = ParseBackendError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::from_name(name).ok_or(ParseBackendError)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseBackendError {}

#[cfg(feature = "std")]
static BACKEND: std::sync::LazyLock<Backend> = std::sync::LazyLock::new(resolve_backend);

#[cfg(feature = "std")]
fn resolve_backend() -> Backend {
    let detected = Backend::detect();
    // Downgrade-only override, for differential testing and for operators who
    // need to sidestep a misbehaving instruction path. Upgrades are refused:
    // running AVX2 code on a CPU without AVX2 is undefined behaviour, not a
    // configuration choice.
    match std::env::var("FFF_BACKEND") {
        Ok(name) => match Backend::from_name(name.trim()) {
            Some(requested) if requested.is_for_current_arch() && requested >= detected => {
                requested
            }
            _ => detected,
        },
        Err(_) => detected,
    }
}

/// The backend these kernels use, detected once per process.
///
/// May be downgraded at startup via the `FFF_BACKEND` environment variable
/// (`avx512`, `gfni`, `avx2`, `ssse3`, `pmull`, `neon`, `simd128`, `scalar`).
/// Requests for a backend the host cannot run are ignored.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "std")]
    {
        *BACKEND
    }
    #[cfg(not(feature = "std"))]
    {
        Backend::Scalar
    }
}
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
        Backend::Avx512 => x86::avx512::xor(dst, src),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::Gfni | Backend::Avx2 => x86::xor_avx2(dst, src),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::Ssse3 => x86::xor_sse2(dst, src),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        Backend::Neon | Backend::Pmull => aarch64::xor_neon(dst, src),
        #[cfg(all(feature = "simd", target_arch = "wasm32"))]
        Backend::Simd128 => wasm32::xor_simd128(dst, src),
        _ => scalar::xor(dst, src),
    }
}
