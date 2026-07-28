//! The public vector operation surface.
//!
//! Every function here is generic over [`Field`] and monomorphizes to that
//! field's dispatched SIMD kernel. Buffers are plain `&[u8]` holding packed
//! elements in the field's stable little-endian representation, so payloads
//! from the network or disk need no conversion.
//!
//! # Naming
//!
//! `mul_add` is the fused `dst ^= coeff * src` (an AXPY). `_scatter` fans one
//! source out to many rows, `_gather` folds many sources into one row, and
//! `_matrix` does many-to-many with the destination held in registers across
//! sources. These three shapes cover systematic encoding, symbol
//! reconstruction, and erasure decoding respectively.
//!
//! # Reusing a coefficient
//!
//! Each `mul_*` call resolves its coefficient into the form the host backend
//! wants. For GF(2^8) that is an array index; for GF(2^16) on a shuffle
//! backend it is four nibble tables. Amortized over a large buffer that cost
//! vanishes, but across many short buffers with the same coefficient it does
//! not. [`Coeff`] hoists it:
//!
//! ```
//! use fff::{Gf16, gf16, ops};
//!
//! let coeff = ops::Coeff::<Gf16>::new(gf16::Elem(0x0108));
//! let src = [0u8; 64];
//! for _ in 0..3 {
//!     let mut symbol = [0u8; 64];
//!     ops::mul_add_with(&mut symbol, &coeff, &src);   // prepared once
//! }
//! ```
//!
//! # Preconditions
//!
//! Buffer lengths must be whole multiples of `F::BYTES`, paired buffers must
//! be equal in length, and `dst`/`src` must not alias. All are checked; a
//! violation panics rather than silently corrupting.

use crate::field::{Elem, Field};
use crate::kernel::{self, FieldKernels};

/// A coefficient already resolved into the host backend's preferred form.
///
/// Build once, use many times. Construction is not free on every field — see
/// the module docs — but it is idempotent and the result is immutable, so a
/// `Coeff` can be cached alongside a coding matrix for the life of a codec.
///
/// Bound to the process's backend at construction. That is not a hazard in
/// practice (the backend is fixed after first use), but it does mean a
/// `Coeff` is not meaningful to serialize — rebuild it from the element.
pub struct Coeff<F: FieldKernels> {
    prepared: F::Prepared,
}

impl<F: FieldKernels> Coeff<F> {
    /// Resolve `coeff` for this host.
    #[inline]
    #[must_use]
    pub fn new(coeff: F::Elem) -> Self {
        Self {
            prepared: F::prepare(coeff),
        }
    }

    /// The field element this was built from.
    #[inline]
    #[must_use]
    pub fn value(&self) -> F::Elem {
        F::prepared_coeff(&self.prepared)
    }
}

impl<F: FieldKernels> Clone for Coeff<F> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
        }
    }
}

impl<F: FieldKernels> core::fmt::Debug for Coeff<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Coeff").field(&self.value()).finish()
    }
}

mod private {
    pub trait Sealed {}
}

/// A borrowed or owned backend-prepared coefficient.
///
/// This trait is sealed: values come from [`Coeff`] or, with `std`, a `Plan`.
/// It exists so the `_with` operations can consume either without cloning the
/// prepared tables.
pub trait PreparedCoefficient<F: FieldKernels>: private::Sealed {
    /// The field element this prepared representation multiplies by.
    fn value(&self) -> F::Elem;

    /// Borrow the backend-private representation.
    #[doc(hidden)]
    fn prepared(&self) -> &F::Prepared;
}

impl<F: FieldKernels> private::Sealed for Coeff<F> {}

impl<F: FieldKernels> PreparedCoefficient<F> for Coeff<F> {
    #[inline]
    fn value(&self) -> F::Elem {
        Coeff::value(self)
    }

    #[inline]
    fn prepared(&self) -> &F::Prepared {
        &self.prepared
    }
}

/// A borrowed coefficient inside a [`Plan`].
///
/// Unlike cloning a [`Coeff`], this copies only a pointer even when the
/// backend representation is a full GF(2^16) nibble-table set.
#[cfg(feature = "std")]
#[derive(Clone, Copy)]
pub struct CoeffRef<'a, F: FieldKernels> {
    prepared: &'a F::Prepared,
    field: core::marker::PhantomData<F>,
}

#[cfg(feature = "std")]
impl<F: FieldKernels> CoeffRef<'_, F> {
    /// The field element this prepared representation multiplies by.
    #[inline]
    #[must_use]
    pub fn value(self) -> F::Elem {
        F::prepared_coeff(self.prepared)
    }
}

#[cfg(feature = "std")]
impl<F: FieldKernels> private::Sealed for CoeffRef<'_, F> {}

#[cfg(feature = "std")]
impl<F: FieldKernels> PreparedCoefficient<F> for CoeffRef<'_, F> {
    #[inline]
    fn value(&self) -> F::Elem {
        F::prepared_coeff(self.prepared)
    }

    #[inline]
    fn prepared(&self) -> &F::Prepared {
        self.prepared
    }
}

/// A reusable vector or row-major matrix of prepared coefficients.
///
/// Plans are available with `std` because they own a dynamically sized
/// coefficient collection. Preparation happens once in the constructor;
/// [`Plan::get`] then borrows an entry without rebuilding or copying its
/// backend tables. Store a plan beside the coding matrix it represents.
#[cfg(feature = "std")]
pub struct Plan<F: FieldKernels> {
    prepared: std::boxed::Box<[F::Prepared]>,
    rows: usize,
    cols: usize,
}

#[cfg(feature = "std")]
impl<F: FieldKernels> Plan<F> {
    /// Prepare a one-dimensional coefficient vector.
    #[must_use]
    pub fn new(coeffs: &[F::Elem]) -> Self {
        let prepared = coeffs
            .iter()
            .copied()
            .map(F::prepare)
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();
        Self {
            prepared,
            rows: 1,
            cols: coeffs.len(),
        }
    }

    /// Prepare a row-major `rows × cols` coefficient matrix.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != rows * cols` or the product overflows.
    #[must_use]
    pub fn matrix(rows: usize, cols: usize, coeffs: &[F::Elem]) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("Plan::matrix: dimensions overflow");
        assert_eq!(
            coeffs.len(),
            len,
            "Plan::matrix: {} coefficients for {rows}x{cols} matrix",
            coeffs.len()
        );
        let mut plan = Self::new(coeffs);
        plan.rows = rows;
        plan.cols = cols;
        plan
    }

    /// Number of prepared coefficients.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.prepared.len()
    }

    /// Whether the plan contains no coefficients.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Matrix dimensions as `(rows, cols)`.
    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Borrow coefficient `index`, or return `None` when out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<CoeffRef<'_, F>> {
        self.prepared.get(index).map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        })
    }
    /// Borrow coefficient `(row, col)`, or return `None` when out of bounds.
    #[inline]
    #[must_use]
    pub fn get_at(&self, row: usize, col: usize) -> Option<CoeffRef<'_, F>> {
        (row < self.rows && col < self.cols)
            .then(|| row * self.cols + col)
            .and_then(|index| self.get(index))
    }
    /// Iterate over prepared coefficients without copying their backend form.
    #[must_use]
    pub fn coeffs(&self) -> impl ExactSizeIterator<Item = CoeffRef<'_, F>> {
        self.prepared.iter().map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        })
    }
    /// Iterate over one matrix row, or return `None` when out of bounds.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<impl ExactSizeIterator<Item = CoeffRef<'_, F>> + '_> {
        let start = row.checked_mul(self.cols)?;
        let prepared = self.prepared.get(start..start + self.cols)?;
        Some(prepared.iter().map(|prepared| CoeffRef {
            prepared,
            field: core::marker::PhantomData,
        }))
    }

    /// Iterate over the original field elements in row-major order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = F::Elem> + '_ {
        self.prepared.iter().map(F::prepared_coeff)
    }
}

#[cfg(feature = "std")]
impl<F: FieldKernels> Clone for Plan<F> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[cfg(feature = "std")]
impl<F: FieldKernels> core::fmt::Debug for Plan<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("dimensions", &self.dimensions())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[inline]
fn check_width<F: Field>(name: &str, len: usize) {
    assert!(
        len.is_multiple_of(F::BYTES),
        "{name}: buffer of {len} bytes is not a whole number of {} elements",
        F::NAME
    );
}

#[inline]
fn check_pair<F: Field>(name: &str, left_name: &str, left: usize, right_name: &str, right: usize) {
    assert_eq!(
        left, right,
        "{name}: {left_name} is {left} bytes but {right_name} is {right} bytes"
    );
    check_width::<F>(name, left);
}

/// `dst += src`, i.e. `dst ^= src`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn add_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    check_pair::<F>("add_assign", "dst", dst.len(), "src", src.len());
    kernel::xor(dst, src);
}

/// `dst -= src`. Identical to [`add_assign`] in characteristic two.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn sub_assign<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    add_assign::<F>(dst, src);
}

/// `dst ^= coeff * src`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    check_pair::<F>("mul_add", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        return;
    }
    if coeff.is_one() {
        kernel::xor(dst, src);
        return;
    }
    F::mul_add(dst, &F::prepare(coeff), src);
}

/// `dst ^= coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_add_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        return;
    }
    if value.is_one() {
        kernel::xor(dst, src);
        return;
    }
    F::mul_add(dst, coeff.prepared(), src);
}

/// `dst = coeff * src`, overwriting `dst`.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem, src: &[u8]) {
    check_pair::<F>("mul_into", "dst", dst.len(), "src", src.len());
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    if coeff.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(dst, &F::prepare(coeff), src);
}

/// `dst = coeff * src`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
#[inline]
pub fn mul_into_with<F: FieldKernels>(
    dst: &mut [u8],
    coeff: &impl PreparedCoefficient<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_into_with", "dst", dst.len(), "src", src.len());
    let value = coeff.value();
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    if value.is_one() {
        dst.copy_from_slice(src);
        return;
    }
    F::mul_into(dst, coeff.prepared(), src);
}

/// `dst *= coeff`, in place.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign<F: FieldKernels>(dst: &mut [u8], coeff: F::Elem) {
    check_width::<F>("mul_assign", dst.len());
    if coeff.is_one() {
        return;
    }
    if coeff.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(dst, &F::prepare(coeff));
}

/// `dst *= coeff`, reusing an already-prepared coefficient.
///
/// # Panics
/// Panics on a partial trailing element.
#[inline]
pub fn mul_assign_with<F: FieldKernels>(dst: &mut [u8], coeff: &impl PreparedCoefficient<F>) {
    check_width::<F>("mul_assign_with", dst.len());
    let value = coeff.value();
    if value.is_one() {
        return;
    }
    if value.is_zero() {
        dst.fill(0);
        return;
    }
    F::mul_assign(dst, coeff.prepared());
}

/// Fan one source out to many rows: `rows[j] ^= coeffs[j] * src`.
///
/// `rows` is a flat buffer of `coeffs.len()` contiguous rows, each `row_len`
/// bytes and the same length as `src`. This is the systematic-encode shape:
/// one arriving data symbol updates every parity row.
///
/// # Panics
/// Panics unless `rows` holds at least `coeffs.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
pub fn mul_add_scatter<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    coeffs: &[F::Elem],
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter", "row_len", row_len, "src", src.len());
    let used = coeffs
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        coeffs.len(),
    );
    if coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_scatter(&mut rows[..used], row_len, coeffs, src);
}

/// Fan one source out to many rows using a prepared coefficient plan.
///
/// Unlike [`mul_add_scatter`], repeated calls do not rebuild backend-specific
/// coefficient tables.
///
/// # Panics
/// Panics unless `rows` holds at least `plan.len()` rows of `row_len` bytes
/// and `row_len == src.len()`, or on a partial trailing element.
#[cfg(feature = "std")]
pub fn mul_add_scatter_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    plan: &Plan<F>,
    src: &[u8],
) {
    check_pair::<F>("mul_add_scatter_with", "row_len", row_len, "src", src.len());
    let used = plan
        .len()
        .checked_mul(row_len)
        .expect("mul_add_scatter_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_scatter_with: rows is {} bytes but {} rows of {row_len} bytes need {used}",
        rows.len(),
        plan.len(),
    );
    if plan.values().all(Elem::is_zero) {
        return;
    }
    F::mul_add_scatter_with(&mut rows[..used], row_len, &plan.prepared, src);
}

/// Fold many sources into one row: `dst ^= sum(coeffs[i] * srcs[i])`.
///
/// The transpose of [`mul_add_scatter`], and the shape that rebuilds a single
/// lost symbol from its survivors. Blocked backends hold the destination tile
/// in registers while every source is folded in, so `dst` is read and written
/// once per tile rather than once per source — prefer this over a loop of
/// [`mul_add`] whenever the sources share a destination.
///
/// # Panics
/// Panics unless `coeffs.len() == srcs.len()` and every source matches
/// `dst` in length, or on a partial trailing element.
pub fn mul_add_gather<F: FieldKernels>(dst: &mut [u8], coeffs: &[F::Elem], srcs: &[&[u8]]) {
    check_width::<F>("mul_add_gather", dst.len());
    assert_eq!(
        coeffs.len(),
        srcs.len(),
        "mul_add_gather: {} coefficients for {} sources",
        coeffs.len(),
        srcs.len()
    );
    for &src in srcs {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather: source is {} bytes, expected {}",
            src.len(),
            dst.len()
        );
    }
    if srcs.is_empty() || coeffs.iter().all(|c| c.is_zero()) {
        return;
    }
    F::mul_add_gather(dst, coeffs, srcs);
}

/// Fold many sources into one row using a prepared coefficient plan.
///
/// # Panics
/// Panics unless `plan.len() == srcs.len()` and every source matches `dst` in
/// length, or on a partial trailing element.
#[cfg(feature = "std")]
pub fn mul_add_gather_with<F: FieldKernels>(dst: &mut [u8], plan: &Plan<F>, srcs: &[&[u8]]) {
    check_width::<F>("mul_add_gather_with", dst.len());
    assert_eq!(
        plan.len(),
        srcs.len(),
        "mul_add_gather_with: plan has {} coefficients but there are {} sources",
        plan.len(),
        srcs.len(),
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            dst.len(),
            src.len(),
            "mul_add_gather_with: source {index} is {} bytes, expected {}",
            src.len(),
            dst.len(),
        );
    }
    if srcs.is_empty() || plan.values().all(Elem::is_zero) {
        return;
    }
    F::mul_add_gather_with(dst, &plan.prepared, srcs);
}

/// Apply many sources to many rows: for each `(coeffs, src)` term,
/// `rows[j] ^= coeffs[j] * src` for every `j` in `0..nrows`.
///
/// Semantically a [`mul_add_scatter`] per term. The difference is memory
/// traffic: blocked backends keep a destination tile in registers while every
/// source is folded in, so the destination is read and written once per tile
/// rather than once per term. That is the whole game once
/// `nrows * row_len` exceeds L1 — which is the normal case for erasure
/// reconstruction.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes, every
/// term supplies `nrows` coefficients, and every source is `row_len` bytes.
pub fn mul_add_matrix<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[F::Elem], &[u8])],
) {
    check_width::<F>("mul_add_matrix", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    for &(coeffs, src) in terms {
        assert_eq!(
            coeffs.len(),
            nrows,
            "mul_add_matrix: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix: source is {} bytes, expected {row_len}",
            src.len()
        );
    }
    if nrows == 0 || terms.is_empty() {
        return;
    }
    F::mul_add_matrix(&mut rows[..used], row_len, nrows, terms);
}

/// Apply many sources to many rows using a prepared row-major coefficient plan.
///
/// `plan` must have dimensions `(srcs.len(), nrows)`: each plan row holds the
/// coefficients for one source term.
///
/// # Panics
/// Panics unless `rows` holds at least `nrows` rows of `row_len` bytes,
/// `plan.dimensions() == (srcs.len(), nrows)`, and every source is `row_len`
/// bytes.
#[cfg(feature = "std")]
pub fn mul_add_matrix_with<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    plan: &Plan<F>,
    srcs: &[&[u8]],
) {
    check_width::<F>("mul_add_matrix_with", row_len);
    let used = nrows
        .checked_mul(row_len)
        .expect("mul_add_matrix_with: row geometry overflows");
    assert!(
        rows.len() >= used,
        "mul_add_matrix_with: rows is {} bytes but {nrows} rows of {row_len} bytes need {used}",
        rows.len(),
    );
    assert_eq!(
        plan.dimensions(),
        (srcs.len(), nrows),
        "mul_add_matrix_with: plan dimensions do not match (sources, rows)",
    );
    for (index, &src) in srcs.iter().enumerate() {
        assert_eq!(
            src.len(),
            row_len,
            "mul_add_matrix_with: source {index} is {} bytes, expected {row_len}",
            src.len(),
        );
    }
    if nrows == 0 || srcs.is_empty() {
        return;
    }
    let terms = srcs
        .iter()
        .enumerate()
        .map(|(index, &src)| {
            let start = index * nrows;
            (&plan.prepared[start..start + nrows], src)
        })
        .collect::<std::vec::Vec<_>>();
    F::mul_add_matrix_with(&mut rows[..used], row_len, nrows, &terms);
}

/// Elementwise product: `dst[i] = a[i] * b[i]`.
///
/// Both operands vary per lane, so there is no coefficient to broadcast and
/// no nibble table to index. GFNI multiplies the vectors directly; `AArch64`
/// uses PMULL when available and a lane-parallel shift/reduce sequence
/// otherwise; Wasm `simd128` uses the same shift/reduce construction.
/// Shuffle-only x86 backends fall back to the scalar reference path.
///
/// # Panics
/// Panics on a length mismatch or a partial trailing element.
pub fn mul_elementwise<F: FieldKernels>(dst: &mut [u8], a: &[u8], b: &[u8]) {
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "a", a.len());
    check_pair::<F>("mul_elementwise", "dst", dst.len(), "b", b.len());
    F::mul_elementwise(dst, a, b);
}

/// Pack field elements into their stable little-endian byte representation.
///
/// # Panics
/// Panics unless `dst.len() == elems.len() * F::BYTES`.
pub fn pack<F: Field>(dst: &mut [u8], elems: &[F::Elem]) {
    let expected = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack: buffer geometry overflows");
    assert_eq!(
        dst.len(),
        expected,
        "pack: dst is {} bytes but {} {} elements need {expected}",
        dst.len(),
        elems.len(),
        F::NAME,
    );
    for (bytes, &elem) in dst.chunks_exact_mut(F::BYTES).zip(elems) {
        F::write(bytes, elem);
    }
}

/// Decode packed little-endian bytes into field elements.
///
/// # Panics
/// Panics unless `src.len() == dst.len() * F::BYTES`.
pub fn unpack<F: Field>(dst: &mut [F::Elem], src: &[u8]) {
    let expected = dst
        .len()
        .checked_mul(F::BYTES)
        .expect("unpack: buffer geometry overflows");
    assert_eq!(
        src.len(),
        expected,
        "unpack: src is {} bytes but dst holds {} {} elements ({expected} bytes)",
        src.len(),
        dst.len(),
        F::NAME,
    );
    for (elem, bytes) in dst.iter_mut().zip(src.chunks_exact(F::BYTES)) {
        *elem = F::read(bytes);
    }
}

/// Pack field elements into a newly allocated byte vector.
///
/// # Panics
/// Panics if the required byte length overflows `usize`.
#[cfg(feature = "std")]
#[must_use]
pub fn pack_to_vec<F: Field>(elems: &[F::Elem]) -> std::vec::Vec<u8> {
    let len = elems
        .len()
        .checked_mul(F::BYTES)
        .expect("pack_to_vec: buffer geometry overflows");
    let mut bytes = std::vec![0; len];
    pack::<F>(&mut bytes, elems);
    bytes
}
