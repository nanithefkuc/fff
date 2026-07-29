//! GF(2^8) NEON kernels.
//!
//! NEON has no byte-wide field multiply, so every product here is the
//! split-nibble form: `vqtbl1q_u8` performs a 16-entry lookup on all 16 lanes
//! at once, and `c * x` becomes `lo[x & 0xf] ^ hi[x >> 4]` against the
//! coefficient's precomputed [`ScaleTable`].
//!
//! Three shapes, in increasing arithmetic intensity:
//!
//! - [`mul_add_neon`] and [`mul_assign_neon`] — one buffer, two 16-byte lanes
//!   per iteration.
//! - [`scatter_neon`] — one source load feeds four contiguous rows, so the
//!   source is read once per group of four instead of once per row.
//! - [`matrix_neon`] — a four-row destination tile is loaded into
//!   accumulators once, every `(coeffs, src)` term is folded in, and the tile
//!   is stored once. Destination traffic is therefore independent of
//!   `terms.len()`, which is the entire reason the shape exists.
//!
//! Every vector loop hands its sub-lane remainder to the scalar nibble
//! kernels in [`crate::kernel::gf8`]; buffer lengths are arbitrary and are
//! never assumed to be lane-aligned.

use core::arch::aarch64::*;

use crate::field::gf8::Elem;
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::tables::{ScaleTable, scale_table};

/// Load a coefficient's nibble tables into two vector registers.
#[inline]
#[target_feature(enable = "neon")]
fn load_tables(table: &ScaleTable) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: `lo` and `hi` are `[u8; 16]` fields of a live reference, so each
    // pointer is valid for exactly the one 16-byte vector being read.
    unsafe { (vld1q_u8(table.lo.as_ptr()), vld1q_u8(table.hi.as_ptr())) }
}

/// `coeff * x` for one 16-byte lane, given that coefficient's nibble tables.
///
/// `vshrq_n_u8` is a logical shift, so the high-nibble indices already land in
/// `0..16` and need no extra mask before the lookup.
#[inline]
#[target_feature(enable = "neon")]
fn scale(lo: uint8x16_t, hi: uint8x16_t, x: uint8x16_t) -> uint8x16_t {
    let low = vandq_u8(x, vdupq_n_u8(0x0f));
    let high = vshrq_n_u8(x, 4);
    veorq_u8(vqtbl1q_u8(lo, low), vqtbl1q_u8(hi, high))
}

/// Which of the three cases a row's coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: the row contributes nothing.
    Skip,
    /// Coefficient one: XOR the source in unscaled, no lookup needed.
    Identity,
    /// Any other coefficient: two nibble lookups and an XOR.
    Table,
}

/// One coefficient resolved into the form the vector loops consume.
///
/// Branching on [`Kind::Skip`] and [`Kind::Identity`] pays for itself: the
/// coefficient arrays handed to [`scatter_neon`] and [`matrix_neon`] are full
/// of zeros and ones, and each case removes two `TBL`s per lane — a skip also
/// removes the destination load and store entirely.
#[derive(Clone, Copy)]
struct Scaling {
    /// Low-nibble lookup table. Meaningless unless `kind` is [`Kind::Table`].
    lo: uint8x16_t,
    /// High-nibble lookup table. Meaningless unless `kind` is [`Kind::Table`].
    hi: uint8x16_t,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff` against the shared table bank.
    #[inline]
    #[target_feature(enable = "neon")]
    fn new(coeff: Elem) -> Self {
        let zero = vdupq_n_u8(0);
        if coeff == Elem::ZERO {
            return Self {
                lo: zero,
                hi: zero,
                kind: Kind::Skip,
            };
        }
        if coeff == Elem::ONE {
            return Self {
                lo: zero,
                hi: zero,
                kind: Kind::Identity,
            };
        }
        let (lo, hi) = load_tables(scale_table(coeff));
        Self {
            lo,
            hi,
            kind: Kind::Table,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[target_feature(enable = "neon")]
    fn fold(self, acc: uint8x16_t, x: uint8x16_t) -> uint8x16_t {
        match self.kind {
            Kind::Skip => acc,
            Kind::Identity => veorq_u8(acc, x),
            Kind::Table => veorq_u8(acc, scale(self.lo, self.hi, x)),
        }
    }
}

/// One destination row of a [`scatter_neon`] group, resolved once per group.
#[derive(Clone, Copy)]
struct RowPlan {
    /// First byte of the row.
    ptr: *mut u8,
    /// Nibble tables for this row's coefficient, used by the scalar tail.
    ///
    /// Correct for all three [`Kind`]s: the tables of `0` are all zero and
    /// the tables of `1` are the identity map, so the tail needs no case
    /// analysis of its own.
    table: &'static ScaleTable,
    /// Vector strategy for this row's coefficient.
    scaling: Scaling,
}

impl RowPlan {
    /// Resolve the row starting at `ptr` for coefficient `coeff`.
    ///
    /// `ptr` is only recorded here; the loops that dereference it carry the
    /// in-bounds and disjointness argument.
    #[inline]
    #[target_feature(enable = "neon")]
    fn new(ptr: *mut u8, coeff: Elem) -> Self {
        Self {
            ptr,
            table: scale_table(coeff),
            scaling: Scaling::new(coeff),
        }
    }
}

/// `dst ^= coeff * src`, two 16-byte NEON lanes per iteration.
pub fn mul_add_neon(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: NEON is baseline on AArch64; the slices are independently
    // borrowed and the kernel never runs past the shorter of the two.
    unsafe { mul_add_impl(dst, table, src) }
}

#[target_feature(enable = "neon")]
unsafe fn mul_add_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len());
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let (lo, hi) = load_tables(table);

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, which bounds both buffers; unaligned
        // NEON loads and stores are allowed, and `dst` does not alias `src`.
        unsafe {
            let x0 = vld1q_u8(src_ptr.add(offset));
            let x1 = vld1q_u8(src_ptr.add(offset + 16));
            let d0 = vld1q_u8(dst_ptr.add(offset));
            let d1 = vld1q_u8(dst_ptr.add(offset + 16));
            vst1q_u8(dst_ptr.add(offset), veorq_u8(d0, scale(lo, hi, x0)));
            vst1q_u8(dst_ptr.add(offset + 16), veorq_u8(d1, scale(lo, hi, x1)));
        }
        offset += 32;
    }
    if offset + 16 <= len {
        // SAFETY: `offset + 16 <= len`, which bounds both buffers.
        unsafe {
            let x = vld1q_u8(src_ptr.add(offset));
            let d = vld1q_u8(dst_ptr.add(offset));
            vst1q_u8(dst_ptr.add(offset), veorq_u8(d, scale(lo, hi, x)));
        }
        offset += 16;
    }

    mul_add_nibble(&mut dst[offset..len], table, &src[offset..len]);
}

/// `dst = coeff * dst`, 16 bytes per iteration.
///
/// Deliberately unblocked: scaling a row in place is a setup step, not a hot
/// loop, and the simple shape is the one that is obviously right.
pub fn mul_assign_neon(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: NEON is baseline on AArch64 and `dst` is uniquely borrowed.
    unsafe { mul_assign_impl(dst, table) }
}

#[target_feature(enable = "neon")]
unsafe fn mul_assign_impl(dst: &mut [u8], table: &ScaleTable) {
    let len = dst.len() & !15;
    let dst_ptr = dst.as_mut_ptr();
    let (lo, hi) = load_tables(table);

    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len()`.
        unsafe {
            let d = vld1q_u8(dst_ptr.add(offset));
            vst1q_u8(dst_ptr.add(offset), scale(lo, hi, d));
        }
        offset += 16;
    }

    mul_assign_nibble(&mut dst[len..], table);
}

/// `dst = coeff * src`, out of place, over 16-byte NEON lanes.
///
/// Fuses what would otherwise be a copy followed by an in-place scale: one
/// pass, and `dst` is never read.
pub fn mul_into_neon(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: NEON is baseline on AArch64; the slices are independently
    // borrowed and the kernel never runs past the shorter of the two.
    unsafe { mul_into_impl(dst, table, src) }
}

#[target_feature(enable = "neon")]
unsafe fn mul_into_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len());
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let (lo, hi) = load_tables(table);

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, which bounds both buffers; unaligned
        // NEON loads and stores are allowed, and `dst` does not alias `src`.
        unsafe {
            let x0 = vld1q_u8(src_ptr.add(offset));
            let x1 = vld1q_u8(src_ptr.add(offset + 16));
            vst1q_u8(dst_ptr.add(offset), scale(lo, hi, x0));
            vst1q_u8(dst_ptr.add(offset + 16), scale(lo, hi, x1));
        }
        offset += 32;
    }
    if offset + 16 <= len {
        // SAFETY: `offset + 16 <= len`, which bounds both buffers.
        unsafe {
            let x = vld1q_u8(src_ptr.add(offset));
            vst1q_u8(dst_ptr.add(offset), scale(lo, hi, x));
        }
        offset += 16;
    }

    mul_into_nibble(&mut dst[offset..len], table, &src[offset..len]);
}

/// `rows[j] ^= coeffs[j] * src` for every row, four rows per source load.
///
/// `rows` is `coeffs.len()` contiguous rows of `row_len` bytes, and
/// `row_len == src.len()`.
pub fn scatter_neon(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    debug_assert_eq!(row_len, src.len());
    debug_assert!(rows.len() >= coeffs.len().saturating_mul(row_len));
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY: NEON is baseline on AArch64. `rows` is uniquely borrowed, and
    // the kernel clamps its row count to the number of whole rows the buffer
    // actually holds.
    unsafe { scatter_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "neon")]
unsafe fn scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let span = row_len.min(src.len());
    let nrows = coeffs.len().min(rows.len() / row_len);
    let base = rows.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= nrows {
        // SAFETY: `j + 4 <= nrows` and `nrows * row_len <= rows.len()`, so all
        // four pointers start in-bounds `row_len`-byte spans. Distinct row
        // indices name disjoint spans, so no store below aliases another
        // row's load.
        let quad = unsafe {
            [
                RowPlan::new(base.add(j * row_len), coeffs[j]),
                RowPlan::new(base.add((j + 1) * row_len), coeffs[j + 1]),
                RowPlan::new(base.add((j + 2) * row_len), coeffs[j + 2]),
                RowPlan::new(base.add((j + 3) * row_len), coeffs[j + 3]),
            ]
        };
        // SAFETY: as above, plus `span <= src.len()` bounds the source reads.
        unsafe { scatter_quad(&quad, src, span) };
        j += 4;
    }
    while j < nrows {
        let coeff = coeffs[j];
        if coeff != Elem::ZERO {
            // SAFETY: `(j + 1) * row_len <= rows.len()` and `span <= row_len`,
            // so this window is in bounds; rows are disjoint by construction,
            // so the `&mut` is unique and does not alias `src`.
            let row = unsafe { core::slice::from_raw_parts_mut(base.add(j * row_len), span) };
            // SAFETY: the `neon` feature is enabled for this whole function.
            unsafe { mul_add_impl(row, scale_table(coeff), &src[..span]) };
        }
        j += 1;
    }
}

/// Fold one source into four rows: one source load, four destination updates.
///
/// # Safety
/// Every plan pointer must start a writable span of at least `span` bytes,
/// the four spans must be pairwise disjoint and disjoint from `src`, and
/// `src` must hold at least `span` bytes.
#[target_feature(enable = "neon")]
unsafe fn scatter_quad(plans: &[RowPlan; 4], src: &[u8], span: usize) {
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset + 16 <= span {
        // SAFETY: `offset + 16 <= span <= src.len()`.
        let x = unsafe { vld1q_u8(src_ptr.add(offset)) };
        for plan in plans {
            if plan.scaling.kind == Kind::Skip {
                continue;
            }
            // SAFETY: `offset + 16 <= span` bounds this row, and the four rows
            // are disjoint, so this store cannot alias another row.
            unsafe {
                let dp = plan.ptr.add(offset);
                vst1q_u8(dp, plan.scaling.fold(vld1q_u8(dp), x));
            }
        }
        offset += 16;
    }

    if offset == span {
        return;
    }
    let tail = span - offset;
    for plan in plans {
        if plan.scaling.kind == Kind::Skip {
            continue;
        }
        // SAFETY: `offset + tail == span` bytes remain in this row, and the
        // rows are disjoint, so this `&mut` window is unique.
        let row = unsafe { core::slice::from_raw_parts_mut(plan.ptr.add(offset), tail) };
        mul_add_nibble(row, plan.table, &src[offset..span]);
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked: each group of four rows loads a 32-byte-per-row
/// destination tile into eight accumulators once, folds in every term, and
/// stores once. Destination memory traffic is therefore independent of
/// `terms.len()`, which is what separates this from a `scatter_neon` per term.
pub fn matrix_neon(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
    debug_assert!(rows.len() >= nrows.saturating_mul(row_len));
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    // SAFETY: NEON is baseline on AArch64. `rows` is uniquely borrowed, and
    // the kernel clamps both its row count and its byte span to what the
    // buffers actually hold.
    unsafe { matrix_impl(rows, row_len, nrows, terms) }
}

#[target_feature(enable = "neon")]
unsafe fn matrix_impl(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
    // One pass over `terms` — outside every hot loop — establishes the bounds
    // the raw-pointer loops rely on, so a caller that violates the documented
    // geometry gets a short update rather than out-of-bounds reads.
    let mut span = row_len;
    let mut count = nrows.min(rows.len() / row_len);
    for &(coeffs, src) in terms {
        debug_assert_eq!(coeffs.len(), nrows);
        debug_assert_eq!(src.len(), row_len);
        span = span.min(src.len());
        count = count.min(coeffs.len());
    }
    let base = rows.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= count {
        // SAFETY: `count * row_len <= rows.len()`, so each of rows `j..j + 4`
        // starts an in-bounds `row_len`-byte span; distinct indices give
        // disjoint spans, so the four accumulator streams never alias.
        let quad = unsafe {
            [
                base.add(j * row_len),
                base.add((j + 1) * row_len),
                base.add((j + 2) * row_len),
                base.add((j + 3) * row_len),
            ]
        };
        // SAFETY: as above; `span` bounds every source and every row, and
        // `j + 3 < count <= coeffs.len()` for every term.
        unsafe { matrix_quad(&quad, span, j, terms) };
        j += 4;
    }
    while j < count {
        // SAFETY: row `j` is an in-bounds `row_len`-byte span of `rows`, and
        // `j < count <= coeffs.len()` for every term.
        unsafe { matrix_single(base.add(j * row_len), span, j, terms) };
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold every term, store once.
///
/// # Safety
/// Each pointer in `rows` must start a writable span of at least `span` bytes,
/// the four spans must be pairwise disjoint and disjoint from every source,
/// every term's source must hold at least `span` bytes, and every term's
/// coefficient slice must have more than `first + 3` entries.
#[target_feature(enable = "neon")]
unsafe fn matrix_quad(rows: &[*mut u8; 4], span: usize, first: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 32 <= span {
        // SAFETY: `tile + 32 <= span` bounds all four rows, which are disjoint.
        let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = unsafe {
            (
                vld1q_u8(rows[0].add(tile)),
                vld1q_u8(rows[0].add(tile + 16)),
                vld1q_u8(rows[1].add(tile)),
                vld1q_u8(rows[1].add(tile + 16)),
                vld1q_u8(rows[2].add(tile)),
                vld1q_u8(rows[2].add(tile + 16)),
                vld1q_u8(rows[3].add(tile)),
                vld1q_u8(rows[3].add(tile + 16)),
            )
        };
        for &(coeffs, src) in terms {
            // Each row's tables are resolved once per term and reused for both
            // vectors of the tile; only one row's pair is live at a time,
            // which keeps the eight accumulators off the stack.
            // SAFETY: `tile + 32 <= span <= src.len()` bounds both source
            // loads.
            let (x0, x1) = unsafe {
                let sp = src.as_ptr().add(tile);
                (vld1q_u8(sp), vld1q_u8(sp.add(16)))
            };

            let s = Scaling::new(coeffs[first]);
            a00 = s.fold(a00, x0);
            a01 = s.fold(a01, x1);

            let s = Scaling::new(coeffs[first + 1]);
            a10 = s.fold(a10, x0);
            a11 = s.fold(a11, x1);

            let s = Scaling::new(coeffs[first + 2]);
            a20 = s.fold(a20, x0);
            a21 = s.fold(a21, x1);

            let s = Scaling::new(coeffs[first + 3]);
            a30 = s.fold(a30, x0);
            a31 = s.fold(a31, x1);
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            vst1q_u8(rows[0].add(tile), a00);
            vst1q_u8(rows[0].add(tile + 16), a01);
            vst1q_u8(rows[1].add(tile), a10);
            vst1q_u8(rows[1].add(tile + 16), a11);
            vst1q_u8(rows[2].add(tile), a20);
            vst1q_u8(rows[2].add(tile + 16), a21);
            vst1q_u8(rows[3].add(tile), a30);
            vst1q_u8(rows[3].add(tile + 16), a31);
        }
        tile += 32;
    }

    if tile + 16 <= span {
        // SAFETY: `tile + 16 <= span` bounds all four rows, which are disjoint.
        let (mut a0, mut a1, mut a2, mut a3) = unsafe {
            (
                vld1q_u8(rows[0].add(tile)),
                vld1q_u8(rows[1].add(tile)),
                vld1q_u8(rows[2].add(tile)),
                vld1q_u8(rows[3].add(tile)),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: `tile + 16 <= span <= src.len()`.
            let x = unsafe { vld1q_u8(src.as_ptr().add(tile)) };
            a0 = Scaling::new(coeffs[first]).fold(a0, x);
            a1 = Scaling::new(coeffs[first + 1]).fold(a1, x);
            a2 = Scaling::new(coeffs[first + 2]).fold(a2, x);
            a3 = Scaling::new(coeffs[first + 3]).fold(a3, x);
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            vst1q_u8(rows[0].add(tile), a0);
            vst1q_u8(rows[1].add(tile), a1);
            vst1q_u8(rows[2].add(tile), a2);
            vst1q_u8(rows[3].add(tile), a3);
        }
        tile += 16;
    }

    if tile < span {
        let tail = span - tile;
        for (slot, &ptr) in rows.iter().enumerate() {
            // SAFETY: `tail` bytes remain in this row, and the rows are
            // disjoint, so this `&mut` window is unique.
            let row = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), tail) };
            for &(coeffs, src) in terms {
                let coeff = coeffs[first + slot];
                if coeff != Elem::ZERO {
                    mul_add_nibble(row, scale_table(coeff), &src[tile..span]);
                }
            }
        }
    }
}

/// Register-blocked single row, for the 1-3 rows a group of four leaves over.
///
/// Still loads its destination once and folds in every term, so the blocking
/// property holds for the remainder as well.
///
/// # Safety
/// `ptr` must start a writable span of at least `span` bytes that no other row
/// and no source aliases, every term's source must hold at least `span` bytes,
/// and every term's coefficient slice must have more than `index` entries.
#[target_feature(enable = "neon")]
unsafe fn matrix_single(ptr: *mut u8, span: usize, index: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 32 <= span {
        // SAFETY: `tile + 32 <= span` bounds the row.
        let (mut a0, mut a1) = unsafe { (vld1q_u8(ptr.add(tile)), vld1q_u8(ptr.add(tile + 16))) };
        for &(coeffs, src) in terms {
            // SAFETY: `tile + 32 <= span <= src.len()`.
            let (x0, x1) = unsafe {
                let sp = src.as_ptr().add(tile);
                (vld1q_u8(sp), vld1q_u8(sp.add(16)))
            };
            let s = Scaling::new(coeffs[index]);
            a0 = s.fold(a0, x0);
            a1 = s.fold(a1, x1);
        }
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_u8(ptr.add(tile), a0);
            vst1q_u8(ptr.add(tile + 16), a1);
        }
        tile += 32;
    }

    if tile + 16 <= span {
        // SAFETY: `tile + 16 <= span` bounds the row.
        let mut a = unsafe { vld1q_u8(ptr.add(tile)) };
        for &(coeffs, src) in terms {
            // SAFETY: `tile + 16 <= span <= src.len()`.
            let x = unsafe { vld1q_u8(src.as_ptr().add(tile)) };
            a = Scaling::new(coeffs[index]).fold(a, x);
        }
        // SAFETY: same bounds as the load above.
        unsafe { vst1q_u8(ptr.add(tile), a) };
        tile += 16;
    }

    if tile < span {
        let tail = span - tile;
        // SAFETY: `tail` bytes remain in this row, which nothing else aliases.
        let row = unsafe { core::slice::from_raw_parts_mut(ptr.add(tile), tail) };
        for &(coeffs, src) in terms {
            let coeff = coeffs[index];
            if coeff != Elem::ZERO {
                mul_add_nibble(row, scale_table(coeff), &src[tile..span]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1: gather and varying-operand multiply.
// ---------------------------------------------------------------------------

/// Fold many sources into one destination while keeping each 32-byte
/// destination tile in registers across every source.
pub fn gather_neon(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: NEON is baseline on AArch64 and callers checked source lengths.
    unsafe { gather_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "neon")]
unsafe fn gather_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let len = dst.len() & !31;
    let dst_ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len()`.
        let (mut acc0, mut acc1) = unsafe {
            (
                vld1q_u8(dst_ptr.add(offset)),
                vld1q_u8(dst_ptr.add(offset + 16)),
            )
        };
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let scaling = Scaling::new(coeff);
            // SAFETY: every source is exactly `dst.len()` bytes.
            unsafe {
                acc0 = scaling.fold(acc0, vld1q_u8(src.as_ptr().add(offset)));
                acc1 = scaling.fold(acc1, vld1q_u8(src.as_ptr().add(offset + 16)));
            }
        }
        // SAFETY: the destination window loaded above.
        unsafe {
            vst1q_u8(dst_ptr.add(offset), acc0);
            vst1q_u8(dst_ptr.add(offset + 16), acc1);
        }
        offset += 32;
    }
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add_nibble(&mut dst[len..], scale_table(coeff), &src[len..]);
    }
}

/// Lane-parallel GF(2^8) multiplication for two varying byte vectors.
///
/// `PMULL` is in AArch64's optional crypto extension, not baseline NEON, so
/// the portable NEON backend uses eight branchless shift/reduce rounds. Every
/// round processes all 16 lanes and the scalar tail is shorter than one lane.
#[inline]
#[target_feature(enable = "neon")]
fn multiply_vectors(mut a: uint8x16_t, mut b: uint8x16_t) -> uint8x16_t {
    let one = vdupq_n_u8(1);
    let high_bit = vdupq_n_u8(0x80);
    let reduction = vdupq_n_u8(0x1b);
    let mut product = vdupq_n_u8(0);
    for _ in 0..8 {
        let active = vceqq_u8(vandq_u8(b, one), one);
        product = veorq_u8(product, vandq_u8(a, active));
        let reduce = vceqq_u8(vandq_u8(a, high_bit), high_bit);
        a = veorq_u8(vshlq_n_u8(a, 1), vandq_u8(reduction, reduce));
        b = vshrq_n_u8(b, 1);
    }
    product
}

/// Eight lane-independent carry-less byte products reduced by the AES
/// polynomial. `PMULL` widens each byte lane to 16 bits; two fixed folds of
/// the high half suffice because an 8-by-8 polynomial product has degree 14.
#[inline]
#[target_feature(enable = "neon,aes")]
fn multiply8_pmull(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let product = vreinterpretq_u16_p16(vmull_p8(vreinterpret_p8_u8(a), vreinterpret_p8_u8(b)));
    let mask = vdupq_n_u16(0x00ff);
    let high = vshrq_n_u16(product, 8);
    let mut reduced = veorq_u16(vandq_u16(product, mask), high);
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 1));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 3));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 4));
    let high = vshrq_n_u16(reduced, 8);
    reduced = veorq_u16(vandq_u16(reduced, mask), high);
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 1));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 3));
    reduced = veorq_u16(reduced, vshlq_n_u16(high, 4));
    vmovn_u16(reduced)
}

/// Sixteen lane-independent base-field products using two `PMULL`s.
#[inline]
#[target_feature(enable = "neon,aes")]
pub(super) fn multiply_vectors_pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vcombine_u8(
        multiply8_pmull(vget_low_u8(a), vget_low_u8(b)),
        multiply8_pmull(vget_high_u8(a), vget_high_u8(b)),
    )
}

/// `dst[i] = a[i] * b[i]` using the optional AArch64 crypto extension.
pub fn elementwise_pmull(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: dispatch checked the AArch64 `aes` feature, which includes
    // `PMULL`, and all three lengths match.
    unsafe { elementwise_pmull_impl(dst, a, b) }
}

#[target_feature(enable = "neon,aes")]
unsafe fn elementwise_pmull_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len`, which bounds all three slices.
        unsafe {
            let x = vld1q_u8(a.as_ptr().add(offset));
            let y = vld1q_u8(b.as_ptr().add(offset));
            vst1q_u8(dst.as_mut_ptr().add(offset), multiply_vectors_pmull(x, y));
        }
        offset += 16;
    }
    for ((d, &x), &y) in dst[len..].iter_mut().zip(&a[len..]).zip(&b[len..]) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}

/// `dst[i] = a[i] * b[i]` over byte field elements.
pub fn elementwise_neon(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: NEON is baseline on AArch64 and all three lengths match.
    unsafe { elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "neon")]
unsafe fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 16 <= len`, which bounds all three slices.
        unsafe {
            let x = vld1q_u8(a.as_ptr().add(offset));
            let y = vld1q_u8(b.as_ptr().add(offset));
            vst1q_u8(dst.as_mut_ptr().add(offset), multiply_vectors(x, y));
        }
        offset += 16;
    }
    for ((d, &x), &y) in dst[len..].iter_mut().zip(&a[len..]).zip(&b[len..]) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}
