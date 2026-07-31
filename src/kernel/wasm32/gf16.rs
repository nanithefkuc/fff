//! GF(2^16) tower kernels using WebAssembly `simd128` swizzles.

use core::arch::wasm32::*;

use crate::field::gf16::{Elem, Gf16};
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::TowerTables;
use crate::kernel::wasm32::gf8::multiply_vectors;

/// How many terms the multi-row kernels prepare before sweeping bytes.
///
/// A GF(2^16) coefficient costs four derived nibble tables, so preparation
/// must never happen inside a byte loop. Terms are therefore handled in
/// bounded blocks whose lookup vectors live in a stack array — no allocation,
/// and one preparation per `(term, row)` pair per kernel call.
const TERM_BLOCK: usize = 8;

#[derive(Clone, Copy)]
struct Factors {
    lo: [v128; 4],
    hi: [v128; 4],
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_factors(tables: &TowerTables) -> Factors {
    let factors = &tables.factors;
    // SAFETY: every table half contains exactly 16 readable bytes.
    unsafe {
        Factors {
            lo: core::array::from_fn(|i| v128_load(factors[i].lo.as_ptr().cast())),
            hi: core::array::from_fn(|i| v128_load(factors[i].hi.as_ptr().cast())),
        }
    }
}

/// A `Factors` whose every product is zero, for slots no lookup will read.
#[inline]
#[target_feature(enable = "simd128")]
fn empty_factors() -> Factors {
    let zero = u8x16_splat(0);
    Factors {
        lo: [zero; 4],
        hi: [zero; 4],
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn swap_adjacent(value: v128) -> v128 {
    const SWAP: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];
    // SAFETY: `SWAP` contains exactly 16 readable bytes.
    let mask = unsafe { v128_load(SWAP.as_ptr().cast()) };
    u8x16_swizzle(value, mask)
}

#[inline]
#[target_feature(enable = "simd128")]
fn lookup(lo: v128, hi: v128, low: v128, high: v128) -> v128 {
    v128_xor(u8x16_swizzle(lo, low), u8x16_swizzle(hi, high))
}

#[inline]
#[target_feature(enable = "simd128")]
fn scaled(value: v128, factors: Factors) -> v128 {
    let nibble = u8x16_splat(0x0f);
    let swapped = swap_adjacent(value);
    let low = v128_and(value, nibble);
    let high = u8x16_shr(value, 4);
    let swapped_low = v128_and(swapped, nibble);
    let swapped_high = u8x16_shr(swapped, 4);
    let direct_even = lookup(factors.lo[0], factors.hi[0], low, high);
    let direct_odd = lookup(factors.lo[1], factors.hi[1], low, high);
    let cross_even = lookup(factors.lo[2], factors.hi[2], swapped_low, swapped_high);
    let cross_odd = lookup(factors.lo[3], factors.hi[3], swapped_low, swapped_high);
    let even_lanes = u16x8_splat(0x00ff);
    v128_bitselect(
        v128_xor(direct_even, cross_even),
        v128_xor(direct_odd, cross_odd),
        even_lanes,
    )
}

/// `dst ^= coeff * src` over interleaved tower elements, two lanes at a time.
pub fn mul_add_simd128(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_add_impl(dst, tables, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_add_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let factors = load_factors(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    // Eight swizzles per lane leave plenty of room for a second independent
    // lane's loads and splits, so the block is 32 bytes wide.
    while offset + 32 <= span {
        // SAFETY: `offset + 32 <= span <= min(dst.len(), src.len())`.
        unsafe {
            let dp = dst_ptr.add(offset);
            let sp = src_ptr.add(offset);
            let d0 = v128_load(dp.cast());
            let d1 = v128_load(dp.add(16).cast());
            let s0 = v128_load(sp.cast());
            let s1 = v128_load(sp.add(16).cast());
            v128_store(dp.cast(), v128_xor(d0, scaled(s0, factors)));
            v128_store(dp.add(16).cast(), v128_xor(d1, scaled(s1, factors)));
        }
        offset += 32;
    }
    if offset + 16 <= span {
        // SAFETY: `offset + 16 <= span <= min(dst.len(), src.len())`.
        unsafe {
            let d = v128_load(dst_ptr.add(offset).cast());
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), v128_xor(d, scaled(s, factors)));
        }
        offset += 16;
    }
    mul_add_scalar(&mut dst[offset..span], tables.coeff, &src[offset..span]);
}

/// `dst = coeff * dst` over interleaved tower elements.
pub fn mul_assign_simd128(dst: &mut [u8], tables: &TowerTables) {
    // SAFETY: the binary requires `simd128`.
    unsafe { mul_assign_impl(dst, tables) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_assign_impl(dst: &mut [u8], tables: &TowerTables) {
    let len = dst.len();
    let factors = load_factors(tables);
    let ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len()`.
        unsafe {
            let dp = ptr.add(offset);
            let d0 = v128_load(dp.cast());
            let d1 = v128_load(dp.add(16).cast());
            v128_store(dp.cast(), scaled(d0, factors));
            v128_store(dp.add(16).cast(), scaled(d1, factors));
        }
        offset += 32;
    }
    if offset + 16 <= len {
        // SAFETY: `offset + 16 <= len == dst.len()`.
        unsafe {
            let d = v128_load(ptr.add(offset).cast());
            v128_store(ptr.add(offset).cast(), scaled(d, factors));
        }
        offset += 16;
    }
    mul_assign_scalar(&mut dst[offset..], tables.coeff);
}

/// `dst = coeff * src` out of place over interleaved tower elements.
///
/// Fuses what would otherwise be a copy followed by an in-place scale: one
/// pass over the destination, which is never read.
pub fn mul_into_simd128(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_into_impl(dst, tables, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_into_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let span = dst.len().min(src.len());
    let factors = load_factors(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset + 32 <= span {
        // SAFETY: `offset + 32 <= span <= min(dst.len(), src.len())`.
        unsafe {
            let dp = dst_ptr.add(offset);
            let sp = src_ptr.add(offset);
            let s0 = v128_load(sp.cast());
            let s1 = v128_load(sp.add(16).cast());
            v128_store(dp.cast(), scaled(s0, factors));
            v128_store(dp.add(16).cast(), scaled(s1, factors));
        }
        offset += 32;
    }
    if offset + 16 <= span {
        // SAFETY: `offset + 16 <= span <= min(dst.len(), src.len())`.
        unsafe {
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), scaled(s, factors));
        }
        offset += 16;
    }
    mul_into_scalar(&mut dst[offset..span], tables.coeff, &src[offset..span]);
}

/// `dst[i] = a[i] * b[i]` over interleaved tower elements.
pub fn elementwise_simd128(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the binary requires `simd128`, and all geometry was validated.
    unsafe { elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "simd128")]
unsafe fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let even = u16x8_splat(0x00ff);
    let delta_even = u16x8_splat(u16::from_le_bytes([crate::field::gf16::DELTA.0, 0]));
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = v128_load(a_ptr.add(offset).cast());
            let y = v128_load(b_ptr.add(offset).cast());
            let direct = multiply_vectors(x, y);
            let crossed = multiply_vectors(x, swap_adjacent(y));
            let delta_bd = multiply_vectors(swap_adjacent(direct), delta_even);
            let constant = v128_xor(direct, delta_bd);
            let extension = v128_xor(v128_xor(crossed, swap_adjacent(crossed)), direct);
            v128_store(
                dst_ptr.add(offset).cast(),
                v128_bitselect(constant, extension, even),
            );
        }
        offset += 16;
    }
    crate::kernel::scalar::mul_elementwise::<Gf16>(&mut dst[len..], &a[len..], &b[len..]);
}

/// Which of the three cases a coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: nothing to fold in.
    Skip,
    /// Coefficient one: a plain XOR, no swizzle and no tables.
    Identity,
    /// Anything else: the full eight-swizzle tower multiply.
    Table,
}

/// One coefficient resolved into the form the multi-row loops consume.
///
/// Resolving is what the naive per-row loops kept redoing: `Kind::Skip` and
/// `Kind::Identity` — both common in coding matrices — skip the four-table
/// derivation entirely, and `Kind::Table` pays for it once per kernel call
/// instead of once per `(term, row, block)`.
#[derive(Clone, Copy)]
struct Scaling {
    /// Lookup vectors. Meaningless unless `kind` is [`Kind::Table`].
    factors: Factors,
    /// The coefficient itself, for the element-aligned tail.
    coeff: Elem,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff`, deriving nibble tables only when they will be used.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn new(coeff: Elem) -> Self {
        if coeff == Elem::ZERO {
            return Self {
                factors: empty_factors(),
                coeff,
                kind: Kind::Skip,
            };
        }
        if coeff == Elem::ONE {
            return Self {
                factors: empty_factors(),
                coeff,
                kind: Kind::Identity,
            };
        }
        Self {
            factors: load_factors(&TowerTables::new(coeff)),
            coeff,
            kind: Kind::Table,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn fold(&self, acc: v128, x: v128) -> v128 {
        match self.kind {
            Kind::Skip => acc,
            Kind::Identity => v128_xor(acc, x),
            Kind::Table => v128_xor(acc, scaled(x, self.factors)),
        }
    }
}

/// `rows[j] ^= coeffs[j] * src` for every row, four rows per source load.
///
/// `rows` is `coeffs.len()` contiguous rows of `row_len` bytes, and
/// `row_len == src.len()`.
pub fn scatter_simd128(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    debug_assert_eq!(row_len, src.len());
    debug_assert!(rows.len() >= coeffs.len().saturating_mul(row_len));
    if row_len == 0 || coeffs.is_empty() {
        return;
    }
    // SAFETY: the binary requires `simd128`, and the kernel clamps its row
    // count and byte span to what the buffers actually hold.
    unsafe { scatter_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn scatter_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let span = row_len.min(src.len());
    let count = coeffs.len().min(rows.len() / row_len);
    let src = &src[..span];
    let mut rest = &mut rows[..count * row_len];

    let mut j = 0;
    while j + 4 <= count {
        let (block, tail) = rest.split_at_mut(4 * row_len);
        rest = tail;
        let (r0, block) = block.split_at_mut(row_len);
        let (r1, block) = block.split_at_mut(row_len);
        let (r2, r3) = block.split_at_mut(row_len);
        let plans = [
            Scaling::new(coeffs[j]),
            Scaling::new(coeffs[j + 1]),
            Scaling::new(coeffs[j + 2]),
            Scaling::new(coeffs[j + 3]),
        ];
        // SAFETY: `split_at_mut` gives four disjoint rows, each truncated to
        // the `span` bytes the source also holds.
        unsafe {
            scatter_quad(
                [
                    &mut r0[..span],
                    &mut r1[..span],
                    &mut r2[..span],
                    &mut r3[..span],
                ],
                &plans,
                src,
            );
        }
        j += 4;
    }
    while j < count {
        let (row, tail) = rest.split_at_mut(row_len);
        rest = tail;
        let plan = Scaling::new(coeffs[j]);
        match plan.kind {
            Kind::Skip => {}
            // SAFETY (both arms): `simd128` is enabled for this whole
            // function, and the row and source windows are both `span` bytes.
            Kind::Identity => unsafe { xor_row(&mut row[..span], src) },
            Kind::Table => unsafe {
                mul_add_impl(&mut row[..span], &TowerTables::new(plan.coeff), src);
            },
        }
        j += 1;
    }
}

/// `dst ^= src`, the whole job when a coefficient is one.
///
/// # Safety
/// `simd128` must be available; both slices must hold the same length.
#[target_feature(enable = "simd128")]
unsafe fn xor_row(dst: &mut [u8], src: &[u8]) {
    let span = dst.len().min(src.len());
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset + 16 <= span {
        // SAFETY: `offset + 16 <= span`, which bounds both slices.
        unsafe {
            let dp = dst_ptr.add(offset).cast();
            v128_store(
                dp,
                v128_xor(v128_load(dp), v128_load(src_ptr.add(offset).cast())),
            );
        }
        offset += 16;
    }
    for (d, &s) in dst[offset..span].iter_mut().zip(&src[offset..span]) {
        *d ^= s;
    }
}

/// Fold one source into four rows: one source load, four destination updates.
///
/// # Safety
/// Every row and `src` must hold exactly the same number of bytes.
#[target_feature(enable = "simd128")]
unsafe fn scatter_quad(mut rows: [&mut [u8]; 4], plans: &[Scaling; 4], src: &[u8]) {
    let span = src.len();
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset + 16 <= span {
        // SAFETY: `offset + 16 <= span == src.len()`.
        let x = unsafe { v128_load(src_ptr.add(offset).cast()) };
        for (row, plan) in rows.iter_mut().zip(plans) {
            if plan.kind == Kind::Skip {
                continue;
            }
            // SAFETY: every row holds `span` bytes and the four rows are
            // disjoint, so this store cannot alias another row or the source.
            unsafe {
                let dp = row.as_mut_ptr().add(offset).cast();
                v128_store(dp, plan.fold(v128_load(dp), x));
            }
        }
        offset += 16;
    }

    for (row, plan) in rows.iter_mut().zip(plans) {
        mul_add_scalar(&mut row[offset..], plan.coeff, &src[offset..]);
    }
}

/// `dst ^= sum(coeffs[k] * srcs[k])`, holding a 32-byte destination tile in
/// registers across a block of sources so `dst` is touched once per block.
///
/// Sources are handled [`TERM_BLOCK`] at a time: their coefficients are
/// resolved once per block, before the byte loop, so the four-table
/// derivation never repeats per tile.
pub fn gather_simd128(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    if dst.is_empty() || coeffs.is_empty() {
        return;
    }
    // SAFETY: the binary requires `simd128`, and the kernel clamps its byte
    // span to the shortest buffer involved.
    unsafe { gather_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "simd128")]
unsafe fn gather_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let count = coeffs.len().min(srcs.len());
    let mut span = dst.len();
    for &src in &srcs[..count] {
        span = span.min(src.len());
    }
    let dst_ptr = dst.as_mut_ptr();

    for first in (0..count).step_by(TERM_BLOCK) {
        let block = (count - first).min(TERM_BLOCK);
        let mut plans = [Scaling::new(Elem::ZERO); TERM_BLOCK];
        for (plan, &coeff) in plans[..block].iter_mut().zip(&coeffs[first..first + block]) {
            *plan = Scaling::new(coeff);
        }

        let mut offset = 0;
        while offset + 32 <= span {
            // SAFETY: `offset + 32 <= span <= dst.len()`.
            let (mut a0, mut a1) = unsafe {
                let dp = dst_ptr.add(offset);
                (v128_load(dp.cast()), v128_load(dp.add(16).cast()))
            };
            for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
                if plan.kind == Kind::Skip {
                    continue;
                }
                // SAFETY: `offset + 32 <= span <= src.len()`.
                unsafe {
                    let sp = src.as_ptr().add(offset);
                    a0 = plan.fold(a0, v128_load(sp.cast()));
                    a1 = plan.fold(a1, v128_load(sp.add(16).cast()));
                }
            }
            // SAFETY: same bounds as the load above; `dst` is uniquely
            // borrowed and the sources are read-only.
            unsafe {
                let dp = dst_ptr.add(offset);
                v128_store(dp.cast(), a0);
                v128_store(dp.add(16).cast(), a1);
            }
            offset += 32;
        }
        if offset + 16 <= span {
            // SAFETY: `offset + 16 <= span <= dst.len()`.
            let mut acc = unsafe { v128_load(dst_ptr.add(offset).cast()) };
            for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
                if plan.kind == Kind::Skip {
                    continue;
                }
                // SAFETY: `offset + 16 <= span <= src.len()`.
                unsafe { acc = plan.fold(acc, v128_load(src.as_ptr().add(offset).cast())) };
            }
            // SAFETY: as above.
            unsafe { v128_store(dst_ptr.add(offset).cast(), acc) };
            offset += 16;
        }

        for (plan, &src) in plans[..block].iter().zip(&srcs[first..first + block]) {
            mul_add_scalar(&mut dst[offset..span], plan.coeff, &src[offset..span]);
        }
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked in both directions: each group of four rows loads a
/// 16-byte-per-row destination tile once, folds in a block of terms, and
/// stores once. Every `(term, row)` coefficient in the block is resolved
/// before the byte loop, so nothing is derived per tile.
pub fn matrix_simd128(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
    debug_assert!(rows.len() >= nrows.saturating_mul(row_len));
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    // SAFETY: the binary requires `simd128`, and the kernel clamps both its
    // row count and its byte span to what the buffers actually hold.
    unsafe { matrix_impl(rows, row_len, nrows, terms) }
}

#[target_feature(enable = "simd128")]
unsafe fn matrix_impl(rows: &mut [u8], row_len: usize, nrows: usize, terms: &[(&[Elem], &[u8])]) {
    // One pass over `terms` — outside every hot loop — establishes the bounds
    // the vector loops rely on, so a caller that violates the documented
    // geometry gets a short update rather than out-of-bounds reads.
    let mut span = row_len;
    let mut count = nrows.min(rows.len() / row_len);
    for &(coeffs, src) in terms {
        debug_assert_eq!(coeffs.len(), nrows);
        debug_assert_eq!(src.len(), row_len);
        span = span.min(src.len());
        count = count.min(coeffs.len());
    }
    let mut rest = &mut rows[..count * row_len];

    let mut j = 0;
    while j + 4 <= count {
        let (block, tail) = rest.split_at_mut(4 * row_len);
        rest = tail;
        let (r0, block) = block.split_at_mut(row_len);
        let (r1, block) = block.split_at_mut(row_len);
        let (r2, r3) = block.split_at_mut(row_len);
        // SAFETY: `split_at_mut` gives four disjoint rows, each truncated to
        // the `span` bytes every source also holds, and `j + 3 < count`
        // indexes every term's coefficients.
        unsafe {
            matrix_quad(
                [
                    &mut r0[..span],
                    &mut r1[..span],
                    &mut r2[..span],
                    &mut r3[..span],
                ],
                j,
                terms,
            );
        }
        j += 4;
    }
    while j < count {
        let (row, tail) = rest.split_at_mut(row_len);
        rest = tail;
        for &(coeffs, src) in terms {
            let plan = Scaling::new(coeffs[j]);
            match plan.kind {
                Kind::Skip => {}
                // SAFETY (both arms): `simd128` is enabled for this whole
                // function and both windows are exactly `span` bytes.
                Kind::Identity => unsafe { xor_row(&mut row[..span], &src[..span]) },
                Kind::Table => unsafe {
                    mul_add_impl(
                        &mut row[..span],
                        &TowerTables::new(plan.coeff),
                        &src[..span],
                    );
                },
            }
        }
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold a block of terms, store
/// once.
///
/// # Safety
/// Every row must hold the same number of bytes, no more than the length of
/// any term's source, and every term's coefficient slice must have more than
/// `first + 3` entries.
#[target_feature(enable = "simd128")]
unsafe fn matrix_quad(mut rows: [&mut [u8]; 4], first: usize, terms: &[(&[Elem], &[u8])]) {
    let span = rows[0].len();

    for block in terms.chunks(TERM_BLOCK) {
        let mut plans = [[Scaling::new(Elem::ZERO); 4]; TERM_BLOCK];
        for (&(coeffs, _), slots) in block.iter().zip(&mut plans) {
            for (slot, &coeff) in slots.iter_mut().zip(&coeffs[first..first + 4]) {
                *slot = Scaling::new(coeff);
            }
        }

        let mut offset = 0;
        while offset + 16 <= span {
            // SAFETY: `offset + 16 <= span`, the common length of all four
            // rows, which are disjoint.
            let mut acc = unsafe {
                [
                    v128_load(rows[0].as_ptr().add(offset).cast()),
                    v128_load(rows[1].as_ptr().add(offset).cast()),
                    v128_load(rows[2].as_ptr().add(offset).cast()),
                    v128_load(rows[3].as_ptr().add(offset).cast()),
                ]
            };
            for (&(_, src), slots) in block.iter().zip(&plans) {
                // SAFETY: `offset + 16 <= span <= src.len()`.
                let x = unsafe { v128_load(src.as_ptr().add(offset).cast()) };
                for (a, plan) in acc.iter_mut().zip(slots) {
                    *a = plan.fold(*a, x);
                }
            }
            for (row, &a) in rows.iter_mut().zip(&acc) {
                // SAFETY: same bounds as the loads; the rows are disjoint and
                // the sources are read-only.
                unsafe { v128_store(row.as_mut_ptr().add(offset).cast(), a) };
            }
            offset += 16;
        }

        for (i, row) in rows.iter_mut().enumerate() {
            for &(coeffs, src) in block {
                mul_add_scalar(&mut row[offset..], coeffs[first + i], &src[offset..span]);
            }
        }
    }
}
