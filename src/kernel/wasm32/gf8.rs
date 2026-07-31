//! GF(2^8) kernels using WebAssembly `simd128` swizzles.

use core::arch::wasm32::*;

use crate::field::gf8::{Elem, Gf8};
use crate::kernel::gf8::mul_add_nibble;
use crate::kernel::tables::{ScaleTable, scale_table};

#[derive(Clone, Copy)]
struct Factors {
    lo: v128,
    hi: v128,
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_factors(table: &ScaleTable) -> Factors {
    // SAFETY: both arrays contain exactly 16 readable bytes.
    unsafe {
        Factors {
            lo: v128_load(table.lo.as_ptr().cast()),
            hi: v128_load(table.hi.as_ptr().cast()),
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn scaled(value: v128, factors: Factors) -> v128 {
    let low = v128_and(value, u8x16_splat(0x0f));
    let high = u8x16_shr(value, 4);
    v128_xor(
        u8x16_swizzle(factors.lo, low),
        u8x16_swizzle(factors.hi, high),
    )
}

/// `dst ^= coeff * src` over 16-byte SIMD lanes.
///
/// Deliberately one lane per iteration. A two-lane unroll — the shape the
/// GF(2^16) kernels gain from on the same runtime — measured as no change
/// here (BENCHMARKS.md): two swizzles per lane is too little work to have any
/// latency left to hide.
pub fn mul_add_simd128(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_add_impl(dst, table, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_add_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len()) & !15;
    let factors = load_factors(table);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let d = v128_load(dst_ptr.add(offset).cast());
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), v128_xor(d, scaled(s, factors)));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_add_nibble(&mut dst[len..], table, &src[len..]);
}

/// `dst = coeff * dst` over 16-byte SIMD lanes.
pub fn mul_assign_simd128(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: the binary requires `simd128`.
    unsafe { mul_assign_impl(dst, table) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_assign_impl(dst: &mut [u8], table: &ScaleTable) {
    let len = dst.len() & !15;
    let factors = load_factors(table);
    let ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in `dst`.
        unsafe {
            let d = v128_load(ptr.add(offset).cast());
            v128_store(ptr.add(offset).cast(), scaled(d, factors));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_assign_nibble(&mut dst[len..], table);
}

/// `dst = coeff * src`, out of place, over 16-byte SIMD lanes.
///
/// Fuses what would otherwise be a copy then an in-place scale: one pass, no `dst` read.
pub fn mul_into_simd128(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the binary requires `simd128`, and slices are independently borrowed.
    unsafe { mul_into_impl(dst, table, src) }
}

#[target_feature(enable = "simd128")]
unsafe fn mul_into_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    let len = dst.len().min(src.len()) & !15;
    let factors = load_factors(table);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in both slices.
        unsafe {
            let s = v128_load(src_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), scaled(s, factors));
        }
        offset += 16;
    }
    crate::kernel::gf8::mul_into_nibble(&mut dst[len..], table, &src[len..]);
}

/// Lane-parallel multiply for two varying base-field vectors.
#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn multiply_vectors(mut a: v128, mut b: v128) -> v128 {
    let one = u8x16_splat(1);
    let reduction = u8x16_splat(0x1b);
    let mut product = u8x16_splat(0);
    for _ in 0..8 {
        let selected = u8x16_eq(v128_and(b, one), one);
        product = v128_xor(product, v128_and(a, selected));
        let high = u8x16_eq(u8x16_shr(a, 7), one);
        a = v128_xor(u8x16_shl(a, 1), v128_and(reduction, high));
        b = u8x16_shr(b, 1);
    }
    product
}

/// `dst[i] = a[i] * b[i]` over 16-byte SIMD lanes.
pub fn elementwise_simd128(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the binary requires `simd128`, and all geometry was validated.
    unsafe { elementwise_impl(dst, a, b) }
}

#[target_feature(enable = "simd128")]
unsafe fn elementwise_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !15;
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: one complete vector remains in all three slices.
        unsafe {
            let x = v128_load(a_ptr.add(offset).cast());
            let y = v128_load(b_ptr.add(offset).cast());
            v128_store(dst_ptr.add(offset).cast(), multiply_vectors(x, y));
        }
        offset += 16;
    }
    crate::kernel::scalar::mul_elementwise::<Gf8>(&mut dst[len..], &a[len..], &b[len..]);
}

/// Which of the three cases a row's coefficient falls into.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Coefficient zero: the row contributes nothing.
    Skip,
    /// Coefficient one: XOR the source in unscaled, no swizzle needed.
    Identity,
    /// Any other coefficient: two nibble swizzles and an XOR.
    Table,
}

/// One coefficient resolved into the form the multi-row loops consume.
///
/// Branching on [`Kind::Skip`] and [`Kind::Identity`] pays for itself: the
/// coefficient arrays handed to [`scatter_simd128`] and [`matrix_simd128`] are
/// full of zeros and ones, and each case removes two `i8x16.swizzle`s per lane.
#[derive(Clone, Copy)]
struct Scaling {
    /// Nibble tables in registers. Meaningless unless `kind` is [`Kind::Table`].
    factors: Factors,
    /// The same tables in memory, for the element-aligned tail.
    table: &'static ScaleTable,
    /// The case this coefficient falls into.
    kind: Kind,
}

impl Scaling {
    /// Resolve `coeff` against the shared table bank.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn new(coeff: Elem) -> Self {
        let table = scale_table(coeff);
        let kind = if coeff == Elem::ZERO {
            Kind::Skip
        } else if coeff == Elem::ONE {
            Kind::Identity
        } else {
            Kind::Table
        };
        Self {
            factors: load_factors(table),
            table,
            kind,
        }
    }

    /// `acc ^= coeff * x` for one 16-byte lane.
    #[inline]
    #[target_feature(enable = "simd128")]
    fn fold(self, acc: v128, x: v128) -> v128 {
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
        let coeff = coeffs[j];
        if coeff != Elem::ZERO {
            // SAFETY: `simd128` is enabled for this whole function, and row
            // and source windows are both exactly `span` bytes.
            unsafe { mul_add_impl(&mut row[..span], scale_table(coeff), src) };
        }
        j += 1;
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
        if plan.kind == Kind::Skip {
            continue;
        }
        mul_add_nibble(&mut row[offset..], plan.table, &src[offset..]);
    }
}

/// `dst ^= sum(coeffs[k] * srcs[k])`, holding a 32-byte destination tile in
/// registers across every source so `dst` is read and written once.
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

    let mut offset = 0;
    while offset + 32 <= span {
        // SAFETY: `offset + 32 <= span <= dst.len()`.
        let (mut a0, mut a1) = unsafe {
            let dp = dst_ptr.add(offset);
            (v128_load(dp.cast()), v128_load(dp.add(16).cast()))
        };
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            // SAFETY: `offset + 32 <= span <= srcs[k].len()`.
            unsafe {
                let sp = srcs[k].as_ptr().add(offset);
                a0 = plan.fold(a0, v128_load(sp.cast()));
                a1 = plan.fold(a1, v128_load(sp.add(16).cast()));
            }
        }
        // SAFETY: same bounds as the load above; `dst` is uniquely borrowed
        // and the sources are read-only, so no store aliases a live read.
        unsafe {
            let dp = dst_ptr.add(offset);
            v128_store(dp.cast(), a0);
            v128_store(dp.add(16).cast(), a1);
        }
        offset += 32;
    }
    if offset + 16 <= span {
        // SAFETY: `offset + 16 <= span <= dst.len()`.
        let mut a = unsafe { v128_load(dst_ptr.add(offset).cast()) };
        for k in 0..count {
            let plan = Scaling::new(coeffs[k]);
            if plan.kind == Kind::Skip {
                continue;
            }
            // SAFETY: `offset + 16 <= span <= srcs[k].len()`.
            unsafe { a = plan.fold(a, v128_load(srcs[k].as_ptr().add(offset).cast())) };
        }
        // SAFETY: as above.
        unsafe { v128_store(dst_ptr.add(offset).cast(), a) };
        offset += 16;
    }

    for k in 0..count {
        let coeff = coeffs[k];
        if coeff != Elem::ZERO {
            mul_add_nibble(
                &mut dst[offset..span],
                scale_table(coeff),
                &srcs[k][offset..span],
            );
        }
    }
}

/// Apply every `(coeffs, src)` term to the first `nrows` contiguous rows.
///
/// Register-blocked: each group of four rows loads a 16-byte-per-row
/// destination tile into four accumulators once, folds in every term, and
/// stores once, so destination traffic is independent of `terms.len()`.
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
            let coeff = coeffs[j];
            if coeff != Elem::ZERO {
                // SAFETY: `simd128` is enabled for this whole function, and
                // both windows are exactly `span` bytes.
                unsafe { mul_add_impl(&mut row[..span], scale_table(coeff), &src[..span]) };
            }
        }
        j += 1;
    }
}

/// Register-blocked four-row tile: load once, fold every term, store once.
///
/// # Safety
/// Every row must hold the same number of bytes, no more than the length of
/// any term's source, and every term's coefficient slice must have more than
/// `first + 3` entries.
#[target_feature(enable = "simd128")]
unsafe fn matrix_quad(mut rows: [&mut [u8]; 4], first: usize, terms: &[(&[Elem], &[u8])]) {
    let span = rows[0].len();

    let mut offset = 0;
    while offset + 16 <= span {
        // SAFETY: `offset + 16 <= span`, the common length of all four rows,
        // which are disjoint.
        let mut acc = unsafe {
            [
                v128_load(rows[0].as_ptr().add(offset).cast()),
                v128_load(rows[1].as_ptr().add(offset).cast()),
                v128_load(rows[2].as_ptr().add(offset).cast()),
                v128_load(rows[3].as_ptr().add(offset).cast()),
            ]
        };
        for &(coeffs, src) in terms {
            // SAFETY: `offset + 16 <= span <= src.len()`.
            let x = unsafe { v128_load(src.as_ptr().add(offset).cast()) };
            for (a, &coeff) in acc.iter_mut().zip(&coeffs[first..first + 4]) {
                *a = Scaling::new(coeff).fold(*a, x);
            }
        }
        for (row, &a) in rows.iter_mut().zip(&acc) {
            // SAFETY: same bounds as the loads; the rows are disjoint and the
            // sources are read-only.
            unsafe { v128_store(row.as_mut_ptr().add(offset).cast(), a) };
        }
        offset += 16;
    }

    for (i, row) in rows.iter_mut().enumerate() {
        for &(coeffs, src) in terms {
            let coeff = coeffs[first + i];
            if coeff != Elem::ZERO {
                mul_add_nibble(&mut row[offset..], scale_table(coeff), &src[offset..span]);
            }
        }
    }
}
