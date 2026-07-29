//! GF(2^8) kernels for x86 and `x86_64`.
//!
//! Two multiply strategies, selected by the
//! [`Backend`](crate::kernel::Backend) the caller already resolved:
//!
//! - **GFNI.** `GF2P8MULB` is a native `GF(2)[x] / 0x11B` multiply across 32
//!   byte lanes, so a coefficient is nothing but a broadcast byte. It is
//!   pipelined but not single-cycle, which shapes the loops below: the
//!   single-buffer AXPY keeps four independent multiply chains in flight, and
//!   the blocked kernels hold a destination tile in registers so the
//!   multiplier, not memory, is the limit.
//! - **Nibble shuffle (AVX2, SSSE3).** With no byte-wide multiply, `c * x`
//!   splits into `c * (x & 0xf) ^ c * (x & 0xf0)`, two `PSHUFB` lookups
//!   against a [`ScaleTable`]. `PSHUFB` indexes within each 128-bit half, so
//!   the AVX2 form broadcasts the 16-byte table into both halves first.
//!
//! Buffer lengths are arbitrary, so x86 kernels descend through 32-byte and
//! 16-byte SIMD lanes before handing only the sub-XMM remainder to the scalar
//! nibble kernels in [`crate::kernel::gf8`].

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::field::gf8::Elem;
use crate::kernel::gf8::{mul_add_nibble, mul_assign_nibble, mul_into_nibble};
use crate::kernel::tables::{ScaleTable, scale_table};

// ---------------------------------------------------------------------------
// Single buffer: GFNI.
// ---------------------------------------------------------------------------

/// `dst ^= coeff * src` using `GF2P8MULB` over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_add_gfni(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the GFNI backend, which detected both AVX2
    // and GFNI; the slices are equal-length and independently borrowed.
    unsafe { mul_add_gfni_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_add_gfni_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);
    let len = dst.len();
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    // Four independent multiply chains per iteration. A single destination
    // AXPY is latency-bound on `GF2P8MULB`, and the unroll is what covers it.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len == dst.len() == src.len()`, so all
        // twelve unaligned accesses stay inside their slice.
        unsafe {
            let sp = src_ptr.add(offset);
            let dp = dst_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            let d0 = _mm256_loadu_si256(dp.cast());
            let d1 = _mm256_loadu_si256(dp.add(32).cast());
            let d2 = _mm256_loadu_si256(dp.add(64).cast());
            let d3 = _mm256_loadu_si256(dp.add(96).cast());
            let r0 = _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x0, factor));
            let r1 = _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x1, factor));
            let r2 = _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x2, factor));
            let r3 = _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x3, factor));
            _mm256_storeu_si256(dp.cast(), r0);
            _mm256_storeu_si256(dp.add(32).cast(), r1);
            _mm256_storeu_si256(dp.add(64).cast(), r2);
            _mm256_storeu_si256(dp.add(96).cast(), r3);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len() == src.len()`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let r = _mm256_xor_si256(d, _mm256_gf2p8mul_epi8(x, factor));
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), r);
        }
        offset += 32;
    }
    if offset + 16 <= len {
        // SAFETY: `offset + 16 <= len == dst.len() == src.len()`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let r = _mm_xor_si128(d, _mm_gf2p8mul_epi8(x, factor128));
            _mm_storeu_si128(dst_ptr.add(offset).cast(), r);
        }
        offset += 16;
    }

    mul_add_nibble(&mut dst[offset..], scale_table(coeff), &src[offset..]);
}

/// `dst = coeff * dst` using `GF2P8MULB` over 32-byte lanes.
pub(crate) fn mul_assign_gfni(dst: &mut [u8], coeff: Elem) {
    // SAFETY: the caller selected the GFNI backend, which detected both AVX2
    // and GFNI.
    unsafe { mul_assign_gfni_impl(dst, coeff) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_assign_gfni_impl(dst: &mut [u8], coeff: Elem) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);
    let len = dst.len() & !31;
    let dst_ptr = dst.as_mut_ptr();
    let mut offset = 0;
    // In-place scaling is store-bound and rare next to the AXPY shapes, so
    // one accumulator is enough; the loads have no dependency to cover.
    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len()`.
        unsafe {
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_gf2p8mul_epi8(d, factor));
        }
        offset += 32;
    }
    let mut offset = len;
    if offset + 16 <= dst.len() {
        // SAFETY: `offset + 16 <= dst.len()`.
        unsafe {
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_gf2p8mul_epi8(d, factor128));
        }
        offset += 16;
    }

    mul_assign_nibble(&mut dst[offset..], scale_table(coeff));
}

/// `dst = coeff * src` using `GF2P8MULB` over 32-byte lanes.
///
/// Fused out-of-place multiply: one pass, one read of `src` and one write of
/// `dst`, versus the copy-then-scale pair the trait default runs. On large
/// buffers that halves destination traffic and is worth roughly 2x.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_into_gfni(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the GFNI backend, which detected both AVX2
    // and GFNI; the slices are equal-length and independently borrowed.
    unsafe { mul_into_gfni_impl(dst, coeff, src) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_into_gfni_impl(dst: &mut [u8], coeff: Elem, src: &[u8]) {
    let factor = _mm256_set1_epi8(coeff.0.cast_signed());
    let factor128 = _mm256_castsi256_si128(factor);
    let len = dst.len();
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    // Four independent multiply chains, as in the AXPY: `GF2P8MULB` is
    // pipelined, and with no destination read there is even less other work
    // to hide its latency behind.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len == dst.len() == src.len()`.
        unsafe {
            let sp = src_ptr.add(offset);
            let dp = dst_ptr.add(offset);
            let r0 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(sp.cast()), factor);
            let r1 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(sp.add(32).cast()), factor);
            let r2 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(sp.add(64).cast()), factor);
            let r3 = _mm256_gf2p8mul_epi8(_mm256_loadu_si256(sp.add(96).cast()), factor);
            _mm256_storeu_si256(dp.cast(), r0);
            _mm256_storeu_si256(dp.add(32).cast(), r1);
            _mm256_storeu_si256(dp.add(64).cast(), r2);
            _mm256_storeu_si256(dp.add(96).cast(), r3);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len() == src.len()`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_gf2p8mul_epi8(x, factor));
        }
        offset += 32;
    }
    if offset + 16 <= len {
        // SAFETY: `offset + 16 <= len == dst.len() == src.len()`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_gf2p8mul_epi8(x, factor128));
        }
        offset += 16;
    }

    mul_into_nibble(&mut dst[offset..], scale_table(coeff), &src[offset..]);
}

// ---------------------------------------------------------------------------
// Single buffer: nibble shuffle.
// ---------------------------------------------------------------------------

/// `dst ^= coeff * src` by nibble shuffle over 32-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_add_avx2(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the AVX2 backend; the slices are
    // equal-length and independently borrowed.
    unsafe { mul_add_avx2_impl(dst, table, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_add_avx2_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast())),
        )
    };
    let mask = _mm256_set1_epi8(0x0f);
    let len = dst.len() & !31;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len() == src.len()`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
            let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
            let product = _mm256_xor_si256(lo, hi);
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(d, product));
        }
        offset += 32;
    }

    // SAFETY: AVX2 implies SSSE3, and the remainders keep equal lengths.
    unsafe { mul_add_ssse3_impl(&mut dst[len..], table, &src[len..]) }
}

/// `dst ^= coeff * src` by nibble shuffle over 16-byte lanes.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_add_ssse3(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the SSSE3 backend; the slices are
    // equal-length and independently borrowed.
    unsafe { mul_add_ssse3_impl(dst, table, src) }
}

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn mul_add_ssse3_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm_loadu_si128(table.lo.as_ptr().cast()),
            _mm_loadu_si128(table.hi.as_ptr().cast()),
        )
    };
    let mask = _mm_set1_epi8(0x0f);
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
            let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
            let product = _mm_xor_si128(lo, hi);
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, product));
        }
        offset += 16;
    }

    mul_add_nibble(&mut dst[len..], table, &src[len..]);
}

/// `dst = coeff * src` by nibble shuffle over 32-byte lanes.
///
/// Fused out-of-place multiply: the `mul_add` body without the destination
/// read, so one pass instead of copy-then-scale.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_into_avx2(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the AVX2 backend; the slices are
    // equal-length and independently borrowed.
    unsafe { mul_into_avx2_impl(dst, table, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_into_avx2_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast())),
        )
    };
    let mask = _mm256_set1_epi8(0x0f);
    let len = dst.len() & !31;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len() == src.len()`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
            let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(lo, hi));
        }
        offset += 32;
    }

    // SAFETY: AVX2 implies SSSE3, and the remainders keep equal lengths.
    unsafe { mul_into_ssse3_impl(&mut dst[len..], table, &src[len..]) }
}

/// `dst = coeff * dst` by nibble shuffle over 32-byte lanes.
pub(crate) fn mul_assign_avx2(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: the caller selected the AVX2 backend.
    unsafe { mul_assign_avx2_impl(dst, table) }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_avx2_impl(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast())),
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast())),
        )
    };
    let mask = _mm256_set1_epi8(0x0f);
    let len = dst.len() & !31;
    let dst_ptr = dst.as_mut_ptr();
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 32 <= len <= dst.len()`.
        unsafe {
            let x = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(x, mask));
            let hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask));
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(lo, hi));
        }
        offset += 32;
    }

    // SAFETY: AVX2 implies SSSE3.
    unsafe { mul_assign_ssse3_impl(&mut dst[len..], table) }
}

/// `dst = coeff * dst` by nibble shuffle over 16-byte lanes.
pub(crate) fn mul_assign_ssse3(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: the caller selected the SSSE3 backend.
    unsafe { mul_assign_ssse3_impl(dst, table) }
}

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn mul_assign_ssse3_impl(dst: &mut [u8], table: &ScaleTable) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm_loadu_si128(table.lo.as_ptr().cast()),
            _mm_loadu_si128(table.hi.as_ptr().cast()),
        )
    };
    let mask = _mm_set1_epi8(0x0f);
    let len = dst.len() & !15;
    let dst_ptr = dst.as_mut_ptr();
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len()`.
        unsafe {
            let x = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
            let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(lo, hi));
        }
        offset += 16;
    }

    mul_assign_nibble(&mut dst[len..], table);
}

/// `dst = coeff * src` by nibble shuffle over 16-byte lanes.
///
/// Fused out-of-place multiply: the `mul_add` body without the destination
/// read, so one pass instead of copy-then-scale.
///
/// # Panics
/// Panics if the slices differ in length.
pub(crate) fn mul_into_ssse3(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the SSSE3 backend; the slices are
    // equal-length and independently borrowed.
    unsafe { mul_into_ssse3_impl(dst, table, src) }
}

#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn mul_into_ssse3_impl(dst: &mut [u8], table: &ScaleTable, src: &[u8]) {
    // SAFETY: `lo` and `hi` are 16-byte arrays, exactly one `__m128i` each.
    let (lo_tbl, hi_tbl) = unsafe {
        (
            _mm_loadu_si128(table.lo.as_ptr().cast()),
            _mm_loadu_si128(table.hi.as_ptr().cast()),
        )
    };
    let mask = _mm_set1_epi8(0x0f);
    let len = dst.len() & !15;
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());
    let mut offset = 0;

    while offset < len {
        // SAFETY: `offset + 16 <= len <= dst.len() == src.len()`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            let lo = _mm_shuffle_epi8(lo_tbl, _mm_and_si128(x, mask));
            let hi = _mm_shuffle_epi8(hi_tbl, _mm_and_si128(_mm_srli_epi16::<4>(x), mask));
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(lo, hi));
        }
        offset += 16;
    }

    mul_into_nibble(&mut dst[len..], table, &src[len..]);
}

// ---------------------------------------------------------------------------
// One source, many rows.
// ---------------------------------------------------------------------------

/// `rows[j] ^= coeffs[j] * src` for every row, four rows at a time.
///
/// Rows with a zero coefficient contribute nothing and are dropped before
/// grouping, so a sparse coefficient vector costs no row traffic at all. The
/// surviving rows are batched in fours (then a pair, then a single) so one
/// source load feeds several destinations.
///
/// # Panics
/// Panics unless `src.len() == row_len` and `rows` holds `coeffs.len()` rows
/// of `row_len` bytes.
pub(crate) fn scatter_gfni(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    assert_eq!(row_len, src.len());
    assert!(
        coeffs
            .len()
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "scatter_gfni: rows buffer does not hold {} rows of {row_len} bytes",
        coeffs.len()
    );
    // SAFETY: the caller selected the GFNI backend, which detected both AVX2
    // and GFNI, and the geometry asserted above bounds every row.
    unsafe { scatter_gfni_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn scatter_gfni_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    let base = rows.as_mut_ptr();
    let mut ptrs = [base; 4];
    let mut group = [Elem::ZERO; 4];
    let mut filled = 0;

    for (j, &coeff) in coeffs.iter().enumerate() {
        if coeff.0 == 0 {
            // `row ^= 0 * src` is the identity: skip the row entirely rather
            // than stream it through the multiplier to add zero.
            continue;
        }
        // SAFETY: `j < coeffs.len()` and `coeffs.len() * row_len <=
        // rows.len()`, so row `j` lies wholly inside `rows`. Distinct `j` are
        // `row_len` apart, so the grouped rows never overlap.
        ptrs[filled] = unsafe { base.add(j * row_len) };
        group[filled] = coeff;
        filled += 1;
        if filled == 4 {
            // SAFETY: four in-bounds, pairwise disjoint rows of `src.len()`
            // bytes; `Elem(1)` and every other nonzero coefficient is exact
            // under `GF2P8MULB`.
            unsafe { scatter_rows4(ptrs, group, src) }
            filled = 0;
        }
    }

    // 3 left over is a pair plus a single; 2 a pair; 1 a single.
    if filled >= 2 {
        // SAFETY: as above, for the first two staged rows.
        unsafe { scatter_rows2([ptrs[0], ptrs[1]], [group[0], group[1]], src) }
    }
    if filled == 1 || filled == 3 {
        let last = filled - 1;
        // SAFETY: `ptrs[last]` is an in-bounds row of `src.len()` bytes, and
        // no other slice into `rows` is live.
        let row = unsafe { core::slice::from_raw_parts_mut(ptrs[last], src.len()) };
        // SAFETY: this function's target features are a superset.
        unsafe { mul_add_gfni_impl(row, group[last], src) }
    }
}

/// Bytes of scalar lead-in that put a row group's vector accesses on a
/// 32-byte boundary, or `0` when the row is too short to repay it.
///
/// A 32-byte `vmovdqu` at an odd multiple of 32 straddles two cache lines,
/// and the scatter body issues one load and one store per row per vector: a
/// misaligned destination therefore doubles the line traffic of eight of the
/// nine accesses a four-row group makes at each vector position. Measured on
/// 64 KiB rows the aligned form runs ~1.4x the misaligned one, so peeling at
/// most 31 bytes per row buys the aligned body for the rest of the pass.
/// Rows of a group sit `row_len` apart, so aligning one aligns them all
/// whenever `row_len` is a multiple of 32 — the usual case. The source is
/// left where it falls: it is the one access against eight, and only the
/// destination is ours to choose.
#[inline]
fn peel_to_align(ptr: *const u8, len: usize) -> usize {
    // The peel uses 128-bit GFNI where possible and scalar code below one XMM
    // lane. Keep the conservative crossover measured for the former all-scalar
    // peel: it won from about 2 KiB up and lost badly below 1 KiB. Sixteen
    // turns of the 128-byte body remains the floor.
    const FLOOR: usize = 16 * 128;
    let head = ptr.align_offset(32);
    // `align_offset` reports `usize::MAX` only for an unreachable alignment,
    // which cannot happen for a byte pointer; `head >= len` also covers the
    // degenerate case of a peel longer than the row.
    if len < FLOOR || head >= len { 0 } else { head }
}

/// `rows[i] ^= coeffs[i] * src` for four disjoint rows of `src.len()` bytes.
#[target_feature(enable = "avx2,gfni")]
unsafe fn scatter_rows4(ptrs: [*mut u8; 4], coeffs: [Elem; 4], src: &[u8]) {
    let factors = [
        _mm256_set1_epi8(coeffs[0].0.cast_signed()),
        _mm256_set1_epi8(coeffs[1].0.cast_signed()),
        _mm256_set1_epi8(coeffs[2].0.cast_signed()),
        _mm256_set1_epi8(coeffs[3].0.cast_signed()),
    ];
    let len = src.len();
    let src_ptr = src.as_ptr();

    // Bring the destinations to a 32-byte boundary before the vector body
    // starts; see `peel_to_align`.
    let head = peel_to_align(ptrs[0], len);
    // SAFETY: `head <= len`, the length shared by `src` and every row, so
    // `0..head` is a sub-range of each; the four rows are distinct and
    // in-bounds, and no slice into them is live here.
    unsafe { scatter_span(ptrs.iter().zip(&coeffs), 0, head, src) }
    let mut offset = head;

    // One 128-byte source window feeds all four rows, and each row runs four
    // independent multiply chains over it.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len == src.len()`, which is also the length
        // of every row; the rows are pairwise disjoint.
        unsafe {
            let sp = src_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d0 = _mm256_loadu_si256(rp.cast());
                let d1 = _mm256_loadu_si256(rp.add(32).cast());
                let d2 = _mm256_loadu_si256(rp.add(64).cast());
                let d3 = _mm256_loadu_si256(rp.add(96).cast());
                let r0 = _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x0, factor));
                let r1 = _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x1, factor));
                let r2 = _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x2, factor));
                let r3 = _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x3, factor));
                _mm256_storeu_si256(rp.cast(), r0);
                _mm256_storeu_si256(rp.add(32).cast(), r1);
                _mm256_storeu_si256(rp.add(64).cast(), r2);
                _mm256_storeu_si256(rp.add(96).cast(), r3);
            }
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, the common length of `src` and every
        // row; the rows are pairwise disjoint.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d = _mm256_loadu_si256(rp.cast());
                _mm256_storeu_si256(
                    rp.cast(),
                    _mm256_xor_si256(d, _mm256_gf2p8mul_epi8(x, factor)),
                );
            }
        }
        offset += 32;
    }

    // SAFETY: the four rows are distinct, in-bounds, and `src.len()` bytes
    // long; no slice into them is live here.
    unsafe { scatter_span(ptrs.iter().zip(&coeffs), offset, len, src) }
}

/// `rows[i] ^= coeffs[i] * src` for two disjoint rows of `src.len()` bytes.
#[target_feature(enable = "avx2,gfni")]
unsafe fn scatter_rows2(ptrs: [*mut u8; 2], coeffs: [Elem; 2], src: &[u8]) {
    let factors = [
        _mm256_set1_epi8(coeffs[0].0.cast_signed()),
        _mm256_set1_epi8(coeffs[1].0.cast_signed()),
    ];
    let len = src.len();
    let src_ptr = src.as_ptr();

    // As in `scatter_rows4`, with four of every five accesses on the
    // destination side.
    let head = peel_to_align(ptrs[0], len);
    // SAFETY: `head <= len`, the length shared by `src` and both rows, so
    // `0..head` is a sub-range of each; the rows are distinct, in-bounds,
    // and no slice into them is live here.
    unsafe { scatter_span(ptrs.iter().zip(&coeffs), 0, head, src) }
    let mut offset = head;

    // With only two rows the 128-byte window leaves room for eight chains.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len == src.len()`, which is also the length
        // of both rows; the rows are disjoint.
        unsafe {
            let sp = src_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d0 = _mm256_loadu_si256(rp.cast());
                let d1 = _mm256_loadu_si256(rp.add(32).cast());
                let d2 = _mm256_loadu_si256(rp.add(64).cast());
                let d3 = _mm256_loadu_si256(rp.add(96).cast());
                let r0 = _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x0, factor));
                let r1 = _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x1, factor));
                let r2 = _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x2, factor));
                let r3 = _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x3, factor));
                _mm256_storeu_si256(rp.cast(), r0);
                _mm256_storeu_si256(rp.add(32).cast(), r1);
                _mm256_storeu_si256(rp.add(64).cast(), r2);
                _mm256_storeu_si256(rp.add(96).cast(), r3);
            }
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, the common length of `src` and both
        // rows; the rows are disjoint.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            for (&row, &factor) in ptrs.iter().zip(&factors) {
                let rp = row.add(offset);
                let d = _mm256_loadu_si256(rp.cast());
                _mm256_storeu_si256(
                    rp.cast(),
                    _mm256_xor_si256(d, _mm256_gf2p8mul_epi8(x, factor)),
                );
            }
        }
        offset += 32;
    }

    // SAFETY: the two rows are distinct, in-bounds, and `src.len()` bytes
    // long; no slice into them is live here.
    unsafe { scatter_span(ptrs.iter().zip(&coeffs), offset, len, src) }
}

/// Narrow SIMD span shared by the GFNI scatter row groups.
///
/// Processes complete 16-byte lanes with GFNI and leaves fewer than 16 bytes
/// to the scalar nibble kernel.
///
/// # Safety
/// Every pointer must address a distinct, in-bounds row of at least
/// `src.len()` bytes, `start..end` must lie within `0..=src.len()`, no slice
/// into those rows may be live, and the caller must have selected GFNI.
#[target_feature(enable = "sse2,gfni")]
unsafe fn scatter_span<'a>(
    rows: impl Iterator<Item = (&'a *mut u8, &'a Elem)>,
    start: usize,
    end: usize,
    src: &[u8],
) {
    if start >= end {
        return;
    }
    let vector_end = start + ((end - start) & !15);
    for (&row, &coeff) in rows {
        let factor = _mm_set1_epi8(coeff.0.cast_signed());
        let mut offset = start;
        while offset < vector_end {
            // SAFETY: `offset + 16 <= vector_end <= end <= src.len()` bounds
            // both source and row accesses.
            unsafe {
                let x = _mm_loadu_si128(src.as_ptr().add(offset).cast());
                let ptr = row.add(offset);
                let d = _mm_loadu_si128(ptr.cast());
                _mm_storeu_si128(ptr.cast(), _mm_xor_si128(d, _mm_gf2p8mul_epi8(x, factor)));
            }
            offset += 16;
        }
        if vector_end < end {
            // SAFETY: `vector_end..end` is the row's final sub-lane span; rows
            // are disjoint and only one mutable slice is live at a time.
            let dst =
                unsafe { core::slice::from_raw_parts_mut(row.add(vector_end), end - vector_end) };
            mul_add_nibble(dst, scale_table(coeff), &src[vector_end..end]);
        }
    }
}

// ---------------------------------------------------------------------------
// Many sources, many rows.
// ---------------------------------------------------------------------------

/// For each `(coeffs, src)` term and each row `j < nrows`,
/// `rows[j] ^= coeffs[j] * src`.
///
/// Register-blocked over the destination: a group of rows is loaded into
/// accumulators once, every term is folded in, and the tile is stored once.
/// Destination traffic is therefore independent of `terms.len()`, which is
/// the whole point of the fused shape. Rows are grouped in fours (64-byte
/// tiles, eight accumulators), then a pair and a single (128-byte tiles).
///
/// Unlike [`scatter_gfni`] the term loop does not skip zero coefficients. The
/// predicate depends only on the term and the row group, never on the tile,
/// so testing it where it would have to live — innermost, once per tile — is
/// pure overhead on a dense matrix, and it costs more than the branch: the
/// coefficients have to reach a GPR to be tested, which stops each factor
/// broadcast folding into a memory-operand `vpbroadcastb`. Measured on eight
/// terms over 64 KiB rows the check cost ~9%. Sparsity belongs in the scatter
/// shape, which drops zero rows before grouping and outside any loop.
///
/// # Panics
/// Panics unless `rows` holds `nrows` rows of `row_len` bytes and every term
/// supplies `nrows` coefficients for a source of `row_len` bytes.
pub(crate) fn matrix_gfni(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    assert!(
        nrows
            .checked_mul(row_len)
            .is_some_and(|needed| needed <= rows.len()),
        "matrix_gfni: rows buffer does not hold {nrows} rows of {row_len} bytes"
    );
    for &(coeffs, src) in terms {
        assert!(
            coeffs.len() >= nrows,
            "matrix_gfni: term supplies {} coefficients for {nrows} rows",
            coeffs.len()
        );
        assert_eq!(src.len(), row_len);
    }
    if terms.is_empty() {
        return;
    }
    // SAFETY: the caller selected the GFNI backend, which detected both AVX2
    // and GFNI; the geometry asserted above bounds every row and every term.
    unsafe { matrix_gfni_impl(rows, row_len, nrows, terms) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_gfni_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    let base = rows.as_mut_ptr();
    // Rows are grouped four at a time, then a pair, then whatever single row
    // is left: `nrows - g` is at most three after the first loop and at most
    // one after the pair.
    let mut g = 0;
    while g + 4 <= nrows {
        // SAFETY: row `j` starts at `base + j * row_len` and spans `row_len`
        // bytes, which the wrapper checked lies inside `rows`.
        let ptrs = unsafe {
            [
                base.add(g * row_len),
                base.add((g + 1) * row_len),
                base.add((g + 2) * row_len),
                base.add((g + 3) * row_len),
            ]
        };
        // SAFETY: four in-bounds rows of `row_len` bytes, pairwise disjoint
        // because distinct rows are `row_len` apart; the wrapper checked every
        // term for `nrows` coefficients over a `row_len`-byte source.
        unsafe { matrix_rows4(ptrs, row_len, g, terms) }
        g += 4;
    }
    if g + 2 <= nrows {
        // SAFETY: as above, for the two rows at `g` and `g + 1`.
        let ptrs = unsafe { [base.add(g * row_len), base.add((g + 1) * row_len)] };
        // SAFETY: two in-bounds, disjoint rows of `row_len` bytes, and every
        // term covers row `g + 1`.
        unsafe { matrix_rows2(ptrs, row_len, g, terms) }
        g += 2;
    }
    if g < nrows {
        // SAFETY: as above, for the last row.
        let ptr = unsafe { base.add(g * row_len) };
        // SAFETY: one in-bounds row of `row_len` bytes, and every term covers
        // row `g`.
        unsafe { matrix_rows1(ptr, row_len, g, terms) }
    }
}

/// Fold every term into four rows, 64 bytes of each row at a time.
///
/// # Safety
/// The four pointers must address distinct, in-bounds rows of `row_len`
/// bytes, every term's source must be `row_len` bytes, and every term must
/// supply coefficients through index `g + 3`.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_rows4(ptrs: [*mut u8; 4], row_len: usize, g: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 64 <= row_len {
        // SAFETY: `tile + 64 <= row_len`, the length of every row; the rows
        // are disjoint.
        let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = unsafe {
            (
                _mm256_loadu_si256(ptrs[0].add(tile).cast()),
                _mm256_loadu_si256(ptrs[0].add(tile + 32).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile + 32).cast()),
                _mm256_loadu_si256(ptrs[2].add(tile).cast()),
                _mm256_loadu_si256(ptrs[2].add(tile + 32).cast()),
                _mm256_loadu_si256(ptrs[3].add(tile).cast()),
                _mm256_loadu_si256(ptrs[3].add(tile + 32).cast()),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g + 3`, and
            // `src.len() == row_len`, so `tile + 64` bounds both loads.
            unsafe {
                let cp = coeffs.as_ptr().add(g);
                let sp = src.as_ptr().add(tile);
                let x0 = _mm256_loadu_si256(sp.cast());
                let x1 = _mm256_loadu_si256(sp.add(32).cast());
                let f0 = _mm256_set1_epi8((*cp).0.cast_signed());
                let f1 = _mm256_set1_epi8((*cp.add(1)).0.cast_signed());
                let f2 = _mm256_set1_epi8((*cp.add(2)).0.cast_signed());
                let f3 = _mm256_set1_epi8((*cp.add(3)).0.cast_signed());
                a00 = _mm256_xor_si256(a00, _mm256_gf2p8mul_epi8(x0, f0));
                a01 = _mm256_xor_si256(a01, _mm256_gf2p8mul_epi8(x1, f0));
                a10 = _mm256_xor_si256(a10, _mm256_gf2p8mul_epi8(x0, f1));
                a11 = _mm256_xor_si256(a11, _mm256_gf2p8mul_epi8(x1, f1));
                a20 = _mm256_xor_si256(a20, _mm256_gf2p8mul_epi8(x0, f2));
                a21 = _mm256_xor_si256(a21, _mm256_gf2p8mul_epi8(x1, f2));
                a30 = _mm256_xor_si256(a30, _mm256_gf2p8mul_epi8(x0, f3));
                a31 = _mm256_xor_si256(a31, _mm256_gf2p8mul_epi8(x1, f3));
            }
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            _mm256_storeu_si256(ptrs[0].add(tile).cast(), a00);
            _mm256_storeu_si256(ptrs[0].add(tile + 32).cast(), a01);
            _mm256_storeu_si256(ptrs[1].add(tile).cast(), a10);
            _mm256_storeu_si256(ptrs[1].add(tile + 32).cast(), a11);
            _mm256_storeu_si256(ptrs[2].add(tile).cast(), a20);
            _mm256_storeu_si256(ptrs[2].add(tile + 32).cast(), a21);
            _mm256_storeu_si256(ptrs[3].add(tile).cast(), a30);
            _mm256_storeu_si256(ptrs[3].add(tile + 32).cast(), a31);
        }
        tile += 64;
    }
    // At most one 32-byte tile survives the 64-byte loop.
    if tile + 32 <= row_len {
        // SAFETY: `tile + 32 <= row_len`; the rows are disjoint.
        let (mut a0, mut a1, mut a2, mut a3) = unsafe {
            (
                _mm256_loadu_si256(ptrs[0].add(tile).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile).cast()),
                _mm256_loadu_si256(ptrs[2].add(tile).cast()),
                _mm256_loadu_si256(ptrs[3].add(tile).cast()),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g + 3`, and
            // `src.len() == row_len` bounds the load.
            unsafe {
                let cp = coeffs.as_ptr();
                let x = _mm256_loadu_si256(src.as_ptr().add(tile).cast());
                let f0 = _mm256_set1_epi8((*cp.add(g)).0.cast_signed());
                let f1 = _mm256_set1_epi8((*cp.add(g + 1)).0.cast_signed());
                let f2 = _mm256_set1_epi8((*cp.add(g + 2)).0.cast_signed());
                let f3 = _mm256_set1_epi8((*cp.add(g + 3)).0.cast_signed());
                a0 = _mm256_xor_si256(a0, _mm256_gf2p8mul_epi8(x, f0));
                a1 = _mm256_xor_si256(a1, _mm256_gf2p8mul_epi8(x, f1));
                a2 = _mm256_xor_si256(a2, _mm256_gf2p8mul_epi8(x, f2));
                a3 = _mm256_xor_si256(a3, _mm256_gf2p8mul_epi8(x, f3));
            }
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            _mm256_storeu_si256(ptrs[0].add(tile).cast(), a0);
            _mm256_storeu_si256(ptrs[1].add(tile).cast(), a1);
            _mm256_storeu_si256(ptrs[2].add(tile).cast(), a2);
            _mm256_storeu_si256(ptrs[3].add(tile).cast(), a3);
        }
        tile += 32;
    }
    // SAFETY: the pointers address distinct in-bounds rows of `row_len` bytes.
    unsafe { matrix_tail(&ptrs, row_len, g, tile, terms) }
}

/// Fold every term into two rows, 128 bytes of each row at a time.
///
/// # Safety
/// As [`matrix_rows4`], for two rows and coefficients through `g + 1`.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_rows2(ptrs: [*mut u8; 2], row_len: usize, g: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 128 <= row_len {
        // SAFETY: `tile + 128 <= row_len`, the length of both rows; the rows
        // are disjoint.
        let (mut a00, mut a01, mut a02, mut a03, mut a10, mut a11, mut a12, mut a13) = unsafe {
            (
                _mm256_loadu_si256(ptrs[0].add(tile).cast()),
                _mm256_loadu_si256(ptrs[0].add(tile + 32).cast()),
                _mm256_loadu_si256(ptrs[0].add(tile + 64).cast()),
                _mm256_loadu_si256(ptrs[0].add(tile + 96).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile + 32).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile + 64).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile + 96).cast()),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g + 1`, and
            // `src.len() == row_len`, so `tile + 128` bounds the loads.
            unsafe {
                let cp = coeffs.as_ptr().add(g);
                let sp = src.as_ptr().add(tile);
                let x0 = _mm256_loadu_si256(sp.cast());
                let x1 = _mm256_loadu_si256(sp.add(32).cast());
                let x2 = _mm256_loadu_si256(sp.add(64).cast());
                let x3 = _mm256_loadu_si256(sp.add(96).cast());
                let f0 = _mm256_set1_epi8((*cp).0.cast_signed());
                let f1 = _mm256_set1_epi8((*cp.add(1)).0.cast_signed());
                a00 = _mm256_xor_si256(a00, _mm256_gf2p8mul_epi8(x0, f0));
                a01 = _mm256_xor_si256(a01, _mm256_gf2p8mul_epi8(x1, f0));
                a02 = _mm256_xor_si256(a02, _mm256_gf2p8mul_epi8(x2, f0));
                a03 = _mm256_xor_si256(a03, _mm256_gf2p8mul_epi8(x3, f0));
                a10 = _mm256_xor_si256(a10, _mm256_gf2p8mul_epi8(x0, f1));
                a11 = _mm256_xor_si256(a11, _mm256_gf2p8mul_epi8(x1, f1));
                a12 = _mm256_xor_si256(a12, _mm256_gf2p8mul_epi8(x2, f1));
                a13 = _mm256_xor_si256(a13, _mm256_gf2p8mul_epi8(x3, f1));
            }
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            _mm256_storeu_si256(ptrs[0].add(tile).cast(), a00);
            _mm256_storeu_si256(ptrs[0].add(tile + 32).cast(), a01);
            _mm256_storeu_si256(ptrs[0].add(tile + 64).cast(), a02);
            _mm256_storeu_si256(ptrs[0].add(tile + 96).cast(), a03);
            _mm256_storeu_si256(ptrs[1].add(tile).cast(), a10);
            _mm256_storeu_si256(ptrs[1].add(tile + 32).cast(), a11);
            _mm256_storeu_si256(ptrs[1].add(tile + 64).cast(), a12);
            _mm256_storeu_si256(ptrs[1].add(tile + 96).cast(), a13);
        }
        tile += 128;
    }
    while tile + 32 <= row_len {
        // SAFETY: `tile + 32 <= row_len`; the rows are disjoint.
        let (mut a0, mut a1) = unsafe {
            (
                _mm256_loadu_si256(ptrs[0].add(tile).cast()),
                _mm256_loadu_si256(ptrs[1].add(tile).cast()),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g + 1`, and
            // `src.len() == row_len` bounds the load.
            unsafe {
                let cp = coeffs.as_ptr();
                let x = _mm256_loadu_si256(src.as_ptr().add(tile).cast());
                let f0 = _mm256_set1_epi8((*cp.add(g)).0.cast_signed());
                let f1 = _mm256_set1_epi8((*cp.add(g + 1)).0.cast_signed());
                a0 = _mm256_xor_si256(a0, _mm256_gf2p8mul_epi8(x, f0));
                a1 = _mm256_xor_si256(a1, _mm256_gf2p8mul_epi8(x, f1));
            }
        }
        // SAFETY: same bounds and disjointness as the loads above.
        unsafe {
            _mm256_storeu_si256(ptrs[0].add(tile).cast(), a0);
            _mm256_storeu_si256(ptrs[1].add(tile).cast(), a1);
        }
        tile += 32;
    }
    // SAFETY: the pointers address distinct in-bounds rows of `row_len` bytes.
    unsafe { matrix_tail(&ptrs, row_len, g, tile, terms) }
}

/// Fold every term into one row, 128 bytes at a time.
///
/// # Safety
/// As [`matrix_rows4`], for one row and coefficient `g`.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_rows1(ptr: *mut u8, row_len: usize, g: usize, terms: &[(&[Elem], &[u8])]) {
    let mut tile = 0;
    while tile + 128 <= row_len {
        // SAFETY: `tile + 128 <= row_len`, the length of the row.
        let (mut a0, mut a1, mut a2, mut a3) = unsafe {
            (
                _mm256_loadu_si256(ptr.add(tile).cast()),
                _mm256_loadu_si256(ptr.add(tile + 32).cast()),
                _mm256_loadu_si256(ptr.add(tile + 64).cast()),
                _mm256_loadu_si256(ptr.add(tile + 96).cast()),
            )
        };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g`, and
            // `src.len() == row_len`, so `tile + 128` bounds the loads.
            unsafe {
                let sp = src.as_ptr().add(tile);
                let f = _mm256_set1_epi8((*coeffs.as_ptr().add(g)).0.cast_signed());
                let x0 = _mm256_loadu_si256(sp.cast());
                let x1 = _mm256_loadu_si256(sp.add(32).cast());
                let x2 = _mm256_loadu_si256(sp.add(64).cast());
                let x3 = _mm256_loadu_si256(sp.add(96).cast());
                a0 = _mm256_xor_si256(a0, _mm256_gf2p8mul_epi8(x0, f));
                a1 = _mm256_xor_si256(a1, _mm256_gf2p8mul_epi8(x1, f));
                a2 = _mm256_xor_si256(a2, _mm256_gf2p8mul_epi8(x2, f));
                a3 = _mm256_xor_si256(a3, _mm256_gf2p8mul_epi8(x3, f));
            }
        }
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(ptr.add(tile).cast(), a0);
            _mm256_storeu_si256(ptr.add(tile + 32).cast(), a1);
            _mm256_storeu_si256(ptr.add(tile + 64).cast(), a2);
            _mm256_storeu_si256(ptr.add(tile + 96).cast(), a3);
        }
        tile += 128;
    }
    while tile + 32 <= row_len {
        // SAFETY: `tile + 32 <= row_len`.
        let mut a0 = unsafe { _mm256_loadu_si256(ptr.add(tile).cast()) };
        for &(coeffs, src) in terms {
            // SAFETY: the caller guarantees `coeffs.len() > g`, and
            // `src.len() == row_len` bounds the load.
            unsafe {
                let f = _mm256_set1_epi8((*coeffs.as_ptr().add(g)).0.cast_signed());
                let x = _mm256_loadu_si256(src.as_ptr().add(tile).cast());
                a0 = _mm256_xor_si256(a0, _mm256_gf2p8mul_epi8(x, f));
            }
        }
        // SAFETY: same bounds as the load above.
        unsafe { _mm256_storeu_si256(ptr.add(tile).cast(), a0) }
        tile += 32;
    }

    // SAFETY: the pointer addresses an in-bounds row of `row_len` bytes.
    unsafe { matrix_tail(&[ptr], row_len, g, tile, terms) }
}

/// Hierarchical remainder shared by the GFNI matrix row groups.
///
/// Complete 32- and 16-byte lanes stay on GFNI through
/// [`mul_add_gfni_impl`]; only the final sub-XMM bytes use scalar arithmetic.
///
/// # Safety
/// Every pointer must address a distinct, in-bounds row of `row_len` bytes,
/// no slice into those rows may be live, every term must supply coefficients
/// through `g + ptrs.len() - 1` over a source of `row_len` bytes, and the
/// caller must have selected AVX2 + GFNI.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_tail(
    ptrs: &[*mut u8],
    row_len: usize,
    g: usize,
    tile: usize,
    terms: &[(&[Elem], &[u8])],
) {
    if tile == row_len {
        return;
    }
    let remaining = row_len - tile;
    for (slot, &row) in ptrs.iter().enumerate() {
        // SAFETY: `tile + remaining == row_len`, so this is the row's own
        // tail; the rows are disjoint and only one tail slice is live at a
        // time.
        let tail = unsafe { core::slice::from_raw_parts_mut(row.add(tile), remaining) };
        for &(coeffs, src) in terms {
            let coeff = coeffs[g + slot];
            if coeff.0 != 0 {
                // SAFETY: the tail and source remainder have equal lengths,
                // and this function's target features match the callee.
                unsafe { mul_add_gfni_impl(tail, coeff, &src[tile..]) }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1: blocked gather and nibble multi-row kernels.
// ---------------------------------------------------------------------------

/// `dst[i] = a[i] * b[i]` using vector-by-vector `GF2P8MULB`.
pub(crate) fn elementwise_gfni(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the selected backend guarantees AVX2 and GFNI.
    unsafe { elementwise_gfni_impl(dst, a, b) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn elementwise_gfni_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let total_len = dst.len().min(a.len()).min(b.len());
    let len = total_len & !31;
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 32 <= len`, which bounds all three slices.
        unsafe {
            let x = _mm256_loadu_si256(a_ptr.add(offset).cast());
            let y = _mm256_loadu_si256(b_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_gf2p8mul_epi8(x, y));
        }
        offset += 32;
    }
    let mut offset = len;
    if offset + 16 <= total_len {
        // SAFETY: `offset + 16 <= total_len` bounds all three slices.
        unsafe {
            let x = _mm_loadu_si128(a_ptr.add(offset).cast());
            let y = _mm_loadu_si128(b_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_gf2p8mul_epi8(x, y));
        }
        offset += 16;
    }
    for ((d, &x), &y) in dst[offset..].iter_mut().zip(&a[offset..]).zip(&b[offset..]) {
        *d = Elem(x).mul(Elem(y)).0;
    }
}

/// Many sources into one destination, register-blocked over 128-byte tiles.
pub(crate) fn gather_gfni(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees AVX2 and GFNI; callers checked
    // every source length against `dst`.
    unsafe { gather_gfni_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn gather_gfni_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let len = dst.len() & !127;
    let dst_ptr = dst.as_mut_ptr();
    let mut offset = 0;
    while offset < len {
        // SAFETY: `offset + 128 <= len <= dst.len()`.
        let mut acc = unsafe {
            [
                _mm256_loadu_si256(dst_ptr.add(offset).cast()),
                _mm256_loadu_si256(dst_ptr.add(offset + 32).cast()),
                _mm256_loadu_si256(dst_ptr.add(offset + 64).cast()),
                _mm256_loadu_si256(dst_ptr.add(offset + 96).cast()),
            ]
        };
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let factor = _mm256_set1_epi8(coeff.0.cast_signed());
            let src_ptr = src.as_ptr();
            // SAFETY: every source is at least `dst.len()` bytes.
            unsafe {
                for (lane, slot) in acc.iter_mut().enumerate() {
                    let x = _mm256_loadu_si256(src_ptr.add(offset + lane * 32).cast());
                    *slot = _mm256_xor_si256(*slot, _mm256_gf2p8mul_epi8(x, factor));
                }
            }
        }
        // SAFETY: the same 128-byte destination window loaded above.
        unsafe {
            for (lane, &value) in acc.iter().enumerate() {
                _mm256_storeu_si256(dst_ptr.add(offset + lane * 32).cast(), value);
            }
        }
        offset += 128;
    }
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        // SAFETY: the destination and source remainders have equal lengths,
        // and this function's target features match the callee.
        unsafe { mul_add_gfni_impl(&mut dst[len..], coeff, &src[len..]) }
    }
}

/// Many sources into one destination using AVX2 nibble shuffles.
pub(crate) fn gather_avx2(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees AVX2.
    unsafe { gather_avx2_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "avx2")]
unsafe fn gather_avx2_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let len = dst.len() & !63;
    let dst_ptr = dst.as_mut_ptr();
    let mask = _mm256_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm256_setzero_si256(); 4];
        let mut hi = [_mm256_setzero_si256(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // SAFETY: each table half is exactly 16 bytes.
            unsafe {
                lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast()));
                hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast()));
            }
        }
        let mut offset = 0;
        while offset < len {
            // SAFETY: `offset + 64 <= len <= dst.len()`.
            let (mut acc0, mut acc1) = unsafe {
                (
                    _mm256_loadu_si256(dst_ptr.add(offset).cast()),
                    _mm256_loadu_si256(dst_ptr.add(offset + 32).cast()),
                )
            };
            for slot in 0..count {
                // SAFETY: every source is at least `dst.len()` bytes.
                unsafe {
                    let src = srcs[group + slot].as_ptr();
                    let x0 = _mm256_loadu_si256(src.add(offset).cast());
                    let x1 = _mm256_loadu_si256(src.add(offset + 32).cast());
                    let p0 = _mm256_xor_si256(
                        _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x0, mask)),
                        _mm256_shuffle_epi8(
                            hi[slot],
                            _mm256_and_si256(_mm256_srli_epi16::<4>(x0), mask),
                        ),
                    );
                    let p1 = _mm256_xor_si256(
                        _mm256_shuffle_epi8(lo[slot], _mm256_and_si256(x1, mask)),
                        _mm256_shuffle_epi8(
                            hi[slot],
                            _mm256_and_si256(_mm256_srli_epi16::<4>(x1), mask),
                        ),
                    );
                    acc0 = _mm256_xor_si256(acc0, p0);
                    acc1 = _mm256_xor_si256(acc1, p1);
                }
            }
            // SAFETY: the destination window loaded above remains in bounds.
            unsafe {
                _mm256_storeu_si256(dst_ptr.add(offset).cast(), acc0);
                _mm256_storeu_si256(dst_ptr.add(offset + 32).cast(), acc1);
            }
            offset += 64;
        }
        for slot in 0..count {
            // SAFETY: AVX2 implies SSSE3; the destination and source
            // remainders have equal lengths.
            unsafe {
                mul_add_avx2_impl(
                    &mut dst[len..],
                    scale_table(coeffs[group + slot]),
                    &srcs[group + slot][len..],
                );
            }
        }
    }
}

/// Many sources into one destination using SSSE3 nibble shuffles.
pub(crate) fn gather_ssse3(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees SSSE3.
    unsafe { gather_ssse3_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "ssse3")]
unsafe fn gather_ssse3_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let len = dst.len() & !63;
    let dst_ptr = dst.as_mut_ptr();
    let mask = _mm_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // SAFETY: each table half is exactly 16 bytes.
            unsafe {
                lo[slot] = _mm_loadu_si128(table.lo.as_ptr().cast());
                hi[slot] = _mm_loadu_si128(table.hi.as_ptr().cast());
            }
        }
        let mut offset = 0;
        while offset < len {
            // SAFETY: `offset + 64 <= len <= dst.len()`.
            let mut acc = unsafe {
                [
                    _mm_loadu_si128(dst_ptr.add(offset).cast()),
                    _mm_loadu_si128(dst_ptr.add(offset + 16).cast()),
                    _mm_loadu_si128(dst_ptr.add(offset + 32).cast()),
                    _mm_loadu_si128(dst_ptr.add(offset + 48).cast()),
                ]
            };
            for slot in 0..count {
                // SAFETY: every source is at least `dst.len()` bytes.
                unsafe {
                    let src = srcs[group + slot].as_ptr();
                    for (lane, value) in acc.iter_mut().enumerate() {
                        let x = _mm_loadu_si128(src.add(offset + lane * 16).cast());
                        let product = _mm_xor_si128(
                            _mm_shuffle_epi8(lo[slot], _mm_and_si128(x, mask)),
                            _mm_shuffle_epi8(hi[slot], _mm_and_si128(_mm_srli_epi16::<4>(x), mask)),
                        );
                        *value = _mm_xor_si128(*value, product);
                    }
                }
            }
            // SAFETY: the destination window loaded above remains in bounds.
            unsafe {
                for (lane, &value) in acc.iter().enumerate() {
                    _mm_storeu_si128(dst_ptr.add(offset + lane * 16).cast(), value);
                }
            }
            offset += 64;
        }
        for slot in 0..count {
            // SAFETY: this function runs under SSSE3, and the destination and
            // source remainders have equal lengths.
            unsafe {
                mul_add_ssse3_impl(
                    &mut dst[len..],
                    scale_table(coeffs[group + slot]),
                    &srcs[group + slot][len..],
                );
            }
        }
    }
}

/// One source into many rows using one AVX2 source load per tile.
pub(crate) fn scatter_avx2(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    debug_assert_eq!(row_len, src.len());
    // SAFETY: the selected backend guarantees AVX2 and row geometry was
    // checked by the public wrapper.
    unsafe { scatter_avx2_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn scatter_avx2_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !31;
    let base = rows.as_mut_ptr();
    let mask = _mm256_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm256_setzero_si256(); 4];
        let mut hi = [_mm256_setzero_si256(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // SAFETY: each table half is exactly 16 bytes.
            unsafe {
                lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast()));
                hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast()));
            }
        }
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: `offset + 32 <= row_len == src.len()`.
            let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
            let indices_lo = _mm256_and_si256(x, mask);
            let indices_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask);
            for slot in 0..count {
                let product = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo[slot], indices_lo),
                    _mm256_shuffle_epi8(hi[slot], indices_hi),
                );
                // SAFETY: the selected row window is in bounds and disjoint.
                unsafe {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm256_storeu_si256(
                        ptr.cast(),
                        _mm256_xor_si256(_mm256_loadu_si256(ptr.cast()), product),
                    );
                }
            }
            offset += 32;
        }
        for slot in 0..count {
            // SAFETY: this is one row's disjoint tail, and AVX2 implies
            // SSSE3. The remainder keeps equal source and destination lengths.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            unsafe {
                mul_add_ssse3_impl(tail, scale_table(coeffs[group + slot]), &src[vector_len..]);
            }
        }
    }
}

/// One source into many rows using one SSSE3 source load per tile.
pub(crate) fn scatter_ssse3(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    debug_assert_eq!(row_len, src.len());
    // SAFETY: the selected backend guarantees SSSE3.
    unsafe { scatter_ssse3_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "ssse3")]
unsafe fn scatter_ssse3_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !15;
    let base = rows.as_mut_ptr();
    let mask = _mm_set1_epi8(0x0f);
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let mut lo = [_mm_setzero_si128(); 4];
        let mut hi = [_mm_setzero_si128(); 4];
        for slot in 0..count {
            let table = scale_table(coeffs[group + slot]);
            // SAFETY: each table half is exactly 16 bytes.
            unsafe {
                lo[slot] = _mm_loadu_si128(table.lo.as_ptr().cast());
                hi[slot] = _mm_loadu_si128(table.hi.as_ptr().cast());
            }
        }
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: `offset + 16 <= row_len == src.len()`.
            let x = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast()) };
            let indices_lo = _mm_and_si128(x, mask);
            let indices_hi = _mm_and_si128(_mm_srli_epi16::<4>(x), mask);
            for slot in 0..count {
                let product = _mm_xor_si128(
                    _mm_shuffle_epi8(lo[slot], indices_lo),
                    _mm_shuffle_epi8(hi[slot], indices_hi),
                );
                // SAFETY: the selected row window is in bounds and disjoint.
                unsafe {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm_storeu_si128(
                        ptr.cast(),
                        _mm_xor_si128(_mm_loadu_si128(ptr.cast()), product),
                    );
                }
            }
            offset += 16;
        }
        for slot in 0..count {
            // SAFETY: this is one row's disjoint scalar tail.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            mul_add_nibble(tail, scale_table(coeffs[group + slot]), &src[vector_len..]);
        }
    }
}

/// Many sources into many rows using AVX2 nibble shuffles.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn matrix_avx2(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    // SAFETY: the selected backend guarantees AVX2 and geometry was checked.
    unsafe { matrix_avx2_impl(rows, row_len, nrows, terms) }
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "avx2")]
unsafe fn matrix_avx2_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !31;
    let base = rows.as_mut_ptr();
    let mask = _mm256_set1_epi8(0x0f);
    let mut group = 0;
    while group < nrows {
        let count = (nrows - group).min(4);
        let mut offset = 0;
        while offset < vector_len {
            let mut acc = [_mm256_setzero_si256(); 4];
            // SAFETY: every selected row contains this 32-byte window.
            unsafe {
                for (slot, value) in acc.iter_mut().take(count).enumerate() {
                    *value = _mm256_loadu_si256(base.add((group + slot) * row_len + offset).cast());
                }
            }
            for &(coeffs, src) in terms {
                // SAFETY: every term source is exactly `row_len` bytes.
                let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
                for slot in 0..count {
                    let table = scale_table(coeffs[group + slot]);
                    // SAFETY: each table half is exactly 16 bytes.
                    unsafe {
                        let lo =
                            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.lo.as_ptr().cast()));
                        let hi =
                            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.hi.as_ptr().cast()));
                        let product = _mm256_xor_si256(
                            _mm256_shuffle_epi8(lo, _mm256_and_si256(x, mask)),
                            _mm256_shuffle_epi8(
                                hi,
                                _mm256_and_si256(_mm256_srli_epi16::<4>(x), mask),
                            ),
                        );
                        acc[slot] = _mm256_xor_si256(acc[slot], product);
                    }
                }
            }
            // SAFETY: the same disjoint row windows loaded above.
            unsafe {
                for (slot, &value) in acc.iter().take(count).enumerate() {
                    _mm256_storeu_si256(base.add((group + slot) * row_len + offset).cast(), value);
                }
            }
            offset += 32;
        }
        for slot in 0..count {
            // SAFETY: this is the selected row's disjoint tail, AVX2 implies
            // SSSE3, and every source remainder has the same length.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            for &(coeffs, src) in terms {
                unsafe {
                    mul_add_ssse3_impl(tail, scale_table(coeffs[group + slot]), &src[vector_len..]);
                }
            }
        }
        group += count;
    }
}

/// Many sources into many rows using SSSE3 nibble shuffles.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn matrix_ssse3(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    // SAFETY: the selected backend guarantees SSSE3 and geometry was checked.
    unsafe { matrix_ssse3_impl(rows, row_len, nrows, terms) }
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "ssse3")]
unsafe fn matrix_ssse3_impl(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !15;
    let base = rows.as_mut_ptr();
    let mask = _mm_set1_epi8(0x0f);
    let mut group = 0;
    while group < nrows {
        let count = (nrows - group).min(4);
        let mut offset = 0;
        while offset < vector_len {
            let mut acc = [_mm_setzero_si128(); 4];
            // SAFETY: every selected row contains this 16-byte window.
            unsafe {
                for (slot, value) in acc.iter_mut().take(count).enumerate() {
                    *value = _mm_loadu_si128(base.add((group + slot) * row_len + offset).cast());
                }
            }
            for &(coeffs, src) in terms {
                // SAFETY: every term source is exactly `row_len` bytes.
                let x = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast()) };
                for slot in 0..count {
                    let table = scale_table(coeffs[group + slot]);
                    // SAFETY: each table half is exactly 16 bytes.
                    unsafe {
                        let lo = _mm_loadu_si128(table.lo.as_ptr().cast());
                        let hi = _mm_loadu_si128(table.hi.as_ptr().cast());
                        let product = _mm_xor_si128(
                            _mm_shuffle_epi8(lo, _mm_and_si128(x, mask)),
                            _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16::<4>(x), mask)),
                        );
                        acc[slot] = _mm_xor_si128(acc[slot], product);
                    }
                }
            }
            // SAFETY: the same disjoint row windows loaded above.
            unsafe {
                for (slot, &value) in acc.iter().take(count).enumerate() {
                    _mm_storeu_si128(base.add((group + slot) * row_len + offset).cast(), value);
                }
            }
            offset += 16;
        }
        for slot in 0..count {
            // SAFETY: this is the selected row's disjoint scalar tail.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            for &(coeffs, src) in terms {
                mul_add_nibble(tail, scale_table(coeffs[group + slot]), &src[vector_len..]);
            }
        }
        group += count;
    }
}
