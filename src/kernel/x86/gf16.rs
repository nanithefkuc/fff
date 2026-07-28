//! GF(2^16) tower kernels for x86 / `x86_64`.
//!
//! Every kernel here is the same identity, spelled three ways. Interleaved
//! bytes `[a, b]` are `a + b*u`; multiplying by `c0 + c1*u` gives
//!
//! ```text
//! even lane = c0*a       ^ (DELTA*c1)*b
//! odd  lane = (c0+c1)*b  ^ c1*a
//! ```
//!
//! Both lanes are therefore one alternating-coefficient byte multiply of the
//! source, `XORed` with one of the source with adjacent bytes exchanged. No
//! planar de-interleave, no 16-bit multiply, no 128 KiB table.
//!
//! - **GFNI** does each byte multiply with `GF2P8MULB` and gets the two
//!   alternating coefficients straight out of
//!   [`TowerCoeff`](crate::kernel::tables::TowerCoeff) — one `vpbroadcastw`
//!   each, no table at all.
//! - **AVX2 / SSSE3** have no field multiply, so each of the four base-field
//!   factors becomes a split-nibble `PSHUFB` pair against
//!   [`TowerTables`](crate::kernel::tables::TowerTables); the even and odd
//!   byte lanes are then selected with a `0x00ff` halfword mask.
//!
//! The multi-row kernels are GFNI-only: without a native byte multiply the
//! shuffle ports, not the destination traffic, are the bottleneck, so
//! blocking buys nothing there.

use crate::field::gf16::Elem;
use crate::kernel::gf16::{mul_add_scalar, mul_assign_scalar, mul_into_scalar};
use crate::kernel::tables::{TowerCoeff, TowerTables};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// `PSHUFB` control that exchanges the two bytes of every element.
///
/// The 16-byte pattern is repeated because `PSHUFB` indexes within each
/// 128-bit lane independently — which costs nothing here, since an element
/// never straddles a lane boundary.
const SWAP_ADJACENT: [u8; 32] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, //
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
];

/// How many terms [`matrix_gfni`] folds into a destination tile at once.
///
/// A GF(2^16) coefficient must be turned into two broadcast words before it
/// can be used, which — unlike the GF(2^8) kernel's single byte splat — is
/// too expensive to redo for every tile. So a block of terms is derived once
/// per row group and then folded into every tile of that group. Destination
/// traffic is one read/write per tile per *block*, not per term, and every
/// coefficient is derived exactly once. At or below this many terms the
/// destination is touched exactly once, which is the case that matters.
const TERM_BLOCK: usize = 16;

/// The two broadcast words of a coefficient, as the halfwords `set1_epi16`
/// takes.
///
/// `to_ne_bytes`/`from_ne_bytes` rather than a cast: this is a
/// reinterpretation of the same 16 bits, not a numeric conversion.
#[inline]
fn broadcast_words(coeff: TowerCoeff) -> (i16, i16) {
    (
        i16::from_ne_bytes(coeff.same.to_ne_bytes()),
        i16::from_ne_bytes(coeff.cross.to_ne_bytes()),
    )
}

/// Load the 32-byte adjacent-exchange shuffle control.
#[inline]
#[target_feature(enable = "avx2")]
fn swap_mask256() -> __m256i {
    // SAFETY: `SWAP_ADJACENT` is 32 bytes, exactly the width of the load.
    unsafe { _mm256_loadu_si256(SWAP_ADJACENT.as_ptr().cast()) }
}

/// Load the 16-byte adjacent-exchange shuffle control.
#[inline]
#[target_feature(enable = "ssse3")]
fn swap_mask128() -> __m128i {
    // SAFETY: `SWAP_ADJACENT` is 32 bytes, so a 16-byte load of its first
    // half is in bounds; both halves hold the same pattern.
    unsafe { _mm_loadu_si128(SWAP_ADJACENT.as_ptr().cast()) }
}

/// `coeff * src` for one 32-byte lane, given the source and its
/// adjacent-exchanged self.
///
/// The exchange is the caller's job so that the multi-row kernels can shuffle
/// once and reuse the result for every destination row.
#[inline]
#[target_feature(enable = "avx2,gfni")]
fn scale_gfni(src: __m256i, swapped: __m256i, same: __m256i, cross: __m256i) -> __m256i {
    _mm256_xor_si256(
        _mm256_gf2p8mul_epi8(src, same),
        _mm256_gf2p8mul_epi8(swapped, cross),
    )
}

/// `dst ^= coeff * src` with `GF2P8MULB` over 32-byte lanes.
pub(crate) fn mul_add_gfni(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the GFNI backend, so AVX2 and GFNI are
    // present; `dst` and `src` are separately borrowed slices.
    unsafe { mul_add_gfni_impl(dst, coeff, src) }
}

/// # Safety
/// AVX2 and GFNI must be available on the host.
#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_add_gfni_impl(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let len = dst.len().min(src.len());
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    // Four independent multiply chains: `GF2P8MULB` has far more throughput
    // than latency, and a single-destination update has no other work to
    // hide it behind.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len <= dst.len().min(src.len())`, so all
        // eight loads and four stores stay inside their slices.
        unsafe {
            let sp = src_ptr.add(offset);
            let dp = dst_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            let p0 = scale_gfni(x0, _mm256_shuffle_epi8(x0, swap), same, cross);
            let p1 = scale_gfni(x1, _mm256_shuffle_epi8(x1, swap), same, cross);
            let p2 = scale_gfni(x2, _mm256_shuffle_epi8(x2, swap), same, cross);
            let p3 = scale_gfni(x3, _mm256_shuffle_epi8(x3, swap), same, cross);
            let d0 = _mm256_loadu_si256(dp.cast());
            let d1 = _mm256_loadu_si256(dp.add(32).cast());
            let d2 = _mm256_loadu_si256(dp.add(64).cast());
            let d3 = _mm256_loadu_si256(dp.add(96).cast());
            _mm256_storeu_si256(dp.cast(), _mm256_xor_si256(d0, p0));
            _mm256_storeu_si256(dp.add(32).cast(), _mm256_xor_si256(d1, p1));
            _mm256_storeu_si256(dp.add(64).cast(), _mm256_xor_si256(d2, p2));
            _mm256_storeu_si256(dp.add(96).cast(), _mm256_xor_si256(d3, p3));
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len` bounds the load pair and the store.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let scaled = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(d, scaled));
        }
        offset += 32;
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary.
    mul_add_scalar(&mut dst[offset..len], coeff.coeff, &src[offset..len]);
}

/// `dst = coeff * dst` with `GF2P8MULB` over 32-byte lanes.
pub(crate) fn mul_assign_gfni(dst: &mut [u8], coeff: TowerCoeff) {
    // SAFETY: the caller selected the GFNI backend, so AVX2 and GFNI are
    // present.
    unsafe { mul_assign_gfni_impl(dst, coeff) }
}

/// # Safety
/// AVX2 and GFNI must be available on the host.
#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_assign_gfni_impl(dst: &mut [u8], coeff: TowerCoeff) {
    let len = dst.len();
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();
    let dst_ptr = dst.as_mut_ptr();

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len()` bounds the load and store.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm256_loadu_si256(p.cast());
            _mm256_storeu_si256(
                p.cast(),
                scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross),
            );
        }
        offset += 32;
    }
    mul_assign_scalar(&mut dst[offset..], coeff.coeff);
}

/// `dst = coeff * src` with `GF2P8MULB` over 32-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
pub(crate) fn mul_into_gfni(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the GFNI backend, so AVX2 and GFNI are
    // present; `dst` and `src` are separately borrowed slices.
    unsafe { mul_into_gfni_impl(dst, coeff, src) }
}

/// # Safety
/// AVX2 and GFNI must be available on the host.
#[target_feature(enable = "avx2,gfni")]
unsafe fn mul_into_gfni_impl(dst: &mut [u8], coeff: TowerCoeff, src: &[u8]) {
    let len = dst.len().min(src.len());
    let (same_word, cross_word) = broadcast_words(coeff);
    let same = _mm256_set1_epi16(same_word);
    let cross = _mm256_set1_epi16(cross_word);
    let swap = swap_mask256();
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    // Four independent multiply chains, as in the AXPY.
    while offset + 128 <= len {
        // SAFETY: `offset + 128 <= len <= dst.len().min(src.len())`.
        unsafe {
            let sp = src_ptr.add(offset);
            let dp = dst_ptr.add(offset);
            let x0 = _mm256_loadu_si256(sp.cast());
            let x1 = _mm256_loadu_si256(sp.add(32).cast());
            let x2 = _mm256_loadu_si256(sp.add(64).cast());
            let x3 = _mm256_loadu_si256(sp.add(96).cast());
            let p0 = scale_gfni(x0, _mm256_shuffle_epi8(x0, swap), same, cross);
            let p1 = scale_gfni(x1, _mm256_shuffle_epi8(x1, swap), same, cross);
            let p2 = scale_gfni(x2, _mm256_shuffle_epi8(x2, swap), same, cross);
            let p3 = scale_gfni(x3, _mm256_shuffle_epi8(x3, swap), same, cross);
            _mm256_storeu_si256(dp.cast(), p0);
            _mm256_storeu_si256(dp.add(32).cast(), p1);
            _mm256_storeu_si256(dp.add(64).cast(), p2);
            _mm256_storeu_si256(dp.add(96).cast(), p3);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len` bounds the load pair and the store.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let scaled = scale_gfni(x, _mm256_shuffle_epi8(x, swap), same, cross);
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), scaled);
        }
        offset += 32;
    }
    // Every step above is a whole number of elements, so the tail starts on
    // an element boundary.
    mul_into_scalar(&mut dst[offset..len], coeff.coeff, &src[offset..len]);
}

/// Nibble tables and lane masks for one coefficient, held in AVX2 registers.
struct NibbleAvx2 {
    /// Low-nibble table of factor `i`, the same 16 entries in both halves.
    lo: [__m256i; 4],
    /// High-nibble table of factor `i`, the same 16 entries in both halves.
    hi: [__m256i; 4],
    /// `0x0f` in every byte: the nibble-extraction mask.
    nibble: __m256i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    even: __m256i,
    /// Adjacent-byte exchange control.
    swap: __m256i,
}

/// Widen the 16-byte tables of `tables` into AVX2 registers.
#[inline]
#[target_feature(enable = "avx2")]
fn nibble_avx2(tables: &TowerTables) -> NibbleAvx2 {
    let mut lo = [_mm256_setzero_si256(); 4];
    let mut hi = [_mm256_setzero_si256(); 4];
    for (slot, factor) in tables.factors.iter().enumerate() {
        // SAFETY: `lo` and `hi` are `[u8; 16]`, exactly the width of the load.
        unsafe {
            lo[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(factor.lo.as_ptr().cast()));
            hi[slot] = _mm256_broadcastsi128_si256(_mm_loadu_si128(factor.hi.as_ptr().cast()));
        }
    }
    NibbleAvx2 {
        lo,
        hi,
        nibble: _mm256_set1_epi8(0x0f),
        even: _mm256_set1_epi16(0x00ff),
        swap: swap_mask256(),
    }
}

/// Split `value` into its low and high nibbles, both as byte indices.
#[inline]
#[target_feature(enable = "avx2")]
fn split_avx2(value: __m256i, nibble: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_and_si256(value, nibble),
        _mm256_and_si256(_mm256_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[inline]
#[target_feature(enable = "avx2")]
fn lookup_avx2(lo: __m256i, hi: __m256i, split: (__m256i, __m256i)) -> __m256i {
    _mm256_xor_si256(
        _mm256_shuffle_epi8(lo, split.0),
        _mm256_shuffle_epi8(hi, split.1),
    )
}

/// `coeff * src` for one 32-byte lane, via four nibble-shuffle multiplies.
///
/// The even and odd contributions are summed *before* masking rather than
/// after: `AND` distributes over `XOR`, so grouping by lane parity halves the
/// mask operations relative to grouping by direct/crossed.
#[inline]
#[target_feature(enable = "avx2")]
fn scale_avx2(src: __m256i, tables: &NibbleAvx2) -> __m256i {
    let swapped = _mm256_shuffle_epi8(src, tables.swap);
    let direct = split_avx2(src, tables.nibble);
    let crossed = split_avx2(swapped, tables.nibble);
    let even = _mm256_xor_si256(
        lookup_avx2(tables.lo[0], tables.hi[0], direct),
        lookup_avx2(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm256_xor_si256(
        lookup_avx2(tables.lo[1], tables.hi[1], direct),
        lookup_avx2(tables.lo[3], tables.hi[3], crossed),
    );
    _mm256_xor_si256(
        _mm256_and_si256(even, tables.even),
        _mm256_andnot_si256(tables.even, odd),
    )
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 32-byte lanes.
pub(crate) fn mul_add_avx2(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the AVX2 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_add_avx2_impl(dst, tables, src) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_add_avx2_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let len = dst.len().min(src.len());
    let vectors = nibble_avx2(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len <= dst.len().min(src.len())`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            let d = _mm256_loadu_si256(dst_ptr.add(offset).cast());
            let scaled = scale_avx2(x, &vectors);
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), _mm256_xor_si256(d, scaled));
        }
        offset += 32;
    }
    mul_add_scalar(&mut dst[offset..len], tables.coeff, &src[offset..len]);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 32-byte lanes.
pub(crate) fn mul_assign_avx2(dst: &mut [u8], tables: &TowerTables) {
    // SAFETY: the caller selected the AVX2 backend.
    unsafe { mul_assign_avx2_impl(dst, tables) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_assign_avx2_impl(dst: &mut [u8], tables: &TowerTables) {
    let len = dst.len();
    let vectors = nibble_avx2(tables);
    let dst_ptr = dst.as_mut_ptr();

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len()`.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm256_loadu_si256(p.cast());
            _mm256_storeu_si256(p.cast(), scale_avx2(x, &vectors));
        }
        offset += 32;
    }
    mul_assign_scalar(&mut dst[offset..], tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
pub(crate) fn mul_into_avx2(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the AVX2 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_into_avx2_impl(dst, tables, src) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_into_avx2_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let len = dst.len().min(src.len());
    let vectors = nibble_avx2(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len <= dst.len().min(src.len())`.
        unsafe {
            let x = _mm256_loadu_si256(src_ptr.add(offset).cast());
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), scale_avx2(x, &vectors));
        }
        offset += 32;
    }
    mul_into_scalar(&mut dst[offset..len], tables.coeff, &src[offset..len]);
}

/// Nibble tables and lane masks for one coefficient, held in SSE registers.
struct NibbleSsse3 {
    /// Low-nibble table of factor `i`.
    lo: [__m128i; 4],
    /// High-nibble table of factor `i`.
    hi: [__m128i; 4],
    /// `0x0f` in every byte: the nibble-extraction mask.
    nibble: __m128i,
    /// `0x00ff` in every halfword: selects each element's even (low) byte.
    even: __m128i,
    /// Adjacent-byte exchange control.
    swap: __m128i,
}

/// Load the 16-byte tables of `tables` into SSE registers.
#[inline]
#[target_feature(enable = "ssse3")]
fn nibble_ssse3(tables: &TowerTables) -> NibbleSsse3 {
    let mut lo = [_mm_setzero_si128(); 4];
    let mut hi = [_mm_setzero_si128(); 4];
    for (slot, factor) in tables.factors.iter().enumerate() {
        // SAFETY: `lo` and `hi` are `[u8; 16]`, exactly the width of the load.
        unsafe {
            lo[slot] = _mm_loadu_si128(factor.lo.as_ptr().cast());
            hi[slot] = _mm_loadu_si128(factor.hi.as_ptr().cast());
        }
    }
    NibbleSsse3 {
        lo,
        hi,
        nibble: _mm_set1_epi8(0x0f),
        even: _mm_set1_epi16(0x00ff),
        swap: swap_mask128(),
    }
}

/// Split `value` into its low and high nibbles, both as byte indices.
#[inline]
#[target_feature(enable = "ssse3")]
fn split_ssse3(value: __m128i, nibble: __m128i) -> (__m128i, __m128i) {
    (
        _mm_and_si128(value, nibble),
        _mm_and_si128(_mm_srli_epi16(value, 4), nibble),
    )
}

/// One base-field byte multiply: two table lookups over pre-split nibbles.
#[inline]
#[target_feature(enable = "ssse3")]
fn lookup_ssse3(lo: __m128i, hi: __m128i, split: (__m128i, __m128i)) -> __m128i {
    _mm_xor_si128(_mm_shuffle_epi8(lo, split.0), _mm_shuffle_epi8(hi, split.1))
}

/// `coeff * src` for one 16-byte lane, via four nibble-shuffle multiplies.
#[inline]
#[target_feature(enable = "ssse3")]
fn scale_ssse3(src: __m128i, tables: &NibbleSsse3) -> __m128i {
    let swapped = _mm_shuffle_epi8(src, tables.swap);
    let direct = split_ssse3(src, tables.nibble);
    let crossed = split_ssse3(swapped, tables.nibble);
    let even = _mm_xor_si128(
        lookup_ssse3(tables.lo[0], tables.hi[0], direct),
        lookup_ssse3(tables.lo[2], tables.hi[2], crossed),
    );
    let odd = _mm_xor_si128(
        lookup_ssse3(tables.lo[1], tables.hi[1], direct),
        lookup_ssse3(tables.lo[3], tables.hi[3], crossed),
    );
    _mm_xor_si128(
        _mm_and_si128(even, tables.even),
        _mm_andnot_si128(tables.even, odd),
    )
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 16-byte lanes.
pub(crate) fn mul_add_ssse3(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the SSSE3 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_add_ssse3_impl(dst, tables, src) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_add_ssse3_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let len = dst.len().min(src.len());
    let vectors = nibble_ssse3(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len <= dst.len().min(src.len())`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let scaled = scale_ssse3(x, &vectors);
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, scaled));
        }
        offset += 16;
    }
    mul_add_scalar(&mut dst[offset..len], tables.coeff, &src[offset..len]);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 16-byte lanes.
pub(crate) fn mul_assign_ssse3(dst: &mut [u8], tables: &TowerTables) {
    // SAFETY: the caller selected the SSSE3 backend.
    unsafe { mul_assign_ssse3_impl(dst, tables) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_assign_ssse3_impl(dst: &mut [u8], tables: &TowerTables) {
    let len = dst.len();
    let vectors = nibble_ssse3(tables);
    let dst_ptr = dst.as_mut_ptr();

    let mut offset = 0;
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len == dst.len()`.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm_loadu_si128(p.cast());
            _mm_storeu_si128(p.cast(), scale_ssse3(x, &vectors));
        }
        offset += 16;
    }
    mul_assign_scalar(&mut dst[offset..], tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes, out of place.
///
/// Fused form of copy-then-scale: the `mul_add` body without the destination
/// read, one pass.
pub(crate) fn mul_into_ssse3(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: the caller selected the SSSE3 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_into_ssse3_impl(dst, tables, src) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_into_ssse3_impl(dst: &mut [u8], tables: &TowerTables, src: &[u8]) {
    let len = dst.len().min(src.len());
    let vectors = nibble_ssse3(tables);
    let (dst_ptr, src_ptr) = (dst.as_mut_ptr(), src.as_ptr());

    let mut offset = 0;
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len <= dst.len().min(src.len())`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), scale_ssse3(x, &vectors));
        }
        offset += 16;
    }
    mul_into_scalar(&mut dst[offset..len], tables.coeff, &src[offset..len]);
}

/// `rows[j] ^= coeffs[j] * src` for every row, with `GF2P8MULB`.
///
/// Rows are updated in groups of four, then a pair, then one at a time. Each
/// group loads the source once and exchanges its bytes once, so those costs
/// are amortized over the whole group instead of paid per row.
///
/// Coefficients of zero and one need no special case: `TowerCoeff` reduces
/// them to the broadcast pairs `(0, 0)` and `(0x0101, 0)`, which the same
/// multiply turns into a no-op and a plain XOR respectively.
pub(crate) fn scatter_gfni(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    debug_assert_eq!(row_len, src.len());
    if row_len == 0 {
        return;
    }
    debug_assert_eq!(rows.len() / row_len, coeffs.len());
    let nrows = coeffs.len().min(rows.len() / row_len);
    let span = row_len.min(src.len());
    if nrows == 0 || span == 0 {
        return;
    }
    // SAFETY: the caller selected the GFNI backend, so AVX2 and GFNI are
    // present. `nrows` is clamped so rows `0..nrows` all fit in `rows`, and
    // `span <= row_len` keeps each row's window inside its own stride.
    unsafe { scatter_gfni_impl(rows, row_len, span, &coeffs[..nrows], src) }
}

/// # Safety
/// AVX2 and GFNI must be available, `coeffs.len() * stride <= rows.len()`,
/// and `span <= stride.min(src.len())`.
#[target_feature(enable = "avx2,gfni")]
unsafe fn scatter_gfni_impl(
    rows: &mut [u8],
    stride: usize,
    span: usize,
    coeffs: &[Elem],
    src: &[u8],
) {
    let base = rows.as_mut_ptr();
    let swap = swap_mask256();
    let mut j = 0;
    while j + 4 <= coeffs.len() {
        let mut group = [Elem(0); 4];
        group.copy_from_slice(&coeffs[j..j + 4]);
        // SAFETY: rows `j..j + 4` lie within `rows` and, being `stride` bytes
        // apart with a `span <= stride` window, are pairwise disjoint.
        unsafe { scatter_group_gfni(base.add(j * stride), stride, span, group, src, swap) };
        j += 4;
    }
    if j + 2 <= coeffs.len() {
        let mut group = [Elem(0); 2];
        group.copy_from_slice(&coeffs[j..j + 2]);
        // SAFETY: as above, for the two-row remainder.
        unsafe { scatter_group_gfni(base.add(j * stride), stride, span, group, src, swap) };
        j += 2;
    }
    if j < coeffs.len() {
        // SAFETY: row `j` is the last row and lies within `rows`; the
        // unrolled single-destination kernel is the better shape for it.
        unsafe {
            let row = core::slice::from_raw_parts_mut(base.add(j * stride), span);
            mul_add_gfni_impl(row, TowerCoeff::new(coeffs[j]), &src[..span]);
        }
    }
}

/// `row[k] ^= coeffs[k] * src` for `N` consecutive rows starting at `base`.
///
/// # Safety
/// AVX2 and GFNI must be available, the `N` rows at `base + k * stride` must
/// be readable and writable for `span` bytes, and `span <= src.len()`.
#[target_feature(enable = "avx2,gfni")]
unsafe fn scatter_group_gfni<const N: usize>(
    base: *mut u8,
    stride: usize,
    span: usize,
    coeffs: [Elem; N],
    src: &[u8],
    swap: __m256i,
) {
    let mut rows = [core::ptr::null_mut::<u8>(); N];
    for (k, row) in rows.iter_mut().enumerate() {
        // SAFETY: the caller guarantees all `N` rows are in bounds.
        *row = unsafe { base.add(k * stride) };
    }
    // Derived once per group, never inside the byte loop.
    let mut same = [_mm256_setzero_si256(); N];
    let mut cross = [_mm256_setzero_si256(); N];
    for (k, &coeff) in coeffs.iter().enumerate() {
        let (same_word, cross_word) = broadcast_words(TowerCoeff::new(coeff));
        same[k] = _mm256_set1_epi16(same_word);
        cross[k] = _mm256_set1_epi16(cross_word);
    }

    let src_ptr = src.as_ptr();
    let mut offset = 0;
    while offset + 32 <= span {
        // SAFETY: `offset + 32 <= span <= src.len()` bounds the source load.
        let x = unsafe { _mm256_loadu_si256(src_ptr.add(offset).cast()) };
        let swapped = _mm256_shuffle_epi8(x, swap);
        for (k, &row) in rows.iter().enumerate() {
            // SAFETY: `offset + 32 <= span`, so this row's load and store are
            // inside it; distinct `k` address disjoint rows.
            unsafe {
                let p = row.add(offset);
                let d = _mm256_loadu_si256(p.cast());
                let scaled = scale_gfni(x, swapped, same[k], cross[k]);
                _mm256_storeu_si256(p.cast(), _mm256_xor_si256(d, scaled));
            }
        }
        offset += 32;
    }
    for (k, &coeff) in coeffs.iter().enumerate() {
        // SAFETY: `offset..span` is the untouched tail of row `k`, and 32-byte
        // steps leave it starting on an element boundary.
        let tail = unsafe { core::slice::from_raw_parts_mut(rows[k].add(offset), span - offset) };
        mul_add_scalar(tail, coeff, &src[offset..span]);
    }
}

/// Apply every `(coeffs, src)` term to all `nrows` rows, with `GF2P8MULB`.
///
/// Register-blocked: a destination tile is loaded into accumulators once,
/// every term of a block is folded in, and the tile is stored once. The
/// non-blocked shape re-streams each destination row per term, which is what
/// dominates once `nrows * row_len` leaves L1.
pub(crate) fn matrix_gfni(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    debug_assert!(
        terms
            .iter()
            .all(|&(coeffs, src)| coeffs.len() == nrows && src.len() == row_len)
    );
    if row_len == 0 || nrows == 0 || terms.is_empty() {
        return;
    }
    let mut nrows = nrows.min(rows.len() / row_len);
    let mut span = row_len;
    for &(coeffs, src) in terms {
        nrows = nrows.min(coeffs.len());
        span = span.min(src.len());
    }
    if nrows == 0 || span == 0 {
        return;
    }
    // SAFETY: the caller selected the GFNI backend, so AVX2 and GFNI are
    // present. `nrows` is clamped to what `rows` holds and to the shortest
    // coefficient array; `span` to the shortest source.
    unsafe { matrix_gfni_impl(rows, row_len, span, nrows, terms) }
}

/// # Safety
/// AVX2 and GFNI must be available, `nrows * stride <= rows.len()`, every
/// term must supply at least `nrows` coefficients, and `span` must be at most
/// `stride` and at most every term's source length.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_gfni_impl(
    rows: &mut [u8],
    stride: usize,
    span: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    let base = rows.as_mut_ptr();
    let swap = swap_mask256();
    let mut j = 0;
    while j + 4 <= nrows {
        // SAFETY: rows `j..j + 4` lie within `rows` and, being `stride` bytes
        // apart with a `span <= stride` window, are pairwise disjoint.
        unsafe { matrix_group_gfni::<4>(base.add(j * stride), stride, span, j, terms, swap) };
        j += 4;
    }
    if j + 2 <= nrows {
        // SAFETY: as above, for the two-row remainder.
        unsafe { matrix_group_gfni::<2>(base.add(j * stride), stride, span, j, terms, swap) };
        j += 2;
    }
    if j < nrows {
        // SAFETY: as above, for the final row.
        unsafe { matrix_group_gfni::<1>(base.add(j * stride), stride, span, j, terms, swap) };
    }
}

/// Fold every term into `N` consecutive rows starting at `base`, which is row
/// `first` of the destination.
///
/// # Safety
/// AVX2 and GFNI must be available, the `N` rows at `base + k * stride` must
/// be readable and writable for `span` bytes, every term must supply more
/// than `first + N - 1` coefficients, and `span` must not exceed any term's
/// source length.
#[target_feature(enable = "avx2,gfni")]
unsafe fn matrix_group_gfni<const N: usize>(
    base: *mut u8,
    stride: usize,
    span: usize,
    first: usize,
    terms: &[(&[Elem], &[u8])],
    swap: __m256i,
) {
    let mut rows = [core::ptr::null_mut::<u8>(); N];
    for (k, row) in rows.iter_mut().enumerate() {
        // SAFETY: the caller guarantees all `N` rows are in bounds.
        *row = unsafe { base.add(k * stride) };
    }

    for block in terms.chunks(TERM_BLOCK) {
        // Every coefficient of the block is derived exactly once, here,
        // outside the byte loop. Kept as the raw broadcast words rather than
        // as vectors: `vpbroadcastw` reads memory directly, so a register
        // spilled to the stack would cost the same reload anyway.
        let mut words = [[(0i16, 0i16); N]; TERM_BLOCK];
        for (t, &(coeffs, _)) in block.iter().enumerate() {
            for (k, slot) in words[t].iter_mut().enumerate() {
                *slot = broadcast_words(TowerCoeff::new(coeffs[first + k]));
            }
        }

        let mut offset = 0;
        while offset + 32 <= span {
            let mut acc = [_mm256_setzero_si256(); N];
            for (k, a) in acc.iter_mut().enumerate() {
                // SAFETY: `offset + 32 <= span` bounds this row's load.
                *a = unsafe { _mm256_loadu_si256(rows[k].add(offset).cast()) };
            }
            for (t, &(_, src)) in block.iter().enumerate() {
                // SAFETY: `offset + 32 <= span <= src.len()`.
                let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
                let swapped = _mm256_shuffle_epi8(x, swap);
                for (k, a) in acc.iter_mut().enumerate() {
                    let (same, cross) = words[t][k];
                    let scaled = scale_gfni(
                        x,
                        swapped,
                        _mm256_set1_epi16(same),
                        _mm256_set1_epi16(cross),
                    );
                    *a = _mm256_xor_si256(*a, scaled);
                }
            }
            for (k, &a) in acc.iter().enumerate() {
                // SAFETY: same bound and disjointness as the matching load.
                unsafe { _mm256_storeu_si256(rows[k].add(offset).cast(), a) };
            }
            offset += 32;
        }

        if offset < span {
            for (k, &row) in rows.iter().enumerate() {
                // SAFETY: `offset..span` is the untouched tail of row `k`, and
                // 32-byte steps leave it on an element boundary.
                let tail =
                    unsafe { core::slice::from_raw_parts_mut(row.add(offset), span - offset) };
                for &(coeffs, src) in block {
                    mul_add_scalar(tail, coeffs[first + k], &src[offset..span]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1: varying operands and blocked shuffle shapes.
// ---------------------------------------------------------------------------

/// `dst[i] = a[i] * b[i]` over interleaved tower elements using GFNI.
pub(crate) fn elementwise_gfni(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    // SAFETY: the selected backend guarantees AVX2 and GFNI.
    unsafe { elementwise_gfni_impl(dst, a, b) }
}

#[target_feature(enable = "avx2,gfni")]
unsafe fn elementwise_gfni_impl(dst: &mut [u8], a: &[u8], b: &[u8]) {
    let len = dst.len().min(a.len()).min(b.len()) & !31;
    let (dst_ptr, a_ptr, b_ptr) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let swap = swap_mask256();
    let even = _mm256_set1_epi16(0x00ff);
    let delta_even = _mm256_set1_epi16(i16::from_ne_bytes([crate::field::gf16::DELTA.0, 0]));
    let mut offset = 0;
    while offset < len {
        // For x=[a,b], y=[c,d]:
        // constant = ac ^ DELTA*bd
        // extension = ad ^ bc ^ bd.
        // SAFETY: `offset + 32 <= len`, which bounds all three slices.
        unsafe {
            let x = _mm256_loadu_si256(a_ptr.add(offset).cast());
            let y = _mm256_loadu_si256(b_ptr.add(offset).cast());
            let direct = _mm256_gf2p8mul_epi8(x, y); // [ac, bd]
            let crossed = _mm256_gf2p8mul_epi8(x, _mm256_shuffle_epi8(y, swap)); // [ad, bc]
            let delta_bd = _mm256_gf2p8mul_epi8(_mm256_shuffle_epi8(direct, swap), delta_even);
            let constant = _mm256_xor_si256(direct, delta_bd);
            let cross_sum = _mm256_xor_si256(crossed, _mm256_shuffle_epi8(crossed, swap));
            let extension = _mm256_xor_si256(cross_sum, direct);
            let product = _mm256_xor_si256(
                _mm256_and_si256(constant, even),
                _mm256_andnot_si256(even, extension),
            );
            _mm256_storeu_si256(dst_ptr.add(offset).cast(), product);
        }
        offset += 32;
    }
    for ((d, x), y) in dst[len..]
        .chunks_exact_mut(2)
        .zip(a[len..].chunks_exact(2))
        .zip(b[len..].chunks_exact(2))
    {
        d.copy_from_slice(
            &Elem::from_bytes([x[0], x[1]])
                .mul(Elem::from_bytes([y[0], y[1]]))
                .to_bytes(),
        );
    }
}

const TERM_TILE: usize = 8;

/// Many sources into one destination, eight coefficients prepared per pass.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gather_avx2(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees AVX2.
    unsafe { gather_avx2_impl(dst, coeffs, srcs) }
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "avx2")]
unsafe fn gather_avx2_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let vector_len = dst.len() & !31;
    for block in (0..coeffs.len()).step_by(TERM_TILE) {
        let count = (coeffs.len() - block).min(TERM_TILE);
        let tables: [TowerTables; TERM_TILE] =
            core::array::from_fn(|i| TowerTables::new(coeffs[block + i.min(count - 1)]));
        let vectors: [NibbleAvx2; TERM_TILE] = core::array::from_fn(|i| nibble_avx2(&tables[i]));
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: this 32-byte window lies within `dst`.
            let mut acc = unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast()) };
            for i in 0..count {
                // SAFETY: every source has `dst.len()` bytes.
                let source =
                    unsafe { _mm256_loadu_si256(srcs[block + i].as_ptr().add(offset).cast()) };
                acc = _mm256_xor_si256(acc, scale_avx2(source, &vectors[i]));
            }
            // SAFETY: the destination window loaded above.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(offset).cast(), acc) };
            offset += 32;
        }
        for i in 0..count {
            mul_add_scalar(
                &mut dst[vector_len..],
                coeffs[block + i],
                &srcs[block + i][vector_len..],
            );
        }
    }
}

/// Many sources into one destination, eight coefficients prepared per pass.
pub(crate) fn gather_ssse3(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees SSSE3.
    unsafe { gather_ssse3_impl(dst, coeffs, srcs) }
}

#[target_feature(enable = "ssse3")]
unsafe fn gather_ssse3_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let vector_len = dst.len() & !15;
    for block in (0..coeffs.len()).step_by(TERM_TILE) {
        let count = (coeffs.len() - block).min(TERM_TILE);
        let tables: [TowerTables; TERM_TILE] =
            core::array::from_fn(|i| TowerTables::new(coeffs[block + i.min(count - 1)]));
        let vectors: [NibbleSsse3; TERM_TILE] = core::array::from_fn(|i| nibble_ssse3(&tables[i]));
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: this 16-byte window lies within `dst`.
            let mut acc = unsafe { _mm_loadu_si128(dst.as_ptr().add(offset).cast()) };
            for i in 0..count {
                // SAFETY: every source has `dst.len()` bytes.
                let source =
                    unsafe { _mm_loadu_si128(srcs[block + i].as_ptr().add(offset).cast()) };
                acc = _mm_xor_si128(acc, scale_ssse3(source, &vectors[i]));
            }
            // SAFETY: the destination window loaded above.
            unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast(), acc) };
            offset += 16;
        }
        for i in 0..count {
            mul_add_scalar(
                &mut dst[vector_len..],
                coeffs[block + i],
                &srcs[block + i][vector_len..],
            );
        }
    }
}

/// GFNI gather: all coefficients remain compact, so one pass can fold every
/// source without term blocking.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gather_gfni(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    debug_assert_eq!(coeffs.len(), srcs.len());
    // SAFETY: the selected backend guarantees AVX2 and GFNI.
    unsafe { gather_gfni_impl(dst, coeffs, srcs) }
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "avx2,gfni")]
unsafe fn gather_gfni_impl(dst: &mut [u8], coeffs: &[Elem], srcs: &[&[u8]]) {
    let vector_len = dst.len() & !31;
    let swap = swap_mask256();
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: this 32-byte window lies within `dst`.
        let mut acc = unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast()) };
        for (&coeff, &src) in coeffs.iter().zip(srcs) {
            let compact = TowerCoeff::new(coeff);
            let (same_word, cross_word) = broadcast_words(compact);
            let same = _mm256_set1_epi16(same_word);
            let cross = _mm256_set1_epi16(cross_word);
            // SAFETY: every source has `dst.len()` bytes.
            let source = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
            acc = _mm256_xor_si256(
                acc,
                scale_gfni(source, _mm256_shuffle_epi8(source, swap), same, cross),
            );
        }
        // SAFETY: the destination window loaded above.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(offset).cast(), acc) };
        offset += 32;
    }
    for (&coeff, &src) in coeffs.iter().zip(srcs) {
        mul_add_scalar(&mut dst[vector_len..], coeff, &src[vector_len..]);
    }
}

/// One source into many rows using four AVX2 table sets at a time.
pub(crate) fn scatter_avx2(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    // SAFETY: the selected backend guarantees AVX2 and geometry was checked.
    unsafe { scatter_avx2_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "avx2")]
unsafe fn scatter_avx2_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !31;
    let base = rows.as_mut_ptr();
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let tables: [TowerTables; 4] =
            core::array::from_fn(|i| TowerTables::new(coeffs[group + i.min(count - 1)]));
        let vectors: [NibbleAvx2; 4] = core::array::from_fn(|i| nibble_avx2(&tables[i]));
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: this source window is in bounds.
            let source = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast()) };
            for (slot, vector) in vectors.iter().take(count).enumerate() {
                // SAFETY: the selected row window is in bounds and disjoint.
                unsafe {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm256_storeu_si256(
                        ptr.cast(),
                        _mm256_xor_si256(
                            _mm256_loadu_si256(ptr.cast()),
                            scale_avx2(source, vector),
                        ),
                    );
                }
            }
            offset += 32;
        }
        for slot in 0..count {
            // SAFETY: this is one row's disjoint scalar tail.
            let tail = unsafe {
                core::slice::from_raw_parts_mut(
                    base.add((group + slot) * row_len + vector_len),
                    row_len - vector_len,
                )
            };
            mul_add_scalar(tail, coeffs[group + slot], &src[vector_len..]);
        }
    }
}

/// One source into many rows using four SSSE3 table sets at a time.
pub(crate) fn scatter_ssse3(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    // SAFETY: the selected backend guarantees SSSE3 and geometry was checked.
    unsafe { scatter_ssse3_impl(rows, row_len, coeffs, src) }
}

#[target_feature(enable = "ssse3")]
unsafe fn scatter_ssse3_impl(rows: &mut [u8], row_len: usize, coeffs: &[Elem], src: &[u8]) {
    if row_len == 0 {
        return;
    }
    let vector_len = row_len & !15;
    let base = rows.as_mut_ptr();
    for group in (0..coeffs.len()).step_by(4) {
        let count = (coeffs.len() - group).min(4);
        let tables: [TowerTables; 4] =
            core::array::from_fn(|i| TowerTables::new(coeffs[group + i.min(count - 1)]));
        let vectors: [NibbleSsse3; 4] = core::array::from_fn(|i| nibble_ssse3(&tables[i]));
        let mut offset = 0;
        while offset < vector_len {
            // SAFETY: this source window is in bounds.
            let source = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast()) };
            for (slot, vector) in vectors.iter().take(count).enumerate() {
                // SAFETY: the selected row window is in bounds and disjoint.
                unsafe {
                    let ptr = base.add((group + slot) * row_len + offset);
                    _mm_storeu_si128(
                        ptr.cast(),
                        _mm_xor_si128(_mm_loadu_si128(ptr.cast()), scale_ssse3(source, vector)),
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
            mul_add_scalar(tail, coeffs[group + slot], &src[vector_len..]);
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
    for group in (0..nrows).step_by(4) {
        let row_count = (nrows - group).min(4);
        for block in (0..terms.len()).step_by(TERM_TILE) {
            let term_count = (terms.len() - block).min(TERM_TILE);
            let tables: [[TowerTables; 4]; TERM_TILE] = core::array::from_fn(|t| {
                core::array::from_fn(|r| {
                    TowerTables::new(
                        terms[block + t.min(term_count - 1)].0[group + r.min(row_count - 1)],
                    )
                })
            });
            let vectors: [[NibbleAvx2; 4]; TERM_TILE] =
                core::array::from_fn(|t| core::array::from_fn(|r| nibble_avx2(&tables[t][r])));
            let mut offset = 0;
            while offset < vector_len {
                let mut acc = [_mm256_setzero_si256(); 4];
                // SAFETY: every selected row contains this window.
                unsafe {
                    for (r, slot) in acc.iter_mut().take(row_count).enumerate() {
                        *slot = _mm256_loadu_si256(base.add((group + r) * row_len + offset).cast());
                    }
                }
                for t in 0..term_count {
                    // SAFETY: every term source is exactly `row_len` bytes.
                    let source = unsafe {
                        _mm256_loadu_si256(terms[block + t].1.as_ptr().add(offset).cast())
                    };
                    for r in 0..row_count {
                        acc[r] = _mm256_xor_si256(acc[r], scale_avx2(source, &vectors[t][r]));
                    }
                }
                // SAFETY: the same disjoint row windows loaded above.
                unsafe {
                    for (r, &slot) in acc.iter().take(row_count).enumerate() {
                        _mm256_storeu_si256(base.add((group + r) * row_len + offset).cast(), slot);
                    }
                }
                offset += 32;
            }
            for r in 0..row_count {
                // SAFETY: this is one row's disjoint scalar tail.
                let tail = unsafe {
                    core::slice::from_raw_parts_mut(
                        base.add((group + r) * row_len + vector_len),
                        row_len - vector_len,
                    )
                };
                for &(coeffs, src) in &terms[block..block + term_count] {
                    mul_add_scalar(tail, coeffs[group + r], &src[vector_len..]);
                }
            }
        }
    }
}

/// Many sources into many rows using SSSE3 nibble shuffles.
pub(crate) fn matrix_ssse3(
    rows: &mut [u8],
    row_len: usize,
    nrows: usize,
    terms: &[(&[Elem], &[u8])],
) {
    // SAFETY: the selected backend guarantees SSSE3 and geometry was checked.
    unsafe { matrix_ssse3_impl(rows, row_len, nrows, terms) }
}

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
    for group in (0..nrows).step_by(4) {
        let row_count = (nrows - group).min(4);
        for block in (0..terms.len()).step_by(TERM_TILE) {
            let term_count = (terms.len() - block).min(TERM_TILE);
            let tables: [[TowerTables; 4]; TERM_TILE] = core::array::from_fn(|t| {
                core::array::from_fn(|r| {
                    TowerTables::new(
                        terms[block + t.min(term_count - 1)].0[group + r.min(row_count - 1)],
                    )
                })
            });
            let vectors: [[NibbleSsse3; 4]; TERM_TILE] =
                core::array::from_fn(|t| core::array::from_fn(|r| nibble_ssse3(&tables[t][r])));
            let mut offset = 0;
            while offset < vector_len {
                let mut acc = [_mm_setzero_si128(); 4];
                // SAFETY: every selected row contains this window.
                unsafe {
                    for (r, slot) in acc.iter_mut().take(row_count).enumerate() {
                        *slot = _mm_loadu_si128(base.add((group + r) * row_len + offset).cast());
                    }
                }
                for t in 0..term_count {
                    // SAFETY: every term source is exactly `row_len` bytes.
                    let source =
                        unsafe { _mm_loadu_si128(terms[block + t].1.as_ptr().add(offset).cast()) };
                    for r in 0..row_count {
                        acc[r] = _mm_xor_si128(acc[r], scale_ssse3(source, &vectors[t][r]));
                    }
                }
                // SAFETY: the same disjoint row windows loaded above.
                unsafe {
                    for (r, &slot) in acc.iter().take(row_count).enumerate() {
                        _mm_storeu_si128(base.add((group + r) * row_len + offset).cast(), slot);
                    }
                }
                offset += 16;
            }
            for r in 0..row_count {
                // SAFETY: this is one row's disjoint scalar tail.
                let tail = unsafe {
                    core::slice::from_raw_parts_mut(
                        base.add((group + r) * row_len + vector_len),
                        row_len - vector_len,
                    )
                };
                for &(coeffs, src) in &terms[block..block + term_count] {
                    mul_add_scalar(tail, coeffs[group + r], &src[vector_len..]);
                }
            }
        }
    }
}
