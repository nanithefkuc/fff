//! Canonical Fan–Paar tower kernels for x86 / `x86_64`.
//!
//! A Fan–Paar GF(2^16) multiply is the same four-nibble-shuffle shape as the
//! polynomial GF(2^16) kernel: two alternating `fp8` byte multiplies of the
//! source and its adjacent-swap, under the period-2 coefficient pair from
//! [`crate::kernel::tables::FpTowerTables`]. The only difference is the base
//! field — the canonical Fan–Paar byte field is *not* the AES field, so the
//! nibble tables are filled from `fp8` arithmetic instead of GF(2^8), and
//! there is no GFNI fast path. The shuffle multiply core in `kernel::x86::gf16`
//! reads only the nibble tables, so this module is just the per-kernel dispatch
//! loops over a pre-built [`FpTowerTables`].
//!
//! See `kernel/fan_paar.rs` for the dispatch and the algebraic fold that
//! keeps `mul_alpha` in coefficient preparation.

use crate::field::fan_paar::FanPaar16;
use crate::kernel::scalar;
use crate::kernel::tables::FpTowerTables;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::gf16::{nibble_avx2, nibble_ssse3, scale_avx2, scale_ssse3};

/// `dst ^= coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2).
pub fn mul_add_avx2(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch selected the AVX2 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_add_avx2_impl(dst, tables, src) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_add_avx2_impl(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
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
    // One 128-bit step down before the scalar tail; SSSE3 is implied by AVX2.
    if offset + 16 <= len {
        let narrow = nibble_ssse3(tables);
        // SAFETY: `offset + 16 <= len <= dst.len().min(src.len())` bounds the
        // load pair and the store.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            let d = _mm_loadu_si128(dst_ptr.add(offset).cast());
            let scaled = scale_ssse3(x, &narrow);
            _mm_storeu_si128(dst_ptr.add(offset).cast(), _mm_xor_si128(d, scaled));
        }
        offset += 16;
    }
    // Both steps above are a whole number of 2-byte elements, so the tail
    // starts on an element boundary. Recover the coefficient from the tables
    // it was built from for the portable fallback.
    let coeff = tables.coeff;
    scalar::mul_add::<FanPaar16>(&mut dst[offset..len], coeff, &src[offset..len]);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 32-byte lanes (AVX2).
pub fn mul_assign_avx2(dst: &mut [u8], tables: &FpTowerTables) {
    // SAFETY: dispatch selected the AVX2 backend.
    unsafe { mul_assign_avx2_impl(dst, tables) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_assign_avx2_impl(dst: &mut [u8], tables: &FpTowerTables) {
    let len = dst.len();
    let vectors = nibble_avx2(tables);
    let dst_ptr = dst.as_mut_ptr();

    let mut offset = 0;
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len == dst.len()` bounds the load and store.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm256_loadu_si256(p.cast());
            _mm256_storeu_si256(p.cast(), scale_avx2(x, &vectors));
        }
        offset += 32;
    }
    if offset + 16 <= len {
        let narrow = nibble_ssse3(tables);
        // SAFETY: `offset + 16 <= len == dst.len()` bounds the load and store.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm_loadu_si128(p.cast());
            _mm_storeu_si128(p.cast(), scale_ssse3(x, &narrow));
        }
        offset += 16;
    }
    scalar::mul_assign::<FanPaar16>(&mut dst[offset..len], tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 32-byte lanes (AVX2), fused.
pub fn mul_into_avx2(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch selected the AVX2 backend; `dst` and `src` are
    // separately borrowed slices.
    unsafe { mul_into_avx2_impl(dst, tables, src) }
}

/// # Safety
/// AVX2 must be available on the host.
#[target_feature(enable = "avx2")]
unsafe fn mul_into_avx2_impl(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
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
    if offset + 16 <= len {
        let narrow = nibble_ssse3(tables);
        // SAFETY: `offset + 16 <= len <= dst.len().min(src.len())`.
        unsafe {
            let x = _mm_loadu_si128(src_ptr.add(offset).cast());
            _mm_storeu_si128(dst_ptr.add(offset).cast(), scale_ssse3(x, &narrow));
        }
        offset += 16;
    }
    // Copy-then-scale the sub-lane tail: the scalar kernel reads `dst` as its
    // own source, so seeding it with `src` first matches the fused body.
    dst[offset..len].copy_from_slice(&src[offset..len]);
    scalar::mul_assign::<FanPaar16>(&mut dst[offset..len], tables.coeff);
}

/// `dst ^= coeff * src` with `PSHUFB` lookups over 16-byte lanes (SSSE3).
pub fn mul_add_ssse3(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch selected the SSSE3 backend.
    unsafe { mul_add_ssse3_impl(dst, tables, src) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_add_ssse3_impl(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
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
    scalar::mul_add::<FanPaar16>(&mut dst[offset..len], tables.coeff, &src[offset..len]);
}

/// `dst = coeff * dst` with `PSHUFB` lookups over 16-byte lanes (SSSE3).
pub fn mul_assign_ssse3(dst: &mut [u8], tables: &FpTowerTables) {
    // SAFETY: dispatch selected the SSSE3 backend.
    unsafe { mul_assign_ssse3_impl(dst, tables) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_assign_ssse3_impl(dst: &mut [u8], tables: &FpTowerTables) {
    let len = dst.len();
    let vectors = nibble_ssse3(tables);
    let dst_ptr = dst.as_mut_ptr();

    let mut offset = 0;
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len == dst.len()` bounds the load and store.
        unsafe {
            let p = dst_ptr.add(offset);
            let x = _mm_loadu_si128(p.cast());
            _mm_storeu_si128(p.cast(), scale_ssse3(x, &vectors));
        }
        offset += 16;
    }
    scalar::mul_assign::<FanPaar16>(&mut dst[offset..len], tables.coeff);
}

/// `dst = coeff * src` with `PSHUFB` lookups over 16-byte lanes (SSSE3), fused.
pub fn mul_into_ssse3(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: dispatch selected the SSSE3 backend.
    unsafe { mul_into_ssse3_impl(dst, tables, src) }
}

/// # Safety
/// SSSE3 must be available on the host.
#[target_feature(enable = "ssse3")]
unsafe fn mul_into_ssse3_impl(dst: &mut [u8], tables: &FpTowerTables, src: &[u8]) {
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
    dst[offset..len].copy_from_slice(&src[offset..len]);
    scalar::mul_assign::<FanPaar16>(&mut dst[offset..len], tables.coeff);
}
