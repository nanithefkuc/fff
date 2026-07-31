//! Per-backend differential tests.
//!
//! [`crate::ops`] can only exercise the one backend the host selected. These
//! tests reach past dispatch and call every architecture kernel the CPU can
//! actually run, comparing each against the portable reference in
//! [`crate::kernel::scalar`]. On an AVX-512 GFNI machine that means AVX-512,
//! AVX2 GFNI, AVX2, SSSE3, and scalar are covered by one `cargo test`.
//!
//! Buffer lengths deliberately straddle every lane and unroll boundary. Most
//! SIMD bugs live in the tail, not the body.
#![cfg_attr(not(feature = "simd"), allow(dead_code))]
// Seeds and geometry are deliberately reduced to field widths below.
#![allow(clippy::cast_possible_truncation)]

extern crate std;

use std::vec;
use std::vec::Vec;

use crate::field::{FanPaar16, FanPaar32, FanPaar64, Gf32, Gf64, fan_paar, gf8, gf16, gf32, gf64};
use crate::kernel::scalar;
use crate::kernel::tables::{FpTowerTables, ScaleTable, TowerCoeff, TowerTables, scale_table};

/// Lengths covering: empty, sub-lane, exact lanes, lane+1, several unroll
/// tiles, and a large odd size. All even so GF(2^16) can use the same list.
const LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 96, 126, 128, 130, 254, 256, 258, 512, 1022,
];

fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// GF(2^8) coefficients worth testing: the two short-circuits, the extremes,
/// and a spread through the field.
fn gf8_coeffs() -> Vec<gf8::Elem> {
    let mut coeffs = vec![gf8::Elem(0), gf8::Elem(1), gf8::Elem(2), gf8::Elem(0xff)];
    coeffs.extend((0..=u8::MAX).step_by(23).map(gf8::Elem));
    coeffs
}

/// GF(2^16) coefficients: pure base-field, pure extension, and mixed. The
/// tower kernels have distinct code paths for the `same` and `cross` factors,
/// and a coefficient with a zero component silently masks one of them.
fn gf16_coeffs() -> Vec<gf16::Elem> {
    let mut coeffs = vec![
        gf16::Elem(0x0000),
        gf16::Elem(0x0001),
        gf16::Elem(0x0002),
        gf16::Elem(0x00ff),
        gf16::Elem(0x0100),
        gf16::Elem(0xff00),
        gf16::Elem(0xffff),
        gf16::Elem(0x0108),
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..24 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        coeffs.push(gf16::Elem((state >> 32) as u16));
    }
    coeffs
}

/// Row geometries for the multi-row kernels. Row counts straddle the
/// four-row and two-row grouping boundaries the blocked kernels use.
const ROW_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 13];
const ROW_LENS: &[usize] = &[2, 16, 32, 34, 64, 66, 128, 300];

// ---------------------------------------------------------------------------
// Generic differential drivers, shared by every architecture below.
// ---------------------------------------------------------------------------

/// Compare a GF(2^8) `mul_add` kernel against the reference at every length
/// and coefficient.
fn check_gf8_mul_add(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x51);
        for coeff in gf8_coeffs() {
            let mut got = noise(len, 0x62);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_add::<gf8::Gf8>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_assign(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable)) {
    for &len in LENGTHS {
        for coeff in gf8_coeffs() {
            let mut got = noise(len, 0x73);
            let mut want = got.clone();
            kernel(&mut got, scale_table(coeff));
            scalar::mul_assign::<gf8::Gf8>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_add_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x84);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x95);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_add::<gf16::Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_assign_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables)) {
    for &len in LENGTHS {
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xa6);
            let mut want = got.clone();
            kernel(&mut got, &TowerTables::new(coeff));
            scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf8_mul_into(name: &str, kernel: impl Fn(&mut [u8], &ScaleTable, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x136);
        for coeff in gf8_coeffs() {
            // Pre-fill the destination with noise: a fused kernel must
            // overwrite it, never accumulate into it.
            let mut got = noise(len, 0x147);
            let mut want = src.clone();
            kernel(&mut got, scale_table(coeff), &src);
            scalar::mul_assign::<gf8::Gf8>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

fn check_gf16_mul_into_tables(name: &str, kernel: impl Fn(&mut [u8], &TowerTables, &[u8])) {
    for &len in LENGTHS {
        let src = noise(len, 0x158);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0x169);
            let mut want = src.clone();
            kernel(&mut got, &TowerTables::new(coeff), &src);
            scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
            assert_eq!(got, want, "{name}: len {len}, coeff {coeff:?}");
        }
    }
}

/// Compare a scatter kernel against a per-row reference AXPY.
fn check_scatter<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, &[E], &[u8]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            let src = noise(row_len, 0xb7);
            let coeffs: Vec<E> = (0..nrows).map(&coeff_at).collect();
            let mut got = noise(row_len * nrows, 0xc8);
            let mut want = got.clone();

            kernel(&mut got, row_len, &coeffs, &src);
            for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
                reference(row, coeff, &src);
            }
            assert_eq!(got, want, "{name}: row_len {row_len}, nrows {nrows}");
        }
    }
}

/// Compare a matrix kernel against a per-term, per-row reference AXPY.
fn check_matrix<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize, usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], usize, usize, &[(&[E], &[u8])]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &row_len in ROW_LENS {
        for &nrows in ROW_COUNTS {
            for nterms in [1usize, 2, 3, 7, 8, 9, 17] {
                let sources: Vec<Vec<u8>> = (0..nterms)
                    .map(|t| noise(row_len, 0x400 + t as u64))
                    .collect();
                let coeff_sets: Vec<Vec<E>> = (0..nterms)
                    .map(|t| (0..nrows).map(|j| coeff_at(t, j)).collect())
                    .collect();
                let terms: Vec<(&[E], &[u8])> = coeff_sets
                    .iter()
                    .zip(&sources)
                    .map(|(c, s)| (c.as_slice(), s.as_slice()))
                    .collect();

                let mut got = noise(row_len * nrows, 0xd9);
                let mut want = got.clone();

                kernel(&mut got, row_len, nrows, &terms);
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
                        reference(row, coeff, src);
                    }
                }
                assert_eq!(
                    got, want,
                    "{name}: row_len {row_len}, nrows {nrows}, terms {nterms}"
                );
            }
        }
    }
}

/// Compare a gather kernel against repeated scalar AXPY calls.
fn check_gather<E: Copy, F>(
    name: &str,
    coeff_at: impl Fn(usize) -> E,
    reference: F,
    kernel: impl Fn(&mut [u8], &[E], &[&[u8]]),
) where
    F: Fn(&mut [u8], E, &[u8]),
{
    for &len in ROW_LENS {
        for nterms in [0usize, 1, 2, 7, 8, 9, 17] {
            let sources: Vec<Vec<u8>> = (0..nterms).map(|i| noise(len, 0x500 + i as u64)).collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<E> = (0..nterms).map(&coeff_at).collect();
            let mut got = noise(len, 0xe1);
            let mut want = got.clone();
            kernel(&mut got, &coeffs, &srcs);
            for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                reference(&mut want, coeff, src);
            }
            assert_eq!(got, want, "{name}: len {len}, terms {nterms}");
        }
    }
}

fn check_gf8_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0xf2);
        let b = noise(len, 0x103);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<gf8::Gf8>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

fn check_gf16_elementwise(name: &str, kernel: impl Fn(&mut [u8], &[u8], &[u8])) {
    for &len in LENGTHS {
        let a = noise(len, 0x114);
        let b = noise(len, 0x125);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        kernel(&mut got, &a, &b);
        scalar::mul_elementwise::<gf16::Gf16>(&mut want, &a, &b);
        assert_eq!(got, want, "{name}: len {len}");
    }
}

/// Compare a tower-field `mul_add` kernel against the portable reference at
/// every length and coefficient.
fn check_tower_mul_add<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let src = noise(len, 0x90 ^ (ci as u64).wrapping_mul(7));
            let mut want = noise(len, 0x80 ^ (ci as u64).wrapping_mul(13));
            let mut got = want.clone();
            reference(&mut want, coeff, &src);
            kernel(&mut got, coeff, &src);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

/// Compare a tower-field `mul_assign` kernel against the portable reference.
fn check_tower_mul_assign<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E),
    kernel: impl Fn(&mut [u8], E),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let mut want = noise(len, 0x80 ^ (ci as u64).wrapping_mul(13));
            let mut got = want.clone();
            reference(&mut want, coeff);
            kernel(&mut got, coeff);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

/// Compare a tower-field `mul_into` kernel against copy-then-scale.
fn check_tower_mul_into<E: Copy + core::fmt::Debug>(
    name: &str,
    lengths: &[usize],
    coeffs: &[E],
    reference: impl Fn(&mut [u8], E, &[u8]),
    kernel: impl Fn(&mut [u8], E, &[u8]),
) {
    for &len in lengths {
        for (ci, &coeff) in coeffs.iter().enumerate() {
            let src = noise(len, 0x90 ^ (ci as u64).wrapping_mul(7));
            let mut want = vec![0; len];
            let mut got = vec![0; len];
            reference(&mut want, coeff, &src);
            kernel(&mut got, coeff, &src);
            assert_eq!(got, want, "{name}: len {len} coeff {coeff:?}");
        }
    }
}

fn gf8_coeff_at(j: usize) -> gf8::Elem {
    // Includes 0 and 1 as j sweeps, which is what we want: the blocked
    // kernels must handle degenerate coefficients per row, not per call.
    gf8::Elem((j as u8).wrapping_mul(29))
}

fn gf8_coeff_at2(t: usize, j: usize) -> gf8::Elem {
    gf8::Elem(((t * 31 + j * 29) % 256) as u8)
}

fn gf16_coeff_at(j: usize) -> gf16::Elem {
    gf16::Elem((j as u16).wrapping_mul(7411))
}

fn gf16_coeff_at2(t: usize, j: usize) -> gf16::Elem {
    gf16::Elem(((t * 7919 + j * 613) % 65536) as u16)
}

fn gf8_reference(dst: &mut [u8], coeff: gf8::Elem, src: &[u8]) {
    scalar::mul_add::<gf8::Gf8>(dst, coeff, src);
}

fn gf16_reference(dst: &mut [u8], coeff: gf16::Elem, src: &[u8]) {
    scalar::mul_add::<gf16::Gf16>(dst, coeff, src);
}

/// Fan–Paar GF(2^16) coefficients: short-circuits, extremes, the tower
/// generator and its components, and a deterministic spread.
fn fp16_coeffs() -> Vec<fan_paar::fp16::Elem> {
    use fan_paar::{fp8, fp16};
    let mut v = vec![
        fp16::Elem::ZERO,
        fp16::Elem::ONE,
        fp16::Elem(u16::MAX),
        // Pure base-field and pure extension.
        fp16::Elem::from_components(fp8::Elem::ONE, fp8::Elem::ZERO),
        fp16::Elem::from_components(fp8::Elem::ZERO, fp8::Elem::ONE),
        // The tower generator and its subfield tower generator.
        fp16::GENERATOR,
        fp16::ALPHA,
    ];
    let mut s = 0xf491u16;
    for _ in 0..16 {
        s = s.wrapping_mul(2057).wrapping_add(13849);
        v.push(fp16::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^16) lengths: multiples of 2 straddling 16/32-byte lanes.
const FP16_LENGTHS: &[usize] = &[
    0, 2, 4, 8, 14, 16, 18, 30, 32, 34, 62, 64, 66, 126, 128, 130, 254, 256, 510, 512, 1022,
];

fn fp16_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar16>(dst, coeff, src);
}
fn fp16_assign_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem) {
    scalar::mul_assign::<FanPaar16>(dst, coeff);
}
fn fp16_into_reference(dst: &mut [u8], coeff: fan_paar::fp16::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar16>(dst, coeff);
}

/// Fan–Paar GF(2^32) coefficients, the same shape as the polynomial towers.
fn fp32_coeffs() -> Vec<fan_paar::fp32::Elem> {
    use fan_paar::{fp16, fp32};
    let mut v = vec![
        fp32::Elem::ZERO,
        fp32::Elem::ONE,
        fp32::Elem(u32::MAX),
        fp32::Elem::from_components(fp16::Elem::ONE, fp16::Elem::ZERO),
        fp32::Elem::from_components(fp16::Elem::ZERO, fp16::Elem::ONE),
        fp32::ALPHA,
        fp32::Elem::from_components(fp16::Elem::ZERO, fp16::ALPHA),
        fp32::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(fp32::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^64) coefficients.
fn fp64_coeffs() -> Vec<fan_paar::fp64::Elem> {
    use fan_paar::{fp32, fp64};
    let mut v = vec![
        fp64::Elem::ZERO,
        fp64::Elem::ONE,
        fp64::Elem(u64::MAX),
        fp64::Elem::from_components(fp32::Elem::ONE, fp32::Elem::ZERO),
        fp64::ALPHA,
        fp64::Elem::from_components(fp32::Elem::ZERO, fp32::ALPHA),
        fp64::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(fp64::Elem(s));
    }
    v
}

/// Fan–Paar GF(2^32) lengths: multiples of 4 straddling 32-byte lanes.
const FP32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// Fan–Paar GF(2^64) lengths: multiples of 8.
const FP64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

fn fp32_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar32>(dst, coeff, src);
}
fn fp32_assign_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem) {
    scalar::mul_assign::<FanPaar32>(dst, coeff);
}
fn fp32_into_reference(dst: &mut [u8], coeff: fan_paar::fp32::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar32>(dst, coeff);
}
fn fp64_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    scalar::mul_add::<FanPaar64>(dst, coeff, src);
}
fn fp64_assign_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem) {
    scalar::mul_assign::<FanPaar64>(dst, coeff);
}
fn fp64_into_reference(dst: &mut [u8], coeff: fan_paar::fp64::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<FanPaar64>(dst, coeff);
}

/// GF(2^32) coefficients: the short-circuits, the extremes, pure tower
/// components, the tower constant, the generator, and a deterministic spread.
fn gf32_coeffs() -> Vec<gf32::Elem> {
    let mut v = vec![
        gf32::Elem::ZERO,
        gf32::Elem::ONE,
        gf32::Elem(u32::MAX),
        gf32::Elem(0x0000_ffff),
        gf32::Elem(0xffff_0000),
        // Pure base field and pure extension.
        gf32::Elem::from_components(gf16::Elem::ONE, gf16::Elem::ZERO),
        gf32::Elem::from_components(gf16::Elem::ZERO, gf16::Elem::ONE),
        // The tower constant in each component.
        gf32::Elem::from_components(gf32::DELTA, gf16::Elem::ZERO),
        gf32::Elem::from_components(gf16::Elem::ZERO, gf32::DELTA),
        gf32::GENERATOR,
    ];
    let mut s = 0x243f_6a88u32;
    for _ in 0..16 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push(gf32::Elem(s));
    }
    v
}

/// GF(2^64) coefficients, the same shape one level up.
fn gf64_coeffs() -> Vec<gf64::Elem> {
    let mut v = vec![
        gf64::Elem::ZERO,
        gf64::Elem::ONE,
        gf64::Elem(u64::MAX),
        gf64::Elem(0x0000_0000_ffff_ffff),
        gf64::Elem(0xffff_ffff_0000_0000),
        gf64::Elem::from_components(gf32::Elem::ONE, gf32::Elem::ZERO),
        gf64::Elem::from_components(gf32::Elem::ZERO, gf32::Elem::ONE),
        gf64::Elem::from_components(gf64::DELTA, gf32::Elem::ZERO),
        gf64::Elem::from_components(gf32::Elem::ZERO, gf64::DELTA),
        gf64::GENERATOR,
    ];
    let mut s = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..16 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(gf64::Elem(s));
    }
    v
}

/// GF(2^32) lengths: multiples of 4 straddling every 32-byte lane boundary.
const GF32_LENGTHS: &[usize] = &[
    0, 4, 8, 16, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1020, 1024,
];
/// GF(2^64) lengths: multiples of 8.
const GF64_LENGTHS: &[usize] = &[
    0, 8, 16, 24, 32, 40, 56, 64, 72, 120, 128, 136, 248, 256, 264, 504, 512, 1016, 1024,
];

fn gf32_reference(dst: &mut [u8], coeff: gf32::Elem, src: &[u8]) {
    scalar::mul_add::<Gf32>(dst, coeff, src);
}
fn gf32_assign_reference(dst: &mut [u8], coeff: gf32::Elem) {
    scalar::mul_assign::<Gf32>(dst, coeff);
}
fn gf32_into_reference(dst: &mut [u8], coeff: gf32::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf32>(dst, coeff);
}
fn gf64_reference(dst: &mut [u8], coeff: gf64::Elem, src: &[u8]) {
    scalar::mul_add::<Gf64>(dst, coeff, src);
}
fn gf64_assign_reference(dst: &mut [u8], coeff: gf64::Elem) {
    scalar::mul_assign::<Gf64>(dst, coeff);
}
fn gf64_into_reference(dst: &mut [u8], coeff: gf64::Elem, src: &[u8]) {
    dst.copy_from_slice(src);
    scalar::mul_assign::<Gf64>(dst, coeff);
}

// ---------------------------------------------------------------------------
// Backend-independent
// ---------------------------------------------------------------------------

#[test]
fn backend_overrides_reject_foreign_architectures() {
    use super::Backend;

    assert!(Backend::Scalar.is_for_current_arch());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        assert!(Backend::Avx512.is_for_current_arch());
        assert!(Backend::Gfni.is_for_current_arch());
        assert!(Backend::Avx2.is_for_current_arch());
        assert!(Backend::Ssse3.is_for_current_arch());
        assert!(!Backend::Neon.is_for_current_arch());
        assert!(!Backend::Simd128.is_for_current_arch());
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(Backend::Neon.is_for_current_arch());
        assert!(!Backend::Avx512.is_for_current_arch());
        assert!(!Backend::Simd128.is_for_current_arch());
    }
    #[cfg(target_arch = "wasm32")]
    {
        assert!(Backend::Simd128.is_for_current_arch());
        assert!(!Backend::Avx512.is_for_current_arch());
        assert!(!Backend::Neon.is_for_current_arch());
    }
}

#[test]
fn scalar_nibble_paths_match_the_generic_reference() {
    check_gf8_mul_add("gf8 nibble", |dst, table, src| {
        super::gf8::mul_add_nibble(dst, table, src);
    });
    check_gf8_mul_assign("gf8 nibble", super::gf8::mul_assign_nibble);
    for &len in LENGTHS {
        let src = noise(len, 0xea);
        for coeff in gf16_coeffs() {
            let mut got = noise(len, 0xfb);
            let mut want = got.clone();
            super::gf16::mul_add_scalar(&mut got, coeff, &src);
            scalar::mul_add::<gf16::Gf16>(&mut want, coeff, &src);
            assert_eq!(got, want, "gf16 scalar: len {len}, coeff {coeff:?}");
        }
    }
}

#[test]
fn tower_coefficient_derivation_is_self_consistent() {
    for coeff in gf16_coeffs() {
        let compact = TowerCoeff::new(coeff);
        let tables = TowerTables::new(coeff);
        for (factor, table) in compact.factors().iter().zip(&tables.factors) {
            assert_eq!(*factor, table.coeff, "table/factor mismatch for {coeff:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// x86
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    use super::*;
    use crate::kernel::x86;

    #[test]
    fn avx512_kernels_match_reference() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("gfni"))
        {
            eprintln!("skipping: no AVX-512F+AVX-512BW+GFNI on this host");
            return;
        }

        check_gf8_mul_add("gf8 avx512", |dst, table, src| {
            x86::avx512::gf8_mul_add(dst, table.coeff, src);
        });
        check_gf8_mul_assign("gf8 avx512", |dst, table| {
            x86::avx512::gf8_mul_assign(dst, table.coeff);
        });
        check_gf16_mul_add_tables("gf16 avx512", |dst, tables, src| {
            x86::avx512::gf16_mul_add(dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf16 avx512", |dst, tables| {
            x86::avx512::gf16_mul_assign(dst, TowerCoeff::new(tables.coeff));
        });
        check_scatter(
            "gf8 avx512 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::avx512::gf8_scatter,
        );
        check_scatter(
            "gf16 avx512 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::avx512::gf16_scatter,
        );
        check_gather(
            "gf8 avx512 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::avx512::gf8_gather,
        );
        check_gather(
            "gf16 avx512 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::avx512::gf16_gather,
        );
        check_matrix(
            "gf8 avx512 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::avx512::gf8_matrix,
        );
        check_matrix(
            "gf16 avx512 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::avx512::gf16_matrix,
        );
        check_gf8_elementwise("gf8 avx512 elementwise", x86::avx512::gf8_elementwise);
        check_gf16_elementwise("gf16 avx512 elementwise", x86::avx512::gf16_elementwise);
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut simd = want.clone();
            scalar::xor(&mut want, &src);
            x86::avx512::xor(&mut simd, &src);
            assert_eq!(simd, want, "avx512 xor: len {len}");
        }
    }

    #[test]
    fn gfni_kernels_match_reference() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni")) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }

        check_gf8_mul_add("gf8 gfni", |dst, table, src| {
            x86::gf8::mul_add_gfni(dst, table.coeff, src);
        });
        check_gf8_mul_assign("gf8 gfni", |dst, table| {
            x86::gf8::mul_assign_gfni(dst, table.coeff);
        });
        check_gf16_mul_add_tables("gf16 gfni", |dst, tables, src| {
            x86::gf16::mul_add_gfni(dst, TowerCoeff::new(tables.coeff), src);
        });
        check_gf16_mul_assign_tables("gf16 gfni", |dst, tables| {
            x86::gf16::mul_assign_gfni(dst, TowerCoeff::new(tables.coeff));
        });

        check_scatter(
            "gf8 gfni scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_gfni,
        );
        check_scatter(
            "gf16 gfni scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_gfni,
        );
        check_matrix(
            "gf8 gfni matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_gfni,
        );
        check_matrix(
            "gf16 gfni matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_gfni,
        );
        check_gather(
            "gf8 gfni gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_gfni,
        );
        check_gather(
            "gf16 gfni gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_gfni,
        );
        check_gf8_elementwise("gf8 gfni elementwise", x86::gf8::elementwise_gfni);
        check_gf16_elementwise("gf16 gfni elementwise", x86::gf16::elementwise_gfni);
        // Level-2/3 tower kernels: the period-2 lane multiply one and two
        // levels up from the GF(2^16) kernel.
        check_tower_gfni_kernels();
    }

    /// Differential-check the GFNI GF(2^32) and GF(2^64) kernels — the level-2
    /// and level-3 tower multiplies — against the portable scalar oracle.
    fn check_tower_gfni_kernels() {
        check_tower_mul_add(
            "gf32 gfni mul_add",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_reference,
            |dst, coeff, src| {
                x86::gf32::mul_add_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf32 gfni mul_assign",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_assign_reference,
            |dst, coeff| {
                x86::gf32::mul_assign_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf32 gfni mul_into",
            GF32_LENGTHS,
            &gf32_coeffs(),
            gf32_into_reference,
            |dst, coeff, src| {
                x86::gf32::mul_into_gfni(dst, coeff, x86::gf32::gf32_tiles(coeff), src);
            },
        );
        // Level-3: the same identity over GF(2^32) lanes.
        check_tower_mul_add(
            "gf64 gfni mul_add",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_reference,
            |dst, coeff, src| {
                x86::gf64::mul_add_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
        check_tower_mul_assign(
            "gf64 gfni mul_assign",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_assign_reference,
            |dst, coeff| {
                x86::gf64::mul_assign_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff));
            },
        );
        check_tower_mul_into(
            "gf64 gfni mul_into",
            GF64_LENGTHS,
            &gf64_coeffs(),
            gf64_into_reference,
            |dst, coeff, src| {
                x86::gf64::mul_into_gfni(dst, coeff, x86::gf64::gf64_tiles(coeff), src);
            },
        );
    }

    #[test]
    fn avx2_kernels_match_reference() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no AVX2 on this host");
            return;
        }
        check_gf8_mul_add("gf8 avx2", x86::gf8::mul_add_avx2);
        check_gf8_mul_assign("gf8 avx2", x86::gf8::mul_assign_avx2);
        check_gf16_mul_add_tables("gf16 avx2", x86::gf16::mul_add_avx2);
        check_gf16_mul_assign_tables("gf16 avx2", x86::gf16::mul_assign_avx2);
        check_gf8_mul_into("gf8 avx2 mul_into", x86::gf8::mul_into_avx2);
        check_gf16_mul_into_tables("gf16 avx2 mul_into", x86::gf16::mul_into_avx2);
        check_scatter(
            "gf8 avx2 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_avx2,
        );
        check_scatter(
            "gf16 avx2 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_avx2,
        );
        check_gather(
            "gf8 avx2 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_avx2,
        );
        check_gather(
            "gf16 avx2 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_avx2,
        );
        check_matrix(
            "gf8 avx2 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_avx2,
        );
        check_matrix(
            "gf16 avx2 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_avx2,
        );
        check_gf8_elementwise("gf8 avx2 elementwise", x86::gf8::elementwise_avx2);
        check_gf16_elementwise("gf16 avx2 elementwise", x86::gf16::elementwise_avx2);
        // Fan–Paar tower (GF(2^16)/32/64): the fp8 nibble tower and its
        // period-2 lane-mul extensions.
        check_fan_paar_avx2_kernels();
    }

    /// Differential-check the Fan–Paar GF(2^16)/32/64 AVX2 kernels — the fp8
    /// nibble tower and its period-2 lane-mul extensions — against the scalar
    /// oracle.
    fn check_fan_paar_avx2_kernels() {
        check_tower_mul_add(
            "fp16 avx2 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_avx2(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 avx2 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_avx2(dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 avx2 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_avx2(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_add(
            "fp32 avx2 mul_add",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_fp32_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp32 avx2 mul_assign",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_fp32_avx2(dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp32 avx2 mul_into",
            FP32_LENGTHS,
            &fp32_coeffs(),
            fp32_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_fp32_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_add(
            "fp64 avx2 mul_add",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_fp64_avx2(dst, coeff, src);
            },
        );
        check_tower_mul_assign(
            "fp64 avx2 mul_assign",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_fp64_avx2(dst, coeff);
            },
        );
        check_tower_mul_into(
            "fp64 avx2 mul_into",
            FP64_LENGTHS,
            &fp64_coeffs(),
            fp64_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_fp64_avx2(dst, coeff, src);
            },
        );
    }

    #[test]
    fn ssse3_kernels_match_reference() {
        if !std::is_x86_feature_detected!("ssse3") {
            eprintln!("skipping: no SSSE3 on this host");
            return;
        }
        check_gf8_mul_add("gf8 ssse3", x86::gf8::mul_add_ssse3);
        check_gf8_mul_assign("gf8 ssse3", x86::gf8::mul_assign_ssse3);
        check_gf16_mul_add_tables("gf16 ssse3", x86::gf16::mul_add_ssse3);
        check_gf16_mul_assign_tables("gf16 ssse3", x86::gf16::mul_assign_ssse3);
        check_gf8_mul_into("gf8 ssse3 mul_into", x86::gf8::mul_into_ssse3);
        check_gf16_mul_into_tables("gf16 ssse3 mul_into", x86::gf16::mul_into_ssse3);
        check_scatter(
            "gf8 ssse3 scatter",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::scatter_ssse3,
        );
        check_scatter(
            "gf16 ssse3 scatter",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::scatter_ssse3,
        );
        check_gather(
            "gf8 ssse3 gather",
            gf8_coeff_at,
            gf8_reference,
            x86::gf8::gather_ssse3,
        );
        check_gather(
            "gf16 ssse3 gather",
            gf16_coeff_at,
            gf16_reference,
            x86::gf16::gather_ssse3,
        );
        check_matrix(
            "gf8 ssse3 matrix",
            gf8_coeff_at2,
            gf8_reference,
            x86::gf8::matrix_ssse3,
        );
        check_matrix(
            "gf16 ssse3 matrix",
            gf16_coeff_at2,
            gf16_reference,
            x86::gf16::matrix_ssse3,
        );
        check_gf8_elementwise("gf8 ssse3 elementwise", x86::gf8::elementwise_ssse3);
        check_gf16_elementwise("gf16 ssse3 elementwise", x86::gf16::elementwise_ssse3);
        check_tower_mul_add(
            "fp16 ssse3 mul_add",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_add_ssse3(dst, &FpTowerTables::new(coeff), src);
            },
        );
        check_tower_mul_assign(
            "fp16 ssse3 mul_assign",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_assign_reference,
            |dst, coeff| {
                x86::fan_paar::mul_assign_ssse3(dst, &FpTowerTables::new(coeff));
            },
        );
        check_tower_mul_into(
            "fp16 ssse3 mul_into",
            FP16_LENGTHS,
            &fp16_coeffs(),
            fp16_into_reference,
            |dst, coeff, src| {
                x86::fan_paar::mul_into_ssse3(dst, &FpTowerTables::new(coeff), src);
            },
        );
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut avx2 = want.clone();
            let mut sse2 = want.clone();
            scalar::xor(&mut want, &src);
            if std::is_x86_feature_detected!("avx2") {
                x86::xor_avx2(&mut avx2, &src);
                assert_eq!(avx2, want, "avx2 xor: len {len}");
            }
            x86::xor_sse2(&mut sse2, &src);
            assert_eq!(sse2, want, "sse2 xor: len {len}");
        }
    }

    /// `mul_into` over a buffer past the non-temporal store threshold.
    ///
    /// The shared `LENGTHS` are all far below it, so nothing else reaches the
    /// `vmovntdq` body. Destination offsets 0 and 1 cover both sides of the
    /// alignment peel — offset 1 leaves the GF(2^16) peel an odd number of
    /// bytes, which must fall back to ordinary stores rather than split a
    /// buffer mid-element.
    #[test]
    fn non_temporal_mul_into_matches_reference() {
        const NT_LEN: usize = (2 << 20) + 130;
        // The non-temporal split is a store-side choice, independent of the
        // coefficient, so a zero, a one and a mixed value are enough.
        const GF8_COEFFS: [gf8::Elem; 3] = [gf8::Elem(0), gf8::Elem(1), gf8::Elem(0x53)];
        const GF16_COEFFS: [gf16::Elem; 3] = [gf16::Elem(0), gf16::Elem(1), gf16::Elem(0x53a7)];

        let source = noise(NT_LEN + 2, 0x1a7);
        for offset in [0, 1] {
            let src = &source[offset..offset + NT_LEN];
            let mut got = vec![0u8; NT_LEN + 1];

            for coeff in GF8_COEFFS {
                let table = scale_table(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf8::Gf8>(&mut want, coeff);
                if std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("avx2") {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_gfni(got, coeff, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if std::is_x86_feature_detected!("avx2") {
                    let got = &mut got[offset..offset + NT_LEN];
                    x86::gf8::mul_into_avx2(got, table, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf8 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + NT_LEN];
                x86::gf8::mul_into_ssse3(got, table, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf8 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }

            // GF(2^16) needs an even length and an even peel.
            let src = &src[..NT_LEN - 1];
            for coeff in GF16_COEFFS {
                let tables = TowerTables::new(coeff);
                let mut want = src.to_vec();
                scalar::mul_assign::<gf16::Gf16>(&mut want, coeff);
                if std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("avx2") {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_gfni(got, TowerCoeff::new(coeff), src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 gfni nt mul_into: coeff {coeff:?}"
                    );
                }
                if std::is_x86_feature_detected!("avx2") {
                    let got = &mut got[offset..offset + src.len()];
                    x86::gf16::mul_into_avx2(got, &tables, src);
                    assert_eq!(
                        got,
                        want.as_slice(),
                        "gf16 avx2 nt mul_into: coeff {coeff:?}"
                    );
                }
                let got = &mut got[offset..offset + src.len()];
                x86::gf16::mul_into_ssse3(got, &tables, src);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "gf16 ssse3 nt mul_into: coeff {coeff:?}"
                );
            }
        }
    }

    /// Multi-row scatter over rows long enough to take the alignment peel.
    ///
    /// `ROW_LENS` tops out at 300 bytes, below the peel's row-length floor, so
    /// no other test reaches the peeled path. The destination is offset by
    /// every residue that matters: 0 and 16 pick different peel lengths, and 1
    /// makes a 32-byte boundary unreachable in whole GF(2^16) elements, which
    /// must skip the peel rather than split an element.
    #[test]
    fn aligned_scatter_matches_reference() {
        const ROW_LEN: usize = 4096;

        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni")) {
            eprintln!("skipping: no AVX2+GFNI on this host");
            return;
        }
        for nrows in [1usize, 4, 6, 9] {
            for skew in [0usize, 1, 16] {
                let src = noise(ROW_LEN, 0x1b8);
                let mut backing = noise(ROW_LEN * nrows + skew, 0x1c9);
                let rows = &mut backing[skew..];

                let coeffs8: Vec<_> = (0..nrows).map(gf8_coeff_at).collect();
                let mut want = rows.to_vec();
                x86::gf8::scatter_gfni(rows, ROW_LEN, &coeffs8, &src);
                for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs8) {
                    gf8_reference(row, coeff, &src);
                }
                assert_eq!(rows, want.as_slice(), "gf8 peeled scatter: {nrows}/{skew}");

                let coeffs16: Vec<_> = (0..nrows).map(gf16_coeff_at).collect();
                let mut want = rows.to_vec();
                x86::gf16::scatter_gfni(rows, ROW_LEN, &coeffs16, &src);
                for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs16) {
                    gf16_reference(row, coeff, &src);
                }
                assert_eq!(rows, want.as_slice(), "gf16 peeled scatter: {nrows}/{skew}");

                for (name, kernel) in [
                    (
                        "avx2",
                        x86::gf16::scatter_avx2::<gf16::Elem>
                            as fn(&mut [u8], usize, &[gf16::Elem], &[u8]),
                    ),
                    ("ssse3", x86::gf16::scatter_ssse3::<gf16::Elem>),
                ] {
                    let mut want = rows.to_vec();
                    kernel(rows, ROW_LEN, &coeffs16, &src);
                    for (row, &coeff) in want.chunks_exact_mut(ROW_LEN).zip(&coeffs16) {
                        gf16_reference(row, coeff, &src);
                    }
                    assert_eq!(
                        rows,
                        want.as_slice(),
                        "gf16 {name} peeled scatter: {nrows}/{skew}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AArch64
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64 {
    use super::*;
    use crate::kernel::aarch64;

    #[test]
    fn neon_kernels_match_reference() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipping: no NEON on this host");
            return;
        }
        check_gf8_mul_add("gf8 neon", aarch64::gf8::mul_add_neon);
        check_gf8_mul_assign("gf8 neon", aarch64::gf8::mul_assign_neon);
        check_gf16_mul_add_tables("gf16 neon", aarch64::gf16::mul_add_neon);
        check_gf16_mul_assign_tables("gf16 neon", aarch64::gf16::mul_assign_neon);
        check_gf8_mul_into("gf8 neon mul_into", aarch64::gf8::mul_into_neon);
        check_gf16_mul_into_tables("gf16 neon mul_into", aarch64::gf16::mul_into_neon);

        check_scatter(
            "gf8 neon scatter",
            gf8_coeff_at,
            gf8_reference,
            aarch64::gf8::scatter_neon,
        );
        check_scatter(
            "gf16 neon scatter",
            gf16_coeff_at,
            gf16_reference,
            aarch64::gf16::scatter_neon,
        );
        check_matrix(
            "gf8 neon matrix",
            gf8_coeff_at2,
            gf8_reference,
            aarch64::gf8::matrix_neon,
        );
        check_matrix(
            "gf16 neon matrix",
            gf16_coeff_at2,
            gf16_reference,
            aarch64::gf16::matrix_neon,
        );
        check_gather(
            "gf8 neon gather",
            gf8_coeff_at,
            gf8_reference,
            aarch64::gf8::gather_neon,
        );
        check_gather(
            "gf16 neon gather",
            gf16_coeff_at,
            gf16_reference,
            aarch64::gf16::gather_neon,
        );
        check_gf8_elementwise("gf8 neon elementwise", aarch64::gf8::elementwise_neon);
        check_gf16_elementwise("gf16 neon elementwise", aarch64::gf16::elementwise_neon);
    }

    #[test]
    fn pmull_kernels_match_reference() {
        if !std::arch::is_aarch64_feature_detected!("pmull") {
            eprintln!("skipping: no AArch64 PMULL extension on this host");
            return;
        }
        check_gf8_elementwise("gf8 pmull elementwise", aarch64::gf8::elementwise_pmull);
        // The tower elementwise and every fixed-coefficient PMULL kernel were
        // measured against the nibble/bit-serial paths and lost; GF(2^8)
        // elementwise is the shape that won and the only one dispatch selects.
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut neon = want.clone();
            scalar::xor(&mut want, &src);
            aarch64::xor_neon(&mut neon, &src);
            assert_eq!(neon, want, "neon xor: len {len}");
        }
    }
}

// ---------------------------------------------------------------------------
// WebAssembly
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32 {
    use super::*;
    use crate::kernel::wasm32;

    #[test]
    fn simd128_kernels_match_reference() {
        check_gf8_mul_add("gf8 simd128", wasm32::gf8::mul_add_simd128);
        check_gf8_mul_assign("gf8 simd128", wasm32::gf8::mul_assign_simd128);
        check_gf16_mul_add_tables("gf16 simd128", wasm32::gf16::mul_add_simd128);
        check_gf16_mul_assign_tables("gf16 simd128", wasm32::gf16::mul_assign_simd128);
        check_gf8_mul_into("gf8 simd128 mul_into", wasm32::gf8::mul_into_simd128);
        check_gf16_mul_into_tables("gf16 simd128 mul_into", wasm32::gf16::mul_into_simd128);
        check_scatter(
            "gf8 simd128 scatter",
            gf8_coeff_at,
            gf8_reference,
            wasm32::gf8::scatter_simd128,
        );
        check_scatter(
            "gf16 simd128 scatter",
            gf16_coeff_at,
            gf16_reference,
            wasm32::gf16::scatter_simd128,
        );
        check_gather(
            "gf8 simd128 gather",
            gf8_coeff_at,
            gf8_reference,
            wasm32::gf8::gather_simd128,
        );
        check_gather(
            "gf16 simd128 gather",
            gf16_coeff_at,
            gf16_reference,
            wasm32::gf16::gather_simd128,
        );
        check_matrix(
            "gf8 simd128 matrix",
            gf8_coeff_at2,
            gf8_reference,
            wasm32::gf8::matrix_simd128,
        );
        check_matrix(
            "gf16 simd128 matrix",
            gf16_coeff_at2,
            gf16_reference,
            wasm32::gf16::matrix_simd128,
        );
        check_gf8_elementwise("gf8 simd128 elementwise", wasm32::gf8::elementwise_simd128);
        check_gf16_elementwise(
            "gf16 simd128 elementwise",
            wasm32::gf16::elementwise_simd128,
        );
    }

    #[test]
    fn vector_xor_matches_scalar_xor() {
        for &len in LENGTHS {
            let src = noise(len, 0x1c);
            let mut want = noise(len, 0x2d);
            let mut simd = want.clone();
            scalar::xor(&mut want, &src);
            wasm32::xor_simd128(&mut simd, &src);
            assert_eq!(simd, want, "simd128 xor: len {len}");
        }
    }
}
