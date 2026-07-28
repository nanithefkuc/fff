//! Scalar field axioms and cross-backend algebra checks.
//!
//! The independent oracle for GF(2^8) is shift-and-XOR; every tower oracle
//! uses schoolbook expansion over its already-tested base field. Nothing here
//! uses the Karatsuba form under test, so reduction bugs remain visible rather
//! than self-consistent.

use fff::field::Field;
use fff::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8, Gf16, Gf32, Gf64, fan_paar, gf8, gf16, gf32,
    gf64,
};

/// Every nonzero element, plus zero, in ascending order.
fn all_gf8() -> impl Iterator<Item = gf8::Elem> {
    (0..=u8::MAX).map(gf8::Elem)
}

/// A spread of GF(2^16) elements: boundaries, both component planes, and a
/// deterministic pseudo-random spray. Exhaustive would be 4 billion pairs.
fn sample_gf16() -> Vec<gf16::Elem> {
    let mut values = vec![
        gf16::Elem(0),
        gf16::Elem(1),
        gf16::Elem(0x0100),
        gf16::Elem(0x00ff),
        gf16::Elem(0xff00),
        gf16::Elem(0xffff),
        gf16::GENERATOR,
    ];
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        values.push(gf16::Elem((state >> 32) as u16));
    }
    values
}

// ---------------------------------------------------------------------------
// GF(2^8)
// ---------------------------------------------------------------------------

#[test]
fn gf8_table_multiply_matches_shift_and_xor() {
    for a in all_gf8() {
        for b in all_gf8() {
            assert_eq!(
                a.mul(b),
                a.mul_xtime(b),
                "table and xtime disagree on {a:?} * {b:?}"
            );
        }
    }
}

#[test]
fn gf8_inverse_matches_fermat_and_round_trips() {
    assert_eq!(gf8::Elem::ZERO.inv(), gf8::Elem::ZERO, "inv(0) must be 0");
    for a in all_gf8().skip(1) {
        assert_eq!(a.inv(), a.inv_xtime(), "inverse backends disagree on {a:?}");
        assert_eq!(a.mul(a.inv()), gf8::Elem::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), gf8::Elem::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf8_generator_has_full_order() {
    // 0x03 must generate all 255 nonzero elements and no fewer.
    let mut seen = [false; 256];
    let mut value = gf8::Elem::ONE;
    for step in 0..255u32 {
        assert!(!seen[value.0 as usize], "generator repeats at step {step}");
        seen[value.0 as usize] = true;
        value = value.mul(Gf8::GENERATOR);
    }
    assert_eq!(value, gf8::Elem::ONE, "generator order is not 255");
    assert!(
        seen.iter().skip(1).all(|&hit| hit),
        "orbit misses an element"
    );
}

#[test]
fn gf8_field_axioms() {
    let sample: Vec<_> = all_gf8().step_by(7).collect();
    for &a in &sample {
        assert_eq!(a.add(gf8::Elem::ZERO), a);
        assert_eq!(a.mul(gf8::Elem::ONE), a);
        assert_eq!(a.mul(gf8::Elem::ZERO), gf8::Elem::ZERO);
        assert_eq!(a.add(a), gf8::Elem::ZERO, "characteristic two");
        assert_eq!(a.sub(a), gf8::Elem::ZERO);
        for &b in &sample {
            assert_eq!(a.add(b), b.add(a), "addition commutes");
            assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
            for &c in &sample {
                assert_eq!(a.add(b).add(c), a.add(b.add(c)), "addition associates");
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "multiplication associates"
                );
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "multiplication distributes"
                );
            }
        }
    }
}

#[test]
fn gf8_pow_matches_repeated_multiplication() {
    for a in all_gf8().step_by(11) {
        let mut expected = gf8::Elem::ONE;
        for exponent in 0..20u64 {
            assert_eq!(a.pow(exponent), expected, "{a:?}^{exponent}");
            expected = expected.mul(a);
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^16)
// ---------------------------------------------------------------------------

/// Schoolbook `(a + b*u)(c + d*u)` reduced by `u^2 = u + DELTA`, using the
/// GF(2^8) shift-and-XOR multiply. Independent of the Karatsuba form under
/// test and of the log tables.
fn gf16_mul_oracle(x: gf16::Elem, y: gf16::Elem) -> gf16::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul_xtime(c);
    let ad = a.mul_xtime(d);
    let bc = b.mul_xtime(c);
    let bd = b.mul_xtime(d);
    // ac + (ad + bc)u + bd*u^2, and u^2 = u + DELTA.
    let constant = ac.add(gf16::DELTA.mul_xtime(bd));
    let extension = ad.add(bc).add(bd);
    gf16::Elem::from_components(constant, extension)
}

#[test]
fn gf16_karatsuba_matches_schoolbook() {
    let sample = sample_gf16();
    for &a in &sample {
        for &b in &sample {
            assert_eq!(a.mul(b), gf16_mul_oracle(a, b), "{a:?} * {b:?}");
        }
    }
}

#[test]
fn gf16_square_matches_multiply() {
    for a in sample_gf16() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
    }
}

#[test]
fn gf16_inverse_round_trips() {
    assert_eq!(gf16::Elem::ZERO.inv(), gf16::Elem::ZERO, "inv(0) must be 0");
    for a in sample_gf16() {
        if a == gf16::Elem::ZERO {
            continue;
        }
        assert_eq!(a.mul(a.inv()), gf16::Elem::ONE, "{a:?} * inv({a:?}) != 1");
        assert_eq!(a.div(a), gf16::Elem::ONE, "{a:?} / {a:?} != 1");
    }
}

#[test]
fn gf16_generator_has_full_order() {
    // Order must be exactly 65535: g^65535 == 1 and g^(65535/p) != 1 for each
    // prime factor p of 65535 = 3 * 5 * 17 * 257.
    let g = Gf16::GENERATOR;
    assert_eq!(g.pow(65_535), gf16::Elem::ONE, "g^65535 != 1");
    for factor in [3u64, 5, 17, 257] {
        assert_ne!(
            g.pow(65_535 / factor),
            gf16::Elem::ONE,
            "generator order divides 65535/{factor}"
        );
    }
}

#[test]
fn gf16_field_axioms() {
    let sample: Vec<_> = sample_gf16().into_iter().step_by(37).collect();
    for &a in &sample {
        assert_eq!(a.add(gf16::Elem::ZERO), a);
        assert_eq!(a.mul(gf16::Elem::ONE), a);
        assert_eq!(a.mul(gf16::Elem::ZERO), gf16::Elem::ZERO);
        assert_eq!(a.add(a), gf16::Elem::ZERO, "characteristic two");
        for &b in &sample {
            assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
            for &c in &sample {
                assert_eq!(
                    a.mul(b).mul(c),
                    a.mul(b.mul(c)),
                    "multiplication associates"
                );
                assert_eq!(
                    a.mul(b.add(c)),
                    a.mul(b).add(a.mul(c)),
                    "multiplication distributes"
                );
            }
        }
    }
}

#[test]
fn gf16_embeds_the_base_field() {
    // Elements with a zero extension component must multiply exactly as
    // GF(2^8) does. If the tower reduction were wrong this would break.
    for a in all_gf8().step_by(5) {
        for b in all_gf8().step_by(7) {
            let lifted = gf16::Elem::from_components(a, b)
                .mul(gf16::Elem::from_components(gf8::Elem(0), gf8::Elem(0)));
            assert_eq!(lifted, gf16::Elem::ZERO);

            let x = gf16::Elem::from_components(a, gf8::Elem(0));
            let y = gf16::Elem::from_components(b, gf8::Elem(0));
            assert_eq!(
                x.mul(y),
                gf16::Elem::from_components(a.mul(b), gf8::Elem(0)),
                "base-field embedding broken for {a:?} * {b:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2^32) and GF(2^64)
// ---------------------------------------------------------------------------

fn sample_gf32() -> Vec<gf32::Elem> {
    let mut values = vec![
        gf32::Elem::ZERO,
        gf32::Elem::ONE,
        gf32::Elem(u32::MAX),
        gf32::Elem(0x0000_ffff),
        gf32::Elem(0xffff_0000),
        gf32::Elem::from_components(gf32::DELTA, gf16::Elem::ZERO),
        Gf32::GENERATOR,
    ];
    let mut state = 0x243f_6a88u32;
    for _ in 0..48 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(gf32::Elem(state));
    }
    values
}

fn sample_gf64() -> Vec<gf64::Elem> {
    let mut values = vec![
        gf64::Elem::ZERO,
        gf64::Elem::ONE,
        gf64::Elem(u64::MAX),
        gf64::Elem(0x0000_0000_ffff_ffff),
        gf64::Elem(0xffff_ffff_0000_0000),
        Gf64::GENERATOR,
    ];
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(gf64::Elem(state));
    }
    values
}

fn gf32_mul_oracle(x: gf32::Elem, y: gf32::Elem) -> gf32::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    gf32::Elem::from_components(ac.add(gf32::DELTA.mul(bd)), ad.add(bc).add(bd))
}

fn gf64_mul_oracle(x: gf64::Elem, y: gf64::Elem) -> gf64::Elem {
    let (a, b) = x.components();
    let (c, d) = y.components();
    let ac = a.mul(c);
    let ad = a.mul(d);
    let bc = b.mul(c);
    let bd = b.mul(d);
    gf64::Elem::from_components(ac.add(gf64::DELTA.mul(bd)), ad.add(bc).add(bd))
}

#[test]
fn gf32_tower_arithmetic() {
    let sample = sample_gf32();
    for (i, &a) in sample.iter().enumerate() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
        assert_eq!(a.add(a), gf32::Elem::ZERO);
        if a != gf32::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), gf32::Elem::ONE, "inverse({a:?})");
        }
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        assert_eq!(a.mul(b), gf32_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }
}

#[test]
fn gf64_tower_arithmetic() {
    let sample = sample_gf64();
    for (i, &a) in sample.iter().enumerate() {
        assert_eq!(a.square(), a.mul(a), "square({a:?})");
        assert_eq!(a.add(a), gf64::Elem::ZERO);
        if a != gf64::Elem::ZERO {
            assert_eq!(a.mul(a.inv()), gf64::Elem::ONE, "inverse({a:?})");
        }
        let b = sample[(i * 7 + 3) % sample.len()];
        let c = sample[(i * 13 + 5) % sample.len()];
        assert_eq!(a.mul(b), gf64_mul_oracle(a, b), "{a:?} * {b:?}");
        assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }
}

#[test]
fn larger_tower_generators_have_full_order() {
    let g32 = Gf32::GENERATOR;
    let order32 = u32::MAX as u64;
    assert_eq!(g32.pow(order32), gf32::Elem::ONE);
    for factor in [3u64, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), gf32::Elem::ONE);
    }

    let g64 = Gf64::GENERATOR;
    let order64 = u64::MAX;
    assert_eq!(g64.pow(order64), gf64::Elem::ONE);
    for factor in [3u64, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(g64.pow(order64 / factor), gf64::Elem::ONE);
    }
}

// ---------------------------------------------------------------------------
// Canonical Fan-Paar tower
// ---------------------------------------------------------------------------

#[test]
fn fan_paar_matches_canonical_vectors() {
    assert_eq!(
        fan_paar::fp8::Elem(0x1b).mul(fan_paar::fp8::Elem(0xa8)),
        fan_paar::fp8::Elem(0x09)
    );
    assert_eq!(
        fan_paar::fp16::Elem(0x48a8).mul(fan_paar::fp16::Elem(0xf8a4)),
        fan_paar::fp16::Elem(0x3656)
    );
    assert_eq!(
        fan_paar::fp16::Elem(0xf8a4).square(),
        fan_paar::fp16::Elem(0xe7e6)
    );
    assert_eq!(
        fan_paar::fp64::Elem(0xc84d_6191_1083_1cef)
            .mul(fan_paar::fp64::Elem(0x0000_0000_0000_a14f)),
        fan_paar::fp64::Elem(0x3565_086d_6b9e_f595)
    );
}

#[test]
fn fan_paar_arithmetic_round_trips() {
    macro_rules! check {
        ($module:ident, $($value:expr),+ $(,)?) => {
            $(
                let a = fan_paar::$module::Elem($value);
                assert_eq!(a.square(), a.mul(a));
                assert_eq!(a.add(a), fan_paar::$module::Elem::ZERO);
                if a != fan_paar::$module::Elem::ZERO {
                    assert_eq!(a.mul(a.inv()), fan_paar::$module::Elem::ONE);
                }
            )+
        };
    }

    check!(fp8, 0, 1, 0x2d, 0x53, 0xff);
    check!(fp16, 0, 1, 0xe2de, 0x1234, 0xffff);
    check!(fp32, 0, 1, 0x03e2_1cea, 0xdead_beef, u32::MAX);
    check!(
        fp64,
        0,
        1,
        0x070f_870d_cd9c_1d88,
        0x0123_4567_89ab_cdef,
        u64::MAX,
    );
}

#[test]
fn fan_paar_generators_have_full_order() {
    let g8 = FanPaar8::GENERATOR;
    for factor in [3u64, 5, 17] {
        assert_ne!(g8.pow(255 / factor), fan_paar::fp8::Elem::ONE);
    }
    assert_eq!(g8.pow(255), fan_paar::fp8::Elem::ONE);

    let g16 = FanPaar16::GENERATOR;
    for factor in [3u64, 5, 17, 257] {
        assert_ne!(g16.pow(65_535 / factor), fan_paar::fp16::Elem::ONE);
    }
    assert_eq!(g16.pow(65_535), fan_paar::fp16::Elem::ONE);

    let g32 = FanPaar32::GENERATOR;
    let order32 = u32::MAX as u64;
    for factor in [3u64, 5, 17, 257, 65_537] {
        assert_ne!(g32.pow(order32 / factor), fan_paar::fp32::Elem::ONE);
    }
    assert_eq!(g32.pow(order32), fan_paar::fp32::Elem::ONE);

    let g64 = FanPaar64::GENERATOR;
    for factor in [3u64, 5, 17, 257, 641, 65_537, 6_700_417] {
        assert_ne!(g64.pow(u64::MAX / factor), fan_paar::fp64::Elem::ONE);
    }
    assert_eq!(g64.pow(u64::MAX), fan_paar::fp64::Elem::ONE);
}

#[test]
fn fan_paar_subfield_encodings_are_nested() {
    for (a, b) in [(0x1bu8, 0xa8u8), (0x53, 0xca), (0xff, 0x42)] {
        let product = fan_paar::fp8::Elem(a).mul(fan_paar::fp8::Elem(b)).0;
        assert_eq!(
            fan_paar::fp16::Elem(a.into())
                .mul(fan_paar::fp16::Elem(b.into()))
                .0,
            product.into()
        );
        assert_eq!(
            fan_paar::fp32::Elem(a.into())
                .mul(fan_paar::fp32::Elem(b.into()))
                .0,
            product.into()
        );
        assert_eq!(
            fan_paar::fp64::Elem(a.into())
                .mul(fan_paar::fp64::Elem(b.into()))
                .0,
            product.into()
        );
    }
}

// ---------------------------------------------------------------------------
// Representation
// ---------------------------------------------------------------------------

#[test]
fn byte_representation_round_trips() {
    for a in all_gf8() {
        let mut buffer = [0u8; 1];
        Gf8::write(&mut buffer, a);
        assert_eq!(Gf8::read(&buffer), a);
    }
    for a in sample_gf16() {
        let mut buffer = [0u8; 2];
        Gf16::write(&mut buffer, a);
        assert_eq!(Gf16::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf32() {
        let mut buffer = [0u8; 4];
        Gf32::write(&mut buffer, a);
        assert_eq!(Gf32::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    for a in sample_gf64() {
        let mut buffer = [0u8; 8];
        Gf64::write(&mut buffer, a);
        assert_eq!(Gf64::read(&buffer), a);
        assert_eq!(buffer, a.to_raw().to_le_bytes(), "representation is not LE");
    }
    macro_rules! check_fan_paar_repr {
        ($field:ty, $elem:expr, $bytes:literal) => {{
            let value = $elem;
            let mut buffer = [0u8; $bytes];
            <$field>::write(&mut buffer, value);
            assert_eq!(<$field>::read(&buffer), value);
            assert_eq!(buffer, value.to_bytes());
        }};
    }
    check_fan_paar_repr!(FanPaar8, fan_paar::fp8::Elem(0xa5), 1);
    check_fan_paar_repr!(FanPaar16, fan_paar::fp16::Elem(0xa55a), 2);
    check_fan_paar_repr!(FanPaar32, fan_paar::fp32::Elem(0xa55a_1234), 4);
    check_fan_paar_repr!(FanPaar64, fan_paar::fp64::Elem(0xa55a_1234_dead_beef), 8);
}

#[test]
fn field_constants_are_consistent() {
    assert_eq!(Gf8::BYTES, 1);
    assert_eq!(Gf8::ORDER, 1u128 << Gf8::BITS);
    assert_eq!(Gf16::BYTES, 2);
    assert_eq!(Gf16::ORDER, 1u128 << Gf16::BITS);
    assert_eq!(Gf32::BYTES, 4);
    assert_eq!(Gf32::ORDER, 1u128 << Gf32::BITS);
    assert_eq!(Gf64::BYTES, 8);
    assert_eq!(Gf64::ORDER, 1u128 << Gf64::BITS);
    for (bytes, bits) in [
        (Gf8::BYTES, Gf8::BITS),
        (Gf16::BYTES, Gf16::BITS),
        (Gf32::BYTES, Gf32::BITS),
        (Gf64::BYTES, Gf64::BITS),
        (FanPaar8::BYTES, FanPaar8::BITS),
        (FanPaar16::BYTES, FanPaar16::BITS),
        (FanPaar32::BYTES, FanPaar32::BITS),
        (FanPaar64::BYTES, FanPaar64::BITS),
    ] {
        assert_eq!(bytes * 8, bits as usize);
    }
}
