//! Precomputed per-coefficient multiplication tables.
//!
//! Two families, matching the two ways SIMD hardware can multiply in a binary
//! field:
//!
//! - [`ScaleTable`] — split-nibble shuffle tables for one GF(2^8) coefficient.
//!   `c * x` is decomposed as `c * (x & 0xf) ^ c * (x & 0xf0)`, each half a
//!   16-entry lookup that `PSHUFB`/`TBL` performs on every lane at once.
//! - [`TowerCoeff`] — the two broadcast words that turn one GF(2^16) tower
//!   multiply into two byte-wide multiplies of the source and its
//!   adjacent-swapped self. No table at all, which is what makes the GFNI and
//!   `PMULL` paths cheap.
//!
//! The GF(2^8) table bank is shared and built once; all 256 coefficients cost
//! 8 KiB and stay resident in L1/L2. GF(2^16) has 65536 coefficients, so a
//! full bank would be ~9 MiB and thrash cache. A GF(2^16) coefficient is
//! instead resolved per call into its four base-field factors (two base
//! multiplies, [`TowerCoeff::new`]) and, on shuffle backends, four table
//! copies out of the shared bank ([`TowerTables::new`]) — amortized over the
//! whole buffer, or hoisted out entirely with `Coeff`/`Plan`.

use crate::field::gf8;
use crate::field::gf16;

/// Split-nibble multiplication tables for one GF(2^8) coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleTable {
    /// The coefficient these tables multiply by.
    pub coeff: gf8::Elem,
    /// `lo[i] = coeff * i` for the low nibble.
    pub lo: [u8; 16],
    /// `hi[i] = coeff * (i << 4)` for the high nibble.
    pub hi: [u8; 16],
}

impl ScaleTable {
    /// Build the nibble tables for `coeff`.
    #[must_use]
    // Loop counters are bounded by the array sizes (16, 256), so every cast
    // below is exact; `const fn` rules out `try_into`.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new(coeff: gf8::Elem) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut i = 0;
        while i < 16 {
            lo[i] = gf8::Elem(i as u8).mul(coeff).0;
            hi[i] = gf8::Elem((i as u8) << 4).mul(coeff).0;
            i += 1;
        }
        Self { coeff, lo, hi }
    }
}

/// Shared table bank, one entry per GF(2^8) coefficient.
///
/// A `static` rather than a `const`: a `const` array would be materialized
/// onto the stack at every use site instead of being borrowed from rodata.
static SCALE_TABLE_BANK: [ScaleTable; 256] = build_bank();

#[allow(clippy::cast_possible_truncation)]
const fn build_bank() -> [ScaleTable; 256] {
    let mut bank = [ScaleTable::new(gf8::Elem(0)); 256];
    let mut i = 0;
    while i < 256 {
        bank[i] = ScaleTable::new(gf8::Elem(i as u8));
        i += 1;
    }
    bank
}

/// Return the shared nibble tables for a GF(2^8) coefficient.
#[inline]
#[must_use]
pub fn scale_table(coeff: gf8::Elem) -> &'static ScaleTable {
    &SCALE_TABLE_BANK[coeff.0 as usize]
}

/// Broadcast factors that express one GF(2^16) tower multiply as two
/// byte-wide GF(2^8) multiplies.
///
/// Interleaved source bytes are `[a, b]` meaning `a + b*u`. Multiplying by
/// `c0 + c1*u` gives
///
/// ```text
/// (c0*a + DELTA*c1*b) + (c1*a + (c0 + c1)*b) * u.
/// ```
///
/// The first term of each component multiplies the source in place; the
/// second multiplies the source with adjacent bytes swapped. So a single
/// alternating-coefficient byte multiply of `src` by [`TowerCoeff::same`],
/// `XORed` with the same of `swap16(src)` by [`TowerCoeff::cross`], produces
/// both components with no planar de-interleave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerCoeff {
    /// The GF(2^16) coefficient.
    pub coeff: gf16::Elem,
    /// Little-endian `[c0, c0 + c1]`, applied to the source directly.
    pub same: u16,
    /// Little-endian `[DELTA*c1, c1]`, applied to the byte-swapped source.
    pub cross: u16,
}

impl TowerCoeff {
    /// Derive the broadcast factors for `coeff`. Two base multiplies.
    #[inline]
    #[must_use]
    pub const fn new(coeff: gf16::Elem) -> Self {
        let (c0, c1) = coeff.components();
        let same = u16::from_le_bytes([c0.0, c0.add(c1).0]);
        let cross = u16::from_le_bytes([gf16::DELTA.mul(c1).0, c1.0]);
        Self { coeff, same, cross }
    }

    /// The four base-field coefficients in `[same.0, same.1, cross.0,
    /// cross.1]` order, i.e. `[c0, c0+c1, DELTA*c1, c1]`.
    #[inline]
    #[must_use]
    pub const fn factors(self) -> [gf8::Elem; 4] {
        let [s0, s1] = self.same.to_le_bytes();
        let [x0, x1] = self.cross.to_le_bytes();
        [gf8::Elem(s0), gf8::Elem(s1), gf8::Elem(x0), gf8::Elem(x1)]
    }
}

/// Nibble tables for the four base-field factors of a GF(2^16) coefficient.
///
/// Used by the shuffle backends (AVX2, SSSE3, NEON), which have no byte-wide
/// field multiply instruction and must emulate one per factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TowerTables {
    /// The GF(2^16) coefficient.
    pub coeff: gf16::Elem,
    /// Tables for `[c0, c0+c1, DELTA*c1, c1]`. Entries 0 and 1 apply to the
    /// source's even and odd byte lanes; entries 2 and 3 apply to the
    /// adjacent-swapped source's even and odd lanes.
    pub factors: [ScaleTable; 4],
}

impl TowerTables {
    /// Build the four nibble tables for `coeff`.
    #[inline]
    #[must_use]
    pub fn new(coeff: gf16::Elem) -> Self {
        let f = TowerCoeff::new(coeff).factors();
        Self {
            coeff,
            factors: [
                *scale_table(f[0]),
                *scale_table(f[1]),
                *scale_table(f[2]),
                *scale_table(f[3]),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::gf8::Elem as E8;

    #[test]
    fn nibble_tables_reconstruct_the_product() {
        for c in 0..=u8::MAX {
            let table = scale_table(E8(c));
            for x in 0..=u8::MAX {
                let split = table.lo[(x & 0x0f) as usize] ^ table.hi[(x >> 4) as usize];
                assert_eq!(split, E8(x).mul(E8(c)).0, "coeff {c:#04x} value {x:#04x}");
            }
        }
    }

    #[test]
    fn tower_factors_reconstruct_the_product() {
        // Exercise the identity the SIMD kernels rely on, scalar-side.
        for coeff in [0u16, 1, 0x0108, 0x2000, 0xffff, 0x1234] {
            let coeff = gf16::Elem(coeff);
            let tc = TowerCoeff::new(coeff);
            let [f0, f1, f2, f3] = tc.factors();
            for value in [0u16, 1, 0x00ff, 0xff00, 0xbeef] {
                let value = gf16::Elem(value);
                let [a, b] = value.to_bytes();
                let (a, b) = (E8(a), E8(b));
                // even lane: same.0 * a  ^  cross.0 * b   (b is a's swap partner)
                // odd  lane: same.1 * b  ^  cross.1 * a
                let even = f0.mul(a).add(f2.mul(b));
                let odd = f1.mul(b).add(f3.mul(a));
                let got = gf16::Elem::from_bytes([even.0, odd.0]);
                assert_eq!(got, value.mul(coeff), "{value:?} * {coeff:?}");
            }
        }
    }
}
