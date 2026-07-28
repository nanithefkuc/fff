//! GF(2^8) under the AES/Rijndael polynomial `0x11B`.
//!
//! The field is `GF(2)[x] / (x^8 + x^4 + x^3 + x + 1)`. Addition and
//! subtraction are bitwise XOR. Multiplication uses compile-time discrete-log
//! tables built from generator `0x03`.
//!
//! This polynomial is not an arbitrary choice: it is the one implemented in
//! hardware by the x86 `GF2P8MULB` instruction (GFNI). Picking it lets the
//! SIMD backend issue a single instruction per 64 field multiplications
//! instead of a nibble-shuffle emulation.
//!
//! # Backends
//!
//! - [`Elem::mul_xtime`] is the reference: shift-and-XOR, `const`, no static
//!   storage. Every other backend is differentially tested against it.
//! - [`Elem::mul`] uses the `LOG`/`EXP` tables (511 bytes, in L1).
//!
//! # Compile-time coding matrices
//!
//! Every scalar operation here is `const`, so a Reed-Solomon generator matrix
//! is a `const` item rather than a lazily-built table. This builds the
//! 3-by-4 Vandermonde matrix `V[i][j] = x_j^i` at compile time:
//!
//! ```
//! use fff::gf8::Elem;
//!
//! const POINTS: [Elem; 4] = [Elem(1), Elem(2), Elem(3), Elem(4)];
//! const V: [[Elem; 4]; 3] = {
//!     let mut rows = [[Elem::ZERO; 4]; 3];
//!     let mut i = 0;
//!     while i < 3 {
//!         let mut j = 0;
//!         while j < 4 {
//!             rows[i][j] = POINTS[j].pow(i as u64);
//!             j += 1;
//!         }
//!         i += 1;
//!     }
//!     rows
//! };
//!
//! assert_eq!(V[0], [Elem::ONE; 4]);
//! assert_eq!(V[1], POINTS);
//! assert_eq!(V[2][3], Elem(4).square());
//!
//! // A parity symbol is one Vandermonde row dotted with the data symbols.
//! let data = [Elem(0x11), Elem(0x22), Elem(0x33), Elem(0x44)];
//! let parity: Elem = V[1].iter().zip(data).map(|(&v, d)| v.mul(d)).sum();
//! assert_eq!(parity, Elem(0x11).mul(Elem(1)).add(Elem(0x22).mul(Elem(2)))
//!     .add(Elem(0x33).mul(Elem(3))).add(Elem(0x44).mul(Elem(4))));
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Elem(0x57).div(Elem::ZERO).to_raw() == 0);
//! ```

use core::fmt;

use super::{Elem as ElemTrait, Field};

/// Irreducible reduction polynomial `x^8 + x^4 + x^3 + x + 1`.
pub const REDUCTION_POLY: u16 = 0x11B;

/// Low byte of [`REDUCTION_POLY`], `XORed` in when a shift overflows the field.
pub const REDUCTION_LOW: u8 = 0x1B;

/// Generator of the multiplicative group under this polynomial.
///
/// `0x03` is the conventional AES generator, which keeps the tables below
/// byte-for-byte comparable with published references.
pub const GENERATOR: Elem = Elem(0x03);

/// Marker type for GF(2^8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf8;

/// An element of GF(2^8), stored as its polynomial coefficient vector.
///
/// The derived [`Ord`] is raw-representation order, useful for map keys and
/// deterministic iteration; it carries no field-theoretic meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u8);

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Wrap a raw coefficient vector.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Unwrap to the raw coefficient vector.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u8 {
        self.0
    }

    /// Decode from the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        Self(bytes[0])
    }

    /// Encode to the stable single-byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 1] {
        [self.0]
    }

    /// Field addition. Identical to subtraction and to bitwise XOR.
    #[inline]
    #[must_use]
    pub const fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    /// Field subtraction. Identical to [`Elem::add`].
    #[inline]
    #[must_use]
    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }

    /// Reference multiplication: shift-and-XOR ("Russian peasant").
    ///
    /// `const`, allocation-free, and independent of the tables. This is the
    /// oracle the table and SIMD backends are validated against.
    #[must_use]
    pub const fn mul_xtime(self, rhs: Self) -> Self {
        let mut a = self.0;
        let b = rhs.0;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            let overflow = a & 0x80 != 0;
            a <<= 1;
            if overflow {
                a ^= REDUCTION_LOW;
            }
            i += 1;
        }
        Self(acc)
    }

    /// Reference inverse via Fermat's little theorem: `a^254 == a^-1`.
    #[must_use]
    pub const fn inv_xtime(self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        // 254 = 0b1111_1110. Square every step; multiply on every set bit,
        // scanning most-significant first.
        let mut result = Self::ONE;
        let mut i = 0;
        while i < 8 {
            result = result.mul_xtime(result);
            if (254u32 >> (7 - i)) & 1 == 1 {
                result = result.mul_xtime(self);
            }
            i += 1;
        }
        result
    }

    /// Field multiplication through the discrete-log tables.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        // LOG has no entry for zero, so the absorbing case must short-circuit.
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        let la = LOG[self.0 as usize] as usize;
        let lb = LOG[rhs.0 as usize] as usize;
        Self(EXP[(la + lb) % 255])
    }

    /// Square.
    ///
    /// A flat field has no cheaper form than a general multiply; the tower
    /// fields above do. Present for parity with them.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Multiplicative inverse. Maps zero to zero by crate convention.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        // The `% 255` matters only for `self == 1`, where `LOG` is 0 and
        // `255 - 0` would run off the end of the 255-entry table.
        let l = LOG[self.0 as usize] as usize;
        Self(EXP[(255 - l) % 255])
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context. A `debug_assert!` here would turn any `const` division by zero
    /// into a compile error in debug builds and silence in release ones.
    #[inline]
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if self.0 == 0 || rhs.0 == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise to an unsigned integer power.
    #[must_use]
    pub const fn pow(self, mut exponent: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }
}

impl ElemTrait for Elem {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
    #[inline]
    fn square(self) -> Self {
        Elem::square(self)
    }
    #[inline]
    fn inv(self) -> Self {
        Elem::inv(self)
    }
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
    #[inline]
    fn pow(self, exponent: u64) -> Self {
        Elem::pow(self, exponent)
    }
}

impl Field for Gf8 {
    type Elem = Elem;

    const NAME: &'static str = "GF(2^8)";
    const BITS: u32 = 8;
    const BYTES: usize = 1;
    const ORDER: u128 = 256;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let bytes: [u8; 1] = bytes
            .try_into()
            .expect("GF(2^8) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(bytes.len(), 1, "GF(2^8) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf8({:#04x})", self.0)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}", self.0)
    }
}

impl core::ops::Add for Elem {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Elem::add(self, rhs)
    }
}

impl core::ops::Sub for Elem {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Elem::sub(self, rhs)
    }
}

impl core::ops::Mul for Elem {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Elem::mul(self, rhs)
    }
}

impl core::ops::Div for Elem {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Elem::div(self, rhs)
    }
}

impl core::ops::AddAssign for Elem {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Elem::add(*self, rhs);
    }
}

impl core::ops::SubAssign for Elem {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Elem::sub(*self, rhs);
    }
}

impl core::ops::MulAssign for Elem {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Elem::mul(*self, rhs);
    }
}

impl core::ops::DivAssign for Elem {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Elem::div(*self, rhs);
    }
}

impl core::iter::Sum for Elem {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Elem::add)
    }
}

impl<'a> core::iter::Sum<&'a Elem> for Elem {
    #[inline]
    fn sum<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| Elem::add(acc, x))
    }
}

impl core::iter::Product for Elem {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Elem::mul)
    }
}

impl<'a> core::iter::Product<&'a Elem> for Elem {
    #[inline]
    fn product<I: Iterator<Item = &'a Elem>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| Elem::mul(acc, x))
    }
}

// ---------------------------------------------------------------------------
// Compile-time discrete-log tables.
// ---------------------------------------------------------------------------

/// `EXP[i] = GENERATOR^i` for `i in 0..255`.
pub(crate) const EXP: [u8; 255] = build_exp();

/// `LOG[EXP[i]] = i` for nonzero elements. `LOG[0]` is undefined and set to
/// zero; callers must short-circuit on zero before indexing.
pub(crate) const LOG: [u8; 256] = build_log();

const fn build_exp() -> [u8; 255] {
    let mut table = [0u8; 255];
    let mut value: u8 = 1;
    let mut i = 0;
    while i < 255 {
        table[i] = value;
        value = Elem(value).mul_xtime(GENERATOR).0;
        i += 1;
    }
    table
}

// `i < 255`, so the cast is exact; `const fn` rules out `try_into`.
#[allow(clippy::cast_possible_truncation)]
const fn build_log() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 255 {
        table[EXP[i] as usize] = i as u8;
        i += 1;
    }
    table
}
