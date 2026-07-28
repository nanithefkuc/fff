//! GF(2^16) as a quadratic tower over [`crate::field::gf8`].
//!
//! With `F = GF(2^8)` under the Rijndael polynomial, an element here is
//! `a + b*u` with `a, b in F` and
//!
//! ```text
//! u^2 + u + DELTA = 0,   DELTA = 0x20.
//! ```
//!
//! The absolute trace of `0x20` in `F` is one, so the quadratic is
//! irreducible.
//!
//! # Why a tower and not a primitive 16-bit polynomial
//!
//! A flat GF(2^16) needs either a 128 KiB log table (blows L1 and is not
//! vectorizable) or a four-level nibble shuffle. The tower instead reduces
//! every 16-bit multiply to base-field multiplies, which the *byte-wide*
//! hardware — `GF2P8MULB` on x86, `PMULL`/nibble-shuffle on NEON — already
//! does 16 or 32 lanes at a time. The whole 16-bit kernel is then two
//! byte-wide multiplies plus a lane swap.
//!
//! # Representation
//!
//! The stable byte form is `[a, b]`: constant component first, extension
//! component second. That is exactly the little-endian encoding of the
//! wrapped `u16`, so a buffer of GF(2^16) elements is a byte-interleaved
//! pair of base-field planes at stride two.
//!
//! ```
//! use fff::gf16::{self, Elem, DELTA};
//! use fff::gf8;
//!
//! let x = Elem::from_components(gf8::Elem(0x12), gf8::Elem(0x34));
//! assert_eq!(x.to_raw(), 0x3412);
//! assert_eq!(x.to_bytes(), [0x12, 0x34]);
//!
//! // The defining relation: u^2 == u + DELTA.
//! const U: Elem = Elem::from_components(gf8::Elem::ZERO, gf8::Elem::ONE);
//! const DELTA_LIFTED: Elem = Elem::from_components(DELTA, gf8::Elem::ZERO);
//! const _: () = assert!(U.square().to_raw() == U.add(DELTA_LIFTED).to_raw());
//!
//! // The documented generator really does have order 65535.
//! assert_eq!(gf16::GENERATOR.pow(65_535), Elem::ONE);
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Elem(0x0108).div(Elem::ZERO).to_raw() == 0);
//! ```

use core::fmt;

use super::gf8::Elem as Base;
use super::{Elem as ElemTrait, Field};

/// Constant term of the irreducible tower polynomial `u^2 + u + DELTA`.
pub const DELTA: Base = Base(0x20);

/// A primitive element of the extension field, `0x08 + u`.
///
/// Its multiplicative order is 65535.
pub const GENERATOR: Elem = Elem(0x0108);

/// Marker type for GF(2^16).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf16;

/// An element of `GF((2^8)^2)`, stored as `a + b*u` with `a` in the low byte.
///
/// The derived [`Ord`] is raw-representation order, useful for map keys and
/// deterministic iteration; it carries no field-theoretic meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u16);

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Construct `a + b*u` from its two base-field components.
    #[inline]
    #[must_use]
    pub const fn from_components(a: Base, b: Base) -> Self {
        Self((a.0 as u16) | ((b.0 as u16) << 8))
    }

    /// Return the `(a, b)` components of `a + b*u`.
    #[inline]
    #[must_use]
    // Splitting a u16 into its two bytes: the truncation IS the operation.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn components(self) -> (Base, Base) {
        (Base(self.0 as u8), Base((self.0 >> 8) as u8))
    }

    /// Decode from the stable `[a, b]` byte representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// Encode to the stable `[a, b]` byte representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Wrap raw component bits.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u16) -> Self {
        Self(value)
    }

    /// Unwrap to the raw component bits.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u16 {
        self.0
    }

    /// Field addition: component-wise XOR.
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

    /// Karatsuba multiplication followed by `u^2 = u + DELTA` reduction.
    ///
    /// Three base multiplies for the Karatsuba products plus one for the
    /// `DELTA` fold, versus four for the schoolbook form.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.components();
        let (c, d) = rhs.components();
        let ac = a.mul(c);
        let bd = b.mul(d);
        let constant = ac.add(DELTA.mul(bd));
        // (a+b)(c+d) + ac = ad + bc + bd, which is the coefficient of u once
        // bd*u^2 has been rewritten as bd*u + DELTA*bd.
        let extension = a.add(b).mul(c.add(d)).add(ac);
        Self::from_components(constant, extension)
    }

    /// Square using the tower relation. Cheaper than a general multiply:
    /// cross terms vanish in characteristic two.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        let (a, b) = self.components();
        let a2 = a.mul(a);
        let b2 = b.mul(b);
        Self::from_components(a2.add(DELTA.mul(b2)), b2)
    }

    /// Multiplicative inverse through the quadratic conjugate and norm.
    ///
    /// `conj(a + b*u) = (a + b) + b*u` and `norm = a^2 + a*b + DELTA*b^2`,
    /// so the inverse is `conj / norm` with a single base-field inversion.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        let (a, b) = self.components();
        if a.0 == 0 && b.0 == 0 {
            return Self::ZERO;
        }
        let norm = a.mul(a).add(a.mul(b)).add(DELTA.mul(b.mul(b)));
        let norm_inv = norm.inv();
        Self::from_components(a.add(b).mul(norm_inv), b.mul(norm_inv))
    }

    /// Field division. Returns zero when either operand is zero.
    ///
    /// `x / 0 == 0` is a definition, not an oversight: keeping division total
    /// leaves hot loops branch-free and keeps this callable from `const`
    /// context. A `debug_assert!` here would turn any `const` division by zero
    /// into a compile error in debug builds and silence in release ones.
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

impl Field for Gf16 {
    type Elem = Elem;

    const NAME: &'static str = "GF(2^16)";
    const BITS: u32 = 16;
    const BYTES: usize = 2;
    const ORDER: u128 = 65_536;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let bytes: [u8; 2] = bytes
            .try_into()
            .expect("GF(2^16) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(bytes.len(), 2, "GF(2^16) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b) = self.components();
        write!(f, "Gf16({:#04x}+{:#04x}u)", a.0, b.0)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}", self.0)
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
