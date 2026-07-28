//! GF(2^32) as a quadratic tower over [`crate::field::gf16`].
//!
//! An element is `a + b*v` with `a, b in GF(2^16)` and
//!
//! ```text
//! v^2 + v + DELTA = 0,   DELTA = 0x2000.
//! ```
//!
//! The absolute trace of `DELTA` in GF(2^16) is one, so the quadratic is
//! irreducible. The stable byte form is the little-endian component encoding
//! `[a, b]`.
//!
//! ```
//! use fff::gf32::{self, Elem};
//! use fff::gf16;
//!
//! const X: Elem = Elem::from_components(gf16::Elem(0x1234), gf16::Elem(0x5678));
//! const _: () = assert!(X.to_raw() == 0x5678_1234);
//!
//! // `inv` is `const`, so a reciprocal table can be a `const` item.
//! const RECIP: Elem = X.inv();
//! const _: () = assert!(X.mul(RECIP).to_raw() == Elem::ONE.to_raw());
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(X.div(Elem::ZERO).to_raw() == 0);
//!
//! assert_eq!(X.components(), (gf16::Elem(0x1234), gf16::Elem(0x5678)));
//! assert_eq!(gf32::GENERATOR.pow(u64::from(u32::MAX)), Elem::ONE);
//! ```

use core::fmt;

use super::gf16::Elem as Base;
use super::{Elem as ElemTrait, Field};

/// Constant term of the irreducible tower polynomial `v^2 + v + DELTA`.
pub const DELTA: Base = Base(0x2000);

/// A primitive element of the extension field, `0x0002 + v`.
///
/// Its multiplicative order is `2^32 - 1`.
pub const GENERATOR: Elem = Elem(0x0001_0002);

/// Marker type for GF(2^32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Gf32;

/// An element of `GF((2^16)^2)`, stored as `a + b*v` with `a` in the low half.
///
/// The derived [`Ord`] is raw-representation order, useful for map keys and
/// deterministic iteration; it carries no field-theoretic meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Elem(pub u32);

impl Elem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Construct `a + b*v` from its two base-field components.
    #[inline]
    #[must_use]
    pub const fn from_components(a: Base, b: Base) -> Self {
        Self((a.0 as u32) | ((b.0 as u32) << 16))
    }

    /// Return the `(a, b)` components of `a + b*v`.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn components(self) -> (Base, Base) {
        (Base(self.0 as u16), Base((self.0 >> 16) as u16))
    }

    /// Decode from the stable little-endian component representation.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }

    /// Encode to the stable little-endian component representation.
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Wrap raw component bits.
    #[inline]
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    /// Unwrap to the raw component bits.
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u32 {
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

    /// Karatsuba multiplication followed by `v^2 = v + DELTA` reduction.
    #[inline]
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.components();
        let (c, d) = rhs.components();
        let ac = a.mul(c);
        let bd = b.mul(d);
        let constant = ac.add(DELTA.mul(bd));
        let extension = a.add(b).mul(c.add(d)).add(ac);
        Self::from_components(constant, extension)
    }

    /// Square using the tower relation.
    #[inline]
    #[must_use]
    pub const fn square(self) -> Self {
        let (a, b) = self.components();
        let a2 = a.square();
        let b2 = b.square();
        Self::from_components(a2.add(DELTA.mul(b2)), b2)
    }

    /// Multiplicative inverse through the quadratic conjugate and norm.
    #[inline]
    #[must_use]
    pub const fn inv(self) -> Self {
        let (a, b) = self.components();
        if a.0 == 0 && b.0 == 0 {
            return Self::ZERO;
        }
        let norm = a.square().add(a.mul(b)).add(DELTA.mul(b.square()));
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

impl Field for Gf32 {
    type Elem = Elem;

    const NAME: &'static str = "GF(2^32)";
    const BITS: u32 = 32;
    const BYTES: usize = 4;
    const ORDER: u128 = 1u128 << 32;
    const GENERATOR: Elem = GENERATOR;

    #[inline]
    fn read(bytes: &[u8]) -> Elem {
        let bytes: [u8; 4] = bytes
            .try_into()
            .expect("GF(2^32) element has the wrong byte width");
        Elem::from_bytes(bytes)
    }

    #[inline]
    fn write(bytes: &mut [u8], value: Elem) {
        assert_eq!(bytes.len(), 4, "GF(2^32) element has the wrong byte width");
        bytes.copy_from_slice(&value.to_bytes());
    }
}

impl fmt::Debug for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b) = self.components();
        write!(f, "Gf32({:#06x}+{:#06x}v)", a.0, b.0)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08x}", self.0)
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
