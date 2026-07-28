//! Field definitions and the scalar algebra contract.
//!
//! A [`Field`] is a zero-sized marker type carrying compile-time facts about a
//! binary field: its element type, its width, and its stable byte
//! representation. Kernels in [`crate::ops`] are generic over `Field`, so
//! callers write one algorithm and get a monomorphized, SIMD-dispatched
//! implementation per field.
//!
//! # Byte representation
//!
//! Every field fixes a stable little-endian byte encoding of width
//! [`Field::BYTES`]. Vector kernels operate on `&[u8]` buffers holding a packed
//! array of such encodings. This keeps the kernel surface alignment-free and
//! lets callers hand raw network/disk payloads straight to the field ops
//! without a transmute.

pub mod fan_paar;
pub mod gf16;
pub mod gf32;
pub mod gf64;
pub mod gf8;

pub use fan_paar::{FanPaar8, FanPaar16, FanPaar32, FanPaar64};
pub use gf8::Gf8;
pub use gf16::Gf16;
pub use gf32::Gf32;
pub use gf64::Gf64;

/// Scalar arithmetic over a binary field.
///
/// All binary fields have characteristic two, so addition and subtraction are
/// the same operation (XOR). Both are provided because algorithms read more
/// clearly when they say what they mean.
///
/// By library-wide convention `inv(0) == 0` and `x / 0 == 0`, in every build
/// profile and under `const` evaluation alike. This is a total-function
/// convention chosen so hot loops never branch on an impossible case; it is
/// *not* a claim that zero is invertible.
pub trait Elem:
    Copy + Clone + PartialEq + Eq + core::fmt::Debug + core::hash::Hash + Default + 'static
{
    /// The additive identity, and the absorbing element for multiplication.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    /// Field addition (XOR).
    #[must_use]
    fn add(self, rhs: Self) -> Self;
    /// Field subtraction. Identical to [`Elem::add`] in characteristic two.
    #[must_use]
    fn sub(self, rhs: Self) -> Self;
    /// Field multiplication.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;
    /// Multiplicative inverse. Maps zero to zero by convention.
    #[must_use]
    fn inv(self) -> Self;
    /// Field division. Returns zero when the divisor is zero.
    #[must_use]
    fn div(self, rhs: Self) -> Self;

    /// Square.
    ///
    /// Defaults to `self.mul(self)`. Tower fields override it: squaring is
    /// linear in characteristic two, so the cross terms vanish and the
    /// dedicated routine is strictly cheaper than a general multiply.
    #[inline]
    #[must_use]
    fn square(self) -> Self {
        self.mul(self)
    }

    /// Raise to an unsigned integer power by square-and-multiply.
    ///
    /// Present on the trait so generic code can exponentiate without knowing
    /// the field's group order. `pow(0)` is [`Elem::ONE`] for every element,
    /// zero included.
    #[must_use]
    fn pow(self, exponent: u64) -> Self {
        let mut base = self;
        let mut exponent = exponent;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exponent >>= 1;
        }
        result
    }

    /// Whether this element is the additive identity.
    #[inline]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    /// Whether this element is the multiplicative identity.
    #[inline]
    fn is_one(self) -> bool {
        self == Self::ONE
    }
}

/// A binary field supported by this crate.
pub trait Field: Copy + Clone + core::fmt::Debug + 'static {
    /// The scalar element type.
    type Elem: Elem;

    /// Human-readable field name, e.g. `"GF(2^8)"`.
    const NAME: &'static str;
    /// Extension degree over GF(2).
    const BITS: u32;
    /// Width of the stable byte representation of one element.
    const BYTES: usize;
    /// Number of elements in the field.
    const ORDER: u128;
    /// A generator of the multiplicative group.
    const GENERATOR: Self::Elem;

    /// Decode one element from its stable little-endian byte representation.
    ///
    /// # Panics
    /// Panics if `bytes.len() != Self::BYTES`.
    fn read(bytes: &[u8]) -> Self::Elem;

    /// Encode one element into its stable little-endian byte representation.
    ///
    /// # Panics
    /// Panics if `bytes.len() != Self::BYTES`.
    fn write(bytes: &mut [u8], value: Self::Elem);

    /// Number of whole elements a byte buffer holds.
    #[inline]
    #[must_use]
    fn elem_count(bytes: usize) -> usize {
        bytes / Self::BYTES
    }
}
