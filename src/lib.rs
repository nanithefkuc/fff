//! # Faster Finite Fields
//!
//! SIMD-optimized binary finite fields and the vector kernels erasure codes
//! are built out of.
//!
//! Supported field families:
//!
//! | Field | Type | Element | Construction |
//! | --- | --- | --- | --- |
//! | GF(2^8) | [`Gf8`] | [`gf8::Elem`] | AES polynomial `0x11B` |
//! | GF(2^16) | [`Gf16`] | [`gf16::Elem`] | quadratic tower over [`Gf8`] |
//! | GF(2^32) | [`Gf32`] | [`gf32::Elem`] | quadratic tower over [`Gf16`] |
//! | GF(2^64) | [`Gf64`] | [`gf64::Elem`] | quadratic tower over [`Gf32`] |
//! | GF(2^8)..GF(2^64) | [`FanPaar8`]..[`FanPaar64`] | [`fan_paar::fp8::Elem`]..[`fan_paar::fp64::Elem`] | canonical Fan–Paar tower |
//!
//! `Gf8` and `Gf16` have hand-written SIMD backends. The wider polynomial
//! towers `Gf32`/`Gf64` run the same tower identity on x86 GFNI, and the
//! canonical Fan–Paar `FanPaar16`/`FanPaar32`/`FanPaar64` run their nibble-
//! shuffle tower on x86 AVX2 (and `FanPaar16` on SSSE3); on every other target
//! those and `FanPaar8` use the portable kernels. All types share the same
//! checked [`ops`] surface and stable little-endian encoding.
//!
//! ## Two layers
//!
//! **Scalar algebra** — [`field::Elem`] gives
//! `add`/`sub`/`mul`/`square`/`inv`/`div`/`pow` over single elements. The
//! concrete element types carry the same methods inherently and `const`, so
//! coding matrices can be built at compile time; `use fff::field::Elem;` to
//! get them in scope when writing code generic over the field.
//!
//! **Vector kernels** — [`ops`] operates on `&[u8]` buffers of packed
//! elements, dispatching once per process to the best backend the host
//! supports ([`Backend`]).
//!
//! ```
//! use fff::{Gf8, gf8, ops};
//!
//! let src = [0x01u8, 0x02, 0x03, 0x04];
//! let mut dst = [0u8; 4];
//!
//! // dst ^= 0x03 * src
//! ops::mul_add::<Gf8>(&mut dst, gf8::Elem(0x03), &src);
//! assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
//!
//! // Undo it: adding the same term back is subtracting it.
//! ops::mul_add::<Gf8>(&mut dst, gf8::Elem(0x03), &src);
//! assert_eq!(dst, [0, 0, 0, 0]);
//! ```
//!
//! The same code over GF(2^16), where buffers hold little-endian element
//! pairs:
//!
//! ```
//! use fff::{Gf16, gf16, ops};
//!
//! let src = 0x1234u16.to_le_bytes();
//! let mut dst = [0u8; 2];
//! ops::mul_add::<Gf16>(&mut dst, gf16::Elem(0x0108), &src);
//!
//! let expected = gf16::Elem(0x1234).mul(gf16::Elem(0x0108));
//! assert_eq!(dst, expected.to_bytes());
//! ```
//!
//! ## Choosing an operation
//!
//! | Shape | One-shot | Prepared | Where it appears |
//! | --- | --- | --- | --- |
//! | `dst ^= src` | [`ops::add_assign`] | — | XOR-only parity |
//! | `dst ^= c * src` | [`ops::mul_add`] | [`ops::mul_add_with`] | AXPY |
//! | `dst = c * src` | [`ops::mul_into`] | [`ops::mul_into_with`] | row scaling |
//! | `dst *= c` | [`ops::mul_assign`] | [`ops::mul_assign_with`] | in-place scaling |
//! | one source, many rows | [`ops::mul_add_scatter`] | `ops::mul_add_scatter_with` | systematic encode |
//! | many sources, one row | [`ops::mul_add_gather`] | `ops::mul_add_gather_with` | recovered symbol |
//! | many sources, many rows | [`ops::mul_add_matrix`] | `ops::mul_add_matrix_with` | reconstruction |
//! | varying pair per lane | [`ops::mul_elementwise`] | — | pointwise products |
//!
//! Prefer the widest shape that fits: [`ops::mul_add_matrix`] holds its
//! destination tiles in registers across all sources, so its destination
//! memory traffic does not grow with the source count.
//!
//! [`ops::Coeff`] prepares one coefficient. With `std`, `ops::Plan` stores a
//! prepared vector or row-major matrix for the multi-row `_with` operations.
//! [`ops::pack`], [`ops::unpack`], and `ops::pack_to_vec` bridge typed elements
//! and packed byte buffers.
//!
//! ## Features
//!
//! - `std` (default) — the standards library (and its lazily-built table
//!   banks).
//! - `simd` (default, implies `std`) — the vector backends. Disabling leaves
//!   the portable scalar kernels, which are correct but slow.
//!
//! [`kernel::backend()`] reports the process-wide SIMD selection over the
//! tiers this crate implements; [`backend_for`] reports the backend used by a
//! specific field, so portable-only wider fields do not appear accelerated on
//! a SIMD host. Detection and ordering are single-source: [`Backend`] is a
//! re-export of [`simdispatch::Backend`](https://docs.rs/simdispatch), and
//! selection is [`simdispatch`](https://github.com/nanithefkuc/simdispatch)'s
//! `Selection` resolved over [`kernel::FFF_TIERS`], with the downgrade-only
//! `SIMD_BACKEND` override.
//!
//! ## Safety and scope
//!
//! The public API is safe. Unsafe intrinsics are confined to private
//! architecture modules and entered only after runtime feature detection.
//! Every backend is differentially tested against the portable implementation.
//!
//! This crate does not build coding matrices or own shards. Cauchy/Vandermonde
//! recipes, matrix inversion, and streaming recovery belong in a codec layer.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Arch intrinsics are imported wholesale by universal convention; naming
    // each of the ~200 used here would be unmaintainable and would have to be
    // duplicated across the x86 and x86_64 cfg arms.
    clippy::wildcard_imports,
    clippy::inline_always,
    clippy::module_name_repetitions
)]

pub mod field;
pub mod kernel;
pub mod ops;

pub use field::{
    FanPaar8, FanPaar16, FanPaar32, FanPaar64, Field, Gf8, Gf16, Gf32, Gf64, fan_paar, gf8, gf16,
    gf32, gf64,
};
pub use kernel::{
    Backend, FieldKernels, KernelBackend, ParseBackendError, backend, backend_for,
    has_vector_elementwise,
};
