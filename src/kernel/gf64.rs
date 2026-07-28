//! GF(2^64) portable kernel dispatch.

use crate::field::gf64::Gf64;

crate::kernel::scalar::impl_field_kernels!(Gf64);
