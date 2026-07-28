//! GF(2^32) portable kernel dispatch.

use crate::field::gf32::Gf32;

crate::kernel::scalar::impl_field_kernels!(Gf32);
