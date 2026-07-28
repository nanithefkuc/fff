//! Portable kernel dispatch for canonical Fan–Paar fields.

use crate::field::fan_paar::{FanPaar8, FanPaar16, FanPaar32, FanPaar64};

crate::kernel::scalar::impl_field_kernels!(FanPaar8);
crate::kernel::scalar::impl_field_kernels!(FanPaar16);
crate::kernel::scalar::impl_field_kernels!(FanPaar32);
crate::kernel::scalar::impl_field_kernels!(FanPaar64);
