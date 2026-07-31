# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- crates.io publishing metadata and docs.rs configuration; the crate is
  distributed through git only.

### Added

- `internals` feature exposing the kernel modules and preparation types
  (`ScaleTable`, `TowerCoeff`, `TowerTables`, backend-specific SIMD entry
  points) for downstream libraries that build directly on the kernels.
- wasm32 `simd128` GF(2^8) multi-row kernels: register-blocked
  `mul_add_scatter`, `mul_add_gather`, and `mul_add_matrix`, with
  zero/one coefficient specialization. `mul_add_matrix` no longer falls back
  to the scalar kernel on wasm.

### Changed

- GF(2^16) scalar kernels (`Backend::Scalar` and every vector tail) multiply
  through the shared nibble tables instead of a per-element Karatsuba
  multiply: ~2.8x on `mul_add`/`mul_assign`/`mul_into`. The scalar scatter,
  gather, and matrix fallbacks route through the same path.

### Fixed

- wasm32 `simd128` test build, which referenced GF(2^8) scatter/gather/matrix
  kernels that did not exist. The comparison dev-dependency is now scoped to
  non-wasm targets so `cargo test --target wasm32-*` builds.

## [0.1.1] - 2026-07-29

### Added

- Prepared `Plan` consumers for scatter, gather, and matrix operations.
- Stable `pack`, `unpack`, and `pack_to_vec` element/buffer conversions.
- `Backend::ALL`, `Display`, `FromStr`, per-field `backend_for`, and
  `has_vector_elementwise` capability reporting.
- Uniform field element `Display`, assignment, iterator, byte-conversion,
  component, and raw-representation APIs.
- Fan–Paar level modules (`fan_paar::fp8` through `fp64`).
- Release metadata, CI, contributing guidance, public roadmap, benchmarks
  guide, and MIT license.

### Changed

- The public `FieldKernels` trait is sealed and re-exported from the crate root.
- The `Elem` trait remains at `fff::field::Elem` instead of colliding with
  concrete `Elem` types at the crate root.
- Row geometry and panic messages are consistent across vector operations.
- Internal scalar kernels, SIMD table layouts, and raw XOR dispatch are no
  longer public semver surface.
- x86 GF(2^8) kernels retain SIMD processing through 16-byte tails.
- Prepared GF(2^8) and GF(2^16) plans retain register-blocked x86 kernels
  where they outperform repeated prepared AXPY.

## [0.1.0] - 2026-07-29

Initial public release.

[Unreleased]: https://github.com/nanithefkuc/fff/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nanithefkuc/fff/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nanithefkuc/fff/releases/tag/v0.1.0
