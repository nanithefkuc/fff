# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

## [0.1.0] - Unreleased

Initial public release.

[Unreleased]: https://github.com/nanithefkuc/fff/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nanithefkuc/fff/releases/tag/v0.1.0
