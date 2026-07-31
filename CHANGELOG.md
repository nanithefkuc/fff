# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-07-31

This release adds vector kernels for the fields that had none, widens the
shapes the existing kernels cover, and tunes x86 store and alignment
behaviour. Every performance claim, the hardware it was measured on, and the
experiments that were rejected are in [BENCHMARKS.md](BENCHMARKS.md); numbers
are deliberately not repeated here.

### Added

- Vector kernels for GF(2^32) and GF(2^64) on x86 GFNI, replacing the scalar
  path for `mul_add`, `mul_assign`, and `mul_into`.
- Vector kernels for the canonical Fan–Paar fields on x86: `FanPaar16` on AVX2
  and SSSE3, `FanPaar32` and `FanPaar64` on AVX2. `FanPaar8` keeps the portable
  path, which is already table-driven.
- `mul_elementwise` vector kernels for GF(2^8) and GF(2^16) on AVX2 and SSSE3.
  `has_vector_elementwise` now reports `true` for those fields on those
  backends.
- `Backend::Pmull`, the AArch64 NEON plus PMULL combination. It replaces a
  per-call feature probe inside `mul_elementwise`, so the capability is
  detected once per process and `FFF_BACKEND=neon` can turn it off. Accepted by
  `FFF_BACKEND`, `Backend::ALL`, `Display`, and `FromStr` like every other
  backend.
- Multi-row GF(2^8) kernels (`mul_add_scatter`, `mul_add_gather`,
  `mul_add_matrix`) and dedicated GF(2^16) scatter, gather, and matrix kernels
  for WebAssembly `simd128`. These shapes previously fell back to per-row or
  scalar loops.
- Fused GF(2^16) `mul_into` for NEON and WebAssembly `simd128`, which used
  copy-then-scale before.
- `internals` feature, exposing the kernel modules and coefficient preparation
  types for downstream crates that build directly on the kernels. It is not a
  stable semver surface.
- Differential kernel coverage for `mul_into` on every backend that implements
  it; the shape had no direct test before.
- Benchmark sections for the shapes the new tuning policies are set from:
  large destinations, destination alignment, small multi-row shapes,
  preparation crossover, and blocked kernels against repeated AXPY.
  `BENCHMARKS.md` documents the AArch64 and WebAssembly runner recipes.

### Changed

- **Breaking:** `Backend::ALL` is `[Backend; 8]`, up from `[Backend; 7]`.
- **Breaking:** AArch64 hosts with the PMULL extension now report
  `Backend::Pmull` from `backend()` rather than `Backend::Neon`.
- **Breaking:** `has_vector_elementwise::<Gf8>()` and `::<Gf16>()` return
  `true` on AVX2 and SSSE3 hosts, where they previously returned `false`.
- GF(2^16) scalar kernels, including the tail of every vector kernel, multiply
  through the shared nibble tables instead of a per-element Karatsuba multiply.
- x86 `mul_into` uses non-temporal stores once the destination is large enough
  that an ordinary store would evict more than it saves. `mul_add` and
  `mul_assign` read their destination and are unchanged.
- The GF(2^16) multi-row scatter kernels on GFNI and AVX2 peel a misaligned
  destination before the main pass, generalizing the existing GF(2^8) peel.
- The NEON and WebAssembly `simd128` GF(2^16) kernels process two lanes per
  iteration.
- GF(2^16) `mul_elementwise` no longer uses PMULL; the NEON path serves PMULL
  hosts. GF(2^8) `mul_elementwise` keeps it.
- GF(2^16) GFNI `mul_add_gather` and GF(2^8) AVX2 `mul_add_matrix` use their
  register-blocked kernels instead of dispatching to repeated AXPY.
- GF(2^16) 256-bit kernels descend through a 128-bit vector before the scalar
  tail, so short remainders are no longer fully scalar.

### Fixed

- WebAssembly `simd128` test build, which referenced GF(2^8) multi-row kernels
  that did not exist. The comparison dev-dependency is now scoped to non-wasm
  targets so `cargo test --target wasm32-*` builds.
- `--no-default-features --all-targets` builds. The bench targets use
  `ops::Plan` and the comparison dev-dependency, both of which need `std`, so
  they now declare `required-features = ["std"]`; a Fan–Paar test import that
  is only reachable on x86 SIMD builds is cfg-gated to match.

### Removed

- crates.io publishing metadata and docs.rs configuration; the crate is
  distributed through git only.

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

[Unreleased]: https://github.com/nanithefkuc/fff/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nanithefkuc/fff/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nanithefkuc/fff/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nanithefkuc/fff/releases/tag/v0.1.0
