# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- crates.io publishing metadata and docs.rs configuration; the crate is
  distributed through git only.

### Added
- GF(2^32) and GF(2^64) GFNI kernels for x86: `mul_add`, `mul_assign`, and
  `mul_into` as the tower identity one and two levels above the GF(2^16)
  kernel — two (Gf32) or four (Gf64) GF(2^16) lane multiplies, each two
  `GF2P8MULB`, folded into 4-/8-byte periodic broadcasts. On a 256 KiB
  `mul_add` (Core Ultra 7 258V, Linux, rustc 1.93, gfni) this is ~63× (Gf32,
  0.22 → 13.9 GiB/s) and ~56× (Gf64, 0.12 → 6.85 GiB/s) over the scalar
  Karatsuba. Shuffle backends, NEON, wasm, and `mul_elementwise` keep the
  portable path. A representation-agnostic `Tower2Coeff` carries the period-2
  subfield coefficient pair for the GF(p) effort to reuse.
- Fan–Paar `mul_add`/`mul_assign`/`mul_into` kernels for x86, all levels that
  leave the portable path: GF(2^16) on AVX2/SSSE3 (the four-nibble-shuffle
  tower over a new `fp8` nibble bank — fp8 is not the AES field, so no GFNI
  fast path) and GF(2^32)/GF(2^64) on AVX2 (period-2 fp16/fp32 lane muls).
  The `mul_alpha` recurrence folds into coefficient preparation by subfield
  commutativity (`alpha·(c1·x1) = (alpha·c1)·x1`), so the kernel is purely two
  alternating subfield lane muls — no novel in-kernel `mul_alpha` network. A
  shared `NibbleFactors` trait reuses the tested `scale_avx2`/`scale_ssse3`
  core. On a 256 KiB `mul_add` (Core Ultra 7 258V, Linux, rustc 1.93, gfni
  dispatching AVX2): FanPaar16 ~0.11 → 7.6 GiB/s, FanPaar32 ~0.08 → 2.9
  GiB/s, FanPaar64 ~0.05 → 1.1 GiB/s (order-of-magnitude over the portable
  path; numbers vary on this hybrid CPU without core pinning). FanPaar8
  (already fast via the log table), SSSE3 for 32/64, and NEON/wasm keep the
  portable path.
- `internals` feature exposing the kernel modules and preparation types
  (`ScaleTable`, `TowerCoeff`, `TowerTables`, backend-specific SIMD entry
  points) for downstream libraries that build directly on the kernels.
- wasm32 `simd128` GF(2^8) multi-row kernels: register-blocked
  `mul_add_scatter`, `mul_add_gather`, and `mul_add_matrix`, with
  zero/one coefficient specialization. `mul_add_matrix` no longer falls back
  to the scalar kernel on wasm.
- `bench_preparation_crossover` and `bench_blocked_vs_axpy` sections in
  `benches/kernels.rs`; the latter needs the `internals` feature and measures
  blocked multi-row kernels against repeated AXPY with dispatch bypassed.

### Changed

- GF(2^16) scalar kernels (`Backend::Scalar` and every vector tail) multiply
  through the shared nibble tables instead of a per-element Karatsuba
  multiply: ~2.8x on `mul_add`/`mul_assign`/`mul_into`. The scalar scatter,
  gather, and matrix fallbacks route through the same path.
- GF(2^16) GFNI `mul_add_gather` now uses the blocked kernel, which derives
  its broadcast factors once per four-source group instead of inside the byte
  loop: measured 1.03–1.59x over the previous repeated-AXPY dispatch.
- GF(2^8) AVX2 `mul_add_matrix` shares each source's nibble split across the
  rows of a group and amortizes table broadcasts over a 64-byte tile: +20%.
- GF(2^16) 256-bit kernels descend through one 128-bit vector before the
  scalar tail, so sub-32-byte remainders are no longer fully scalar.

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
