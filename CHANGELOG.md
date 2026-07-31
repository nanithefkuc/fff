# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- crates.io publishing metadata and docs.rs configuration; the crate is
  distributed through git only.

### Added
- `Backend::Pmull`: `AArch64` NEON plus the optional PMULL extension, detected
  once at startup and selected ahead of `Backend::Neon`. It replaces a
  per-call `is_aarch64_feature_detected!("aes")` probe inside
  `mul_elementwise`, probes the accurate feature name (`pmull`, not the AES
  bundle), and makes `FFF_BACKEND=neon` able to turn PMULL off. Accepted by
  `FFF_BACKEND`, `Backend::ALL`, `Display` and `FromStr` like every other
  identifier.
- Fused GF(2^16) `mul_into` for NEON and wasm `simd128`; both backends used the
  default copy-then-scale before. Both fields' GF(2^16) kernels are
  compute-bound on those targets, so halving destination traffic is worth a few
  percent, not the ~2x it buys on bandwidth-bound x86.
- Two-lane (32-byte) blocks in the NEON and wasm `simd128` GF(2^16)
  `mul_add`/`mul_assign`/`mul_into` kernels: eight table lookups per lane leave
  enough latency to hide a second lane's loads and nibble splits. +11% at
  4 KiB … 8 MiB on a pinned Snapdragon 8 Gen 3 core; +13–25% under Node/V8.
  The same unroll measured 1.00x for the GF(2^8) wasm kernels and was not
  taken; the reason is recorded in the code.
- wasm `simd128` GF(2^16) `scatter`/`gather`/`matrix` are real kernels instead
  of per-row `mul_add` loops that rebuilt a coefficient's four nibble tables
  once per `(term, row)`: preparation is hoisted to one resolve per coefficient
  per call, zero and one coefficients skip it entirely, `gather` keeps a
  32-byte destination tile in registers across a block of eight sources, and
  `matrix` keeps a four-row tile across a block of eight terms. Rows ≥ 256 B:
  scatter ~2.1x, gather ~1.7x, matrix 1.3–1.6x under Node/V8.
- Non-temporal stores (`vmovntdq`) in the x86 `mul_into` kernels once the
  destination reaches 2 MiB: GF(2^8) on GFNI/AVX2/SSSE3 and GF(2^16) on
  GFNI/AVX2. `mul_into` never reads its destination, so an ordinary store still
  fetches every line it overwrites. Core Ultra 7 258V, Linux, rustc 1.93, one
  core, 8 MiB / 32 MiB: gf8 18.8 → 33.2 and 13.6 → 22.6 GiB/s (gfni),
  18.2 → 33.9 and 13.1 → 17.8 (avx2), 19.7 → 32.5 and 13.3 → 16.6 (ssse3);
  gf16 18.1 → 32.9 and 12.5 → 20.5 (gfni), 16.5 → 20.9 and 11.3 → 12.5 (avx2).
  Destinations below 2 MiB, `mul_add`, and `mul_assign` are unchanged: the
  threshold is where an encode-then-read-back loop stops losing from the
  eviction, and `mul_assign` reads its destination anyway (measured 22.0 GiB/s
  either way at 64 MiB, 0.5x at 256 KiB). GF(2^16) on SSSE3 keeps ordinary
  stores: at ~5.9 GiB/s it is shuffle-bound, and non-temporal stores cost it
  11%.
- `bench_large_destination` in `benches/kernels.rs`: `mul_into` at 1/8/32 MiB
  with an encode-then-read-back row and `mul_add` as the unaffected control —
  the shapes the non-temporal store threshold is set from.
- `bench_small_row_shapes` in `benches/kernels.rs`: GF(2^16) multi-row shapes at
  preparation-dominated row lengths, plus fused `mul_into` against
  copy-then-scale. `BENCHMARKS.md` documents the aarch64 (adb/cargo-ndk) and
  wasm32 (Node WASI) runner recipes and the variance rules those hosts need.
- Differential coverage for `mul_into` on every backend that implements it
  (`check_gf8_mul_into`, `check_gf16_mul_into_tables`); the shape had no direct
  kernel test before.
- `mul_elementwise` vector kernels for the shuffle-only x86 backends
  (AVX2, SSSE3) for GF(2^8) and GF(2^16). With both operands varying there is
  no nibble table to build, so the base-field product is the eight-round
  branchless shift/reduce sequence already used on NEON and wasm, ported with
  `PADDB`/`PCMPGTB` standing in for the byte shift x86 lacks; the GF(2^16)
  tower keeps a nibble table for the one constant (`DELTA`) multiply. Core
  Ultra 7 258V, Linux, rustc 1.93, 256 KiB: gf8 0.78 → 7.97 GiB/s (avx2) /
  2.62 GiB/s (ssse3), gf16 0.43 → 4.39 GiB/s (avx2) / 1.48 GiB/s (ssse3).
  `has_vector_elementwise::<Gf8/Gf16>()` now reports `true` on these backends.
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

- GF(2^16) `mul_elementwise` no longer uses PMULL on `AArch64`. The
  three-multiply tower identity over `vmull_p8` measured 0.88–0.92x against the
  bit-serial rounds on a Snapdragon 8 Gen 3, so that kernel was removed and the
  NEON path serves PMULL hosts too. GF(2^8) `mul_elementwise`, where PMULL
  replaces eight bit-serial rounds with two multiplies, keeps it at 1.55x.
- Fixed-coefficient PMULL kernels were written, measured, and **not** kept:
  0.13x (GF(2^8)) and 0.26x (GF(2^16)) against the split-nibble shuffle at every
  size from 4 KiB to 8 MiB — two `vmull_p8`s plus a twenty-instruction reduction
  network against five instructions of `vqtbl1q_u8`. The numbers now sit beside
  the surviving kernels in `src/kernel/aarch64/`.
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
