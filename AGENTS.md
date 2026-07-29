# Repository Guidelines

## Project Overview

`fff` is a dependency-free Rust library for binary finite-field arithmetic. It
provides const-capable scalar elements and safe, runtime-dispatched kernels over
packed byte buffers for erasure coders, proof systems, and similar consumers.
It is deliberately not a codec: matrix recipes, shard ownership, inversion,
and streaming recovery belong above this crate.

The public fields are `Gf8`, `Gf16`, `Gf32`, `Gf64`, and canonical Fan–Paar
`FanPaar8/16/32/64`. Only `Gf8` and `Gf16` have hand-written SIMD kernels;
wider and Fan–Paar fields use the portable implementation.

## Architecture & Data Flow

1. `src/field/` defines scalar algebra and stable representations through
   `Elem` and zero-sized `Field` markers. Concrete inherent arithmetic remains
   `const` where possible.
2. Callers pass packed, little-endian `&[u8]`/`&mut [u8]` buffers to
   `src/ops.rs`. This safe façade validates element widths, paired lengths,
   coefficient counts, and row geometry.
3. `Coeff<F>` or std-only `Plan<F>` may prepare backend-specific coefficients
   before repeated work. `_with` operations consume prepared forms.
4. The sealed `FieldKernels` contract selects field-specific dispatch.
   `backend()` is process-wide; use `backend_for::<F>()` when field support
   matters because wider fields remain scalar.
5. Safe crate-private wrappers under `src/kernel/{x86,aarch64,wasm32}/` enter
   target-feature-specific unsafe intrinsics, process full lanes/blocked tiles,
   then use portable element-aligned tails.

Preserve these invariants:

- Encodings are stable, fixed-width, little-endian, and alignment-free.
- Addition and subtraction are XOR. By convention `inv(0) == 0` and
  `x / 0 == 0`; do not turn these total operations into errors.
- Backend selection is cached once and `FFF_BACKEND` is downgrade-only.
- Coefficients `0` and `1` are handled before expensive preparation.
- Unsafe code stays inside architecture modules. Root code denies unsafe.
- Raw multi-row kernels may be register-blocked, but not every
  `(field, backend, shape)` is; preserve documented measured crossovers.

There is no async runtime, dependency injection, service container, or mutable
application-state framework. Field selection is static generic dispatch; the
only process-global state is immutable-after-initialization backend selection
and immutable lookup data.

## Key Directories

- `src/field/` — `Elem`/`Field` contracts, concrete fields, constants,
  conversions, and scalar algebra.
- `src/kernel/` — sealed dispatch contract, portable oracle/fallback,
  coefficient tables, per-field routing, and backend reporting.
- `src/kernel/{x86,aarch64,wasm32}/` — crate-private intrinsic implementations
  and the only allowed unsafe boundary.
- `tests/` — public integration tests: `algebra.rs` for field laws and
  `ops.rs` for checked/dispatched buffer operations.
- `benches/` — custom throughput binaries, not Criterion harnesses.
- `external-bench/` — ignored, Linux/x86-oriented comparisons against external
  libraries; dependencies must be installed or built separately.
- `.github/workflows/` — CI policy and platform/toolchain matrix.

## Development Commands

```sh
# Main feature matrix
cargo test --all-features
cargo test --no-default-features

# Focused suites
cargo test --test algebra
cargo test --test ops
cargo test --lib kernel::tests -- --nocapture
cargo test --doc

# Required backend processes on capable x86 hardware
FFF_BACKEND=avx2 cargo test --all-features
FFF_BACKEND=ssse3 cargo test --all-features
FFF_BACKEND=scalar cargo test --all-features

# Formatting, linting, and docs
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Cross-builds and MSRV
cargo build --target aarch64-unknown-linux-gnu
cargo build --target wasm32-unknown-unknown
cargo +1.89.0 build --all-features

# Benchmarks
cargo bench --bench kernels
cargo bench --bench compare
```

Nightly scalar-path Miri: `cargo miri test --no-default-features`.
External comparisons use `sh external-bench/run.sh [cpu]`; read the script
first because it assumes system libraries, local prefixes, GNU/Linux tools,
and host-specific compiler flags.

## Code Conventions & Common Patterns

- Use rustfmt defaults. The crate denies missing docs and unsafe code, warns on
  Clippy pedantic, and has only narrow lint allowances in `src/lib.rs`.
- Naming: lowercase field module, CamelCase zero-sized marker, module-local
  `Elem`; conversions are `from_raw`/`to_raw`, `from_bytes`/`to_bytes`, and
  tower `from_components`/`components`. Algebraic constants are uppercase.
- Extend `define_fan_paar_level!` and `impl_field_kernels!` instead of creating
  parallel repetition for sibling fields.
- Keep inherent scalar APIs `const`, `#[must_use]`, and small/hot wrappers
  `#[inline]` where the surrounding code does. Const algorithms use fixed
  arrays and explicit loops rather than allocation.
- Public buffer misuse is a programmer error: validate in `ops`, then panic
  with operation-specific, operand-naming messages. Use `checked_mul` for row
  geometry. Query APIs such as `Plan::get`, `get_at`, and `row` return `Option`.
- Internal kernels rely on `ops` validation and normally use `debug_assert!`.
  A new unsafe function needs a `# Safety` contract; each call needs an
  adjacent `// SAFETY:` explanation naming the dispatch/geometry proof.
- Avoid allocations and copies in hot paths. Prepare outside repeated short
  operations, borrow `CoeffRef`/`Plan` instead of cloning large GF16 tables,
  use fused/wide operation shapes when they preserve destination traffic, and
  leave measured crossover comments intact.
- Do not serialize prepared coefficients; they are tied to the process backend.
  Serialize the scalar element and rebuild preparation.
- `Backend` declaration order encodes capability and downgrade comparison;
  reordering variants is a behavioral and safety change.

## Important Files

- `Cargo.toml` — package metadata, Rust 2024/MSRV 1.89, `std`/`simd` feature
  graph, bench profile and targets. Publishing is disabled (`publish =
  false`); the crate is git-only.
- `src/lib.rs` — crate scope, lint/safety policy, public modules and re-exports.
- `src/field/mod.rs` — canonical scalar/marker contracts and byte invariant.
- `src/ops.rs` — validated public operations, `Coeff`, `Plan`, packing helpers.
- `src/kernel/mod.rs` — `Backend`, cached downgrade override, sealed kernels.
- `src/kernel/scalar.rs` — correctness oracle, universal fallback, SIMD tails.
- `src/kernel/tables.rs` — static GF8 table bank and GF16 prepared layouts.
- `tests/algebra.rs`, `tests/ops.rs`, `src/kernel/tests.rs` — three QA layers.
- `README.md` — user-facing supported fields, operations, platforms, and scope.
- `CONTRIBUTING.md` — unsafe and differential-testing policies.
- `BENCHMARKS.md` — benchmark interpretation and reproducibility requirements.
- `ROADMAP.md` — future work, not current API contract.
- `CHANGELOG.md` — notable user-facing changes; keep unreleased entries current.
- `.github/workflows/ci.yml` — authoritative automated command matrix.

## Runtime/Tooling Preferences

- Use Cargo and Rust; there is no Node/Bun/package-manager workflow.
- Edition: Rust 2024. MSRV: Rust 1.89. No root toolchain pin exists, so select
  `+1.89.0`, stable, or nightly explicitly when the check requires it.
- Default features are `std` + `simd`; `simd` implies `std`.
  `--no-default-features` is the portable `no_std` configuration.
- Normal builds have no third-party dependencies. The sole direct
  dev-dependency is `reed-solomon-erasure` with `simd-accel` for comparison.
- No custom rustfmt/clippy config or Cargo aliases exist; use the commands above.
- `FFF_BACKEND` requests can be ignored when unavailable. A green forced run is
  not proof that an incapable host executed that ISA; inspect reported backend
  or direct-kernel skip output.
- The committed `Cargo.lock` is Cargo-generated; do not edit it manually.
- `/target` and `/external-bench` are ignored. Crate packaging also excludes
  `/external-bench` and `/.github`.

## Testing & QA

Tests use the built-in Rust harness—no property, snapshot, async, or fixture
framework. Reuse existing deterministic helpers and independent oracles:

- `tests/algebra.rs` exhaustively checks GF8 and deterministically samples wider
  fields against independent shift/XOR or schoolbook formulations.
- `tests/ops.rs` compares public operations with elementwise/repeated-operation
  oracles, including prepared plans, packing, geometry panics, surplus rows,
  zero/one shortcuts, empty buffers, and erasure recovery.
- `src/kernel/tests.rs` bypasses dispatch and differentially compares runnable
  architecture kernels with `kernel::scalar` across lane tails, row blocks,
  source counts, and coefficients. Unsupported hardware prints a skip and
  returns; tests are not marked ignored.

Use the existing fixed-seed `noise(len, seed)` LCG and shared boundary arrays;
do not add nondeterministic randomness or a second fixture convention. Include
coefficients `0`, `1`, mixed/component-specific values, and maxima. GF16 byte
lengths must remain even. Panic tests use `#[should_panic(expected = "stable
message fragment")]`.

When changing behavior:

- New field: implement `Elem` + `Field`, wire `FieldKernels`, add an independent
  algebra oracle/laws/encoding/generator coverage, extend public ops coverage,
  and update the field documentation table.
- New operation: add a public compositional oracle, empty/zero/one cases,
  invalid geometry, lane/unroll/tail boundaries, and scalar differentials for
  every optimized backend.
- New backend: add cfg + runtime feature guards, invoke all shared `check_*`
  drivers and XOR checks, then run a separate forced public-dispatch process on
  hardware that can genuinely select it.
- Prepared-path change: compare `_with` byte-for-byte with one-shot operations;
  preserve plan dimensions, iteration/index access, and scatter/gather/matrix
  equivalence.

Benchmarks are not CI correctness checks. When quoting results, record CPU, OS,
Rust version, actual process/per-field backend, row size/count, and source
count. Do not reuse historical numbers from `external-bench/PLAN.md` without
rerunning; that file is an experiment log with stale and superseded sections.
