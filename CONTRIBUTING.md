# Contributing

## Ground rules for this crate

`fff` is a small crate with an unusually large unsafe surface. Two rules follow
from that and are not negotiable:

1. **Every backend is differentially tested against the portable reference.**
   `src/kernel/scalar.rs` is the oracle. A new kernel is not done until
   `src/kernel/tests.rs` exercises it past dispatch, over lane boundaries, zero
   and one coefficients, and odd geometry.
2. **All `unsafe` lives in `src/kernel/{x86,aarch64,wasm32}`.** These modules
   are `pub(crate)`; nothing above them may be unsafe. Each `unsafe fn` carries
   a `# Safety` section naming the target features it requires, and each call
   site carries a `// SAFETY:` comment naming the dispatch arm that guarantees
   them.

## Running the tests

```sh
cargo test --all-features
cargo test --no-default-features   # portable kernels only
```

Backend dispatch resolves once per process, so one run only covers the host's
best backend. Sweep the weaker ones explicitly:

```sh
FFF_BACKEND=avx2   cargo test
FFF_BACKEND=ssse3  cargo test
FFF_BACKEND=scalar cargo test
```

`FFF_BACKEND` is downgrade-only: asking for a backend the host cannot execute
is ignored rather than faked.

## Benchmarks

```sh
cargo bench --bench kernels    # this crate's kernel shapes
cargo bench --bench compare    # against reed-solomon-erasure
```

Both print throughput tables rather than using a harness. Record the CPU when
quoting a number; see `BENCHMARKS.md` for the existing measurements and the
methodology they were taken under.

## Before opening a PR

```sh
cargo fmt
cargo clippy --all-features --all-targets
cargo doc --all-features --no-deps
```

The crate denies `missing_docs` and warns on `clippy::pedantic`, so new public
items need doc comments. MSRV is 1.89 and is checked in CI; do not reach for
newer standard-library APIs without raising it deliberately.

## Adding a field

A new field needs a `Field` + `Elem` impl in `src/field/`, a `FieldKernels`
impl in `src/kernel/`, and coverage in `tests/algebra.rs`. If it has no
hand-written backend, wire it to the portable kernels with
`impl_field_kernels!` and say so in the crate docs — the field table records
which families have SIMD backends and which do not.
