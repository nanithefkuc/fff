# Benchmarks

`fff` uses small custom benchmark binaries rather than a statistical harness.
They print throughput directly so operation shape, row size, and backend remain
visible beside each result.

## Reproduce

```sh
cargo bench --bench kernels
cargo bench --bench compare
```

Pin a weaker backend to measure a dispatch crossover:

```sh
FFF_BACKEND=avx2  cargo bench --bench kernels
FFF_BACKEND=ssse3 cargo bench --bench kernels
FFF_BACKEND=scalar cargo bench --bench kernels
```

Record the CPU, operating system, Rust version, selected backend, row size, row
count, and source count with any quoted result. `backend_for::<F>()` matters for
wider and Fan–Paar fields: they currently use the portable kernels even when
the process-wide backend is `avx512` or `gfni`.

## Interpreting the shapes

- `mul_add` is the single-row AXPY baseline.
- `mul_add_scatter` tests source-load sharing across destination rows.
- `mul_add_gather` and `mul_add_matrix` test whether destination tiles stay in
  registers across sources.
- `_with` operations separate coefficient preparation from byte-loop cost.
- `mul_elementwise` has no broadcast coefficient. It vectorizes on
  AVX-512/GFNI, NEON, and Wasm `simd128`; AVX2/SSSE3 use the scalar reference.

Small GF(2^16) rows are sensitive to coefficient preparation because a shuffle
backend builds four nibble tables per coefficient. Use `Coeff` or `Plan` when a
coding matrix is reused. Large rows amortize the same setup in the byte loop.

## Comparative benchmark

`benches/compare.rs` compares compatible GF(2^8) operations against
`reed-solomon-erasure` with its `simd-accel` feature. It is a development
comparison, not a claim that the crates expose identical abstractions. Run it
on the same machine and toolchain before quoting a ratio.

Historical measurements from local development are intentionally not copied
into the crate landing page: results without their original CPU and command
line are not reproducible evidence.
