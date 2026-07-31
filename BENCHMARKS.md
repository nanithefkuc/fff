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

Non-x86 targets need a runner. The `aarch64` and `wasm32` kernels were
measured with these:

```sh
# aarch64, on an adb-connected arm64 device (cargo-ndk):
cargo ndk -t arm64-v8a build --release --bench kernels
adb push target/aarch64-linux-android/release/deps/kernels-<hash> /data/local/tmp/
adb shell 'cd /data/local/tmp && taskset 80 ./kernels-<hash>'

# wasm32, under Node's WASI shim:
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release --target wasm32-wasip1 --bench kernels
node wasi-run.mjs target/wasm32-wasip1/release/deps/kernels-<hash>.wasm
```

`wasi-run.mjs` is the whole shim (the same one works as
`CARGO_TARGET_WASM32_WASIP1_RUNNER` for `cargo test --target wasm32-wasip1`):

```js
import { WASI } from "node:wasi";
import { argv, env, exit } from "node:process";
import { readFile } from "node:fs/promises";

const [wasmPath, ...rest] = argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [wasmPath, ...rest], env,
                        preopens: { "/": "/" }, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(wasmPath));
exit(wasi.start(await WebAssembly.instantiate(module, wasi.getImportObject())));
```

Both runners are far noisier than a pinned desktop CPU, so a single pair of
runs proves nothing:

- On a phone, `taskset` to **one** big core, not the cluster: migration
  between cores produces a bimodal distribution (two clean clusters ~1.5x
  apart), and a whole run can land in either mode.
- Under Node, tiering and GC shift results by up to 40% run to run.
- Interleave the two binaries (`base, new, base, new, …`), take the **maximum**
  of at least three runs per key, and always include an unchanged operation as
  a control. Every number below was accepted only with its control at 1.00x.

Record the CPU, operating system, Rust version, selected backend, row size, row
count, and source count with any quoted result. `backend_for::<F>()` matters
for the wider towers and Fan–Paar family: GF(2^32)/GF(2^64) use GFNI on
x86 GFNI hosts, the canonical Fan–Paar GF(2^16)/32/64 use AVX2/SSSE3 on x86,
but FanPaar8, SSSE3 for the wider Fan–Paar fields, and every shuffle-only
backend still report the portable kernels even when the process-wide backend
is `avx512` or `gfni`.

## Interpreting the shapes

- `mul_add` is the single-row AXPY baseline.
- `mul_add_scatter` tests source-load sharing across destination rows.
- `mul_add_gather` and `mul_add_matrix` test whether destination tiles stay in
  registers across sources.
- `_with` operations separate coefficient preparation from byte-loop cost.
- `mul_elementwise` has no broadcast coefficient. It vectorizes on
  AVX-512/GFNI (native `GF2P8MULB`) and, via a branchless shift/reduce vector
  multiply, on AVX2/SSSE3, NEON, and Wasm `simd128`. Measured on a Core Ultra
  7 258V (Linux, rustc 1.93), 256 KiB buffers: gf8 0.78 → 7.97 GiB/s (AVX2)
  and 2.62 GiB/s (SSSE3); gf16 0.43 → 4.39 GiB/s (AVX2) and 1.48 GiB/s
  (SSSE3). The wider fields still use the scalar reference.

Small GF(2^16) rows are sensitive to coefficient preparation because a shuffle
backend builds four nibble tables per coefficient. Use `Coeff` or `Plan` when a
coding matrix is reused. Large rows amortize the same setup in the byte loop.
GF(2^8) preparation is free at every size — its 256-entry table bank is static
— and GFNI preparation is two broadcast words, so the effect is a shuffle-
backend GF(2^16) concern only.

`bench_preparation_crossover` in `benches/kernels.rs` measures exactly where
that stops mattering, one-shot against prepared over 16 B … 64 KiB rows. Run
it rather than quoting a remembered threshold; on a hybrid CPU pin the process
(`taskset -c <p-core>`) or the two core types will produce two answers.

`bench_blocked_vs_axpy` measures the blocked multi-row GF(2^16) kernels
against repeated single-row AXPY by calling both directly, bypassing dispatch.
It needs the `internals` feature (`cargo bench --bench kernels --features
internals`) and is the harness to rerun before changing which shape a backend
dispatches to.

`bench_small_row_shapes` measures the GF(2^16) multi-row shapes at row lengths
where a coefficient's four nibble tables are still a visible share of the work,
with a coefficient set that is one-fifth zeros and one-fifth ones so the
zero/one specializations are exercised. It also times fused `mul_into` against
the copy-then-scale it replaces.

`bench_large_destination` measures `mul_into` at 1, 8 and 32 MiB. Past 2 MiB
the x86 kernels store the destination non-temporally, which skips the
read-for-ownership fetch of lines they overwrite whole; the cost is that the
destination is not left cached, so the section also times an
encode-then-read-back loop, and `mul_add` as a control the change cannot touch.
Rerun it before moving that threshold, and pin the process: the effect is
entirely memory-side, so an unpinned run on a hybrid CPU reports the wrong
size at which it starts paying.

## aarch64 and wasm32 kernel numbers (2026-07-31)

Snapdragon 8 Gen 3 (`arm64-v8a`, Android, rustc 1.93, backend `neon`), one big
core, maximum of four interleaved runs, GF(2^8) buffers as control (1.00x):

| Shape | Before | After |
| --- | --- | --- |
| gf16 `mul_add`, 4 KiB … 8 MiB | 2.09–2.17 GiB/s | 2.30–2.41 GiB/s (+11%) |
| gf16 `mul_assign`, ≤ 256 KiB | 2.30–2.35 GiB/s | 2.39–2.45 GiB/s (+4%) |
| gf16 `mul_into`, 16 KiB rows | copy-then-scale | fused, +6% over copy+scale |

The gf16 two-lane unroll is what moves `mul_add`; it wins at every size from
4 KiB to 8 MiB on a pinned core. Measured on the *cluster* instead, the same
binary reports anything from 0.65x to 1.11x — that spread is core migration,
not the kernel.

### PMULL against the nibble shuffle

`Backend::Pmull` and `Backend::Neon` differ by dispatch alone, so
`FFF_BACKEND=pmull` against `FFF_BACKEND=neon` in **one binary** is the
cleanest A/B this crate has. Same device, one pinned big core, max of three
runs each:

| Shape | `neon` | `pmull` | |
| --- | --- | --- | --- |
| gf8 `mul_elementwise`, 4 KiB … 8 MiB | 0.83 GiB/s | 1.28–1.30 GiB/s | **1.55x** |
| gf16 `mul_elementwise`, 4 KiB … 8 MiB | 0.39–0.41 GiB/s | 0.36 GiB/s | 0.88–0.92x |
| gf8 `mul_add` fixed coefficient | 9.0–9.2 GiB/s | 1.17–1.23 GiB/s | 0.13x |
| gf16 `mul_add` fixed coefficient | 2.4 GiB/s | 0.62 GiB/s | 0.26x |

Only the first line survived into dispatch. `PMULL` pays when it replaces
eight bit-serial rounds; against `vqtbl1q_u8`'s five instructions it cannot
carry its twenty-instruction reduction network, and against the tower's
three-multiply identity it merely ties. The losing kernels were deleted, with
the numbers kept in `src/kernel/aarch64/`.

Everything else in the run matches within 3% — except rows of 16–128 B, where
dispatch-identical code varied by up to 1.36x between the two runs. Treat
anything measured on rows that small as noise unless it survives many runs.

Node 26 WASI (`wasm32-wasip1`, `simd128`, rustc 1.93), maximum of four
interleaved runs, GF(2^8) `xor`/`elementwise` as control (1.00x):

| Shape | Before | After |
| --- | --- | --- |
| gf16 `mul_add`, 4 KiB … 8 MiB | 4.5–5.4 GiB/s | 5.6–6.1 GiB/s (+13–24%) |
| gf16 `mul_assign`, 4 KiB … 8 MiB | 5.0–5.2 GiB/s | 6.0–6.2 GiB/s (+14–25%) |
| gf16 `scatter`, 16 rows, ≥ 256 B | 4.4–5.1 GiB/s | 9.8–11.1 GiB/s (~2.1x) |
| gf16 `gather`, 8 sources, ≥ 256 B | 4.5–5.4 GiB/s | 7.5–9.1 GiB/s (~1.7x) |
| gf16 `matrix`, 16x8, ≥ 256 B | 4.8–5.0 GiB/s | 6.2–7.9 GiB/s (1.3–1.6x) |

The multi-row factors come from hoisting the four-table derivation out of the
per-`(term, row)` loop and skipping zero/one coefficients; at 64 B rows the
result is inside Node's noise band. A two-lane unroll of the *GF(2^8)* wasm
kernels measured 1.00x and was therefore not kept: two swizzles per lane leave
no latency to hide, unlike GF(2^16)'s eight.

## Comparative benchmark

`benches/compare.rs` compares compatible GF(2^8) operations against
`reed-solomon-erasure` with its `simd-accel` feature. It is a development
comparison, not a claim that the crates expose identical abstractions. Run it
on the same machine and toolchain before quoting a ratio.

Historical measurements from local development are intentionally not copied
into the crate landing page: results without their original CPU and command
line are not reproducible evidence.
