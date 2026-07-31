# fff — Faster Finite Fields

`fff` provides scalar arithmetic and runtime-dispatched vector kernels for
binary finite fields. It is the arithmetic layer underneath erasure coders,
proof systems, and other applications that operate on packed field elements;
it is deliberately not a codec.

- Stable little-endian byte encodings.
- Portable `no_std` scalar implementation.
- SIMD kernels for GF(2^8) and GF(2^16) on every SIMD backend; GF(2^32) and
  GF(2^64) on x86 GFNI, with the same tower identity one and two levels up.
- Const-callable scalar arithmetic on every concrete element type.
- No dependencies in normal builds.

## Install

`fff` is distributed through git only; it is not published to crates.io.

```toml
[dependencies]
fff = { git = "https://github.com/nanithefkuc/fff" }
```

The default enables `std` and SIMD dispatch. Portable `no_std` builds use:

```toml
[dependencies]
fff = { git = "https://github.com/nanithefkuc/fff", default-features = false }
```

MSRV is Rust 1.89.

## Fields

| Field | Marker | Element | Construction | Vector backend |
| --- | --- | --- | --- | --- |
| GF(2^8) | `Gf8` | `gf8::Elem` | AES polynomial `0x11B` | SIMD |
| GF(2^16) | `Gf16` | `gf16::Elem` | quadratic tower over `Gf8` | SIMD |
| GF(2^32) | `Gf32` | `gf32::Elem` | quadratic tower over `Gf16` | GFNI x86 |
| GF(2^64) | `Gf64` | `gf64::Elem` | quadratic tower over `Gf32` | GFNI x86 |
| Fan–Paar GF(2^8) | `FanPaar8` | `fan_paar::fp8::Elem` | canonical recursive tower | portable |
| Fan–Paar GF(2^16) | `FanPaar16` | `fan_paar::fp16::Elem` | canonical recursive tower | x86 AVX2/SSSE3 |
| Fan–Paar GF(2^32) | `FanPaar32` | `fan_paar::fp32::Elem` | canonical recursive tower | x86 AVX2 |
| Fan–Paar GF(2^64) | `FanPaar64` | `fan_paar::fp64::Elem` | canonical recursive tower | x86 AVX2 |

## Scalar arithmetic

Concrete methods are `const fn`, so coefficients and coding matrices can be
built at compile time. Import `fff::field::Elem` when generic code needs the
trait methods in scope.

```rust
use fff::gf16;

const A: gf16::Elem = gf16::Elem(0x1234);
const B: gf16::Elem = gf16::Elem(0x0108);
const PRODUCT: gf16::Elem = A.mul(B);

assert_eq!(PRODUCT.div(B), A);
assert_eq!(A + B, A.sub(B)); // characteristic two
```

By library-wide convention `inv(0) == 0` and `x / 0 == 0`. This makes the
scalar contract total; it does not claim zero is mathematically invertible.

All element families implement `Add`, `Sub`, `Mul`, `Div`, their assignment
forms, `Sum`, `Product`, `Display`, and representation-order `Ord`. `Ord` is
for map keys only and has no field-theoretic meaning.

## Packed vector operations

Buffers contain consecutive stable little-endian element encodings. Their
length must be a multiple of `F::BYTES`.

```rust
use fff::{Gf8, gf8, ops};

let src = [0x01u8, 0x02, 0x03, 0x04];
let mut dst = [0u8; 4];

ops::mul_add::<Gf8>(&mut dst, gf8::Elem(0x03), &src);
assert_eq!(dst, [0x03, 0x06, 0x05, 0x0c]);
```

| Shape | Function | Typical use |
| --- | --- | --- |
| `dst ^= src` | `add_assign` / `sub_assign` | XOR parity |
| `dst ^= c * src` | `mul_add` / `mul_add_with` | AXPY |
| `dst = c * src` | `mul_into` / `mul_into_with` | scale a row |
| `dst *= c` | `mul_assign` / `mul_assign_with` | in-place scale |
| one source, many rows | `mul_add_scatter` / `mul_add_scatter_with` | systematic encode |
| many sources, one row | `mul_add_gather` / `mul_add_gather_with` | recover one symbol |
| many sources, many rows | `mul_add_matrix` / `mul_add_matrix_with` | reconstruction |
| varying pair per lane | `mul_elementwise` | pointwise products |

Prefer the widest shape that matches the operation. The blocked kernels can
retain destination tiles in registers across sources.

### Reusing coefficients

`Coeff<F>` prepares one coefficient. With `std`, `Plan<F>` prepares a vector or
row-major matrix once and drives every multi-row `_with` operation directly.
A matrix plan has dimensions `(sources, destination_rows)`:

```rust
use fff::{Gf16, gf16, ops};

let coeffs = [
    gf16::Elem(1), gf16::Elem(2),
    gf16::Elem(3), gf16::Elem(4),
];
let plan = ops::Plan::<Gf16>::matrix(2, 2, &coeffs);
let a = [1u8, 0, 2, 0];
let b = [3u8, 0, 4, 0];
let sources = [&a[..], &b[..]];
let mut rows = [0u8; 8];

ops::mul_add_matrix_with(&mut rows, 4, 2, &plan, &sources);
```

Use `ops::pack`, `ops::unpack`, or `ops::pack_to_vec` at element/buffer
boundaries instead of writing chunk loops by hand.

## Backends

`backend()` reports the process-wide SIMD selection. `backend_for::<F>()`
reports what a particular field actually uses; the GF(2^32)/GF(2^64) towers
report the GFNI backend on x86 GFNI hosts, the canonical Fan–Paar GF(2^16)/
GF(2^32)/GF(2^64) report AVX2/SSSE3 on x86, and the remaining fields report
`scalar`. `has_vector_elementwise::<F>()` exposes the notable performance
boundary of `mul_elementwise`.

| Identifier | Target and requirements | Lane width |
| --- | --- | --- |
| `avx512` | x86 AVX-512F + AVX-512BW + GFNI | 64 bytes |
| `gfni` | x86 AVX2 + GFNI | 32 bytes |
| `avx2` | x86 AVX2 shuffle | 32 bytes |
| `ssse3` | x86 SSSE3 shuffle | 16 bytes |
| `pmull` | AArch64 NEON + PMULL; adds a vector `mul_elementwise` for GF(2^8) | 16 bytes |
| `neon` | AArch64 NEON split-nibble shuffle | 16 bytes |
| `simd128` | WebAssembly `simd128` | 16 bytes |
| `scalar` | portable fallback | scalar |

`FFF_BACKEND=avx512|gfni|avx2|ssse3|pmull|neon|simd128|scalar` requests a backend at
process startup. It is downgrade-only: an unsupported upgrade is ignored.
`Backend::ALL`, `Display`, and `FromStr` support diagnostics and CLI wiring.

## Features and platforms

| Configuration | Result |
| --- | --- |
| default (`std`, `simd`) | runtime CPU detection and vector kernels |
| `std` without `simd` | portable kernels with allocation-backed plans |
| `--no-default-features` | `no_std`, portable kernels, allocation-free API |
| x86/x86_64 | AVX-512/GFNI/AVX2/SSSE3 runtime dispatch |
| AArch64 | NEON runtime dispatch, optional PMULL |
| wasm32 + `simd128` | WebAssembly vector kernels |
| other targets | portable scalar kernels |

## Safety

The public API is safe. Unsafe code is confined to architecture modules under
`src/kernel/`; runtime dispatch establishes the target-feature preconditions
before those functions are called. Every backend is differentially tested
against the portable scalar implementation across lane boundaries and odd row
geometry. See [CONTRIBUTING.md](CONTRIBUTING.md) for the enforced policy and
[ROADMAP.md](ROADMAP.md) for remaining hardware-verification work.

## Scope

`fff` provides fields, representations, and vector arithmetic. It does not
provide Cauchy/Vandermonde construction, matrix inversion, shard ownership,
streaming decoders, or an erasure-code recipe. Those belong in a codec layer;
keeping them separate lets this crate stay useful to proof systems and other
non-codec users.

## Benchmarks

`cargo bench --bench kernels` measures the operation shapes;
`cargo bench --bench compare` compares against `reed-solomon-erasure`.
Measurements and reproduction notes live in [BENCHMARKS.md](BENCHMARKS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). CI covers stable Rust on Linux, macOS,
and Windows, the 1.89 MSRV, scalar/no_std builds, AArch64 and Wasm cross-builds,
backend sweeps, clippy, rustdoc, and scalar-path Miri.

## License

MIT — see [LICENSE](LICENSE).
