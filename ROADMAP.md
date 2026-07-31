# Roadmap

`fff` 0.1 focuses on a small, stable arithmetic layer: eight binary fields,
packed vector operations, prepared coefficient plans, and safe runtime SIMD
dispatch. This file records work that remains valuable without presenting it
as part of the current contract.

## Correctness and hardware coverage

- Run the AVX-512 differential suite on AVX-512F + AVX-512BW + GFNI silicon.
  The backend cross-compiles and its emitted assembly has been inspected, but
  the current development host cannot execute it.
- The optional AArch64 PMULL path is exercised on Snapdragon 8 Gen 3 hardware,
  as is baseline NEON: `cargo ndk-test -t arm64-v8a` against an adb-connected
  arm64 device runs the differential and public suites, and `Backend::Pmull`
  makes the capability a cached, overridable dispatch choice. An AES/PMULL
  capable CI runner would keep that coverage without a device attached. Using
  PMULL for *fixed*-coefficient kernels is a closed question, not a gap: it
  measured far behind the nibble shuffle (BENCHMARKS.md).
- Add fuzz targets for public row geometry (`row_len`, `nrows`, source count,
  coefficient count) and packed element boundaries.
- Keep scalar/no-default-feature paths under Miri. Intrinsics themselves are
  outside Miri's execution model.

These are validation goals, not hidden API prerequisites: unsupported CPU
features are never selected at runtime and every target retains the portable
implementation.

## Performance

- Revisit the measured crossover rules per `(field, backend, shape)`. A backend
  name alone cannot predict whether a particular operation is register-blocked.
- GF(2^32)/GF(2^64) now have an x86 GFNI kernel (the tower identity one and two
  levels above GF(2^16)); the canonical Fan–Paar GF(2^16)/32/64 have x86
  AVX2/SSSE3 kernels (the `mul_alpha` fold lands in coefficient preparation,
  so each reuses the same four-shuffle shape as the polynomial tower, composed
  recursively for the wider fields). The remaining vectorization work is the
  shuffle/NEON/wasm arms of the same identities, the AVX2-512 and SSSE3
  variants for the wider fields, and FanPaar8, which still uses the portable
  reference.

## Deliberate non-goals

- Runtime-selectable reduction polynomials. GF(2^8) deliberately matches the
  polynomial implemented by x86 GFNI; another polynomial should be another
  field type.
- A flat GF(2^16) log-table implementation. Its table is too large for L1 and
  does not vectorize as well as the current tower.
- Codec policy: Cauchy/Vandermonde recipes, shard ownership, streaming recovery,
  and public decoder APIs belong in a crate above `fff`.
- GPU offload. The transfer cost loses to CPU kernels at realistic buffer sizes.
- External `FieldKernels` implementations. SIMD dispatch is field-specific;
  the trait is sealed so the crate does not promise acceleration it cannot
  provide.
