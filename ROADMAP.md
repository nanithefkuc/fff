# Roadmap

`fff` 0.1 focuses on a small, stable arithmetic layer: eight binary fields,
packed vector operations, prepared coefficient plans, and safe runtime SIMD
dispatch. This file records work that remains valuable without presenting it
as part of the current contract.

## Correctness and hardware coverage

- Run the AVX-512 differential suite on AVX-512F + AVX-512BW + GFNI silicon.
  The backend cross-compiles and its emitted assembly has been inspected, but
  the current development host cannot execute it.
- Run the optional AArch64 PMULL path on an AES/PMULL-capable CI runner.
  Baseline NEON has been exercised on Snapdragon 8 Gen 3 hardware.
- Add fuzz targets for public row geometry (`row_len`, `nrows`, source count,
  coefficient count) and packed element boundaries.
- Keep scalar/no-default-feature paths under Miri. Intrinsics themselves are
  outside Miri's execution model.

These are validation goals, not hidden API prerequisites: unsupported CPU
features are never selected at runtime and every target retains the portable
implementation.

## Performance

- Add direct prepared, register-blocked scatter/gather/matrix entry points for
  backends where they beat repeated prepared AXPY. `Plan` already avoids table
  construction; this work would also recover the destination-memory advantage
  of each backend's blocked raw-coefficient path.
- Revisit the measured crossover rules per `(field, backend, shape)`. A backend
  name alone cannot predict whether a particular operation is register-blocked.
- Investigate vector implementations for `Gf32`, `Gf64`, and the canonical
  Fan–Paar tower only when a representation-specific design beats the portable
  reference. A generic intrinsic layer without a field that benefits is dead
  code.

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
