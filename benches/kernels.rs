//! Throughput harness for the vector kernels.
//!
//! No criterion: these kernels are memory-bound straight-line loops with no
//! branchy tail, so a warm loop and a wall-clock median is enough to see the
//! effects that matter — backend, buffer size relative to cache, and whether
//! the blocked multi-row shapes actually beat repeated single-row AXPY.
//!
//! ```sh
//! cargo bench --bench kernels
//! FFF_BACKEND=avx2   cargo bench --bench kernels   # compare backends
//! FFF_BACKEND=scalar cargo bench --bench kernels
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use fff::{
    FanPaar16, FanPaar32, FanPaar64, Gf8, Gf16, Gf32, Gf64, backend, fan_paar, gf8, gf16, gf32,
    gf64, ops,
};

fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Run `body` until it has been timed enough times to trust the median, and
/// report bytes per second over `bytes` of logical traffic per iteration.
///
/// Returns the median iteration time so callers can print a ratio between two
/// shapes measured back to back in the same process.
fn bench(label: &str, bytes: usize, mut body: impl FnMut()) -> Duration {
    // Warm caches, branch predictors, and the lazy backend detection.
    for _ in 0..16 {
        body();
    }

    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        let reps = 32;
        for _ in 0..reps {
            body();
        }
        samples.push(start.elapsed() / reps);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];

    let gib_per_sec = bytes as f64 / median.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("  {label:<44} {:>9.2?}  {gib_per_sec:>7.2} GiB/s", median);
    median
}

/// Row lengths for the preparation crossover sweep: below one lane through
/// well past the point where the byte loop dominates table derivation.
const CROSSOVER_LENGTHS: &[usize] = &[
    16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 16_384, 65_536,
];

/// Where does preparing a coefficient stop paying for itself?
///
/// One-shot `mul_add` derives the backend's coefficient form on every call —
/// two broadcast words on GFNI, four nibble tables on a shuffle backend. The
/// `_with` form derives it once. The ratio printed here is the whole reason
/// `Coeff`/`Plan` exist, and it is the number `BENCHMARKS.md` refers to.
fn bench_preparation_crossover() {
    println!("preparation crossover — one-shot vs prepared, by row length:");
    for &len in CROSSOVER_LENGTHS {
        let src = noise(len, 0xa00 + len as u64);
        let mut dst = noise(len, 0xb00 + len as u64);
        let coeff8 = gf8::Elem(0x53);
        let coeff16 = gf16::Elem(0x53a7);
        let prepared8 = ops::Coeff::<Gf8>::new(coeff8);
        let prepared16 = ops::Coeff::<Gf16>::new(coeff16);

        println!("  row {len} B:");
        let one8 = bench("  mul_add one-shot           gf8", len, || {
            ops::mul_add::<Gf8>(black_box(&mut dst), coeff8, black_box(&src));
        });
        let with8 = bench("  mul_add prepared           gf8", len, || {
            ops::mul_add_with::<Gf8>(black_box(&mut dst), &prepared8, black_box(&src));
        });
        let one16 = bench("  mul_add one-shot          gf16", len, || {
            ops::mul_add::<Gf16>(black_box(&mut dst), coeff16, black_box(&src));
        });
        let with16 = bench("  mul_add prepared          gf16", len, || {
            ops::mul_add_with::<Gf16>(black_box(&mut dst), &prepared16, black_box(&src));
        });
        println!(
            "    one-shot/prepared: gf8 {:.2}x  gf16 {:.2}x",
            one8.as_secs_f64() / with8.as_secs_f64(),
            one16.as_secs_f64() / with16.as_secs_f64()
        );
    }
    println!();
}

/// Payload lengths used by network-facing consumers.
///
/// The run around 1,200 bytes alternates exact 32-byte lanes with 16-byte
/// remainders so SIMD tail cliffs stay visible.
const NETWORK_LENGTHS: &[usize] = &[
    64, 256, 512, 1_152, 1_168, 1_184, 1_200, 1_216, 1_232, 1_248, 1_400,
];

fn bench_network_payloads() {
    println!("network-size GF(2^8) payloads:");
    for &len in NETWORK_LENGTHS {
        println!("  payload {len} B:");
        let src = noise(len, 0x700 + len as u64);
        let mut dst = noise(len, 0x800 + len as u64);

        bench("xor", len, || {
            ops::add_assign::<Gf8>(black_box(&mut dst), black_box(&src));
        });
        bench("mul_add", len, || {
            ops::mul_add::<Gf8>(black_box(&mut dst), gf8::Elem(0x53), black_box(&src));
        });
        bench("mul_assign", len, || {
            ops::mul_assign::<Gf8>(black_box(&mut dst), gf8::Elem(0x53));
        });

        for nrows in [4usize, 16] {
            let coeffs: Vec<_> = (0..nrows)
                .map(|row| gf8::Elem((row as u8).wrapping_mul(37).wrapping_add(2)))
                .collect();
            let mut rows = noise(len * nrows, 0x900 + nrows as u64);
            let label = format!("scatter ({nrows} rows)");
            bench(&label, len * nrows, || {
                ops::mul_add_scatter::<Gf8>(
                    black_box(&mut rows),
                    len,
                    black_box(&coeffs),
                    black_box(&src),
                );
            });
        }
    }
    println!();
}

/// Blocked multi-row GF(2^16) kernels measured against repeated single-row
/// AXPY, bypassing dispatch.
///
/// Dispatch currently selects AXPY for GFNI gather, AVX2 gather, and AVX2
/// matrix (`src/kernel/gf16.rs:119-149`). That choice is a measurement, not a
/// theory, so it needs a harness that can run both sides in one process:
/// hence the `internals` feature and the direct kernel calls.
#[cfg(all(
    feature = "internals",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn bench_blocked_vs_axpy() {
    use fff::kernel::tables::{TowerCoeff, TowerTables};
    use fff::kernel::x86;

    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
    let has_gfni = has_avx2 && std::arch::is_x86_feature_detected!("gfni");
    let has_ssse3 = std::arch::is_x86_feature_detected!("ssse3");
    if !has_ssse3 {
        return;
    }

    println!("blocked vs AXPY — direct GF(2^16) kernel calls (dispatch bypassed):");
    for &row_len in &[4 * 1024usize, 16 * 1024, 64 * 1024] {
        for &nsrc in &[2usize, 4, 8, 16] {
            let sources: Vec<Vec<u8>> = (0..nsrc)
                .map(|t| noise(row_len, 0xc00 + t as u64))
                .collect();
            let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let coeffs: Vec<gf16::Elem> = (0..nsrc)
                .map(|t| gf16::Elem(((t * 7919) as u16).wrapping_add(3)))
                .collect();
            let mut dst = noise(row_len, 0xd00);
            let traffic = row_len * nsrc;

            println!("  gather {nsrc} sources x {} KiB:", row_len / 1024);
            let blocked = bench("  gather blocked           ssse3", traffic, || {
                x86::gf16::gather_ssse3(black_box(&mut dst), &coeffs, black_box(&srcs));
            });
            let axpy = bench("  gather AXPY              ssse3", traffic, || {
                for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                    x86::gf16::mul_add_ssse3(
                        black_box(&mut dst),
                        &TowerTables::new(coeff),
                        black_box(src),
                    );
                }
            });
            println!(
                "    ssse3 blocked/AXPY: {:.2}x",
                axpy.as_secs_f64() / blocked.as_secs_f64()
            );

            if has_avx2 {
                let blocked = bench("  gather blocked            avx2", traffic, || {
                    x86::gf16::gather_avx2(black_box(&mut dst), &coeffs, black_box(&srcs));
                });
                let axpy = bench("  gather AXPY               avx2", traffic, || {
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        x86::gf16::mul_add_avx2(
                            black_box(&mut dst),
                            &TowerTables::new(coeff),
                            black_box(src),
                        );
                    }
                });
                println!(
                    "    avx2  blocked/AXPY: {:.2}x",
                    axpy.as_secs_f64() / blocked.as_secs_f64()
                );
            }
            if has_gfni {
                let blocked = bench("  gather blocked            gfni", traffic, || {
                    x86::gf16::gather_gfni(black_box(&mut dst), &coeffs, black_box(&srcs));
                });
                let axpy = bench("  gather AXPY               gfni", traffic, || {
                    for (&coeff, &src) in coeffs.iter().zip(&srcs) {
                        x86::gf16::mul_add_gfni(
                            black_box(&mut dst),
                            TowerCoeff::new(coeff),
                            black_box(src),
                        );
                    }
                });
                println!(
                    "    gfni  blocked/AXPY: {:.2}x",
                    axpy.as_secs_f64() / blocked.as_secs_f64()
                );
            }

            if !has_avx2 {
                continue;
            }
            // Matrix: `nsrc` sources folded into 4 rows.
            let nrows = 4;
            let mut rows = noise(row_len * nrows, 0xe00);
            let coeff_sets: Vec<Vec<gf16::Elem>> = (0..nsrc)
                .map(|t| {
                    (0..nrows)
                        .map(|j| gf16::Elem(((t * 613 + j * 97) as u16).wrapping_add(1)))
                        .collect()
                })
                .collect();
            let terms: Vec<(&[gf16::Elem], &[u8])> = coeff_sets
                .iter()
                .zip(&sources)
                .map(|(c, s)| (c.as_slice(), s.as_slice()))
                .collect();
            let traffic = row_len * nrows * nsrc;
            println!(
                "  matrix {nsrc} sources x {nrows} rows x {} KiB:",
                row_len / 1024
            );
            let blocked = bench("  matrix blocked            avx2", traffic, || {
                x86::gf16::matrix_avx2(black_box(&mut rows), row_len, nrows, &terms);
            });
            let axpy = bench("  matrix AXPY               avx2", traffic, || {
                for &(coeffs, src) in &terms {
                    for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                        x86::gf16::mul_add_avx2(
                            black_box(row),
                            &TowerTables::new(coeff),
                            black_box(src),
                        );
                    }
                }
            });
            println!(
                "    avx2  blocked/AXPY: {:.2}x",
                axpy.as_secs_f64() / blocked.as_secs_f64()
            );
        }
    }
    println!();
}

/// Destination sizes straddling the non-temporal store threshold
/// (`x86::NT_STORE_MIN`, 2 MiB): one below it as a control, two above.
const LARGE_DST_LENGTHS: &[usize] = &[1 << 20, 8 << 20, 32 << 20];

/// `mul_into` over destinations large enough for the store policy to matter.
///
/// `mul_into` never reads its destination, so an ordinary store still fetches
/// every line it overwrites. Past 2 MiB the x86 kernels switch to `vmovntdq`,
/// which skips that fetch and does not keep the destination cached. The
/// read-back row is the workload that trade pessimizes: encode, then consume
/// the result immediately. `mul_add` is the control — it reads its
/// destination, so it is unaffected either way.
fn bench_large_destination() {
    println!("large destinations — mul_into store policy:");
    for &len in LARGE_DST_LENGTHS {
        let src = noise(len, 0xd00);
        let mut dst = noise(len, 0xd01);
        let mib = len / (1024 * 1024);

        bench(&format!("{mib:3} MiB mul_into          gf8"), len, || {
            ops::mul_into::<Gf8>(black_box(&mut dst), gf8::Elem(0x53), black_box(&src));
        });
        bench(&format!("{mib:3} MiB mul_into         gf16"), len, || {
            ops::mul_into::<Gf16>(black_box(&mut dst), gf16::Elem(0x53a7), black_box(&src));
        });
        bench(&format!("{mib:3} MiB mul_into+read     gf8"), len, || {
            ops::mul_into::<Gf8>(black_box(&mut dst), gf8::Elem(0x53), black_box(&src));
            let mut acc = 0u64;
            for word in dst.chunks_exact(8) {
                acc ^= u64::from_le_bytes(word.try_into().unwrap());
            }
            black_box(acc);
        });
        bench(&format!("{mib:3} MiB mul_add           gf8"), len, || {
            ops::mul_add::<Gf8>(black_box(&mut dst), gf8::Elem(0x53), black_box(&src));
        });
    }
    println!();
}

/// Row lengths where a GF(2^16) coefficient's four nibble tables are a
/// visible share of the work, plus one length where they are not.
const SMALL_ROW_LENGTHS: &[usize] = &[64, 256, 1_024, 16_384];

/// Multi-row GF(2^16) shapes at row lengths where preparation dominates.
///
/// A GF(2^16) coefficient costs four derived nibble tables. Multi-row kernels
/// that rebuild them per `(term, row)` — as the wasm kernels used to — pay
/// that per row instead of once, which only shows up when the row is short
/// enough that the byte loop cannot hide it. Coefficient sets deliberately
/// include zeros and ones, the case the zero/one specializations exist for.
fn bench_small_row_shapes() {
    println!("small-row GF(2^16) multi-row shapes — preparation-dominated:");
    let nrows = 16;
    let nsrc = 8;
    for &row_len in SMALL_ROW_LENGTHS {
        let sources: Vec<Vec<u8>> = (0..nsrc)
            .map(|t| noise(row_len, 0xf00 + t as u64))
            .collect();
        let srcs: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
        // Every fourth coefficient is zero and every fifth is one.
        let coeff_at = |i: usize| match i % 5 {
            0 => gf16::Elem::ZERO,
            1 => gf16::Elem::ONE,
            _ => gf16::Elem(((i * 7919) as u16) | 0x0100),
        };
        let row_coeffs: Vec<gf16::Elem> = (0..nrows).map(coeff_at).collect();
        let coeff_sets: Vec<Vec<gf16::Elem>> = (0..nsrc)
            .map(|t| (0..nrows).map(|j| coeff_at(t * 3 + j)).collect())
            .collect();
        let terms: Vec<(&[gf16::Elem], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let mut rows = noise(row_len * nrows, 0x1000);
        let mut dst = noise(row_len, 0x1100);

        println!("  row {row_len} B, {nrows} rows, {nsrc} sources:");
        bench("  scatter                   gf16", row_len * nrows, || {
            ops::mul_add_scatter::<Gf16>(
                black_box(&mut rows),
                row_len,
                &row_coeffs,
                black_box(srcs[0]),
            );
        });
        bench("  gather                    gf16", row_len * nsrc, || {
            ops::mul_add_gather::<Gf16>(black_box(&mut dst), &row_coeffs[..nsrc], black_box(&srcs));
        });
        bench(
            "  matrix                    gf16",
            row_len * nrows * nsrc,
            || {
                ops::mul_add_matrix::<Gf16>(black_box(&mut rows), row_len, nrows, &terms);
            },
        );

        // Fused `mul_into` against the copy-then-scale it replaces: the
        // fused kernel touches the destination once.
        let coeff = gf16::Elem(0x53a7);
        let fused = bench("  mul_into fused            gf16", row_len, || {
            ops::mul_into::<Gf16>(black_box(&mut dst), coeff, black_box(srcs[0]));
        });
        let copied = bench("  mul_into copy+scale       gf16", row_len, || {
            let dst = black_box(&mut dst);
            dst.copy_from_slice(black_box(srcs[0]));
            ops::mul_assign::<Gf16>(dst, coeff);
        });
        println!(
            "    copy+scale/fused: {:.2}x",
            copied.as_secs_f64() / fused.as_secs_f64()
        );
    }
    println!();
}

fn main() {
    println!("fff kernel benchmark — backend: {}", backend().name());
    println!("  (override with FFF_BACKEND=avx512|gfni|avx2|ssse3|pmull|neon|scalar)\n");

    bench_preparation_crossover();
    bench_small_row_shapes();
    bench_large_destination();
    #[cfg(all(
        feature = "internals",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    bench_blocked_vs_axpy();

    bench_network_payloads();

    // L1-resident, L2-resident, and DRAM-resident.
    for &len in &[4 * 1024usize, 256 * 1024, 8 * 1024 * 1024] {
        let human = if len >= 1024 * 1024 {
            format!("{} MiB", len / (1024 * 1024))
        } else {
            format!("{} KiB", len / 1024)
        };
        println!("buffer {human}:");

        let src = noise(len, 1);
        let mut dst = noise(len, 2);
        let prepared16 = ops::Coeff::<Gf16>::new(gf16::Elem(0x53a7));
        let rhs = noise(len, 0x602);
        let mut product = vec![0; len];

        bench("xor                       gf8", len, || {
            ops::add_assign::<Gf8>(black_box(&mut dst), black_box(&src));
        });
        bench("mul_add                   gf8", len, || {
            ops::mul_add::<Gf8>(black_box(&mut dst), gf8::Elem(0x53), black_box(&src));
        });
        bench("mul_add                  gf16", len, || {
            ops::mul_add::<Gf16>(black_box(&mut dst), gf16::Elem(0x53a7), black_box(&src));
        });
        bench("mul_add prepared         gf16", len, || {
            ops::mul_add_with::<Gf16>(black_box(&mut dst), &prepared16, black_box(&src));
        });
        bench("mul_assign                gf8", len, || {
            ops::mul_assign::<Gf8>(black_box(&mut dst), gf8::Elem(0x53));
        });
        bench("elementwise                gf8", len, || {
            ops::mul_elementwise::<Gf8>(black_box(&mut product), black_box(&src), black_box(&rhs));
        });
        bench("elementwise               gf16", len, || {
            ops::mul_elementwise::<Gf16>(black_box(&mut product), black_box(&src), black_box(&rhs));
        });
        bench("mul_assign               gf16", len, || {
            ops::mul_assign::<Gf16>(black_box(&mut dst), gf16::Elem(0x53a7));
        });
        println!();
    }

    // Multi-row shapes. Geometry is a realistic erasure code: k data rows
    // folded into m parity rows, each row a 64 KiB symbol.
    let row_len = 64 * 1024;
    for &nrows in &[2usize, 4, 8, 16] {
        println!("scatter/matrix — {nrows} rows x {} KiB:", row_len / 1024);

        let src = noise(row_len, 3);
        let mut rows = noise(row_len * nrows, 4);
        let coeffs8: Vec<_> = (0..nrows)
            .map(|j| gf8::Elem((j as u8).wrapping_mul(37).wrapping_add(2)))
            .collect();
        let coeffs16: Vec<_> = (0..nrows)
            .map(|j| gf16::Elem((j as u16).wrapping_mul(9871).wrapping_add(2)))
            .collect();
        let scatter_plan8 = ops::Plan::<Gf8>::new(&coeffs8);
        let scatter_plan16 = ops::Plan::<Gf16>::new(&coeffs16);

        let traffic = row_len * nrows;
        bench("scatter                   gf8", traffic, || {
            ops::mul_add_scatter::<Gf8>(black_box(&mut rows), row_len, &coeffs8, black_box(&src));
        });
        bench("scatter                  gf16", traffic, || {
            ops::mul_add_scatter::<Gf16>(black_box(&mut rows), row_len, &coeffs16, black_box(&src));
        });
        bench("scatter prepared          gf8", traffic, || {
            ops::mul_add_scatter_with::<Gf8>(
                black_box(&mut rows),
                row_len,
                &scatter_plan8,
                black_box(&src),
            );
        });
        bench("scatter prepared         gf16", traffic, || {
            ops::mul_add_scatter_with::<Gf16>(
                black_box(&mut rows),
                row_len,
                &scatter_plan16,
                black_box(&src),
            );
        });
        bench("scatter (unblocked)       gf8", traffic, || {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs8) {
                ops::mul_add::<Gf8>(black_box(row), coeff, black_box(&src));
            }
        });
        bench("scatter (unblocked)      gf16", traffic, || {
            for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(&coeffs16) {
                ops::mul_add::<Gf16>(black_box(row), coeff, black_box(&src));
            }
        });

        // The blocked-vs-unblocked comparison: 8 sources into `nrows` rows,
        // as one matrix call versus eight scatter calls. Same arithmetic,
        // different destination memory traffic.
        let sources: Vec<Vec<u8>> = (0..16).map(|t| noise(row_len, 100 + t as u64)).collect();
        let coeff_sets: Vec<Vec<gf8::Elem>> = (0..8)
            .map(|t| {
                (0..nrows)
                    .map(|j| gf8::Elem(((t * 31 + j * 17) as u8).wrapping_add(1)))
                    .collect()
            })
            .collect();
        let coeff_sets16: Vec<Vec<gf16::Elem>> = (0..8)
            .map(|t| {
                (0..nrows)
                    .map(|j| gf16::Elem(((t * 7919 + j * 613) as u16).wrapping_add(1)))
                    .collect()
            })
            .collect();
        let terms: Vec<(&[gf8::Elem], &[u8])> = coeff_sets
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let terms16: Vec<(&[gf16::Elem], &[u8])> = coeff_sets16
            .iter()
            .zip(&sources)
            .map(|(c, s)| (c.as_slice(), s.as_slice()))
            .collect();
        let matrix_coeffs8: Vec<_> = coeff_sets.iter().flatten().copied().collect();
        let matrix_coeffs16: Vec<_> = coeff_sets16.iter().flatten().copied().collect();
        let matrix_plan8 = ops::Plan::<Gf8>::matrix(8, nrows, &matrix_coeffs8);
        let matrix_plan16 = ops::Plan::<Gf16>::matrix(8, nrows, &matrix_coeffs16);
        let matrix_srcs: Vec<&[u8]> = sources.iter().take(8).map(Vec::as_slice).collect();

        let traffic = row_len * nrows * 8;
        bench("matrix (selected)          gf8", traffic, || {
            ops::mul_add_matrix::<Gf8>(black_box(&mut rows), row_len, nrows, &terms);
        });
        bench("matrix (unblocked AXPY)   gf8", traffic, || {
            for &(coeffs, src) in &terms {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    ops::mul_add::<Gf8>(black_box(row), coeff, src);
                }
            }
        });
        bench("matrix (selected)         gf16", traffic, || {
            ops::mul_add_matrix::<Gf16>(black_box(&mut rows), row_len, nrows, &terms16);
        });
        bench("matrix prepared           gf8", traffic, || {
            ops::mul_add_matrix_with::<Gf8>(
                black_box(&mut rows),
                row_len,
                nrows,
                &matrix_plan8,
                black_box(&matrix_srcs),
            );
        });
        bench("matrix prepared          gf16", traffic, || {
            ops::mul_add_matrix_with::<Gf16>(
                black_box(&mut rows),
                row_len,
                nrows,
                &matrix_plan16,
                black_box(&matrix_srcs),
            );
        });
        bench("matrix (unblocked AXPY)  gf16", traffic, || {
            for &(coeffs, src) in &terms16 {
                for (row, &coeff) in rows.chunks_exact_mut(row_len).zip(coeffs) {
                    ops::mul_add::<Gf16>(black_box(row), coeff, src);
                }
            }
        });

        let gather_srcs: Vec<&[u8]> = sources.iter().take(nrows).map(Vec::as_slice).collect();
        let mut gathered = noise(row_len, 5);
        let gather_plan8 = ops::Plan::<Gf8>::new(&coeffs8);
        let gather_plan16 = ops::Plan::<Gf16>::new(&coeffs16);
        let gather_traffic = row_len * nrows;
        bench("gather (selected)          gf8", gather_traffic, || {
            ops::mul_add_gather::<Gf8>(black_box(&mut gathered), &coeffs8, black_box(&gather_srcs));
        });
        bench("gather (unblocked)        gf8", gather_traffic, || {
            for (&coeff, &source) in coeffs8.iter().zip(&gather_srcs) {
                ops::mul_add::<Gf8>(black_box(&mut gathered), coeff, black_box(source));
            }
        });
        bench("gather (selected)         gf16", gather_traffic, || {
            ops::mul_add_gather::<Gf16>(
                black_box(&mut gathered),
                &coeffs16,
                black_box(&gather_srcs),
            );
        });
        bench("gather prepared           gf8", gather_traffic, || {
            ops::mul_add_gather_with::<Gf8>(
                black_box(&mut gathered),
                &gather_plan8,
                black_box(&gather_srcs),
            );
        });
        bench("gather prepared          gf16", gather_traffic, || {
            ops::mul_add_gather_with::<Gf16>(
                black_box(&mut gathered),
                &gather_plan16,
                black_box(&gather_srcs),
            );
        });
        bench("gather (unblocked)       gf16", gather_traffic, || {
            for (&coeff, &source) in coeffs16.iter().zip(&gather_srcs) {
                ops::mul_add::<Gf16>(black_box(&mut gathered), coeff, black_box(source));
            }
        });
        println!();
    }

    // Same byte volume means the same information volume: one GF(2^32)
    // symbol occupies exactly two GF(2^16) symbols. This makes the cost of
    // choosing a wider field directly visible rather than hiding it behind a
    // symbol-count comparison.
    let tier3_len = 256 * 1024;
    let tier3_src = noise(tier3_len, 0x900);
    let mut tier3_dst = noise(tier3_len, 0x901);
    println!("Tier 3 field cost — {tier3_len} bytes:");
    bench("mul_add polynomial tower     gf16", tier3_len, || {
        ops::mul_add::<Gf16>(
            black_box(&mut tier3_dst),
            gf16::Elem(0x53a7),
            black_box(&tier3_src),
        );
    });
    bench("mul_add polynomial tower     gf32", tier3_len, || {
        ops::mul_add::<Gf32>(
            black_box(&mut tier3_dst),
            gf32::Elem(0xdead_beef),
            black_box(&tier3_src),
        );
    });
    bench("mul_add polynomial tower     gf64", tier3_len, || {
        ops::mul_add::<Gf64>(
            black_box(&mut tier3_dst),
            gf64::Elem(0x0123_4567_89ab_cdef),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp16", tier3_len, || {
        ops::mul_add::<FanPaar16>(
            black_box(&mut tier3_dst),
            fan_paar::fp16::Elem(0xe2de),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp32", tier3_len, || {
        ops::mul_add::<FanPaar32>(
            black_box(&mut tier3_dst),
            fan_paar::fp32::Elem(0x03e2_1cea),
            black_box(&tier3_src),
        );
    });
    bench("mul_add canonical Fan-Paar  fp64", tier3_len, || {
        ops::mul_add::<FanPaar64>(
            black_box(&mut tier3_dst),
            fan_paar::fp64::Elem(0x070f_870d_cd9c_1d88),
            black_box(&tier3_src),
        );
    });
}
