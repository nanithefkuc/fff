//! `fff` side of the external-library comparison (see external-bench/PLAN.md).
//!
//! Same harness shape as the C/C++ side: scalar multiply and
//! `dst ^= c * src` over 64 KiB, GF(2^8) and GF(2^16).
//!
//! ```sh
//! cargo bench --bench compare
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use fff::{Gf8, Gf16, backend, gf8, gf16, ops};

const BYTES: usize = 64 * 1024;
const SCALAR_ITERS: usize = 1 << 20;

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

fn bench_region(label: &str, bytes: usize, mut body: impl FnMut()) {
    for _ in 0..16 {
        body();
    }
    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        for _ in 0..32 {
            body();
        }
        samples.push(start.elapsed() / 32);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let gib = bytes as f64 / median.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("  {label:<44} {:>9.2?}  {gib:>7.2} GiB/s", median);
}

fn bench_scalar(label: &str, mut body: impl FnMut()) {
    for _ in 0..4 {
        body();
    }
    let mut samples = Vec::with_capacity(64);
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline && samples.len() < 64 {
        let start = Instant::now();
        body();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let ns = median.as_secs_f64() * 1e9 / SCALAR_ITERS as f64;
    let mops = SCALAR_ITERS as f64 / median.as_secs_f64() / 1e6;
    println!("  {label:<44} {ns:>9.2} ns/op  {mops:>7.2} Mops/s");
}

fn main() {
    println!("fff — backend: {}", backend().name());

    let src8 = noise(BYTES, 0x600);
    let mut dst8 = noise(BYTES, 0x601);
    let src16 = noise(BYTES, 0x602);
    let mut dst16 = noise(BYTES, 0x603);

    // w8
    let c8 = gf8::Elem(0x53);
    bench_scalar("w8 scalar mul", || {
        let mut x = gf8::Elem(0xA5);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(gf8::Elem((0x53u8).wrapping_add(i as u8)));
        }
        black_box(x);
    });
    bench_region("w8 mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf8>(black_box(&mut dst8), c8, black_box(&src8));
    });
    bench_region("w8 mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf8>(black_box(&mut dst8), c8, black_box(&src8));
    });

    // w16
    let c16 = gf16::Elem(0x53A7);
    bench_scalar("w16 scalar mul", || {
        let mut x = gf16::Elem(0x1234);
        for i in 0..SCALAR_ITERS {
            x = black_box(x).mul(gf16::Elem((0x53A7u16).wrapping_add(i as u16)));
        }
        black_box(x);
    });
    bench_region("w16 mul_add (dst ^= c*src)", BYTES, || {
        ops::mul_add::<Gf16>(black_box(&mut dst16), c16, black_box(&src16));
    });
    bench_region("w16 mul_into (dst = c*src)", BYTES, || {
        ops::mul_into::<Gf16>(black_box(&mut dst16), c16, black_box(&src16));
    });

    // reed-solomon-erasure (simd-accel: SSSE3/AVX2). GF(2^8), poly 0x11D.
    // mul_slice_xor(c, input, out) == dst ^= c * src, the exact mul_add analog.
    println!("reed-solomon-erasure 6 (simd-accel)");
    let mut rse_dst = noise(BYTES, 0x701);
    let rse_src = noise(BYTES, 0x702);
    bench_region("w8 mul_slice_xor (dst ^= c*src)", BYTES, || {
        reed_solomon_erasure::galois_8::mul_slice_xor(
            0x53,
            black_box(&rse_src),
            black_box(&mut rse_dst),
        );
    });
    bench_region("w8 mul_slice (dst = c*src)", BYTES, || {
        reed_solomon_erasure::galois_8::mul_slice(
            0x53,
            black_box(&rse_src),
            black_box(&mut rse_dst),
        );
    });
}
