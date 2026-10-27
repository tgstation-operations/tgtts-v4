//! The hand-written GLSL kernels from `nix build .#glsl-kernels` on the GPU, each against a
//! reference in f64 over the same f16 inputs, at small shapes and at the base LM's, and at
//! either subgroup size. Built with the `device` feature; the kernels are read from the
//! directory `GLSL_KERNELS` names, which `nix run .#device-tests` sets.
//!
//! Each bound is what f32 arithmetic and the f16 roundings the kernel shares with upstream can
//! account for, and each is shown to catch a defect of the kind it is for.
#![cfg(feature = "device")]

use std::path::PathBuf;

use half::f16;
use vulkan::{Buffer, Context, Kernel, KernelDesc, SpecValue};

const WORKGROUP: u32 = 256;

fn context() -> Context {
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    if !ctx.validating() {
        eprintln!("the validation layer is off for this run; `nix run .#device-tests` turns it on");
    }
    ctx
}

fn assert_no_validation_messages(ctx: &Context) {
    let messages = ctx.validation_messages();
    assert!(
        messages.is_empty(),
        "the validation layer reported:\n{}",
        messages.join("\n")
    );
}

fn kernel(
    ctx: &Context,
    name: &str,
    bindings: u32,
    subgroup: Option<u32>,
    constants: &[(u32, SpecValue)],
) -> Kernel {
    let dir = PathBuf::from(
        std::env::var_os("GLSL_KERNELS")
            .expect("GLSL_KERNELS names no directory; `nix run .#device-tests` sets it"),
    );
    let path = dir.join(format!("{name}.spv"));
    let spirv = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let desc = KernelDesc {
        storage_buffers: bindings,
        subgroup_size: subgroup,
        full_subgroups: subgroup.is_some(),
        constants,
        ..KernelDesc::default()
    };
    ctx.kernel(&spirv, &desc)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// A uniform sequence in [-scale, scale) from a fixed xorshift sequence.
fn uniform(len: usize, seed: u32, scale: f64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn halves(values: &[f64]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f64(v)).collect()
}

fn bits(values: &[f16]) -> Vec<u16> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn buffer_of(ctx: &Context, values: &[f16]) -> Buffer {
    let mut b = ctx.buffer(2 * values.len() as u64).unwrap();
    b.write(&bits(values));
    b
}

fn read_f16(b: &Buffer) -> Vec<f64> {
    b.read::<u16>()
        .into_iter()
        .map(|h| f16::from_bits(h).to_f64())
        .collect()
}

/// v rounded to f16 and back.
fn r16(v: f64) -> f64 {
    f16::from_f64(v).to_f64()
}

/// The gap between f16 values around v: what one rounding step to f16 can move it by.
fn ulp16(v: f64) -> f64 {
    let h = f16::from_f64(v.abs());
    if h.is_infinite() {
        return f64::INFINITY;
    }
    (f16::from_bits(h.to_bits() + 1).to_f64() - h.to_f64()).max(2f64.powi(-24))
}

/// Unit roundoff of f32.
const U32: f64 = 1.0 / (1u64 << 24) as f64;

/// The outputs past their bound: (index, error / bound), worst first.
fn past(got: &[f64], want: &[f64], bound: impl Fn(usize) -> f64) -> Vec<(usize, f64)> {
    let mut bad: Vec<_> = got
        .iter()
        .zip(want)
        .enumerate()
        .map(|(i, (g, w))| (i, (g - w).abs() / bound(i)))
        .filter(|(_, ratio)| ratio.is_nan() || *ratio > 1.0)
        .collect();
    bad.sort_by(|a, b| b.1.total_cmp(&a.1));
    bad
}

fn worst(got: &[f64], want: &[f64], bound: impl Fn(usize) -> f64) -> f64 {
    got.iter()
        .zip(want)
        .enumerate()
        .map(|(i, (g, w))| (g - w).abs() / bound(i))
        .fold(0.0, f64::max)
}

/// RMSNorm in f64, rounded to f16: x times the reciprocal root of its mean square, times the
/// gain.
fn rmsnorm(x: &[f16], g: &[f16], eps: f64) -> Vec<f64> {
    let mean = x.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(g)
        .map(|(x, g)| r16(x.to_f64() * scale * g.to_f64()))
        .collect()
}

/// Rows `rows` of W (`k` columns) times `x`, in f64, with each sum's magnitude.
fn dots(w: &[f16], x: &[f64], k: usize, rows: impl Iterator<Item = usize>) -> Vec<(f64, f64)> {
    rows.map(|r| {
        w[r * k..][..k]
            .iter()
            .zip(x)
            .map(|(w, x)| w.to_f64() * x)
            .fold((0.0, 0.0), |(s, m), p| (s + p, m + p.abs()))
    })
    .collect()
}

/// The bound on an f32 sum of `k` products against the exact one: k unit roundoffs of the sum
/// of their magnitudes.
fn sum_bound(k: usize, magnitude: f64) -> f64 {
    k as f64 * U32 * magnitude
}

fn timed(ctx: &Context, kernel: &Kernel, buffers: &mut [&mut Buffer], groups: [u32; 3]) -> f64 {
    let t = ctx
        .dispatch(kernel, buffers, groups)
        .unwrap_or_else(|e| panic!("{e}"));
    t.nanoseconds
}

#[test]
fn rmsnorm_rounds_as_upstream_in_f16() {
    let ctx = context();
    for (k, subgroup) in [(256, 64), (2048, 64), (2048, 32)] {
        let x = halves(&uniform(k, 3, 2.0));
        let g = halves(
            &uniform(k, 5, 1.0)
                .iter()
                .map(|v| 1.0 + v / 4.0)
                .collect::<Vec<_>>(),
        );
        let eps = 1e-5;
        let constants = [
            (0, u(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
            (2, u(k)),
            (3, SpecValue::F32(eps as f32)),
        ];
        let kernel = kernel(&ctx, "rmsnorm", 3, Some(subgroup as u32), &constants);
        let (mut xb, mut gb) = (buffer_of(&ctx, &x), buffer_of(&ctx, &g));
        let mut out = ctx.buffer(2 * k as u64).unwrap();
        let ns = timed(&ctx, &kernel, &mut [&mut xb, &mut gb, &mut out], [1, 1, 1]);
        let got = read_f16(&out);
        let want = rmsnorm(&x, &g, eps);
        // f32's product differs from f64's in its last bits, which can send the rounding to
        // the next f16.
        let bound = |i: usize| ulp16(want[i]);
        let bad = past(&got, &want, bound);
        assert!(bad.is_empty(), "K {k}, wave{subgroup}: {bad:?}");
        let exact = got.iter().zip(&want).filter(|(g, w)| g == w).count();
        // Defect: the gain left out.
        let ungained = rmsnorm(&x, &vec![f16::ONE; k], eps);
        assert!(past(&got, &ungained, bound).len() > k / 2);
        eprintln!(
            "rmsnorm K {k} wave{subgroup}: {exact} of {k} bit-exact, worst at {:.2} of its bound; \
             {:.1} us",
            worst(&got, &want, bound),
            ns / 1e3
        );
    }
    assert_no_validation_messages(&ctx);
}

/// W `n x k` of weights of about `scale`, x, and a residual, with row `subnormal` of W all f16
/// subnormals and its residual zero, so that a flush to zero would show.
struct GemvCase {
    w: Vec<f16>,
    x: Vec<f16>,
    residual: Vec<f16>,
    subnormal: usize,
}

fn gemv_case(n: usize, k: usize, seed: u32) -> GemvCase {
    let mut w = halves(&uniform(n * k, seed, 0.04));
    let subnormal = n / 3;
    for (i, v) in w[subnormal * k..][..k].iter_mut().enumerate() {
        // Magnitudes from 2^-24 to just under 2^-14, both signs.
        let m = ((i * 37) % 1023 + 1) as u16;
        *v = f16::from_bits(if i % 2 == 0 { m } else { m | 0x8000 });
        assert!(!v.is_normal() && *v != f16::ZERO);
    }
    let mut residual = halves(&uniform(n, seed + 2, 1.0));
    residual[subnormal] = f16::ZERO;
    GemvCase {
        w,
        x: halves(&uniform(k, seed + 1, 1.5)),
        residual,
        subnormal,
    }
}

#[test]
fn gemv_with_the_residual_sums_in_f32_and_keeps_subnormal_weights() {
    let ctx = context();
    for (n, k, subgroup, rows) in [
        (64, 512, 64, 2),
        (2048, 2048, 64, 2),
        (2048, 2048, 32, 2),
        (2048, 6144, 64, 2),
        (2048, 6144, 64, 4),
        (2048, 6144, 32, 4),
    ] {
        let case = gemv_case(n, k, 11);
        let constants = [
            (0, u(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
            (2, u(k)),
            (3, u(rows)),
        ];
        let kernel = kernel(&ctx, "gemv_residual", 4, Some(subgroup as u32), &constants);
        let mut xb = buffer_of(&ctx, &case.x);
        let mut wb = buffer_of(&ctx, &case.w);
        let mut rb = buffer_of(&ctx, &case.residual);
        let mut out = ctx.buffer(2 * n as u64).unwrap();
        let groups = (n / (WORKGROUP as usize / subgroup * rows)) as u32;
        let mut ns = Vec::new();
        for _ in 0..4 {
            let buffers = &mut [&mut xb, &mut wb, &mut rb, &mut out];
            ns.push(timed(&ctx, &kernel, buffers, [groups, 1, 1]));
        }
        let got = read_f16(&out);
        let x: Vec<f64> = case.x.iter().map(|v| v.to_f64()).collect();
        let sums = dots(&case.w, &x, k, 0..n);
        let want: Vec<f64> = sums
            .iter()
            .zip(&case.residual)
            .map(|(&(s, _), r)| r16(r.to_f64() + s))
            .collect();
        // The f32 sum, then the rounding to f16, which its error may send the other way.
        let bound = |i: usize| ulp16(want[i]) + sum_bound(k, sums[i].1);
        let bad = past(&got, &want, bound);
        assert!(
            bad.is_empty(),
            "{n}x{k} wave{subgroup} rows {rows}: {bad:?}"
        );
        assert!(got[case.subnormal] != 0.0);
        // Defect: subnormal weights flushed to zero.
        let flushed = r16(0.0);
        assert!(
            (got[case.subnormal] - flushed).abs() > bound(case.subnormal),
            "the subnormal row would pass flushed"
        );
        let bytes = (n * k * 2) as f64;
        let best = ns[1..].iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!(
            "gemv_residual {n}x{k} wave{subgroup} rows {rows}: worst at {:.2} of its bound; best \
             of 3 {:.1} us, {:.0} GB/s of weights",
            worst(&got, &want, bound),
            best / 1e3,
            bytes / best
        );
    }
    assert_no_validation_messages(&ctx);
}

/// The bound on a row of W times the normed x the GPU computed, against the f64 one: the f32
/// sum's, and four elements of the normed x one f16 gap away, should f32's scale send their
/// rounding the other way.
fn normed_dot_bound(k: usize, magnitude: f64, row: &[f16], normed: &[f64]) -> f64 {
    let flips = row
        .iter()
        .zip(normed)
        .map(|(w, h)| w.to_f64().abs() * ulp16(*h))
        .fold(0.0, f64::max);
    sum_bound(k, magnitude) + 4.0 * flips
}

#[test]
fn gate_up_is_silu_of_the_gate_times_up_after_the_norm() {
    let ctx = context();
    for (k, inner, subgroup, pairs) in [
        (512, 256, 64, 1),
        (2048, 6144, 64, 1),
        (2048, 6144, 32, 1),
        (2048, 6144, 64, 2),
    ] {
        let case = gemv_case(2 * inner, k, 23);
        let g = halves(
            &uniform(k, 29, 0.5)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
        );
        let eps = 1e-5;
        let constants = [
            (0, u(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
            (2, u(k)),
            (3, u(inner)),
            (4, SpecValue::F32(eps as f32)),
            (5, u(pairs)),
        ];
        let kernel = kernel(&ctx, "gemv_gate_up", 4, Some(subgroup as u32), &constants);
        let mut xb = buffer_of(&ctx, &case.x);
        let mut gb = buffer_of(&ctx, &g);
        let mut wb = buffer_of(&ctx, &case.w);
        let mut out = ctx.buffer(2 * inner as u64).unwrap();
        let groups = (inner / (WORKGROUP as usize / subgroup * pairs)) as u32;
        let mut ns = Vec::new();
        for _ in 0..4 {
            let buffers = &mut [&mut xb, &mut gb, &mut wb, &mut out];
            ns.push(timed(&ctx, &kernel, buffers, [groups, 1, 1]));
        }
        let got = read_f16(&out);
        let normed = rmsnorm(&case.x, &g, eps);
        let sums = dots(&case.w, &normed, k, 0..2 * inner);
        let silu = |v: f64| v / (1.0 + (-v).exp());
        let want: Vec<f64> = (0..inner)
            .map(|r| r16(silu(sums[r].0) * sums[inner + r].0))
            .collect();
        // Each sum within its bound; SiLU's slope is under 1.1, and the product moves by each
        // factor's error times the other; then the rounding to f16.
        let bound = |r: usize| {
            let e = |row: usize| normed_dot_bound(k, sums[row].1, &case.w[row * k..][..k], &normed);
            let (gate, up) = (sums[r].0, sums[inner + r].0);
            let eg = 1.1 * e(r);
            ulp16(want[r]) + eg * up.abs() + e(inner + r) * silu(gate).abs() + eg * e(inner + r)
        };
        let bad = past(&got, &want, bound);
        assert!(
            bad.is_empty(),
            "{inner}x{k} wave{subgroup} pairs {pairs}: {bad:?}"
        );
        // Defect: gate and up swapped.
        let swapped: Vec<f64> = (0..inner)
            .map(|r| r16(silu(sums[inner + r].0) * sums[r].0))
            .collect();
        assert!(past(&got, &swapped, bound).len() > inner / 2);
        let bytes = (2 * inner * k * 2) as f64;
        let best = ns[1..].iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!(
            "gemv_gate_up {inner}x{k} wave{subgroup} pairs {pairs}: worst at {:.2} of its bound; \
             best of 3 {:.1} us, {:.0} GB/s of weights",
            worst(&got, &want, bound),
            best / 1e3,
            bytes / best
        );
    }
    assert_no_validation_messages(&ctx);
}

/// LongRoPE's tables as minicpm4 computes them, for the factors of VoxCPM2's config.
fn rope_tables(capacity: usize, dim: usize, factors: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let theta = 10_000f32;
    let rates: Vec<(f32, f32)> = factors
        .iter()
        .zip((0..dim).step_by(2))
        .map(|(&f, i)| (1.0 / f as f32, 1.0 / theta.powf(i as f32 / dim as f32)))
        .collect();
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for t in 0..capacity {
        for &(r, inv) in &rates {
            let f = t as f32 * r * inv;
            cos.push(f.cos());
            sin.push(f.sin());
        }
    }
    (cos, sin)
}

#[test]
fn qkv_projects_rotates_and_writes_the_cache_at_the_position() {
    let ctx = context();
    let (k, heads, kv_heads, dim, capacity) = (2048, 16, 2, 128, 64);
    let factors: Vec<f64> = (0..dim / 2).map(|i| 1.0 + i as f64 * 0.47).collect();
    let (cos, sin) = rope_tables(capacity, dim, &factors);
    let rows = (heads + 2 * kv_heads) * dim;
    let case = gemv_case(rows, k, 31);
    let g = halves(
        &uniform(k, 37, 0.5)
            .iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>(),
    );
    let eps = 1e-5;
    let normed = rmsnorm(&case.x, &g, eps);
    let sums = dots(&case.w, &normed, k, 0..rows);
    let sentinel = f16::from_f64(-7.0);
    for (subgroup, pairs, position) in [(64, 1, 0), (64, 1, 37), (32, 1, 63), (64, 2, 5)] {
        let constants = [
            (0, u(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
            (2, u(k)),
            (3, u(heads)),
            (4, u(kv_heads)),
            (5, u(dim)),
            (6, u(capacity)),
            (7, SpecValue::F32(eps as f32)),
            (8, u(pairs)),
        ];
        let kernel = kernel(&ctx, "gemv_qkv_rope", 9, Some(subgroup as u32), &constants);
        let mut xb = buffer_of(&ctx, &case.x);
        let mut gb = buffer_of(&ctx, &g);
        let mut wb = buffer_of(&ctx, &case.w);
        let mut cb = ctx.buffer(4 * cos.len() as u64).unwrap();
        cb.write(&cos);
        let mut sb = ctx.buffer(4 * sin.len() as u64).unwrap();
        sb.write(&sin);
        let mut pb = ctx.buffer(4).unwrap();
        pb.write(&[position as u32]);
        let mut qb = ctx.buffer(2 * (heads * dim) as u64).unwrap();
        let cache = vec![sentinel; kv_heads * capacity * dim];
        let (mut kb, mut vb) = (buffer_of(&ctx, &cache), buffer_of(&ctx, &cache));
        let groups = (rows / 2 / (WORKGROUP as usize / subgroup * pairs)) as u32;
        let buffers = &mut [
            &mut xb, &mut gb, &mut wb, &mut cb, &mut sb, &mut pb, &mut qb, &mut kb, &mut vb,
        ];
        let ns = timed(&ctx, &kernel, buffers, [groups, 1, 1]);
        let (q, kc, vc) = (read_f16(&qb), read_f16(&kb), read_f16(&vb));

        let half = dim / 2;
        // Every output as (got, want, bound), rotated as at `at`.
        let expected = |at: usize| {
            let mut rows = Vec::new();
            for head in 0..heads + 2 * kv_heads {
                for i in 0..half {
                    let (r0, r1) = (head * dim + i, head * dim + i + half);
                    let e =
                        |r: usize| normed_dot_bound(k, sums[r].1, &case.w[r * k..][..k], &normed);
                    let (a, b) = (sums[r0].0, sums[r1].0);
                    let c = f64::from(cos[at * half + i]);
                    let s = f64::from(sin[at * half + i]);
                    let (ra, rb, ea, eb) = if head < heads + kv_heads {
                        // Each sum's error, and the rotation's own roundings in f32.
                        let err = e(r0) + e(r1) + 4.0 * U32 * (a.abs() + b.abs());
                        (a * c - b * s, b * c + a * s, err, err)
                    } else {
                        (a, b, e(r0), e(r1))
                    };
                    let (ga, gb) = if head < heads {
                        (q[head * dim + i], q[head * dim + i + half])
                    } else {
                        let (cache, kv) = if head < heads + kv_heads {
                            (&kc, head - heads)
                        } else {
                            (&vc, head - heads - kv_heads)
                        };
                        let row = (kv * capacity + position) * dim + i;
                        (cache[row], cache[row + half])
                    };
                    rows.push((ga, r16(ra), ea + ulp16(ra)));
                    rows.push((gb, r16(rb), eb + ulp16(rb)));
                }
            }
            rows
        };
        let rows = expected(position);
        let got: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let want: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let bound = |i: usize| rows[i].2;
        let bad = past(&got, &want, bound);
        assert!(
            bad.is_empty(),
            "wave{subgroup} pairs {pairs} at {position}: {bad:?}"
        );
        let untouched = |cache: &[f64]| {
            cache
                .chunks(dim)
                .enumerate()
                .filter(|(row, _)| row % capacity != position)
                .all(|(_, r)| r.iter().all(|&v| v == -7.0))
        };
        assert!(
            untouched(&kc) && untouched(&vc),
            "a cache row past the position was written"
        );
        // Defect: the rotation of the next position, which turns the fastest frequencies of
        // every head of q and k, about one in twenty of their outputs, by a visible angle.
        let other = expected((position + 1) % capacity);
        let want_other: Vec<f64> = other.iter().map(|r| r.1).collect();
        assert!(past(&got, &want_other, bound).len() > got.len() / 50);
        eprintln!(
            "gemv_qkv_rope wave{subgroup} pairs {pairs} at {position}: worst at {:.2} of its \
             bound; {:.1} us",
            worst(&got, &want, bound),
            ns / 1e3
        );
    }
    assert_no_validation_messages(&ctx);
}

/// Attention of each query head over the cache up to `position`, in f64.
fn attention(
    q: &[f16],
    kc: &[f16],
    vc: &[f16],
    (heads, kv_heads, dim, capacity): (usize, usize, usize, usize),
    position: usize,
) -> Vec<f64> {
    let group = heads / kv_heads;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut out = vec![0.0; heads * dim];
    for h in 0..heads {
        let kv = h / group;
        let row = |c: &[f16], p: usize| -> Vec<f64> {
            c[(kv * capacity + p) * dim..][..dim]
                .iter()
                .map(|v| v.to_f64())
                .collect()
        };
        let qh: Vec<f64> = q[h * dim..][..dim].iter().map(|v| v.to_f64()).collect();
        let scores: Vec<f64> = (0..=position)
            .map(|p| row(kc, p).iter().zip(&qh).map(|(a, b)| a * b).sum::<f64>() * scale)
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let total: f64 = e.iter().sum();
        for (p, w) in e.iter().enumerate() {
            for (o, v) in out[h * dim..][..dim].iter_mut().zip(row(vc, p)) {
                *o += w / total * v;
            }
        }
    }
    out
}

#[test]
fn split_attention_joined_matches_softmax_attention_up_to_the_position() {
    let ctx = context();
    let (heads, kv_heads, dim, capacity) = (16, 2, 128, 512);
    let chunk = 32;
    let chunks = capacity / chunk;
    let group = heads / kv_heads;
    let q = halves(&uniform(heads * dim, 41, 2.0));
    let kc = halves(&uniform(kv_heads * capacity * dim, 43, 2.0));
    let vc = halves(&uniform(kv_heads * capacity * dim, 47, 1.0));
    for subgroup in [64, 32] {
        let split = kernel(
            &ctx,
            "attention_split",
            6,
            Some(subgroup),
            &[
                (0, SpecValue::U32(subgroup)),
                (1, u(group * chunk)),
                (2, u(heads)),
                (3, u(kv_heads)),
                (4, u(dim)),
                (5, u(capacity)),
                (6, SpecValue::F32(1.0 / (dim as f32).sqrt())),
            ],
        );
        let combine = kernel(
            &ctx,
            "attention_combine",
            4,
            None,
            &[
                (0, SpecValue::U32(32)),
                (1, u(dim / 4)),
                (2, u(heads)),
                (3, u(dim)),
                (4, u(capacity)),
            ],
        );
        for position in [0, 5, 31, 32, 33, 100, 446, capacity - 1] {
            // What lies past the position must not reach the output, NaN included.
            let (mut k_here, mut v_here) = (kc.clone(), vc.clone());
            for kv in 0..kv_heads {
                for p in position + 1..capacity {
                    for d in 0..dim {
                        k_here[(kv * capacity + p) * dim + d] = f16::NAN;
                        v_here[(kv * capacity + p) * dim + d] = f16::NAN;
                    }
                }
            }
            let mut qb = buffer_of(&ctx, &q);
            let mut kb = buffer_of(&ctx, &k_here);
            let mut vb = buffer_of(&ctx, &v_here);
            let mut pb = ctx.buffer(4).unwrap();
            pb.write(&[position as u32]);
            let mut ob = ctx.buffer((heads * chunks * dim * 4) as u64).unwrap();
            let mut mlb = ctx.buffer((heads * chunks * 8) as u64).unwrap();
            let mut out = ctx.buffer((heads * dim * 2) as u64).unwrap();
            let split_ns = timed(
                &ctx,
                &split,
                &mut [&mut qb, &mut kb, &mut vb, &mut pb, &mut ob, &mut mlb],
                [chunks as u32, kv_heads as u32, 1],
            );
            let combine_ns = timed(
                &ctx,
                &combine,
                &mut [&mut pb, &mut ob, &mut mlb, &mut out],
                [heads as u32, 1, 1],
            );
            let got = read_f16(&out);
            let want = attention(&q, &kc, &vc, (heads, kv_heads, dim, capacity), position);
            // The f16 rounding of the output, and f32's error in the scores, the
            // exponentials and the weighted sums: a few hundred roundings, each relative to
            // the largest value.
            let bound = |i: usize| ulp16(want[i]) + 1024.0 * U32;
            let bad = past(&got, &want, bound);
            assert!(bad.is_empty(), "wave{subgroup} at {position}: {bad:?}");
            // Defect: the last position left out.
            if position > 0 {
                let short = attention(&q, &kc, &vc, (heads, kv_heads, dim, capacity), position - 1);
                assert!(!past(&got, &short, bound).is_empty(), "at {position}");
            }
            eprintln!(
                "attention wave{subgroup} at {position}: worst at {:.2} of its bound; split \
                 {:.1} us, combine {:.1} us",
                worst(&got, &want, bound),
                split_ns / 1e3,
                combine_ns / 1e3
            );
        }
    }
    assert_no_validation_messages(&ctx);
}
