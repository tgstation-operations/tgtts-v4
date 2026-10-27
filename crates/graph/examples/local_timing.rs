//! Times the patch sampler and the local encoder on the GPU at batch 1: per dispatch from GPU
//! timestamps, summed by kernel, and whole samples from submission to fence, against the bytes
//! of weights they read and against the floor a streaming read of device memory sets.
//!
//! ```text
//! cargo run --release -p graph --example local_timing -- [--timesteps 10] [--runs 20]
//!     [--subgroup S] [--workgroup W] [--qkv-pairs P] [--gate-up-pairs P] [--o-rows R]
//!     [--down-rows R] [--chunk C] [--split 0|1] [--groups G] [--warmup-ms 2000]
//!     [--floor-mib 512]
//! ```
//!
//! The tuning defaults to [`Tuning::for_device`]'s. `--floor-mib 0` skips the streaming read.
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. What is sampled
//! does not change the time: the inputs are fixed sequences of about their scale.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use graph::{
    checkpoint::Checkpoint,
    graph::Pipeline,
    kernels::{Kernels, STREAM_READ},
    local::{PatchConfig, Tuning, dit::Sampler, encoder::Encoder},
};
use vulkan::{Context, Residence, SpecValue, Timing, memory::STAGING_BYTES};

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .map(|i| {
            args[i + 1]
                .parse()
                .unwrap_or_else(|e| panic!("{name}: {e}"))
        })
        .unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Values in [-scale, scale) from a fixed xorshift sequence.
fn uniform(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The best rate, in GB/s, at which `stream_read` reads `bytes` of device memory, over 1, 4 and
/// 16 loads per invocation in workgroups of 256, each the best of nine dispatches after reading
/// for `warmup`.
fn stream_read(
    ctx: &Context,
    kernels: &mut Kernels,
    subgroup: u32,
    bytes: u64,
    warmup: Duration,
) -> f64 {
    let mut data = ctx
        .buffer_in(bytes, Residence::Device)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut state = 0x2545_f491_u32;
    let chunk = (STAGING_BYTES / 4) as usize;
    for at in (0..bytes / 4).step_by(chunk) {
        let words: Vec<u32> = (0..chunk.min((bytes / 4 - at) as usize))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            })
            .collect();
        data.write_at(4 * at, &words)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let workgroup = 256;
    let mut best = 0f64;
    for loads in [1, 4, 16] {
        let pipeline = Pipeline {
            kernel: &STREAM_READ,
            constants: vec![
                (0, SpecValue::U32(subgroup)),
                (1, SpecValue::U32(workgroup)),
                (2, SpecValue::U32(loads)),
            ],
            subgroup_size: Some(subgroup),
        };
        let kernel = kernels.get(&pipeline).unwrap_or_else(|e| panic!("{e}"));
        let groups = bytes / 16 / u64::from(workgroup * loads);
        let mut out = ctx
            .buffer_in(
                4 * groups * u64::from(workgroup / subgroup),
                Residence::Readback,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let warming = Instant::now();
        while warming.elapsed() < warmup {
            ctx.dispatch(&kernel, &mut [&mut data, &mut out], [groups as u32, 1, 1])
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let rates: Vec<f64> = (0..10)
            .map(|_| {
                let t = ctx
                    .dispatch(&kernel, &mut [&mut data, &mut out], [groups as u32, 1, 1])
                    .unwrap_or_else(|e| panic!("{e}"));
                bytes as f64 / t.nanoseconds
            })
            .collect();
        best = best.max(rates[1..].iter().copied().fold(0.0, f64::max));
    }
    best
}

/// A dispatch's kernel, from its label: the last part of it.
fn kernel_of(label: &str) -> &str {
    label.rsplit('.').next().unwrap_or(label)
}

/// Each kernel's GPU time summed over a run, and the run's total.
fn sums(timings: &[(&str, Timing)]) -> (BTreeMap<String, f64>, f64) {
    let mut sums: BTreeMap<String, f64> = BTreeMap::new();
    for (label, t) in timings {
        *sums.entry(kernel_of(label).to_string()).or_default() += t.nanoseconds;
    }
    (sums, timings.iter().map(|(_, t)| t.nanoseconds).sum())
}

fn report(
    what: &str,
    per_kernel: BTreeMap<String, Vec<f64>>,
    gpu: Vec<f64>,
    walls: Vec<f64>,
    bytes: &BTreeMap<&str, f64>,
    floor: Option<f64>,
) {
    let of_floor = |rate: f64| {
        floor.map_or(String::new(), |f| {
            format!(", {:.0}% of the floor", 100.0 * rate / f)
        })
    };
    let total: f64 = bytes.values().sum();
    let gpu = median(gpu);
    println!(
        "{what}: {:.1} MB of weights read per run; per kernel (median of {} runs):",
        total / 1e6,
        walls.len()
    );
    for (kernel, ns) in per_kernel {
        let m = median(ns);
        let rate = bytes.get(kernel.as_str()).map_or(String::new(), |&b| {
            format!("  {:4.0} GB/s{}", b / m, of_floor(b / m))
        });
        println!(
            "  {kernel:<12} {:8.3} ms  {:5.1}%{rate}",
            m / 1e6,
            100.0 * m / gpu
        );
    }
    let wall = median(walls.clone());
    let best = walls.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "{what}: GPU time per run (timestamps) median {:.3} ms, {:.0} GB/s{}; submit to fence \
         median {:.3} ms, best {:.3} ms, {:.0} GB/s at the median{}",
        gpu / 1e6,
        total / gpu,
        of_floor(total / gpu),
        wall / 1e6,
        best / 1e6,
        total / wall,
        of_floor(total / wall)
    );
    if let Some(f) = floor {
        println!(
            "{what}: the floor, {total:.0} bytes at {f:.0} GB/s, is {:.3} ms",
            total / f / 1e6
        );
    }
}

fn main() {
    let timesteps = arg("--timesteps", 10);
    let runs = arg("--runs", 20);
    let floor_mib = arg("--floor-mib", 512) as u64;
    // The GPU's clocks ramp up over a second or two of load and fall back when it idles, as
    // between the recordings: each recording, and the streaming read, runs this long before it
    // is timed.
    let warmup = Duration::from_millis(arg("--warmup-ms", 2000) as u64);
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let checkpoint =
        Checkpoint::open(dir.join("model.safetensors")).unwrap_or_else(|e| panic!("{e}"));
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let device_report = ctx.report();
    let device = Tuning::for_device(device_report).unwrap_or_else(|e| panic!("{e}"));
    let base = match arg("--subgroup", 0) {
        0 => device,
        s => Tuning {
            subgroup: s as u32,
            ..device
        },
    };
    let tuning = Tuning {
        workgroup: arg("--workgroup", base.workgroup as usize) as u32,
        qkv_pairs: arg("--qkv-pairs", base.qkv_pairs as usize) as u32,
        gate_up_pairs: arg("--gate-up-pairs", base.gate_up_pairs as usize) as u32,
        o_rows: arg("--o-rows", base.o_rows as usize) as u32,
        down_rows: arg("--down-rows", base.down_rows as usize) as u32,
        chunk: arg("--chunk", base.chunk as usize) as u32,
        split: arg("--split", usize::from(base.split)) != 0,
        groups: arg("--groups", base.groups as usize) as u32,
        ..base
    };
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), validation {}; {timesteps} timesteps, {tuning:?}",
        device_report.name,
        device_report.driver_info,
        if ctx.validating() { "on" } else { "off" },
    );
    let floor = (floor_mib > 0)
        .then(|| stream_read(&ctx, &mut kernels, tuning.subgroup, floor_mib << 20, warmup));
    if let Some(f) = floor {
        println!("floor: stream_read at its best, {f:.0} GB/s");
    }

    let hidden = config.dit_config.stack.hidden_dim;
    let len = config.feat_dim * config.patch_size;
    let mu: Vec<u16> = uniform(2 * hidden, 1, 1.7)
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let cond = uniform(len, 2, 1.0);
    let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let (mut gpu, mut walls) = (Vec::new(), Vec::new());
    let mut steps = 0;
    let mut stack_bytes = BTreeMap::new();
    for timed in [true, false] {
        let start = Instant::now();
        let mut sampler = Sampler::new(&ctx, &mut kernels, &config, timesteps, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        sampler.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
        let t = sampler.tensors();
        steps = t.steps.len();
        stack_bytes = layer_bytes(&t.stack.shape, steps);
        eprintln!(
            "sampler, {}: {} dispatches, {} barriers, arena {} KiB; built and loaded in {:.1} s",
            if timed { "timed" } else { "untimed" },
            sampler.graph_step().dispatches(),
            sampler.graph_step().barriers(),
            sampler.graph_step().arena_bytes() / 1024,
            start.elapsed().as_secs_f64()
        );
        let warming = Instant::now();
        let mut i = 0;
        while warming.elapsed() < warmup {
            let noise = uniform(len, 10 + i, 1.7);
            sampler
                .sample(&mu, &cond, &noise, 1.0, 2.0)
                .unwrap_or_else(|e| panic!("{e}"));
            i += 1;
        }
        for i in 0..runs {
            let noise = uniform(len, 10 + i as u32, 1.7);
            let start = Instant::now();
            let patch = sampler
                .sample(&mu, &cond, &noise, 1.0, 2.0)
                .unwrap_or_else(|e| panic!("{e}"));
            let wall = start.elapsed().as_secs_f64() * 1e9;
            assert!(patch.iter().all(|v| v.is_finite()));
            if timed {
                let timings = sampler.timings().unwrap().expect("timed");
                let (sums, total) = sums(&timings);
                for (kernel, ns) in sums {
                    per_kernel.entry(kernel).or_default().push(ns);
                }
                gpu.push(total);
            } else {
                walls.push(wall);
            }
        }
    }
    let small = |n: usize, k: usize| (2 * (n * k + n)) as f64;
    let (c, h) = (config.feat_dim, hidden);
    let mut bytes = stack_bytes;
    bytes.insert("in_proj", steps as f64 * small(h, c));
    bytes.insert("cond_proj", small(h, c));
    bytes.insert("euler", steps as f64 * (small(c, h) + 2.0 * h as f64));
    report(
        &format!("sampler, {steps} runs of the DiT"),
        per_kernel,
        gpu,
        walls,
        &bytes,
        floor,
    );

    let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let (mut gpu, mut walls) = (Vec::new(), Vec::new());
    let mut bytes = BTreeMap::new();
    for timed in [true, false] {
        let mut encoder = Encoder::new(&ctx, &mut kernels, &config, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        encoder.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
        bytes = layer_bytes(&encoder.tensors().stack.shape, 1);
        bytes.insert("in_proj", small(h, c));
        bytes.insert("norm", 2.0 * h as f64);
        let warming = Instant::now();
        while warming.elapsed() < warmup {
            encoder
                .encode(&uniform(len, 99, 1.0))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        for i in 0..runs {
            let patch = uniform(len, 100 + i as u32, 1.0);
            let start = Instant::now();
            encoder.encode(&patch).unwrap_or_else(|e| panic!("{e}"));
            let wall = start.elapsed().as_secs_f64() * 1e9;
            if timed {
                let timings = encoder.timings().unwrap().expect("timed");
                let (sums, total) = sums(&timings);
                for (kernel, ns) in sums {
                    per_kernel.entry(kernel).or_default().push(ns);
                }
                gpu.push(total);
            } else {
                walls.push(wall);
            }
        }
    }
    report("encoder", per_kernel, gpu, walls, &bytes, floor);
    if ctx.validating() {
        let messages = ctx.validation_messages();
        println!("the validation layer reported {} messages", messages.len());
        for m in messages {
            println!("  {m}");
        }
    }
}

/// The bytes each of a stack's kernels reads over `runs` runs of its layers.
fn layer_bytes(s: &graph::decode::Shape, runs: usize) -> BTreeMap<&'static str, f64> {
    let (h, i, d, l) = (s.hidden, s.inner, s.head_dim, s.layers);
    let half = |elements: usize| (2 * l * runs * elements) as f64;
    BTreeMap::from([
        ("qkv_rope", half((s.heads + 2 * s.kv_heads) * d * h + h)),
        ("o_proj", half(h * s.heads * d)),
        ("gate_up", half(2 * i * h + h)),
        ("down_proj", half(h * i)),
    ])
}
