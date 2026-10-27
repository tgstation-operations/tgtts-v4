//! Times the base LM's decode step on the GPU at batch 1: per dispatch from GPU timestamps,
//! summed by kernel, and whole steps from submission to fence, against the bytes a step reads
//! and against the floor a streaming read of device memory sets.
//!
//! ```text
//! cargo run --release -p graph --example decode_timing -- [--layers 28] [--context 481]
//!     [--capacity 768] [--steps 30] [--subgroup S] [--workgroup W] [--qkv-pairs 1]
//!     [--gate-up-pairs 1] [--residual-rows 1] [--floor-mib 512]
//! ```
//!
//! The subgroup size defaults to the widest the kernels are written for that the device can
//! require, and the rest of the tuning to [`Tuning::for_subgroup`]'s at it. `--floor-mib 0`
//! skips the streaming read.
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. The steps run at
//! positions from `--context` on, over a cache whose earlier positions are zero: attention's cost
//! depends on the position, not on what the cache holds.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, Shape, Tuning},
    graph::Pipeline,
    kernels::{Kernels, STREAM_READ},
};
use vulkan::{Context, Residence, SpecValue, memory::STAGING_BYTES};

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

/// A hidden state of the base LM's scale, from a fixed xorshift sequence.
fn input(hidden: usize, seed: u32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..hidden)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            half::f16::from_f64(v * 0.1).to_bits()
        })
        .collect()
}

/// The best rate, in GB/s, at which `stream_read` reads `bytes` of device memory, over 1, 4 and
/// 16 loads per invocation in workgroups of 256, each the best of nine dispatches after one.
fn stream_read(ctx: &Context, kernels: &mut Kernels, subgroup: u32, bytes: u64) -> f64 {
    let mut data = ctx
        .buffer_in(bytes, Residence::Device)
        .unwrap_or_else(|e| panic!("{e}"));
    // Words that do not repeat, filled a staging buffer at a time.
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
        let mut rates = Vec::new();
        for _ in 0..10 {
            let t = ctx
                .dispatch(&kernel, &mut [&mut data, &mut out], [groups as u32, 1, 1])
                .unwrap_or_else(|e| panic!("{e}"));
            rates.push(bytes as f64 / t.nanoseconds);
        }
        let rate = rates[1..].iter().copied().fold(0.0, f64::max);
        println!(
            "stream_read of {} MiB, {loads} loads per invocation: best {rate:.0} GB/s",
            bytes >> 20
        );
        best = best.max(rate);
    }
    best
}

/// The bytes each kernel reads in a step at `position`, summed over the layers: its weights,
/// and attention's keys and values up to the position.
fn kernel_bytes(s: &Shape, position: usize) -> BTreeMap<&'static str, f64> {
    let (h, i, d, l) = (s.hidden, s.inner, s.head_dim, s.layers);
    let half = |elements: usize| (2 * l * elements) as f64;
    BTreeMap::from([
        ("qkv_rope", half((s.heads + 2 * s.kv_heads) * d * h)),
        ("o_proj", half(h * s.heads * d)),
        ("gate_up", half(2 * i * h)),
        ("down_proj", half(h * i)),
        ("attention_split", s.cache_bytes(position) as f64),
    ])
}

fn main() {
    let layers = arg("--layers", 28);
    let context = arg("--context", 481);
    let capacity = arg("--capacity", 768);
    let steps = arg("--steps", 30);
    let floor_mib = arg("--floor-mib", 512) as u64;
    let warmup = 5;
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let shape = Shape::new(&config, layers, capacity).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        context + warmup + steps <= capacity,
        "the steps pass the capacity"
    );
    let checkpoint =
        Checkpoint::open(dir.join("model.safetensors")).unwrap_or_else(|e| panic!("{e}"));
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    let device = Tuning::for_device(report).unwrap_or_else(|e| panic!("{e}"));
    let base = Tuning::for_subgroup(arg("--subgroup", device.subgroup as usize) as u32);
    let tuning = Tuning {
        workgroup: arg("--workgroup", base.workgroup as usize) as u32,
        qkv_pairs: arg("--qkv-pairs", base.qkv_pairs as usize) as u32,
        gate_up_pairs: arg("--gate-up-pairs", base.gate_up_pairs as usize) as u32,
        residual_rows: arg("--residual-rows", base.residual_rows as usize) as u32,
        ..base
    };
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), subgroups of {} to {}, validation {}, robustBufferAccess2 {}; {layers} layers, \
         {tuning:?}",
        report.name,
        report.driver_info,
        report.min_subgroup_size,
        report.max_subgroup_size,
        if ctx.validating() { "on" } else { "off" },
        ctx.robust_buffer_access2()
    );
    let floor =
        (floor_mib > 0).then(|| stream_read(&ctx, &mut kernels, tuning.subgroup, floor_mib << 20));

    // Each dispatch timed, then the same step untimed, from submission to fence.
    let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut gpu_totals = Vec::new();
    let mut walls = Vec::new();
    let mut host = Vec::new();
    let mut records = Vec::new();
    for timed in [true, false] {
        let start = Instant::now();
        let mut decoder = Decoder::new(&ctx, &mut kernels, &shape, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        decoder
            .load(&checkpoint, "base_lm.")
            .unwrap_or_else(|e| panic!("{e}"));
        records.push(decoder.graph_step().recorded_in().as_secs_f64() * 1e3);
        eprintln!(
            "{} recording: {} dispatches, {} barriers, arena {} KiB; built and loaded in {:.1} s",
            if timed { "timed" } else { "untimed" },
            decoder.graph_step().dispatches(),
            decoder.graph_step().barriers(),
            decoder.graph_step().arena_bytes() / 1024,
            start.elapsed().as_secs_f64()
        );
        for i in 0..warmup + steps {
            let x = input(shape.hidden, i as u32 + 1);
            let start = Instant::now();
            let out = decoder
                .step(context + i, &x)
                .unwrap_or_else(|e| panic!("{e}"));
            let wall = start.elapsed().as_secs_f64() * 1e9;
            assert!(
                out.iter()
                    .all(|&h| half::f16::from_bits(h).to_f32().is_finite())
            );
            if i < warmup {
                continue;
            }
            if timed {
                let timings = decoder.timings().unwrap().expect("timed");
                let mut sums: BTreeMap<&str, f64> = BTreeMap::new();
                for (label, t) in &timings {
                    *sums.entry(label).or_default() += t.nanoseconds;
                }
                for (label, ns) in sums {
                    per_kernel.entry(label.to_string()).or_default().push(ns);
                }
                let gpu = timings.iter().map(|(_, t)| t.nanoseconds).sum::<f64>();
                gpu_totals.push(gpu);
                host.push(wall - gpu);
            } else {
                walls.push(wall);
            }
        }
    }

    let position = context + warmup + steps / 2;
    let weights = shape.weight_bytes() as f64;
    let cache = shape.cache_bytes(position) as f64;
    let bytes = weights + cache;
    let of_floor = |rate: f64| {
        floor.map_or(String::new(), |f| {
            format!(", {:.0}% of the floor", 100.0 * rate / f)
        })
    };
    println!(
        "bytes read per step: {:.3} GB of weights, {:.1} MB of KV cache at position {position}",
        weights / 1e9,
        cache / 1e6,
    );
    if let Some(f) = floor {
        println!("floor: stream_read at its best, {f:.0} GB/s");
    }
    println!("per kernel, summed over the step's layers (median of {steps} steps):");
    let gpu = median(gpu_totals.clone());
    let read = kernel_bytes(&shape, position);
    for (label, ns) in &per_kernel {
        let m = median(ns.clone());
        let rate = read.get(label.as_str()).map_or(String::new(), |&b| {
            format!("  {:4.0} GB/s{}", b / m, of_floor(b / m))
        });
        println!(
            "  {label:<18} {:8.3} ms  {:5.1}%{rate}",
            m / 1e6,
            100.0 * m / gpu
        );
    }
    let wall = median(walls.clone());
    let best = walls.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "GPU time per step (timestamps, timed recording): median {:.3} ms, {:.0} GB/s{}",
        gpu / 1e6,
        bytes / gpu,
        of_floor(bytes / gpu)
    );
    println!(
        "host per step, beyond the GPU's timestamps (writing the input and position, submitting, \
         waking on the fence, reading the output): median {:.1} us; recording the step once \
         (descriptor sets and commands): {:.2} ms",
        median(host) / 1e3,
        records.iter().copied().fold(f64::INFINITY, f64::min)
    );
    println!(
        "submit to fence per step (untimed recording, with the host's write and read): median \
         {:.3} ms, best {:.3} ms, {:.0} GB/s at the median{}",
        wall / 1e6,
        best / 1e6,
        bytes / wall,
        of_floor(bytes / wall)
    );
}
