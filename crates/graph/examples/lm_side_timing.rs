//! Times VoxCPM2's LM side of a patch on the GPU at batch 1: per dispatch from GPU timestamps,
//! summed by part (the base LM, the residual LM, the layers around them) and by label, and whole
//! patches from submission to fence, against the bytes a patch reads.
//!
//! ```text
//! cargo run --release -p graph --example lm_side_timing -- [--context 481] [--capacity 768]
//!     [--steps 30] [--subgroup S]
//! ```
//!
//! The subgroup size defaults to the widest the kernels are written for that the device can
//! require, and the rest of the tuning to [`Tuning::for_subgroup`]'s at it.
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. The patches run at
//! positions from `--context` on, over caches whose earlier positions are zero: attention's cost
//! depends on the position, not on what the caches hold.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::Tuning,
    kernels::Kernels,
    lm_side::{self, LmSide, LmSideShape},
};
use vulkan::Context;

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

/// A local encoder output of about the real ones' scale, from a fixed xorshift sequence.
fn input(width: usize, seed: u32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            half::f16::from_f64(v).to_bits()
        })
        .collect()
}

/// The part of the LM side a dispatch's label puts it in.
fn part(label: &str) -> &str {
    label
        .split_once('.')
        .map(|(part, _)| part)
        .filter(|part| part.ends_with("_lm"))
        .unwrap_or("heads")
}

fn main() {
    let context = arg("--context", 481);
    let capacity = arg("--capacity", 768);
    let steps = arg("--steps", 30);
    let warmup = 5;
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let shape = LmSideShape::new(&config, capacity).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        context + warmup + steps <= capacity,
        "the steps pass the capacity"
    );
    let checkpoint =
        Checkpoint::open(dir.join("model.safetensors")).unwrap_or_else(|e| panic!("{e}"));
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    let device = Tuning::for_device(report).unwrap_or_else(|e| panic!("{e}"));
    let tuning = Tuning::for_subgroup(arg("--subgroup", device.subgroup as usize) as u32);
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), validation {}; {} and {} layers, {tuning:?}",
        report.name,
        report.driver_info,
        if ctx.validating() { "on" } else { "off" },
        shape.base.layers,
        shape.residual.layers,
    );

    // Each dispatch timed, then the same patch untimed, from submission to fence.
    let mut per_part: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut gpu_totals = Vec::new();
    let mut walls = Vec::new();
    for timed in [true, false] {
        let start = Instant::now();
        let mut side = LmSide::new(&ctx, &mut kernels, &shape, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        side.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
        let step = side.graph_step();
        eprintln!(
            "{} recording: {} dispatches, {} barriers, arena {} KiB; built and loaded in {:.1} s",
            if timed { "timed" } else { "untimed" },
            step.dispatches(),
            step.barriers(),
            step.arena_bytes() / 1024,
            start.elapsed().as_secs_f64()
        );
        for i in 0..warmup + steps {
            let enc = input(shape.encoder, i as u32 + 1);
            let start = Instant::now();
            let out = side
                .step(context + i, &enc)
                .unwrap_or_else(|e| panic!("{e}"));
            let wall = start.elapsed().as_secs_f64() * 1e9;
            assert!(out.stop_logits.iter().all(|l| l.is_finite()));
            if i < warmup {
                continue;
            }
            if !timed {
                walls.push(wall);
                continue;
            }
            let timings = side.timings().unwrap().expect("timed");
            let mut parts: BTreeMap<&str, f64> = BTreeMap::new();
            let mut kernels: BTreeMap<&str, f64> = BTreeMap::new();
            for (label, t) in &timings {
                *parts.entry(part(label)).or_default() += t.nanoseconds;
                *kernels.entry(label).or_default() += t.nanoseconds;
            }
            for (part, ns) in parts {
                per_part.entry(part.to_string()).or_default().push(ns);
            }
            for (kernel, ns) in kernels {
                per_kernel.entry(kernel.to_string()).or_default().push(ns);
            }
            gpu_totals.push(timings.iter().map(|(_, t)| t.nanoseconds).sum::<f64>());
        }
    }

    let position = context + warmup + steps / 2;
    let weights = (shape.base.weight_bytes() + shape.residual.weight_bytes()) as f64;
    let (h, d) = (shape.base.hidden, shape.dit);
    let heads =
        2.0 * (h * shape.encoder + 2 * h * shape.latent + 2 * h * h + 2 * d * h + h * h) as f64;
    let cache = (shape.base.cache_bytes(position) + shape.residual.cache_bytes(position)) as f64;
    let bytes = weights + heads + cache;
    println!(
        "bytes read per patch: {:.3} GB of the LMs' weights, {:.1} MB of the layers around them, \
         {:.1} MB of KV cache at position {position}",
        weights / 1e9,
        heads / 1e6,
        cache / 1e6
    );
    let gpu = median(gpu_totals);
    println!("per part and per label, summed over the patch (median of {steps} patches):");
    for (label, ns) in per_part.iter().chain(&per_kernel) {
        let m = median(ns.clone());
        println!("  {label:<30} {:8.3} ms  {:5.1}%", m / 1e6, 100.0 * m / gpu);
    }
    println!(
        "GPU time per patch (timestamps, timed recording): median {:.3} ms, {:.0} GB/s",
        gpu / 1e6,
        bytes / gpu
    );
    let wall = median(walls.clone());
    let best = walls.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "submit to fence per patch (untimed recording, with the host's write and read): median \
         {:.3} ms, best {:.3} ms, {:.0} GB/s at the median",
        wall / 1e6,
        best / 1e6,
        bytes / wall
    );
}
