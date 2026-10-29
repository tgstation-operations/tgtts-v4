//! Times the audio VAE's decoder on the GPU, a chunk of latent frames at a time: per dispatch
//! from GPU timestamps, summed by layer kind, and whole chunks from submission to fence.
//!
//! ```text
//! cargo run --release -p graph --example vae_timing -- [--chunk 25] [--chunks 40]
//! ```
//!
//! `VOXCPM2_DIR` names the checkpoint, `VOICES_DIR` a voices directory whose first voice's
//! latents are decoded over and over, and `GLSL_KERNELS` the compiled kernels.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use graph::{
    graph::Kind,
    kernels::Kernels,
    pth::Pth,
    vae::{VaeConfig, VaeDecoder, VaeShape, vae_graph},
};
use voices::Voices;
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

fn main() {
    let chunk = arg("--chunk", 25);
    let chunks = arg("--chunks", 40);
    let warmup = 3;
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config: VaeConfig = serde_json::from_value(json["audio_vae_config"].clone()).unwrap();
    let shape = VaeShape::new(&config, chunk, 1).unwrap_or_else(|e| panic!("{e}"));
    let weights = Pth::open(dir.join("audiovae.pth")).unwrap_or_else(|e| panic!("{e}"));
    let voices = Voices::open(std::env::var_os("VOICES_DIR").expect("VOICES_DIR"))
        .unwrap_or_else(|e| panic!("{e}"));
    let voice = voices.names().next().expect("a voice").to_owned();
    let latent = voices.latent(&voice).unwrap_or_else(|e| panic!("{e}"));
    let frames = latent.frames.as_flattened();
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    eprintln!(
        "{} ({}), validation {}, robustBufferAccess2 {}; chunks of {chunk} frames of {voice}'s \
         {} frames",
        report.name,
        report.driver_info,
        if ctx.validating() { "on" } else { "off" },
        ctx.robust_buffer_access2(),
        latent.frames.len()
    );
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));

    // Each dispatch timed, then the same chunks untimed, from submission to fence.
    let mut per_kind: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut gpu_totals = Vec::new();
    let mut slowest: (String, f64) = (String::new(), 0.0);
    let mut walls = Vec::new();
    for timed in [true, false] {
        let start = Instant::now();
        let mut decoder =
            VaeDecoder::new(&ctx, &mut kernels, &shape, timed).unwrap_or_else(|e| panic!("{e}"));
        decoder
            .load(|name| weights.floats(&format!("decoder.{name}")))
            .unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "{} recording: {} dispatches, {} barriers, arena {} KiB; built and loaded in {:.1} s",
            if timed { "timed" } else { "untimed" },
            decoder.graph_step().dispatches(),
            decoder.graph_step().barriers(),
            decoder.graph_step().arena_bytes() / 1024,
            start.elapsed().as_secs_f64()
        );
        let mut at = 0;
        for i in 0..warmup + chunks {
            let latents: Vec<f32> = frames
                .iter()
                .cycle()
                .skip(at * shape.latent)
                .take(chunk * shape.latent)
                .copied()
                .collect();
            at = (at + chunk) % latent.frames.len();
            let start = Instant::now();
            let audio = decoder.push(0, &latents).unwrap_or_else(|e| panic!("{e}"));
            let wall = start.elapsed().as_secs_f64() * 1e9;
            assert!(audio.iter().all(|v| v.is_finite()));
            if i < warmup {
                continue;
            }
            if !timed {
                walls.push(wall);
                continue;
            }
            let timings = decoder.timings().unwrap().expect("timed");
            let mut sums: BTreeMap<&str, f64> = BTreeMap::new();
            for (label, t) in &timings {
                let kind = label.rsplit('.').next().unwrap_or(label);
                *sums.entry(kind).or_default() += t.nanoseconds;
                if t.nanoseconds > slowest.1 {
                    slowest = (label.to_string(), t.nanoseconds);
                }
            }
            for (kind, ns) in sums {
                per_kind.entry(kind.to_string()).or_default().push(ns);
            }
            gpu_totals.push(timings.iter().map(|(_, t)| t.nanoseconds).sum::<f64>());
        }
    }

    let tensors = vae_graph(&shape)
        .unwrap_or_else(|e| panic!("{e}"))
        .graph
        .tensors;
    let bytes = |kind| -> u64 {
        tensors
            .iter()
            .filter(|t| t.kind == kind)
            .map(|t| t.bytes())
            .sum()
    };
    println!(
        "weights {:.1} MB, rings {:.1} KB",
        bytes(Kind::Constant) as f64 / 1e6,
        bytes(Kind::State) as f64 / 1e3
    );
    let gpu = median(gpu_totals);
    println!("per kind of dispatch, summed over a chunk (median of {chunks} chunks):");
    for (kind, ns) in &per_kind {
        let m = median(ns.clone());
        println!("  {kind:<10} {:8.3} ms  {:5.1}%", m / 1e6, 100.0 * m / gpu);
    }
    println!(
        "slowest dispatch of any chunk: {} at {:.3} ms",
        slowest.0,
        slowest.1 / 1e6
    );
    println!(
        "GPU time per chunk (timestamps, timed recording): median {:.3} ms",
        gpu / 1e6
    );
    let wall = median(walls.clone());
    println!(
        "submit to fence per chunk (untimed recording, with the host's write and read): median \
         {:.3} ms, best {:.3} ms; {:.1} s of audio a second",
        wall / 1e6,
        walls.iter().copied().fold(f64::INFINITY, f64::min) / 1e6,
        chunk as f64 / 25.0 / (wall / 1e9)
    );
}
