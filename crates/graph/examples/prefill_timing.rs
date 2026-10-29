//! Times prompts' prefills through the base LM on the GPU at batch 1: per dispatch from GPU
//! timestamps, summed by kernel, and whole runs from submission to fence.
//!
//! ```text
//! cargo run --release -p graph --example prefill_timing -- [OPTIONS]
//! ```
//!
//! `VOXCPM2_DIR` names the checkpoint, and `GLSL_KERNELS` and `VULKAN_KERNELS` the compiled
//! kernels. A prompt is hidden states of nothing in particular at the scale of the base LM's
//! embeddings, from position 0: a prefill's cost depends on its length, not on what it holds.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use argh::FromArgs;
use graph::{
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, Shape, Tuning},
    kernels::Kernels,
    prefill::{Prefill, TILE},
};
use list::List;
use vulkan::{Context, F16};

mod list;

/// Times prompts' prefills through the base LM on the GPU at batch 1.
#[derive(FromArgs)]
struct Args {
    /// the base LM's layers to run, 28 by default
    #[argh(option, default = "28")]
    layers: usize,
    /// the prompts' lengths, 128,256,512 by default
    #[argh(option, default = "List(vec![128, 256, 512])")]
    lengths: List,
    /// the runs timed of each length, after one to warm up, 5 by default
    #[argh(option, default = "5")]
    runs: usize,
    /// the subgroup size, by default the widest the kernels are written for that the device can
    /// require
    #[argh(option)]
    subgroup: Option<u32>,
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `len` hidden states of the base LM's embeddings' scale, from a fixed xorshift sequence.
fn prompt(len: usize, hidden: usize) -> Vec<F16> {
    let mut state = 0x9e37_79b9_u32;
    (0..len * hidden)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            F16::from_f64(v * 0.05)
        })
        .collect()
}

fn main() {
    let args: Args = argh::from_env();
    let (layers, lengths, runs) = (args.layers, &args.lengths.0, args.runs);
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let longest = lengths.iter().copied().max().expect("a length");
    let max_rows = longest.next_multiple_of(TILE);
    let shape = Shape::new(&config, layers, (longest + 1).next_multiple_of(32))
        .unwrap_or_else(|e| panic!("{e}"));
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    let device = Tuning::for_device(report).unwrap_or_else(|e| panic!("{e}"));
    let tuning = Tuning::for_subgroup(args.subgroup.unwrap_or(device.subgroup));
    eprintln!(
        "{} ({}), validation {}; {layers} layers, programs of up to {max_rows} rows, the \
         kernels at {tuning:?}",
        report.name,
        report.driver_info,
        if ctx.validating() { "on" } else { "off" },
    );
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let start = Instant::now();
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint =
        Checkpoint::open(dir.join("model.safetensors")).unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, "base_lm.")
        .unwrap_or_else(|e| panic!("{e}"));
    let prefill = Prefill::new(&ctx, &mut kernels, &mut decoder, max_rows, &tuning, true)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "loaded and recorded in {:.1} s, {} pipelines",
        start.elapsed().as_secs_f64(),
        kernels.len()
    );

    let (h, i) = (shape.hidden, shape.inner);
    let qkv = (shape.heads + 2 * shape.kv_heads) * shape.head_dim;
    // The columns of C and of A of each projection.
    let projections = BTreeMap::from([
        ("qkv_proj", (qkv, h)),
        ("o_proj", (h, shape.heads * shape.head_dim)),
        ("gate_up_proj", (2 * i, h)),
        ("down_proj", (h, i)),
    ]);
    for &len in lengths {
        let x = prompt(len, h);
        let rows = len.next_multiple_of(TILE);
        let mut hosts = Vec::new();
        let mut gpus = Vec::new();
        let mut by_kernel = BTreeMap::<String, Vec<f64>>::new();
        let mut longest_dispatch = 0f64;
        for run in 0..=runs {
            let start = Instant::now();
            prefill
                .run(&mut decoder, 0, &x)
                .unwrap_or_else(|e| panic!("{e}"));
            let host = start.elapsed().as_secs_f64() * 1e3;
            let timings = decoder
                .timings()
                .unwrap_or_else(|e| panic!("{e}"))
                .expect("a timed prefill");
            if run == 0 {
                continue;
            }
            let mut sums = BTreeMap::<&str, f64>::new();
            for (label, t) in &timings {
                let ms = t.nanoseconds / 1e6;
                *sums.entry(label).or_default() += ms;
                longest_dispatch = longest_dispatch.max(ms);
            }
            gpus.push(sums.values().sum::<f64>());
            hosts.push(host);
            for (label, ms) in sums {
                by_kernel.entry(label.to_string()).or_default().push(ms);
            }
        }
        println!(
            "{len} positions in {rows} rows: {:.1} ms on the GPU, {:.1} ms on the host, the \
             median of {runs}; the longest dispatch {longest_dispatch:.2} ms",
            median(&gpus),
            median(&hosts)
        );
        for (label, ms) in &by_kernel {
            let ms = median(ms);
            let rate = projections
                .get(label.as_str())
                .map_or(String::new(), |&(n, k)| {
                    let flops = 2.0 * (rows * n * k * layers) as f64;
                    format!(", {:.1} TFLOP/s over the rows run", flops / ms / 1e9)
                });
            println!("  {label}: {ms:.2} ms{rate}");
        }
    }
}
