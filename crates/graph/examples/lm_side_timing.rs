//! Times VoxCPM2's LM side of a patch on the GPU, for a batch of each size `--batch` lists: per
//! dispatch from GPU timestamps, summed by part (the base LM, the residual LM, the layers around
//! them) and by label, and whole patches from the GPU's timestamps around them and from
//! submission to fence, against the bytes a patch reads and the floor a streaming read sets.
//!
//! ```text
//! cargo run --release -p graph --example lm_side_timing -- [OPTIONS]
//! ```
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. The patches run at
//! positions from `--context` on, each request of a batch in a slot of the caches of its own,
//! over caches whose earlier positions are zero: attention's cost depends on the position, not on
//! what the caches hold.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use argh::FromArgs;
use graph::{
    checkpoint::Checkpoint,
    decode::{Batch, Row, Tuning},
    kernels::Kernels,
    lm_side::{self, LmSide, LmSideShape},
};
use list::List;
use stream_read::stream_read;
use vulkan::{Context, F16};

mod list;
mod stream_read;

/// Times VoxCPM2's LM side of a patch on the GPU, for a batch of each size --batch lists.
#[derive(FromArgs)]
struct Args {
    /// the position the first patch runs at, 481 by default
    #[argh(option, default = "481")]
    context: usize,
    /// the positions each slot of the caches holds, 768 by default
    #[argh(option, default = "768")]
    capacity: usize,
    /// the patches timed, after 5 to warm up, 30 by default
    #[argh(option, default = "30")]
    steps: usize,
    /// the batch sizes to time, 1 by default
    #[argh(option, default = "List(vec![1])")]
    batch: List,
    /// the subgroup size, by default the widest the kernels are written for that the device can
    /// require; the rest of the tuning defaults to Tuning::for_batch's at it for each batch
    #[argh(option)]
    subgroup: Option<u32>,
    /// the GEMVs' workgroup width
    #[argh(option)]
    workgroup: Option<u32>,
    /// row pairs per subgroup of the q/k/v projection
    #[argh(option)]
    qkv_pairs: Option<u32>,
    /// row pairs per subgroup of the gate/up projection
    #[argh(option)]
    gate_up_pairs: Option<u32>,
    /// rows per subgroup of the output and down projections
    #[argh(option)]
    residual_rows: Option<u32>,
    /// the steps of its rows each GEMV lane loads before the kernel's prologue
    #[argh(option)]
    ahead: Option<u32>,
    /// the bytes of shared memory a GEMV takes at most
    #[argh(option)]
    stage: Option<u32>,
    /// the rows of a batch a GEMV workgroup takes at most
    #[argh(option)]
    vectors: Option<u32>,
    /// the MiB of device memory the streaming read reads, 512 by default; 0 skips it
    #[argh(option, default = "512")]
    floor_mib: u64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A local encoder output of about the real ones' scale, from a fixed xorshift sequence.
fn input(width: usize, seed: u32) -> Vec<F16> {
    let mut state = seed | 1;
    (0..width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            F16::from_f64(v)
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

/// What a batch's patches took, in nanoseconds: the GPU's time untimed, from the timestamps
/// around each patch, and submission to fence.
struct Measured {
    rows: usize,
    gpu: f64,
    wall: f64,
    bytes: f64,
}

fn main() {
    let args: Args = argh::from_env();
    let (context, capacity, steps) = (args.context, args.capacity, args.steps);
    let batches = &args.batch.0;
    let floor_mib = args.floor_mib;
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
    let subgroup = args.subgroup.unwrap_or(device.subgroup);
    let tuning = |rows: usize| {
        let base = Tuning::for_batch(subgroup, rows);
        Tuning {
            workgroup: args.workgroup.unwrap_or(base.workgroup),
            qkv_pairs: args.qkv_pairs.unwrap_or(base.qkv_pairs),
            gate_up_pairs: args.gate_up_pairs.unwrap_or(base.gate_up_pairs),
            residual_rows: args.residual_rows.unwrap_or(base.residual_rows),
            ahead: args.ahead.unwrap_or(base.ahead),
            stage: args.stage.unwrap_or(base.stage),
            vectors: args.vectors.unwrap_or(base.vectors),
            ..base
        }
    };
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), validation {}; {} and {} layers",
        report.name,
        report.driver_info,
        if ctx.validating() { "on" } else { "off" },
        shape.base.layers,
        shape.residual.layers,
    );
    let floor = (floor_mib > 0).then(|| stream_read(&ctx, &mut kernels, subgroup, floor_mib << 20));

    let position = context + warmup + steps / 2;
    let weights = (shape.base.weight_bytes() + shape.residual.weight_bytes()) as f64;
    let (h, d) = (shape.base.hidden, shape.dit);
    let heads =
        2.0 * (h * shape.encoder + 2 * h * shape.latent + 2 * h * h + 2 * d * h + h * h) as f64;
    let cache = (shape.base.cache_bytes(position) + shape.residual.cache_bytes(position)) as f64;
    println!(
        "bytes read per patch: {:.3} GB of the LMs' weights, {:.1} MB of the layers around them, \
         {:.1} MB of KV cache at position {position} for each request",
        weights / 1e9,
        heads / 1e6,
        cache / 1e6
    );
    let mut measured = Vec::new();
    for &rows in batches {
        let batch = Batch { rows, slots: rows };
        let tuning = tuning(rows);
        let start = Instant::now();
        let mut side = LmSide::with_batch(&ctx, &mut kernels, &shape, &tuning, batch, false)
            .unwrap_or_else(|e| panic!("{e}"));
        side.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
        let step = side.graph_step();
        eprintln!(
            "{rows} rows at {tuning:?}: {} dispatches, {} barriers, arena {} KiB; built and \
             loaded in {:.1} s",
            step.dispatches(),
            step.barriers(),
            step.arena_bytes() / 1024,
            start.elapsed().as_secs_f64()
        );
        let enc: Vec<F16> = (0..rows as u32)
            .flat_map(|r| input(shape.encoder, r + 1))
            .collect();
        let placed = |i: usize| -> Vec<Option<Row>> {
            (0..rows)
                .map(|slot| {
                    Some(Row {
                        position: context + i,
                        slot,
                    })
                })
                .collect()
        };

        // The same patches untimed, then each dispatch timed.
        let (mut gpu, mut walls) = (Vec::new(), Vec::new());
        let mut per_part: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut timed_totals = Vec::new();
        for timed in [false, true] {
            if timed {
                side = side
                    .retuned(&ctx, &mut kernels, &tuning, true)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
            for i in 0..warmup + steps {
                let start = Instant::now();
                let out = side
                    .step_rows(&placed(i), &enc)
                    .unwrap_or_else(|e| panic!("{e}"));
                let wall = start.elapsed().as_secs_f64() * 1e9;
                assert!(
                    out.iter()
                        .flatten()
                        .all(|c| c.stop_logits.iter().all(|l| l.is_finite()))
                );
                if i < warmup {
                    continue;
                }
                if !timed {
                    let elapsed = side.graph_step().elapsed();
                    gpu.push(elapsed.unwrap_or_else(|e| panic!("{e}")).nanoseconds);
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
                timed_totals.push(timings.iter().map(|(_, t)| t.nanoseconds).sum::<f64>());
            }
        }
        let timed = median(timed_totals);
        println!("{rows} rows, per part and per label, summed over the patch (median of {steps}):");
        for (label, ns) in per_part.iter().chain(&per_kernel) {
            let m = median(ns.clone());
            println!(
                "  {label:<30} {:8.3} ms  {:5.1}%",
                m / 1e6,
                100.0 * m / timed
            );
        }
        let bytes = weights + heads + rows as f64 * cache;
        let (gpu, wall) = (median(gpu), median(walls));
        println!(
            "{rows} rows: GPU time per patch {:.3} ms untimed, {:.3} ms timed per dispatch; \
             submit to fence {wall_ms:.3} ms",
            gpu / 1e6,
            timed / 1e6,
            wall_ms = wall / 1e6
        );
        measured.push(Measured {
            rows,
            gpu,
            wall,
            bytes,
        });
    }

    println!(
        "per patch, untimed: rows, GB read, GPU ms, GB/s, submit to fence ms, GPU ms per request{}",
        if floor.is_some() {
            ", floor ms, % of the floor"
        } else {
            ""
        }
    );
    for m in &measured {
        let at_floor = floor.map_or(String::new(), |f| {
            format!(
                ", {:.3}, {:.0}%",
                m.bytes / f / 1e6,
                100.0 * (m.bytes / f) / m.gpu
            )
        });
        println!(
            "  {:2}, {:.3}, {:.3}, {:.0}, {:.3}, {:.3}{at_floor}",
            m.rows,
            m.bytes / 1e9,
            m.gpu / 1e6,
            m.bytes / m.gpu,
            m.wall / 1e6,
            m.gpu / 1e6 / m.rows as f64
        );
    }
    if let Some(f) = floor {
        println!("floor: stream_read at its best, {f:.0} GB/s");
    }
    let messages = ctx.validation_messages();
    if ctx.validating() {
        println!("validation: {} messages", messages.len());
        for m in messages {
            println!("  {m}");
        }
    }
}
