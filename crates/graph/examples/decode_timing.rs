//! Times the base LM's decode step on the GPU at batch 1, and says what a step's wall time is made
//! of: the GPU's time from timestamps, and the host's around it, which is writing the input,
//! submitting, the queue's delay before the GPU starts, waking on the fence after it ends, and
//! reading the output. The delay and the wake are placed on the host's clock where the device
//! offers calibrated timestamps. In order, after the streaming-read floor and the weights' load:
//!
//! - the warm-up from the load, step by step, which shows the GPU's clocks ramping up;
//! - steps back to back, once per fence spin of `--spins-us`, which tells the host's wake from
//!   the GPU's time;
//! - steps with the host idle between them, once per gap of `--gaps-ms`, and steps after
//!   `--idle-ms` of idling, which show what idling costs the steps after it;
//! - the step recorded untimed and timed per dispatch in turn, `--rounds` times, and each
//!   kernel's time summed over the layers;
//! - a sweep over the tunings `--sweep-workgroups`, `--sweep-rows` (row pairs or rows to a
//!   subgroup, of every GEMV), `--sweep-ahead`, `--sweep-stage` and `--sweep-vectors` make
//!   together, each recorded over the same weights, `--rounds` times round robin, with what the
//!   driver compiled each GEMV into.
//!
//! ```text
//! cargo run --release -p graph --example decode_timing -- [OPTIONS]
//! ```
//!
//! An empty list skips its part; the sweep runs when any of its lists is given, the others
//! defaulting to the tuning's own value. Each block of `--steps` steps follows `--settle-ms` of
//! steps back to back, so that it starts at the clocks of a busy GPU.
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. Every step runs at
//! position `--context`, over a cache whose positions are zero until a step writes them:
//! attention's cost depends on the position, not on what the cache holds. A step runs `--batch`
//! requests at once, each in a slot of the cache of its own.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use argh::FromArgs;
use graph::{
    checkpoint::Checkpoint,
    decode::{Batch, Decoder, LmConfig, Row, Shape, Tuning, decode_batch_graph},
    kernels::Kernels,
};
use list::List;
use stream_read::stream_read;
use vulkan::{Calibration, Context, F16};

mod list;
mod stream_read;

/// Times the base LM's decode step on the GPU, and says what a step's wall time is made of.
#[derive(FromArgs)]
struct Args {
    /// the base LM's layers to run, 28 by default
    #[argh(option, default = "28")]
    layers: usize,
    /// the position every step runs at, 481 by default
    #[argh(option, default = "481")]
    context: usize,
    /// the positions each slot of the cache holds, 768 by default
    #[argh(option, default = "768")]
    capacity: usize,
    /// the steps of each block, 50 by default
    #[argh(option, default = "50")]
    steps: usize,
    /// the requests a step runs, 1 by default
    #[argh(option, default = "1")]
    batch: usize,
    /// the subgroup size, by default the widest the kernels are written for that the device can
    /// require; the rest of the tuning defaults to Tuning::for_batch's at it
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
    /// how long the warm-up from the load runs, 2000 by default
    #[argh(option, default = "2000")]
    warmup_ms: u64,
    /// how long steps run back to back before each block, 200 by default
    #[argh(option, default = "200")]
    settle_ms: u64,
    /// the fence spins, a block of steps back to back at each, 0,50000 by default
    #[argh(option, default = "List(vec![0, 50000])")]
    spins_us: List,
    /// the host's idle gaps before each step, a block at each, 0,1,4,16 by default
    #[argh(option, default = "List(vec![0, 1, 4, 16])")]
    gaps_ms: List,
    /// how long the host idles before the steps after it, 1000 by default; 0 skips them
    #[argh(option, default = "1000")]
    idle_ms: u64,
    /// the rounds of the step untimed and timed per dispatch, and of the sweep, 3 by default
    #[argh(option, default = "3")]
    rounds: usize,
    /// the sweep's workgroup widths
    #[argh(option)]
    sweep_workgroups: Option<List>,
    /// the sweep's row pairs or rows to a subgroup, of every GEMV
    #[argh(option)]
    sweep_rows: Option<List>,
    /// the sweep's loads ahead
    #[argh(option)]
    sweep_ahead: Option<List>,
    /// the sweep's stages
    #[argh(option)]
    sweep_stage: Option<List>,
    /// the sweep's vectors
    #[argh(option)]
    sweep_vectors: Option<List>,
}

/// The value at quantile `q` of `v`, from 0 for the least to 1 for the greatest.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// The host's `CLOCK_MONOTONIC` in nanoseconds, the clock a calibration reads beside the GPU's.
fn monotonic_ns() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a clock every Linux has, and a timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as f64 * 1e9 + t.tv_nsec as f64
}

/// A hidden state of the base LM's scale, from a fixed xorshift sequence.
fn input(hidden: usize, seed: u32) -> Vec<F16> {
    let mut state = seed | 1;
    (0..hidden)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            F16::from_f64(v * 0.1)
        })
        .collect()
}

/// The bytes each kernel reads in a step of `rows` rows at `position`, summed over the layers:
/// its weights, and attention's keys and values up to the position in each row's slot.
fn kernel_bytes(s: &Shape, position: usize, rows: usize) -> BTreeMap<&'static str, f64> {
    let (h, i, d, l) = (s.hidden, s.inner, s.head_dim, s.layers);
    let half = |elements: usize| (2 * l * elements) as f64;
    BTreeMap::from([
        ("qkv_rope", half((s.heads + 2 * s.kv_heads) * d * h)),
        ("o_proj", half(h * s.heads * d)),
        ("gate_up", half(2 * i * h)),
        ("down_proj", half(h * i)),
        (
            "attention_split",
            (rows as u64 * s.cache_bytes(position)) as f64,
        ),
    ])
}

/// GPU nanoseconds of each step, by kernel, summed over the step's layers.
type PerKernel = BTreeMap<String, Vec<f64>>;

/// One step's parts, in nanoseconds: on the host's clock around the GPU's time, and the queue's
/// delay and the host's wake on it too when the device calibrates its timestamps.
#[derive(Clone, Copy, Debug)]
struct Phases {
    /// Writing the input and the position.
    write: f64,
    /// The submission call.
    submit: f64,
    /// From the start of the submission call to the GPU's first timestamp.
    queue: Option<f64>,
    gpu: f64,
    /// From the GPU's last timestamp to the wait's return.
    wake: Option<f64>,
    /// How far the queue's delay and the wake may be off: the calibration's deviation.
    deviation: Option<f64>,
    /// Reading the output.
    read: f64,
    wall: f64,
}

/// The decoder, and what every step runs on.
struct Bench {
    ctx: Context,
    decoder: Option<Decoder>,
    /// Where each row runs.
    rows: Vec<Option<Row>>,
    inputs: Vec<Vec<F16>>,
    next: usize,
}

impl Bench {
    fn decoder(&mut self) -> &mut Decoder {
        self.decoder.as_mut().expect("a decoder")
    }

    /// The decoder recorded again at `tuning`, timed per dispatch when `timed`.
    fn retune(&mut self, kernels: &mut Kernels, tuning: &Tuning, timed: bool) {
        let decoder = self.decoder.take().expect("a decoder");
        self.decoder = Some(
            decoder
                .retuned(&self.ctx, kernels, tuning, timed)
                .unwrap_or_else(|e| panic!("{e}")),
        );
    }

    /// One step, and its parts placed on the host's clock by `calibration`.
    fn step(&mut self, calibration: Option<&Calibration>) -> Phases {
        let x = &self.inputs[self.next % self.inputs.len()];
        self.next += 1;
        let decoder = self.decoder.as_mut().expect("a decoder");
        let t0 = monotonic_ns();
        decoder
            .set_rows(&self.rows, x)
            .unwrap_or_else(|e| panic!("{e}"));
        let t1 = monotonic_ns();
        let submission = decoder.start().unwrap_or_else(|e| panic!("{e}"));
        let t2 = monotonic_ns();
        submission.wait().unwrap_or_else(|e| panic!("{e}"));
        let t3 = monotonic_ns();
        let out = decoder.output().unwrap_or_else(|e| panic!("{e}"));
        let t4 = monotonic_ns();
        assert!(out.iter().all(|h| h.to_f32().is_finite()));
        let step = decoder.graph_step();
        let gpu = step.elapsed().unwrap_or_else(|e| panic!("{e}")).nanoseconds;
        let ends = calibration.map(|c| {
            step.ends()
                .unwrap_or_else(|e| panic!("{e}"))
                .map(|t| c.monotonic_at(t))
        });
        Phases {
            write: t1 - t0,
            submit: t2 - t1,
            queue: ends.map(|[start, _]| start - t1),
            gpu,
            wake: ends.map(|[_, end]| t3 - end),
            deviation: calibration.map(|c| c.deviation_ns as f64),
            read: t4 - t3,
            wall: t4 - t0,
        }
    }

    /// The closest of a few calibrations: one can be far looser than the rest, as one of three on
    /// the Radeon 8060S was, within 1.2 ms where the others were within 20 us.
    fn calibration(&self) -> Option<Calibration> {
        (0..8)
            .filter_map(|_| self.ctx.calibrate().unwrap_or_else(|e| panic!("{e}")))
            .min_by_key(|c| c.deviation_ns)
    }

    /// Steps back to back for `duration`, one at least; their phases.
    fn run_for(&mut self, duration: Duration) -> Vec<Phases> {
        let calibration = self.calibration();
        let start = Instant::now();
        let mut phases = vec![self.step(calibration.as_ref())];
        while start.elapsed() < duration {
            phases.push(self.step(calibration.as_ref()));
        }
        phases
    }

    /// `steps` steps with the host idle for `gap` before each, after `settle` back to back; their
    /// phases, and the GPU time of each dispatch summed by kernel when the step is timed.
    fn block(&mut self, steps: usize, gap: Duration, settle: Duration) -> (Vec<Phases>, PerKernel) {
        self.run_for(settle);
        let calibration = self.calibration();
        let mut phases = Vec::with_capacity(steps);
        let mut per_kernel: PerKernel = BTreeMap::new();
        for _ in 0..steps {
            if !gap.is_zero() {
                std::thread::sleep(gap);
            }
            phases.push(self.step(calibration.as_ref()));
            if let Some(timings) = self.decoder().timings().unwrap_or_else(|e| panic!("{e}")) {
                let mut sums: BTreeMap<&str, f64> = BTreeMap::new();
                for (label, t) in &timings {
                    *sums.entry(label).or_default() += t.nanoseconds;
                }
                for (label, ns) in sums {
                    per_kernel.entry(label.to_string()).or_default().push(ns);
                }
            }
        }
        (phases, per_kernel)
    }
}

/// A line of what a block's steps were made of: medians, and the tails where they vary.
fn summary(phases: &[Phases]) -> String {
    let all = |f: fn(&Phases) -> f64| phases.iter().map(f).collect::<Vec<_>>();
    let some = |f: fn(&Phases) -> Option<f64>| phases.iter().filter_map(f).collect::<Vec<_>>();
    let (gpu, wall) = (all(|p| p.gpu), all(|p| p.wall));
    let ms = |v: &[f64], q: f64| quantile(v, q) / 1e6;
    let us = |v: &[f64]| median(v) / 1e3;
    let mut line = format!(
        "GPU {:.3} ms (p10 {:.3}, p90 {:.3}), wall {:.3} ms (p10 {:.3}, p90 {:.3}, max {:.3}); \
         write {:.1} us, submit {:.1} us",
        ms(&gpu, 0.5),
        ms(&gpu, 0.1),
        ms(&gpu, 0.9),
        ms(&wall, 0.5),
        ms(&wall, 0.1),
        ms(&wall, 0.9),
        ms(&wall, 1.0),
        us(&all(|p| p.write)),
        us(&all(|p| p.submit)),
    );
    let (queue, wake) = (some(|p| p.queue), some(|p| p.wake));
    if !queue.is_empty() {
        line += &format!(
            ", queue {:.1} us (p90 {:.1}), wake {:.1} us (p90 {:.1}, max {:.1}), each within {:.1} \
             us",
            us(&queue),
            quantile(&queue, 0.9) / 1e3,
            us(&wake),
            quantile(&wake, 0.9) / 1e3,
            quantile(&wake, 1.0) / 1e3,
            quantile(&some(|p| p.deviation), 1.0) / 1e3
        );
    }
    line + &format!(", read {:.1} us", us(&all(|p| p.read)))
}

/// Each GEMV pipeline of `tuning`'s step, by label, with what the driver says it compiled it into.
fn gemv_statistics(
    kernels: &mut Kernels,
    shape: &Shape,
    tuning: &Tuning,
    batch: Batch,
) -> Vec<String> {
    let graph = decode_batch_graph(shape, tuning, batch).unwrap_or_else(|e| panic!("{e}"));
    let mut pipelines = BTreeMap::new();
    for op in &graph.graph.ops {
        if op.pipeline.kernel.name.starts_with("gemv") {
            pipelines
                .entry(op.label.clone())
                .or_insert_with(|| op.pipeline.clone());
        }
    }
    pipelines
        .into_iter()
        .map(|(label, pipeline)| {
            let kernel = kernels.get(&pipeline).unwrap_or_else(|e| panic!("{e}"));
            let statistics = match kernel.executables() {
                Ok(executables) => executables
                    .iter()
                    .flat_map(|e| &e.statistics)
                    .map(|s| format!("{} {}", s.name, s.value.as_f64()))
                    .collect::<Vec<_>>()
                    .join(", "),
                Err(e) => e.to_string(),
            };
            format!("{label}: {statistics}")
        })
        .collect()
}

fn main() {
    let args: Args = argh::from_env();
    let (layers, context, capacity) = (args.layers, args.context, args.capacity);
    let (steps, rows, rounds) = (args.steps, args.batch, args.rounds);
    let batch = Batch { rows, slots: rows };
    let floor_mib = args.floor_mib;
    let warmup = Duration::from_millis(args.warmup_ms);
    let settle = Duration::from_millis(args.settle_ms);
    let idle = Duration::from_millis(args.idle_ms);
    let (spins, gaps) = (&args.spins_us.0, &args.gaps_ms.0);
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let shape = Shape::new(&config, layers, capacity).unwrap_or_else(|e| panic!("{e}"));
    assert!(context < capacity, "position {context} past the capacity");
    let checkpoint =
        Checkpoint::open(dir.join("model.safetensors")).unwrap_or_else(|e| panic!("{e}"));
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    let device = Tuning::for_device(report).unwrap_or_else(|e| panic!("{e}"));
    let base = Tuning::for_batch(args.subgroup.unwrap_or(device.subgroup), rows);
    let tuning = Tuning {
        workgroup: args.workgroup.unwrap_or(base.workgroup),
        qkv_pairs: args.qkv_pairs.unwrap_or(base.qkv_pairs),
        gate_up_pairs: args.gate_up_pairs.unwrap_or(base.gate_up_pairs),
        residual_rows: args.residual_rows.unwrap_or(base.residual_rows),
        ahead: args.ahead.unwrap_or(base.ahead),
        stage: args.stage.unwrap_or(base.stage),
        vectors: args.vectors.unwrap_or(base.vectors),
        ..base
    };
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), subgroups of {} to {}, validation {}, robustBufferAccess2 {}, calibrated \
         timestamps {}; {layers} layers at position {context} in {rows} rows, {tuning:?}",
        report.name,
        report.driver_info,
        report.min_subgroup_size,
        report.max_subgroup_size,
        if ctx.validating() { "on" } else { "off" },
        ctx.robust_buffer_access2(),
        ctx.calibrate().unwrap_or_else(|e| panic!("{e}")).is_some(),
    );
    let floor =
        (floor_mib > 0).then(|| stream_read(&ctx, &mut kernels, tuning.subgroup, floor_mib << 20));
    let weights = shape.weight_bytes() as f64;
    let cache = (rows as u64 * shape.cache_bytes(context)) as f64;
    let bytes = weights + cache;
    println!(
        "bytes read per step: {:.3} GB of weights, {:.1} MB of KV cache at position {context} in \
         {rows} rows",
        weights / 1e9,
        cache / 1e6,
    );
    if let Some(f) = floor {
        println!(
            "floor: stream_read at its best, {f:.0} GB/s, {:.3} ms a step",
            bytes / f / 1e6
        );
    }
    let of_floor = |rate: f64| {
        floor.map_or(String::new(), |f| {
            format!(", {:.0}% of the floor", 100.0 * rate / f)
        })
    };

    let start = Instant::now();
    let mut decoder = Decoder::with_batch(&ctx, &mut kernels, &shape, &tuning, batch, false)
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, "base_lm.")
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "recorded: {} dispatches, {} barriers, arena {} KiB; built and loaded in {:.1} s, \
         recorded in {:.2} ms",
        decoder.graph_step().dispatches(),
        decoder.graph_step().barriers(),
        decoder.graph_step().arena_bytes() / 1024,
        start.elapsed().as_secs_f64(),
        decoder.graph_step().recorded_in().as_secs_f64() * 1e3
    );
    let mut bench = Bench {
        ctx: ctx.clone(),
        decoder: Some(decoder),
        rows: (0..rows)
            .map(|slot| {
                Some(Row {
                    position: context,
                    slot,
                })
            })
            .collect(),
        inputs: (1..=8)
            .map(|seed| {
                let rows = (0..rows as u32).map(|r| input(shape.hidden, seed + 8 * r));
                rows.flatten().collect()
            })
            .collect(),
        next: 0,
    };

    // The warm-up from the load, as the GPU's clocks ramp.
    let warm = bench.run_for(warmup);
    let gpu_ms: Vec<f64> = warm.iter().map(|p| p.gpu / 1e6).collect();
    let fifths: Vec<String> = gpu_ms
        .chunks(gpu_ms.len().div_ceil(5))
        .map(|c| format!("{:.3}", median(c)))
        .collect();
    println!(
        "warm-up: {} steps in {:.2} s; GPU ms of the first 20: {}; median of each fifth: {}",
        warm.len(),
        warmup.as_secs_f64(),
        gpu_ms
            .iter()
            .take(20)
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join(" "),
        fifths.join(" ")
    );

    for &us in spins {
        ctx.set_fence_spin(Duration::from_micros(us as u64));
        let (phases, _) = bench.block(steps, Duration::ZERO, settle);
        println!("back to back, spin {us} us: {}", summary(&phases));
    }
    ctx.set_fence_spin(Duration::ZERO);
    for &ms in gaps {
        let (phases, _) = bench.block(steps, Duration::from_millis(ms as u64), settle);
        println!("gap {ms} ms: {}", summary(&phases));
    }
    if !idle.is_zero() {
        bench.run_for(settle);
        std::thread::sleep(idle);
        let calibration = bench.calibration();
        let after: Vec<Phases> = (0..20).map(|_| bench.step(calibration.as_ref())).collect();
        let each = |f: fn(&Phases) -> f64, scale: f64| {
            after
                .iter()
                .map(|p| format!("{:.2}", f(p) / scale))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "after {} ms idle, 20 steps: GPU ms {}; wall ms {}",
            idle.as_millis(),
            each(|p| p.gpu, 1e6),
            each(|p| p.wall, 1e6)
        );
    }

    // Untimed and timed in turn, so that what changes over the run falls on both alike.
    let mut per_kernel: PerKernel = BTreeMap::new();
    let mut medians = [Vec::new(), Vec::new()];
    for round in 0..rounds {
        let mut parts = Vec::new();
        for timed in [false, true] {
            bench.retune(&mut kernels, &tuning, timed);
            let (phases, kernels_ns) = bench.block(steps, Duration::ZERO, settle);
            for (label, ns) in kernels_ns {
                per_kernel.entry(label).or_default().extend(ns);
            }
            let gpu = median(&phases.iter().map(|p| p.gpu).collect::<Vec<_>>());
            medians[usize::from(timed)].push(gpu);
            let which = if timed { "timed" } else { "untimed" };
            parts.push(format!("{which} {}", summary(&phases)));
        }
        println!("round {round}: {}", parts.join("; "));
    }
    bench.retune(&mut kernels, &tuning, false);
    if rounds > 0 {
        let [untimed, timed] = medians.map(|m| median(&m));
        println!(
            "GPU time per step at {tuning:?}: untimed {:.3} ms, {:.0} GB/s{}; timed per dispatch \
             {:.3} ms",
            untimed / 1e6,
            bytes / untimed,
            of_floor(bytes / untimed),
            timed / 1e6
        );
        println!(
            "per kernel, summed over the step's layers (median of {} timed steps):",
            rounds * steps
        );
        let read = kernel_bytes(&shape, context, rows);
        for (label, ns) in &per_kernel {
            let m = median(ns);
            let rate = read.get(label.as_str()).map_or(String::new(), |&b| {
                format!("  {:4.0} GB/s{}", b / m, of_floor(b / m))
            });
            println!(
                "  {label:<18} {:8.3} ms  {:5.1}%{rate}",
                m / 1e6,
                100.0 * m / timed
            );
        }
    }

    let sweeping = [
        &args.sweep_workgroups,
        &args.sweep_rows,
        &args.sweep_ahead,
        &args.sweep_stage,
        &args.sweep_vectors,
    ]
    .iter()
    .any(|list| list.is_some());
    if sweeping {
        sweep(&mut bench, &mut kernels, &shape, &tuning, &args, floor);
    }
    let messages = ctx.validation_messages();
    if ctx.validating() {
        println!("validation: {} messages", messages.len());
        for m in messages {
            println!("  {m}");
        }
    }
}

/// Each tuning the sweep's lists make together, recorded over the bench's weights and measured
/// untimed and timed, `--rounds` times round robin; and the best.
fn sweep(
    bench: &mut Bench,
    kernels: &mut Kernels,
    shape: &Shape,
    tuning: &Tuning,
    args: &Args,
    floor: Option<f64>,
) {
    let (steps, rounds) = (args.steps, args.rounds);
    let settle = Duration::from_millis(args.settle_ms);
    let batch = bench.decoder().batch();
    let context = bench.rows[0].map_or(0, |row| row.position);
    let cache = batch.rows as u64 * shape.cache_bytes(context);
    let bytes = (shape.weight_bytes() + cache) as f64;
    let of_floor = |rate: f64| {
        floor.map_or(String::new(), |f| {
            format!(", {:.0}% of the floor", 100.0 * rate / f)
        })
    };
    let given = |list: &Option<List>, or: &[u32]| match list {
        Some(list) => list.0.clone(),
        None => or.iter().map(|&v| v as usize).collect(),
    };
    let workgroups = given(&args.sweep_workgroups, &[tuning.workgroup]);
    let rows = given(&args.sweep_rows, &[]);
    let aheads = given(&args.sweep_ahead, &[tuning.ahead]);
    let stages = given(&args.sweep_stage, &[tuning.stage]);
    let vectors = given(&args.sweep_vectors, &[tuning.vectors]);
    let mut tunings = Vec::new();
    for &workgroup in &workgroups {
        for &ahead in &aheads {
            for (&stage, &vectors) in stages
                .iter()
                .flat_map(|s| vectors.iter().map(move |v| (s, v)))
            {
                let cut = Tuning {
                    workgroup: workgroup as u32,
                    ahead: ahead as u32,
                    stage: stage as u32,
                    vectors: vectors as u32,
                    ..*tuning
                };
                if rows.is_empty() {
                    tunings.push(cut);
                }
                for &r in &rows {
                    let r = r as u32;
                    tunings.push(Tuning {
                        qkv_pairs: r,
                        gate_up_pairs: r,
                        residual_rows: r,
                        ..cut
                    });
                }
            }
        }
    }
    // Refused cuts, such as rows that do not divide a projection, are said and left out.
    tunings.retain(|t| match decode_batch_graph(shape, t, batch) {
        Ok(_) => true,
        Err(e) => {
            println!("sweep: {t:?} left out: {e}");
            false
        }
    });
    let gemvs = ["qkv_rope", "o_proj", "gate_up", "down_proj"];
    let mut results: Vec<(Vec<f64>, PerKernel)> =
        vec![(Vec::new(), BTreeMap::new()); tunings.len()];
    for _ in 0..rounds.max(1) {
        for (t, (untimed, per_kernel)) in tunings.iter().zip(&mut results) {
            bench.retune(kernels, t, false);
            let (phases, _) = bench.block(steps, Duration::ZERO, settle);
            untimed.push(median(&phases.iter().map(|p| p.gpu).collect::<Vec<_>>()));
            bench.retune(kernels, t, true);
            let (_, kernels_ns) = bench.block(steps, Duration::ZERO, settle);
            for (label, ns) in kernels_ns {
                per_kernel.entry(label).or_default().extend(ns);
            }
        }
    }
    println!(
        "sweep, GPU ms per step untimed (median of each round's median) and each GEMV's ms timed:"
    );
    let read = kernel_bytes(shape, context, batch.rows);
    let mut best: Option<(f64, Tuning)> = None;
    for (t, (untimed, per_kernel)) in tunings.iter().zip(&results) {
        let m = median(untimed);
        let gemv = gemvs
            .iter()
            .map(|g| {
                let ns = median(&per_kernel[*g]);
                format!("{g} {:.3} ({:.0} GB/s)", ns / 1e6, read[g] / ns)
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  w{} r{}/{}/{} a{} s{} v{}: {:.3} ms [{}]; {gemv}",
            t.workgroup,
            t.qkv_pairs,
            t.gate_up_pairs,
            t.residual_rows,
            t.ahead,
            t.stage,
            t.vectors,
            m / 1e6,
            untimed
                .iter()
                .map(|v| format!("{:.3}", v / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for line in gemv_statistics(kernels, shape, t, batch) {
            println!("    {line}");
        }
        if best.is_none_or(|(b, _)| m < b) {
            best = Some((m, *t));
        }
    }
    if let Some((m, t)) = best {
        println!(
            "best: {t:?}, {:.3} ms, {:.0} GB/s{}",
            m / 1e6,
            bytes / m,
            of_floor(bytes / m)
        );
    }
}
