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
//!   subgroup, of every GEMV) and `--sweep-ahead` make together, each recorded over the same
//!   weights, `--rounds` times round robin, with what the driver compiled each GEMV into.
//!
//! ```text
//! cargo run --release -p graph --example decode_timing -- [--layers 28] [--context 481]
//!     [--capacity 768] [--steps 50] [--subgroup S] [--workgroup W] [--qkv-pairs 1]
//!     [--gate-up-pairs 1] [--residual-rows 1] [--ahead 0] [--floor-mib 512]
//!     [--warmup-ms 2000] [--settle-ms 200] [--spins-us 0,50000] [--gaps-ms 0,1,4,16]
//!     [--idle-ms 1000] [--rounds 3] [--sweep-workgroups 64,128] [--sweep-rows 1,2]
//!     [--sweep-ahead 0,4,8]
//! ```
//!
//! The subgroup size defaults to the widest the kernels are written for that the device can
//! require, and the rest of the tuning to [`Tuning::for_subgroup`]'s at it. `--floor-mib 0` skips
//! the streaming read, and an empty list skips its part; the sweep runs when any of its lists is
//! given, the others defaulting to the tuning's own value. Each block of `--steps` steps follows
//! `--settle-ms` of steps back to back, so that it starts at the clocks of a busy GPU.
//!
//! `VOXCPM2_DIR` names the checkpoint and `GLSL_KERNELS` the compiled kernels. Every step runs at
//! position `--context`, over a cache whose positions are zero until a step writes them:
//! attention's cost depends on the position, not on what the cache holds.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use graph::{
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, Shape, Tuning, decode_graph},
    graph::Pipeline,
    kernels::{Kernels, STREAM_READ},
};
use vulkan::{Calibration, Context, Residence, SpecValue, memory::STAGING_BYTES};

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

/// A comma-separated list of numbers; `default` when the argument is not given.
fn list(name: &str, default: &str) -> Vec<usize> {
    let args: Vec<String> = std::env::args().collect();
    let text = args
        .iter()
        .position(|a| a == name)
        .map_or(default, |i| args[i + 1].as_str());
    text.split(',')
        .filter(|v| !v.is_empty())
        .map(|v| v.parse().unwrap_or_else(|e| panic!("{name}: {e}")))
        .collect()
}

fn given(name: &str) -> bool {
    std::env::args().any(|a| a == name)
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
    /// Reading the output.
    read: f64,
    wall: f64,
}

/// The decoder, and what every step runs on.
struct Bench {
    ctx: Context,
    decoder: Option<Decoder>,
    position: usize,
    inputs: Vec<Vec<u16>>,
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
        let position = self.position;
        let decoder = self.decoder.as_mut().expect("a decoder");
        let t0 = monotonic_ns();
        decoder
            .set_input(position, x)
            .unwrap_or_else(|e| panic!("{e}"));
        let t1 = monotonic_ns();
        let submission = decoder.start().unwrap_or_else(|e| panic!("{e}"));
        let t2 = monotonic_ns();
        submission.wait().unwrap_or_else(|e| panic!("{e}"));
        let t3 = monotonic_ns();
        let out = decoder.output().unwrap_or_else(|e| panic!("{e}"));
        let t4 = monotonic_ns();
        assert!(
            out.iter()
                .all(|&h| half::f16::from_bits(h).to_f32().is_finite())
        );
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
            read: t4 - t3,
            wall: t4 - t0,
        }
    }

    fn calibration(&self) -> Option<Calibration> {
        self.ctx.calibrate().unwrap_or_else(|e| panic!("{e}"))
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
            ", queue {:.1} us (p90 {:.1}), wake {:.1} us (p90 {:.1}, max {:.1})",
            us(&queue),
            quantile(&queue, 0.9) / 1e3,
            us(&wake),
            quantile(&wake, 0.9) / 1e3,
            quantile(&wake, 1.0) / 1e3
        );
    }
    line + &format!(", read {:.1} us", us(&all(|p| p.read)))
}

/// Each GEMV pipeline of `tuning`'s step, by label, with what the driver says it compiled it into.
fn gemv_statistics(kernels: &mut Kernels, shape: &Shape, tuning: &Tuning) -> Vec<String> {
    let graph = decode_graph(shape, tuning).unwrap_or_else(|e| panic!("{e}"));
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
    let layers = arg("--layers", 28);
    let context = arg("--context", 481);
    let capacity = arg("--capacity", 768);
    let steps = arg("--steps", 50);
    let floor_mib = arg("--floor-mib", 512) as u64;
    let warmup = Duration::from_millis(arg("--warmup-ms", 2000) as u64);
    let settle = Duration::from_millis(arg("--settle-ms", 200) as u64);
    let idle = Duration::from_millis(arg("--idle-ms", 1000) as u64);
    let rounds = arg("--rounds", 3);
    let spins = list("--spins-us", "0,50000");
    let gaps = list("--gaps-ms", "0,1,4,16");
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
    let base = Tuning::for_subgroup(arg("--subgroup", device.subgroup as usize) as u32);
    let tuning = Tuning {
        workgroup: arg("--workgroup", base.workgroup as usize) as u32,
        qkv_pairs: arg("--qkv-pairs", base.qkv_pairs as usize) as u32,
        gate_up_pairs: arg("--gate-up-pairs", base.gate_up_pairs as usize) as u32,
        residual_rows: arg("--residual-rows", base.residual_rows as usize) as u32,
        ahead: arg("--ahead", base.ahead as usize) as u32,
        ..base
    };
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{} ({}), subgroups of {} to {}, validation {}, robustBufferAccess2 {}, calibrated \
         timestamps {}; {layers} layers at position {context}, {tuning:?}",
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
    let bytes = weights + shape.cache_bytes(context) as f64;
    println!(
        "bytes read per step: {:.3} GB of weights, {:.1} MB of KV cache at position {context}",
        weights / 1e9,
        shape.cache_bytes(context) as f64 / 1e6,
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
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
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
        position: context,
        inputs: (1..=8).map(|seed| input(shape.hidden, seed)).collect(),
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

    for &us in &spins {
        ctx.set_fence_spin(Duration::from_micros(us as u64));
        let (phases, _) = bench.block(steps, Duration::ZERO, settle);
        println!("back to back, spin {us} us: {}", summary(&phases));
    }
    ctx.set_fence_spin(Duration::ZERO);
    for &ms in &gaps {
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
        let read = kernel_bytes(&shape, context);
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

    let sweeping = ["--sweep-workgroups", "--sweep-rows", "--sweep-ahead"]
        .iter()
        .any(|a| given(a));
    if !sweeping {
        return;
    }
    let workgroups = list("--sweep-workgroups", &tuning.workgroup.to_string());
    let rows = list("--sweep-rows", "");
    let aheads = list("--sweep-ahead", &tuning.ahead.to_string());
    let mut tunings = Vec::new();
    for &workgroup in &workgroups {
        for &ahead in &aheads {
            let cut = Tuning {
                workgroup: workgroup as u32,
                ahead: ahead as u32,
                ..tuning
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
    // Refused cuts, such as rows that do not divide a projection, are said and left out.
    tunings.retain(|t| match decode_graph(&shape, t) {
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
            bench.retune(&mut kernels, t, false);
            let (phases, _) = bench.block(steps, Duration::ZERO, settle);
            untimed.push(median(&phases.iter().map(|p| p.gpu).collect::<Vec<_>>()));
            bench.retune(&mut kernels, t, true);
            let (_, kernels_ns) = bench.block(steps, Duration::ZERO, settle);
            for (label, ns) in kernels_ns {
                per_kernel.entry(label).or_default().extend(ns);
            }
        }
    }
    println!(
        "sweep, GPU ms per step untimed (median of each round's median) and each GEMV's ms timed:"
    );
    let read = kernel_bytes(&shape, context);
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
            "  w{} r{}/{}/{} a{}: {:.3} ms [{}]; {gemv}",
            t.workgroup,
            t.qkv_pairs,
            t.gate_up_pairs,
            t.residual_rows,
            t.ahead,
            m / 1e6,
            untimed
                .iter()
                .map(|v| format!("{:.3}", v / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for line in gemv_statistics(&mut kernels, &shape, t) {
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
