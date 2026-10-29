//! The patch step on the GPU for several requests at once, on the real weights: each request a
//! row of the step at its own position in its own slot of the caches, beside spare rows, on the
//! program of the fewest rows that holds those under way. Each comes out bit for bit as it does
//! alone, in the patch step at batch 1, whatever runs beside it. Built with the `device` feature,
//! and run with `VOXCPM2_DIR` naming the checkpoint and `GLSL_KERNELS` the compiled kernels, as
//! `nix run .#device-tests` sets the last.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{self, Batch, Row, SUBGROUPS},
    graph::Kind,
    kernels::Kernels,
    lm_side::{self, LmSideShape},
    local::{self, PatchConfig},
    patch::{PatchGraph, PatchRow, PatchStep, Sampled, patch_batch_graph},
};
use testkit::checkpoint_dir;
use vulkan::Context;

/// The rows of the programs the service records, over caches of as many slots as the most.
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];
const SLOTS: usize = 16;

/// The positions each slot of the caches holds.
const CAPACITY: usize = 64;

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

/// A request: from step `start` on, `len` patches, a step each, from position 0 of slot `slot`,
/// each sampled after the one before from noise made up from `seed`.
#[derive(Clone, Copy, Debug)]
struct Request {
    start: usize,
    len: usize,
    slot: usize,
    temperature: f32,
    cfg: f32,
}

const fn request(start: usize, len: usize, slot: usize, sampling: (f32, f32)) -> Request {
    Request {
        start,
        len,
        slot,
        temperature: sampling.0,
        cfg: sampling.1,
    }
}

const SERVICE: (f32, f32) = (1.0, 2.0);
const COLD: (f32, f32) = (0.7, 1.5);
const HOT: (f32, f32) = (1.3, 3.0);

/// One request alone, then two and three; then sixteen at once, every slot taken; then fewer as
/// they end, through the programs of eight, four and two rows; and two more in slots others have
/// left, the last of them alone at the end.
const REQUESTS: [Request; 18] = [
    request(0, 40, 3, SERVICE),
    request(2, 30, 9, HOT),
    request(3, 21, 0, COLD),
    request(5, 14, 1, SERVICE),
    request(6, 13, 2, HOT),
    request(7, 15, 4, SERVICE),
    request(8, 12, 5, COLD),
    request(9, 16, 6, SERVICE),
    request(10, 11, 7, HOT),
    request(11, 17, 8, SERVICE),
    request(12, 10, 10, COLD),
    request(13, 19, 11, SERVICE),
    request(14, 9, 12, HOT),
    request(15, 18, 13, SERVICE),
    request(16, 8, 14, COLD),
    request(17, 20, 15, SERVICE),
    request(26, 12, 0, HOT),
    request(36, 14, 2, SERVICE),
];

/// Values in [-scale, scale), `len` of them, from a fixed xorshift sequence for each seed.
fn made_up(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0) as f32 * scale
        })
        .collect()
}

/// The noise request `i` samples its `t`-th patch from.
fn noise(i: usize, t: usize, len: usize) -> Vec<f32> {
    made_up(len, (1000 * (i + 1) + t) as u32, 1.7)
}

/// The patch request `i` samples its first patch after.
fn first(i: usize, len: usize) -> Vec<f32> {
    made_up(len, i as u32 + 1, 1.0)
}

/// What a request sampled, as bits: each patch and the stop logits beside it.
fn bits(sampled: &Sampled) -> Vec<u32> {
    let logits = sampled.stop_logits.iter();
    sampled
        .patch
        .iter()
        .chain(logits)
        .map(|v| v.to_bits())
        .collect()
}

/// VoxCPM2 with two layers of the base LM and one of the residual LM, and its checkpoint.
struct Model {
    ctx: Context,
    kernels: Kernels,
    checkpoint: Checkpoint,
    side: LmSideShape,
    config: PatchConfig,
}

impl Model {
    fn new() -> Self {
        let ctx = context();
        let kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
        let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
        let mut side: lm_side::Config = serde_json::from_str(&json).unwrap();
        side.lm_config.num_hidden_layers = 2;
        side.residual_lm_num_layers = 1;
        let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
            .unwrap_or_else(|e| panic!("{e}"));
        Model {
            ctx,
            kernels,
            checkpoint,
            side: LmSideShape::new(&side, CAPACITY).unwrap(),
            config: serde_json::from_str(&json).unwrap(),
        }
    }

    /// The patch step at `rows` rows over caches of `slots` slots, the LM side cut as a batch
    /// of as many rows is at `subgroup`.
    fn graph(&self, subgroup: u32, rows: usize, slots: usize) -> PatchGraph {
        let lm = decode::Tuning::for_batch(subgroup, rows);
        let local = local::Tuning::for_subgroup(subgroup)
            .fit(self.ctx.report())
            .unwrap_or_else(|e| panic!("{e}"));
        let batch = Batch { rows, slots };
        patch_batch_graph(&self.side, &self.config, 10, &lm, &local, batch)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// `graphs` recorded and loaded.
    fn step(&mut self, graphs: Vec<PatchGraph>) -> PatchStep {
        let start = Instant::now();
        let mut step = PatchStep::with_rows(&self.ctx, &mut self.kernels, graphs, false)
            .unwrap_or_else(|e| panic!("{e}"));
        step.load(&self.checkpoint)
            .unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "programs of {:?} rows over {} slots recorded and loaded: {:.1} s",
            step.rows(),
            step.slots(),
            start.elapsed().as_secs_f64()
        );
        step
    }
}

/// Runs [`REQUESTS`] on `step`, those under way at a step in rows of the program of the fewest
/// rows that holds them, rotating from step to step so that a request is seldom in the same row
/// twice running, the other rows spare; each patch sampled after the one before. `before` is
/// called with each step's placement before it runs. Gives what each request sampled.
fn run(
    step: &mut PatchStep,
    mut before: impl FnMut(&mut PatchStep, &[Option<Row>], &[Option<usize>]),
) -> Vec<Vec<Vec<u32>>> {
    let len = 64 * 4;
    let steps = REQUESTS.iter().map(|r| r.start + r.len).max().unwrap();
    let mut conds: Vec<Vec<f32>> = (0..REQUESTS.len()).map(|i| first(i, len)).collect();
    let mut got: Vec<Vec<Vec<u32>>> = REQUESTS.iter().map(|_| vec![]).collect();
    let mut used = vec![];
    for t in 0..steps {
        let running: Vec<usize> = (0..REQUESTS.len())
            .filter(|&i| (REQUESTS[i].start..REQUESTS[i].start + REQUESTS[i].len).contains(&t))
            .collect();
        let rows = *step
            .rows()
            .iter()
            .find(|&&r| r >= running.len())
            .expect("a program that holds them");
        used.push((rows, running.len()));
        let noises: Vec<Vec<f32>> = running
            .iter()
            .map(|&i| noise(i, t - REQUESTS[i].start, len))
            .collect();
        let mut placed = vec![None; rows];
        let mut which = vec![None; rows];
        let mut patch_rows = vec![None; rows];
        for (k, &i) in running.iter().enumerate() {
            let row = (k + t) % rows;
            let r = REQUESTS[i];
            let at = Row {
                position: t - r.start,
                slot: r.slot,
            };
            placed[row] = Some(at);
            which[row] = Some(i);
            patch_rows[row] = Some(PatchRow {
                at,
                cond: &conds[i],
                noise: &noises[k],
                temperature: r.temperature,
                cfg: r.cfg,
            });
        }
        before(step, &placed, &which);
        let sampled = step.run_rows(&patch_rows).unwrap_or_else(|e| panic!("{e}"));
        drop(patch_rows);
        for (row, out) in sampled.into_iter().enumerate() {
            if let (Some(i), Some(out)) = (which[row], out) {
                got[i].push(bits(&out));
                conds[i] = out.patch;
            }
        }
    }
    // Every program runs, and those that can with rows spare.
    for program in step.rows() {
        let runs: Vec<usize> = used
            .iter()
            .filter(|u| u.0 == program)
            .map(|u| u.1)
            .collect();
        assert!(!runs.is_empty(), "the program of {program} rows never ran");
        assert!(
            program <= 2 || runs.iter().any(|&n| n < program),
            "the program of {program} rows ran with no row spare"
        );
    }
    got
}

/// Each request alone on `step`, at batch 1, in the caches' first slot.
fn alone(step: &mut PatchStep) -> Vec<Vec<Vec<u32>>> {
    let len = 64 * 4;
    REQUESTS
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut cond = first(i, len);
            (0..r.len)
                .map(|t| {
                    let noise = noise(i, t, len);
                    let out = step
                        .run(t, &cond, &noise, r.temperature, r.cfg)
                        .unwrap_or_else(|e| panic!("{e}"));
                    let got = bits(&out);
                    cond = out.patch;
                    got
                })
                .collect()
        })
        .collect()
}

/// The first step of each request at which `got` is apart from `want`.
fn first_apart(got: &[Vec<Vec<u32>>], want: &[Vec<Vec<u32>>]) -> Vec<Option<usize>> {
    got.iter()
        .zip(want)
        .map(|(g, w)| (0..w.len()).find(|&t| g.get(t) != Some(&w[t])))
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn rows_sample_bit_for_bit_as_they_do_alone() {
    let mut model = Model::new();
    let report = model.ctx.report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let one = model.graph(subgroup, 1, 1);
        let mut one = model.step(vec![one]);
        let want = alone(&mut one);
        drop(one);

        let graphs = ROWS.map(|rows| model.graph(subgroup, rows, SLOTS)).to_vec();
        let mut batched = model.step(graphs);
        let start = Instant::now();
        let got = run(&mut batched, |_, _, _| {});
        eprintln!(
            "wave{subgroup}: {} requests at once at most, {:.1} s",
            SLOTS,
            start.elapsed().as_secs_f64()
        );
        let apart = first_apart(&got, &want);
        assert!(
            apart.iter().all(Option::is_none),
            "wave{subgroup}: the first patch of each request apart from it alone, {apart:?}"
        );
        drop(batched);

        // Defect: request 0's attention reads request 2's slot and the reverse wherever both run,
        // while their keys and values go to their own. Only those two may come out otherwise.
        let mut crossed = model.graph(subgroup, 16, SLOTS);
        let words = crossed.graph.info(crossed.position).shape.clone();
        let wrong = crossed
            .graph
            .tensor::<u32>("wrong_position", &words, Kind::Param);
        for op in &mut crossed.graph.ops {
            if op.label.ends_with("attention_split") {
                op.bindings[3] = wrong.id();
            }
        }
        let mut crossed = model.step(vec![crossed]);
        let got = run(&mut crossed, |step, placed, which| {
            let batch = Batch {
                rows: placed.len(),
                slots: SLOTS,
            };
            let mut words = batch.placement(placed, CAPACITY).unwrap();
            let row_of = |i| which.iter().position(|&w| w == Some(i));
            if let (Some(a), Some(c)) = (row_of(0), row_of(2)) {
                words.swap(batch.rows + a, batch.rows + c);
            }
            step.parts_mut().1.write(wrong, &words).unwrap();
        });
        let apart = first_apart(&got, &want);
        eprintln!("wave{subgroup}: with the slots read crossed, first apart at {apart:?}");
        let crossed: Vec<bool> = apart.iter().map(Option::is_some).collect();
        let mut expected = vec![false; REQUESTS.len()];
        expected[0] = true;
        expected[2] = true;
        assert_eq!(crossed, expected, "wave{subgroup}: {apart:?}");
    }
    assert_no_validation_messages(&model.ctx);
}
