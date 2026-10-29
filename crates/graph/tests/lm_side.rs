//! The LM side on the GPU, on VoxCPM2's real weights, held to its golden. It is given the local
//! encoder's output for every patch of a generation burn's CPU backend ran in f32, prompted with a
//! voice's patches alone and generating from seeded noise, the prompt's one at a time, and gives
//! the condition and the stop logits from the prompt's last position on. Built with the `device`
//! feature, and run with `VOXCPM2_DIR` naming the checkpoint, `GOLDEN_DIR` the goldens and
//! `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests` sets the last. Each tolerance
//! is set as testkit's `golden` says.
//!
//! The LM side runs several requests at once too, a row each, each at its own position in its own
//! slot of the caches: a row comes out bit for bit as it does alone, which a few layers are held
//! to over inputs made up for each request, and the golden's generation.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{Batch, Row, SUBGROUPS, Tuning},
    graph::{Kind, TensorId},
    kernels::Kernels,
    lm_side::{self, Conditioning, LmSide, LmSideShape, lm_side_batch_graph, load_lm_side},
    step::Step,
};
use serde::Deserialize;
use testkit::{checkpoint_dir, golden, golden_input};
use vulkan::{Context, F16};

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

/// VoxCPM2's config, as the LM side reads it, with `layers` of the base LM and
/// `residual_layers` of the residual LM.
fn config(layers: usize, residual_layers: usize) -> lm_side::Config {
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let mut config: lm_side::Config = serde_json::from_str(&json).unwrap();
    config.lm_config.num_hidden_layers = layers;
    config.residual_lm_num_layers = residual_layers;
    config
}

/// What the LM side is given: the local encoder's output for every patch of a generation burn's
/// CPU backend ran in f32 with `layers` of the base LM and `residual_layers` of the residual LM,
/// prompted with Baraka's patches alone and generating 8 patches from `SeededNoise::new(0)`,
/// the prompt's and then the generated ones, rounded to f16: `[patches][encoder]` as f16 bits.
#[derive(Deserialize)]
struct Input {
    enc: Vec<u16>,
    /// The patches of the prompt.
    prompt: usize,
}

fn input(config: &lm_side::Config) -> Input {
    let (layers, residual) = (
        config.lm_config.num_hidden_layers,
        config.residual_lm_num_layers,
    );
    golden_input(&format!("input-lm-side-{layers}-{residual}"))
}

/// The golden of `config` over `input`: the condition `[patches][2 * dit]` and the stop logits
/// `[patches][2]` after the prompt and after each generated patch, at the device's default
/// tuning.
fn golden_of(config: &lm_side::Config, input: &Input) -> (Vec<f32>, Vec<f32>) {
    let (layers, residual) = (
        config.lm_config.num_hidden_layers,
        config.residual_lm_num_layers,
    );
    golden(&format!("lm-side-{layers}-{residual}"), || {
        gpu(config, input, None, None, 1).remove(0)
    })
}

/// A request a batch runs: from step `start` on, positions 0 to `len - 1`, a position a step, in
/// slot `slot` of the caches.
#[derive(Clone, Copy, Debug)]
struct Request {
    start: usize,
    len: usize,
    slot: usize,
}

/// Runs `requests` through a batch of `rows`, step by step: those running at a step in rows that
/// rotate from step to step, so that a request is seldom in the same row twice running, and the
/// other rows spare. `step` takes where each row runs and which request runs there, and gives
/// what each row gives, none for a spare row; returns what each request was given at each of its
/// positions.
fn run_batch<T>(
    requests: &[Request],
    rows: usize,
    mut step: impl FnMut(&[Option<Row>], &[Option<usize>]) -> Vec<Option<T>>,
) -> Vec<Vec<T>> {
    let steps = requests.iter().map(|r| r.start + r.len).max().unwrap_or(0);
    let mut got: Vec<Vec<T>> = requests.iter().map(|_| Vec::new()).collect();
    for t in 0..steps {
        let running: Vec<usize> = (0..requests.len())
            .filter(|&i| (requests[i].start..requests[i].start + requests[i].len).contains(&t))
            .collect();
        assert!(
            running.len() <= rows,
            "{running:?} at step {t} in {rows} rows"
        );
        let mut placed = vec![None; rows];
        let mut which = vec![None; rows];
        for (k, &i) in running.iter().enumerate() {
            let row = (k + t) % rows;
            let Request { start, slot, .. } = requests[i];
            placed[row] = Some(Row {
                position: t - start,
                slot,
            });
            which[row] = Some(i);
        }
        let mut out = step(&placed, &which);
        for (k, &i) in running.iter().enumerate() {
            let row = (k + t) % rows;
            got[i].push(out[row].take().expect("a row that runs gives something"));
        }
    }
    got
}

/// Each row's input, `[rows][width]`: of the request that runs there, at its position, from
/// `inputs`, the inputs of every request at every position; zero in a spare row.
fn row_inputs(
    placed: &[Option<Row>],
    which: &[Option<usize>],
    inputs: &[&[F16]],
    width: usize,
) -> Vec<F16> {
    let mut x = vec![F16::ZERO; placed.len() * width];
    for (b, (row, request)) in placed.iter().zip(which).enumerate() {
        if let (Some(row), Some(i)) = (row, request) {
            let at = row.position * width;
            x[b * width..(b + 1) * width].copy_from_slice(&inputs[*i][at..at + width]);
        }
    }
    x
}

/// The GPU's conditions and stop logits for `input`'s patches, from the prompt's last on, with
/// the kernels at `subgroup` (the device's default when `None`), the residual LM rotated by RoPE
/// as `shape` says unless `residual_rope` says otherwise; for each of `rows` requests run at
/// once, cut as [`Tuning::for_batch`] cuts them, request r from step 2 r on in slot
/// `rows - 1 - r` of the caches.
fn gpu(
    config: &lm_side::Config,
    input: &Input,
    subgroup: Option<u32>,
    residual_rope: Option<bool>,
    rows: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let positions = input.enc.len() / config.encoder_config.hidden_dim;
    let mut shape = LmSideShape::new(config, positions.next_multiple_of(32)).unwrap();
    if let Some(rope) = residual_rope {
        shape.residual.rope = rope;
    }
    let subgroup = subgroup.unwrap_or_else(|| {
        let device = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
        device.subgroup
    });
    let tuning = Tuning::for_batch(subgroup, rows);
    let start = Instant::now();
    let batch = Batch { rows, slots: rows };
    let mut side = LmSide::with_batch(&ctx, &mut kernels, &shape, &tuning, batch, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    side.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let step = side.graph_step();
    eprintln!(
        "{} and {} layers in {rows} rows recorded and loaded on the GPU at {tuning:?}: {:.1} s, {} \
         dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        shape.base.layers,
        shape.residual.layers,
        start.elapsed().as_secs_f64(),
        step.dispatches(),
        step.barriers(),
        kernels.len(),
        step.arena_bytes() / 1024
    );
    let requests: Vec<Request> = (0..rows)
        .map(|r| Request {
            start: 2 * r,
            len: positions,
            slot: rows - 1 - r,
        })
        .collect();
    let width = shape.encoder;
    let enc: Vec<F16> = input.enc.iter().map(|&h| F16::from_bits(h)).collect();
    let inputs = vec![enc.as_slice(); rows];
    let got = run_batch(&requests, rows, |placed, which| {
        let enc = row_inputs(placed, which, &inputs, width);
        side.step_rows(placed, &enc)
            .unwrap_or_else(|e| panic!("{e}"))
    });
    assert_no_validation_messages(&ctx);
    got.into_iter()
        .map(|patches| {
            let (mut mu, mut stop) = (vec![], vec![]);
            for out in &patches[input.prompt - 1..] {
                mu.extend(out.mu.iter().map(|h| h.to_f32()));
                stop.extend(out.stop_logits);
            }
            (mu, stop)
        })
        .collect()
}

/// How far one run's outputs are from another's.
#[derive(Clone, Copy)]
struct Distance {
    max_abs: f64,
    /// RMS of the differences over RMS of the reference.
    relative_rms: f64,
}

impl Distance {
    fn within(self, tolerance: Distance) -> bool {
        self.max_abs <= tolerance.max_abs && self.relative_rms <= tolerance.relative_rms
    }
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |d| {:.2e}, relative RMS {:.2e}",
            self.max_abs, self.relative_rms
        )
    }
}

fn distance(got: &[f32], want: &[f32]) -> Distance {
    let (mut max_abs, mut err2, mut ref2) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        let d = (g - w).abs();
        max_abs = max_abs.max(if d.is_finite() { d } else { f64::INFINITY });
        err2 += d * d;
        ref2 += w * w;
    }
    Distance {
        max_abs,
        relative_rms: (err2 / ref2).sqrt(),
    }
}

/// A run's tolerances: of the condition, and of the stop logits.
struct Tolerance {
    mu: Distance,
    stop: Distance,
}

/// Whether the GPU's conditions and stop logits are within `tolerance` of the golden's, by their
/// largest error and relative RMS error over every patch. Errors are compared over all the
/// patches, not at each: the scalar quantizer rounds a value either way of a level by luck, and
/// one level is a ninth.
fn check(
    name: &str,
    (mu, stop): &(Vec<f32>, Vec<f32>),
    (want, want_stop): &(Vec<f32>, Vec<f32>),
    tolerance: &Tolerance,
) -> bool {
    let patches = want_stop.len() / 2;
    let width = want.len() / patches;
    for i in 0..patches {
        let at = |v: &[f32]| v[i * width..(i + 1) * width].to_vec();
        let logits = |v: &[f32]| [v[2 * i], v[2 * i + 1]];
        eprintln!(
            "{name}, after patch {i}: condition {:?}; stop logits {:?}, the golden's {:?}",
            distance(&at(mu), &at(want)),
            logits(stop),
            logits(want_stop),
        );
    }
    let (ours, ours_stop) = (distance(mu, want), distance(stop, want_stop));
    eprintln!("{name} against the golden: condition {ours:?}; stop logits {ours_stop:?}");
    ours.within(tolerance.mu) && ours_stop.within(tolerance.stop)
}

/// Whether the GPU and the golden decide alike whether to stop, wherever the golden's margin
/// between going on and stopping is wider than the GPU's distance from it in the logits.
fn decides_alike(want: &[f32], stop: &[f32]) -> bool {
    let error = distance(stop, want).max_abs as f32;
    want.chunks(2).zip(stop.chunks(2)).all(|(w, g)| {
        let margin = w[1] - w[0];
        margin.abs() <= 2.0 * error || (margin > 0.0) == (g[1] > g[0])
    })
}

/// The full stacks' layer counts, from the checkpoint's config.
fn full() -> (usize, usize) {
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let full: lm_side::Config = serde_json::from_str(&json).unwrap();
    (
        full.lm_config.num_hidden_layers,
        full.residual_lm_num_layers,
    )
}

/// A few layers' tolerances: against burn's CPU in f32, the largest errors and relative RMS
/// errors of any run on the Radeon 8060S and the RTX 4070 Ti SUPER were 7.37e-2 and 4.39e-3 over
/// the condition and 1.94e-3 and 2.16e-3 over the stop logits, and the golden's 7.37e-2 and
/// 4.32e-3, and 1.94e-3 and 2.07e-3.
const A_FEW_LAYERS: Tolerance = Tolerance {
    mu: Distance {
        max_abs: 1.9e-1,
        relative_rms: 1.1e-2,
    },
    stop: Distance {
        max_abs: 4.9e-3,
        relative_rms: 5.3e-3,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn a_few_layers_condition_as_their_golden() {
    let config = config(2, 1);
    let input = input(&config);
    let want = golden_of(&config, &input);
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(&config, &input, Some(subgroup), None, 1).remove(0);
        let name = format!("wave{subgroup}");
        assert!(check(&name, &got, &want, &A_FEW_LAYERS), "{name}");
        assert!(decides_alike(&want.1, &got.1), "{name}");
        // Three requests at once, each at its own position: each as it is alone.
        let rows = gpu(&config, &input, Some(subgroup), None, 3);
        for (r, row) in rows.iter().enumerate() {
            assert!(row == &got, "{name}: row {r} of 3 is not as it is alone");
        }
    }
    // Defect: the residual LM rotated by RoPE, as the base LM is.
    let rotated = gpu(&config, &input, None, Some(true), 1).remove(0);
    assert!(
        !check("the residual LM rotated", &rotated, &want, &A_FEW_LAYERS),
        "a residual LM with RoPE passes"
    );
}

/// Requests in four rows over six slots: four at once for a while, mostly three and so a spare
/// row, each at its own position; the last in a slot the first has left.
const REQUESTS: [Request; 6] = [
    Request {
        start: 0,
        len: 40,
        slot: 5,
    },
    Request {
        start: 3,
        len: 30,
        slot: 0,
    },
    Request {
        start: 10,
        len: 25,
        slot: 2,
    },
    Request {
        start: 12,
        len: 8,
        slot: 3,
    },
    Request {
        start: 34,
        len: 20,
        slot: 1,
    },
    Request {
        start: 41,
        len: 10,
        slot: 5,
    },
];

/// Local encoder outputs of about the real ones' scale, `[len][width]`, from a fixed xorshift
/// sequence for each seed.
fn made_up(len: usize, width: usize, seed: u32) -> Vec<F16> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..len * width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            F16::from_f64(f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0)
        })
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn rows_condition_bit_for_bit_as_they_do_alone() {
    let config = config(2, 1);
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    let shape = LmSideShape::new(&config, 64).unwrap();
    let width = shape.encoder;
    let encs: Vec<Vec<F16>> = (0..REQUESTS.len())
        .map(|i| made_up(REQUESTS[i].len, width, i as u32 + 1))
        .collect();
    let inputs: Vec<&[F16]> = encs.iter().map(Vec::as_slice).collect();
    let batch = Batch { rows: 4, slots: 6 };
    let report = ctx.report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let base = Tuning::for_subgroup(subgroup);
        let side = |tuning: &Tuning, batch: Batch, kernels: &mut Kernels| {
            let mut side = LmSide::with_batch(&ctx, kernels, &shape, tuning, batch, false)
                .unwrap_or_else(|e| panic!("{e}"));
            side.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
            side
        };
        // Alone: each request at batch 1, one after another in the one slot.
        let mut one = side(&base, Batch::ONE, &mut kernels);
        let alone: Vec<Vec<Conditioning>> = encs
            .iter()
            .map(|enc| {
                let steps = enc.chunks(width).enumerate();
                steps
                    .map(|(t, e)| one.step(t, e).unwrap_or_else(|e| panic!("{e}")))
                    .collect()
            })
            .collect();
        drop(one);
        let first_apart = |got: &[Vec<Conditioning>]| -> Vec<Option<usize>> {
            got.iter()
                .zip(&alone)
                .map(|(g, w)| (0..w.len()).find(|&t| g.get(t) != Some(&w[t])))
                .collect()
        };
        // Cuts that change where and when a row is summed, never in what order: the workgroup
        // and the rows to a subgroup, loads ahead, the vectors staged a step at a time, in a stage
        // short of two, and the batch shared out over workgroups a row or two each.
        let cuts = [
            base,
            Tuning {
                workgroup: 256,
                qkv_pairs: 2,
                gate_up_pairs: 2,
                residual_rows: 2,
                ahead: 2,
                ..base
            },
            Tuning {
                workgroup: 2 * subgroup,
                stage: (2 * batch.rows * 8 * subgroup as usize * 2 - 1) as u32,
                ..base
            },
            Tuning {
                workgroup: 256,
                vectors: 1,
                ..base
            },
            Tuning { vectors: 2, ..base },
            Tuning::for_batch(subgroup, 16),
        ];
        for tuning in cuts {
            let mut rows = side(&tuning, batch, &mut kernels);
            let got = run_batch(&REQUESTS, batch.rows, |placed, which| {
                let enc = row_inputs(placed, which, &inputs, width);
                rows.step_rows(placed, &enc)
                    .unwrap_or_else(|e| panic!("{e}"))
            });
            let apart = first_apart(&got);
            assert!(
                apart.iter().all(Option::is_none),
                "wave{subgroup}, {tuning:?}: the first position of each request apart from it \
                 alone, {apart:?}"
            );
        }
        // One row over three slots, a request in each in turn.
        let mut row = side(&base, Batch { rows: 1, slots: 3 }, &mut kernels);
        let (d, f) = (REQUESTS[3], REQUESTS[5]);
        let turns = [
            Request {
                start: 0,
                len: d.len,
                slot: 2,
            },
            Request {
                start: d.len,
                len: f.len,
                slot: 1,
            },
        ];
        let got = run_batch(&turns, 1, |placed, which| {
            let enc = row_inputs(placed, which, &[inputs[3], inputs[5]], width);
            row.step_rows(placed, &enc)
                .unwrap_or_else(|e| panic!("{e}"))
        });
        assert!(
            got[0] == alone[3] && got[1] == alone[5],
            "wave{subgroup}: one row over three slots"
        );
        drop(row);

        // Defect: request 0's attention reads request 2's slot and the reverse wherever both run,
        // while their keys and values go to their own. Only those two may come out otherwise.
        let mut tensors =
            lm_side_batch_graph(&shape, &base, batch).unwrap_or_else(|e| panic!("{e}"));
        let words = tensors.graph.info(tensors.position).shape().to_vec();
        let wrong = tensors
            .graph
            .tensor::<u32>("wrong_position", &words, Kind::Param);
        for op in &mut tensors.graph.ops {
            if op.label.ends_with("attention_split") {
                op.bindings[3] = wrong.id();
            }
        }
        let mut step =
            Step::new(&ctx, &mut kernels, &tensors.graph, false).unwrap_or_else(|e| panic!("{e}"));
        let (cos, sin) = shape.base.rope_tables();
        step.write(tensors.cos, &cos).unwrap();
        step.write(tensors.sin, &sin).unwrap();
        let ids: Vec<TensorId> = (0..tensors.graph.tensors.len()).map(TensorId).collect();
        load_lm_side(&mut step, &tensors.graph, &tensors, &ids, &checkpoint)
            .unwrap_or_else(|e| panic!("{e}"));
        let got = run_batch(&REQUESTS, batch.rows, |placed, which| {
            let words = batch.placement(placed, shape.base.capacity).unwrap();
            let mut swapped = words.clone();
            let row_of = |i| which.iter().position(|&w| w == Some(i));
            if let (Some(a), Some(c)) = (row_of(0), row_of(2)) {
                swapped.swap(batch.rows + a, batch.rows + c);
            }
            step.write(tensors.enc, &row_inputs(placed, which, &inputs, width))
                .unwrap();
            step.write(tensors.position, &words).unwrap();
            step.write(wrong, &swapped).unwrap();
            step.submit().unwrap();
            let mu: Vec<F16> = step.read(tensors.mu).unwrap();
            let logits: Vec<F16> = step.read(tensors.stop_logits).unwrap();
            let logit = |i: usize| logits[i].to_f32();
            let condition = mu.len() / batch.rows;
            let rows = placed.iter().enumerate();
            rows.map(|(b, row)| {
                row.map(|_| Conditioning {
                    mu: mu[b * condition..(b + 1) * condition].to_vec(),
                    stop_logits: [logit(2 * b), logit(2 * b + 1)],
                })
            })
            .collect()
        });
        let apart = first_apart(&got);
        eprintln!("wave{subgroup}: with the slots read crossed, first apart at {apart:?}");
        let crossed: Vec<bool> = apart.iter().map(Option::is_some).collect();
        assert_eq!(
            crossed,
            [true, false, true, false, false, false],
            "wave{subgroup}: {apart:?}"
        );
    }
    assert_no_validation_messages(&ctx);
}

/// The full stacks' tolerances: against f32, any run's were 1.49e-1 and 9.94e-3 over the
/// condition and 3.55e-2 and 4.56e-3 over the stop logits, and the golden's 1.43e-1 and 8.99e-3,
/// and 3.55e-2 and 4.56e-3.
const THE_LM_SIDE: Tolerance = Tolerance {
    mu: Distance {
        max_abs: 3.7e-1,
        relative_rms: 2.4e-2,
    },
    stop: Distance {
        max_abs: 8.9e-2,
        relative_rms: 1.2e-2,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_lm_side_conditions_as_its_golden() {
    let (layers, residual_layers) = full();
    let config = config(layers, residual_layers);
    let input = input(&config);
    let want = golden_of(&config, &input);
    let got = gpu(&config, &input, None, None, 1).remove(0);
    assert!(check("the LM side", &got, &want, &THE_LM_SIDE));
    assert!(decides_alike(&want.1, &got.1));
    // Sixteen requests at once, each at its own position: each as it is alone.
    let rows = gpu(&config, &input, None, None, 16);
    for (r, row) in rows.iter().enumerate() {
        assert!(row == &got, "row {r} of 16 is not as it is alone");
    }
}
