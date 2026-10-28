//! The decode step on the GPU against minicpm4's stack on burn's CPU backend, on VoxCPM2's real
//! base LM over a real prompt: a prefill on the CPU and one position at a time on the GPU, then
//! decode steps on both. Built with the `device` feature, and run with `VOXCPM2_DIR` naming the
//! checkpoint and `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests` sets it. The
//! GPU also runs the prompt as several requests at once, a row each, each row at its own position
//! in its own slot of the cache.
//!
//! The CPU's part is kept under the directory `ORACLE_DIR` names, when it is set, and read from
//! there by later runs: `keeps_the_oracles` runs it alone, touching no GPU.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use burn::tensor::{FloatDType, Int, Tensor, TensorData};
use graph::{
    checkpoint::Checkpoint,
    decode::{Batch, Decoder, LmConfig, Row, SUBGROUPS, Shape, Tuning},
    kernels::Kernels,
};
use half::f16;
use minicpm4::{Config, Model};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use testkit::{Fixture, checkpoint_dir, device, kept, oracle_dir};
use vulkan::Context;

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

/// Token ids upstream's tokenizer gave for two of the game's lines, 79 in all: positions in three
/// of attention's chunks of 32.
fn prompt_ids() -> Vec<i64> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_tokenizer_checkpoint.safetensors"
    );
    let fixture = Fixture::open(path);
    ["messages.15", "messages.24"]
        .iter()
        .flat_map(|name| {
            let ids = fixture.int_tensor::<1>(name, &device());
            ids.into_data().iter::<i64>().collect::<Vec<_>>()
        })
        .collect()
}

/// How far one run's outputs are from another's.
#[derive(Clone, Copy)]
struct Distance {
    max_abs: f64,
    /// The largest |a - b| / (atol + rtol |b|).
    worst: f64,
    /// RMS of the differences over RMS of the reference.
    relative_rms: f64,
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |d| {:.2e}, worst at {:.2} of the tolerance, relative RMS {:.2e}",
            self.max_abs, self.worst, self.relative_rms
        )
    }
}

fn distance(got: &[f32], want: &[f32], atol: f64, rtol: f64) -> Distance {
    let (mut max_abs, mut worst, mut err2, mut ref2) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        worst = worst.max(d / (atol + rtol * w.abs()));
        if !d.is_finite() {
            worst = f64::INFINITY;
        }
        err2 += d * d;
        ref2 += w * w;
    }
    Distance {
        max_abs,
        worst,
        relative_rms: (err2 / ref2).sqrt(),
    }
}

/// A stack of VoxCPM2's: its base LM; its residual LM, which goes without RoPE; or, as a defect,
/// the residual LM rotated as the base LM is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stack {
    Base,
    Residual,
    RotatedResidual,
}

impl Stack {
    fn prefix(self) -> &'static str {
        match self {
            Stack::Base => "base_lm.",
            _ => "residual_lm.",
        }
    }
}

/// What minicpm4 on the CPU gives for the first `layers` layers of a stack and its final norm,
/// over the prompt's embeddings rounded to f16, the first `prefill` positions at once and the
/// rest one at a time: the outputs in f32 and in f16, `[positions][hidden]` flattened, and the
/// embeddings as f16 bits.
#[derive(Serialize, Deserialize)]
struct Oracle {
    f32: Vec<f32>,
    f16: Vec<f32>,
    x: Vec<u16>,
}

fn config() -> Value {
    let dir = checkpoint_dir();
    serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn oracle(stack: Stack, layers: usize, prefill: usize) -> Oracle {
    kept(&format!("decode-{stack:?}-{layers}-{prefill}"), || {
        compute_oracle(stack, layers, prefill)
    })
}

fn compute_oracle(stack: Stack, layers: usize, prefill: usize) -> Oracle {
    let weights = checkpoint_dir().join("model.safetensors");
    let ids = prompt_ids();
    let len = ids.len();
    let cpu = device();
    let base: Config = serde_json::from_value(config()["lm_config"].clone()).unwrap();
    let config = match stack {
        Stack::Base => Config {
            num_hidden_layers: layers,
            ..base.clone()
        },
        _ => base.residual_lm(layers, stack == Stack::Residual),
    };
    let hidden = config.hidden_size;
    let load = |config: &Config, prefix, dtype| {
        Model::load(config, &weights, prefix, dtype, &cpu).unwrap_or_else(|e| panic!("{e}"))
    };

    let start = Instant::now();
    let embedding = Config {
        num_hidden_layers: 0,
        ..base
    };
    let embedding = load(&embedding, "base_lm.", FloatDType::F32);
    let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &cpu);
    let embeds: Vec<f32> = embedding.embed(ids).into_data().iter::<f32>().collect();
    drop(embedding);
    let x16: Vec<u16> = embeds.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let x: Vec<f32> = x16.iter().map(|&h| f16::from_bits(h).to_f32()).collect();
    let decode = |model: &Model, dtype: FloatDType| -> Vec<f32> {
        let x =
            Tensor::<3>::from_data(TensorData::new(x.clone(), [1, len, hidden]), &cpu).cast(dtype);
        let mut cache = model.cache(1, len);
        let mut outputs = vec![model.forward_cached(x.clone().narrow(1, 0, prefill), &mut cache)];
        for t in prefill..len {
            outputs.push(model.forward_cached(x.clone().narrow(1, t, 1), &mut cache));
        }
        let out = Tensor::cat(outputs, 1).cast(FloatDType::F32);
        out.into_data().iter::<f32>().collect()
    };
    let want = decode(
        &load(&config, stack.prefix(), FloatDType::F32),
        FloatDType::F32,
    );
    let f16_model = load(&config, stack.prefix(), FloatDType::F16);
    let burn_f16 = decode(&f16_model, FloatDType::F16);
    eprintln!(
        "{layers} layers on the CPU, f32 and f16: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    Oracle {
        f32: want,
        f16: burn_f16,
        x: x16,
    }
}

/// The GPU's outputs for the same layers of `stack` over the embeddings `x`, every position one at
/// a time, each at its position plus `shift` in the cache, with the kernels at `subgroup` (the
/// device's default when `None`).
fn gpu(stack: Stack, layers: usize, x: &[u16], shift: usize, subgroup: Option<u32>) -> Vec<f32> {
    gpu_rows(stack, layers, x, shift, subgroup, 1).remove(0)
}

/// [`gpu`]'s outputs for each of `rows` requests run at once: request r from step 2 r on, in slot
/// `rows - 1 - r` of the cache, in a row that changes from step to step, the rows without one
/// spare.
fn gpu_rows(
    stack: Stack,
    layers: usize,
    x: &[u16],
    shift: usize,
    subgroup: Option<u32>,
    rows: usize,
) -> Vec<Vec<f32>> {
    let json = config();
    let hidden = json["lm_config"]["hidden_size"].as_u64().unwrap() as usize;
    let len = x.len() / hidden;
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let config = LmConfig {
        no_rope: stack == Stack::Residual,
        ..serde_json::from_value(json["lm_config"].clone()).unwrap()
    };
    let shape = Shape::new(&config, layers, (len + shift).next_multiple_of(32)).unwrap();
    let start = Instant::now();
    let tuning = match subgroup {
        Some(s) => Tuning::for_subgroup(s),
        None => Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}")),
    };
    let batch = Batch { rows, slots: rows };
    let mut decoder = Decoder::with_batch(&ctx, &mut kernels, &shape, &tuning, batch, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, stack.prefix())
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{layers} layers in {rows} rows recorded and loaded on the GPU at {tuning:?}: {:.1} s, \
         {} dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        decoder.graph_step().dispatches(),
        decoder.graph_step().barriers(),
        kernels.len(),
        decoder.graph_step().arena_bytes() / 1024
    );
    let mut got = vec![Vec::with_capacity(len * hidden); rows];
    for t in 0..len + 2 * (rows - 1) {
        let mut placed = vec![None; rows];
        let mut input = vec![0u16; rows * hidden];
        for r in 0..rows {
            let Some(i) = t.checked_sub(2 * r).filter(|&i| i < len) else {
                continue;
            };
            let row = (r + t) % rows;
            placed[row] = Some(Row {
                position: i + shift,
                slot: rows - 1 - r,
            });
            input[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&x[i * hidden..(i + 1) * hidden]);
        }
        let out = decoder
            .step_rows(&placed, &input)
            .unwrap_or_else(|e| panic!("{e}"));
        for r in (0..rows).filter(|r| (2 * r..2 * r + len).contains(&t)) {
            let row = (r + t) % rows;
            let out = &out[row * hidden..(row + 1) * hidden];
            got[r].extend(out.iter().map(|&h| f16::from_bits(h).to_f32()));
        }
    }
    assert_no_validation_messages(&ctx);
    got
}

/// How far the GPU's outputs and burn's f16 ones are from f32 at each position.
fn per_position(
    want: &[f32],
    burn_f16: &[f32],
    got: &[f32],
    hidden: usize,
    (atol, rtol): (f64, f64),
) -> Vec<(Distance, Distance)> {
    want.chunks(hidden)
        .zip(burn_f16.chunks(hidden))
        .zip(got.chunks(hidden))
        .map(|((w, b), g)| (distance(g, w, atol, rtol), distance(b, w, atol, rtol)))
        .collect()
}

/// The served f16 tolerance of crates/gpu's real-weight tests, set for 8 layers.
const SERVED: (f64, f64) = (2e-2, 4e-3);

/// Checks that the GPU stays as close to f32 as f16 arithmetic on the CPU does: over the prefill,
/// its largest error and its relative RMS error within 1.25 times burn's f16; over the decode
/// steps, where burn's f16 drifts on the CPU, both within 1.25 times what burn's f16 reached over
/// the prefill. Errors are compared over many positions: in a deep stack they gather in a few
/// outlier channels, and at any one position which of two runs rounds those worse is luck.
fn check(want: &[f32], burn_f16: &[f32], got: &[f32], prefill: usize) -> (Distance, [f64; 4]) {
    let hidden = 2048;
    let distances = per_position(want, burn_f16, got, hidden, SERVED);
    let mut worst_position = (0, 0f64);
    for (t, (ours, burn)) in distances.iter().enumerate() {
        if t % 8 == 0 || t + 1 == distances.len() {
            eprintln!("position {t}: GPU {ours:?}; burn f16 {burn:?}");
        }
        let ratio = ours.relative_rms / burn.relative_rms;
        if t < prefill && ratio > worst_position.1 {
            worst_position = (t, ratio);
        }
    }
    let split = prefill * hidden;
    let ours_prefill = distance(&got[..split], &want[..split], SERVED.0, SERVED.1);
    let burn_prefill = distance(&burn_f16[..split], &want[..split], SERVED.0, SERVED.1);
    let ours_decode = distance(&got[split..], &want[split..], SERVED.0, SERVED.1);
    let margin = [
        ours_prefill.max_abs / burn_prefill.max_abs,
        ours_prefill.relative_rms / burn_prefill.relative_rms,
        ours_decode.max_abs / burn_prefill.max_abs,
        ours_decode.relative_rms / burn_prefill.relative_rms,
    ];
    eprintln!(
        "prefill: GPU {ours_prefill:?}; burn f16 {burn_prefill:?}\ndecode steps: GPU \
         {ours_decode:?}\nthe GPU's against burn's f16: {margin:.2?}; at one position its \
         relative RMS is at most {:.2} of burn's, at {}",
        worst_position.1, worst_position.0
    );
    (distance(got, want, SERVED.0, SERVED.1), margin)
}

fn within(margin: [f64; 4]) -> bool {
    margin.iter().all(|&m| m <= 1.25)
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and ORACLE_DIR a directory to keep \
            the oracles in"]
fn keeps_the_oracles() {
    if oracle_dir().is_none() {
        eprintln!("ORACLE_DIR is not set: nothing to keep the oracles in");
        return;
    }
    oracle(Stack::Base, 1, 70);
    oracle(Stack::Base, 28, 70);
    oracle(Stack::Residual, 8, 70);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_layer_decodes_as_minicpm4_does_on_the_cpu() {
    let oracle = oracle(Stack::Base, 1, 70);
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(Stack::Base, 1, &oracle.x, 0, Some(subgroup));
        let (ours, margin) = check(&oracle.f32, &oracle.f16, &got, 70);
        assert!(within(margin), "wave{subgroup}: {margin:?}");
        // One layer is well inside the served tolerance, which was set for eight.
        assert!(ours.worst <= 1.0, "wave{subgroup}: {ours:?}");
        // Three requests at once, each at its own position: each as it is alone.
        let rows = gpu_rows(Stack::Base, 1, &oracle.x, 0, Some(subgroup), 3);
        for (r, row) in rows.iter().enumerate() {
            assert!(
                row == &got,
                "wave{subgroup}: row {r} of 3 is not as it is alone"
            );
        }
    }
    // Defect: every position one slot on in the cache, so rotated by the next position's angle.
    let shifted = gpu(Stack::Base, 1, &oracle.x, 1, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &shifted, 70);
    assert!(!within(margin), "positions one slot on pass: {margin:?}");
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_layer_recorded_again_with_its_rows_cut_otherwise_decodes_the_same() {
    let json = config();
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let positions = 40;
    let shape = Shape::new(&config, 1, 64).unwrap();
    let base = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &base, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, "base_lm.")
        .unwrap_or_else(|e| panic!("{e}"));
    // Hidden states of about the embeddings' scale, from a fixed xorshift sequence.
    let mut state = 0x9e37_79b9_u32;
    let x: Vec<u16> = (0..positions * shape.hidden)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            f16::from_f64((f64::from(state) / f64::from(u32::MAX) - 0.5) * 0.2).to_bits()
        })
        .collect();
    let run = |decoder: &mut Decoder| -> Vec<u16> {
        x.chunks(shape.hidden)
            .enumerate()
            .flat_map(|(t, x)| decoder.step(t, x).unwrap_or_else(|e| panic!("{e}")))
            .collect()
    };
    let want = run(&mut decoder);
    // Rows to a subgroup, loads ahead and the workgroup width decide where and when each row is
    // summed, not in what order: the norms are summed in the order of the default's workgroup.
    let cuts = [
        (
            Tuning {
                workgroup: 4 * base.workgroup,
                ..base
            },
            false,
        ),
        (
            Tuning {
                qkv_pairs: 2,
                gate_up_pairs: 2,
                residual_rows: 2,
                ..base
            },
            false,
        ),
        (Tuning { ahead: 64, ..base }, false),
        (base, true),
    ];
    for (tuning, timed) in cuts {
        decoder = decoder
            .retuned(&ctx, &mut kernels, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        let got = run(&mut decoder);
        let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(
            differ,
            0,
            "{tuning:?}, timed {timed}: {differ} of {} differ",
            want.len()
        );
    }
    eprintln!("{positions} positions alike at {base:?} and recorded again as {cuts:?}");
    assert_no_validation_messages(&ctx);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_base_lm_decodes_as_minicpm4_does_on_the_cpu() {
    let oracle = oracle(Stack::Base, 28, 70);
    let got = gpu(Stack::Base, 28, &oracle.x, 0, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &got, 70);
    assert!(within(margin), "{margin:?}");
    // Sixteen requests at once, each at its own position: each as it is alone.
    let rows = gpu_rows(Stack::Base, 28, &oracle.x, 0, None, 16);
    for (r, row) in rows.iter().enumerate() {
        assert!(row == &got, "row {r} of 16 is not as it is alone");
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_residual_lm_decodes_without_rope_as_minicpm4_does_on_the_cpu() {
    let oracle = oracle(Stack::Residual, 8, 70);
    let got = gpu(Stack::Residual, 8, &oracle.x, 0, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &got, 70);
    assert!(within(margin), "{margin:?}");
    let rows = gpu_rows(Stack::Residual, 8, &oracle.x, 0, None, 4);
    for (r, row) in rows.iter().enumerate() {
        assert!(row == &got, "row {r} of 4 is not as it is alone");
    }
    // Defect: the residual LM rotated by RoPE.
    let rotated = gpu(Stack::RotatedResidual, 8, &oracle.x, 0, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &rotated, 70);
    assert!(
        !within(margin),
        "the residual LM rotated passes: {margin:?}"
    );
}
