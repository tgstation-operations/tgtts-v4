//! The decode step on the GPU against minicpm4's stack on burn's CPU backend, on VoxCPM2's real
//! base LM over a real prompt: a prefill on the CPU and one position at a time on the GPU, then
//! decode steps on both. Built with the `device` feature, and run with `VOXCPM2_DIR` naming the
//! checkpoint and `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests` sets it.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use burn::tensor::{FloatDType, Int, Tensor, TensorData};
use graph::{
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, SUBGROUPS, Shape, Tuning},
    kernels::Kernels,
};
use half::f16;
use minicpm4::{Config, Model};
use serde_json::Value;
use testkit::{Fixture, checkpoint_dir, device};
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

/// What minicpm4 on the CPU gives for the base LM's first `layers` layers and its final norm,
/// over the prompt's embeddings rounded to f16, the first `prefill` positions at once and the
/// rest one at a time: the outputs in f32 and in f16, `[positions][hidden]` flattened, and the
/// embeddings as f16 bits.
struct Oracle {
    f32: Vec<f32>,
    f16: Vec<f32>,
    x: Vec<u16>,
}

fn config() -> Value {
    let dir = checkpoint_dir();
    serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn oracle(layers: usize, prefill: usize) -> Oracle {
    let weights = checkpoint_dir().join("model.safetensors");
    let ids = prompt_ids();
    let len = ids.len();
    let cpu = device();
    let mut config: Config = serde_json::from_value(config()["lm_config"].clone()).unwrap();
    config.num_hidden_layers = layers;
    let hidden = config.hidden_size;
    let load = |dtype| {
        Model::load(&config, &weights, "base_lm.", dtype, &cpu).unwrap_or_else(|e| panic!("{e}"))
    };

    let start = Instant::now();
    let f32_model = load(FloatDType::F32);
    let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &cpu);
    let embeds: Vec<f32> = f32_model.embed(ids).into_data().iter::<f32>().collect();
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
    let want = decode(&f32_model, FloatDType::F32);
    drop(f32_model);
    let burn_f16 = decode(&load(FloatDType::F16), FloatDType::F16);
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

/// The GPU's outputs for the same layers over the embeddings `x`, every position one at a time,
/// each at its position plus `shift` in the cache, with the kernels at `subgroup` (the device's
/// default when `None`).
fn gpu(layers: usize, x: &[u16], shift: usize, subgroup: Option<u32>) -> Vec<f32> {
    let json = config();
    let hidden = json["lm_config"]["hidden_size"].as_u64().unwrap() as usize;
    let len = x.len() / hidden;
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let shape = Shape::new(&config, layers, (len + shift).next_multiple_of(32)).unwrap();
    let start = Instant::now();
    let tuning = match subgroup {
        Some(s) => Tuning::for_subgroup(s),
        None => Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}")),
    };
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, "base_lm.")
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{layers} layers recorded and loaded on the GPU at {tuning:?}: {:.1} s, {} dispatches, \
         {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        decoder.graph_step().dispatches(),
        decoder.graph_step().barriers(),
        kernels.len(),
        decoder.graph_step().arena_bytes() / 1024
    );
    let mut got = Vec::with_capacity(len * hidden);
    for (t, x) in x.chunks(hidden).enumerate() {
        let out = decoder.step(t + shift, x).unwrap_or_else(|e| panic!("{e}"));
        got.extend(out.iter().map(|&h| f16::from_bits(h).to_f32()));
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
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_layer_decodes_as_minicpm4_does_on_the_cpu() {
    let oracle = oracle(1, 70);
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(1, &oracle.x, 0, Some(subgroup));
        let (ours, margin) = check(&oracle.f32, &oracle.f16, &got, 70);
        assert!(within(margin), "wave{subgroup}: {margin:?}");
        // One layer is well inside the served tolerance, which was set for eight.
        assert!(ours.worst <= 1.0, "wave{subgroup}: {ours:?}");
    }
    // Defect: every position one slot on in the cache, so rotated by the next position's angle.
    let shifted = gpu(1, &oracle.x, 1, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &shifted, 70);
    assert!(!within(margin), "positions one slot on pass: {margin:?}");
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_base_lm_decodes_as_minicpm4_does_on_the_cpu() {
    let oracle = oracle(28, 70);
    let got = gpu(28, &oracle.x, 0, None);
    let (_, margin) = check(&oracle.f32, &oracle.f16, &got, 70);
    assert!(within(margin), "{margin:?}");
}
