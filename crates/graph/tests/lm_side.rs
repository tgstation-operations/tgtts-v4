//! The LM side on the GPU against voxcpm's model on burn's CPU backend, on VoxCPM2's real weights.
//! The model is prompted with a voice's patches alone, which it runs as it runs the patches it
//! generates, and generates patches from seeded noise; the GPU is given the local encoder's output
//! for every patch, the prompt's one at a time, and gives the condition and the stop logits the
//! model's `conditioning` gives, from the prompt's last position on. burn's f16 is fed the f32
//! run's patches, so that all three follow one generation. Built with the `device` feature, and run
//! with `VOXCPM2_DIR` naming the checkpoint, `VOICES_DIR` a voices directory and `GLSL_KERNELS`
//! the compiled kernels, as `nix run .#device-tests` sets the last.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use burn::tensor::{FloatDType, Tensor, TensorData};
use graph::{
    checkpoint::Checkpoint,
    decode::{SUBGROUPS, Tuning},
    kernels::Kernels,
    lm_side::{self, LmSide, LmSideShape},
};
use half::f16;
use testkit::{Fixture, checkpoint_dir, device};
use voices::Latent;
use voxcpm::{Config, Dtypes, Generation, LocalEncoder, Model, Prompt, Sampling, SeededNoise};
use vulkan::Context;

/// Patches generated after the prompt.
const GENERATED: usize = 8;

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

/// The latent frames of the voice voxcpm's golden was made with.
fn voice() -> Vec<f32> {
    let golden = Fixture::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_checkpoint.safetensors"
    ));
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let dir = std::path::absolute(dir).unwrap();
    let path = voices::latent_path(&dir, &golden.metadata()["voice"]);
    let latent = Latent::read(&path).unwrap_or_else(|e| panic!("{e}"));
    latent.frames.as_flattened().to_vec()
}

/// VoxCPM2's config, as voxcpm and as the LM side read it, with `layers` of the base LM and
/// `residual_layers` of the residual LM.
fn configs(layers: usize, residual_layers: usize) -> (Config, lm_side::Config) {
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let mut config = Config::from_json(&json).unwrap();
    config.lm_config.num_hidden_layers = layers;
    config.residual_lm_num_layers = residual_layers;
    let mut ours: lm_side::Config = serde_json::from_str(&json).unwrap();
    ours.lm_config.num_hidden_layers = layers;
    ours.residual_lm_num_layers = residual_layers;
    (config, ours)
}

/// What voxcpm's model gives on the CPU.
struct Oracle {
    /// The local encoder's output for every patch, the prompt's and then the generated ones,
    /// rounded to f16: `[patches][encoder]` as f16 bits.
    enc: Vec<u16>,
    /// The patches of the prompt.
    prompt: usize,
    /// The condition `[GENERATED + 1][2 * dit]` and the stop logits `[GENERATED + 1][2]` after the
    /// prompt and after each generated patch, in f32 and in burn's f16.
    mu: [Vec<f32>; 2],
    stop: [Vec<f32>; 2],
}

fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32).into_data().iter::<f32>().collect()
}

fn oracle(config: &Config) -> Oracle {
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let voice = voice();
    let patch = config.patch_size * config.feat_dim;
    let len = voice.len() / patch;
    let prompt = Prompt {
        tokens: vec![0; len],
        patches: voice.clone(),
        audio: vec![true; len],
    };
    let load = |lm| {
        let dtypes = Dtypes {
            lm,
            local_encoder: FloatDType::F32,
            local_dit: FloatDType::F32,
            audio_decoder: FloatDType::F32,
        };
        Model::load(config, &weights, &dtypes, &cpu).unwrap_or_else(|e| panic!("{e}"))
    };
    let start = Instant::now();

    let model = load(FloatDType::F32);
    let mut generation = model.start(&prompt, GENERATED).unwrap();
    let mut noise = SeededNoise::new(0);
    let (mut mu, mut stop, mut patches) = (vec![], vec![], vec![]);
    for i in 0..=GENERATED {
        let (m, s) = generation.conditioning();
        mu.extend(floats(m));
        stop.extend(floats(s));
        if i < GENERATED {
            let step = generation.sample(model.noise(&mut noise), &Sampling::default());
            patches.push(step.patch.clone());
            generation.advance(step.patch);
        }
    }
    drop(generation);
    drop(model);

    let encoder = LocalEncoder::load(config, &weights, FloatDType::F32, &cpu).unwrap();
    let shape = [1, len, config.patch_size, config.feat_dim];
    let prompt_patches = Tensor::<4>::from_data(TensorData::new(voice, shape), &cpu);
    let generated = Tensor::cat(patches.clone(), 0).unsqueeze_dim::<4>(0);
    let enc = encoder.forward(Tensor::cat(vec![prompt_patches, generated], 1));
    let enc = floats(enc)
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect();
    drop(encoder);

    let model = load(FloatDType::F16);
    let mut generation = model.start(&prompt, GENERATED).unwrap();
    let (mut mu16, mut stop16) = (vec![], vec![]);
    let mut record = |generation: &Generation| {
        let (m, s) = generation.conditioning();
        mu16.extend(floats(m));
        stop16.extend(floats(s));
    };
    record(&generation);
    for patch in patches {
        generation.advance(patch);
        record(&generation);
    }
    eprintln!(
        "{} and {} layers on the CPU, f32 and f16: {:.1} s",
        config.lm_config.num_hidden_layers,
        config.residual_lm_num_layers,
        start.elapsed().as_secs_f64()
    );
    Oracle {
        enc,
        prompt: len,
        mu: [mu, mu16],
        stop: [stop, stop16],
    }
}

/// The GPU's conditions and stop logits for the oracle's patches, from the prompt's last on, with
/// the kernels at `subgroup` (the device's default when `None`), the residual LM rotated by RoPE
/// as `shape` says unless `residual_rope` says otherwise.
fn gpu(
    config: &lm_side::Config,
    oracle: &Oracle,
    subgroup: Option<u32>,
    residual_rope: Option<bool>,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let positions = oracle.enc.len() / config.encoder_config.hidden_dim;
    let mut shape = LmSideShape::new(config, positions.next_multiple_of(32)).unwrap();
    if let Some(rope) = residual_rope {
        shape.residual.rope = rope;
    }
    let tuning = match subgroup {
        Some(s) => Tuning::for_subgroup(s),
        None => Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}")),
    };
    let start = Instant::now();
    let mut side =
        LmSide::new(&ctx, &mut kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    side.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let step = side.graph_step();
    eprintln!(
        "{} and {} layers recorded and loaded on the GPU at {tuning:?}: {:.1} s, {} dispatches, {} \
         barriers, {} pipelines, an arena of {} KiB",
        shape.base.layers,
        shape.residual.layers,
        start.elapsed().as_secs_f64(),
        step.dispatches(),
        step.barriers(),
        kernels.len(),
        step.arena_bytes() / 1024
    );
    let (mut mu, mut stop) = (vec![], vec![]);
    for (t, enc) in oracle.enc.chunks(shape.encoder).enumerate() {
        let out = side.step(t, enc).unwrap_or_else(|e| panic!("{e}"));
        if t + 1 >= oracle.prompt {
            mu.extend(out.mu.iter().map(|&h| f16::from_bits(h).to_f32()));
            stop.extend(out.stop_logits);
        }
    }
    assert_no_validation_messages(&ctx);
    (mu, stop)
}

/// How far one run's outputs are from another's.
#[derive(Clone, Copy)]
struct Distance {
    max_abs: f64,
    /// RMS of the differences over RMS of the reference.
    relative_rms: f64,
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

/// How the GPU's conditions and stop logits compare with burn's f16, both against f32: the GPU's
/// largest error and relative RMS error over every patch, each over burn's, for the condition and
/// for the stop logits. Errors are compared over all the patches, not at each: the scalar
/// quantizer rounds a value either way of a level by luck, and one level is a ninth.
fn check(oracle: &Oracle, (mu, stop): &(Vec<f32>, Vec<f32>)) -> [f64; 4] {
    let [want, burn] = &oracle.mu;
    let width = want.len() / (GENERATED + 1);
    for i in 0..=GENERATED {
        let at = |v: &[f32]| v[i * width..(i + 1) * width].to_vec();
        let logits = |v: &[f32]| [v[2 * i], v[2 * i + 1]];
        eprintln!(
            "after patch {i}: condition GPU {:?}, burn f16 {:?}; stop logits f32 {:?}, GPU {:?}, \
             burn f16 {:?}",
            distance(&at(mu), &at(want)),
            distance(&at(burn), &at(want)),
            logits(&oracle.stop[0]),
            logits(stop),
            logits(&oracle.stop[1]),
        );
    }
    let (ours, theirs) = (distance(mu, want), distance(burn, want));
    let (ours_stop, theirs_stop) = (
        distance(stop, &oracle.stop[0]),
        distance(&oracle.stop[1], &oracle.stop[0]),
    );
    let margin = [
        ours.max_abs / theirs.max_abs,
        ours.relative_rms / theirs.relative_rms,
        ours_stop.max_abs / theirs_stop.max_abs,
        ours_stop.relative_rms / theirs_stop.relative_rms,
    ];
    eprintln!(
        "condition: GPU {ours:?}; burn f16 {theirs:?}\nstop logits: GPU {ours_stop:?}; burn f16 \
         {theirs_stop:?}\nthe GPU's against burn's f16: {margin:.2?}"
    );
    margin
}

fn within(margin: [f64; 4]) -> bool {
    margin.iter().all(|&m| m <= 1.25)
}

/// Whether the GPU and f32 decide alike whether to stop, wherever f32's margin between going on
/// and stopping is wider than the GPU's error in the logits.
fn decides_alike(oracle: &Oracle, stop: &[f32]) -> bool {
    let want = &oracle.stop[0];
    let error = distance(stop, want).max_abs as f32;
    want.chunks(2).zip(stop.chunks(2)).all(|(w, g)| {
        let margin = w[1] - w[0];
        margin.abs() <= 2.0 * error || (margin > 0.0) == (g[1] > g[0])
    })
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn a_few_layers_condition_as_voxcpm_does_on_the_cpu() {
    let (config, ours) = configs(2, 1);
    let oracle = oracle(&config);
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(&ours, &oracle, Some(subgroup), None);
        let margin = check(&oracle, &got);
        assert!(within(margin), "wave{subgroup}: {margin:?}");
        assert!(decides_alike(&oracle, &got.1), "wave{subgroup}");
    }
    // Defect: the residual LM rotated by RoPE, as the base LM is.
    let rotated = gpu(&ours, &oracle, None, Some(true));
    let margin = check(&oracle, &rotated);
    assert!(
        !within(margin),
        "a residual LM with RoPE passes: {margin:?}"
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn the_lm_side_conditions_as_voxcpm_does_on_the_cpu() {
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let full: lm_side::Config = serde_json::from_str(&json).unwrap();
    let (config, ours) = configs(
        full.lm_config.num_hidden_layers,
        full.residual_lm_num_layers,
    );
    let oracle = oracle(&config);
    let got = gpu(&ours, &oracle, None, None);
    let margin = check(&oracle, &got);
    assert!(within(margin), "{margin:?}");
    assert!(decides_alike(&oracle, &got.1));
}
