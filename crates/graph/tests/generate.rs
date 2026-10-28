//! VoxCPM2's generation on the GPU against voxcpm's on burn's CPU backend, on the real weights.
//! The CPU generates patches from seeded noise in f32, and again in burn's f16 sampling each patch
//! after f32's patch before; the GPU samples each from the same noise after f32's patch before
//! too, so that all three follow one generation. Built with the `device` feature, and run with
//! `VOXCPM2_DIR` naming the checkpoint, `VOICES_DIR` a voices directory and `GLSL_KERNELS` and
//! `VULKAN_KERNELS` the compiled kernels, as `nix run .#device-tests` sets the last two.
//!
//! The patch step is checked on a prompt of a voice's patches alone, run through the patch step a
//! position at a time: the sampler conditioned on what the LM side computes, where its own tests
//! make the condition up. The prompt is checked on the one the service builds, of the golden's
//! line in its voice, run at once into the patch step's caches. The generator speaks that line,
//! after f32's patches as above, its audio held to burn's f16's patches' against f32's, all
//! decoded in f32; and run free, it stops where f32 does, its audio what the CPU's decoder makes
//! of its own patches.
//!
//! The scalar quantizer rounds the base LM's output to levels, and a value near a boundary
//! between two lands on either by luck of rounding: a level apart from f32's is a discrete event,
//! which the sampler amplifies into the patch, not an error of the rounding's size. On the real
//! weights levels are apart at most positions, for burn's f16 as for the GPU. So the GPU is held
//! to burn's f16 on the quantizer's input, a continuous value, by its largest and its RMS error;
//! on how many levels are apart; on the stop logits by their largest and RMS error; on the
//! decisions to stop; and on the patches by being the closer to f32's at half of them at least,
//! and at each where none of its levels is apart when a run has `CLEAN` such patches. Only a
//! run with fewer is held to the patches' RMS error over all of them: where the GPU's error is
//! otherwise small, one level apart that the sampler amplifies decides that error, and how many
//! are apart is held already. Teacher forcing keeps a level's effect on a patch from reaching the
//! next through the patch.
//!
//! The CPU's part is kept under the directory `ORACLE_DIR` names, when it is set, and read from
//! there by later runs: `keeps_the_oracles` runs it alone, touching no GPU.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use audiovae::Weights;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    store::{
        FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    },
    tensor::{Bool, FloatDType, Int, Tensor, TensorData},
};
use graph::{
    checkpoint::Checkpoint,
    decode::{self, SUBGROUPS},
    generator::{Generator, Settings},
    graph::{Kind, TensorId},
    kernels::Kernels,
    lm_side::{self, LmSideShape},
    local::{self, PatchConfig},
    patch::{PatchStep, patch_graph},
    prefill::TILE,
    prompt::Prompter,
};
use half::f16;
use serde::{Deserialize, Serialize};
use testkit::{Fixture, checkpoint_dir, device, kept, oracle_dir};
use voices::Latent;
use voxcpm::{
    Config, Dtypes, LocalEncoder, Model, Noise, Params, Prompt, Sampling, SeededNoise,
    SpeechDecoder, Tokenizer,
};
use vulkan::Context;

/// Patches sampled after the voice's patches alone.
const GENERATED: usize = 24;

const SEED: u64 = 0;

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

fn config_json() -> String {
    fs::read_to_string(checkpoint_dir().join("config.json")).unwrap()
}

fn config() -> Config {
    Config::from_json(&config_json()).unwrap()
}

fn golden() -> Fixture {
    Fixture::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_checkpoint.safetensors"
    ))
}

/// The voice voxcpm's golden was made with, by its name.
fn voice() -> (String, Latent) {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let dir = std::path::absolute(dir).unwrap();
    let name = golden().metadata()["voice"].clone();
    let latent = Latent::read(&voices::latent_path(&dir, &name)).unwrap_or_else(|e| panic!("{e}"));
    (name, latent)
}

/// What voxcpm's model gives on the CPU for a prompt, in f32 and in burn's f16, the second each
/// patch after the first's patch before.
#[derive(Serialize, Deserialize)]
struct Oracle {
    /// The prompt, as voxcpm's `Prompt` holds it.
    tokens: Vec<u32>,
    prompt: Vec<f32>,
    audio: Vec<bool>,
    /// The noise each patch is sampled from, `[feat_dim][patch_size]`.
    noise: Vec<Vec<f32>>,
    /// Each patch, `[patch_size][feat_dim]`.
    patches: [Vec<Vec<f32>>; 2],
    /// The stop logits each patch was sampled beside.
    stop: [Vec<[f32; 2]>; 2],
    /// The base LM's output at the position each patch is sampled after, the scalar quantizer's
    /// input.
    base: [Vec<Vec<f32>>; 2],
    /// The quantizer's levels there, integers.
    levels: [Vec<Vec<f32>>; 2],
}

impl Oracle {
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32).into_data().iter::<f32>().collect()
}

/// The voice's patches alone as the prompt, and `GENERATED` patches after it whatever the stop
/// head says.
fn voice_oracle() -> Oracle {
    let (name, voice) = voice();
    kept(&format!("patch-{name}-{GENERATED}-{SEED}"), || {
        let frames = voice.frames.as_flattened().to_vec();
        let len = frames.len() / (64 * 4);
        let prompt = Prompt {
            tokens: vec![0; len],
            patches: frames,
            audio: vec![true; len],
        };
        compute_oracle(&prompt, GENERATED, false)
    })
}

/// The prompt the service builds for the golden's line in its voice, and the patches after it
/// until the stop head says to stop, as the service stops.
fn line_oracle() -> Oracle {
    let (name, voice) = voice();
    kept(&format!("line-{name}-{SEED}"), || {
        let dir = checkpoint_dir();
        let tokenizer = fs::read_to_string(dir.join("tokenizer.json")).unwrap();
        let tokenizer = Tokenizer::from_json(&tokenizer).unwrap();
        let text = golden().metadata()["text"].clone();
        let tokens = tokenizer.encode(&format!("{}{text}", voice.text));
        let prompt = Prompt::cloning(&tokens, voice.frames.as_flattened(), &config()).unwrap();
        compute_oracle(&prompt, Params::default().max_patches, true)
    })
}

fn dtypes(dtype: FloatDType) -> Dtypes {
    Dtypes {
        lm: dtype,
        local_encoder: dtype,
        local_dit: dtype,
        audio_decoder: FloatDType::F32,
    }
}

/// Up to `most` patches after `prompt`, `until_stop` after the first the stop head says to stop
/// after.
fn compute_oracle(prompt: &Prompt, most: usize, until_stop: bool) -> Oracle {
    let config = config();
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let (patch_size, feat_dim) = (config.patch_size, config.feat_dim);
    let sampling = Sampling::default();
    let tensor = |values: &[f32], shape: [usize; 3]| {
        Tensor::<3>::from_data(TensorData::new(values.to_vec(), shape), &cpu)
    };
    let start = Instant::now();
    let mut draws = SeededNoise::new(SEED);
    let (mut noise, mut patches, mut stop) = (vec![], [vec![], vec![]], [vec![], vec![]]);
    let mut base = [vec![], vec![]];
    let mut levels = [vec![], vec![]];
    for (run, dtype) in [FloatDType::F32, FloatDType::F16].into_iter().enumerate() {
        let model =
            Model::load(&config, &weights, &dtypes(dtype), &cpu).unwrap_or_else(|e| panic!("{e}"));
        let mut generation = model.start(prompt, most).unwrap();
        for i in 0..most {
            if run == 0 {
                noise.push(draws.next(feat_dim * patch_size));
            } else if i == noise.len() {
                break;
            }
            let step = generation.sample(tensor(&noise[i], [1, feat_dim, patch_size]), &sampling);
            let logits = floats(step.stop_logits.clone());
            patches[run].push(floats(step.patch.clone()));
            stop[run].push([logits[0], logits[1]]);
            if run == 0 && until_stop && step.stops() {
                break;
            }
            let before = &patches[0][i];
            generation.advance(tensor(before, [1, patch_size, feat_dim]));
        }
        drop(generation);
        drop(model);
        let fed = &patches[0][..noise.len() - 1];
        [base[run], levels[run]] = quantizer(&config, prompt, fed, most, dtype);
    }
    eprintln!(
        "{} positions of prompt and {} patches after it on the CPU, f32 and f16: {:.1} s",
        prompt.len(),
        noise.len(),
        start.elapsed().as_secs_f64()
    );
    Oracle {
        tokens: prompt.tokens.clone(),
        prompt: prompt.patches.clone(),
        audio: prompt.audio.clone(),
        noise,
        patches,
        stop,
        base,
        levels,
    }
}

/// The layers of VoxCPM2's around the base LM that the quantizer's input and levels need, under
/// their names in the checkpoint.
#[derive(Module, Debug)]
struct Heads {
    enc_to_lm_proj: Linear,
    fsq_layer: Quantizer,
}

#[derive(Module, Debug)]
struct Quantizer {
    in_proj: Linear,
}

/// The base LM's outputs and the scalar quantizer's levels at the position before each patch,
/// the prompt's last and then `fed`'s, as voxcpm's model computes them in `dtype` with caches for
/// `most` patches after the prompt, which keeps them to itself: the prompt embedded as
/// `run_prompt` embeds it, the local encoder and enc_to_lm_proj over all its patches at once and
/// the base LM's embedding table at its tokens, and each of `fed` by the encoder and the
/// projection; the base LM over the prompt at once and then a position at a time; and the
/// quantizer over its outputs as they come.
fn quantizer(
    config: &Config,
    prompt: &Prompt,
    fed: &[Vec<f32>],
    most: usize,
    dtype: FloatDType,
) -> [Vec<Vec<f32>>; 2] {
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let (patch_size, feat_dim) = (config.patch_size, config.feat_dim);
    let lm = &config.lm_config;
    let hidden = lm.hidden_size;
    let len = prompt.len();
    let encoder = LocalEncoder::load(config, &weights, dtype, &cpu).unwrap();
    let base = minicpm4::Model::load(lm, &weights, "base_lm.", dtype, &cpu).unwrap();
    let mut heads = Heads {
        enc_to_lm_proj: LinearConfig::new(config.encoder_config.hidden_dim, hidden).init(&cpu),
        fsq_layer: Quantizer {
            in_proj: LinearConfig::new(hidden, config.scalar_quantization_latent_dim).init(&cpu),
        },
    };
    let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype.into()));
    let mut store = SafetensorsStore::from_file(&weights).with_from_adapter(adapter);
    heads
        .load_from(&mut store)
        .unwrap_or_else(|e| panic!("{e}"));
    let scale = f64::from(config.scalar_quantization_scale);
    let embed = |patches: Vec<f32>, len: usize| {
        let shape = [1, len, patch_size, feat_dim];
        let patches = Tensor::<4>::from_data(TensorData::new(patches, shape), &cpu);
        heads
            .enc_to_lm_proj
            .forward(encoder.forward(patches).cast(dtype))
    };
    let tokens: Vec<i64> = prompt.tokens.iter().map(|&t| i64::from(t)).collect();
    let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, len]), &cpu);
    let audio = TensorData::new(prompt.audio.clone(), [1, len, 1]);
    let audio = Tensor::<3, Bool>::from_data(audio, &cpu).expand([1, len, hidden]);
    let x = base
        .embed(tokens)
        .mask_where(audio, embed(prompt.patches.clone(), len));
    let mut cache = base.cache(1, len + most);
    let mut outputs = vec![base.forward_cached(x, &mut cache)];
    for patch in fed {
        outputs.push(base.forward_cached(embed(patch.clone(), 1), &mut cache));
    }
    let mut found = [vec![], vec![]];
    for out in outputs {
        let levels = heads.fsq_layer.in_proj.forward(out.clone());
        let levels = levels.tanh().mul_scalar(scale).round();
        let [_, rows, _] = out.dims();
        let last = |x: Tensor<3>| floats(x.narrow(1, rows - 1, 1));
        found[1].push(last(levels));
        found[0].push(last(out));
    }
    found
}

/// `[rows][columns]` as `[columns][rows]`.
fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
    let columns = values.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect()
}

/// The patch step on the GPU for `oracle`'s prompt and patches, recorded and loaded, the LM side's
/// kernels at `subgroup` and the local stacks' at the device's tuning at that subgroup size (the
/// device's default when `None`); and the tensors of the base LM's output and the quantizer's
/// levels, made outputs for the host to read.
fn patch_step(
    ctx: &Context,
    kernels: &mut Kernels,
    oracle: &Oracle,
    subgroup: Option<u32>,
) -> (PatchStep, [TensorId; 2], decode::Tuning) {
    let json = config_json();
    let side_config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let capacity = (oracle.len() + oracle.noise.len()).next_multiple_of(32);
    let shape = LmSideShape::new(&side_config, capacity).unwrap();
    let report = ctx.report().clone();
    let lm_default = decode::Tuning::for_device(&report).unwrap_or_else(|e| panic!("{e}"));
    let subgroup = subgroup.unwrap_or(lm_default.subgroup);
    let lm = decode::Tuning::for_subgroup(subgroup);
    let local = local::Tuning {
        subgroup,
        ..local::Tuning::for_device(&report).unwrap_or_else(|e| panic!("{e}"))
    };
    let start = Instant::now();
    let timesteps = Sampling::default().timesteps;
    let mut graph =
        patch_graph(&shape, &config, timesteps, &lm, &local).unwrap_or_else(|e| panic!("{e}"));
    let probes = [graph.side.base_out, graph.side.levels].map(|t| graph.side_ids[t.0]);
    for t in probes {
        graph.graph.tensors[t.0].kind = Kind::Output;
    }
    let mut step = PatchStep::new(ctx, kernels, graph, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    step.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let recorded = step.graph_step();
    eprintln!(
        "patch step recorded and loaded on the GPU at subgroups of {subgroup}: {:.1} s, {} \
         dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        recorded.dispatches(),
        recorded.barriers(),
        kernels.len(),
        recorded.arena_bytes() / 1024
    );
    (step, probes, lm)
}

/// What the GPU gives for each patch: the patch `[patch_size][feat_dim]`, the stop logits, the
/// base LM's output before it and the quantizer's levels.
struct Run {
    patches: Vec<Vec<f32>>,
    stop: Vec<[f32; 2]>,
    base: Vec<Vec<f32>>,
    levels: Vec<Vec<f32>>,
}

/// Samples each of `oracle`'s patches on `step`, whose caches hold its prompt but the last
/// position, after the patch before, its `stale`-th before rather than f32's last.
fn sample(step: &mut PatchStep, probes: [TensorId; 2], oracle: &Oracle, stale: usize) -> Run {
    let sampling = Sampling::default();
    let len = oracle.len();
    let patch = oracle.prompt.len() / len;
    let mut out = Run {
        patches: vec![],
        stop: vec![],
        base: vec![],
        levels: vec![],
    };
    let mut elapsed = 0.0;
    for (i, noise) in oracle.noise.iter().enumerate() {
        let before = match i.checked_sub(stale) {
            Some(at) => &oracle.patches[0][at],
            None => &oracle.prompt[(len - 1) * patch..],
        };
        let start = Instant::now();
        let sampled = step
            .run(
                len - 1 + i,
                &transposed(before, 4),
                noise,
                sampling.temperature,
                sampling.cfg_value,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        elapsed += start.elapsed().as_secs_f64();
        let read = |t: TensorId| -> Vec<f32> {
            let bits: Vec<u16> = step.graph_step().read(t).unwrap();
            bits.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
        };
        let levels = read(probes[1]).iter().map(|&l| (l * 9.0).round()).collect();
        out.patches.push(transposed(&sampled.patch, 64));
        out.stop.push(sampled.stop_logits);
        out.base.push(read(probes[0]));
        out.levels.push(levels);
    }
    eprintln!(
        "{} patches at {:.1} ms each from submission to fence (smoke, not a measurement)",
        oracle.noise.len(),
        elapsed * 1e3 / oracle.noise.len() as f64
    );
    out
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
    assert_eq!(got.len(), want.len());
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

/// The GPU's largest error and relative RMS error against f32 over `got`, each over burn's f16's
/// over `burn`.
fn margins(name: &str, got: &[f32], burn: &[f32], want: &[f32]) -> [f64; 2] {
    let (ours, theirs) = (distance(got, want), distance(burn, want));
    eprintln!("{name}: GPU {ours:?}; burn f16 {theirs:?}");
    [
        ours.max_abs / theirs.max_abs,
        ours.relative_rms / theirs.relative_rms,
    ]
}

/// The levels of each position that differ from f32's.
fn flips(levels: &[Vec<f32>], want: &[Vec<f32>]) -> Vec<usize> {
    let differ = |(a, b): (&Vec<f32>, &Vec<f32>)| a.iter().zip(b).filter(|(a, b)| a != b).count();
    levels.iter().zip(want).map(differ).collect()
}

/// The patches where none of the GPU's levels is apart from f32's that a run must have for the
/// GPU to be held to being the closer at each of them.
const CLEAN: usize = 4;

/// How the GPU compares with burn's f16, both against f32.
#[derive(Debug)]
struct Verdict {
    /// The quantizer's input: largest error and relative RMS error over burn's.
    base: [f64; 2],
    /// The levels apart from f32's, the GPU's and burn's.
    flips: [usize; 2],
    /// The stop logits' largest error and relative RMS error over burn's.
    stop: [f64; 2],
    /// The patches' relative RMS error over burn's, over all of them.
    patches: f64,
    /// How many patches of how many are closer to f32's than burn's are at their largest error,
    /// and how many of those where none of the GPU's levels is apart.
    closer: [usize; 2],
    clean: [usize; 2],
    /// Whether the GPU and f32 decide alike whether to stop, wherever f32's margin between
    /// going on and stopping is wider than the GPU's error in the logits.
    decides_alike: bool,
}

impl Verdict {
    /// The criteria it fails. The patches are held to their error over all of them only where
    /// too few are clear of levels apart for the GPU to be held to being the closer at those.
    fn fails(&self) -> Vec<&'static str> {
        let within = |m: f64| m <= 1.25;
        let patches = if self.clean[1] >= CLEAN {
            self.clean[0] == self.clean[1]
        } else {
            within(self.patches)
        };
        [
            ("quantizer input", self.base.into_iter().all(within)),
            ("levels apart", self.flips[0] <= self.flips[1]),
            ("stop logits", self.stop.into_iter().all(within)),
            ("patches", 2 * self.closer[0] >= self.closer[1] && patches),
            ("stop decisions", self.decides_alike),
        ]
        .into_iter()
        .filter(|&(_, holds)| !holds)
        .map(|(name, _)| name)
        .collect()
    }
}

fn judge(oracle: &Oracle, run: &Run) -> Verdict {
    let [want, burn] = &oracle.patches;
    let gpu_flips = flips(&run.levels, &oracle.levels[0]);
    let burn_flips = flips(&oracle.levels[1], &oracle.levels[0]);
    let (mut closer, mut clean) = (0, [0, 0]);
    for i in 0..run.patches.len() {
        let (ours, theirs) = (
            distance(&run.patches[i], &want[i]),
            distance(&burn[i], &want[i]),
        );
        let nearer = ours.max_abs < theirs.max_abs;
        closer += usize::from(nearer);
        if gpu_flips[i] == 0 {
            clean[0] += usize::from(nearer);
            clean[1] += 1;
        }
        eprintln!(
            "patch {i}: levels apart GPU {}, burn f16 {}; GPU {ours:?}, burn f16 {theirs:?}; stop \
             logits f32 {:?}, GPU {:?}, burn f16 {:?}",
            gpu_flips[i], burn_flips[i], oracle.stop[0][i], run.stop[i], oracle.stop[1][i],
        );
    }
    let all = |v: &[Vec<f32>]| v.concat();
    let base = margins(
        "the quantizer's input",
        &all(&run.base),
        &all(&oracle.base[1]),
        &all(&oracle.base[0]),
    );
    let [stop, burn_stop, want_stop] =
        [&run.stop, &oracle.stop[1], &oracle.stop[0]].map(|s| s.concat());
    let stop_margins = margins("stop logits", &stop, &burn_stop, &want_stop);
    let patches = margins("patches", &all(&run.patches), &all(burn), &all(want))[1];
    let error = distance(&stop, &want_stop).max_abs as f32;
    let decides_alike = want_stop.chunks(2).zip(stop.chunks(2)).all(|(w, g)| {
        let margin = w[1] - w[0];
        margin.abs() <= 2.0 * error || (margin > 0.0) == (g[1] > g[0])
    });
    let verdict = Verdict {
        base,
        flips: [gpu_flips.iter().sum(), burn_flips.iter().sum()],
        stop: stop_margins,
        patches,
        closer: [closer, run.patches.len()],
        clean,
        decides_alike,
    };
    eprintln!("{verdict:.2?}");
    verdict
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ORACLE_DIR a directory to keep the oracles in"]
fn keeps_the_oracles() {
    if oracle_dir().is_none() {
        eprintln!("ORACLE_DIR is not set: nothing to keep the oracles in");
        return;
    }
    voice_oracle();
    line_audio(&line_oracle());
}

/// The patch step over `oracle`, the voice's patches alone, its prompt run through the patch
/// step a position at a time.
fn patches_after_a_voice(oracle: &Oracle, subgroup: Option<u32>, stale: usize) -> Verdict {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let (mut step, probes, _) = patch_step(&ctx, &mut kernels, oracle, subgroup);
    let sampling = Sampling::default();
    let patch = oracle.prompt.len() / oracle.len();
    let silence = vec![0.0; patch];
    for (t, cond) in oracle
        .prompt
        .chunks(patch)
        .take(oracle.len() - 1)
        .enumerate()
    {
        let cond = transposed(cond, 4);
        step.run(t, &cond, &silence, sampling.temperature, sampling.cfg_value)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let run = sample(&mut step, probes, oracle, stale);
    assert_no_validation_messages(&ctx);
    judge(oracle, &run)
}

/// The subgroup sizes the device can require: 32, as on NVIDIA, also on the Radeon.
fn subgroups() -> Vec<u32> {
    let report = context().report().clone();
    let can = |&s: &u32| report.can_require_subgroup_size(s);
    SUBGROUPS.into_iter().filter(can).collect()
}

/// A run at a subgroup size, with a defect planted and the criterion it must fail, or none.
struct Checked {
    subgroup: u32,
    defect: Option<(String, &'static str)>,
    verdict: Verdict,
}

/// Asserts that each run passes every criterion, or with a defect fails the defect's.
fn assert_checked(runs: &[Checked]) {
    for run in runs {
        let (subgroup, fails) = (run.subgroup, run.verdict.fails());
        match &run.defect {
            None => assert!(
                fails.is_empty(),
                "wave{subgroup}: {fails:?}: {:?}",
                run.verdict
            ),
            Some((defect, criterion)) => assert!(
                fails.contains(criterion),
                "wave{subgroup}: {defect} passes {criterion}: {:?}",
                run.verdict
            ),
        }
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn patches_sample_as_voxcpm_generates_on_the_cpu() {
    let oracle = voice_oracle();
    let mut runs = vec![];
    for subgroup in subgroups() {
        runs.push(Checked {
            subgroup,
            defect: None,
            verdict: patches_after_a_voice(&oracle, Some(subgroup), 1),
        });
        runs.push(Checked {
            subgroup,
            defect: Some((
                "each patch after the one two before".into(),
                "quantizer input",
            )),
            verdict: patches_after_a_voice(&oracle, Some(subgroup), 2),
        });
    }
    assert_checked(&runs);
}

/// How a prompt is run.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Prompting {
    AsVoxcpm,
    /// The quantizer's output taken at text positions too.
    QuantizedText,
    /// The prompt one position on in the caches.
    Shifted,
}

/// The prompt and the patch step over `oracle`, its prompt run at once.
fn patches_after_a_prompt(oracle: &Oracle, subgroup: Option<u32>, how: Prompting) -> Verdict {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let (mut step, probes, tuning) = patch_step(&ctx, &mut kernels, oracle, subgroup);
    let prompter = Prompter::new(&ctx, &mut kernels, &mut step, 2 * TILE, &tuning, true)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    let prompt = graph::prompt::Prompt {
        tokens: &oracle.tokens,
        patches: &oracle.prompt,
        audio: &oracle.audio,
    };
    let start = Instant::now();
    let (x, audio_x) = prompter
        .embeddings(&mut step, &checkpoint, prompt)
        .unwrap_or_else(|e| panic!("{e}"));
    let embedded = start.elapsed().as_secs_f64() * 1e3;
    let ran = oracle.len() - 1;
    let audio = match how {
        Prompting::QuantizedText => vec![true; ran],
        _ => oracle.audio[..ran].to_vec(),
    };
    let at = usize::from(how == Prompting::Shifted);
    let start = Instant::now();
    prompter
        .run_rows(&mut step, at, &x, &audio_x, &audio)
        .unwrap_or_else(|e| panic!("{e}"));
    let ran = start.elapsed().as_secs_f64() * 1e3;
    let timings = step.timings().unwrap().expect("a timed prompt");
    let ms = |label: &str| -> f64 {
        let of = timings.iter().filter(|(l, _)| l.starts_with(label));
        of.map(|(_, t)| t.nanoseconds / 1e6).sum()
    };
    eprintln!(
        "{} positions of prompt: embedded in {embedded:.1} ms and run in {ran:.1} ms on the host, \
         {:.1} ms on the GPU of which {:.2} between the LMs (smoke, not a measurement)",
        oracle.len(),
        ms(""),
        ms("fsq_layer") + ms("select_rows") + ms("fusion_concat_proj")
    );
    let run = sample(&mut step, probes, oracle, 1);
    assert_no_validation_messages(&ctx);
    judge(oracle, &run)
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn the_prompt_runs_as_voxcpm_runs_it_on_the_cpu() {
    let oracle = line_oracle();
    // The quantizer's output taken at text positions leaves the base LM as it is, and moves the
    // residual LM's state, so the patches; the prompt one position on moves both.
    let defects = [
        (Prompting::QuantizedText, "patches"),
        (Prompting::Shifted, "quantizer input"),
    ];
    let mut runs = vec![];
    for subgroup in subgroups() {
        runs.push(Checked {
            subgroup,
            defect: None,
            verdict: patches_after_a_prompt(&oracle, Some(subgroup), Prompting::AsVoxcpm),
        });
        for (how, criterion) in defects {
            runs.push(Checked {
                subgroup,
                defect: Some((format!("{how:?}"), criterion)),
                verdict: patches_after_a_prompt(&oracle, Some(subgroup), how),
            });
        }
    }
    assert_checked(&runs);
}

/// The audio voxcpm decodes on the CPU, after the voice's last frames and in the service's chunks,
/// of `oracle`'s patches in f32 and of burn's f16's, the decoder in f32 for both.
fn line_audio(oracle: &Oracle) -> [Vec<f32>; 2] {
    let (name, voice) = voice();
    kept(&format!("line-audio-{name}-{SEED}"), || {
        let decoder = cpu_decoder();
        let frames = voice.frames.as_flattened();
        oracle
            .patches
            .each_ref()
            .map(|patches| cpu_speech(&decoder, frames, patches))
    })
}

fn cpu_decoder() -> audiovae::Decoder {
    let weights = Weights::open(checkpoint_dir().join("audiovae.pth")).unwrap();
    audiovae::Decoder::load(&config().audio_vae(), &weights, FloatDType::F32, &device()).unwrap()
}

/// `patches`' audio as the service decodes it on the CPU, after `voice`'s last frames.
fn cpu_speech(decoder: &audiovae::Decoder, voice: &[f32], patches: &[Vec<f32>]) -> Vec<f32> {
    let chunk = Params::default().decode_chunk_frames;
    let mut speech = SpeechDecoder::new(decoder, voice, 64, chunk);
    let shape = [patches.len(), 4, 64];
    speech.push(Tensor::from_data(
        TensorData::new(patches.concat(), shape),
        &device(),
    ))
}

/// The generator for the checkpoint, as the service would run it, with caches for `capacity`
/// positions, its kernels at `subgroup`.
fn generator(ctx: &Context, kernels: &mut Kernels, capacity: usize, subgroup: u32) -> Generator {
    let weights = Weights::open(checkpoint_dir().join("audiovae.pth")).unwrap();
    let settings = Settings {
        capacity,
        max_rows: 2 * TILE,
        timesteps: Sampling::default().timesteps,
        chunk_frames: Params::default().decode_chunk_frames.get(),
        timed: false,
        prompt_cache_bytes: 0,
        subgroup: Some(subgroup),
    };
    let start = Instant::now();
    let parameter = |name: &str| {
        let name = format!("decoder.{name}");
        weights
            .floats(&name)
            .map_err(|e| graph::Error::Checkpoint(e.to_string()))
    };
    let generator = Generator::new(ctx, kernels, &checkpoint_dir(), &settings, parameter)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "generator recorded and loaded at subgroups of {subgroup}: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    generator
}

/// A request's patches, `[patch_size][feat_dim]` each, and its audio.
struct Speech {
    patches: Vec<Vec<f32>>,
    audio: Vec<f32>,
}

/// `oracle`'s line spoken by `generator` from its noise: each patch sampled after f32's
/// `stale`-th before, as teacher forcing does, or after its own when `stale` is `None`; the
/// frames waiting decoded at once after each patch whose number `early` has, counting from 1.
fn speak(
    generator: &mut Generator,
    oracle: &Oracle,
    stale: Option<usize>,
    early: &[usize],
) -> Speech {
    let (_, voice) = voice();
    let sampling = Sampling::default();
    let prompt = graph::prompt::Prompt {
        tokens: &oracle.tokens,
        patches: &oracle.prompt,
        audio: &oracle.audio,
    };
    let start = Instant::now();
    generator
        .start(
            prompt,
            voice.frames.as_flattened(),
            Params::default().max_patches,
            sampling.temperature,
            sampling.cfg_value,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let (mut patches, mut audio) = (vec![], vec![]);
    for (i, noise) in oracle.noise.iter().enumerate() {
        if let Some(before) = stale.and_then(|stale| i.checked_sub(stale)) {
            generator
                .sample_after(&oracle.patches[0][before])
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let Some(patch) = generator
            .next_patch(noise)
            .unwrap_or_else(|e| panic!("{e}"))
        else {
            break;
        };
        if i == 0 {
            eprintln!(
                "the first patch {:.1} ms after the request (smoke, not a measurement)",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
        patches.push(patch.frames);
        if early.contains(&patches.len()) {
            generator.decode_waiting().unwrap_or_else(|e| panic!("{e}"));
        }
        audio.extend(generator.take_audio());
        if patch.last {
            break;
        }
    }
    audio.extend(generator.finish().unwrap_or_else(|e| panic!("{e}")));
    eprintln!(
        "{} patches and {} samples in {:.2} s (smoke, not a measurement)",
        patches.len(),
        audio.len(),
        start.elapsed().as_secs_f64()
    );
    Speech { patches, audio }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn speech_comes_out_as_voxcpm_speaks_it_on_the_cpu() {
    let oracle = line_oracle();
    let [want, burn] = line_audio(&oracle);
    let patches = oracle.noise.len();
    let samples = patches * 4 * 1920;
    assert_eq!(want.len(), samples);
    let decoder = cpu_decoder();
    let (_, voice) = voice();
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let capacity = (oracle.len() + Params::default().max_patches).next_multiple_of(32);
    let mut failed = vec![];
    for subgroup in subgroups() {
        let mut generator = generator(&ctx, &mut kernels, capacity, subgroup);
        let fails = speech_at(&mut generator, &oracle, &decoder, &voice, [&want, &burn]);
        assert_no_validation_messages(&ctx);
        if let Err(fails) = fails {
            failed.push(format!("wave{subgroup}: {fails}"));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The speech test on `generator`, the failures it finds, if any.
fn speech_at(
    generator: &mut Generator,
    oracle: &Oracle,
    decoder: &audiovae::Decoder,
    voice: &Latent,
    [want, burn]: [&[f32]; 2],
) -> Result<(), String> {
    let patches = oracle.noise.len();
    let samples = want.len();
    // Each patch after f32's patch before: it stops where f32 does, and its audio is as near
    // f32's as burn's f16's patches' is, decoded by the same decoder.
    let forced = speak(generator, oracle, Some(1), &[]);
    // Defect: each patch after f32's two before.
    let stale = speak(generator, oracle, Some(2), &[]);
    // Each patch after its own, as the service generates.
    let free = speak(generator, oracle, None, &[]);
    // Decoding the frames waiting after the patches a stream decodes early at.
    let early = speak(generator, oracle, None, &[1, 2, 4, 8]);

    let audio_margins = |name: &str, speech: &Speech| match speech.audio.len() == samples {
        true => margins(name, &speech.audio, &burn, &want),
        false => [f64::INFINITY; 2],
    };
    let margin = audio_margins("audio", &forced);
    let stale_margin = audio_margins("audio after stale patches", &stale);

    // Run free, it stops where f32 does too, and its audio is what the service's decoder makes
    // of its patches on the CPU, to f32 rounding, which f16 storage does not reach.
    let cpu = cpu_speech(decoder, voice.frames.as_flattened(), &free.patches);
    let d = distance(&free.audio, &cpu);
    eprintln!("run free, against the CPU's decoder over its patches: {d:?}");
    // The audio VAE carries its state from chunk to chunk, so audio decoded early in chunks
    // of fewer frames is the same audio.
    if early.audio.len() == free.audio.len() {
        let early = distance(&early.audio, &free.audio);
        eprintln!("decoded early, against run free: {early:?}");
    }
    // Defect: the CPU's decoder not primed with the voice's last frames.
    let unprimed = cpu_speech(decoder, &[], &free.patches);
    let unprimed = distance(&free.audio, &unprimed);
    eprintln!("run free, against the CPU's decoder not primed: {unprimed:?}");

    // Every check is made before any fails, so that a run reports the free run's too.
    let fails: Vec<&str> = [
        ("stops", forced.patches.len() == patches),
        ("audio", margin[1] <= 1.25),
        ("audio after stale patches fails", stale_margin[1] > 1.25),
        ("run free, stops", free.patches.len() == patches),
        ("run free, audio", d.max_abs <= 1e-4),
        ("early, same patches", early.patches == free.patches),
        ("early, same audio", early.audio == free.audio),
        ("audio not primed fails", unprimed.max_abs > 1e-4),
    ]
    .into_iter()
    .filter(|&(_, holds)| !holds)
    .map(|(name, _)| name)
    .collect();
    if fails.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{fails:?}: {} patches of {patches}, {} run free; audio {margin:.2?}, after stale \
         patches {stale_margin:.2?}; run free {d:?}, not primed {unprimed:?}",
        forced.patches.len(),
        free.patches.len()
    ))
}
