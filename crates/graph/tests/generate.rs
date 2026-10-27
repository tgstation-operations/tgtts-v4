//! The patch step on the GPU against voxcpm's generation on burn's CPU backend, on VoxCPM2's real
//! weights. The model is prompted with a voice's patches alone and generates patches from seeded
//! noise; the GPU runs the prompt's patches through the patch step a position at a time, then
//! samples each generated patch from the same noise after the CPU's patch before, as voxcpm's
//! f16 is fed the f32 run's, so that all three follow one generation: the sampler conditioned on
//! what the LM side computes, where its own tests make the condition up. Built with the `device`
//! feature, and run with `VOXCPM2_DIR` naming the checkpoint, `VOICES_DIR` a voices directory and
//! `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests` sets the last.
//!
//! The scalar quantizer rounds the base LM's output to levels, and a value near a boundary
//! between two lands on either by luck of rounding: a level apart from f32's is a discrete event,
//! which the sampler amplifies into the patch, not an error of the rounding's size. On the real
//! weights some levels are apart at every position, for burn's f16 as for the GPU, so the GPU is
//! held to burn's f16 on the quantizer's input, a continuous value, by its largest and its RMS
//! error; on how many levels are apart; on the stop logits by their largest and RMS error; and on
//! the patches by their RMS error over all of them and by being the closer at half of them at
//! least, not by their largest error, which is that of whichever level apart the sampler
//! amplified most. Teacher forcing keeps a level's effect on a patch from reaching the next
//! through the patch.
//!
//! The CPU's part is kept under the directory `ORACLE_DIR` names, when it is set, and read from
//! there by later runs: `keeps_the_oracle` runs it alone, touching no GPU.
#![cfg(feature = "device")]

use std::{fs, path::PathBuf, time::Instant};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    store::{
        FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    },
    tensor::{FloatDType, Tensor, TensorData},
};
use graph::{
    checkpoint::Checkpoint,
    decode::{self, SUBGROUPS},
    graph::Kind,
    kernels::Kernels,
    lm_side::{self, LmSideShape},
    local::{self, PatchConfig},
    patch::{PatchStep, patch_graph},
};
use half::f16;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use testkit::{Fixture, checkpoint_dir, device};
use voices::Latent;
use voxcpm::{Config, Dtypes, LocalEncoder, Model, Noise, Prompt, Sampling, SeededNoise};
use vulkan::Context;

/// Patches generated after the prompt.
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

/// The voice voxcpm's golden was made with, and its latent frames.
fn voice() -> (String, Vec<f32>) {
    let golden = Fixture::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_checkpoint.safetensors"
    ));
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let dir = std::path::absolute(dir).unwrap();
    let name = golden.metadata()["voice"].clone();
    let latent = Latent::read(&voices::latent_path(&dir, &name)).unwrap_or_else(|e| panic!("{e}"));
    (name, latent.frames.as_flattened().to_vec())
}

/// `compute`'s result, kept as JSON under `ORACLE_DIR` when it is set, by `name`: read from
/// there when a run before left it. A change to what computes it wants the file removed.
fn kept<T: Serialize + DeserializeOwned>(name: &str, compute: impl FnOnce() -> T) -> T {
    let Some(dir) = std::env::var_os("ORACLE_DIR").filter(|dir| !dir.is_empty()) else {
        return compute();
    };
    let path = PathBuf::from(dir).join(format!("{name}.json"));
    if let Ok(json) = fs::read(&path) {
        eprintln!("the oracle from {}", path.display());
        return serde_json::from_slice(&json).unwrap_or_else(|e| panic!("{e}"));
    }
    let value = compute();
    fs::write(&path, serde_json::to_vec(&value).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    value
}

/// What voxcpm's model gives on the CPU, in f32 and in burn's f16, the second each step after the
/// first's patch before.
#[derive(Serialize, Deserialize)]
struct Oracle {
    /// The voice's latent frames, the prompt's patches.
    voice: Vec<f32>,
    /// The noise each generated patch is sampled from, `[feat_dim][patch_size]`.
    noise: Vec<Vec<f32>>,
    /// Each generated patch, `[patch_size][feat_dim]`.
    patches: [Vec<Vec<f32>>; 2],
    /// The stop logits each patch was sampled beside.
    stop: [Vec<[f32; 2]>; 2],
    /// The base LM's output at the position each patch is sampled after, the scalar quantizer's
    /// input.
    base: [Vec<Vec<f32>>; 2],
    /// The quantizer's levels there, integers.
    levels: [Vec<Vec<f32>>; 2],
}

fn floats<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32).into_data().iter::<f32>().collect()
}

fn oracle() -> Oracle {
    let (name, voice) = voice();
    kept(&format!("patch-{name}-{GENERATED}-{SEED}"), || {
        compute_oracle(voice)
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

fn compute_oracle(voice: Vec<f32>) -> Oracle {
    let config = Config::from_json(&config_json()).unwrap();
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let (patch_size, feat_dim) = (config.patch_size, config.feat_dim);
    let len = voice.len() / (patch_size * feat_dim);
    let prompt = Prompt {
        tokens: vec![0; len],
        patches: voice.clone(),
        audio: vec![true; len],
    };
    let sampling = Sampling::default();
    let tensor = |values: &[f32], shape: [usize; 3]| {
        Tensor::<3>::from_data(TensorData::new(values.to_vec(), shape), &cpu)
    };
    let start = Instant::now();
    let mut draws = SeededNoise::new(SEED);
    let noise: Vec<Vec<f32>> = (0..GENERATED)
        .map(|_| draws.next(feat_dim * patch_size))
        .collect();
    let mut patches = [vec![], vec![]];
    let mut stop = [vec![], vec![]];
    let mut base = [vec![], vec![]];
    let mut levels = [vec![], vec![]];
    for (run, dtype) in [FloatDType::F32, FloatDType::F16].into_iter().enumerate() {
        let model =
            Model::load(&config, &weights, &dtypes(dtype), &cpu).unwrap_or_else(|e| panic!("{e}"));
        let mut generation = model.start(&prompt, GENERATED).unwrap();
        for (i, noise) in noise.iter().enumerate() {
            let step = generation.sample(tensor(noise, [1, feat_dim, patch_size]), &sampling);
            let logits = floats(step.stop_logits);
            patches[run].push(floats(step.patch));
            stop[run].push([logits[0], logits[1]]);
            let before = &patches[0][i];
            generation.advance(tensor(before, [1, patch_size, feat_dim]));
        }
        drop(generation);
        drop(model);
        let fed = &patches[0][..GENERATED - 1];
        [base[run], levels[run]] = quantizer(&config, &voice, fed, dtype);
    }
    eprintln!(
        "{len} patches of prompt and {GENERATED} generated on the CPU, f32 and f16: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    Oracle {
        voice,
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
/// the prompt's last and then `fed`'s, as voxcpm's model computes them in `dtype`, which keeps
/// them to itself: the local encoder and enc_to_lm_proj over the prompt's patches at once and
/// then over each of `fed`, the base LM over the prompt at once and then a position at a time,
/// and the quantizer over its outputs as they come.
fn quantizer(
    config: &Config,
    voice: &[f32],
    fed: &[Vec<f32>],
    dtype: FloatDType,
) -> [Vec<Vec<f32>>; 2] {
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let (patch_size, feat_dim) = (config.patch_size, config.feat_dim);
    let lm = &config.lm_config;
    let hidden = lm.hidden_size;
    let len = voice.len() / (patch_size * feat_dim);
    let encoder = LocalEncoder::load(config, &weights, dtype, &cpu).unwrap();
    let base = minicpm4::Model::load(lm, &weights, "base_lm.", dtype, &cpu).unwrap();
    let heads = Heads {
        enc_to_lm_proj: LinearConfig::new(config.encoder_config.hidden_dim, hidden).init(&cpu),
        fsq_layer: Quantizer {
            in_proj: LinearConfig::new(hidden, config.scalar_quantization_latent_dim).init(&cpu),
        },
    };
    let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype.into()));
    let mut store = SafetensorsStore::from_file(&weights).with_from_adapter(adapter);
    let mut heads = heads;
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
    let mut cache = base.cache(1, len + GENERATED);
    let mut outputs = vec![base.forward_cached(embed(voice.to_vec(), len), &mut cache)];
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
    assert_eq!(found[0][0].len(), hidden);
    found
}

/// `[rows][columns]` as `[columns][rows]`.
fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
    let columns = values.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect()
}

/// How a run differs from the oracle's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Defect {
    None,
    /// Each generated patch sampled after the patch two before rather than the one before.
    Stale,
}

/// What the GPU gives for each generated patch: the patch `[patch_size][feat_dim]`, the stop
/// logits, the base LM's output before it and the quantizer's levels.
struct Run {
    patches: Vec<Vec<f32>>,
    stop: Vec<[f32; 2]>,
    base: Vec<Vec<f32>>,
    levels: Vec<Vec<f32>>,
}

/// The GPU's run, the LM side's kernels at `subgroup` and the local stacks' at the device's
/// tuning at that subgroup size (the device's default when `None`). The base LM's output and
/// the levels are read where the step leaves them, made outputs for it.
fn gpu(oracle: &Oracle, subgroup: Option<u32>, defect: Defect) -> Run {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let json = config_json();
    let side_config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let (patch_size, feat_dim) = (config.patch_size, config.feat_dim);
    let patch = patch_size * feat_dim;
    let len = oracle.voice.len() / patch;
    let shape = LmSideShape::new(&side_config, (len + GENERATED).next_multiple_of(32)).unwrap();
    let report = ctx.report().clone();
    let lm_default = decode::Tuning::for_device(&report).unwrap_or_else(|e| panic!("{e}"));
    let subgroup = subgroup.unwrap_or(lm_default.subgroup);
    let lm = decode::Tuning::for_subgroup(subgroup);
    let local = local::Tuning {
        subgroup,
        ..local::Tuning::for_device(&report).unwrap_or_else(|e| panic!("{e}"))
    };
    let sampling = Sampling::default();
    let start = Instant::now();
    let mut graph = patch_graph(&shape, &config, sampling.timesteps, &lm, &local)
        .unwrap_or_else(|e| panic!("{e}"));
    let probes = [graph.side.base_out, graph.side.levels].map(|t| graph.side_ids[t.0]);
    for t in probes {
        graph.graph.tensors[t.0].kind = Kind::Output;
    }
    let mut step =
        PatchStep::new(&ctx, &mut kernels, graph, false).unwrap_or_else(|e| panic!("{e}"));
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
    let voice = |t: usize| &oracle.voice[t * patch..(t + 1) * patch];
    let silence = vec![0.0; patch];
    let run = |step: &mut PatchStep, position: usize, cond: &[f32], noise: &[f32]| {
        let cond = transposed(cond, patch_size);
        step.run(
            position,
            &cond,
            noise,
            sampling.temperature,
            sampling.cfg_value,
        )
        .unwrap_or_else(|e| panic!("{e}"))
    };
    for t in 0..len - 1 {
        run(&mut step, t, voice(t), &silence);
    }
    let mut out = Run {
        patches: vec![],
        stop: vec![],
        base: vec![],
        levels: vec![],
    };
    let mut elapsed = 0.0;
    for (i, noise) in oracle.noise.iter().enumerate() {
        let before = match (defect, i) {
            (Defect::Stale, 2..) => &oracle.patches[0][i - 2],
            (_, 0) => voice(len - 1),
            _ => &oracle.patches[0][i - 1],
        };
        let start = Instant::now();
        let sampled = run(&mut step, len - 1 + i, before, noise);
        elapsed += start.elapsed().as_secs_f64();
        let read = |t: graph::graph::TensorId| -> Vec<f32> {
            let bits: Vec<u16> = step.graph_step().read(t).unwrap();
            bits.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
        };
        let levels = read(probes[1]).iter().map(|&l| (l * 9.0).round()).collect();
        out.patches.push(transposed(&sampled.patch, feat_dim));
        out.stop.push(sampled.stop_logits);
        out.base.push(read(probes[0]));
        out.levels.push(levels);
    }
    eprintln!(
        "{GENERATED} patches at {:.1} ms each from submission to fence (smoke, not a \
         measurement)",
        elapsed * 1e3 / GENERATED as f64
    );
    assert_no_validation_messages(&ctx);
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

/// How the GPU compares with burn's f16, both against f32.
#[derive(Debug)]
struct Verdict {
    /// The quantizer's input: largest error and relative RMS error over burn's.
    base: [f64; 2],
    /// The levels apart from f32's, the GPU's and burn's.
    flips: [usize; 2],
    /// The stop logits' largest error and relative RMS error over burn's.
    stop: [f64; 2],
    /// The patches' relative RMS error over burn's, over all of them, and how many of them are
    /// closer to f32's than burn's are at their largest error.
    patches: f64,
    closer: usize,
    /// Whether the GPU and f32 decide alike whether to stop, wherever f32's margin between
    /// going on and stopping is wider than the GPU's error in the logits.
    decides_alike: bool,
}

impl Verdict {
    fn passes(&self) -> bool {
        let within = |m: f64| m <= 1.25;
        self.base.into_iter().all(within)
            && self.flips[0] <= self.flips[1]
            && self.stop.into_iter().all(within)
            && within(self.patches)
            && 2 * self.closer >= GENERATED
            && self.decides_alike
    }
}

fn judge(oracle: &Oracle, run: &Run) -> Verdict {
    let [want, burn] = &oracle.patches;
    let gpu_flips = flips(&run.levels, &oracle.levels[0]);
    let burn_flips = flips(&oracle.levels[1], &oracle.levels[0]);
    let mut closer = 0;
    for i in 0..GENERATED {
        let (ours, theirs) = (
            distance(&run.patches[i], &want[i]),
            distance(&burn[i], &want[i]),
        );
        closer += usize::from(ours.max_abs < theirs.max_abs);
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
        closer,
        decides_alike,
    };
    eprintln!("{verdict:.2?}");
    verdict
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ORACLE_DIR a directory to keep the oracle in"]
fn keeps_the_oracle() {
    if std::env::var_os("ORACLE_DIR").is_none_or(|dir| dir.is_empty()) {
        eprintln!("ORACLE_DIR is not set: nothing to keep the oracle in");
        return;
    }
    oracle();
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn patches_sample_as_voxcpm_generates_on_the_cpu() {
    let oracle = oracle();
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    let verdicts: Vec<_> = SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
        .map(|subgroup| {
            (
                subgroup,
                judge(&oracle, &gpu(&oracle, Some(subgroup), Defect::None)),
            )
        })
        .collect();
    let stale = judge(&oracle, &gpu(&oracle, None, Defect::Stale));
    for (subgroup, verdict) in verdicts {
        assert!(verdict.passes(), "wave{subgroup}: {verdict:?}");
    }
    assert!(
        !stale.passes(),
        "patches after a stale patch pass: {stale:?}"
    );
}
