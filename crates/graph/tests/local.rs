//! The patch sampler and the local encoder on the GPU against voxcpm's on burn's CPU backend,
//! on VoxCPM2's real weights: each case a patch upstream generated, as the patch before, and the
//! noise it drew for the next. Built with the `device` feature, and run with `VOXCPM2_DIR`
//! naming the checkpoint and `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests`
//! sets it.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use burn::tensor::{FloatDType, Tensor, TensorData};
use graph::{
    checkpoint::Checkpoint,
    kernels::Kernels,
    local::{PatchConfig, Tuning, dit::Sampler, encoder::Encoder},
};
use half::f16;
use testkit::{Fixture, checkpoint_dir, device};
use voxcpm::{Cfm, Config, LocalEncoder, Sampling};
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

fn config_json() -> String {
    fs::read_to_string(checkpoint_dir().join("config.json")).unwrap()
}

fn weights() -> std::path::PathBuf {
    checkpoint_dir().join("model.safetensors")
}

/// What one patch is sampled from: the conditions, the patch before and the noise, each laid
/// out as the sampler takes it.
struct Case {
    mu: Vec<f32>,
    cond: Vec<f32>,
    noise: Vec<f32>,
}

/// A uniform sequence in [-scale, scale) from a fixed xorshift sequence.
fn uniform(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Patches upstream generated on the real weights, `[patch_size][feat_dim]` each, and the noise
/// it drew for each, `[feat_dim][patch_size]`.
fn golden() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_checkpoint.safetensors"
    );
    let fixture = Fixture::open(path);
    let rows = |name: &str| -> Vec<Vec<f32>> {
        let tensor = fixture.tensor::<3>(name, &device());
        let [n, ..] = tensor.dims();
        let values: Vec<f32> = tensor.into_data().iter::<f32>().collect();
        values
            .chunks(values.len() / n)
            .map(<[f32]>::to_vec)
            .collect()
    };
    (rows("output.patches"), rows("input.noise"))
}

fn transposed(patch: &[f32], rows: usize) -> Vec<f32> {
    let columns = patch.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| patch[r * columns + c]))
        .collect()
}

/// Two patches of the golden after the ones before them, with conditions of about the scale of
/// a hidden state after a norm.
fn cases(config: &PatchConfig) -> Vec<Case> {
    let (patches, noise) = golden();
    let hidden = config.dit_config.stack.hidden_dim;
    [(3, 1), (12, 2)]
        .into_iter()
        .map(|(at, seed)| Case {
            mu: uniform(2 * hidden, seed, 1.7),
            cond: transposed(&patches[at - 1], config.patch_size),
            noise: noise[at].clone(),
        })
        .collect()
}

fn f16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
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

/// The GPU's largest error and relative RMS error against f32, over burn's f16 own.
fn margins_of(got: &[f32], want: &[f32], burn_f16: &[f32]) -> [f64; 2] {
    let (ours, burn) = (distance(got, want), distance(burn_f16, want));
    eprintln!("GPU {ours:?}; burn f16 {burn:?}");
    [
        ours.max_abs / burn.max_abs,
        ours.relative_rms / burn.relative_rms,
    ]
}

/// Whether the GPU stays as close to f32 as f16 arithmetic on the CPU does, give or take a
/// quarter: which of two runs rounds a few large values worse is luck.
fn within(margins: [f64; 2]) -> bool {
    margins.iter().all(|&m| m <= 1.25)
}

/// What voxcpm's sampler gives for each case in `timesteps` steps, its DiT in `dtype`, the
/// patches `[feat_dim][patch_size]` flattened one after another.
fn oracle(cases: &[Case], timesteps: usize, dtype: FloatDType) -> Vec<f32> {
    let config = Config::from_json(&config_json()).unwrap();
    let (feat_dim, patch_size) = (config.feat_dim, config.patch_size);
    let cpu = device();
    let start = Instant::now();
    let cfm = Cfm::load(&config, &weights(), dtype, &cpu).unwrap_or_else(|e| panic!("{e}"));
    let sampling = Sampling {
        timesteps,
        ..Sampling::default()
    };
    let mut out = Vec::new();
    for case in cases {
        let tensor = |values: &[f32], shape: [usize; 3]| {
            Tensor::<3>::from_data(TensorData::new(values.to_vec(), shape), &cpu)
        };
        let mu = Tensor::<2>::from_data(TensorData::new(case.mu.clone(), [1, case.mu.len()]), &cpu);
        let patch = cfm.sample(
            mu,
            tensor(&case.cond, [1, feat_dim, patch_size]),
            tensor(&case.noise, [1, feat_dim, patch_size]),
            &sampling,
        );
        out.extend(patch.into_data().iter::<f32>());
    }
    eprintln!(
        "{} cases in {timesteps} steps on the CPU, {dtype:?}: {:.1} s",
        cases.len(),
        start.elapsed().as_secs_f64()
    );
    out
}

fn patch_config() -> PatchConfig {
    serde_json::from_str(&config_json()).unwrap()
}

/// The GPU's patches for the same cases, the kernels cut as `tuning` says, the conditions
/// rounded to f16 as the GPU takes them.
fn gpu(ctx: &Context, cases: &[Case], timesteps: usize, tuning: &Tuning) -> Vec<f32> {
    let config = patch_config();
    let mut kernels = Kernels::from_env(ctx).unwrap_or_else(|e| panic!("{e}"));
    let start = Instant::now();
    let mut sampler = Sampler::new(ctx, &mut kernels, &config, timesteps, tuning, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(weights()).unwrap_or_else(|e| panic!("{e}"));
    sampler.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "sampler in {timesteps} steps recorded and loaded on the GPU at {tuning:?}: {:.1} s, {} \
         dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        sampler.graph_step().dispatches(),
        sampler.graph_step().barriers(),
        kernels.len(),
        sampler.graph_step().arena_bytes() / 1024
    );
    let sampling = Sampling::default();
    let mut out = Vec::new();
    for case in cases {
        let patch = sampler
            .sample(
                &f16_bits(&case.mu),
                &case.cond,
                &case.noise,
                sampling.temperature,
                sampling.cfg_value,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        out.extend(patch);
    }
    out
}

/// The mu that burn's DiT sees, rounded to f16 as the GPU's does: the oracles take it so, the
/// difference of their inputs being none of the GPU's error.
fn rounded(cases: Vec<Case>) -> Vec<Case> {
    cases
        .into_iter()
        .map(|case| Case {
            mu: case.mu.iter().map(|&v| f16::from_f32(v).to_f32()).collect(),
            ..case
        })
        .collect()
}

/// The tunings each kernel is checked at, at either subgroup size: one row or pair to a cluster
/// and more, in workgroups of 128 to 1024, one to each block of rows or fewer, which then take
/// several in turn, the tokens staged in chunks of 1024 columns and fewer, and the output and
/// down projections cut along their columns or not.
fn tunings(ctx: &Context) -> Vec<Tuning> {
    let report = ctx.report();
    [64, 32]
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
        .flat_map(|s| {
            let base = Tuning::for_subgroup(s);
            let cut = |workgroup, pairs, o_rows, down_rows, chunk, groups| Tuning {
                workgroup,
                qkv_pairs: pairs,
                gate_up_pairs: pairs,
                o_rows,
                down_rows,
                chunk,
                groups,
                ..base
            };
            [
                base,
                Tuning {
                    subgroup: s,
                    ..Tuning::for_device(report).unwrap()
                },
                cut(128, 2, 2, 4, 512, 0),
                cut(256, 1, 1, 1, 256, 0),
                cut(256, 2, 2, 2, 256, 24),
                cut(128, 1, 1, 1, 256, 40),
                cut(256, 1, 1, 1, 1024, 40),
                cut(256, 1, 1, 1, 512, 80),
                cut(512, 1, 1, 1, 1024, 40),
                cut(1024, 1, 1, 1, 1024, 0),
                Tuning {
                    split: true,
                    ..cut(256, 1, 1, 1, 1024, 40)
                },
                Tuning {
                    split: true,
                    ..cut(256, 1, 1, 1, 1024, 24)
                },
                Tuning {
                    split: true,
                    ..cut(256, 1, 1, 1, 1024, 80)
                },
                Tuning {
                    split: true,
                    ..cut(128, 1, 2, 2, 512, 0)
                },
            ]
        })
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_euler_step_samples_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    // Two steps, the first of which stays where it starts: one run of the DiT.
    let cases = rounded(cases(&config));
    let want = oracle(&cases, 2, FloatDType::F32);
    let burn_f16 = oracle(&cases, 2, FloatDType::F16);
    let ctx = context();
    for tuning in tunings(&ctx) {
        let got = gpu(&ctx, &cases, 2, &tuning);
        let margins = margins_of(&got, &want, &burn_f16);
        assert!(within(margins), "{tuning:?}: {margins:.2?}");
    }
    // Defect: the conditions left out, so the unconditional velocity alone.
    let unguided: Vec<Case> = cases
        .iter()
        .map(|case| Case {
            mu: vec![0.0; case.mu.len()],
            cond: case.cond.clone(),
            noise: case.noise.clone(),
        })
        .collect();
    let got = gpu(&ctx, &unguided, 2, &Tuning::for_subgroup(64));
    let margins = margins_of(&got, &want, &burn_f16);
    assert!(!within(margins), "without mu: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn ten_steps_sample_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    let cases = rounded(cases(&config));
    let timesteps = Sampling::default().timesteps;
    let want = oracle(&cases, timesteps, FloatDType::F32);
    let burn_f16 = oracle(&cases, timesteps, FloatDType::F16);
    let ctx = context();
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let got = gpu(&ctx, &cases, timesteps, &tuning);
    let margins = margins_of(&got, &want, &burn_f16);
    assert!(within(margins), "{margins:.2?}");
    // Defect: a step short, the last step's velocity left out.
    let short = gpu(&ctx, &cases, timesteps - 1, &tuning);
    let margins = margins_of(&short, &want, &burn_f16);
    assert!(!within(margins), "a step short: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}

/// What voxcpm's local encoder gives for each patch `[patch_size][feat_dim]`, in `dtype`.
fn encoder_oracle(patches: &[Vec<f32>], dtype: FloatDType) -> Vec<f32> {
    let config = Config::from_json(&config_json()).unwrap();
    let cpu = device();
    let encoder =
        LocalEncoder::load(&config, &weights(), dtype, &cpu).unwrap_or_else(|e| panic!("{e}"));
    let n = patches.len();
    let values: Vec<f32> = patches.concat();
    let shape = [1, n, config.patch_size, config.feat_dim];
    let patches = Tensor::<4>::from_data(TensorData::new(values, shape), &cpu);
    let out = encoder.forward(patches).cast(FloatDType::F32);
    out.into_data().iter::<f32>().collect()
}

/// The first `count` patches of each of a few voices' latents, `[patch_size][feat_dim]` each,
/// from the voices directory `VOICES_DIR` names.
fn voice_patches(config: &PatchConfig, count: usize) -> Vec<Vec<f32>> {
    let dir = std::env::var_os("VOICES_DIR").expect("VOICES_DIR names no voices directory");
    let len = config.patch_size * config.feat_dim;
    ["abaddon", "femaleoldkindly", "kunkka"]
        .into_iter()
        .flat_map(|voice| {
            let path = std::path::Path::new(&dir).join(format!("latents/{voice}.safetensors"));
            let latents = Fixture::open(path).tensor::<2>("latents", &device());
            let values: Vec<f32> = latents.into_data().iter::<f32>().collect();
            let patches: Vec<Vec<f32>> = values.chunks_exact(len).map(<[f32]>::to_vec).collect();
            patches.into_iter().take(count)
        })
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn the_encoder_embeds_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    // Over many patches: the errors gather in a few outlier channels, and at any one patch
    // which of two runs rounds those worse is luck.
    let mut patches = golden().0;
    patches.extend(voice_patches(&config, 40));
    let want = encoder_oracle(&patches, FloatDType::F32);
    let burn_f16 = encoder_oracle(&patches, FloatDType::F16);
    let ctx = context();
    let checkpoint = Checkpoint::open(weights()).unwrap_or_else(|e| panic!("{e}"));
    for tuning in tunings(&ctx) {
        let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
        let mut encoder = Encoder::new(&ctx, &mut kernels, &config, &tuning, false)
            .unwrap_or_else(|e| panic!("{e}"));
        encoder.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
        let mut got = Vec::new();
        for patch in &patches {
            let out = encoder
                .encode(&transposed(patch, config.patch_size))
                .unwrap_or_else(|e| panic!("{e}"));
            got.extend(out.iter().map(|&h| f16::from_bits(h).to_f32()));
        }
        let margins = margins_of(&got, &want, &burn_f16);
        assert!(within(margins), "{tuning:?}: {margins:.2?}");
    }
    // Defect: the patches' frames in another order, which only RoPE tells apart.
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let tuning = Tuning::for_subgroup(64);
    let mut encoder =
        Encoder::new(&ctx, &mut kernels, &config, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
    encoder.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let mut got = Vec::new();
    for patch in &patches {
        let mut frames: Vec<&[f32]> = patch.chunks(config.feat_dim).collect();
        frames.reverse();
        let out = encoder
            .encode(&transposed(&frames.concat(), config.patch_size))
            .unwrap_or_else(|e| panic!("{e}"));
        got.extend(out.iter().map(|&h| f16::from_bits(h).to_f32()));
    }
    let margins = margins_of(&got, &want, &burn_f16);
    assert!(!within(margins), "frames reversed: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}
