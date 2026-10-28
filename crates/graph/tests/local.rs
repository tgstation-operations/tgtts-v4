//! The patch sampler and the local encoder on the GPU against voxcpm's on burn's CPU backend,
//! on VoxCPM2's real weights: each case a patch upstream generated, as the patch before, and the
//! noise it drew for the next, sampled at a temperature and a guidance scale of its own. Built
//! with the `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint and `GLSL_KERNELS`
//! the compiled kernels, as `nix run .#device-tests` sets it.
//!
//! Each case is run alone, and with others as rows of one run, where it must come out bit for
//! bit as it does alone: what a row's sums come to depends on the tuning alone, never on what
//! else is in a block of tokens (`local`'s documentation), so that a request's audio does not
//! depend on the others served beside it. Alone, each is held to burn's f16.
//!
//! The CPU's part is kept under the directory `ORACLE_DIR` names, when it is set, and read from
//! there by later runs: `keeps_the_oracles` runs it alone, touching no GPU.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use burn::tensor::{FloatDType, Tensor, TensorData};
use graph::{
    checkpoint::Checkpoint,
    graph::{Graph, Kind, TensorId},
    kernels::Kernels,
    lm_side::splice,
    local::{
        PatchConfig, Tuning,
        dit::{Sampler, load_sampler, sampler_graph},
        encoder::{Encoder, encoder_graph, load_encoder},
    },
    step::Step,
};
use half::f16;
use testkit::{Fixture, checkpoint_dir, device, kept, oracle_dir};
use voxcpm::{Cfm, Config, LocalEncoder, Sampling};
use vulkan::{Context, executable::count_mnemonics};

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
/// out as the sampler takes it, and the temperature and guidance scale; named for what it is.
#[derive(Clone)]
struct Case {
    name: String,
    mu: Vec<f32>,
    cond: Vec<f32>,
    noise: Vec<f32>,
    sampling: [f32; 2],
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

/// The most rows a case is run among.
const ROWS: usize = 16;

/// [`ROWS`] patches of the golden after the ones before them, with conditions of about the scale
/// of a hidden state after a norm, the conditions rounded to f16 as the GPU takes them, so that
/// the oracles take them so too, the difference of their inputs being none of the GPU's error.
/// The first two at voxcpm's temperature and guidance scale, the rest each at a pair of its own.
fn cases(config: &PatchConfig) -> Vec<Case> {
    let (patches, noise) = golden();
    let hidden = config.dit_config.stack.hidden_dim;
    let default = Sampling::default();
    let others = (1..).filter(|at| ![3, 12].contains(at)).zip(3..);
    [(3, 1), (12, 2)]
        .into_iter()
        .chain(others)
        .take(ROWS)
        .enumerate()
        .map(|(i, (at, seed))| {
            let sampling = match i {
                0 | 1 => [default.temperature, default.cfg_value],
                _ => [[1.0, 0.8, 1.2][i % 3], [2.0, 1.5, 3.0, 2.5][i % 4]],
            };
            let mu = uniform(2 * hidden, seed, 1.7);
            Case {
                name: format!("golden{at}-mu{seed}-t{}-cfg{}", sampling[0], sampling[1]),
                mu: mu.iter().map(|&v| f16::from_f32(v).to_f32()).collect(),
                cond: transposed(&patches[at - 1], config.patch_size),
                noise: noise[at].clone(),
                sampling,
            }
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
/// patches `[feat_dim][patch_size]` flattened one after another, each kept by the case's name.
fn oracle(cases: &[Case], timesteps: usize, dtype: FloatDType) -> Vec<f32> {
    let config = Config::from_json(&config_json()).unwrap();
    let (feat_dim, patch_size) = (config.feat_dim, config.patch_size);
    let cpu = device();
    let mut cfm = None;
    let mut out = Vec::new();
    for case in cases {
        let name = format!("local-{}-{timesteps}-{dtype:?}", case.name);
        out.extend(kept::<Vec<f32>>(&name, || {
            let start = Instant::now();
            let cfm = cfm.get_or_insert_with(|| {
                Cfm::load(&config, &weights(), dtype, &cpu).unwrap_or_else(|e| panic!("{e}"))
            });
            let sampling = Sampling {
                timesteps,
                temperature: case.sampling[0],
                cfg_value: case.sampling[1],
            };
            let tensor = |values: &[f32], shape: [usize; 3]| {
                Tensor::<3>::from_data(TensorData::new(values.to_vec(), shape), &cpu)
            };
            let mu = TensorData::new(case.mu.clone(), [1, case.mu.len()]);
            let patch = cfm.sample(
                Tensor::<2>::from_data(mu, &cpu),
                tensor(&case.cond, [1, feat_dim, patch_size]),
                tensor(&case.noise, [1, feat_dim, patch_size]),
                &sampling,
            );
            eprintln!(
                "{} in {timesteps} steps on the CPU, {dtype:?}: {:.1} s",
                case.name,
                start.elapsed().as_secs_f64()
            );
            patch.into_data().iter::<f32>().collect()
        }));
    }
    out
}

fn patch_config() -> PatchConfig {
    serde_json::from_str(&config_json()).unwrap()
}

/// The GPU's patches for the cases, sampled `rows` at a time, the kernels cut as `tuning` says:
/// the rows of the last run past the cases are the first case again.
fn gpu(ctx: &Context, cases: &[Case], timesteps: usize, rows: usize, tuning: &Tuning) -> Vec<f32> {
    let config = patch_config();
    let mut kernels = Kernels::from_env(ctx).unwrap_or_else(|e| panic!("{e}"));
    let start = Instant::now();
    let mut sampler = Sampler::new(ctx, &mut kernels, &config, timesteps, rows, tuning, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(weights()).unwrap_or_else(|e| panic!("{e}"));
    sampler.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "sampler in {timesteps} steps over {rows} rows recorded and loaded on the GPU at \
         {tuning:?}: {:.1} s, {} dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        sampler.graph_step().dispatches(),
        sampler.graph_step().barriers(),
        kernels.len(),
        sampler.graph_step().arena_bytes() / 1024
    );
    let len = config.feat_dim * config.patch_size;
    let mut out = Vec::new();
    for chunk in cases.chunks(rows) {
        let mut run = chunk.to_vec();
        run.resize(rows, cases[0].clone());
        let all = |part: fn(&Case) -> &[f32]| -> Vec<f32> {
            run.iter().flat_map(part).copied().collect()
        };
        let sampling: Vec<[f32; 2]> = run.iter().map(|case| case.sampling).collect();
        let patches = sampler
            .sample(
                &f16_bits(&all(|case| &case.mu)),
                &all(|case| &case.cond),
                &all(|case| &case.noise),
                &sampling,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        out.extend(&patches[..len * chunk.len()]);
    }
    out
}

/// Whether `got` is `alone` bit for bit, and if not, where it first differs.
fn as_alone(got: &[f32], alone: &[f32], len: usize) -> Result<(), String> {
    assert_eq!(got.len(), alone.len());
    match got
        .iter()
        .zip(alone)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        None => Ok(()),
        Some(at) => Err(format!(
            "case {} differs from alone at {}: {} for {}",
            at / len,
            at % len,
            got[at],
            alone[at]
        )),
    }
}

/// The tunings each kernel is checked at, at either subgroup size: one tile or pair of tiles to
/// a subgroup and more, in workgroups of 64 to 512, one to each workgroup's worth of tiles or
/// fewer, which then take several in turn; the tokens staged in chunks of 512 columns and fewer,
/// at once or chunk by chunk, in blocks of 16 tokens and more; the weights loaded 1 to 8 steps
/// ahead; and the output and down projections cut into 1 to 8 parts. Those the device's shared
/// memory does not hold are left out, and said so.
fn tunings(ctx: &Context) -> Vec<Tuning> {
    let report = ctx.report();
    let tunings: Vec<Tuning> = subgroups(ctx)
        .into_iter()
        .flat_map(|s| {
            let base = fitted(ctx, Tuning::for_subgroup(s));
            let cut = |workgroup, pairs, tiles, chunk, block, depth, groups| Tuning {
                workgroup,
                qkv_pairs: pairs,
                gate_up_pairs: pairs,
                o_tiles: tiles,
                down_tiles: tiles,
                chunk,
                block,
                depth,
                groups,
                ..base
            };
            let parts = |o_parts, down_parts, tuning| Tuning {
                o_parts,
                down_parts,
                ..tuning
            };
            [
                base,
                cut(64, 1, 1, 256, 32, 4, 0),
                cut(128, 2, 2, 128, 32, 2, 0),
                cut(256, 2, 4, 64, 16, 1, 0),
                cut(256, 1, 1, 512, 16, 8, 40),
                cut(512, 1, 2, 128, 64, 4, 8),
                parts(1, 1, cut(256, 1, 1, 512, 32, 4, 0)),
                parts(4, 8, cut(256, 1, 1, 128, 32, 2, 0)),
                parts(1, 2, cut(128, 1, 2, 256, 48, 4, 3)),
            ]
        })
        .collect();
    let (held, left): (Vec<Tuning>, Vec<Tuning>) = tunings
        .into_iter()
        .partition(|t| t.shared_bytes() <= report.max_shared_memory as usize);
    for tuning in left {
        eprintln!(
            "left out: {tuning:?} takes {} bytes of shared memory, of {}",
            tuning.shared_bytes(),
            report.max_shared_memory
        );
    }
    held
}

/// The subgroup sizes the kernels are written for that the device can require: 32 alone on
/// NVIDIA's.
fn subgroups(ctx: &Context) -> Vec<u32> {
    let report = ctx.report();
    [64, 32]
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
        .collect()
}

/// `tuning` [fit](Tuning::fit) to the device.
fn fitted(ctx: &Context, tuning: Tuning) -> Tuning {
    tuning.fit(ctx.report()).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ORACLE_DIR a directory to keep the oracles in"]
fn keeps_the_oracles() {
    if oracle_dir().is_none() {
        eprintln!("ORACLE_DIR is not set: nothing to keep the oracles in");
        return;
    }
    let config = patch_config();
    let cases = cases(&config);
    let patches = encoder_patches(&config);
    for dtype in [FloatDType::F32, FloatDType::F16] {
        oracle(&cases, 2, dtype);
        oracle(&cases, Sampling::default().timesteps, dtype);
        encoder_oracle(&patches, dtype);
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_euler_step_samples_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    let len = config.feat_dim * config.patch_size;
    // Two steps, the first of which stays where it starts: one run of the DiT.
    let cases = cases(&config);
    let want = oracle(&cases, 2, FloatDType::F32);
    let burn_f16 = oracle(&cases, 2, FloatDType::F16);
    let ctx = context();
    for tuning in tunings(&ctx) {
        let alone = gpu(&ctx, &cases, 2, 1, &tuning);
        let margins = margins_of(&alone, &want, &burn_f16);
        assert!(within(margins), "{tuning:?}: {margins:.2?}");
        for rows in [3, ROWS] {
            let got = gpu(&ctx, &cases, 2, rows, &tuning);
            as_alone(&got, &alone, len).unwrap_or_else(|e| panic!("{rows} rows, {tuning:?}: {e}"));
        }
    }
    // Defect: the conditions left out, so the unconditional velocity alone.
    let unguided: Vec<Case> = cases
        .iter()
        .map(|case| Case {
            mu: vec![0.0; case.mu.len()],
            ..case.clone()
        })
        .collect();
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let got = gpu(&ctx, &unguided, 2, ROWS, &tuning);
    let margins = margins_of(&got, &want, &burn_f16);
    assert!(!within(margins), "without mu: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn ten_steps_sample_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    let len = config.feat_dim * config.patch_size;
    let cases = cases(&config);
    let timesteps = Sampling::default().timesteps;
    let want = oracle(&cases, timesteps, FloatDType::F32);
    let burn_f16 = oracle(&cases, timesteps, FloatDType::F16);
    let ctx = context();
    for subgroup in subgroups(&ctx) {
        let tuning = fitted(&ctx, Tuning::for_subgroup(subgroup));
        let alone = gpu(&ctx, &cases, timesteps, 1, &tuning);
        let margins = margins_of(&alone, &want, &burn_f16);
        assert!(within(margins), "{tuning:?}: {margins:.2?}");
        for rows in [5, ROWS] {
            let got = gpu(&ctx, &cases, timesteps, rows, &tuning);
            as_alone(&got, &alone, len).unwrap_or_else(|e| panic!("{rows} rows, {tuning:?}: {e}"));
        }
    }
    // Defect: a step short, the last step's velocity left out.
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let short = gpu(&ctx, &cases, timesteps - 1, ROWS, &tuning);
    let margins = margins_of(&short, &want, &burn_f16);
    assert!(!within(margins), "a step short: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}

/// [`compute_encoder_oracle`]'s, kept by the dtype alone: the encoder's test embeds the golden's
/// patches and [`voice_patches`]' 40.
fn encoder_oracle(patches: &[Vec<f32>], dtype: FloatDType) -> Vec<f32> {
    kept(&format!("local-encoder-{dtype:?}"), || {
        compute_encoder_oracle(patches, dtype)
    })
}

/// What voxcpm's local encoder gives for each patch `[patch_size][feat_dim]`, in `dtype`.
fn compute_encoder_oracle(patches: &[Vec<f32>], dtype: FloatDType) -> Vec<f32> {
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

/// The patches the encoder is checked on: over many, since the errors gather in a few outlier
/// channels, and at any one patch which of two runs rounds those worse is luck.
fn encoder_patches(config: &PatchConfig) -> Vec<Vec<f32>> {
    let mut patches = golden().0;
    patches.extend(voice_patches(config, 40));
    patches
}

/// The GPU's embedding of each patch `[patch_size][feat_dim]`, `rows` at a time, the kernels cut
/// as `tuning` says: the rows of the last run past the patches are the first patch again.
fn embed(
    ctx: &Context,
    checkpoint: &Checkpoint,
    patches: &[Vec<f32>],
    rows: usize,
    tuning: &Tuning,
) -> Vec<f32> {
    let config = patch_config();
    let mut kernels = Kernels::from_env(ctx).unwrap_or_else(|e| panic!("{e}"));
    let mut encoder = Encoder::new(ctx, &mut kernels, &config, rows, tuning, false)
        .unwrap_or_else(|e| panic!("{e}"));
    encoder.load(checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let hidden = config.encoder_config.hidden_dim;
    let mut got = Vec::new();
    for chunk in patches.chunks(rows) {
        let mut run = chunk.to_vec();
        run.resize(rows, patches[0].clone());
        let laid_out: Vec<f32> = run
            .iter()
            .flat_map(|patch| transposed(patch, config.patch_size))
            .collect();
        let out = encoder.encode(&laid_out).unwrap_or_else(|e| panic!("{e}"));
        got.extend(
            out[..hidden * chunk.len()]
                .iter()
                .map(|&h| f16::from_bits(h).to_f32()),
        );
    }
    got
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn the_encoder_embeds_as_voxcpm_does_on_the_cpu() {
    let config = patch_config();
    let hidden = config.encoder_config.hidden_dim;
    let patches = encoder_patches(&config);
    let want = encoder_oracle(&patches, FloatDType::F32);
    let burn_f16 = encoder_oracle(&patches, FloatDType::F16);
    let ctx = context();
    let checkpoint = Checkpoint::open(weights()).unwrap_or_else(|e| panic!("{e}"));
    for tuning in tunings(&ctx) {
        let alone = embed(&ctx, &checkpoint, &patches, 1, &tuning);
        let margins = margins_of(&alone, &want, &burn_f16);
        assert!(within(margins), "{tuning:?}: {margins:.2?}");
        for rows in [7, ROWS] {
            let got = embed(&ctx, &checkpoint, &patches, rows, &tuning);
            as_alone(&got, &alone, hidden)
                .unwrap_or_else(|e| panic!("{rows} rows, {tuning:?}: {e}"));
        }
    }
    // Defect: the patches' frames in another order, which only RoPE tells apart.
    let reversed: Vec<Vec<f32>> = patches
        .iter()
        .map(|patch| {
            let mut frames: Vec<&[f32]> = patch.chunks(config.feat_dim).collect();
            frames.reverse();
            frames.concat()
        })
        .collect();
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let got = embed(&ctx, &checkpoint, &reversed, ROWS, &tuning);
    let margins = margins_of(&got, &want, &burn_f16);
    assert!(!within(margins), "frames reversed: {margins:.2?}");
    assert_no_validation_messages(&ctx);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_sampler_and_the_encoder_load_into_a_graph_they_are_spliced_into() {
    let config = patch_config();
    let rows = 2;
    let cases = &cases(&config)[..rows];
    let patches = &golden().0[3..3 + rows];
    let ctx = context();
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(weights()).unwrap_or_else(|e| panic!("{e}"));
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let alone = gpu(&ctx, cases, 2, rows, &tuning);
    let embedded = embed(&ctx, &checkpoint, patches, rows, &tuning);

    // A tensor of another graph's first, so that each of theirs is somewhere else: loading
    // them where they were would write the wrong tensors.
    let mut g = Graph::default();
    g.tensor("before", 256, Kind::Constant);
    let sampler = sampler_graph(&config, 2, rows, &tuning).unwrap_or_else(|e| panic!("{e}"));
    let dit = splice(&mut g, &sampler.graph, "dit.", &[]);
    let enc = encoder_graph(&config, rows, &tuning).unwrap_or_else(|e| panic!("{e}"));
    let local = splice(&mut g, &enc.graph, "encoder.", &[]);
    let mut step = Step::new(&ctx, &mut kernels, &g, false).unwrap_or_else(|e| panic!("{e}"));
    load_sampler(&mut step, &g, &sampler, &dit, &checkpoint).unwrap_or_else(|e| panic!("{e}"));
    load_encoder(&mut step, &g, &enc, &local, &checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let at = |ids: &[TensorId], t: TensorId| ids[t.0];
    let all =
        |part: fn(&Case) -> &[f32]| -> Vec<f32> { cases.iter().flat_map(part).copied().collect() };
    step.write(at(&dit, sampler.mu), &f16_bits(&all(|case| &case.mu)))
        .unwrap();
    step.write(at(&dit, sampler.cond), &all(|case| &case.cond))
        .unwrap();
    step.write(at(&dit, sampler.noise), &all(|case| &case.noise))
        .unwrap();
    let sampling: Vec<f32> = cases.iter().flat_map(|case| case.sampling).collect();
    step.write(at(&dit, sampler.sampling), &sampling).unwrap();
    let laid_out: Vec<f32> = patches
        .iter()
        .flat_map(|patch| transposed(patch, config.patch_size))
        .collect();
    step.write(at(&local, enc.patch), &laid_out).unwrap();
    step.submit().unwrap_or_else(|e| panic!("{e}"));
    let got: Vec<f32> = step.read(at(&dit, sampler.patch)).unwrap();
    assert_eq!(got, alone, "the spliced sampler's patches");
    let got: Vec<u16> = step.read(at(&local, enc.out)).unwrap();
    let got: Vec<f32> = got.iter().map(|&h| f16::from_bits(h).to_f32()).collect();
    assert_eq!(got, embedded, "the spliced encoder's embeddings");
    assert_no_validation_messages(&ctx);
}

/// Sub-dword loads among the mnemonics of `assembly` that start with `prefix`.
fn narrow_loads(assembly: &str, prefix: &str) -> Vec<String> {
    count_mnemonics(assembly, prefix)
        .into_keys()
        .filter(|m| {
            !["_b32", "_b64", "_b96", "_b128"]
                .iter()
                .any(|w| m.ends_with(w))
        })
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_projections_compile_to_matrix_products_over_wide_loads() {
    // What the driver makes of the cooperative-matrix loads decides how fast the weights stream,
    // which no output can show: on RADV, whose final assembly is read, each projection's weights,
    // loaded straight from memory, and its tokens, from shared memory, are loaded 8 and 16 bytes
    // to a lane, never 2, and multiplied by v_wmma in f32, with nothing spilled. Another driver's
    // statistics are printed.
    let ctx = context();
    let radv = ctx.report().driver_name == "radv";
    if !ctx.captures_executables() {
        assert!(
            !radv,
            "the device does not offer VK_KHR_pipeline_executable_properties"
        );
        eprintln!(
            "{}: no VK_KHR_pipeline_executable_properties",
            ctx.report().driver_name
        );
        return;
    }
    let config = patch_config();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    for subgroup in subgroups(&ctx) {
        let tuning = fitted(&ctx, Tuning::for_subgroup(subgroup));
        let sampler = sampler_graph(&config, 2, 1, &tuning).unwrap_or_else(|e| panic!("{e}"));
        let mut seen = std::collections::BTreeSet::new();
        for op in &sampler.graph.ops {
            let name = op.pipeline.kernel.name;
            if !["local_qkv_rope", "local_gate_up", "local_partial"].contains(&name)
                || !seen.insert(op.pipeline.key())
            {
                continue;
            }
            let kernel = kernels.get(&op.pipeline).unwrap_or_else(|e| panic!("{e}"));
            let executables = kernel.executables().unwrap_or_else(|e| panic!("{e}"));
            let [e] = executables.as_slice() else {
                panic!("{name}: {} executables", executables.len());
            };
            assert_eq!(e.subgroup_size, subgroup, "{name}");
            let stats: Vec<String> = e
                .statistics
                .iter()
                .map(|s| format!("{} {}", s.name, s.value.as_f64()))
                .collect();
            eprintln!("{name} wave{subgroup}: {stats:?}");
            if !radv {
                continue;
            }
            let asm = e
                .representations
                .iter()
                .find(|r| r.name.contains("Assembly"))
                .and_then(|r| r.text.as_deref())
                .expect("RADV's final assembly");
            let wmma = count_mnemonics(asm, "v_wmma_");
            let wide = count_mnemonics(asm, "buffer_load_b128");
            eprintln!(
                "  {wmma:?}, {wide:?}, LDS {:?}",
                count_mnemonics(asm, "ds_load")
            );
            assert!(
                !wmma.is_empty() && wmma.keys().all(|m| m == "v_wmma_f32_16x16x16_f16"),
                "{name} wave{subgroup}: {wmma:?}"
            );
            assert!(!wide.is_empty(), "{name} wave{subgroup}: no 16-byte loads");
            for prefix in ["buffer_load", "global_load", "ds_load", "ds_read"] {
                let narrow = narrow_loads(asm, prefix);
                assert!(narrow.is_empty(), "{name} wave{subgroup}: {narrow:?}");
            }
            for stat in ["Spilled VGPRs", "Scratch size"] {
                let value = e.statistic(stat).map(|v| v.as_f64());
                assert_eq!(value, Some(0.0), "{name} wave{subgroup}: {stat}");
            }
        }
    }
    assert_no_validation_messages(&ctx);
}
