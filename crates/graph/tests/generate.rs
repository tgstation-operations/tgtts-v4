//! VoxCPM2's generation on the GPU, on the real weights, held to its goldens. Each patch is
//! sampled from the noise burn's CPU backend sampled it from in f32, after the patch it sampled
//! before, so that every run follows that one generation, and the goldens hold what the GPU gave
//! there. Built with the `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint,
//! `VOICES_DIR` a voices directory, `GOLDEN_DIR` the goldens and `GLSL_KERNELS` and
//! `VULKAN_KERNELS` the compiled kernels, as `nix run .#device-tests` sets the last two. Each
//! tolerance is set as testkit's `golden` says.
//!
//! The patch step is checked on a prompt of a voice's patches alone, run through the patch step a
//! position at a time: the sampler conditioned on what the LM side computes, where its own tests
//! make the condition up. The prompt is checked on the one the service builds, of the golden's
//! line in its voice, run at once into the patch step's caches. The generator speaks that line,
//! after those patches as above, its audio held to its golden; and run free, its audio is what
//! the GPU's decoder makes of its own patches. Where a free run stops is not held: a level apart
//! sends a generation elsewhere, and which levels land apart differs from one GPU to another.
//!
//! The scalar quantizer rounds the base LM's output to levels, and a value near a boundary
//! between two lands on either by luck of rounding: a level apart from the golden's is a
//! discrete event, which the sampler amplifies into the patch, not an error of the rounding's
//! size. So a run is held on the quantizer's input, a continuous value, by its largest and its RMS
//! error; on how many levels are apart; on the stop logits by their largest and RMS error; on the
//! decisions to stop; and on the patches by their largest and RMS error over all of them, as far
//! as the levels apart each GPU had took them. Teacher forcing keeps a level's effect on a patch
//! from reaching the next through the patch.
//!
//! The audio is the decoder's of those patches, a level apart that the sampler amplifies with
//! them. The decoder's causal convolutions put a patch's error in the audio of its frames and of
//! the next patch's first, and, measured on the real weights, at most 0.2% of its energy in any
//! frame after: so the audio of a patch's frames after its first is that patch's own. The audio
//! is held over all of it, and, when a run has `CLEAN` patches whose levels are all the
//! golden's, by its RMS error over those patches' spans together, where only rounding is apart.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{self, Row, SUBGROUPS},
    generator::{Generator, Settings},
    graph::{Kind, TensorId},
    kernels::Kernels,
    lm_side::{self, LmSideShape},
    local::{self, PatchConfig},
    patch::{PatchStep, patch_graph},
    prefill::TILE,
    prompt::Prompter,
    pth::Pth,
    vae::{VaeConfig, VaeDecoder, VaeShape},
};
use serde::{Deserialize, Serialize};
use testkit::{Fixture, checkpoint_dir, golden, golden_input};
use voices::Latent;
use voxcpm::{Params, Sampling};
use vulkan::{Context, F16};

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

/// The name of the voice voxcpm's golden was made with.
fn voice_name() -> String {
    let golden = Fixture::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_checkpoint.safetensors"
    ));
    golden.metadata()["voice"].clone()
}

/// The voice voxcpm's golden was made with.
fn voice() -> Latent {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let dir = std::path::absolute(dir).unwrap();
    let path = voices::latent_path(&dir, &voice_name());
    Latent::read(&path).unwrap_or_else(|e| panic!("{e}"))
}

/// A generation burn's CPU backend ran in f32, from `SeededNoise::new(SEED)`: its prompt, as
/// voxcpm's `Prompt` holds it, the noise each patch was sampled from, `[feat_dim][patch_size]`,
/// and each patch, `[patch_size][feat_dim]`.
#[derive(Deserialize)]
struct Input {
    tokens: Vec<u32>,
    prompt: Vec<f32>,
    audio: Vec<bool>,
    noise: Vec<Vec<f32>>,
    patches: Vec<Vec<f32>>,
}

impl Input {
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// The voice's patches alone as the prompt, and `GENERATED` patches after it whatever the stop
/// head said.
fn voice_input() -> Input {
    golden_input(&format!("input-patch-{}-{GENERATED}-{SEED}", voice_name()))
}

/// The prompt the service builds for the golden's line in its voice, and the patches after it
/// until the stop head said to stop, as the service stops.
fn line_input() -> Input {
    golden_input(&format!("input-line-{}-{SEED}", voice_name()))
}

/// `[rows][columns]` as `[columns][rows]`.
fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
    let columns = values.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect()
}

/// The patch step on the GPU for `input`'s prompt and patches, recorded and loaded, the LM side's
/// kernels at `subgroup` and the local stacks' at the device's tuning at that subgroup size (the
/// device's default when `None`); and the tensors of the base LM's output and the quantizer's
/// levels, made outputs for the host to read.
fn patch_step(
    ctx: &Context,
    kernels: &mut Kernels,
    input: &Input,
    subgroup: Option<u32>,
) -> (PatchStep, [TensorId; 2], decode::Tuning) {
    let json = config_json();
    let side_config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let capacity = (input.len() + input.noise.len()).next_multiple_of(32);
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
#[derive(Serialize, Deserialize)]
struct Run {
    patches: Vec<Vec<f32>>,
    stop: Vec<[f32; 2]>,
    base: Vec<Vec<f32>>,
    levels: Vec<Vec<f32>>,
}

/// Samples each of `input`'s patches on `step`, whose caches hold its prompt but the last
/// position, after the patch before, its `stale`-th before rather than the last.
fn sample(step: &mut PatchStep, probes: [TensorId; 2], input: &Input, stale: usize) -> Run {
    let sampling = Sampling::default();
    let len = input.len();
    let patch = input.prompt.len() / len;
    let mut out = Run {
        patches: vec![],
        stop: vec![],
        base: vec![],
        levels: vec![],
    };
    let mut elapsed = 0.0;
    for (i, noise) in input.noise.iter().enumerate() {
        let before = match i.checked_sub(stale) {
            Some(at) => &input.patches[at],
            None => &input.prompt[(len - 1) * patch..],
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
            let values: Vec<F16> = step.graph_step().read(t).unwrap();
            values.iter().map(|h| h.to_f32()).collect()
        };
        let levels = read(probes[1]).iter().map(|&l| (l * 9.0).round()).collect();
        out.patches.push(transposed(&sampled.patch, 64));
        out.stop.push(sampled.stop_logits);
        out.base.push(read(probes[0]));
        out.levels.push(levels);
    }
    eprintln!(
        "{} patches at {:.1} ms each from submission to fence (smoke, not a measurement)",
        input.noise.len(),
        elapsed * 1e3 / input.noise.len() as f64
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

/// The levels of each position that differ from the golden's.
fn flips(levels: &[Vec<f32>], want: &[Vec<f32>]) -> Vec<usize> {
    let differ = |(a, b): (&Vec<f32>, &Vec<f32>)| a.iter().zip(b).filter(|(a, b)| a != b).count();
    levels.iter().zip(want).map(differ).collect()
}

/// How far a run of the patch step may be from its golden.
struct Tolerance {
    /// The quantizer's input.
    base: Distance,
    /// The levels apart.
    flips: usize,
    stop: Distance,
    /// The patches, over all of them.
    patches: Distance,
}

/// How a run compares with its golden.
#[derive(Debug)]
struct Verdict {
    base: Distance,
    flips: usize,
    stop: Distance,
    patches: Distance,
    /// Whether the run and the golden decide alike whether to stop, wherever the golden's margin
    /// between going on and stopping is wider than the run's distance from it in the logits.
    decides_alike: bool,
}

impl Verdict {
    /// The criteria it fails.
    fn fails(&self, tolerance: &Tolerance) -> Vec<&'static str> {
        [
            ("quantizer input", self.base.within(tolerance.base)),
            ("levels apart", self.flips <= tolerance.flips),
            ("stop logits", self.stop.within(tolerance.stop)),
            ("patches", self.patches.within(tolerance.patches)),
            ("stop decisions", self.decides_alike),
        ]
        .into_iter()
        .filter(|&(_, holds)| !holds)
        .map(|(name, _)| name)
        .collect()
    }
}

fn judge(want: &Run, run: &Run) -> Verdict {
    let apart = flips(&run.levels, &want.levels);
    for i in 0..run.patches.len() {
        eprintln!(
            "patch {i}: levels apart {}; {:?}; stop logits {:?}, the golden's {:?}",
            apart[i],
            distance(&run.patches[i], &want.patches[i]),
            run.stop[i],
            want.stop[i],
        );
    }
    let all = |v: &[Vec<f32>]| v.concat();
    let [stop, want_stop] = [&run.stop, &want.stop].map(|s| s.concat());
    let error = distance(&stop, &want_stop).max_abs as f32;
    let decides_alike = want_stop.chunks(2).zip(stop.chunks(2)).all(|(w, g)| {
        let margin = w[1] - w[0];
        margin.abs() <= 2.0 * error || (margin > 0.0) == (g[1] > g[0])
    });
    let verdict = Verdict {
        base: distance(&all(&run.base), &all(&want.base)),
        flips: apart.iter().sum(),
        stop: distance(&stop, &want_stop),
        patches: distance(&all(&run.patches), &all(&want.patches)),
        decides_alike,
    };
    eprintln!("against the golden: {verdict:?}");
    verdict
}

/// The patch step over `input`, the voice's patches alone, its prompt run through the patch
/// step a position at a time.
fn patches_after_a_voice(input: &Input, subgroup: Option<u32>, stale: usize) -> Run {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let (mut step, probes, _) = patch_step(&ctx, &mut kernels, input, subgroup);
    let sampling = Sampling::default();
    let patch = input.prompt.len() / input.len();
    let silence = vec![0.0; patch];
    for (t, cond) in input.prompt.chunks(patch).take(input.len() - 1).enumerate() {
        let cond = transposed(cond, 4);
        step.run(t, &cond, &silence, sampling.temperature, sampling.cfg_value)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    let run = sample(&mut step, probes, input, stale);
    assert_no_validation_messages(&ctx);
    run
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

/// Asserts that each run passes every criterion of `tolerance`, or with a defect fails the
/// defect's.
fn assert_checked(runs: &[Checked], tolerance: &Tolerance) {
    for run in runs {
        let (subgroup, fails) = (run.subgroup, run.verdict.fails(tolerance));
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

/// The patch step's tolerance after the voice: against burn's CPU in f32, the largest error and
/// relative RMS error of any run on the Radeon 8060S and the RTX 4070 Ti SUPER were 3.45e-2 and
/// 4.19e-3 over the quantizer's input, 5.90e-2 and 7.55e-3 over the stop logits and 4.02e-1 and
/// 2.00e-2 over the patches, with up to 222 levels apart; the golden's were 3.12e-2 and 3.78e-3,
/// 5.74e-2 and 7.55e-3, and 4.02e-1 and 2.00e-2, with 208.
const AFTER_A_VOICE: Tolerance = Tolerance {
    base: Distance {
        max_abs: 8.3e-2,
        relative_rms: 1.0e-2,
    },
    flips: 537,
    stop: Distance {
        max_abs: 1.5e-1,
        relative_rms: 1.9e-2,
    },
    patches: Distance {
        max_abs: 1.01,
        relative_rms: 5.0e-2,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn patches_sample_as_their_golden() {
    let input = voice_input();
    let name = format!("patch-{}-{GENERATED}-{SEED}", voice_name());
    let want = golden(&name, || patches_after_a_voice(&input, None, 1));
    let mut runs = vec![];
    for subgroup in subgroups() {
        let run = patches_after_a_voice(&input, Some(subgroup), 1);
        runs.push(Checked {
            subgroup,
            defect: None,
            verdict: judge(&want, &run),
        });
        let run = patches_after_a_voice(&input, Some(subgroup), 2);
        runs.push(Checked {
            subgroup,
            defect: Some((
                "each patch after the one two before".into(),
                "quantizer input",
            )),
            verdict: judge(&want, &run),
        });
    }
    assert_checked(&runs, &AFTER_A_VOICE);
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

/// The prompt and the patch step over `input`, its prompt run at once.
fn patches_after_a_prompt(input: &Input, subgroup: Option<u32>, how: Prompting) -> Run {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let (mut step, probes, tuning) = patch_step(&ctx, &mut kernels, input, subgroup);
    let prompter = Prompter::new(&ctx, &mut kernels, &mut step, 2 * TILE, &tuning, true)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    let prompt = graph::prompt::Prompt {
        tokens: &input.tokens,
        patches: &input.prompt,
        audio: &input.audio,
    };
    let start = Instant::now();
    let (x, audio_x) = prompter
        .embeddings(&mut step, &checkpoint, prompt)
        .unwrap_or_else(|e| panic!("{e}"));
    let embedded = start.elapsed().as_secs_f64() * 1e3;
    let ran = input.len() - 1;
    let audio = match how {
        Prompting::QuantizedText => vec![true; ran],
        _ => input.audio[..ran].to_vec(),
    };
    let at = Row {
        position: usize::from(how == Prompting::Shifted),
        slot: 0,
    };
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
        input.len(),
        ms(""),
        ms("fsq_layer") + ms("select_rows") + ms("fusion_concat_proj")
    );
    let run = sample(&mut step, probes, input, 1);
    assert_no_validation_messages(&ctx);
    run
}

/// The patch step's tolerance after the line's prompt: against f32, the largest error and
/// relative RMS error of any run were 5.31e-3 and 1.28e-3 over the quantizer's input, 1.30e-1 and
/// 3.59e-3 over the stop logits and 1.46e-1 and 7.10e-3 over the patches, with up to 37 levels
/// apart; the golden's were 5.31e-3 and 1.28e-3, 1.22e-1 and 3.59e-3, and 8.84e-2 and 4.94e-3,
/// with 37.
const AFTER_A_PROMPT: Tolerance = Tolerance {
    base: Distance {
        max_abs: 1.33e-2,
        relative_rms: 3.2e-3,
    },
    flips: 92,
    stop: Distance {
        max_abs: 3.2e-1,
        relative_rms: 9.0e-3,
    },
    patches: Distance {
        max_abs: 2.93e-1,
        relative_rms: 1.51e-2,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_prompt_runs_as_its_golden() {
    let input = line_input();
    let name = format!("line-{}-{SEED}", voice_name());
    let want = golden(&name, || {
        patches_after_a_prompt(&input, None, Prompting::AsVoxcpm)
    });
    // The quantizer's output taken at text positions leaves the base LM as it is, and moves the
    // residual LM's state, so the patches; the prompt one position on moves both.
    let defects = [
        (Prompting::QuantizedText, "patches"),
        (Prompting::Shifted, "quantizer input"),
    ];
    let mut runs = vec![];
    for subgroup in subgroups() {
        let run = patches_after_a_prompt(&input, Some(subgroup), Prompting::AsVoxcpm);
        runs.push(Checked {
            subgroup,
            defect: None,
            verdict: judge(&want, &run),
        });
        for (how, criterion) in defects {
            let run = patches_after_a_prompt(&input, Some(subgroup), how);
            runs.push(Checked {
                subgroup,
                defect: Some((format!("{how:?}"), criterion)),
                verdict: judge(&want, &run),
            });
        }
    }
    assert_checked(&runs, &AFTER_A_PROMPT);
}

/// The generator for the checkpoint, as the service would run it, with caches for `capacity`
/// positions, its kernels at `subgroup`.
fn generator(ctx: &Context, kernels: &mut Kernels, capacity: usize, subgroup: u32) -> Generator {
    let weights = Pth::open(checkpoint_dir().join("audiovae.pth")).unwrap();
    let settings = Settings {
        capacity,
        max_rows: 2 * TILE,
        timesteps: Sampling::default().timesteps,
        chunk_frames: Params::default().decode_chunk_frames.get(),
        timed: false,
        prompt_cache_bytes: 0,
        subgroup: Some(subgroup),
        slots: 1,
    };
    let start = Instant::now();
    let parameter = |name: &str| weights.floats(&format!("decoder.{name}"));
    let generator = Generator::new(ctx, kernels, &checkpoint_dir(), &settings, parameter)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "generator recorded and loaded at subgroups of {subgroup}: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    generator
}

/// A request's patches, `[patch_size][feat_dim]` each, the quantizer's levels each was sampled
/// after, and its audio.
#[derive(Clone, Serialize, Deserialize)]
struct Speech {
    patches: Vec<Vec<f32>>,
    levels: Vec<Vec<f32>>,
    audio: Vec<f32>,
}

/// `input`'s line spoken by `generator` after `voice`, each patch from its `noise`: sampled after
/// `input`'s `stale`-th before, as teacher forcing does, or after its own when `stale` is `None`;
/// the frames waiting decoded at once after each patch whose number `early` has, counting from 1.
fn speak(
    generator: &mut Generator,
    input: &Input,
    voice: &Latent,
    noise: &[Vec<f32>],
    stale: Option<usize>,
    early: &[usize],
) -> Speech {
    let sampling = Sampling::default();
    let prompt = graph::prompt::Prompt {
        tokens: &input.tokens,
        patches: &input.prompt,
        audio: &input.audio,
    };
    let start = Instant::now();
    generator
        .start(
            0,
            prompt,
            voice.frames.as_flattened(),
            Params::default().max_patches,
            sampling.temperature,
            sampling.cfg_value,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let (mut patches, mut levels, mut audio) = (vec![], vec![], vec![]);
    for (i, noise) in noise.iter().enumerate() {
        if let Some(before) = stale.and_then(|stale| i.checked_sub(stale)) {
            generator
                .sample_after(0, &input.patches[before])
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let stepped = generator
            .step(|_| noise.clone())
            .unwrap_or_else(|e| panic!("{e}"));
        let Some((_, patch)) = stepped.into_iter().next() else {
            break;
        };
        let patch = patch.unwrap_or_else(|e| panic!("{e}"));
        if i == 0 {
            eprintln!(
                "the first patch {:.1} ms after the request (smoke, not a measurement)",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
        patches.push(patch.frames);
        if early.contains(&patches.len()) {
            generator
                .decode_waiting(0)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        levels.push(generator.levels(0).unwrap_or_else(|e| panic!("{e}")));
        audio.extend(generator.take_audio(0));
        if patch.last {
            break;
        }
    }
    audio.extend(generator.finish(0).unwrap_or_else(|e| panic!("{e}")));
    eprintln!(
        "{} patches and {} samples in {:.2} s (smoke, not a measurement)",
        patches.len(),
        audio.len(),
        start.elapsed().as_secs_f64()
    );
    Speech {
        patches,
        levels,
        audio,
    }
}

/// The patches where none of a run's levels is apart from the golden's that it must have for its
/// audio to be held over their spans.
const CLEAN: usize = 4;

/// How far the generator's audio may be from its golden: over all of it, and in relative RMS
/// error over the spans of the patches where no level is apart.
struct AudioTolerance {
    all: Distance,
    clean_rms: f64,
}

/// How a run's audio compares with its golden's.
#[derive(Debug)]
struct AudioVerdict {
    all: Distance,
    /// The patches where none of the run's levels is apart from the golden's, and the relative
    /// RMS error over their spans together.
    clean: usize,
    clean_rms: f64,
}

impl AudioVerdict {
    /// The criteria it fails. The spans of the patches clear of levels apart are held where a
    /// run has enough of them.
    fn fails(&self, tolerance: &AudioTolerance) -> Vec<&'static str> {
        let enough = self.clean >= CLEAN;
        [
            ("all audio", self.all.within(tolerance.all)),
            (
                "clean audio",
                !enough || self.clean_rms <= tolerance.clean_rms,
            ),
        ]
        .into_iter()
        .filter(|&(_, holds)| !holds)
        .map(|(name, _)| name)
        .collect()
    }
}

/// `speech` against its golden, `want`, by its levels at each patch against the golden's, in spans
/// of each patch's frames after its first; each span printed when `table`.
fn judge_audio(name: &str, speech: &Speech, want: &Speech, table: bool) -> AudioVerdict {
    let (audio, levels) = (&speech.audio[..], &speech.levels[..]);
    let patches = want.levels.len();
    if audio.len() != want.audio.len() || levels.len() != patches {
        eprintln!(
            "{name}: {} samples of {} and the levels of {} patches of {patches}",
            audio.len(),
            want.audio.len(),
            levels.len()
        );
        let infinite = Distance {
            max_abs: f64::INFINITY,
            relative_rms: f64::INFINITY,
        };
        return AudioVerdict {
            all: infinite,
            clean: 0,
            clean_rms: f64::INFINITY,
        };
    }
    let apart = flips(levels, &want.levels);
    let frame = want.audio.len() / (4 * patches);
    let (mut ours, mut theirs) = (vec![], vec![]);
    for (i, &apart) in apart.iter().enumerate() {
        let span = (4 * i + 1) * frame..(4 * i + 4) * frame;
        if apart == 0 {
            ours.extend_from_slice(&audio[span.clone()]);
            theirs.extend_from_slice(&want.audio[span.clone()]);
        }
        if table {
            eprintln!(
                "{name}, patch {i}'s frames after its first: levels apart {apart}; {:?}",
                distance(&audio[span.clone()], &want.audio[span])
            );
        }
    }
    let clean = apart.iter().filter(|&&apart| apart == 0).count();
    let verdict = AudioVerdict {
        all: distance(audio, &want.audio),
        clean,
        clean_rms: distance(&ours, &theirs).relative_rms,
    };
    eprintln!("{name}: {verdict:?}");
    verdict
}

/// The audio's tolerance: against burn's CPU in f32, the largest error and relative RMS error of
/// any run on the Radeon 8060S and the RTX 4070 Ti SUPER were 4.05e-3 and 1.41e-2 over all of it,
/// and the relative RMS error 1.98e-3 over the spans of the patches where no level was apart from
/// f32's; the golden's were 2.58e-3 and 9.34e-3, and 1.64e-3.
const AUDIO: AudioTolerance = AudioTolerance {
    all: Distance {
        max_abs: 8.3e-3,
        relative_rms: 3.0e-2,
    },
    clean_rms: 4.5e-3,
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            GOLDEN_DIR the goldens"]
fn speech_comes_out_as_its_golden() {
    let input = line_input();
    let voice = voice();
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let capacity = (input.len() + Params::default().max_patches).next_multiple_of(32);
    let want = golden(&format!("speech-{}-{SEED}", voice_name()), || {
        let lm = decode::Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
        let mut generator = generator(&ctx, &mut kernels, capacity, lm.subgroup);
        speak(&mut generator, &input, &voice, &input.noise, Some(1), &[])
    });
    assert_eq!(want.audio.len(), input.noise.len() * 4 * 1920);
    let mut failed = vec![];
    for subgroup in subgroups() {
        let mut generator = generator(&ctx, &mut kernels, capacity, subgroup);
        let fails = speech_at(&ctx, &mut kernels, &mut generator, &input, &voice, &want);
        assert_no_validation_messages(&ctx);
        if let Err(fails) = fails {
            failed.push(format!("wave{subgroup}: {fails}"));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The voice's last frames the service primes its decoder with before a speech,
/// nanovllm-voxcpm's `n_decode_pad_frames`.
const CONTEXT_FRAMES: usize = 12;

/// `patches`' audio as the service decodes it on the GPU, in its chunks after `voice`'s last
/// [`CONTEXT_FRAMES`] frames, whose own audio is left out.
fn decoded(ctx: &Context, kernels: &mut Kernels, voice: &[f32], patches: &[Vec<f32>]) -> Vec<f32> {
    let json: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    let config: VaeConfig = serde_json::from_value(json["audio_vae_config"].clone()).unwrap();
    let chunk = Params::default().decode_chunk_frames.get();
    let shape = VaeShape::new(&config, chunk, 1).unwrap_or_else(|e| panic!("{e}"));
    let mut decoder =
        VaeDecoder::new(ctx, kernels, &shape, false).unwrap_or_else(|e| panic!("{e}"));
    let weights = Pth::open(checkpoint_dir().join("audiovae.pth")).unwrap();
    decoder
        .load(|name| weights.floats(&format!("decoder.{name}")))
        .unwrap_or_else(|e| panic!("{e}"));
    let latent = shape.latent;
    let context = (voice.len() / latent).min(CONTEXT_FRAMES);
    let mut frames = voice[voice.len() - context * latent..].to_vec();
    frames.extend(patches.concat());
    let mut audio = Vec::new();
    for chunk in frames.chunks(chunk * latent) {
        audio.extend(decoder.push(0, chunk).unwrap_or_else(|e| panic!("{e}")));
    }
    audio.split_off(context * shape.hop())
}

/// The speech test on `generator`, the failures it finds, if any.
fn speech_at(
    ctx: &Context,
    kernels: &mut Kernels,
    generator: &mut Generator,
    input: &Input,
    voice: &Latent,
    want: &Speech,
) -> Result<(), String> {
    let mut next_noise = input.noise.clone();
    next_noise.rotate_left(1);
    let speak = |generator: &mut Generator, noise: &[Vec<f32>], stale, early: &[usize]| {
        speak(generator, input, voice, noise, stale, early)
    };
    // Each patch after the input's patch before: it stops where the golden does, and its audio
    // is the golden's but for rounding and the levels apart.
    let forced = speak(generator, &input.noise, Some(1), &[]);
    // Defect: each patch after the input's two before, which moves the levels.
    let stale = speak(generator, &input.noise, Some(2), &[]);
    // Defect: each patch from the next one's noise, which leaves the levels as they are.
    let noisy = speak(generator, &next_noise, Some(1), &[]);
    // Each patch after its own, as the service generates.
    let free = speak(generator, &input.noise, None, &[]);
    // Decoding the frames waiting after the patches a stream decodes early at.
    let early = speak(generator, &input.noise, None, &[1, 2, 4, 8]);

    let verdict = judge_audio("audio", &forced, want, true);
    let mut fails: Vec<String> = verdict
        .fails(&AUDIO)
        .into_iter()
        .map(String::from)
        .collect();
    // Defect: the audio 5% louder, as a decoder off in its gain would make it.
    let louder = Speech {
        audio: forced.audio.iter().map(|a| 1.05 * a).collect(),
        ..forced.clone()
    };
    let defects = [
        ("audio after stale patches", &stale, "all audio"),
        ("audio from the next patch's noise", &noisy, "clean audio"),
        ("audio 5% louder", &louder, "clean audio"),
    ];
    for (defect, speech, criterion) in defects {
        let verdict = judge_audio(defect, speech, want, false);
        if !verdict.fails(&AUDIO).contains(&criterion) {
            fails.push(format!("{defect} passes {criterion}"));
        }
    }

    // Run free, its audio is what the GPU's decoder makes of its patches after the voice's last
    // frames, in the service's chunks, as chunks decode as the whole does.
    let frames = voice.frames.as_flattened();
    let d = distance(&free.audio, &decoded(ctx, kernels, frames, &free.patches));
    eprintln!(
        "run free, {} patches; against the decoder over its patches: {d:?}",
        free.patches.len()
    );
    // The audio VAE carries its state from chunk to chunk, so audio decoded early in chunks
    // of fewer frames is the same audio.
    if early.audio.len() == free.audio.len() {
        let early = distance(&early.audio, &free.audio);
        eprintln!("decoded early, against run free: {early:?}");
    }
    // Defect: the decoder not primed with the voice's last frames.
    let unprimed = distance(&free.audio, &decoded(ctx, kernels, &[], &free.patches));
    eprintln!("run free, against the decoder not primed: {unprimed:?}");

    // Every check is made before any fails, so that a run reports the free run's too.
    let checks = [
        ("stops", forced.patches.len() == want.patches.len()),
        ("run free, audio", d.max_abs <= 1e-6),
        ("early, same patches", early.patches == free.patches),
        ("early, same audio", early.audio == free.audio),
        ("audio not primed fails", unprimed.max_abs > 1e-4),
    ];
    let failed = checks.into_iter().filter(|&(_, holds)| !holds);
    fails.extend(failed.map(|(name, _)| name.to_string()));
    if fails.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{fails:?}: {} patches of {}, {} run free; {verdict:?}; run free {d:?}, not primed \
         {unprimed:?}",
        forced.patches.len(),
        want.patches.len(),
        free.patches.len()
    ))
}
