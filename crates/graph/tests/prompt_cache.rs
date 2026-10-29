//! The prompt cache on the GPU, on the real weights: prompts in voices it keeps, run from what it
//! keeps of them, against the same prompts run whole. The LMs are causal and their kernels compute
//! each position's row alone, so a prompt that goes on from the keys and values kept of the
//! positions it begins with samples the same patches from the same noise, bit for bit. So does a
//! prompt run, whole or from what is kept, into any slot of the caches, which leaves the others as
//! they were. Built with the `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint,
//! `VOICES_DIR` a voices directory and `GLSL_KERNELS` and `VULKAN_KERNELS` the compiled kernels,
//! as `nix run .#device-tests` sets the last two.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{self, Batch, Row},
    kernels::Kernels,
    lm_side::{self, LmSideShape},
    local::{self, PatchConfig},
    patch::{PatchRow, PatchStep, patch_batch_graph},
    prefill::TILE,
    prompt::{Prompt, PromptCache, Prompter},
};
use testkit::checkpoint_dir;
use voices::{Latent, Voices};
use voxcpm::{Config, Noise, Sampling, SeededNoise, Tokenizer};
use vulkan::{Context, F16};

/// Patches sampled after each prompt.
const PATCHES: usize = 8;

const SEED: u64 = 0;

/// What the cache keeps unless a test says otherwise, as the service's default.
const BUDGET: usize = 512 << 20;

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

/// The prompts the service builds: `lines` in each of the first two voices of `VOICES_DIR`, a
/// voice's lines after its transcript, and the voices' latent frames.
fn prompts(lines: &[&str]) -> Vec<(Vec<voxcpm::Prompt>, Vec<f32>)> {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let voices = Voices::open(dir).unwrap_or_else(|e| panic!("{e}"));
    let dir = checkpoint_dir();
    let config = Config::from_json(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = fs::read_to_string(dir.join("tokenizer.json")).unwrap();
    let tokenizer = Tokenizer::from_json(&tokenizer).unwrap();
    let names: Vec<&str> = voices.names().take(2).collect();
    names
        .iter()
        .map(|name| {
            let voice: Latent = voices.latent(name).unwrap_or_else(|e| panic!("{e}"));
            let frames = voice.frames.as_flattened().to_vec();
            let prompts = lines
                .iter()
                .map(|line| {
                    let tokens = tokenizer.encode(&format!("{}{line}", voice.text));
                    voxcpm::Prompt::cloning(&tokens, &frames, &config).unwrap()
                })
                .collect();
            (prompts, frames)
        })
        .collect()
}

/// The patch step for prompts of up to `positions` positions and [`PATCHES`] after them in caches
/// of `slots` slots, and a prompter over it whose runs take up to two tiles of positions, so that
/// a prompt runs in pieces.
struct Gpu {
    ctx: Context,
    step: PatchStep,
    prompter: Prompter,
    checkpoint: Checkpoint,
}

fn gpu(positions: usize, slots: usize) -> Gpu {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let side_config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let capacity = (positions + PATCHES).next_multiple_of(32);
    let shape = LmSideShape::new(&side_config, capacity).unwrap();
    let lm = decode::Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let local = local::Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let start = Instant::now();
    let timesteps = Sampling::default().timesteps;
    let batch = Batch { rows: 1, slots };
    let graph = patch_batch_graph(&shape, &config, timesteps, &lm, &local, batch)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut step =
        PatchStep::new(&ctx, &mut kernels, graph, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    step.load(&checkpoint).unwrap_or_else(|e| panic!("{e}"));
    let prompter = Prompter::new(&ctx, &mut kernels, &mut step, 2 * TILE, &lm, false)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "patch step and prompter recorded and loaded: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    Gpu {
        ctx,
        step,
        prompter,
        checkpoint,
    }
}

/// `[rows][columns]` as `[columns][rows]`.
fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
    let columns = values.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect()
}

/// What a prompt gave: the positions that ran through the LMs, and the bits of the patches
/// sampled after it and of the stop logits beside each.
#[derive(PartialEq)]
struct Generated {
    prefilled: usize,
    bits: Vec<u32>,
}

impl Generated {
    /// The values whose bits differ from `want`'s.
    fn differing(&self, want: &Generated) -> usize {
        let pairs = self.bits.iter().zip(&want.bits);
        pairs.filter(|(a, b)| a != b).count() + self.bits.len().abs_diff(want.bits.len())
    }
}

impl std::fmt::Debug for Generated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let first = self.bits.first().map(|&b| f32::from_bits(b));
        write!(
            f,
            "{} positions run, {} values from {first:?}",
            self.prefilled,
            self.bits.len()
        )
    }
}

/// `prompt` in the voice of latent frames `voice` run into slot `slot` from what `cache` keeps,
/// then [`PATCHES`] patches there each after the one before from noise seeded alike for every
/// prompt.
fn generate(
    gpu: &mut Gpu,
    cache: &mut PromptCache,
    prompt: &voxcpm::Prompt,
    voice: &[f32],
    slot: usize,
) -> Generated {
    let positions = Prompt {
        tokens: &prompt.tokens,
        patches: &prompt.patches,
        audio: &prompt.audio,
    };
    let prefilled = gpu
        .prompter
        .run_cached(
            &mut gpu.step,
            slot,
            &gpu.checkpoint,
            positions,
            voice,
            cache,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    sample(gpu, prompt, slot, prefilled)
}

/// [`PATCHES`] patches after `prompt`, which `slot` of the caches holds, as [`generate`] samples
/// them.
fn sample(gpu: &mut Gpu, prompt: &voxcpm::Prompt, slot: usize, prefilled: usize) -> Generated {
    let sampling = Sampling::default();
    let len = prompt.len();
    let patch = prompt.patches.len() / len;
    let mut noise = SeededNoise::new(SEED);
    let mut before = transposed(&prompt.patches[(len - 1) * patch..], 4);
    let mut bits = vec![];
    for i in 0..PATCHES {
        let noise = noise.next(patch);
        let row = PatchRow {
            at: Row {
                position: len - 1 + i,
                slot,
            },
            cond: &before,
            noise: &noise,
            temperature: sampling.temperature,
            cfg: sampling.cfg_value,
        };
        let sampled = gpu
            .step
            .run_rows(&[Some(row)])
            .unwrap_or_else(|e| panic!("{e}"))
            .remove(0)
            .expect("a row that runs");
        bits.extend(sampled.patch.iter().map(|v| v.to_bits()));
        bits.extend(sampled.stop_logits.map(f32::to_bits));
        before = sampled.patch;
    }
    Generated { prefilled, bits }
}

/// The positions of `prompt` before its last run of audio: what the cache keeps of the first
/// prompt in a voice.
fn held(prompt: &voxcpm::Prompt) -> usize {
    prompt.audio.iter().rposition(|&a| !a).unwrap() + 1
}

/// The positions two prompts begin with alike.
fn shared(a: &voxcpm::Prompt, b: &voxcpm::Prompt) -> usize {
    let patch = a.patches.len() / a.len();
    (0..a.len().min(b.len()))
        .take_while(|&p| {
            let at = p * patch..(p + 1) * patch;
            a.tokens[p] == b.tokens[p]
                && a.audio[p] == b.audio[p]
                && a.patches[at.clone()] == b.patches[at]
        })
        .count()
}

/// A line long enough that its prompt in either voice reaches past what the other's prompts keep.
const LONG: &str = "Attention, crew: the supermatter engine is delaminating. All personnel are to \
                    evacuate the engineering department at once and report to the nearest escape \
                    pod. Engineers, if you are still in the chamber, get out now and do not stop \
                    to grab your tools. This is not a drill. I repeat, this is not a drill.";

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn a_prompt_in_a_kept_voice_samples_as_it_does_run_whole() {
    let voices = prompts(&["Hello, crew.", "Someone set the kitchen on fire.", LONG]);
    let (prompts, voice) = &voices[0];
    let [a, b] = [&prompts[0], &prompts[1]];
    let (other, other_voice) = (&voices[1].0[2], &voices[1].1);
    let mut gpu = gpu(
        [a, b, other]
            .map(voxcpm::Prompt::len)
            .into_iter()
            .max()
            .unwrap(),
        1,
    );
    let mut whole = PromptCache::new(&gpu.ctx, 0);
    let want = [a, b].map(|prompt| generate(&mut gpu, &mut whole, prompt, voice, 0));
    for (want, prompt) in want.iter().zip([a, b]) {
        assert_eq!(want.prefilled, prompt.len());
    }
    assert_eq!((whole.voices(), whole.bytes()), (0, 0));
    // A prompt in another voice between two in this one, so that what the caches hold of the
    // first is not this voice's when the second is run.
    assert!(other.len() - 1 + PATCHES >= held(a));
    let mut overwrite = |gpu: &mut Gpu| generate(gpu, &mut whole, other, other_voice, 0);

    let mut cache = PromptCache::new(&gpu.ctx, BUDGET);
    // The first in the voice runs whole, and the cache keeps it up to its last run of audio.
    let first = generate(&mut gpu, &mut cache, a, voice, 0);
    assert_eq!(first, want[0]);
    assert_eq!(cache.voices(), 1);
    // The next goes on from the positions it shares with that: the voice as reference audio and
    // the transcript's tokens as far as the text after it leaves them alike.
    let patches = voice.len() / (a.patches.len() / a.len());
    let alike = shared(a, b);
    assert!(
        alike >= 2 + patches && alike < held(a),
        "{alike} of {}",
        held(a)
    );
    overwrite(&mut gpu);
    let next = generate(&mut gpu, &mut cache, b, voice, 0);
    assert_eq!(next.prefilled, b.len() - alike);
    let differing = next.differing(&want[1]);
    assert_eq!(differing, 0, "a prompt from {alike} positions kept");
    // The first again shares all it kept, and runs only the voice spoken again.
    overwrite(&mut gpu);
    let again = generate(&mut gpu, &mut cache, a, voice, 0);
    assert_eq!(again.prefilled, a.len() - held(a));
    let differing = again.differing(&want[0]);
    assert_eq!(differing, 0, "a prompt from {} positions kept", held(a));
    eprintln!(
        "prompts of {} and {} positions, {alike} of them alike, {} kept: {} bytes kept",
        a.len(),
        b.len(),
        held(a),
        cache.bytes()
    );
    assert_no_validation_messages(&gpu.ctx);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn the_cache_keeps_what_fits_and_drops_the_voice_used_longest_ago() {
    let voices = prompts(&["Clown in the armory."]);
    let [(a, v), (c, w)] = [&voices[0], &voices[1]].map(|(prompts, voice)| (&prompts[0], voice));
    let mut gpu = gpu(a.len().max(c.len()), 1);
    let mut whole = PromptCache::new(&gpu.ctx, 0);
    let want = generate(&mut gpu, &mut whole, a, v, 0);

    let kept = |gpu: &mut Gpu, prompt, voice| {
        let mut cache = PromptCache::new(&gpu.ctx, BUDGET);
        generate(gpu, &mut cache, prompt, voice, 0);
        cache.bytes()
    };
    let (bytes_v, bytes_w) = (kept(&mut gpu, a, v), kept(&mut gpu, c, w));
    // Room for either voice, and not both.
    let mut cache = PromptCache::new(&gpu.ctx, bytes_v.max(bytes_w));
    generate(&mut gpu, &mut cache, a, v, 0);
    assert_eq!((cache.voices(), cache.bytes()), (1, bytes_v));
    generate(&mut gpu, &mut cache, c, w, 0);
    assert_eq!((cache.voices(), cache.bytes()), (1, bytes_w));
    let back = generate(&mut gpu, &mut cache, a, v, 0);
    assert_eq!(back, want, "the voice dropped runs whole");
    assert_eq!((cache.voices(), cache.bytes()), (1, bytes_v));

    // Room for the voice's embeddings and not its keys and values: it keeps the embeddings,
    // and runs whole each time.
    let mut cache = PromptCache::new(&gpu.ctx, bytes_v - 1);
    for _ in 0..2 {
        assert_eq!(generate(&mut gpu, &mut cache, a, v, 0), want);
        assert_eq!(cache.voices(), 1);
        assert!(
            cache.bytes() < bytes_v / 2,
            "{} of {bytes_v}",
            cache.bytes()
        );
    }
    assert_no_validation_messages(&gpu.ctx);
}

/// The slots of the caches a prompt is run into below.
const SLOTS: usize = 4;

/// Each slot of each of both LMs' caches, `[cache][slot]`.
fn caches(gpu: &mut Gpu) -> Vec<Vec<Vec<F16>>> {
    let (graph, step) = gpu.step.parts_mut();
    let side = &graph.side;
    [&side.base, &side.residual]
        .into_iter()
        .flat_map(|stack| &stack.layers)
        .flat_map(|l| [l.k_cache, l.v_cache])
        .map(|t| {
            let held: Vec<F16> = step
                .read(graph.side_ids[t.0])
                .unwrap_or_else(|e| panic!("{e}"));
            let slot = held.len() / side.batch.slots;
            held.chunks(slot).map(<[F16]>::to_vec).collect()
        })
        .collect()
}

/// The slots whose bits differ between two readings of [`caches`].
fn changed(before: &[Vec<Vec<F16>>], after: &[Vec<Vec<F16>>]) -> Vec<usize> {
    (0..SLOTS)
        .filter(|&s| before.iter().zip(after).any(|(b, a)| b[s] != a[s]))
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn a_prompt_runs_into_any_slot_as_into_the_first_and_leaves_the_others_alone() {
    let voices = prompts(&["Hello, crew.", "Someone set the kitchen on fire."]);
    let (prompts, voice) = &voices[0];
    let [a, b] = [&prompts[0], &prompts[1]];
    let (other, other_voice) = (&voices[1].0[1], &voices[1].1);
    let longest = [a, b, other].map(voxcpm::Prompt::len).into_iter().max();
    let mut gpu = gpu(longest.unwrap(), SLOTS);
    let mut whole = PromptCache::new(&gpu.ctx, 0);
    let want = [a, b].map(|prompt| generate(&mut gpu, &mut whole, prompt, voice, 0));
    generate(&mut gpu, &mut whole, other, other_voice, 1);
    generate(&mut gpu, &mut whole, b, voice, 3);

    // Whole, into a slot no prompt has been run into.
    let before = caches(&mut gpu);
    let got = generate(&mut gpu, &mut whole, a, voice, 2);
    assert_eq!(got, want[0], "a prompt run whole into slot 2");
    assert_eq!(changed(&before, &caches(&mut gpu)), [2]);

    // From what a cache keeps of the voice from one slot, into another that holds another
    // prompt.
    for (kept, into) in [(0, 2), (2, 1)] {
        let mut cache = PromptCache::new(&gpu.ctx, BUDGET);
        generate(&mut gpu, &mut cache, a, voice, kept);
        let before = caches(&mut gpu);
        let got = generate(&mut gpu, &mut cache, b, voice, into);
        assert!(got.prefilled < b.len(), "{got:?}");
        let differing = got.differing(&want[1]);
        assert_eq!(
            differing, 0,
            "a prompt kept from slot {kept} run into slot {into}"
        );
        assert_eq!(
            changed(&before, &caches(&mut gpu)),
            [into],
            "from {kept} into {into}"
        );
    }

    // Defect: the prompt run into another slot than the one its patches are sampled in, which
    // holds another prompt in the voice.
    let before = caches(&mut gpu);
    let positions = Prompt {
        tokens: &a.tokens,
        patches: &a.patches,
        audio: &a.audio,
    };
    gpu.prompter
        .run(&mut gpu.step, 1, &gpu.checkpoint, positions)
        .unwrap_or_else(|e| panic!("{e}"));
    let stray = sample(&mut gpu, a, 3, a.len());
    let differing = stray.differing(&want[0]);
    let touched = changed(&before, &caches(&mut gpu));
    eprintln!(
        "the prompt in slot 1 and its patches in slot 3: {differing} values apart, slots \
         {touched:?} changed"
    );
    assert!(
        differing > 0 && touched != [3],
        "a prompt run into another slot passes"
    );
    assert_no_validation_messages(&gpu.ctx);
}
