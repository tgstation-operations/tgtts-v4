//! The generator on the GPU for several requests at once, on the real weights: requests admitted
//! into the slots free as they come, between steps, and finished as they stop, each in a slot of
//! the caches and a stream of the audio VAE of its own. Each comes out bit for bit as it does
//! alone, its patches, their stop logits and its audio, whatever joins and leaves beside it.
//! Built with the `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint,
//! `VOICES_DIR` a voices directory and `GLSL_KERNELS` and `VULKAN_KERNELS` the compiled kernels,
//! as `nix run .#device-tests` sets the last two.
#![cfg(feature = "device")]

use std::{collections::VecDeque, fs, time::Instant};

use graph::{
    generator::{Generator, Settings},
    kernels::Kernels,
    prefill::TILE,
    prompt::Prompt as Positions,
    pth::Pth,
};
use testkit::checkpoint_dir;
use voices::Voices;
use voxcpm::{Config, Noise, Params, Prompt, Sampling, SeededNoise, Tokenizer};
use vulkan::Context;

/// The requests the generator generates at once.
const SLOTS: usize = 4;

/// What the prompt cache keeps, as the service's default.
const PROMPT_CACHE_BYTES: usize = 512 << 20;

/// A request: a line in one of two voices from a seed's noise, up to `max_patches` patches,
/// arriving before step `arrives`.
struct Request {
    voice: usize,
    line: &'static str,
    seed: u64,
    max_patches: usize,
    arrives: usize,
}

const fn request(
    voice: usize,
    line: &'static str,
    seed: u64,
    max_patches: usize,
    arrives: usize,
) -> Request {
    Request {
        voice,
        line,
        seed,
        max_patches,
        arrives,
    }
}

/// Two at once, then four, every slot taken, while a fifth waits for one to free; two of them
/// ended at their most patches, the rest where the stop head says; and a sixth alone at the end.
const REQUESTS: [Request; 6] = [
    request(0, "Hello, crew.", 1, 60, 0),
    request(1, "Clown in the armory.", 2, 6, 0),
    request(0, "Someone set the kitchen on fire.", 3, 60, 2),
    request(1, "Security, report to the bridge.", 4, 60, 3),
    request(0, "Clown in the armory.", 5, 9, 4),
    request(1, "Hello, crew.", 6, 60, 30),
];

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

/// The prompt the service builds for each of [`REQUESTS`], in the first two voices of
/// `VOICES_DIR`, and the voice's latent frames.
fn prompts() -> Vec<(Prompt, Vec<f32>)> {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let voices = Voices::open(dir).unwrap_or_else(|e| panic!("{e}"));
    let dir = checkpoint_dir();
    let config = Config::from_json(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = fs::read_to_string(dir.join("tokenizer.json")).unwrap();
    let tokenizer = Tokenizer::from_json(&tokenizer).unwrap();
    let names: Vec<&str> = voices.names().take(2).collect();
    REQUESTS
        .iter()
        .map(|r| {
            let voice = voices
                .latent(names[r.voice])
                .unwrap_or_else(|e| panic!("{e}"));
            let frames = voice.frames.as_flattened().to_vec();
            let tokens = tokenizer.encode(&format!("{}{}", voice.text, r.line));
            let prompt = Prompt::cloning(&tokens, &frames, &config).unwrap();
            (prompt, frames)
        })
        .collect()
}

/// The generator of [`SLOTS`] slots for `prompts`, and the GPU it runs on.
fn generator(prompts: &[(Prompt, Vec<f32>)]) -> (Context, Generator) {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let weights = Pth::open(checkpoint_dir().join("audiovae.pth")).unwrap();
    let most = REQUESTS.iter().map(|r| r.max_patches).max().unwrap();
    let longest = prompts.iter().map(|(p, _)| p.len()).max().unwrap();
    let settings = Settings {
        capacity: (longest + most).next_multiple_of(32),
        max_rows: 2 * TILE,
        timesteps: Sampling::default().timesteps,
        chunk_frames: Params::default().decode_chunk_frames.get(),
        timed: false,
        prompt_cache_bytes: PROMPT_CACHE_BYTES,
        subgroup: None,
        slots: SLOTS,
    };
    let start = Instant::now();
    let parameter = |name: &str| weights.floats(&format!("decoder.{name}"));
    let generator = Generator::new(&ctx, &mut kernels, &checkpoint_dir(), &settings, parameter)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "generator of {SLOTS} slots recorded and loaded: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    (ctx, generator)
}

/// What a request gave, as bits: each patch's frames and the stop logits beside it, and its
/// audio.
#[derive(Default, PartialEq)]
struct Spoken {
    patches: Vec<Vec<u32>>,
    audio: Vec<u32>,
}

fn bits(values: &[f32]) -> impl Iterator<Item = u32> + '_ {
    values.iter().map(|v| v.to_bits())
}

/// Starts request `r` in `slot`.
fn start(generator: &mut Generator, prompts: &[(Prompt, Vec<f32>)], r: usize, slot: usize) {
    let (prompt, voice) = &prompts[r];
    let positions = Positions {
        tokens: &prompt.tokens,
        patches: &prompt.patches,
        audio: &prompt.audio,
    };
    let sampling = Sampling::default();
    generator
        .start(
            slot,
            positions,
            voice,
            REQUESTS[r].max_patches,
            sampling.temperature,
            sampling.cfg_value,
        )
        .unwrap_or_else(|e| panic!("{e}"));
}

/// [`REQUESTS`] on `generator`, each started as it arrives in the free slot of the lowest
/// index, or as one frees in the order they came; and ended once it has its last patch. At step
/// `swap.0`, requests `swap.1` and `swap.2` sample their patches from each other's noise. Gives
/// what each gave, and the requests under way at each step.
fn together(
    generator: &mut Generator,
    prompts: &[(Prompt, Vec<f32>)],
    swap: Option<(usize, usize, usize)>,
) -> (Vec<Spoken>, Vec<usize>) {
    let patch = 64 * 4;
    let mut noise: Vec<SeededNoise> = REQUESTS.iter().map(|r| SeededNoise::new(r.seed)).collect();
    let mut spoken: Vec<Spoken> = REQUESTS.iter().map(|_| Spoken::default()).collect();
    let mut slots: Vec<Option<usize>> = vec![None; SLOTS];
    let mut waiting = VecDeque::new();
    let (mut ended, mut under_way, mut waited) = (0, vec![], false);
    for t in 0.. {
        if ended == REQUESTS.len() {
            break;
        }
        waiting.extend((0..REQUESTS.len()).filter(|&r| REQUESTS[r].arrives == t));
        for (slot, held) in slots.iter_mut().enumerate() {
            if held.is_none()
                && let Some(r) = waiting.pop_front()
            {
                start(generator, prompts, r, slot);
                *held = Some(r);
                waited |= t > REQUESTS[r].arrives;
            }
        }
        under_way.push(slots.iter().flatten().count());
        let draw = |slot: usize| {
            let r = slots[slot].expect("a request under way");
            let r = match swap {
                Some((at, a, b)) if at == t && r == a => b,
                Some((at, a, b)) if at == t && r == b => a,
                _ => r,
            };
            noise[r].next(patch)
        };
        let stepped = generator.step(draw).unwrap_or_else(|e| panic!("{e}"));
        for (slot, patch) in stepped {
            let r = slots[slot].expect("a request under way");
            let patch = patch.unwrap_or_else(|e| panic!("request {r}: {e}"));
            let logits = patch.stop_logits;
            spoken[r]
                .patches
                .push(bits(&patch.frames).chain(bits(&logits)).collect());
            spoken[r].audio.extend(bits(&generator.take_audio(slot)));
            if patch.last {
                let rest = generator.finish(slot).unwrap_or_else(|e| panic!("{e}"));
                spoken[r].audio.extend(bits(&rest));
                slots[slot] = None;
                ended += 1;
            }
        }
    }
    assert!(waited, "no request waited for a slot: {under_way:?}");
    (spoken, under_way)
}

/// Each of [`REQUESTS`] alone on `generator`, in its first slot.
fn alone(generator: &mut Generator, prompts: &[(Prompt, Vec<f32>)]) -> Vec<Spoken> {
    (0..REQUESTS.len())
        .map(|r| {
            start(generator, prompts, r, 0);
            let mut noise = SeededNoise::new(REQUESTS[r].seed);
            let mut spoken = Spoken::default();
            loop {
                let stepped = generator
                    .step(|_| noise.next(64 * 4))
                    .unwrap_or_else(|e| panic!("{e}"));
                let [(0, patch)] = &stepped[..] else {
                    panic!("request {r} alone: {} stepped", stepped.len());
                };
                let patch = patch.as_ref().unwrap_or_else(|e| panic!("{e}"));
                let logits = patch.stop_logits;
                spoken
                    .patches
                    .push(bits(&patch.frames).chain(bits(&logits)).collect());
                spoken.audio.extend(bits(&generator.take_audio(0)));
                if patch.last {
                    break;
                }
            }
            let rest = generator.finish(0).unwrap_or_else(|e| panic!("{e}"));
            spoken.audio.extend(bits(&rest));
            spoken
        })
        .collect()
}

/// The first patch of each request at which `got` is apart from `want`, its count where only its
/// audio is.
fn first_apart(got: &[Spoken], want: &[Spoken]) -> Vec<Option<usize>> {
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let patches = (0..w.patches.len()).find(|&i| g.patches.get(i) != Some(&w.patches[i]));
            let audio = (g.audio != w.audio).then_some(w.patches.len());
            patches.or(audio)
        })
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn requests_that_come_and_go_speak_bit_for_bit_as_they_do_alone() {
    let prompts = prompts();
    let (ctx, mut generator) = generator(&prompts);
    let start = Instant::now();
    let want = alone(&mut generator, &prompts);
    let lengths: Vec<usize> = want.iter().map(|s| s.patches.len()).collect();
    eprintln!(
        "alone: {lengths:?} patches, {:.1} s",
        start.elapsed().as_secs_f64()
    );
    // Some end at their most patches, and some where the stop head says.
    let most = REQUESTS.iter().map(|r| r.max_patches);
    let at_most: Vec<bool> = lengths.iter().zip(most).map(|(&n, m)| n == m).collect();
    assert!(
        at_most.contains(&true) && at_most.contains(&false),
        "{lengths:?}"
    );

    let start = Instant::now();
    let (got, under_way) = together(&mut generator, &prompts, None);
    eprintln!(
        "together: under way at each step {under_way:?}, {:.1} s",
        start.elapsed().as_secs_f64()
    );
    // Every slot taken at once, and rows spare beside fewer.
    assert!(under_way.contains(&SLOTS) && under_way.contains(&3) && under_way.contains(&1));
    let apart = first_apart(&got, &want);
    assert!(
        apart.iter().all(Option::is_none),
        "the first patch of each request apart from it alone: {apart:?}"
    );

    // Defect: requests 0 and 2 sample a patch from each other's noise, which only they may come
    // out otherwise for.
    let (crossed, _) = together(&mut generator, &prompts, Some((3, 0, 2)));
    let apart = first_apart(&crossed, &want);
    eprintln!("with a patch's noise crossed, first apart at {apart:?}");
    let crossed: Vec<bool> = apart.iter().map(Option::is_some).collect();
    assert_eq!(
        crossed,
        [true, false, true, false, false, false],
        "{apart:?}"
    );
    assert_no_validation_messages(&ctx);
}
