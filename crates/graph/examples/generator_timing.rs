//! Times VoxCPM2's generation on the GPU at batch 1, a request at a time, as the service's
//! requests run: the prompt it builds for a line in a voice, from the request to its first patch,
//! then each patch after it and each chunk of audio, from submission to fence on the host and
//! per dispatch from GPU timestamps.
//!
//! ```text
//! cargo run --release -p graph --example generator_timing -- [--voice-patches 70]
//!     [--patches 32] [--runs 3] [--slots 1]
//! ```
//!
//! `VOXCPM2_DIR` names the checkpoint, `VOICES_DIR` a voices directory, and `GLSL_KERNELS` and
//! `VULKAN_KERNELS` the compiled kernels. The voice is the one of the median length in the
//! directory, its prompt timed whole and with its voice cut to its first `--voice-patches`
//! patches. Each prompt is generated from once to warm up, then `--runs` times, each request
//! `--patches` patches long.
//!
//! With `--slots` above 1, the generator holds that many requests at once, recorded untimed, and
//! each number of them under way from one to its slots is timed instead: the prompt with its
//! voice cut started that many times, each from a seed of its own, then their steps until they
//! end, each from the call to its return on the host, with the patch step's GPU time and the
//! chunks of audio decoded in it.

use std::{path::PathBuf, time::Instant};

use graph::{
    generator::{Generator, Settings},
    kernels::Kernels,
    prompt::Prompt as Positions,
};
use voices::Latent;
use voxcpm::{Config, Noise, Params, Prompt, Sampling, SeededNoise, Tokenizer};
use vulkan::Context;

/// A line twice as long as a median line, so that the stop head does not end a request first.
const LINE: &str = "Hello there, could someone please come to medbay? I need help. The captain \
                    is down in the bridge, and nobody has seen the doctor for an hour.";

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .map(|i| {
            args[i + 1]
                .parse()
                .unwrap_or_else(|e| panic!("{name}: {e}"))
        })
        .unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The GPU time of a run's dispatches, in milliseconds.
fn gpu_ms(timings: Option<Vec<(&str, vulkan::Timing)>>) -> f64 {
    let timings = timings.expect("recorded timed");
    timings.iter().map(|(_, t)| t.nanoseconds / 1e6).sum()
}

/// The voice of the median length in `dir`.
fn median_voice(dir: &std::path::Path) -> (String, Latent) {
    let mut voices: Vec<(usize, String)> = std::fs::read_dir(dir.join("latents"))
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().unwrap();
            let name = name.strip_suffix(".safetensors")?.to_owned();
            let latent = Latent::read(&voices::latent_path(dir, &name)).ok()?;
            Some((latent.frames.len(), name))
        })
        .collect();
    voices.sort();
    let (_, name) = voices[voices.len() / 2].clone();
    let latent = Latent::read(&voices::latent_path(dir, &name)).unwrap();
    (name, latent)
}

/// What one request took, in milliseconds.
struct Request {
    start: f64,
    prompt_gpu: f64,
    first: f64,
    first_gpu: f64,
    patches: Vec<Sampled>,
}

/// A patch's next_patch on the host and its step on the GPU, and the GPU time of the chunk of
/// audio decoded in it, if one was.
struct Sampled {
    host: f64,
    gpu: f64,
    chunk: Option<f64>,
}

fn request(generator: &mut Generator, prompt: &Prompt, voice: &[f32], patches: usize) -> Request {
    let sampling = Sampling::default();
    let positions = Positions {
        tokens: &prompt.tokens,
        patches: &prompt.patches,
        audio: &prompt.audio,
    };
    let mut noise = SeededNoise::new(0);
    let begun = Instant::now();
    generator
        .start(
            0,
            positions,
            voice,
            patches,
            sampling.temperature,
            sampling.cfg_value,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let start = begun.elapsed().as_secs_f64() * 1e3;
    let prompt_gpu = gpu_ms(generator.timings().unwrap());
    let mut out = Request {
        start,
        prompt_gpu,
        first: 0.0,
        first_gpu: 0.0,
        patches: vec![],
    };
    loop {
        let at = Instant::now();
        let stepped = generator.step(|_| noise.next(256));
        let host = at.elapsed().as_secs_f64() * 1e3;
        let stepped = stepped.unwrap_or_else(|e| panic!("{e}"));
        let Some((_, patch)) = stepped.into_iter().next() else {
            break;
        };
        let patch = patch.unwrap_or_else(|e| panic!("{e}"));
        let gpu = gpu_ms(generator.timings().unwrap());
        let audio = generator.take_audio(0);
        let chunk = (!audio.is_empty()).then(|| gpu_ms(generator.audio_timings().unwrap()));
        if out.patches.is_empty() {
            out.first = begun.elapsed().as_secs_f64() * 1e3;
            out.first_gpu = gpu;
        }
        out.patches.push(Sampled { host, gpu, chunk });
        if patch.last {
            break;
        }
    }
    generator.finish(0).unwrap_or_else(|e| panic!("{e}"));
    out
}

/// Each number of requests under way from one to `generator`'s slots, as the module's
/// documentation says: `prompt` in `voice`, up to `patches` patches each.
fn batched(generator: &mut Generator, prompt: &Prompt, voice: &[f32], patches: usize) {
    let sampling = Sampling::default();
    let positions = Positions {
        tokens: &prompt.tokens,
        patches: &prompt.patches,
        audio: &prompt.audio,
    };
    let ms = |since: Instant| since.elapsed().as_secs_f64() * 1e3;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    for under_way in 1..=generator.slots() {
        let begun = Instant::now();
        for slot in 0..under_way {
            generator
                .start(
                    slot,
                    positions,
                    voice,
                    patches,
                    sampling.temperature,
                    sampling.cfg_value,
                )
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let started = ms(begun);
        let mut noise: Vec<SeededNoise> = (0..under_way as u64).map(SeededNoise::new).collect();
        let (mut host, mut quiet, mut gpu, mut chunks) = (vec![], vec![], vec![], 0);
        loop {
            let at = Instant::now();
            let stepped = generator
                .step(|slot| noise[slot].next(256))
                .unwrap_or_else(|e| panic!("{e}"));
            let took = ms(at);
            if stepped.is_empty() {
                break;
            }
            let decoded = stepped
                .iter()
                .filter(|(slot, _)| !generator.take_audio(*slot).is_empty())
                .count();
            if stepped.len() == under_way {
                host.push(took);
                gpu.push(generator.elapsed().unwrap().nanoseconds / 1e6);
                chunks += decoded;
                if decoded == 0 {
                    quiet.push(took);
                }
            }
            for (slot, patch) in stepped {
                if patch.unwrap_or_else(|e| panic!("{e}")).last {
                    generator.finish(slot).unwrap_or_else(|e| panic!("{e}"));
                }
            }
        }
        let step = mean(&host);
        println!(
            "{under_way} under way: {} steps with all of them, each {step:.1} ms on the host, \
             {:.1} without a chunk of audio, the patch step {:.1} ms on the GPU; {:.2} chunks a \
             step; {:.1} steps and {:.1} patches a second; the prompts {:.0} ms each",
            host.len(),
            median(quiet),
            median(gpu),
            chunks as f64 / host.len().max(1) as f64,
            1e3 / step,
            under_way as f64 * 1e3 / step,
            started / under_way as f64,
        );
    }
}

fn main() {
    let patches = arg("--patches", 32);
    let runs = arg("--runs", 3);
    let voice_patches = arg("--voice-patches", 70);
    let slots = arg("--slots", 1);
    let dir = PathBuf::from(std::env::var_os("VOXCPM2_DIR").expect("VOXCPM2_DIR"));
    let voices = PathBuf::from(std::env::var_os("VOICES_DIR").expect("VOICES_DIR"));
    let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap())
        .unwrap_or_else(|e| panic!("{e}"));
    let tokenizer =
        Tokenizer::from_json(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap())
            .unwrap_or_else(|e| panic!("{e}"));
    let (name, voice) = median_voice(&voices);
    let frames = voice.frames.as_flattened();
    let tokens = tokenizer.encode(&format!("{}{LINE}", voice.text));
    let patch = config.patch_size * config.feat_dim;
    let cut = &frames[..(voice_patches * patch).min(frames.len())];
    let prompts = [cut, frames].map(|voice| Prompt::cloning(&tokens, voice, &config).unwrap());
    let longest = prompts.iter().map(Prompt::len).max().unwrap();

    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    eprintln!(
        "{} ({}), validation {}; voice {name} of {} patches, {} tokens of text",
        report.name,
        report.driver_info,
        if ctx.validating() { "on" } else { "off" },
        frames.len() / patch,
        tokens.len(),
    );
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let weights = audiovae::Weights::open(dir.join("audiovae.pth")).unwrap();
    let settings = Settings {
        capacity: (longest + patches).next_multiple_of(32),
        max_rows: longest.next_multiple_of(graph::prefill::TILE),
        timesteps: Sampling::default().timesteps,
        chunk_frames: Params::default().decode_chunk_frames.get(),
        timed: slots == 1,
        prompt_cache_bytes: 0,
        subgroup: None,
        slots,
    };
    let loaded = Instant::now();
    let parameter = |name: &str| {
        weights
            .floats(&format!("decoder.{name}"))
            .map_err(|e| graph::Error::Checkpoint(e.to_string()))
    };
    let mut generator = Generator::new(&ctx, &mut kernels, &dir, &settings, parameter)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "recorded and loaded in {:.1} s, {settings:?}",
        loaded.elapsed().as_secs_f64()
    );
    if slots > 1 {
        batched(&mut generator, &prompts[0], cut, patches);
        return;
    }

    for (prompt, voice) in prompts.iter().zip([cut, frames]) {
        request(&mut generator, prompt, voice, patches);
        let requests: Vec<Request> = (0..runs)
            .map(|_| request(&mut generator, prompt, voice, patches))
            .collect();
        let of = |f: &dyn Fn(&Request) -> f64| median(requests.iter().map(f).collect());
        let all = |f: &dyn Fn(&Sampled) -> Option<f64>| {
            median(
                requests
                    .iter()
                    .flat_map(|r| r.patches[1..].iter().filter_map(f))
                    .collect(),
            )
        };
        let sampled = requests[0].patches.len();
        println!(
            "prompt of {} positions, {} of them the voice's patches: the first patch {:.1} ms \
             after the request; the prompt {:.1} ms, {:.1} of it the prompt's run on the GPU \
             and the rest mostly the voice's patches embedded; the first patch's step {:.1} ms \
             on the GPU",
            prompt.len(),
            2 * (voice.len() / patch),
            of(&|r| r.first),
            of(&|r| r.start),
            of(&|r| r.prompt_gpu),
            of(&|r| r.first_gpu),
        );
        println!(
            "  {sampled} patches: each after the first {:.1} ms from submission to fence and \
             {:.1} on the GPU; {} with a chunk of audio decoded, each {:.1} ms from submission to \
             fence and the chunk {:.1} on the GPU (medians of {runs} requests)",
            all(&|p| p.chunk.is_none().then_some(p.host)),
            all(&|p| Some(p.gpu)),
            requests[0]
                .patches
                .iter()
                .filter(|p| p.chunk.is_some())
                .count(),
            all(&|p| p.chunk.map(|_| p.host)),
            all(&|p| p.chunk),
        );
    }
}
