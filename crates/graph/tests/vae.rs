//! The audio VAE's decoder on the GPU against the audiovae crate's on burn's CPU backend, on
//! VoxCPM2's real weights over a real voice's latents; and decoding several streams, each bit for
//! bit as it does alone. Built with the `device` feature, and run with `VOXCPM2_DIR` naming the
//! checkpoint, `VOICES_DIR` a voices directory and `GLSL_KERNELS` the compiled kernels, as `nix
//! run .#device-tests` sets it.
//!
//! The CPU's part is kept under the directory `ORACLE_DIR` names, when it is set, and read from
//! there by later runs: `keeps_the_oracles` runs it alone, touching no GPU.
#![cfg(feature = "device")]

use std::{collections::VecDeque, fs, path::PathBuf, time::Instant};

use audiovae::{AudioVaeConfig, Decoder, Weights};
use burn::tensor::{FloatDType, Tensor, TensorData};
use graph::{
    kernels::Kernels,
    pth::Pth,
    vae::{VaeConfig, VaeDecoder, VaeShape},
};
use testkit::{checkpoint_dir, device, kept, oracle_dir};
use voices::Latent;
use vulkan::Context;

const LATENT: usize = 64;

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

fn config() -> serde_json::Value {
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json).unwrap();
    json["audio_vae_config"].clone()
}

fn weights() -> Weights {
    Weights::open(checkpoint_dir().join("audiovae.pth")).unwrap_or_else(|e| panic!("{e}"))
}

/// `frames` frames of a voice's latents, `[frames][64]`: speech, past the quiet start.
fn latents(frames: usize) -> Vec<f32> {
    latents_from(4, frames)
}

/// `frames` frames of the voice's latents from its frame `first`.
fn latents_from(first: usize, frames: usize) -> Vec<f32> {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let path = voices::latent_path(&PathBuf::from(dir), "Adachi");
    let voice = Latent::read(&path).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        voice.frames.len() >= first + frames,
        "{} frames",
        voice.frames.len()
    );
    voice.frames[first..first + frames].as_flattened().to_vec()
}

/// [`compute_oracle`]'s, kept by the frames and the dtype alone: the tests decode [`latents`]'.
fn cpu(latents: &[f32], dtype: FloatDType) -> Vec<f32> {
    let frames = latents.len() / LATENT;
    kept(&format!("vae-{frames}-{dtype:?}"), || {
        compute_oracle(latents, dtype)
    })
}

/// What the audiovae crate decodes `latents` into on the CPU, whole, with the decoder in `dtype`.
fn compute_oracle(latents: &[f32], dtype: FloatDType) -> Vec<f32> {
    let start = Instant::now();
    let config: AudioVaeConfig = serde_json::from_value(config()).unwrap();
    let device = device();
    let decoder = Decoder::load(&config, &weights(), dtype, &device).unwrap();
    let frames = latents.len() / LATENT;
    let x = Tensor::<2>::from_data(TensorData::new(latents.to_vec(), [frames, LATENT]), &device);
    let audio = decoder.decode(x.transpose().unsqueeze_dim(0));
    let audio = audio
        .cast(FloatDType::F32)
        .into_data()
        .iter::<f32>()
        .collect();
    eprintln!(
        "{frames} frames on the CPU in {dtype:?}: {:.1} s",
        start.elapsed().as_secs_f64()
    );
    audio
}

/// The decoder on the GPU of `streams` streams in chunks of at most `chunk` frames.
fn gpu(ctx: &Context, chunk: usize, streams: usize, timed: bool) -> VaeDecoder {
    let start = Instant::now();
    let mut kernels = Kernels::from_env(ctx).unwrap_or_else(|e| panic!("{e}"));
    let config: VaeConfig = serde_json::from_value(config()).unwrap();
    let shape = VaeShape::new(&config, chunk, streams).unwrap_or_else(|e| panic!("{e}"));
    let mut decoder =
        VaeDecoder::new(ctx, &mut kernels, &shape, timed).unwrap_or_else(|e| panic!("{e}"));
    let weights =
        Pth::open(checkpoint_dir().join("audiovae.pth")).unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(|name| weights.floats(&format!("decoder.{name}")))
        .unwrap_or_else(|e| panic!("{e}"));
    let step = decoder.graph_step();
    eprintln!(
        "chunks of {chunk} recorded and loaded: {:.1} s, {} dispatches, {} barriers, {} \
         pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        step.dispatches(),
        step.barriers(),
        kernels.len(),
        step.arena_bytes() / 1024
    );
    decoder
}

/// `latents` pushed through `decoder`'s first stream in chunks of `lengths` frames, and their
/// audio joined.
fn streamed(decoder: &mut VaeDecoder, latents: &[f32], lengths: &[usize]) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut start = 0;
    for &length in lengths {
        let chunk = &latents[start * LATENT..(start + length) * LATENT];
        audio.extend(decoder.push(0, chunk).unwrap_or_else(|e| panic!("{e}")));
        start += length;
    }
    audio
}

/// The slowest dispatch of the last chunk, and the chunk's GPU time, in milliseconds.
fn dispatch_times(decoder: &VaeDecoder) -> (String, f64, f64) {
    let timings = decoder.timings().unwrap().expect("recorded timed");
    let ms = |t: &vulkan::Timing| t.nanoseconds / 1e6;
    let total = timings.iter().map(|(_, t)| ms(t)).sum();
    let (label, slowest) = timings
        .iter()
        .max_by(|a, b| ms(&a.1).total_cmp(&ms(&b.1)))
        .map(|(label, t)| (label.to_string(), ms(t)))
        .unwrap();
    (label, slowest, total)
}

/// How far one run's audio is from another's.
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
            "max |d| {:.2e}, relative RMS {:.2e} ({:.1} dB)",
            self.max_abs,
            self.relative_rms,
            -20.0 * self.relative_rms.log10()
        )
    }
}

fn distance(got: &[f32], want: &[f32]) -> Distance {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut err2, mut ref2) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (f64::from(g) - f64::from(w)).abs();
        max_abs = max_abs.max(if d.is_finite() { d } else { f64::INFINITY });
        err2 += d * d;
        ref2 += f64::from(w).powi(2);
    }
    Distance {
        max_abs,
        relative_rms: (err2 / ref2).sqrt(),
    }
}

/// Checks the GPU's audio against the CPU's in f32 and burn's in f16: within `max_abs` of f32,
/// which f32 arithmetic in another order reaches and f16 storage does not, and closer to it than
/// burn's f16 is.
fn check(got: &[f32], want: &[f32], burn_f16: &[f32], max_abs: f64) {
    let (ours, burn) = (distance(got, want), distance(burn_f16, want));
    eprintln!("against f32: GPU {ours:?}; burn f16 {burn:?}");
    assert!(ours.max_abs <= max_abs, "{ours:?}");
    assert!(
        ours.relative_rms <= burn.relative_rms,
        "{ours:?} against {burn:?}"
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ORACLE_DIR a directory to keep the oracles in"]
fn keeps_the_oracles() {
    if oracle_dir().is_none() {
        eprintln!("ORACLE_DIR is not set: nothing to keep the oracles in");
        return;
    }
    for frames in [3, 50] {
        let latents = latents(frames);
        cpu(&latents, FloatDType::F32);
        cpu(&latents, FloatDType::F16);
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn frames_one_at_a_time_decode_as_audiovae_does_on_the_cpu() {
    let frames = 3;
    let latents = latents(frames);
    let (want, burn_f16) = (
        cpu(&latents, FloatDType::F32),
        cpu(&latents, FloatDType::F16),
    );
    let ctx = context();
    let mut decoder = gpu(&ctx, 1, 1, true);
    let got = streamed(&mut decoder, &latents, &vec![1; frames]);
    let (label, slowest, total) = dispatch_times(&decoder);
    eprintln!("a frame: {total:.2} ms on the GPU, the slowest dispatch {label} at {slowest:.2} ms");
    assert_no_validation_messages(&ctx);
    check(&got, &want, &burn_f16, 1e-4);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn chunks_decode_as_audiovae_does_on_the_cpu() {
    // Two chunks of the service's 25 frames.
    let frames = 50;
    let latents = latents(frames);
    let (want, burn_f16) = (
        cpu(&latents, FloatDType::F32),
        cpu(&latents, FloatDType::F16),
    );
    let ctx = context();
    let mut decoder = gpu(&ctx, 25, 1, true);
    let got = streamed(&mut decoder, &latents, &[25, 25]);
    let (label, slowest, total) = dispatch_times(&decoder);
    eprintln!(
        "25 frames: {total:.2} ms on the GPU, the slowest dispatch {label} at {slowest:.2} ms"
    );
    assert_no_validation_messages(&ctx);
    check(&got, &want, &burn_f16, 1e-4);
}

/// Lengths of 1 to `longest` frames, the last cut short, that add up to `frames`, from a seeded
/// generator, so that a rerun draws the same.
fn lengths(frames: usize, longest: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut lengths = vec![];
    let mut left = frames;
    while left > 0 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let length = (1 + (state >> 33) as usize % longest).min(left);
        lengths.push(length);
        left -= length;
    }
    lengths
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn chunks_streamed_decode_as_the_whole() {
    // Longer than the decoder reaches back: 20 frames before a sample's own.
    let frames = 48;
    let latents = latents(frames);
    let ctx = context();
    let whole = streamed(&mut gpu(&ctx, frames, 1, false), &latents, &[frames]);
    let mut decoder = gpu(&ctx, 25, 1, false);
    let mut chunkings = vec![vec![1; frames], vec![25, 23]];
    chunkings.extend((0..3).map(|seed| lengths(frames, 25, seed)));
    for lengths in &chunkings {
        decoder.reset(0).unwrap();
        let audio = streamed(&mut decoder, &latents, lengths);
        let d = distance(&audio, &whole);
        eprintln!("chunks of {lengths:?}: {d:?}");
        // The rings keep what each convolution read, in f32, so on the Radeon 8060S chunks came
        // out bit for bit the whole; a driver free to fuse differently need not.
        assert!(d.max_abs <= 1e-6, "{lengths:?}: {d:?}");
    }
    // Defect: a stream that forgets every chunk before, as if each were its first.
    let mut forgetful = Vec::new();
    for (i, &length) in chunkings[1].iter().enumerate() {
        decoder.reset(0).unwrap();
        let start = chunkings[1][..i].iter().sum::<usize>() * LATENT;
        forgetful.extend(
            decoder
                .push(0, &latents[start..start + length * LATENT])
                .unwrap(),
        );
    }
    let d = distance(&forgetful, &whole);
    assert!(d.max_abs > 1e-2, "chunks that start afresh pass: {d:?}");
    assert_no_validation_messages(&ctx);
}

/// The speech of four requests, each `[frames][64]` from its own part of the voice, and its
/// chunks' lengths; the stream each is decoded in, the fourth in the first's once the first has
/// ended.
fn requests() -> (Vec<Vec<f32>>, Vec<Vec<usize>>, [usize; 4]) {
    let parts = [(4, 48), (100, 37), (200, 60), (300, 30)];
    let speech: Vec<Vec<f32>> = parts.iter().map(|&(at, n)| latents_from(at, n)).collect();
    let chunkings = (0..parts.len())
        .map(|i| lengths(parts[i].1, 25, 10 + i as u64))
        .collect();
    (speech, chunkings, [0, 1, 2, 0])
}

/// Each request's audio as bits, its chunks pushed in `streams` in turn with the other streams',
/// a stream reset before each request in it; one chunk, `stray` (request, chunk, stream), pushed
/// into another stream than its request's.
fn interleaved(decoder: &mut VaeDecoder, stray: Option<(usize, usize, usize)>) -> Vec<Vec<u32>> {
    let (speech, chunkings, streams) = requests();
    let mut waiting: Vec<VecDeque<usize>> = vec![VecDeque::new(); 3];
    for (r, &s) in streams.iter().enumerate() {
        waiting[s].push_back(r);
    }
    let mut current: Vec<Option<usize>> = vec![None; waiting.len()];
    let mut next = vec![0; speech.len()];
    let mut audio = vec![vec![]; speech.len()];
    loop {
        let mut pushed = false;
        for stream in 0..waiting.len() {
            if current[stream].is_none() {
                current[stream] = waiting[stream].pop_front();
                if current[stream].is_some() {
                    decoder.reset(stream).unwrap_or_else(|e| panic!("{e}"));
                }
            }
            let Some(r) = current[stream] else {
                continue;
            };
            let k = next[r];
            let start = chunkings[r][..k].iter().sum::<usize>() * LATENT;
            let chunk = &speech[r][start..start + chunkings[r][k] * LATENT];
            let into = match stray {
                Some((sr, sk, other)) if (sr, sk) == (r, k) => other,
                _ => stream,
            };
            let samples = decoder.push(into, chunk).unwrap_or_else(|e| panic!("{e}"));
            audio[r].extend(samples.iter().map(|v| v.to_bits()));
            next[r] += 1;
            if next[r] == chunkings[r].len() {
                current[stream] = None;
            }
            pushed = true;
        }
        if !pushed {
            return audio;
        }
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn streams_decode_bit_for_bit_as_each_does_alone() {
    let (speech, chunkings, _) = requests();
    let ctx = context();
    let mut one = gpu(&ctx, 25, 1, false);
    let alone: Vec<Vec<u32>> = speech
        .iter()
        .zip(&chunkings)
        .map(|(latents, lengths)| {
            one.reset(0).unwrap();
            let audio = streamed(&mut one, latents, lengths);
            audio.iter().map(|v| v.to_bits()).collect()
        })
        .collect();
    drop(one);
    let mut decoder = gpu(&ctx, 25, 3, false);
    let together = interleaved(&mut decoder, None);
    for (r, (got, want)) in together.iter().zip(&alone).enumerate() {
        assert!(got == want, "request {r} is not as it is alone");
    }
    // Defect: request 1's second chunk pushed into request 2's stream, which only those two may
    // come out otherwise for.
    let crossed = interleaved(&mut decoder, Some((1, 1, 2)));
    let apart: Vec<bool> = crossed.iter().zip(&alone).map(|(g, w)| g != w).collect();
    assert_eq!(apart, [false, true, true, false]);
    assert_no_validation_messages(&ctx);
}
