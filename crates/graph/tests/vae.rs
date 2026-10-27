//! The audio VAE's decoder on the GPU against the audiovae crate's on burn's CPU backend, on
//! VoxCPM2's real weights over a real voice's latents. Built with the `device` feature, and run
//! with `VOXCPM2_DIR` naming the checkpoint, `VOICES_DIR` a voices directory and `GLSL_KERNELS`
//! the compiled kernels, as `nix run .#device-tests` sets it.
#![cfg(feature = "device")]

use std::{fs, path::PathBuf, time::Instant};

use audiovae::{AudioVaeConfig, Decoder, Weights};
use burn::tensor::{FloatDType, Tensor, TensorData};
use graph::{
    kernels::Kernels,
    vae::{VaeConfig, VaeDecoder, VaeShape},
};
use testkit::{checkpoint_dir, device};
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
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let path = voices::latent_path(&PathBuf::from(dir), "Adachi");
    let voice = Latent::read(&path).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        voice.frames.len() >= 4 + frames,
        "{} frames",
        voice.frames.len()
    );
    voice.frames[4..4 + frames].as_flattened().to_vec()
}

/// What the audiovae crate decodes `latents` into on the CPU, whole, with the decoder in `dtype`.
fn cpu(latents: &[f32], dtype: FloatDType) -> Vec<f32> {
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

/// The decoder on the GPU in chunks of at most `chunk` frames.
fn gpu(ctx: &Context, chunk: usize, timed: bool) -> VaeDecoder {
    let start = Instant::now();
    let mut kernels = Kernels::from_env(ctx).unwrap_or_else(|e| panic!("{e}"));
    let config: VaeConfig = serde_json::from_value(config()).unwrap();
    let shape = VaeShape::new(&config, chunk).unwrap_or_else(|e| panic!("{e}"));
    let mut decoder =
        VaeDecoder::new(ctx, &mut kernels, &shape, timed).unwrap_or_else(|e| panic!("{e}"));
    let weights = weights();
    decoder
        .load(|name| {
            weights
                .floats(&format!("decoder.{name}"))
                .map_err(|e| graph::Error::Checkpoint(e.to_string()))
        })
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

/// `latents` pushed through `decoder` in chunks of `lengths` frames, and their audio joined.
fn streamed(decoder: &mut VaeDecoder, latents: &[f32], lengths: &[usize]) -> Vec<f32> {
    let mut audio = Vec::new();
    let mut start = 0;
    for &length in lengths {
        let chunk = &latents[start * LATENT..(start + length) * LATENT];
        audio.extend(decoder.push(chunk).unwrap_or_else(|e| panic!("{e}")));
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
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn frames_one_at_a_time_decode_as_audiovae_does_on_the_cpu() {
    let frames = 3;
    let latents = latents(frames);
    let (want, burn_f16) = (
        cpu(&latents, FloatDType::F32),
        cpu(&latents, FloatDType::F16),
    );
    let ctx = context();
    let mut decoder = gpu(&ctx, 1, true);
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
    let mut decoder = gpu(&ctx, 25, true);
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
    let whole = streamed(&mut gpu(&ctx, frames, false), &latents, &[frames]);
    let mut decoder = gpu(&ctx, 25, false);
    let mut chunkings = vec![vec![1; frames], vec![25, 23]];
    chunkings.extend((0..3).map(|seed| lengths(frames, 25, seed)));
    for lengths in &chunkings {
        decoder.reset().unwrap();
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
        decoder.reset().unwrap();
        let start = chunkings[1][..i].iter().sum::<usize>() * LATENT;
        forgetful.extend(
            decoder
                .push(&latents[start..start + length * LATENT])
                .unwrap(),
        );
    }
    let d = distance(&forgetful, &whole);
    assert!(d.max_abs > 1e-2, "chunks that start afresh pass: {d:?}");
    assert_no_validation_messages(&ctx);
}
