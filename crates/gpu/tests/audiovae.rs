//! audiovae's fixtures and real-weight golden on the GPU, in f32.
//!
//! The decoder runs in f32 on the GPU too: in f16 the first of its transposed convolutions, 2048
//! channels to 1024, sums in f16, which took the real golden's decode from 116 dB of upstream's
//! to 46 dB, where the CPU in f16 reaches 56 dB.
#![cfg(any(feature = "vulkan", feature = "cuda"))]

mod common;

use audiovae::{AudioVaeConfig, Decoder, Encoder, Weights};
use burn::{store::SafetensorsStore, tensor::FloatDType};
use common::fixture;
use testkit::{Fixture, assert_close, checkpoint_dir};

/// The config fixtures/audiovae.py builds its decoder and encoder from.
fn tiny() -> AudioVaeConfig {
    AudioVaeConfig {
        encoder_dim: 2,
        encoder_rates: vec![2, 5],
        latent_dim: 4,
        decoder_dim: 32,
        decoder_rates: vec![3, 2, 2],
        out_sample_rate: 24000,
        ..AudioVaeConfig::default()
    }
}

fn tiny_weights(fixture: &Fixture) -> Weights {
    Weights::from_store(&mut SafetensorsStore::from_file(fixture.path())).unwrap()
}

#[test]
fn decodes_as_upstream() {
    let (fixture, device) = (fixture("audiovae", "audiovae"), gpu::device());
    let weights = tiny_weights(&fixture);
    let decoder = Decoder::load(&tiny(), &weights, FloatDType::F32, &device).unwrap();
    let audio = decoder.decode(fixture.tensor("input.decode.z", &device));
    // As on the CPU: upstream's own f32 result is 2e-5 from its f64 one, and this came within
    // 4.1e-5 of upstream's, the random weights amplifying rounding through thirteen convolutions.
    assert_close(audio, fixture.tensor("output.decode", &device), 1e-4, 0.0);
}

#[test]
fn encodes_as_upstream() {
    let (fixture, device) = (fixture("audiovae", "audiovae"), gpu::device());
    let weights = tiny_weights(&fixture);
    let encoder = Encoder::load(&tiny(), &weights, FloatDType::F32, &device).unwrap();
    let latents = encoder.encode(fixture.tensor("input.encode.audio", &device));
    // Within 3.0e-6 of upstream's, which is 2.4e-6 from its own f64 result.
    let expected = fixture.tensor("output.encode", &device);
    assert_close(latents, expected, 1e-5, 1e-5);
}

fn checkpoint() -> (AudioVaeConfig, Weights) {
    let dir = checkpoint_dir();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let config = serde_json::from_value(config["audio_vae_config"].clone()).unwrap();
    (config, Weights::open(dir.join("audiovae.pth")).unwrap())
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn decodes_a_speaker_latent_as_upstream() {
    let device = gpu::device();
    let (config, weights) = checkpoint();
    let decoder = Decoder::load(&config, &weights, FloatDType::F32, &device).unwrap();
    let golden = fixture("audiovae", "audiovae_checkpoint");
    let audio = decoder.decode(golden.tensor("input.decode", &device));
    // Within 6.0e-6 of upstream's, which is 2.5e-6 from its own f64 result.
    assert_close(audio, golden.tensor("output.decode", &device), 1e-5, 0.0);
}

/// The product encodes voices offline, on the CPU; this checks the GPU's convolutions on a
/// third of a second of speech.
#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn encodes_speech_as_upstream() {
    let device = gpu::device();
    let (config, weights) = checkpoint();
    let encoder = Encoder::load(&config, &weights, FloatDType::F32, &device).unwrap();
    let golden = fixture("audiovae", "audiovae_checkpoint");
    let latents = encoder.encode(golden.tensor("input.encode", &device));
    // Within 8.3e-6 of upstream's, which is 6.0e-6 from its own f64 result.
    assert_close(latents, golden.tensor("output.encode", &device), 3e-5, 0.0);
}
