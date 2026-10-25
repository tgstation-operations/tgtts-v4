//! VoxCPM2's own weights against what upstream computes with them, as
//! fixtures/audiovae_checkpoint.py records it.

use audiovae::{AudioVaeConfig, Decoder, Encoder, Weights};
use burn::tensor::FloatDType;
use testkit::{Fixture, assert_close, checkpoint_dir, device, fixture};

fn checkpoint() -> (AudioVaeConfig, Weights) {
    let dir = checkpoint_dir();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let config = serde_json::from_value(config["audio_vae_config"].clone()).unwrap();
    (config, Weights::open(dir.join("audiovae.pth")).unwrap())
}

fn golden() -> Fixture {
    fixture!("audiovae_checkpoint")
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn decodes_a_speaker_latent_as_upstream() {
    let device = device();
    let (config, weights) = checkpoint();
    let decoder = Decoder::load(&config, &weights, FloatDType::F32, &device).unwrap();
    let golden = golden();
    let audio = decoder.decode(golden.tensor("input.decode", &device));
    // Upstream's own f32 result is 2.5e-6 from its f64 one.
    assert_close(audio, golden.tensor("output.decode", &device), 1e-5, 0.0);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn encodes_speech_as_upstream() {
    let device = device();
    let (config, weights) = checkpoint();
    let encoder = Encoder::load(&config, &weights, FloatDType::F32, &device).unwrap();
    let golden = golden();
    let latents = encoder.encode(golden.tensor("input.encode", &device));
    // Upstream's own f32 result is 6.0e-6 from its f64 one.
    assert_close(latents, golden.tensor("output.encode", &device), 3e-5, 0.0);
}

/// The golden needs the checkpoint to make, so a run of every generator keeps it as it is:
/// this refuses one that run left behind.
#[test]
fn golden_was_made_as_the_other_fixtures_were() {
    let fixture = golden();
    let golden = fixture.metadata();
    let regenerate = "regenerate it with VOXCPM2_DIR and VOICES_DIR set: \
                      nix run .#fixtures audiovae_checkpoint";
    let fixtures = fixture!("audiovae");
    for key in ["upstream", "torch"] {
        assert_eq!(
            golden[key],
            fixtures.metadata()[key],
            "{key} differs; {regenerate}"
        );
    }
    let generator = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/audiovae_checkpoint.py"
    );
    let generator = std::fs::read(generator).unwrap();
    assert_eq!(
        golden["generator_crc32"],
        format!("{:08x}", crc32fast::hash(&generator)),
        "fixtures/audiovae_checkpoint.py changed after making the golden; {regenerate}"
    );
}
