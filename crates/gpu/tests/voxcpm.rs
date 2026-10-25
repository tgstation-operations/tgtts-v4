//! voxcpm on the real weights on the GPU, against what upstream computes with them, as
//! fixtures/voxcpm_checkpoint.py records it: in f32, and in the dtypes the GPU serves in.
#![cfg(feature = "vulkan")]

mod common;
#[path = "../../voxcpm/tests/recorded/mod.rs"]
mod recorded;

use std::path::PathBuf;

use burn::tensor::{FloatDType, Tensor};
use common::fixture;
use recorded::Recorded;
use testkit::{Fixture, assert_close, checkpoint_dir};
use voices::Latent;
use voxcpm::{Dtypes, Params, TextToSpeech};

/// How far what generation gives in some dtypes may be from upstream's in f32.
struct Precision {
    dtypes: Dtypes,
    /// A patch sampled after upstream's before it.
    patch: f32,
    /// The stop head's logits.
    stop_logits: f32,
    /// The audio of the first two patches.
    audio: f32,
}

// As on the CPU, whose test says why: given upstream's patches, each patch came within 2.3e-3
// of upstream's and the stop logits within 3.8e-6, and the first two patches decoded within
// 1.1e-5 of its audio.
const F32: Precision = Precision {
    dtypes: Dtypes {
        lm: FloatDType::F32,
        local_encoder: FloatDType::F32,
        local_dit: FloatDType::F32,
        audio_decoder: FloatDType::F32,
    },
    patch: 1e-2,
    stop_logits: 1e-4,
    audio: 1e-4,
};

// f16 but for the audio decoder (tests/audiovae.rs says why). Over two runs, given upstream's
// patches, each came within 4.0e-2 of upstream's but the two whose quantizer levels sit nearest a
// boundary, the fifteenth (the CPU's test tells of it) at 3.8e-1 and the eighteenth at 1.9e-1,
// and the stop logits within 8.1e-2; the first two patches decoded within 6.3e-4 of upstream's
// audio. The dtypes the Python service ran in, bf16 for the LMs and the DiT and f16 for the
// decoder, came on the CPU within 2.0e-1 but the fifteenth at 1.2, 1.6e-1 and 1.6e-3.
const SERVED: Precision = Precision {
    dtypes: Dtypes {
        lm: FloatDType::F16,
        local_encoder: FloatDType::F16,
        local_dit: FloatDType::F16,
        audio_decoder: FloatDType::F32,
    },
    patch: 8e-1,
    stop_logits: 2e-1,
    audio: 2e-3,
};

fn golden() -> Fixture {
    fixture("voxcpm", "voxcpm_checkpoint")
}

/// The voice the golden was made with, as long as it is still the one it was made with.
fn voice(golden: &Fixture) -> Latent {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    let dir = std::path::absolute(PathBuf::from(dir)).unwrap();
    let metadata = golden.metadata();
    let path = voices::latent_path(&dir, &metadata["voice"]);
    let voice = Latent::read(&path).unwrap();
    let frames = voice.frames.as_flattened().iter();
    let bytes: Vec<u8> = frames.flat_map(|value| value.to_le_bytes()).collect();
    let checksums = [
        format!("{:08x}", crc32fast::hash(&bytes)),
        format!("{:08x}", crc32fast::hash(voice.text.as_bytes())),
    ];
    let expected = [&metadata["voice_crc32"], &metadata["transcript_crc32"]];
    assert_eq!(checksums.each_ref(), expected, "{} changed", path.display());
    voice
}

fn generates_as_upstream(precision: &Precision) {
    let (golden, device) = (golden(), gpu::device());
    let (voice, text) = (voice(&golden), &golden.metadata()["text"]);
    let tts = TextToSpeech::load(&checkpoint_dir(), &precision.dtypes, &device).unwrap();
    let (model, params) = (tts.model(), Params::default());
    let prompt = tts.prompt(text, &voice).unwrap();

    let noise = golden.tensor::<3>("input.noise", &device);
    let patches = golden.tensor::<3>("output.patches", &device);
    let logits = golden.tensor::<2>("output.stop_logits", &device);
    let steps = patches.dims()[0];
    let at = |tensor: &Tensor<3>, i| tensor.clone().narrow(0, i, 1);

    let mut generation = model.start(&prompt, steps).unwrap();
    for i in 0..steps {
        let step = generation.sample(at(&noise, i), &params.sampling);
        assert_eq!(step.stops(), i == steps - 1, "patch {i}");
        assert_close(step.patch, at(&patches, i), precision.patch, 0.0);
        let stop_logits = step.stop_logits.cast(FloatDType::F32);
        assert_close(
            stop_logits,
            logits.clone().narrow(0, i, 1),
            precision.stop_logits,
            0.0,
        );
        generation.advance(at(&patches, i));
    }

    let mut generation = model.start(&prompt, params.max_patches).unwrap();
    let mut stopped = None;
    for i in 0..steps {
        let step = generation.sample(at(&noise, i), &params.sampling);
        if step.stops() {
            stopped = Some(i);
            break;
        }
        generation.advance(step.patch);
    }
    assert_eq!(
        stopped,
        Some(steps - 1),
        "stopped after another patch than upstream"
    );

    let audio = tts
        .synthesize(text, &voice, &params, &mut Recorded::new(noise))
        .unwrap();
    let samples: usize = golden.metadata()["audio_samples"].parse().unwrap();
    assert_eq!(audio.len(), samples);
    let start = golden.tensor::<1>("output.audio.start", &device);
    let audio = Tensor::<1>::from_floats(&audio[..start.dims()[0]], &device);
    assert_close(audio, start, precision.audio, 0.0);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn generates_as_upstream_on_the_real_weights_in_f32() {
    generates_as_upstream(&F32);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn generates_as_upstream_on_the_real_weights_as_served() {
    generates_as_upstream(&SERVED);
}
