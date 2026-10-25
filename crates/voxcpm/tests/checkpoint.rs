//! VoxCPM2's own weights against what upstream computes with them, as
//! fixtures/voxcpm_checkpoint.py records it.

mod recorded;

use std::{fs, path::PathBuf, time::Instant};

use audiovae::{Decoder, Weights};
use burn::tensor::{FloatDType, Tensor};
use recorded::Recorded;
use testkit::{Fixture, assert_close, checkpoint_dir, device, fixture};
use voices::Latent;
use voxcpm::{Config, Dtypes, Params, SeededNoise, SpeechDecoder, TextToSpeech};

const F32: Dtypes = Dtypes {
    lm: FloatDType::F32,
    local_encoder: FloatDType::F32,
    local_dit: FloatDType::F32,
    audio_decoder: FloatDType::F32,
};

fn golden() -> Fixture {
    fixture!("voxcpm_checkpoint")
}

/// The voices directory `VOICES_DIR` names.
fn voices_dir() -> PathBuf {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    std::path::absolute(dir).expect("absolute of a non-empty path")
}

/// The voice the golden was made with, as long as it is still the one it was made with.
fn voice(golden: &Fixture) -> Latent {
    let metadata = golden.metadata();
    let path = voices::latent_path(&voices_dir(), &metadata["voice"]);
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

fn max_diff<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) -> f32 {
    (actual - expected).abs().max().into_scalar()
}

/// Making the golden needs the checkpoint and the voices, so a run of every generator keeps it
/// as it is: this refuses one that such a run has left behind.
#[test]
fn golden_was_made_as_the_fixtures_were() {
    let (golden, fixtures) = (golden(), fixture!("voxcpm_generate"));
    let regenerate = "regenerate it with VOXCPM2_DIR and VOICES_DIR set: \
                      nix run .#fixtures voxcpm_checkpoint";
    for key in ["torch", "upstream"] {
        assert_eq!(
            golden.metadata()[key],
            fixtures.metadata()[key],
            "the golden's {key} differs from the fixtures'; {regenerate}"
        );
    }
    let generator = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/voxcpm_checkpoint.py"
    );
    let crc32 = format!("{:08x}", crc32fast::hash(&fs::read(generator).unwrap()));
    assert_eq!(
        golden.metadata().get("generator_crc32"),
        Some(&crc32),
        "fixtures/voxcpm_checkpoint.py changed after making the golden; {regenerate}"
    );
}

/// One level of the scalar quantizer over the prompt lies within half an ulp of a rounding
/// boundary: upstream has -5.4999995 and rounds it to -5, this -5.500004 and rounds it to -6. The
/// residual LM carries that step into the hidden states every patch is sampled from, and upstream
/// made to round it so moves its own patches by what these differ by, 5.5e-4 at the first and
/// 6.7e-3 at the fifteenth. Otherwise the two agree to f32 rounding: given upstream's inputs, each
/// part comes within 1.5e-6 of it relatively, and the stop head, which sees the base LM only
/// through the quantizer, within 2.8e-5 of logits up to 13. A level of a step's own rounded apart
/// would move those by about 1e-2; given upstream's patches, none is.
const PATCH: f32 = 1e-2;
const STOP_LOGITS: f32 = 1e-4;

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn generates_as_upstream_on_the_real_weights() {
    let (golden, device) = (golden(), device());
    let (voice, text) = (voice(&golden), &golden.metadata()["text"]);
    let started = Instant::now();
    let tts = TextToSpeech::load(&checkpoint_dir(), &F32, &device).unwrap();
    eprintln!("loaded in {:.1?}", started.elapsed());
    let (model, params) = (tts.model(), Params::default());

    let prompt = tts.prompt(text, &voice).unwrap();
    let tokens = prompt
        .tokens
        .iter()
        .flat_map(|&t| i64::from(t).to_le_bytes());
    let tokens = [
        prompt.len() as u64,
        u64::from(crc32fast::hash(&tokens.collect::<Vec<_>>())),
    ];
    let expected: [u64; 2] = serde_json::from_str(&golden.metadata()["prompt_tokens"]).unwrap();
    assert_eq!(tokens, expected, "the prompt's tokens are not upstream's");

    let noise = golden.tensor::<3>("input.noise", &device);
    let patches = golden.tensor::<3>("output.patches", &device);
    let logits = golden.tensor::<2>("output.stop_logits", &device);
    let steps = patches.dims()[0];
    let at = |tensor: &Tensor<3>, i| tensor.clone().narrow(0, i, 1);
    let logits_at = |i| logits.clone().narrow(0, i, 1);

    // Each patch sampled after upstream's before it, so that each starts where upstream's did.
    let mut generation = model.start(&prompt, steps).unwrap();
    for i in 0..steps {
        let step = generation.sample(at(&noise, i), &params.sampling);
        assert_eq!(step.stops(), i == steps - 1, "patch {i}");
        assert_close(step.patch, at(&patches, i), PATCH, 0.0);
        assert_close(step.stop_logits, logits_at(i), STOP_LOGITS, 0.0);
        generation.advance(at(&patches, i));
    }

    // Each patch sampled after this one's own, as generation goes. A level of the quantizer over a
    // step's hidden state can land as near a boundary as the prompt's one, so how far these drift
    // from upstream's depends on how the kernels sum: built as the gate builds it this drifts to
    // 5.4e-2, Flex with rayon and SIMD to 4.9e-1 after rounding levels apart at the eighth and the
    // thirteenth patches too.
    let started = Instant::now();
    let mut generation = model.start(&prompt, params.max_patches).unwrap();
    eprintln!(
        "prefill of {} positions in {:.1?}",
        prompt.len(),
        started.elapsed()
    );
    let started = Instant::now();
    let (mut drift, mut stopped) = (vec![], None);
    for i in 0..steps {
        let step = generation.sample(at(&noise, i), &params.sampling);
        drift.push(max_diff(step.patch.clone(), at(&patches, i)));
        if step.stops() {
            stopped = Some(i);
            break;
        }
        generation.advance(step.patch);
    }
    let per_patch = started.elapsed() / drift.len() as u32;
    eprintln!("{} patches at {per_patch:.2?} each", drift.len());
    let drift: Vec<_> = drift.iter().map(|d| format!("{d:.1e}")).collect();
    eprintln!("each patch from upstream's: {}", drift.join(" "));
    // Upstream's last logits are 0.070 apart, and these drift at most 3.9e-2 from them there.
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
    // The first two patches, 5.7e-4 and 1.9e-4 from upstream's, decode within 1.1e-5 of its
    // audio, which they reach up to 0.053 of.
    let start = golden.tensor::<1>("output.audio.start", &device);
    let kept = start.dims()[0];
    let audio = Tensor::<1>::from_floats(&audio[..kept], &device);
    assert_close(audio, start, 1e-4, 0.0);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn decodes_upstreams_patches_as_upstream_on_the_real_weights() {
    let (golden, device) = (golden(), device());
    let voice = voice(&golden);
    let dir = checkpoint_dir();
    let json = fs::read_to_string(dir.join("config.json")).unwrap();
    let config = Config::from_json(&json).unwrap();
    let weights = Weights::open(dir.join("audiovae.pth")).unwrap();
    let decoder = Decoder::load(&config.audio_vae(), &weights, FloatDType::F32, &device).unwrap();
    let patches = golden.tensor::<3>("output.patches", &device);
    let chunk_frames = Params::default().decode_chunk_frames;
    let mut speech = SpeechDecoder::new(
        &decoder,
        voice.frames.as_flattened(),
        config.feat_dim,
        chunk_frames,
    );
    let audio = speech.push(patches);
    let samples: usize = golden.metadata()["audio_samples"].parse().unwrap();
    assert_eq!(audio.len(), samples);
    let start = golden.tensor::<1>("output.audio.start", &device);
    let end = golden.tensor::<1>("output.audio.end", &device);
    let kept = start.dims()[0];
    let audio = Tensor::<1>::from_floats(audio.as_slice(), &device);
    // f32 here and in upstream: this audio, within (-0.053, 0.053), came within 1.8e-7 of
    // upstream's.
    assert_close(audio.clone().narrow(0, 0, kept), start, 1e-6, 0.0);
    assert_close(audio.narrow(0, samples - kept, kept), end, 1e-6, 0.0);
}

/// Each part in a dtype of its own, so that what passes between them has to be cast: the LMs and
/// the DiT in bf16 and the decoder in f16, as prod runs them, and the local encoder between them
/// in f32.
#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn synthesizes_with_each_part_in_its_own_dtype() {
    let golden = golden();
    let (voice, text) = (voice(&golden), &golden.metadata()["text"]);
    let dtypes = Dtypes {
        lm: FloatDType::BF16,
        local_encoder: FloatDType::F32,
        local_dit: FloatDType::BF16,
        audio_decoder: FloatDType::F16,
    };
    let tts = TextToSpeech::load(&checkpoint_dir(), &dtypes, &device()).unwrap();
    let params = Params {
        max_patches: 3,
        ..Params::default()
    };
    let audio = tts
        .synthesize(text, &voice, &params, &mut SeededNoise::new(0))
        .unwrap();
    let patch = 4 * 1920;
    assert!(
        (1..=3).any(|patches| audio.len() == patches * patch),
        "{} samples",
        audio.len()
    );
    assert!(audio.iter().all(|x| x.is_finite()));
    let peak = audio.iter().fold(0f32, |peak, x| peak.max(x.abs()));
    assert!(peak > 1e-3, "silent, at a peak of {peak}");
}
