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
use voxcpm::{Dtypes, Noise, Params, SeededNoise, TextToSpeech};

/// How far what generation gives in some dtypes may be from upstream's in f32.
struct Precision {
    dtypes: Dtypes,
    /// A patch sampled after upstream's before it.
    patch: f32,
    /// The steps whose patches are held only to being finite: those whose patches a change in
    /// what conditions them moves furthest.
    amplified: &'static [usize],
    /// The stop head's logits.
    stop_logits: f32,
    /// How many patches past upstream's last generation run free may go on for.
    stops_late: usize,
    /// The audio of the first two patches.
    audio: f32,
}

// As on the CPU, whose test says why: given upstream's patches, each patch came within 2.3e-3
// of upstream's and the stop logits within 3.8e-6, where upstream's last margin for stopping is
// 0.070, and the first two patches decoded within 1.1e-5 of its audio.
const F32: Precision = Precision {
    dtypes: Dtypes {
        lm: FloatDType::F32,
        local_encoder: FloatDType::F32,
        local_dit: FloatDType::F32,
        audio_decoder: FloatDType::F32,
    },
    patch: 1e-2,
    amplified: &[],
    stop_logits: 1e-4,
    stops_late: 0,
    audio: 1e-4,
};

// f16 but for the audio decoder (tests/audiovae.rs says why). Autotune picks kernels by timing
// them in each process, so how they round differs from one process to the next, and against our
// f32 on the same inputs 1 to 9 of the scalar quantizer's 512 levels round apart at every step.
// What that does to a patch depends on the step: in f32 one level over the prompt lies 1.7e-6
// from a boundary and rounds either way from process to process, which moves the fifteenth and
// eighteenth patches by up to 1.1e-3 and 2.3e-3 and the others by at most 5.0e-4. Over 22
// processes, given upstream's patches, those two came within 5.2e-1 and 2.0e-1 of upstream's,
// with nothing to bound how far other levels rounding apart would take them, and the others
// within 7.3e-2; the stop logits came within 1.0e-1, and the first two patches decoded within
// 6.3e-4 of upstream's audio. The dtypes the Python service ran in, bf16 for the LMs and the DiT
// and f16 for the decoder, came on the CPU within 2.0e-1 but the fifteenth at 1.2, 1.6e-1 and
// 1.6e-3.
//
// Run free, the stop logits drift with the patches, up to 3.5e-1 from upstream's at its last
// patch over ten processes, where upstream's margin for stopping is 0.070: one of the ten went
// on a patch longer. A patch before, upstream's margin is -6.5, and given upstream's patches the
// margin a patch after is +3.2, so generation stops at upstream's last patch or the next.
const SERVED: Precision = Precision {
    dtypes: Dtypes {
        lm: FloatDType::F16,
        local_encoder: FloatDType::F16,
        local_dit: FloatDType::F16,
        audio_decoder: FloatDType::F32,
    },
    patch: 2e-1,
    amplified: &[14, 17],
    stop_logits: 2e-1,
    stops_late: 1,
    audio: 2e-3,
};

/// Upstream's noise, then seeded noise for any patch after upstream's last.
struct Extended {
    upstream: Recorded,
    left: usize,
    after: SeededNoise,
}

impl Extended {
    fn new(noise: Tensor<3>) -> Self {
        Self {
            left: noise.dims()[0],
            upstream: Recorded::new(noise),
            after: SeededNoise::new(0),
        }
    }
}

impl Noise for Extended {
    fn next(&mut self, len: usize) -> Vec<f32> {
        if self.left == 0 {
            return self.after.next(len);
        }
        self.left -= 1;
        self.upstream.next(len)
    }
}

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
    // Logits each within `stop_logits` of upstream's move the margin between going on and
    // stopping by up to twice that, so the decision is checked where upstream's is wider.
    let decided = |i| {
        let upstream: Vec<f32> = logits.clone().narrow(0, i, 1).into_data().iter().collect();
        (upstream[1] - upstream[0]).abs() > 2.0 * precision.stop_logits
    };

    let mut generation = model.start(&prompt, steps).unwrap();
    for i in 0..steps {
        let step = generation.sample(at(&noise, i), &params.sampling);
        if decided(i) {
            assert_eq!(step.stops(), i == steps - 1, "patch {i}");
        }
        if precision.amplified.contains(&i) {
            let patch: Vec<f32> = step.patch.into_data().iter().collect();
            assert!(patch.iter().all(|x| x.is_finite()), "patch {i}");
        } else {
            assert_close(step.patch, at(&patches, i), precision.patch, 0.0);
        }
        let stop_logits = step.stop_logits.cast(FloatDType::F32);
        assert_close(
            stop_logits,
            logits.clone().narrow(0, i, 1),
            precision.stop_logits,
            0.0,
        );
        generation.advance(at(&patches, i));
    }

    let stops = steps..=steps + precision.stops_late;
    let mut generation = model.start(&prompt, params.max_patches).unwrap();
    let mut free = Extended::new(noise.clone());
    let mut stopped = None;
    for i in 0..*stops.end() {
        let step = generation.sample(model.noise(&mut free), &params.sampling);
        if step.stops() {
            stopped = Some(i + 1);
            break;
        }
        generation.advance(step.patch);
    }
    assert!(
        stopped.is_some_and(|patches| stops.contains(&patches)),
        "stopped after {stopped:?} patches, not {stops:?}"
    );

    let audio = tts
        .synthesize(text, &voice, &params, &mut Extended::new(noise))
        .unwrap();
    let samples: usize = golden.metadata()["audio_samples"].parse().unwrap();
    let per_patch = samples / steps;
    assert_eq!(audio.len() % per_patch, 0, "{} samples", audio.len());
    let spoken = audio.len() / per_patch;
    assert!(stops.contains(&spoken), "spoke {spoken} patches");
    assert!(audio.iter().all(|x| x.is_finite()));
    let start = golden.tensor::<1>("output.audio.start", &device);
    let kept = start.dims()[0];
    // Upstream's last two patches, where its speech trails off, still reach 1.2e-2.
    let peak = audio[kept..].iter().fold(0f32, |peak, x| peak.max(x.abs()));
    assert!(
        peak > 1e-3,
        "silent after the first two patches, at a peak of {peak}"
    );
    let audio = Tensor::<1>::from_floats(&audio[..kept], &device);
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
