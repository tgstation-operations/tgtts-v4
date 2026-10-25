mod common;

use std::collections::VecDeque;

use audiovae::{Decoder, Weights};
use burn::{
    store::SafetensorsStore,
    tensor::{FloatDType, Tensor},
};
use common::tiny;
use testkit::{Fixture, assert_close, device, fixture};
use voxcpm::{Dtypes, Error, Model, Noise, Prompt, Sampling, SeededNoise, decode};

const F32: Dtypes = Dtypes {
    lm: FloatDType::F32,
    local_encoder: FloatDType::F32,
    local_dit: FloatDType::F32,
    audio_decoder: FloatDType::F32,
};

/// Upstream's noise, patch by patch.
struct Recorded(VecDeque<Vec<f32>>);

impl Noise for Recorded {
    fn next(&mut self, len: usize) -> Vec<f32> {
        let noise = self.0.pop_front().expect("upstream drew no more noise");
        assert_eq!(noise.len(), len);
        noise
    }
}

fn recorded(fixture: &Fixture) -> Recorded {
    let noise = fixture.tensor::<3>("input.noise", &device());
    let steps = noise.dims()[0];
    Recorded(
        (0..steps)
            .map(|i| floats(noise.clone().narrow(0, i, 1)))
            .collect(),
    )
}

fn floats<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().iter::<f32>().collect()
}

fn prompt(fixture: &Fixture) -> Prompt {
    let device = device();
    let tokens = fixture.int_tensor::<1>("input.text_tokens", &device);
    let tokens: Vec<u32> = tokens.into_data().iter::<i64>().map(|t| t as u32).collect();
    let voice = floats(fixture.tensor::<2>("input.voice", &device));
    Prompt::cloning(&tokens, &voice, &tiny()).unwrap()
}

fn model(fixture: &Fixture) -> Model {
    Model::load(&tiny(), fixture.path(), &F32, &device()).unwrap()
}

#[test]
fn prompts_with_a_voice_as_the_service_did() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let prompt = prompt(&fixture);
    let ints = |name| -> Vec<i64> {
        let tensor = fixture.int_tensor::<1>(name, &device);
        tensor.into_data().iter().collect()
    };
    let tokens: Vec<i64> = prompt.tokens.iter().map(|&t| i64::from(t)).collect();
    assert_eq!(tokens, ints("input.prompt.tokens"));
    let audio: Vec<i64> = prompt.audio.iter().map(|&a| i64::from(a)).collect();
    assert_eq!(audio, ints("input.prompt.audio"));
    let patches = floats(fixture.tensor::<3>("input.prompt.patches", &device));
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&prompt.patches), bits(&patches));
}

#[test]
fn generates_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let model = model(&fixture);
    let expected = fixture.tensor::<3>("output.patches", &device);
    let expected_logits = fixture.tensor::<2>("output.stop_logits", &device);
    let steps = expected.dims()[0];
    let mut noise = recorded(&fixture);
    let mut generation = model.start(&prompt(&fixture), steps).unwrap();
    // f32 here and in upstream, each fed back its own patches: over the eight steps the patches,
    // of magnitude under 3.4, stayed within 6.7e-7 of upstream's, and the logits within 1.2e-7.
    for i in 0..steps {
        let step = generation.sample(model.noise(&mut noise), &Sampling::default());
        let patch = expected.clone().narrow(0, i, 1);
        assert_close(step.patch.clone(), patch, 2e-6, 1e-6);
        let logits = expected_logits.clone().narrow(0, i, 1);
        assert_close(step.stop_logits, logits, 1e-6, 1e-6);
        generation.advance(step.patch);
    }
}

#[test]
fn stops_after_the_first_patch_the_stop_head_says_to() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let logits = fixture.tensor::<2>("output.stop_logits", &device);
    let stops: Vec<i64> = logits.argmax(1).into_data().iter().collect();
    let first = stops.iter().position(|&stop| stop == 1).unwrap();
    let expected = fixture.tensor::<3>("output.patches", &device);
    let max_patches = expected.dims()[0];
    let patches = model(&fixture)
        .generate(
            &prompt(&fixture),
            &Sampling::default(),
            max_patches,
            &mut recorded(&fixture),
        )
        .unwrap();
    assert_close(patches, expected.narrow(0, 0, first + 1), 2e-6, 1e-6);
}

#[test]
fn stops_at_max_patches() {
    let fixture = fixture!("voxcpm_generate");
    let model = model(&fixture);
    let generate = |max_patches| {
        let mut noise = recorded(&fixture);
        model.generate(
            &prompt(&fixture),
            &Sampling::default(),
            max_patches,
            &mut noise,
        )
    };
    assert_eq!(generate(2).unwrap().dims(), [2, 4, 8]);
    assert!(matches!(generate(0), Err(Error::Request(_))));
}

#[test]
fn a_voice_of_part_of_a_patch_is_refused() {
    let voice = vec![0.0; 3 * 8];
    let error = Prompt::cloning(&[1, 2], &voice, &tiny()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "cannot generate from a voice of 24 values, not a whole number of 4-frame patches of 8 \
         channels"
    );
}

#[test]
fn a_prompt_past_the_lms_positions_is_refused() {
    let fixture = fixture!("voxcpm_generate");
    let prompt = prompt(&fixture);
    let max_patches = 64 - prompt.len() + 1;
    let error = model(&fixture).start(&prompt, max_patches).unwrap_err();
    assert!(matches!(error, Error::Request(_)), "{error}");
}

#[test]
fn seeded_noise_is_standard_normal_and_repeats_for_a_seed() {
    let draw = |seed| SeededNoise::new(seed).next(4096);
    assert_eq!(draw(7), draw(7));
    assert_ne!(draw(7), draw(8));
    let noise = draw(7);
    let mean = noise.iter().sum::<f32>() / 4096.0;
    let variance = noise.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4096.0;
    // Three and four and a half standard errors of 4096 draws.
    assert!(mean.abs() < 0.05, "mean {mean}");
    assert!((variance - 1.0).abs() < 0.1, "variance {variance}");
}

#[test]
fn decodes_after_the_voices_last_frames_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let weights = Weights::from_store(&mut SafetensorsStore::from_file(fixture.path())).unwrap();
    let decoder = Decoder::load(&tiny().audio_vae(), &weights, FloatDType::F32, &device).unwrap();
    let voice = floats(fixture.tensor::<2>("input.voice", &device));
    let audio = decode(&decoder, &voice, fixture.tensor("output.patches", &device));
    let audio = Tensor::<1>::from_floats(audio.as_slice(), &device);
    // f32 here and in upstream, summing in other orders: this audio, within (-0.64, 0.64), came
    // within 1.5e-7 of upstream's.
    let expected = fixture.tensor::<1>("output.audio", &device);
    assert_close(audio, expected, 1e-6, 0.0);
}
