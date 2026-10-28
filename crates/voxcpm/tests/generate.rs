mod common;
mod recorded;

use std::num::NonZeroUsize;

use audiovae::{Decoder, Weights};
use burn::{
    store::SafetensorsStore,
    tensor::{FloatDType, Tensor, TensorData},
};
use common::tiny;
use recorded::Recorded;
use testkit::{Fixture, assert_close, device, fixture};
use voxcpm::{
    Cfm, Dtypes, Error, Model, Noise, Prompt, Sampling, SeededNoise, SpeechChunks, SpeechDecoder,
    Stepping,
};

const F32: Dtypes = Dtypes {
    lm: FloatDType::F32,
    local_encoder: FloatDType::F32,
    local_dit: FloatDType::F32,
    audio_decoder: FloatDType::F32,
};

fn recorded(fixture: &Fixture) -> Recorded {
    Recorded::new(fixture.tensor("input.noise", &device()))
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

/// The condition and the stop logits a generation gives are those its patches are sampled with:
/// the sampler given them samples each patch bit for bit.
#[test]
fn samples_with_the_conditioning_it_gives() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let (model, config) = (model(&fixture), tiny());
    let cfm = Cfm::load(&config, fixture.path(), FloatDType::F32, &device).unwrap();
    let prompt = prompt(&fixture);
    let patch = config.patch_size * config.feat_dim;
    let last = prompt.patches[prompt.patches.len() - patch..].to_vec();
    let shape = [1, config.patch_size, config.feat_dim];
    let mut cond = Tensor::<3>::from_data(TensorData::new(last, shape), &device);
    let mut noise = recorded(&fixture);
    let mut generation = model.start(&prompt, 3).unwrap();
    for _ in 0..3 {
        let draw = model.noise(&mut noise);
        let (mu, stop_logits) = generation.conditioning();
        let step = generation.sample(draw.clone(), &Sampling::default());
        let sampled = cfm.sample(mu, cond.swap_dims(1, 2), draw, &Sampling::default());
        assert_close(sampled.swap_dims(1, 2), step.patch.clone(), 0.0, 0.0);
        assert_close(stop_logits, step.stop_logits, 0.0, 0.0);
        cond = step.patch.clone();
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

/// The patches of the stable step, kernel by kernel and replayed, at the default sampling and at
/// another, twice, the second time in the caches the first left: at this size they come out bit
/// for bit as the eager step's.
#[test]
fn steps_as_the_eager_step() {
    let fixture = fixture!("voxcpm_generate");
    let mut model = model(&fixture);
    let prompt = prompt(&fixture);
    let other = Sampling {
        timesteps: 6,
        temperature: 0.7,
        cfg_value: 1.5,
    };
    for sampling in [Sampling::default(), other] {
        let mut generate = |stepping| {
            model.set_stepping(stepping);
            let mut noise = SeededNoise::new(3);
            model.generate(&prompt, &sampling, 8, &mut noise).unwrap()
        };
        let eager = generate(Stepping::Eager);
        assert!(eager.dims()[0] > 1);
        for stepping in [Stepping::Stable, Stepping::Captured, Stepping::Captured] {
            assert_close(generate(stepping), eager.clone(), 0.0, 0.0);
        }
    }
}

/// A graph of the step reads the model's weights, which live on the stream of the thread that
/// loaded them, so steps are captured on that thread alone.
#[test]
fn steps_are_captured_only_on_the_thread_the_model_loaded_on() {
    let fixture = fixture!("voxcpm_generate");
    let mut model = model(&fixture);
    model.set_stepping(Stepping::Captured);
    let (model, prompt) = (&model, &prompt(&fixture));
    let generate = move || {
        let mut noise = SeededNoise::new(3);
        model.generate(prompt, &Sampling::default(), 2, &mut noise)
    };
    let elsewhere = std::thread::scope(|scope| scope.spawn(generate).join().unwrap());
    assert!(
        matches!(elsewhere, Err(Error::OtherThread)),
        "{elsewhere:?}"
    );
    assert!(generate().is_ok());
}

/// Four speeches decoded a chunk at a time together, one starting two patches after the others,
/// the first in a slot another left, decoded as `captured` says, three at once filling out a
/// graph of four rows: each comes out bit for bit as decoded alone a second at a time.
fn decodes_speeches_a_chunk_at_a_time_together(captured: bool) {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let weights = Weights::from_store(&mut SafetensorsStore::from_file(fixture.path())).unwrap();
    let decoder = Decoder::load(&tiny().audio_vae(), &weights, FloatDType::F32, &device).unwrap();
    let voice = floats(fixture.tensor::<2>("input.voice", &device));
    let patches = fixture.tensor::<3>("output.patches", &device);
    let count = patches.dims()[0];
    let patch = |i: usize| patches.clone().narrow(0, i, 1);
    // Each speech's patches, and the step it starts at.
    let speeches: [(Vec<Tensor<3>>, usize); 4] = [
        ((0..count).map(patch).collect(), 0),
        (
            (0..count - 3).map(|i| patch(count - 1 - i).neg()).collect(),
            2,
        ),
        (
            (0..count - 1).map(|i| patch(i).mul_scalar(0.5)).collect(),
            0,
        ),
        (
            (0..count - 2)
                .map(|i| patch(i + 1).mul_scalar(-2.0))
                .collect(),
            0,
        ),
    ];
    let alone = |patches: &[Tensor<3>]| {
        let second = NonZeroUsize::new(25).unwrap();
        let mut speech = SpeechDecoder::new(&decoder, &voice, 8, second);
        let samples = speech.push(Tensor::cat(patches.to_vec(), 0));
        Tensor::<1>::from_floats(samples.as_slice(), &device)
    };
    let mut chunks = SpeechChunks::new(&decoder, 8, 4, captured, &device);
    chunks.warm_up();
    let mut gone = chunks.start(&voice);
    gone.push(Tensor::cat(vec![patches.clone(); 3], 0));
    chunks.decode(&mut [&mut gone], &[false]);
    chunks.release(gone);
    let mut decoders: Vec<_> = speeches.iter().map(|_| chunks.start(&voice)).collect();
    let mut audio = vec![vec![]; speeches.len()];
    for step in 0..count {
        let (mut chosen, mut last, mut whose) = (vec![], vec![], vec![]);
        for (i, (decoder, (patches, start))) in decoders.iter_mut().zip(&speeches).enumerate() {
            let Some(patch) = step.checked_sub(*start).and_then(|at| patches.get(at)) else {
                continue;
            };
            decoder.push(patch.clone());
            let ends = step - start == patches.len() - 1;
            if decoder.ready() || ends {
                chosen.push(decoder);
                last.push(ends);
                whose.push(i);
            }
        }
        if !chosen.is_empty() {
            for (i, samples) in whose.into_iter().zip(chunks.decode(&mut chosen, &last)) {
                audio[i].extend(samples);
            }
        }
    }
    assert_eq!(chunks.graphs(), if captured { 3 } else { 0 });
    for (audio, (patches, _)) in audio.into_iter().zip(&speeches) {
        let audio = Tensor::<1>::from_floats(audio.as_slice(), &device);
        assert_close(audio, alone(patches), 0.0, 0.0);
    }
}

#[test]
fn decodes_speeches_a_chunk_at_a_time_together_as_they_come() {
    decodes_speeches_a_chunk_at_a_time_together(false);
}

#[test]
fn decodes_speeches_a_chunk_at_a_time_together_by_replays() {
    decodes_speeches_a_chunk_at_a_time_together(true);
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

/// The noise seeds draw, as they always have: an update of rand or rand_distr that drew other
/// noise would change the speech of every seed.
#[test]
fn seeds_draw_the_noise_they_always_have() {
    let first = [
        0.712813, 0.85833144, -2.4362438, 0.16334426, -1.2750102, 1.287171, -1.4814075, 0.61259484,
    ];
    assert_eq!(SeededNoise::new(0).next(8), first);
    let drawn = SeededNoise::new(7).next(4096);
    let bytes: Vec<u8> = drawn.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(crc32fast::hash(&bytes), 0x202b_a547);
}

#[test]
fn decodes_after_the_voices_last_frames_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_generate"), device());
    let weights = Weights::from_store(&mut SafetensorsStore::from_file(fixture.path())).unwrap();
    let decoder = Decoder::load(&tiny().audio_vae(), &weights, FloatDType::F32, &device).unwrap();
    let voice = floats(fixture.tensor::<2>("input.voice", &device));
    let patches = fixture.tensor::<3>("output.patches", &device);
    let expected = fixture.tensor::<1>("output.audio", &device);
    let count = patches.dims()[0];
    // Chunks of the 12 frames of the voice and the 32 of the patches, whole and split, and patches
    // pushed at once and one or a few at a time.
    for (chunk_frames, per_push) in [(44, count), (25, count), (1, 1), (3, 1), (5, 3), (12, 2)] {
        let chunk_frames = NonZeroUsize::new(chunk_frames).unwrap();
        let mut speech = SpeechDecoder::new(&decoder, &voice, 8, chunk_frames);
        let audio: Vec<f32> = (0..count)
            .step_by(per_push)
            .flat_map(|i| speech.push(patches.clone().narrow(0, i, per_push.min(count - i))))
            .collect();
        let audio = Tensor::<1>::from_floats(audio.as_slice(), &device);
        // f32 here and in upstream, summing in other orders: this audio, within (-0.64, 0.64), came
        // within 1.8e-7 of upstream's.
        assert_close(audio, expected.clone(), 1e-6, 0.0);
    }
}
