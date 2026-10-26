mod common;

use burn::tensor::FloatDType;
use common::tiny;
use testkit::{Fixture, assert_close, device, fixture};
use voxcpm::{Dtypes, Generation, Model, Noise, Prompt, PromptCache, Sampling, SeededNoise, Step};

const F32: Dtypes = Dtypes {
    lm: FloatDType::F32,
    local_encoder: FloatDType::F32,
    local_dit: FloatDType::F32,
    audio_decoder: FloatDType::F32,
};

fn model(fixture: &Fixture) -> Model {
    Model::load(&tiny(), fixture.path(), &F32, &device()).unwrap()
}

/// A voice of `patches` random patches, and a prompt in it of the transcript's tokens and then
/// the text's.
fn prompt(voice_seed: u64, patches: usize, transcript: &[u32], text: &[u32]) -> Prompt {
    let config = tiny();
    let voice = SeededNoise::new(voice_seed).next(patches * config.patch_size * config.feat_dim);
    let tokens: Vec<u32> = transcript.iter().chain(text).copied().collect();
    Prompt::cloning(&tokens, &voice, &config).unwrap()
}

// f32 here and run whole, the voice's patches embedded apart from the rest of the prompt and the
// prompt run in other runs of positions: at this size a prompt that went on from what the cache
// kept came within 4.8e-7 of one run whole.
const ATOL: f32 = 2e-6;
const RTOL: f32 = 1e-6;

/// Each step of a generation alone, sampled from `seed`'s noise and fed back its own patches.
fn alone(model: &Model, generation: Generation<'_>, seed: u64, steps: usize) -> Vec<Step> {
    let mut generation = generation;
    let mut noise = SeededNoise::new(seed);
    (0..steps)
        .map(|i| {
            let step = generation.sample(model.noise(&mut noise), &Sampling::default());
            if i + 1 < steps {
                generation.advance(step.patch.clone());
            }
            step
        })
        .collect()
}

/// A prompt whose voice the cache keeps nothing of, one that goes on from the transcript the
/// first left there, and the first again, which goes on from all its text: each run in other
/// runs of positions than the others.
#[test]
fn a_prompt_goes_on_from_what_the_cache_keeps_of_its_voice() {
    let fixture = fixture!("voxcpm_generate");
    let model = model(&fixture);
    let transcript = [5, 17, 42, 9];
    let first = prompt(1, 5, &transcript, &[77, 3, 64]);
    let second = prompt(1, 5, &transcript, &[11, 12, 13, 14, 15]);
    let mut cache = PromptCache::new(usize::MAX);
    for (prompt, cached, runs) in [(&first, 0, 3), (&second, 11, 4), (&first, 14, 100)] {
        let expected = alone(&model, model.start(prompt, 8).unwrap(), 7, 8);
        let mut prefill = model.prefill(prompt, 8, &mut cache).unwrap();
        assert_eq!(prompt.len() - prefill.left(), cached);
        while !prefill.run(runs, &mut cache) {}
        let steps = alone(&model, prefill.finish(), 7, 8);
        for (step, expected) in steps.into_iter().zip(expected) {
            assert_close(step.patch, expected.patch, ATOL, RTOL);
            assert_close(step.stop_logits, expected.stop_logits, ATOL, RTOL);
        }
    }
    assert_eq!(cache.voices(), 1);
}

#[test]
fn a_prompt_cache_keeps_the_voices_used_last_within_its_bytes() {
    let fixture = fixture!("voxcpm_generate");
    let model = model(&fixture);
    let prompts = [1, 2, 3].map(|voice| prompt(voice, 5, &[5, 17], &[9]));
    let mut unbounded = PromptCache::new(usize::MAX);
    let mut prefill = model.prefill(&prompts[0], 8, &mut unbounded).unwrap();
    prefill.run(usize::MAX, &mut unbounded);
    let voice = unbounded.bytes();
    // Room for two voices: the first is dropped for the third, the second having been used
    // since.
    let mut cache = PromptCache::new(2 * voice);
    for i in [0, 1, 0, 1, 2] {
        let mut prefill = model.prefill(&prompts[i], 8, &mut cache).unwrap();
        prefill.run(usize::MAX, &mut cache);
    }
    assert_eq!((cache.voices(), cache.bytes()), (2, 2 * voice));
    let kept = |cache: &mut PromptCache, i: usize| {
        let prefill = model.prefill(&prompts[i], 8, cache).unwrap();
        prompts[i].len() - prefill.left()
    };
    assert_eq!(kept(&mut cache, 1), 2 + 5 + 3);
    assert_eq!(kept(&mut cache, 2), 2 + 5 + 3);
    // Asking for the first drops the second.
    assert_eq!(kept(&mut cache, 0), 0);
    assert_eq!(kept(&mut cache, 1), 0);
    // Room for a voice's embeddings but not its keys and values keeps the embeddings, and less
    // keeps nothing.
    let embeds = 5 * 32 * 4;
    for (budget, kept) in [(voice - 1, (1, embeds)), (embeds - 1, (0, 0))] {
        let mut small = PromptCache::new(budget);
        let mut prefill = model.prefill(&prompts[0], 8, &mut small).unwrap();
        prefill.run(usize::MAX, &mut small);
        assert_eq!((small.voices(), small.bytes()), kept);
    }
}
