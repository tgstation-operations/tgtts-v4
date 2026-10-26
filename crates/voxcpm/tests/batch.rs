mod common;

use burn::tensor::{FloatDType, Tensor};
use common::tiny;
use testkit::{Fixture, assert_close, device, fixture};
use voxcpm::{
    Batch, Dtypes, Error, Model, Noise, Prompt, PromptCache, Sampling, Scheduler, SeededNoise,
    Step, Stepping,
};

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

// f32 here and alone, the rows of a product run as one or as several and attention over keys
// other rows add masked out: at this size each row's patches came out bit for bit as alone, and
// those of a request whose prompt went on from what the cache kept within 4.8e-7.
const ATOL: f32 = 2e-6;
const RTOL: f32 = 1e-6;

fn assert_step(step: &Step, row: usize, expected: &Step) {
    let patch = step.patch.clone().narrow(0, row, 1);
    assert_close(patch, expected.patch.clone(), ATOL, RTOL);
    let logits = step.stop_logits.clone().narrow(0, row, 1);
    assert_close(logits, expected.stop_logits.clone(), ATOL, RTOL);
}

/// Rows of other lengths joining a batch at different patches and leaving it at others, each
/// fed the patches it was given alone.
#[test]
fn rows_of_a_batch_are_sampled_as_alone() {
    let fixture = fixture!("voxcpm_generate");
    let model = model(&fixture);
    let prompts = [
        prompt(1, 5, &[5, 17, 42, 9], &[77, 3, 64]),
        prompt(2, 3, &[8, 9], &[10, 11]),
        prompt(3, 2, &[20, 21, 22, 23, 24], &[25, 26, 27, 28, 29]),
    ];
    let steps = 6;
    let sampling = [
        Sampling::default(),
        Sampling {
            temperature: 0.7,
            ..Sampling::default()
        },
        Sampling {
            cfg_value: 3.0,
            ..Sampling::default()
        },
    ];
    let expected: Vec<Vec<Step>> = (0..3)
        .map(|i| {
            let mut generation = model.start(&prompts[i], steps).unwrap();
            let mut noise = SeededNoise::new(i as u64);
            (0..steps)
                .map(|_| {
                    let step = generation.sample(model.noise(&mut noise), &sampling[i]);
                    generation.advance(step.patch.clone());
                    step
                })
                .collect()
        })
        .collect();
    let mut batch = Batch::new(&model);
    let mut noise: Vec<SeededNoise> = (0..3).map(SeededNoise::new).collect();
    // The second row joins after the first's second patch, and the third after its third into
    // a batch grown to four rows; the first leaves after its fourth patch, the third after its
    // fifth, and the second after its sixth, alone in a batch shrunk to one row.
    let (joins, leaves) = ([0, 2, 3], [4, 6, 5]);
    let mut rows = vec![];
    let mut sampled = [0; 3];
    for t in 0..8 {
        for i in 0..3 {
            if t == joins[i] {
                let generation = model.start(&prompts[i], steps).unwrap();
                rows.push((batch.join(generation, &sampling[i]).unwrap(), i));
            }
        }
        assert_eq!(
            batch.rows(),
            rows.iter().map(|&(id, _)| id).collect::<Vec<_>>()
        );
        let draws: Vec<f32> = rows.iter().flat_map(|&(_, i)| noise[i].next(32)).collect();
        let draws = Tensor::<1>::from_floats(draws.as_slice(), &device());
        let step = batch.sample(draws.reshape([rows.len(), 8, 4]));
        let mut fed = vec![];
        for (k, &(id, i)) in rows.iter().enumerate() {
            assert_step(&step, k, &expected[i][sampled[i]]);
            fed.push(expected[i][sampled[i]].patch.clone());
            sampled[i] += 1;
            if sampled[i] == leaves[i] {
                batch.leave(id);
                fed.pop();
            }
        }
        rows.retain(|&(_, i)| sampled[i] < leaves[i]);
        if !fed.is_empty() {
            batch.advance(Tensor::cat(fed, 0));
        }
    }
    assert_eq!(sampled, leaves);
}

/// Requests coming in while others run, one in the voice of another, each stopping as its stop
/// head says or at its most patches, as `stepping` steps them, against each generated alone by
/// the eager step.
fn a_scheduler_generates_each_request_as_alone(stepping: Stepping) {
    let fixture = fixture!("voxcpm_generate");
    let mut model = model(&fixture);
    model.set_stepping(stepping);
    let voice: Vec<f32> = fixture
        .tensor::<2>("input.voice", &device())
        .into_data()
        .iter()
        .collect();
    let cloning = |voice: &[f32], transcript: &[u32], text: &[u32]| {
        let tokens: Vec<u32> = transcript.iter().chain(text).copied().collect();
        Prompt::cloning(&tokens, voice, &tiny()).unwrap()
    };
    let prompts = [
        cloning(&voice, &[5, 17, 42, 9], &[77, 3, 64]),
        prompt(2, 2, &[8, 9], &[10, 11]),
        cloning(&voice, &[5, 17, 42, 9], &[1, 2]),
        prompt(4, 3, &[20, 21, 22, 23, 24], &[25, 26, 27, 28, 29]),
    ];
    let (seeds, max_patches, arrivals) = ([3, 0, 3, 0], [8, 8, 3, 8], [0, 1, 1, 2]);
    let mut scheduler = Scheduler::new(&model, PromptCache::new(usize::MAX), 8);
    let mut got: Vec<Vec<Tensor<3>>> = vec![vec![]; 4];
    let (mut first, mut done, mut most) = ([None; 4], [false; 4], 0);
    let mut started = [None; 4];
    for t in 0..20 {
        for i in 0..4 {
            if t == arrivals[i] {
                let noise = SeededNoise::new(seeds[i]);
                let sampling = Sampling::default();
                scheduler
                    .add(i, &prompts[i], &sampling, max_patches[i], noise)
                    .unwrap();
            }
        }
        let patches = scheduler.step();
        most = most.max(patches.len());
        for patch in patches {
            assert!(
                !done[patch.key],
                "a patch of request {} after its last",
                patch.key
            );
            first[patch.key].get_or_insert(t);
            let first_patch = got[patch.key].is_empty();
            assert_eq!(
                patch.started.is_some(),
                first_patch,
                "request {}",
                patch.key
            );
            started[patch.key] = started[patch.key].or(patch.started);
            got[patch.key].push(patch.patch.unwrap());
            done[patch.key] = patch.last;
        }
        if scheduler.requests() == 0 && t >= 2 {
            break;
        }
    }
    assert_eq!(done, [true; 4]);
    // The first prompt ran whole, with nothing to wait for it, and the others eight positions
    // at a time between patches: the second's 11 by the third step, then the third's 8 left
    // after what the first kept of its voice, and the last's 19 by the sixth, the others
    // having stopped.
    assert_eq!(first, [Some(0), Some(2), Some(3), Some(5)]);
    assert_eq!(most, 3);
    let prefilled = started.map(|started| started.map(|s| (s.prompt, s.prefilled)));
    let whole = |i: usize| Some((prompts[i].len(), prompts[i].len()));
    let cached = Some((prompts[2].len(), 8));
    assert_eq!(prefilled, [whole(0), whole(1), cached, whole(3)]);
    drop(scheduler);
    model.set_stepping(Stepping::Eager);
    for (i, prompt) in prompts.iter().enumerate() {
        let mut noise = SeededNoise::new(seeds[i]);
        let sampling = Sampling::default();
        let expected = model
            .generate(prompt, &sampling, max_patches[i], &mut noise)
            .unwrap();
        let patches = Tensor::cat(got[i].clone(), 0);
        assert_eq!(patches.dims(), expected.dims(), "request {i}");
        assert_close(patches, expected, ATOL, RTOL);
    }
}

/// A scheduler dropped while its requests generate, as the service's warm-up drops it, leaves its
/// graphs for the next, whose requests still come out as each does alone.
fn requests_go_on_as_alone_after_a_scheduler_left_off(stepping: Stepping) {
    let fixture = fixture!("voxcpm_generate");
    let mut model = model(&fixture);
    let prompts: Vec<Prompt> = (0..5)
        .map(|i| {
            prompt(
                i,
                2 + i as usize % 3,
                &[5, 17, 42, 9],
                &[77, 3, 64][..1 + i as usize % 3],
            )
        })
        .collect();
    let sampling = Sampling::default();
    let expected: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let mut noise = SeededNoise::new(10 + i as u64);
            model.generate(prompt, &sampling, 8, &mut noise).unwrap()
        })
        .collect();
    model.set_stepping(stepping);
    let mut first = Scheduler::new(&model, PromptCache::new(usize::MAX), 8);
    for (i, prompt) in prompts.iter().enumerate().take(4) {
        let noise = SeededNoise::new(20 + i as u64);
        first.add(i, prompt, &sampling, 8, noise).unwrap();
    }
    for _ in 0..4 {
        first.step();
    }
    drop(first);
    let mut scheduler = Scheduler::new(&model, PromptCache::new(usize::MAX), 8);
    for (i, prompt) in prompts.iter().enumerate() {
        let noise = SeededNoise::new(10 + i as u64);
        scheduler.add(i, prompt, &sampling, 8, noise).unwrap();
    }
    let mut got: Vec<Vec<Tensor<3>>> = vec![vec![]; prompts.len()];
    while scheduler.requests() > 0 {
        for patch in scheduler.step() {
            got[patch.key].push(patch.patch.unwrap());
        }
    }
    for (i, expected) in expected.into_iter().enumerate() {
        let patches = Tensor::cat(got[i].clone(), 0);
        assert_eq!(patches.dims(), expected.dims(), "request {i}");
        assert_close(patches, expected, ATOL, RTOL);
    }
}

#[test]
fn requests_go_on_as_alone_after_a_scheduler_left_off_by_the_stable_step() {
    requests_go_on_as_alone_after_a_scheduler_left_off(Stepping::Stable);
}

#[test]
fn requests_go_on_as_alone_after_a_scheduler_left_off_by_replays() {
    requests_go_on_as_alone_after_a_scheduler_left_off(Stepping::Captured);
}

/// Noise that is NaN from the `from`th patch on.
struct Diverging {
    noise: SeededNoise,
    left: usize,
}

impl Noise for Diverging {
    fn next(&mut self, len: usize) -> Vec<f32> {
        let noise = self.noise.next(len);
        if self.left == 0 {
            return vec![f32::NAN; len];
        }
        self.left -= 1;
        noise
    }
}

#[test]
fn a_scheduler_generates_each_request_as_alone_by_the_eager_step() {
    a_scheduler_generates_each_request_as_alone(Stepping::Eager);
}

#[test]
fn a_scheduler_generates_each_request_as_alone_by_the_stable_step() {
    a_scheduler_generates_each_request_as_alone(Stepping::Stable);
}

#[test]
fn a_scheduler_generates_each_request_as_alone_by_replays() {
    a_scheduler_generates_each_request_as_alone(Stepping::Captured);
}

/// A request whose third patch comes out NaN ends with an error in its place, and the request
/// beside it goes on as alone, as `stepping` steps them.
fn a_patch_that_is_not_finite_ends_its_request_with_an_error(stepping: Stepping) {
    let fixture = fixture!("voxcpm_generate");
    let mut model = model(&fixture);
    model.set_stepping(stepping);
    let voice: Vec<f32> = fixture
        .tensor::<2>("input.voice", &device())
        .into_data()
        .iter()
        .collect();
    let tokens = [5, 17, 42, 9, 77, 3, 64];
    let prompt = Prompt::cloning(&tokens, &voice, &tiny()).unwrap();
    let sampling = Sampling::default();
    let diverging = || Diverging {
        noise: SeededNoise::new(3),
        left: 2,
    };
    let alone = model.generate(&prompt, &sampling, 8, &mut diverging());
    assert!(matches!(alone, Err(Error::NotFinite(2))), "{alone:?}");

    let mut scheduler = Scheduler::new(&model, PromptCache::new(usize::MAX), 8);
    scheduler
        .add(0, &prompt, &sampling, 8, diverging())
        .unwrap();
    let mut other = vec![];
    let mut seeded = Some(SeededNoise::new(3));
    let mut failed = vec![];
    while scheduler.requests() > 0 {
        let patches = scheduler.step();
        // The other request joins after the first's first patch, from a noise of its own.
        if let Some(noise) = seeded.take() {
            let noise = Diverging {
                noise,
                left: usize::MAX,
            };
            scheduler.add(1, &prompt, &sampling, 8, noise).unwrap();
        }
        for patch in patches {
            match (patch.key, patch.patch) {
                (0, Ok(_)) => assert!(failed.is_empty()),
                (0, Err(error)) => {
                    assert!(patch.last);
                    failed.push(error);
                }
                (_, patch) => other.push(patch.unwrap()),
            }
        }
    }
    assert!(matches!(failed[..], [Error::NotFinite(2)]), "{failed:?}");
    drop(scheduler);
    model.set_stepping(Stepping::Eager);
    let expected = model
        .generate(&prompt, &sampling, 8, &mut SeededNoise::new(3))
        .unwrap();
    assert_close(Tensor::cat(other, 0), expected, ATOL, RTOL);
}

#[test]
fn a_patch_that_is_not_finite_ends_its_request_with_an_error_by_the_eager_step() {
    a_patch_that_is_not_finite_ends_its_request_with_an_error(Stepping::Eager);
}

#[test]
fn a_patch_that_is_not_finite_ends_its_request_with_an_error_by_replays() {
    a_patch_that_is_not_finite_ends_its_request_with_an_error(Stepping::Captured);
}
