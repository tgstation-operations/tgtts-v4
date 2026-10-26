//! VoxCPM2 as the service's [`Synthesizer`].

use std::{fs, num::NonZeroUsize, path::Path, time::Instant};

use burn::tensor::{Device, Tensor};
use dsp::RATE;
use tracing::info;
use voices::{Latent, Voices};
use voxcpm::{Config, Dtypes, Params, Prompt, SeededNoise, TextToSpeech};

use crate::synth::{self, Speech, Synthesizer};

/// The line the warm-up gives each voice to find the shortest and longest prompts, as long as a
/// median line.
const WARM_UP_LINE: &str = "Hello there, could someone please come to medbay? I need help.";

/// The patches the warm-up generates from one short of each start: enough for the KV length to
/// take each alignment to 16 positions from the start, a multiple of 16, as it goes past it,
/// for each of which autotune chose kernels anew on a GPU.
const WARM_UP_PATCHES: usize = 9;

/// How far below each start the warm-up prefills again, beside the prefill one short of it that
/// it generates from: for the prompt length to take each alignment to 16 positions, which on a
/// GPU cost up to 2 s each the first time.
const WARM_UP_ALIGNMENTS: [usize; 3] = [2, 4, 8];

/// VoxCPM2, synthesizing with the same parameters for every request.
#[derive(Debug)]
pub struct VoxCpm {
    tts: TextToSpeech,
    params: Params,
    samples_per_patch: usize,
    patch_size: usize,
    feat_dim: usize,
    device: Device,
}

impl VoxCpm {
    /// Loads the VoxCPM2 checkpoint in `dir`, each part in its dtype from `dtypes`, on
    /// `device`, to synthesize with `params`, refusing one whose audio is not at the service's
    /// rate before it loads the weights.
    pub fn load(
        dir: &Path,
        dtypes: &Dtypes,
        device: &Device,
        params: Params,
    ) -> Result<Self, voxcpm::Error> {
        let path = dir.join("config.json");
        let json = fs::read_to_string(&path).map_err(|e| voxcpm::Error::Io(path, e))?;
        let config = Config::from_json(&json)?;
        let rate = config.audio_vae().out_sample_rate;
        if rate != RATE {
            let reason = format!("out_sample_rate {rate} where the service works at {RATE} Hz");
            return Err(voxcpm::Error::Unsupported(reason));
        }
        let hop: usize = config.audio_vae().decoder_rates.iter().product();
        Ok(Self {
            tts: TextToSpeech::load(dir, dtypes, device)?,
            params,
            samples_per_patch: config.patch_size * hop,
            patch_size: config.patch_size,
            feat_dim: config.feat_dim,
            device: device.clone(),
        })
    }

    /// Runs the shapes synthesis takes with `voices`, so that a GPU's autotune, which chooses
    /// the kernels of each shape the first time it runs, has chosen them before a request waits
    /// on it: prefills from the shortest prompt the voices give a median line to the longest,
    /// generation past each power of two the KV length goes past before the longest's last
    /// patch, where it takes other kernels, and the audio decoder's every chunk length. Logs
    /// what it ran and how long it took.
    ///
    /// # Errors
    ///
    /// If a voice cannot be read or spoken.
    pub fn warm_up(&self, voices: &Voices) -> Result<(), synth::Error> {
        let started = Instant::now();
        let mut prompts = Vec::new();
        for name in voices.names() {
            prompts.push(self.tts.prompt(WARM_UP_LINE, &voices.latent(name)?)?);
        }
        let shortest = prompts.iter().map(Prompt::len).min().ok_or("no voices")?;
        let longest = prompts
            .iter()
            .max_by_key(|prompt| prompt.len())
            .expect("a voice");
        let max_patches = self.params.max_patches;
        let starts = warm_up_starts(shortest, longest.len(), max_patches);
        let (model, sampling) = (self.tts.model(), &self.params.sampling);
        let mut noise = SeededNoise::new(0);
        for &start in &starts {
            model.start(&resized(longest, start), max_patches)?;
            let mut generation = model.start(&resized(longest, start - 1), max_patches)?;
            for _ in 0..WARM_UP_PATCHES.min(max_patches) {
                let step = generation.sample(model.noise(&mut noise), sampling);
                generation.advance(step.patch);
            }
            for below in WARM_UP_ALIGNMENTS {
                model.start(&resized(longest, start - below), max_patches)?;
            }
        }
        let generated = started.elapsed();
        let silent = Latent {
            frames: Vec::new(),
            text: String::new(),
        };
        let chunks = self.params.decode_chunk_frames.get();
        for chunk in 1..=chunks {
            let params = Params {
                decode_chunk_frames: NonZeroUsize::new(chunk).expect("not zero"),
                ..self.params.clone()
            };
            let shape = [
                chunk.div_ceil(self.patch_size),
                self.patch_size,
                self.feat_dim,
            ];
            let patches = Tensor::zeros(shape, &self.device);
            self.tts.speech_decoder(&silent, &params).push(patches);
        }
        let seconds = started.elapsed().as_secs_f64();
        info!(
            prompts = %format!("{shortest}..={}", longest.len()),
            contexts = %format!("{}..={}", shortest + 1, longest.len() + max_patches),
            from = ?starts,
            patches = WARM_UP_PATCHES,
            decoder_chunks = %format!("1..={chunks}"),
            generation_seconds = generated.as_secs_f64(),
            decoder_seconds = seconds - generated.as_secs_f64(),
            seconds,
            "warmed up"
        );
        Ok(())
    }
}

/// The prompt lengths the warm-up starts from, in the order it prefills them: the shortest and
/// the longest, and each power of two the KV length goes past from the shortest to the longest's
/// last patch, each a multiple of 16 that is the first prefill of its power-of-two bucket.
///
/// Autotune chooses the attention's kernel for a bucket of lengths on the first it runs, and on
/// a GPU one chosen on a length that is not a multiple of 16 took up to 15 times as long on those
/// that are, with the KV cache as long as the prompt and 256 patches: 1.9 s to prefill where
/// others took 0.13 s. One chosen on a multiple of 16 kept every length of its bucket fast.
fn warm_up_starts(shortest: usize, longest: usize, max_patches: usize) -> Vec<usize> {
    let mut starts = vec![shortest.next_multiple_of(16), longest.next_multiple_of(16)];
    let mut boundary = (shortest + 1).next_power_of_two();
    while boundary < longest + max_patches {
        starts.push(boundary);
        boundary *= 2;
    }
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// `prompt` cut to `len` positions, or made as long by repeating its last, for the shapes it
/// runs at.
fn resized(prompt: &Prompt, len: usize) -> Prompt {
    let per_position = prompt.patches.len() / prompt.len();
    let last = prompt.len() - 1;
    let at = |i: usize| i.min(last);
    Prompt {
        tokens: (0..len).map(|i| prompt.tokens[at(i)]).collect(),
        patches: (0..len)
            .flat_map(|i| &prompt.patches[at(i) * per_position..(at(i) + 1) * per_position])
            .copied()
            .collect(),
        audio: (0..len).map(|i| prompt.audio[at(i)]).collect(),
    }
}

impl Synthesizer for VoxCpm {
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        let mut noise = SeededNoise::new(seed);
        let samples = self.tts.synthesize(text, voice, &self.params, &mut noise)?;
        Ok(Speech {
            patches: samples.len() / self.samples_per_patch,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::FloatDType;
    use serde_json::Value;

    use super::*;

    /// voxcpm's tiny config, at `rate`, as the checkpoint `dir` holds it, with nothing else.
    fn checkpoint(dir: &Path, rate: u32) {
        let json = include_str!("../../voxcpm/tests/fixtures/voxcpm.json");
        let mut config: Value = serde_json::from_str(json).unwrap();
        config["audio_vae_config"]["out_sample_rate"] = rate.into();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    fn load(dir: &Path) -> voxcpm::Error {
        let f32 = FloatDType::F32;
        let dtypes = Dtypes {
            lm: f32,
            local_encoder: f32,
            local_dit: f32,
            audio_decoder: f32,
        };
        VoxCpm::load(dir, &dtypes, &Device::flex(), Params::default()).unwrap_err()
    }

    #[test]
    fn warms_up_from_each_power_of_two_the_kv_length_crosses() {
        assert_eq!(warm_up_starts(110, 579, 256), [112, 128, 256, 512, 592]);
        assert_eq!(
            warm_up_starts(100, 900, 256),
            [112, 128, 256, 512, 912, 1024]
        );
        assert_eq!(warm_up_starts(127, 200, 10), [128, 208]);
        assert_eq!(warm_up_starts(128, 200, 10), [128, 208]);
        assert_eq!(warm_up_starts(300, 300, 12), [304]);
    }

    #[test]
    fn resizes_a_prompt_by_its_last_position() {
        let prompt = Prompt {
            tokens: vec![7, 8, 0],
            patches: vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            audio: vec![false, false, true],
        };
        let cut = resized(&prompt, 2);
        assert_eq!(
            (cut.tokens, cut.patches, cut.audio),
            (vec![7, 8], vec![0.0; 4], vec![false, false])
        );
        let longer = resized(&prompt, 4);
        assert_eq!(longer.tokens, [7, 8, 0, 0]);
        assert_eq!(longer.patches[4..], [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(longer.audio, [false, false, true, true]);
    }

    #[test]
    fn refuses_a_model_at_another_rate() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(dir.path(), 44100);
        let error = load(dir.path()).to_string();
        assert!(error.contains("out_sample_rate 44100"), "{error}");
        // At the service's rate it goes on, to what the tiny config fails on in TextToSpeech.
        checkpoint(dir.path(), RATE);
        let error = load(dir.path()).to_string();
        assert!(!error.contains("out_sample_rate"), "{error}");
    }
}
