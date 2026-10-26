//! VoxCPM2 as the service's [`Synthesizer`].

use std::{
    collections::HashMap,
    fs,
    num::NonZeroUsize,
    path::Path,
    sync::{
        Arc,
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use burn::tensor::{Device, Tensor};
use dsp::RATE;
use tracing::{error, info};
use voices::{Latent, Voices};
use voxcpm::{
    Config, Dtypes, Params, Patch, Prompt, PromptCache, Scheduler, SeededNoise, TextToSpeech,
};

use crate::synth::{self, Speech, Synthesizer};

/// The line the warm-up gives each voice to find the shortest and longest prompts, as long as a
/// median line.
const WARM_UP_LINE: &str = "Hello there, could someone please come to medbay? I need help.";

/// The patches each warm-up request generates, from a prompt two short of a power of two: its
/// KV length goes past that power of two, where a GPU takes other kernels, and on to the
/// alignments past it.
const WARM_UP_PATCHES: usize = 9;

/// The bytes on the device the prompt cache keeps of the voices: on a GPU in f16 a median voice
/// takes about 11 MB, so about 90 of them.
const PROMPT_CACHE_BYTES: usize = 1 << 30;

/// The positions of prompts run between two patches while other requests generate: on a GPU in
/// f16 a run of 128 takes about as long as a patch of one request.
const PREFILL_POSITIONS: usize = 128;

/// How requests are batched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Batching {
    /// How long the first request to an idle model waits for others to start with it.
    pub window: Duration,
    /// The most requests at once, which stop the wait.
    pub requests: usize,
}

/// VoxCPM2, synthesizing with the same parameters for every request: the requests generate
/// together, in a batch a thread of its own runs, and each decodes its own speech.
#[derive(Debug)]
pub struct VoxCpm {
    tts: Arc<TextToSpeech>,
    config: Config,
    params: Params,
    batching: Batching,
    requests: Sender<Request>,
    samples_per_patch: usize,
    device: Device,
}

/// A prompt to generate from, and where its patches go.
struct Request {
    prompt: Prompt,
    seed: u64,
    patches: PatchesSender,
}

impl VoxCpm {
    /// Loads the VoxCPM2 checkpoint in `dir`, each part in its dtype from `dtypes`, on
    /// `device`, to synthesize with `params` in batches as `batching` says, refusing one whose
    /// audio is not at the service's rate before it loads the weights.
    pub fn load(
        dir: &Path,
        dtypes: &Dtypes,
        device: &Device,
        params: Params,
        batching: Batching,
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
        let tts = Arc::new(TextToSpeech::load(dir, dtypes, device)?);
        let (requests, received) = mpsc::channel();
        let (generating, generation) = (tts.clone(), params.clone());
        thread::Builder::new()
            .name("voxcpm".into())
            .spawn(move || generate(&generating, &generation, batching, &received))
            .map_err(|e| voxcpm::Error::Io("a thread to generate on".into(), e))?;
        Ok(Self {
            tts,
            samples_per_patch: config.patch_size * hop,
            config,
            params,
            batching,
            requests,
            device: device.clone(),
        })
    }

    /// Runs the shapes synthesis takes with `voices`, so that a GPU's autotune, which chooses
    /// the kernels of each shape the first time it runs, has chosen them before a request waits
    /// on it: as many requests as run at once in voices from the shortest prompt a median line
    /// gives to the longest, twice, the second time from what the prompt cache kept; one request
    /// and as many as run at once going past each power of two the KV length goes past before
    /// the longest's last patch, where a GPU takes other kernels; and the audio decoder's every
    /// chunk length. Logs what it ran and how long it took.
    ///
    /// # Errors
    ///
    /// If a voice cannot be read or spoken.
    pub fn warm_up(&self, voices: &Voices) -> Result<(), synth::Error> {
        let started = Instant::now();
        let mut prompts = Vec::new();
        for name in voices.names() {
            let latent = voices.latent(name)?;
            prompts.push((self.tts.prompt(WARM_UP_LINE, &latent)?, latent));
        }
        prompts.sort_by_key(|(prompt, _)| prompt.len());
        let (Some((shortest, voice)), Some((longest, _))) = (prompts.first(), prompts.last())
        else {
            return Err("no voices".into());
        };
        let requests = self.batching.requests;
        let spread = (0..requests).map(|i| {
            let at = i * (prompts.len() - 1) / requests.saturating_sub(1).max(1);
            prompts[at].0.clone()
        });
        let spread: Vec<Prompt> = spread.collect();
        let mut cache = PromptCache::new(PROMPT_CACHE_BYTES);
        let mut runs = vec![spread.clone(), spread];
        let mut boundary = (shortest.len() + 1).next_power_of_two();
        let mut boundaries = vec![];
        while boundary < longest.len() + self.params.max_patches {
            let length = |less: usize| (boundary - 2 - less).max(shortest.len());
            let (config, voice) = (&self.config, voice.frames.as_flattened());
            runs.push(vec![lengthened(shortest, voice, length(0), config)?]);
            let around = (0..requests).map(|i| lengthened(shortest, voice, length(i), config));
            runs.push(around.collect::<Result<_, _>>()?);
            boundaries.push(boundary);
            boundary *= 2;
        }
        let max_patches = WARM_UP_PATCHES.min(self.params.max_patches);
        for run in &runs {
            let model = self.tts.model();
            let mut scheduler = Scheduler::new(model, cache, PREFILL_POSITIONS);
            for (i, prompt) in run.iter().enumerate() {
                let noise = SeededNoise::new(i as u64);
                scheduler.add(i, prompt, &self.params.sampling, max_patches, noise)?;
            }
            while scheduler.requests() > 0 {
                for patch in scheduler.step() {
                    patch.patch?;
                }
            }
            cache = scheduler.into_cache();
        }
        let generated = started.elapsed();
        let silent = Latent {
            frames: Vec::new(),
            text: String::new(),
        };
        let chunks = self.params.decode_chunk_frames.get();
        let patch_size = self.config.patch_size;
        for chunk in 1..=chunks {
            let params = Params {
                decode_chunk_frames: NonZeroUsize::new(chunk).expect("not zero"),
                ..self.params.clone()
            };
            let shape = [chunk.div_ceil(patch_size), patch_size, self.config.feat_dim];
            let patches = Tensor::zeros(shape, &self.device);
            self.tts.speech_decoder(&silent, &params).push(patches);
        }
        let seconds = started.elapsed().as_secs_f64();
        info!(
            prompts = %format!("{}..={}", shortest.len(), longest.len()),
            requests,
            past = ?boundaries,
            patches = max_patches,
            decoder_chunks = %format!("1..={chunks}"),
            generation_seconds = generated.as_secs_f64(),
            decoder_seconds = seconds - generated.as_secs_f64(),
            seconds,
            "warmed up"
        );
        Ok(())
    }
}

/// A prompt of `len` positions in the voice of `prompt`, its latent frames `voice` one after
/// another, its text as long as it has to be by going round the text of `prompt` again.
fn lengthened(
    prompt: &Prompt,
    voice: &[f32],
    len: usize,
    config: &Config,
) -> Result<Prompt, voxcpm::Error> {
    let patches = voice.len() / (config.patch_size * config.feat_dim);
    let text = &prompt.tokens[2 + patches..prompt.len() - patches - 1];
    let tokens: Vec<u32> = text
        .iter()
        .cycle()
        .take(len - 3 - 2 * patches)
        .copied()
        .collect();
    Prompt::cloning(&tokens, voice, config)
}

/// Generates the patches of each request `requests` brings, in a batch, until every sender of
/// requests is gone: an idle model waits up to `batching.window` after a request for others to
/// start with it.
fn generate(tts: &TextToSpeech, params: &Params, batching: Batching, requests: &Receiver<Request>) {
    let cache = PromptCache::new(PROMPT_CACHE_BYTES);
    let mut batch = Batch {
        scheduler: Scheduler::new(tts.model(), cache, PREFILL_POSITIONS),
        generating: HashMap::new(),
        next: 0,
        params,
    };
    loop {
        if batch.scheduler.requests() == 0 {
            let Ok(request) = requests.recv() else {
                return;
            };
            batch.add(request);
            let until = Instant::now() + batching.window;
            while batch.scheduler.requests() < batching.requests {
                let Some(left) = until.checked_duration_since(Instant::now()) else {
                    break;
                };
                match requests.recv_timeout(left) {
                    Ok(request) => batch.add(request),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        }
        while let Ok(request) = requests.try_recv() {
            batch.add(request);
        }
        batch.step();
    }
}

/// The requests generating, each with where its patches go and those it has so far.
struct Batch<'a> {
    scheduler: Scheduler<'a, u64, SeededNoise>,
    generating: HashMap<u64, (PatchesSender, Vec<Tensor<3>>)>,
    next: u64,
    params: &'a Params,
}

type PatchesSender = Sender<Result<Vec<Tensor<3>>, voxcpm::Error>>;

impl Batch<'_> {
    fn add(&mut self, request: Request) {
        let (key, noise) = (self.next, SeededNoise::new(request.seed));
        self.next += 1;
        let (sampling, max_patches) = (&self.params.sampling, self.params.max_patches);
        match self
            .scheduler
            .add(key, &request.prompt, sampling, max_patches, noise)
        {
            Ok(()) => {
                self.generating.insert(key, (request.patches, vec![]));
            }
            Err(e) => {
                let _ = request.patches.send(Err(e));
            }
        }
    }

    fn step(&mut self) {
        for Patch { key, patch, last } in self.scheduler.step() {
            let (_, patches) = self.generating.get_mut(&key).expect("generating");
            let failed = match patch {
                Ok(patch) => {
                    patches.push(patch);
                    None
                }
                Err(e) => Some(e),
            };
            if last {
                let (sender, patches) = self.generating.remove(&key).expect("generating");
                let _ = sender.send(failed.map_or(Ok(patches), Err));
            }
        }
    }
}

impl Synthesizer for VoxCpm {
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        let prompt = self.tts.prompt(text, voice)?;
        let (patches, received) = mpsc::channel();
        let request = Request {
            prompt,
            seed,
            patches,
        };
        let stopped = "the thread VoxCPM2 generates on has stopped";
        self.requests.send(request).map_err(|_| stopped)?;
        let patches = received.recv().map_err(|_| {
            error!("{stopped}");
            stopped
        })??;
        let samples = self
            .tts
            .speech_decoder(voice, &self.params)
            .push(Tensor::cat(patches, 0));
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
        let batching = Batching {
            window: Duration::ZERO,
            requests: 1,
        };
        VoxCpm::load(dir, &dtypes, &Device::flex(), Params::default(), batching).unwrap_err()
    }

    #[test]
    fn lengthens_a_prompt_by_going_round_its_text() {
        let config = Config::from_json(include_str!("../../voxcpm/tests/fixtures/voxcpm.json"));
        let config = config.unwrap();
        let voice: Vec<f32> = (0..2 * config.patch_size * config.feat_dim)
            .map(|i| i as f32)
            .collect();
        let prompt = Prompt::cloning(&[7, 8, 9], &voice, &config).unwrap();
        let longer = lengthened(&prompt, &voice, prompt.len() + 4, &config).unwrap();
        let text = Prompt::cloning(&[7, 8, 9, 7, 8, 9, 7], &voice, &config).unwrap();
        assert_eq!(longer, text);
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
