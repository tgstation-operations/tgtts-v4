//! VoxCPM2 as the service's [`Synthesizer`].

use std::{
    collections::{HashMap, HashSet},
    fs, io,
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
use tracing::{debug, error, info, warn};
use voices::{Latent, Voices};
use voxcpm::{
    CHUNK_FRAMES, ChunkDecoder, Config, Dtypes, GraphCounts, PREFILL_BATCH, Params, Patch, Prompt,
    PromptCache, Scheduler, SeededNoise, Stepping, TextToSpeech,
};

use crate::{
    device,
    synth::{self, Speech, Start, Synthesizer},
};

/// The line the warm-up gives each voice to find the shortest and longest prompts, as long as a
/// median line.
const WARM_UP_LINE: &str = "Hello there, could someone please come to medbay? I need help.";

/// The patches each warm-up request generates, from a prompt two short of a power of two: its
/// KV length goes past that power of two, where a GPU takes other kernels, and on to the
/// alignments past it.
const WARM_UP_PATCHES: usize = 9;

/// The bytes on the device the prompt cache keeps of the voices: on a GPU in f16 a median voice
/// takes about 11 MB, so about 90 of them.
pub const PROMPT_CACHE_BYTES: usize = 1 << 30;

/// The positions of waiting prompts run between two patches while other requests generate,
/// unless the configuration says otherwise, past which only the first runs.
pub const PREFILL_POSITIONS: usize = 128;

/// How requests are batched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Batching {
    /// How long the first request to an idle model waits for others to start with it.
    pub window: Duration,
    /// The most requests at once, which stop the wait.
    pub requests: usize,
    /// Whether a batch steps by replaying graphs of its step where the device records them.
    pub graphs: bool,
    /// The positions of prompts run between two patches while other requests generate.
    pub prefill_positions: usize,
}

/// VoxCPM2, synthesizing with the same parameters for every request: the requests generate
/// together, in a batch a thread of its own runs, which loads the model and decodes their
/// speech too, a chunk at a time as their patches come, those of every request with a chunk to
/// decode at once. That thread is the only one to use the device, as cubecl gives each thread a
/// stream of its own, and work on several streams at once has faulted the GPU.
#[derive(Debug)]
pub struct VoxCpm {
    tts: Arc<TextToSpeech>,
    config: Config,
    params: Params,
    batching: Batching,
    jobs: Sender<Job>,
    samples_per_patch: usize,
}

/// What the thread that generates is asked to do.
enum Job {
    Generate(Request),
    /// Run what the warm-up runs and say how long generating took.
    WarmUp(WarmUp, Sender<Result<Duration, voxcpm::Error>>),
}

/// What the warm-up runs on the thread that generates.
struct WarmUp {
    /// Runs of prompts generated from in turn, a batch of each.
    runs: Vec<Vec<Prompt>>,
    /// The positions of the longest prompt, the patches of the longest voice, and the most
    /// requests that run at once.
    longest: usize,
    voice_patches: usize,
    requests: usize,
}

/// A prompt to generate from in a voice, and where its speech goes.
struct Request {
    prompt: Prompt,
    voice: Latent,
    seed: u64,
    /// When it was sent to the thread that generates.
    sent: Instant,
    speech: SpeechSender,
}

type SpeechSender = Sender<Result<(Vec<f32>, Option<Start>), voxcpm::Error>>;

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
        let (jobs, received) = mpsc::channel();
        let (loaded, loading) = mpsc::channel();
        let (dir, dtypes, on) = (dir.to_owned(), *dtypes, device.clone());
        let (model_config, model_params) = (config.clone(), params.clone());
        // The weights are loaded on the thread that uses them, whose stream holds them.
        thread::Builder::new()
            .name("voxcpm".into())
            .spawn(move || {
                let tts = match TextToSpeech::load(&dir, &dtypes, &on) {
                    Ok(mut tts) => {
                        if batching.graphs {
                            tts.set_stepping(Stepping::for_device(&on));
                        }
                        Arc::new(tts)
                    }
                    Err(e) => {
                        let _ = loaded.send(Err(e));
                        return;
                    }
                };
                let _ = loaded.send(Ok(tts.clone()));
                let generating = Generating {
                    tts,
                    config: model_config,
                    params: model_params,
                    device: on,
                    prefill_positions: batching.prefill_positions,
                };
                generate(&generating, batching, &received);
            })
            .map_err(|e| voxcpm::Error::Io("a thread to generate on".into(), e))?;
        let stopped = || {
            let stopped = io::ErrorKind::BrokenPipe.into();
            voxcpm::Error::Io("the thread VoxCPM2 loads on".into(), stopped)
        };
        let tts = loading.recv().map_err(|_| stopped())??;
        Ok(Self {
            tts,
            samples_per_patch: config.patch_size * hop,
            config,
            params,
            batching,
            jobs,
        })
    }

    /// How many graphs of its step the model has captured, and destroyed.
    pub fn graph_counts(&self) -> GraphCounts {
        self.tts.model().graph_counts()
    }

    /// Runs the shapes synthesis takes with `voices`, so that a GPU's autotune, which chooses
    /// the kernels of each shape the first time it runs, has chosen them before a request waits
    /// on it: requests at once in voices from the shortest prompt a median line gives to the
    /// longest, each number a graph of the step holds them in at each capacity of cache their
    /// prompts need, later runs going on from what the prompt cache kept of earlier ones; one
    /// request and as many as run at once going past each power of two the KV length goes past
    /// before the longest's last patch, where a GPU takes other kernels; each shape of prompts
    /// run together; and chunks of each power of two of requests at once through the decoder,
    /// all on the thread that generates. Logs what it ran and how long it took.
    ///
    /// The thread that generates keeps the graphs a GPU records of each number of requests and
    /// capacity of cache the warm-up reaches, which the requests after it replay.
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
        let lengths: Vec<usize> = prompts.iter().map(|(prompt, _)| prompt.len()).collect();
        let model = self.tts.model();
        let capacity = |len| model.graph_capacity(len, self.params.max_patches);
        let runs = bucket_runs(&lengths, requests, capacity);
        let run = |run: Vec<usize>| run.into_iter().map(|i| prompts[i].0.clone()).collect();
        let mut runs: Vec<Vec<Prompt>> = runs.into_iter().map(run).collect();
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
        let voice_patches = (voices.names())
            .map(|name| Ok(voices.latent(name)?.frames.len() / self.config.patch_size))
            .collect::<Result<Vec<_>, voices::Error>>()?;
        let warm_up = WarmUp {
            runs,
            longest: longest.len(),
            voice_patches: voice_patches.into_iter().max().unwrap_or(1),
            requests,
        };
        let (done, warmed) = mpsc::channel();
        let stopped = "the thread VoxCPM2 generates on has stopped";
        self.jobs
            .send(Job::WarmUp(warm_up, done))
            .map_err(|_| stopped)?;
        let generated = warmed.recv().map_err(|_| stopped)??;
        let seconds = started.elapsed().as_secs_f64();
        info!(
            prompts = %format!("{}..={}", shortest.len(), longest.len()),
            requests,
            past = ?boundaries,
            patches = WARM_UP_PATCHES,
            decoder_rows = %format!("1..={}, powers of two", requests.next_power_of_two()),
            generation_seconds = generated.as_secs_f64(),
            graphs_captured = self.graph_counts().captured,
            kernels_tuned = device::tuned(),
            decoder_seconds = seconds - generated.as_secs_f64(),
            seconds,
            "warmed up"
        );
        Ok(())
    }
}

/// The runs of prompts that join a scheduler together the warm-up runs, as rows, positions held
/// and positions run: for each number of rows up to `requests` and of positions up to those of
/// the `longest` prompt, each a power of two of at least 32, as autotune rounds the shapes it
/// chooses kernels for, that come to at most [`PREFILL_BATCH`] positions, each after enough
/// held positions to come to each power of two of keys from those positions to the longest's.
/// Each is the largest of the shapes autotune rounds to it, which also holds the most memory.
fn prefill_shapes(longest: usize, requests: usize) -> Vec<(usize, usize, usize)> {
    let powers = |from: usize, to: usize| {
        (0..)
            .map(move |power| 1 << power)
            .skip_while(move |&n| n < from)
            .take_while(move |&n| n <= to)
    };
    let (rows, positions) = (requests.next_power_of_two(), longest.next_power_of_two());
    let mut shapes = vec![];
    for rows in powers(1, rows) {
        for len in powers(32, positions).filter(|&len| rows * len <= PREFILL_BATCH) {
            for keys in powers(len, positions) {
                shapes.push((rows, keys - len, len));
            }
        }
    }
    shapes
}

/// The prompts, by their index among `lengths` in ascending order, of each run of requests at once
/// the warm-up generates from: for each capacity of cache `capacity` gives the prompts and each
/// number of rows a graph of the step holds up to `requests` requests in, a run of prompts that
/// come to it, as many as it holds or `requests`, spread across the lengths that fit it, the
/// longest of them included. The runs go from the shortest prompts to the longest, each also a
/// prompt cache's next voices.
fn bucket_runs(
    lengths: &[usize],
    requests: usize,
    capacity: impl Fn(usize) -> usize,
) -> Vec<Vec<usize>> {
    let mut capacities: Vec<usize> = lengths.iter().map(|&len| capacity(len)).collect();
    capacities.dedup();
    let mut sizes: Vec<usize> = (0..)
        .map(|power| 1 << power)
        .take_while(|&rows| rows < requests)
        .collect();
    sizes.push(requests);
    let mut runs = vec![];
    for &holds in &capacities {
        let fits = lengths
            .iter()
            .filter(|&&len| capacity(len) <= holds)
            .count();
        for &size in &sizes {
            let at = |i: usize| fits - 1 - i * (fits - 1) / (size - 1).max(1);
            runs.push((0..size).map(at).collect());
        }
    }
    runs
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

/// What the thread that generates holds.
struct Generating {
    tts: Arc<TextToSpeech>,
    config: Config,
    params: Params,
    device: Device,
    prefill_positions: usize,
}

/// Does each job `jobs` brings, generating the patches of requests in a batch and decoding the
/// speech of each, until every sender of jobs is gone: an idle model waits up to
/// `batching.window` after a request for others to start with it.
fn generate(generating: &Generating, batching: Batching, jobs: &Receiver<Job>) {
    let cache = PromptCache::new(PROMPT_CACHE_BYTES);
    let mut batch = Batch {
        scheduler: Scheduler::new(generating.tts.model(), cache, generating.prefill_positions),
        generating: HashMap::new(),
        next: 0,
        on: generating,
        stepped: Instant::now(),
        warmed: None,
    };
    loop {
        if batch.scheduler.requests() == 0 {
            let Ok(job) = jobs.recv() else {
                return;
            };
            batch.take(job);
            let until = Instant::now() + batching.window;
            while (1..batching.requests).contains(&batch.scheduler.requests()) {
                let Some(left) = until.checked_duration_since(Instant::now()) else {
                    break;
                };
                match jobs.recv_timeout(left) {
                    Ok(job) => batch.take(job),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
            batch.stepped = Instant::now();
        }
        while let Ok(job) = jobs.try_recv() {
            batch.take(job);
        }
        batch.step();
    }
}

impl Generating {
    /// Generates from each run of `prompts` in turn, a batch of each, as `params` says, from a
    /// prompt cache of their own, until each request has [`WARM_UP_PATCHES`] patches, in caches
    /// of the capacity a request has. Then it runs each shape of the runs of prompts that join
    /// together [`prefill_shapes`] gives and embeds a voice of each power of two of patches,
    /// after every graph is captured, each shape's memory given back before the next, so that
    /// the pools hold what the largest takes beside them; and decodes two chunks of silence for
    /// each power of two of requests at once. Gives how long generating took.
    fn warm_up(&self, warm_up: &WarmUp) -> Result<Duration, voxcpm::Error> {
        let started = Instant::now();
        let (tts, params) = (&*self.tts, &self.params);
        let mut cache = PromptCache::new(PROMPT_CACHE_BYTES);
        for run in &warm_up.runs {
            let mut scheduler = Scheduler::new(tts.model(), cache, self.prefill_positions);
            for (i, prompt) in run.iter().enumerate() {
                let noise = SeededNoise::new(i as u64);
                scheduler.add(i, prompt, &params.sampling, params.max_patches, noise)?;
            }
            let mut left = vec![WARM_UP_PATCHES; run.len()];
            while left.iter().any(|&left| left > 0) && scheduler.requests() > 0 {
                for patch in scheduler.step() {
                    patch.patch?;
                    left[patch.key] = match patch.last {
                        true => 0,
                        false => left[patch.key].saturating_sub(1),
                    };
                }
            }
            cache = scheduler.into_cache();
        }
        let model = tts.model();
        for (rows, held, len) in prefill_shapes(warm_up.longest, warm_up.requests) {
            model.warm_prefill(rows, held, len);
            // What each shape took goes back before the next, or the pools would hold the
            // memory of all of them at once, which no step of serving takes.
            self.device.memory_cleanup();
        }
        let mut patches = 1;
        while patches < warm_up.voice_patches {
            patches *= 2;
            model.warm_voice(patches);
        }
        let generated = started.elapsed();
        let silent = Latent {
            frames: Vec::new(),
            text: String::new(),
        };
        let patch_size = self.config.patch_size;
        let shape = [CHUNK_FRAMES / patch_size, patch_size, self.config.feat_dim];
        // Autotune rounds the rows up to a power of two, as it rounds those of prompts.
        let rows = (0..).map(|power| 1 << power);
        let rows = rows.take_while(|&rows| rows <= warm_up.requests.next_power_of_two());
        for rows in rows {
            let mut decoders: Vec<_> = (0..rows).map(|_| tts.chunk_decoder(&silent)).collect();
            // A first chunk, then one after it.
            for _ in 0..2 {
                for decoder in &mut decoders {
                    decoder.push(Tensor::zeros(shape, &self.device));
                }
                let mut decoding: Vec<_> = decoders.iter_mut().collect();
                ChunkDecoder::decode_together(&mut decoding, &vec![false; rows]);
            }
        }
        Ok(generated)
    }
}

/// The requests generating, each with its voice, where its speech goes, how it started and the
/// patches it has so far.
struct Batch<'a> {
    scheduler: Scheduler<'a, u64, SeededNoise>,
    generating: HashMap<u64, Generation<'a>>,
    next: u64,
    on: &'a Generating,
    /// When the last step returned.
    stepped: Instant,
    /// The graphs of the step captured and destroyed, and the kernels autotune tuned, when the
    /// warm-up ended, and since, if it has.
    warmed: Option<(GraphCounts, usize)>,
}

struct Generation<'a> {
    request: Request,
    start: Option<Start>,
    decoder: ChunkDecoder<'a>,
    samples: Vec<f32>,
}

impl Batch<'_> {
    fn take(&mut self, job: Job) {
        match job {
            Job::Generate(request) => self.add(request),
            Job::WarmUp(warm_up, done) => {
                let warmed = self.on.warm_up(&warm_up);
                self.warmed = Some((self.on.tts.model().graph_counts(), device::tuned()));
                let _ = done.send(warmed);
            }
        }
    }

    fn add(&mut self, request: Request) {
        let (key, noise) = (self.next, SeededNoise::new(request.seed));
        self.next += 1;
        let params = &self.on.params;
        let (sampling, max_patches) = (&params.sampling, params.max_patches);
        match self
            .scheduler
            .add(key, &request.prompt, sampling, max_patches, noise)
        {
            Ok(()) => {
                let generation = Generation {
                    decoder: self.on.tts.chunk_decoder(&request.voice),
                    request,
                    start: None,
                    samples: vec![],
                };
                self.generating.insert(key, generation);
            }
            Err(e) => {
                let _ = request.speech.send(Err(e));
            }
        }
    }

    fn step(&mut self) {
        let between = self.stepped.elapsed();
        let patches = self.scheduler.step();
        let stepped = Instant::now();
        let mut decoding = vec![];
        for Patch {
            key,
            patch,
            last,
            started,
        } in patches
        {
            let generation = self.generating.get_mut(&key).expect("generating");
            if let Some(started) = started {
                generation.start = Some(Start {
                    waited: generation.request.sent.elapsed(),
                    prompt: started.prompt,
                    prefilled: started.prefilled,
                });
            }
            match patch {
                Ok(patch) => {
                    generation.decoder.push(patch);
                    if last || generation.decoder.ready() {
                        decoding.push((key, last));
                    }
                }
                Err(e) => {
                    let generation = self.generating.remove(&key).expect("generating");
                    let _ = generation.request.speech.send(Err(e));
                }
            }
        }
        if !decoding.is_empty() {
            let keys: HashSet<u64> = decoding.iter().map(|&(key, _)| key).collect();
            let (mut decoders, mut last, mut whose) = (vec![], vec![], vec![]);
            for (key, generation) in &mut self.generating {
                if keys.contains(key) {
                    let ends = decoding.iter().any(|&(at, ends)| at == *key && ends);
                    decoders.push(&mut generation.decoder);
                    last.push(ends);
                    whose.push((*key, ends));
                }
            }
            let decoded = ChunkDecoder::decode_together(&mut decoders, &last);
            for ((key, ends), samples) in whose.into_iter().zip(decoded) {
                let generation = self.generating.get_mut(&key).expect("generating");
                generation.samples.extend(samples);
                if ends {
                    let generation = self.generating.remove(&key).expect("generating");
                    let speech = (generation.samples, generation.start);
                    let _ = generation.request.speech.send(Ok(speech));
                }
            }
        }
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let report = self.scheduler.last_step();
        let (graphs, tuned) = (self.on.tts.model().graph_counts(), device::tuned());
        if let Some((warmed, before)) = self.warmed.replace((graphs, tuned)) {
            if warmed != graphs {
                // Each capture after the warm-up is a pause of a few steps, and a destroyed graph
                // is what faulted the GPU on Vulkan.
                warn!(
                    captured = graphs.captured - warmed.captured,
                    destroyed = graphs.destroyed - warmed.destroyed,
                    "graphs of the step captured or destroyed after the warm-up"
                );
            }
            if tuned != before {
                // Cubecl logs the key of each just before; every request waits while it times
                // its candidates.
                warn!(
                    kernels = tuned - before,
                    "kernels autotuned after the warm-up, of shapes it did not run"
                );
            }
        }
        self.stepped = Instant::now();
        debug!(
            rows = report.rows,
            waiting = report.waiting,
            joined = report.joined,
            positions = report.positions,
            prefill_ms = ms(report.prefill),
            step_ms = ms(report.step),
            read_ms = ms(report.read),
            retire_ms = ms(report.retire),
            decode_ms = ms(self.stepped - stepped),
            between_ms = ms(between),
            graphs_captured = graphs.captured,
            "stepped"
        );
    }
}

impl Synthesizer for VoxCpm {
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        let prompt = self.tts.prompt(text, voice)?;
        let (speech, received) = mpsc::channel();
        let request = Request {
            prompt,
            voice: voice.clone(),
            seed,
            sent: Instant::now(),
            speech,
        };
        let stopped = "the thread VoxCPM2 generates on has stopped";
        self.jobs
            .send(Job::Generate(request))
            .map_err(|_| stopped)?;
        let (samples, start) = received.recv().map_err(|_| {
            error!("{stopped}");
            stopped
        })??;
        Ok(Speech {
            patches: samples.len() / self.samples_per_patch,
            samples,
            start,
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
            graphs: true,
            prefill_positions: PREFILL_POSITIONS,
        };
        VoxCpm::load(dir, &dtypes, &Device::flex(), Params::default(), batching).unwrap_err()
    }

    /// Every shape autotune rounds a run of up to 12 prompts of up to 579 positions to, as many
    /// rows, keys and positions run as a power of two each, that the scheduler lets through.
    #[test]
    fn warms_up_each_shape_of_prompts_run_together() {
        let shapes = prefill_shapes(579, 12);
        let rounded = |n: usize| n.next_power_of_two();
        let mut keys: Vec<_> = shapes
            .iter()
            .map(|&(rows, held, len)| (rounded(rows), rounded(held + len), rounded(len)))
            .collect();
        let count = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), count, "a shape warmed up twice");
        for rows in [1, 2, 4, 8, 16] {
            for len in [32, 64, 128, 256, 512, 1024] {
                for held in 0..1024 - len {
                    let key = (rows, rounded(held + len), len);
                    let allowed = rows * len <= PREFILL_BATCH;
                    assert_eq!(keys.contains(&key), allowed, "{key:?}");
                }
            }
        }
        assert!(
            shapes
                .iter()
                .all(|&(rows, _, len)| rows * len <= PREFILL_BATCH)
        );
    }

    /// Prompts of three capacities and requests of up to 12 at once: each capacity is reached
    /// with 1, 2, 4, 8 and 12 requests, which a stepper holds in 16 rows.
    #[test]
    fn warms_each_graph_up_once() {
        let lengths = [300, 310, 400, 500, 700, 760, 900, 1500];
        let capacity = |len: usize| (len + 256).next_power_of_two();
        let runs = bucket_runs(&lengths, 12, capacity);
        let mut reached: Vec<(usize, usize)> = runs
            .iter()
            .map(|run| {
                let held = run.iter().map(|&i| capacity(lengths[i])).max().unwrap();
                (run.len().next_power_of_two(), held)
            })
            .collect();
        let count = reached.len();
        reached.sort_unstable();
        reached.dedup();
        assert_eq!(reached.len(), count, "a graph warmed up twice: {runs:?}");
        let expected: Vec<_> = [1024, 2048]
            .into_iter()
            .flat_map(|held| [1, 2, 4, 8, 16].map(|rows| (rows, held)))
            .collect();
        let mut expected = expected;
        expected.sort_unstable();
        assert_eq!(reached, expected);
        assert!(runs.iter().all(|run| run.len() <= 12));
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
