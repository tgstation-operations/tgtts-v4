//! VoxCPM2 on tts-rs's own Vulkan runtime as the service's [`Synthesizer`], on the `graph`
//! crate's generator, which records the model's work once and submits it a step at a time, a
//! patch of each request under way. A thread of its own owns the GPU and generates there on an
//! [`Engine`], the requests in the order they come and as many at once as the engine holds, the
//! generator sixteen. The prompt is built on the caller's thread, and the speech comes back
//! whole, or for a stream a run of audio at a time as it is decoded too. Each patch is sampled
//! from noise drawn from the request's seed.
//!
//! The generator keeps what a prompt computes of its voice for the next prompts in it, within
//! `prompt_cache_bytes`, so that a request in a voice it keeps runs only its own text and the
//! voice spoken again; it counts the voice's patches' embeddings, kept on the host, beside the
//! keys and values on the GPU.

use std::{
    collections::VecDeque,
    fs, mem,
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use dsp::RATE;
use graph::{
    generator::{Generator, Settings},
    kernels::Kernels,
    pth::Pth,
};
use tracing::{debug, error, info};
use voices::{Latent, Voices};
use voxcpm::{Config, Noise, Params, Prompt, Sampling, SeededNoise, Tokenizer};

use crate::synth::{self, Sink, Speech, Start, Synthesizer};

/// The line the warm-up speaks, as long as a median line.
const WARM_UP_LINE: &str = "Hello there, could someone please come to medbay? I need help.";

/// The patches the warm-up generates: past the audio decoder's first chunk.
const WARM_UP_PATCHES: usize = 9;

/// The longest the babble cap lets a request's speech run, in ms: this, and
/// [`BABBLE_MS_PER_CHAR`] for each character of its text. VoxCPM2 now and then talks on past its
/// text to the last patch it may have; this is about nine times as long as it says a line of
/// that length at the median, where in a gains run all but 2 of the 7191 lines it ended itself
/// came within seven times.
const BABBLE_MS: usize = 3000;
const BABBLE_MS_PER_CHAR: usize = 500;

/// The patches after which a streamed request's frames waiting are decoded at once, rather than
/// once a chunk's worth waits: its first audio after its first patch, then as much again each
/// time, so that what it has given stays ahead of what a player has played of it while the
/// chunks grow to their size.
const RAMP: [usize; 4] = [1, 2, 4, 8];

/// The positions each slot of the generator's caches holds: a prompt of the longest voice, 188
/// patches twice, and the longest message the game sends, 1024 characters, with the patches
/// after it.
const CAPACITY: usize = 2048;

/// The most positions of a prompt one submission runs: a median voice's prompt at once.
const MAX_ROWS: usize = 512;

/// The requests the generator generates at once: the most rows the LM side's batches were timed
/// at.
const SLOTS: usize = 16;

/// A request as the thread that generates takes it.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Prompt,
    /// The voice's latent frames, one after another, which the speech goes on from.
    pub voice: Vec<f32>,
    pub seed: u64,
    /// The most patches it may have.
    pub max_patches: usize,
    /// Whether its audio goes out as it is decoded, which it is then sooner.
    pub streamed: bool,
}

/// What a step gave a request: the audio decoded, and whether its patch was its last.
#[derive(Clone, Debug, PartialEq)]
pub struct Stepped {
    pub audio: Vec<f32>,
    pub last: bool,
}

/// What generates on the thread that owns the GPU: requests in slots, of which it holds
/// [`slots`](Self::slots), each a patch at a time, every request under way stepped at once.
pub trait Engine {
    /// The requests it generates at once.
    fn slots(&self) -> usize;

    /// Starts generating `request` in `slot`, which holds none. Gives the positions of its
    /// prompt that run through the LMs, the rest coming from what it keeps of the voice.
    fn start(&mut self, slot: usize, request: &Request) -> Result<usize, synth::Error>;

    /// Samples the next patch of each request under way, and decodes the audio ready, giving
    /// each request's slot and what it gave.
    fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)>;

    /// Ends the request in `slot`, giving the audio not yet given.
    fn finish(&mut self, slot: usize) -> Result<Vec<f32>, synth::Error>;

    /// Gives back what it holds of the GPU, saying what went wrong there.
    fn close(self: Box<Self>) -> Result<(), synth::Error>;
}

/// A request sent to the thread that generates, and where its speech goes.
struct Job {
    request: Request,
    /// When it was sent.
    sent: Instant,
    replies: Sender<Reply>,
}

/// What the thread that generates sends back for a request: for a streamed one, its audio a run
/// at a time as it is decoded, then for every one, its speech whole, or why there is none.
enum Reply {
    Audio(Vec<f32>),
    Speech(Result<Speech, synth::Error>),
}

/// A request under way.
struct Generation {
    job: Job,
    prefilled: usize,
    start: Option<Start>,
    patches: usize,
    samples: Vec<f32>,
}

impl Generation {
    /// Sends the speech, or why there is none.
    fn end(self, rest: Result<Vec<f32>, synth::Error>) {
        let Generation {
            job,
            start,
            patches,
            mut samples,
            ..
        } = self;
        let speech = rest.map(|rest| {
            if job.request.streamed && !rest.is_empty() {
                let _ = job.replies.send(Reply::Audio(rest.clone()));
            }
            samples.extend(rest);
            debug!(
                prompt = start.map(|s| s.prompt),
                prefilled = start.map(|s| s.prefilled),
                first_patch_ms = start.map(|s| ms(s.waited)),
                patches,
                ms = ms(job.sent.elapsed()),
                "generated"
            );
            Speech {
                samples,
                patches,
                start,
            }
        });
        let _ = job.replies.send(Reply::Speech(speech));
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Generates the requests `jobs` brings on `engine` until every sender of them is gone, starting
/// them in the order they came, as slots free, one between two steps: a prompt runs in
/// submissions of its own, and each started holds up the steps of those under way and the starts
/// of those behind it, so that sixteen at once would hold the first patch of each back by the
/// time of all their prompts rather than of its own.
fn serve(engine: &mut dyn Engine, jobs: &Receiver<Job>) {
    let mut slots: Vec<Option<Generation>> = (0..engine.slots()).map(|_| None).collect();
    let mut waiting = VecDeque::new();
    loop {
        if slots.iter().all(Option::is_none) && waiting.is_empty() {
            let Ok(job) = jobs.recv() else {
                return;
            };
            waiting.push_back(job);
        }
        waiting.extend(jobs.try_iter());
        if let Some(slot) = slots.iter().position(Option::is_none)
            && let Some(job) = waiting.pop_front()
        {
            match engine.start(slot, &job.request) {
                Ok(prefilled) => {
                    slots[slot] = Some(Generation {
                        job,
                        prefilled,
                        start: None,
                        patches: 0,
                        samples: Vec::new(),
                    });
                }
                Err(e) => {
                    let _ = job.replies.send(Reply::Speech(Err(e)));
                }
            }
        }
        if slots.iter().all(Option::is_none) {
            continue;
        }
        for (slot, stepped) in engine.step() {
            let Some(generation) = slots[slot].as_mut() else {
                continue;
            };
            match stepped {
                Ok(Stepped { audio, last }) => {
                    generation.patches += 1;
                    generation.start.get_or_insert_with(|| Start {
                        waited: generation.job.sent.elapsed(),
                        prompt: generation.job.request.prompt.len(),
                        prefilled: generation.prefilled,
                    });
                    if generation.job.request.streamed && !audio.is_empty() {
                        let _ = generation.job.replies.send(Reply::Audio(audio.clone()));
                    }
                    generation.samples.extend(audio);
                    if last {
                        let generation = slots[slot].take().expect("under way");
                        generation.end(engine.finish(slot));
                    }
                }
                Err(e) => {
                    let generation = slots[slot].take().expect("under way");
                    // The slot is freed whatever the audio left came to.
                    let _ = engine.finish(slot);
                    generation.end(Err(e));
                }
            }
        }
    }
}

/// VoxCPM2 on an [`Engine`] on a thread of its own, speaking each request with the same
/// parameters. Shut down or dropped, it waits for the thread to end the requests under way and
/// close the engine, which a process that exits meanwhile would leave half torn down on the GPU.
#[derive(Debug)]
pub struct Rt {
    tokenizer: Tokenizer,
    config: Config,
    max_patches: usize,
    babble_cap: bool,
    jobs: Sender<Job>,
    thread: Option<JoinHandle<Result<(), synth::Error>>>,
}

impl Rt {
    /// Generates on the engine `engine` makes, on a thread of its own, the prompts tokenized by
    /// `tokenizer` and laid out as `config` says, each request up to `max_patches` patches long.
    ///
    /// # Errors
    ///
    /// If the engine cannot be made.
    pub fn spawn(
        tokenizer: Tokenizer,
        config: Config,
        max_patches: usize,
        engine: impl FnOnce() -> Result<Box<dyn Engine>, synth::Error> + Send + 'static,
    ) -> Result<Self, synth::Error> {
        let (jobs, received) = mpsc::channel();
        let (made, making) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("rt".into())
            .spawn(move || match engine() {
                Ok(mut engine) => {
                    let _ = made.send(Ok(()));
                    serve(&mut *engine, &received);
                    engine.close()
                }
                Err(e) => {
                    let _ = made.send(Err(e));
                    Ok(())
                }
            })?;
        making
            .recv()
            .map_err(|_| "the thread VoxCPM2 loads on panicked")??;
        Ok(Self {
            tokenizer,
            config,
            max_patches,
            babble_cap: false,
            jobs,
            thread: Some(thread),
        })
    }

    /// Whether each request's patches are capped by its text, at 3 s and 0.5 s for each
    /// character, as well as by the most any request may have.
    pub fn set_babble_cap(&mut self, capped: bool) {
        self.babble_cap = capped;
    }

    /// The most patches the speech of `text` may have.
    fn patches_for(&self, text: &str) -> usize {
        if !self.babble_cap {
            return self.max_patches;
        }
        let vae = self.config.audio_vae();
        let hop: usize = vae.decoder_rates.iter().product();
        let cap = babble_cap(text, self.config.patch_size * hop, vae.out_sample_rate);
        cap.min(self.max_patches)
    }

    /// Loads the VoxCPM2 checkpoint in `dir` onto the GPU the `vulkan` crate chooses, with the
    /// kernels compiled into `glsl_kernels` and `mlir_kernels`, to speak with `params`, its
    /// prompt cache keeping up to `prompt_cache_bytes`; refusing one whose audio is not at the
    /// service's rate before it loads the weights.
    ///
    /// # Errors
    ///
    /// If the checkpoint cannot be read or the GPU cannot run it.
    pub fn load(
        dir: &Path,
        glsl_kernels: &Path,
        mlir_kernels: &Path,
        params: &Params,
        prompt_cache_bytes: usize,
    ) -> Result<Self, synth::Error> {
        let read = |name: &str| {
            let path = dir.join(name);
            fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let config = Config::from_json(&read("config.json")?)?;
        let rate = config.audio_vae().out_sample_rate;
        if rate != RATE {
            let reason = format!("out_sample_rate {rate} where the service works at {RATE} Hz");
            return Err(reason.into());
        }
        let tokenizer = Tokenizer::from_json(&read("tokenizer.json")?)?;
        let settings = Settings {
            capacity: CAPACITY,
            max_rows: MAX_ROWS,
            timesteps: params.sampling.timesteps,
            chunk_frames: params.decode_chunk_frames.get(),
            timed: false,
            prompt_cache_bytes,
            subgroup: None,
            slots: SLOTS,
        };
        let sampling = params.sampling.clone();
        let patch = config.patch_size * config.feat_dim;
        let (dir, glsl_kernels, mlir_kernels) = (
            dir.to_owned(),
            glsl_kernels.to_owned(),
            mlir_kernels.to_owned(),
        );
        Self::spawn(tokenizer, config, params.max_patches, move || {
            let ctx = vulkan::Context::new()?;
            let report = ctx.report();
            info!(gpu = %report.name, driver = %report.driver_info, "opened the GPU");
            let mut kernels = Kernels::new(&ctx, glsl_kernels).with_mlir(mlir_kernels);
            let weights = Pth::open(dir.join("audiovae.pth"))?;
            let parameter = |name: &str| weights.floats(&format!("decoder.{name}"));
            let generator = Generator::new(&ctx, &mut kernels, &dir, &settings, parameter)?;
            Ok(Box::new(Generating {
                ctx,
                noise: (0..generator.slots()).map(|_| None).collect(),
                streamed: vec![None; generator.slots()],
                generator,
                sampling,
                patch,
            }))
        })
    }

    /// `text` in `voice` from `seed`'s noise, up to `max_patches` patches of it, its audio given
    /// to `audio` as it is decoded where there is one.
    fn generate(
        &self,
        text: &str,
        voice: &Latent,
        seed: u64,
        max_patches: usize,
        mut audio: Option<&mut Sink<'_>>,
    ) -> Result<Speech, synth::Error> {
        let tokens = self.tokenizer.encode(&format!("{}{text}", voice.text));
        let frames = voice.frames.as_flattened();
        let prompt = Prompt::cloning(&tokens, frames, &self.config)?;
        let (replies, received) = mpsc::channel();
        let job = Job {
            request: Request {
                prompt,
                voice: frames.to_vec(),
                seed,
                max_patches,
                streamed: audio.is_some(),
            },
            sent: Instant::now(),
            replies,
        };
        let stopped = "the thread VoxCPM2 generates on has stopped";
        self.jobs.send(job).map_err(|_| stopped)?;
        loop {
            match received.recv().map_err(|_| stopped)? {
                Reply::Audio(samples) => {
                    if let Some(audio) = audio.as_mut() {
                        audio(&samples);
                    }
                }
                Reply::Speech(speech) => return speech,
            }
        }
    }

    /// Speaks a line in the voice of `voices` with the most frames, a few patches of it, so that
    /// a prompt, its patches and their audio have each run once on the GPU before a request
    /// waits on them. Logs how long it took.
    ///
    /// # Errors
    ///
    /// If a voice cannot be read or spoken.
    pub fn warm_up(&self, voices: &Voices) -> Result<(), synth::Error> {
        let started = Instant::now();
        let mut longest: Option<Latent> = None;
        for name in voices.names() {
            let latent = voices.latent(name)?;
            if longest
                .as_ref()
                .is_none_or(|l| latent.frames.len() > l.frames.len())
            {
                longest = Some(latent);
            }
        }
        let voice = longest.ok_or("no voices")?;
        let speech = self.generate(WARM_UP_LINE, &voice, 0, WARM_UP_PATCHES, None)?;
        info!(
            prompt = speech.start.map(|s| s.prompt),
            patches = speech.patches,
            seconds = started.elapsed().as_secs_f64(),
            "warmed up"
        );
        Ok(())
    }
}

/// The generator, with the noise of the request in each of its slots.
struct Generating {
    ctx: vulkan::Context,
    generator: Generator,
    sampling: Sampling,
    /// The values of a patch, and so of its noise.
    patch: usize,
    /// The noise of each slot's request, none in a free slot.
    noise: Vec<Option<SeededNoise>>,
    /// The patches of each slot's streamed request so far, none for one not streamed.
    streamed: Vec<Option<usize>>,
}

impl Engine for Generating {
    fn slots(&self) -> usize {
        self.generator.slots()
    }

    fn start(&mut self, slot: usize, request: &Request) -> Result<usize, synth::Error> {
        let prompt = &request.prompt;
        let positions = graph::prompt::Prompt {
            tokens: &prompt.tokens,
            patches: &prompt.patches,
            audio: &prompt.audio,
        };
        let prefilled = self.generator.start(
            slot,
            positions,
            &request.voice,
            request.max_patches,
            self.sampling.temperature,
            self.sampling.cfg_value,
        )?;
        self.noise[slot] = Some(SeededNoise::new(request.seed));
        self.streamed[slot] = request.streamed.then_some(0);
        let cache = self.generator.prompt_cache();
        debug!(
            prompt = prompt.len(),
            prefilled,
            kept_voices = cache.voices(),
            kept_bytes = cache.bytes(),
            "started"
        );
        Ok(prefilled)
    }

    fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)> {
        let Generating {
            generator,
            noise,
            patch,
            streamed,
            ..
        } = self;
        let drawn = |slot: usize| {
            let noise = noise[slot].as_mut().expect("a request under way");
            noise.next(*patch)
        };
        match generator.step(drawn) {
            Ok(patches) => patches
                .into_iter()
                .map(|(slot, patch)| {
                    let stepped = patch.map_err(synth::Error::from).and_then(|patch| {
                        if let Some(patches) = &mut streamed[slot] {
                            *patches += 1;
                            if RAMP.contains(patches) {
                                generator.decode_waiting(slot)?;
                            }
                        }
                        Ok(Stepped {
                            audio: generator.take_audio(slot),
                            last: patch.last,
                        })
                    });
                    (slot, stepped)
                })
                .collect(),
            // The submission failed, and every request under way with it.
            Err(e) => {
                let held = noise
                    .iter()
                    .enumerate()
                    .filter(|(_, noise)| noise.is_some());
                held.map(|(slot, _)| (slot, Err(e.to_string().into())))
                    .collect()
            }
        }
    }

    /// Fails the request once the validation layer, where it is on, has reported anything.
    fn finish(&mut self, slot: usize) -> Result<Vec<f32>, synth::Error> {
        self.noise[slot] = None;
        self.streamed[slot] = None;
        let audio = self.generator.finish(slot)?;
        validated(&self.ctx)?;
        Ok(audio)
    }

    fn close(self: Box<Self>) -> Result<(), synth::Error> {
        let Generating { ctx, generator, .. } = *self;
        drop(generator);
        validated(&ctx)
    }
}

/// Whatever the validation layer, where it is on, has reported on `ctx`, as an error.
fn validated(ctx: &vulkan::Context) -> Result<(), synth::Error> {
    let messages = ctx.validation_messages();
    if messages.is_empty() {
        return Ok(());
    }
    let reported = messages.join("\n");
    Err(format!("the validation layer reported:\n{reported}").into())
}

impl Rt {
    /// Waits for the requests under way to end and the engine to close, and gives what went
    /// wrong as it did.
    ///
    /// # Errors
    ///
    /// If the engine did not close cleanly, or the thread panicked.
    pub fn shut_down(mut self) -> Result<(), synth::Error> {
        self.stop()
    }

    fn stop(&mut self) -> Result<(), synth::Error> {
        // The thread ends once the requests under way have, and no sender of jobs is left.
        drop(mem::replace(&mut self.jobs, mpsc::channel().0));
        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| "the thread VoxCPM2 generates on panicked")?,
            None => Ok(()),
        }
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            error!("shutting VoxCPM2 down: {e}");
        }
    }
}

impl Synthesizer for Rt {
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        self.generate(text, voice, seed, self.patches_for(text), None)
    }

    fn stream(
        &self,
        text: &str,
        voice: &Latent,
        seed: u64,
        audio: &mut Sink<'_>,
    ) -> Result<Speech, synth::Error> {
        self.generate(text, voice, seed, self.patches_for(text), Some(audio))
    }
}

/// The patches of `samples_per_patch` at `rate` Hz that the babble cap lets the speech of `text`
/// run to.
fn babble_cap(text: &str, samples_per_patch: usize, rate: u32) -> usize {
    let ms = BABBLE_MS + BABBLE_MS_PER_CHAR * text.chars().count();
    (ms * rate as usize / 1000).div_ceil(samples_per_patch)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use super::*;

    /// An engine of `slots` slots whose requests each have as many patches as their seed's last
    /// digit, each patch's audio the request's seed and the patch's number, and a sample more at
    /// the end; a request of seed 0 fails to start, one of seed 99 fails at its second patch.
    struct Fake {
        slots: usize,
        under_way: HashMap<usize, (u64, usize)>,
        /// Each request started, by its seed, and each step's slots under way, in order.
        events: Vec<Event>,
    }

    #[derive(Clone, Debug, PartialEq)]
    enum Event {
        Start(u64),
        Step(Vec<usize>),
    }

    impl Engine for Fake {
        fn slots(&self) -> usize {
            self.slots
        }

        fn start(&mut self, slot: usize, request: &Request) -> Result<usize, synth::Error> {
            assert!(!self.under_way.contains_key(&slot), "slot {slot} is held");
            if request.seed == 0 {
                return Err("no such request".into());
            }
            self.under_way.insert(slot, (request.seed, 0));
            self.events.push(Event::Start(request.seed));
            Ok(request.prompt.len() - 1)
        }

        fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)> {
            let mut slots: Vec<usize> = self.under_way.keys().copied().collect();
            slots.sort_unstable();
            self.events.push(Event::Step(slots.clone()));
            slots
                .into_iter()
                .map(|slot| {
                    let (seed, patches) = self.under_way.get_mut(&slot).expect("under way");
                    *patches += 1;
                    let stepped = match (*seed, *patches) {
                        (99, 2) => Err("not finite".into()),
                        (seed, n) => Ok(Stepped {
                            audio: vec![seed as f32, n as f32],
                            last: n as u64 == seed % 10,
                        }),
                    };
                    (slot, stepped)
                })
                .collect()
        }

        fn finish(&mut self, slot: usize) -> Result<Vec<f32>, synth::Error> {
            let (seed, _) = self.under_way.remove(&slot).expect("under way");
            Ok(vec![-(seed as f32)])
        }

        fn close(self: Box<Self>) -> Result<(), synth::Error> {
            Ok(())
        }
    }

    /// An engine that takes its time to close, then says it has, and how it went.
    struct Closing {
        closed: Arc<AtomicBool>,
        clean: bool,
    }

    impl Engine for Closing {
        fn slots(&self) -> usize {
            1
        }

        fn start(&mut self, _: usize, _: &Request) -> Result<usize, synth::Error> {
            Ok(0)
        }

        fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)> {
            let stepped = Stepped {
                audio: vec![],
                last: true,
            };
            vec![(0, Ok(stepped))]
        }

        fn finish(&mut self, _: usize) -> Result<Vec<f32>, synth::Error> {
            Ok(vec![])
        }

        fn close(self: Box<Self>) -> Result<(), synth::Error> {
            thread::sleep(Duration::from_millis(50));
            self.closed.store(true, Ordering::SeqCst);
            match self.clean {
                true => Ok(()),
                false => Err("left the GPU as it was".into()),
            }
        }
    }

    fn closing(clean: bool) -> (Rt, Arc<AtomicBool>) {
        let tokenizer = include_str!("../../voxcpm/tests/fixtures/voxcpm_tokenizer.json");
        let tokenizer = Tokenizer::from_json(tokenizer).unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        let engine = Closing {
            closed: closed.clone(),
            clean,
        };
        let rt = Rt::spawn(tokenizer, config(), 4, move || Ok(Box::new(engine))).unwrap();
        (rt, closed)
    }

    #[test]
    fn dropped_it_waits_for_its_engine_to_close() {
        let (rt, closed) = closing(true);
        let voice = Latent {
            frames: vec![[0.5; voices::LATENT_DIM]; 4],
            text: "Hello.".to_owned(),
        };
        assert_eq!(rt.synthesize("Hi.", &voice, 1).unwrap().patches, 1);
        drop(rt);
        assert!(
            closed.load(Ordering::SeqCst),
            "dropped before its engine closed"
        );
    }

    #[test]
    fn shut_down_it_says_how_its_engine_closed() {
        let (rt, closed) = closing(true);
        rt.shut_down().unwrap();
        assert!(closed.load(Ordering::SeqCst));
        let (rt, _) = closing(false);
        let error = rt.shut_down().unwrap_err().to_string();
        assert_eq!(error, "left the GPU as it was");
    }

    fn config() -> Config {
        Config::from_json(include_str!("../../voxcpm/tests/fixtures/voxcpm.json")).unwrap()
    }

    /// An engine of a slot that says how many patches each request may have, each a patch long.
    struct Asked(Arc<Mutex<Vec<usize>>>);

    impl Engine for Asked {
        fn slots(&self) -> usize {
            1
        }

        fn start(&mut self, _: usize, request: &Request) -> Result<usize, synth::Error> {
            self.0.lock().unwrap().push(request.max_patches);
            Ok(0)
        }

        fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)> {
            let stepped = Stepped {
                audio: vec![],
                last: true,
            };
            vec![(0, Ok(stepped))]
        }

        fn finish(&mut self, _: usize) -> Result<Vec<f32>, synth::Error> {
            Ok(vec![])
        }

        fn close(self: Box<Self>) -> Result<(), synth::Error> {
            Ok(())
        }
    }

    #[test]
    fn caps_a_requests_patches_by_its_text_where_asked() {
        // At the real model's patch of 160 ms: 5 s, 34 s and 3 s.
        assert_eq!(babble_cap("Yes.", 7680, 48000), 32);
        let line = "Hello there, could someone please come to medbay? I need help.";
        assert_eq!(babble_cap(line, 7680, 48000), 213);
        assert_eq!(babble_cap("", 7680, 48000), 19);

        // The tiny config's patches are 24 samples at 48 kHz, so "Yes." is capped at 10000, and
        // a longer text at the most any request may have.
        let tokenizer = include_str!("../../voxcpm/tests/fixtures/voxcpm_tokenizer.json");
        let tokenizer = Tokenizer::from_json(tokenizer).unwrap();
        let asked = Arc::new(Mutex::new(Vec::new()));
        let engine = Asked(asked.clone());
        let mut rt = Rt::spawn(tokenizer, config(), 20000, move || Ok(Box::new(engine))).unwrap();
        let voice = Latent {
            frames: vec![[0.5; voices::LATENT_DIM]; 4],
            text: "Hello.".to_owned(),
        };
        rt.synthesize("Yes.", &voice, 1).unwrap();
        rt.set_babble_cap(true);
        rt.synthesize("Yes.", &voice, 1).unwrap();
        rt.stream("Yes.", &voice, 1, &mut |_| {}).unwrap();
        rt.synthesize(line, &voice, 1).unwrap();
        rt.shut_down().unwrap();
        assert_eq!(*asked.lock().unwrap(), [20000, 10000, 10000, 20000]);
    }

    /// Sends a request of `seed` to `jobs`, its prompt of `text` tokens, streamed if `streamed`,
    /// and gives where its replies come back.
    fn send(jobs: &Sender<Job>, seed: u64, text: usize, streamed: bool) -> Receiver<Reply> {
        let config = config();
        let voice = vec![0.5; config.patch_size * config.feat_dim];
        let prompt = Prompt::cloning(&vec![7; text], &voice, &config).unwrap();
        let (replies, received) = mpsc::channel();
        let request = Request {
            prompt,
            voice,
            seed,
            max_patches: 16,
            streamed,
        };
        let sent = Instant::now();
        jobs.send(Job {
            request,
            sent,
            replies,
        })
        .unwrap();
        received
    }

    /// The runs of audio `received` brought, then the speech.
    fn replied(received: &Receiver<Reply>) -> (Vec<Vec<f32>>, Result<Speech, synth::Error>) {
        let mut runs = vec![];
        loop {
            match received.recv().unwrap() {
                Reply::Audio(run) => runs.push(run),
                Reply::Speech(speech) => return (runs, speech),
            }
        }
    }

    /// Serves what `send` sends on an engine of `slots` slots, and gives what the engine did.
    fn served_events(slots: usize, send: impl FnOnce(&Sender<Job>)) -> Vec<Event> {
        let (jobs, received) = mpsc::channel();
        send(&jobs);
        drop(jobs);
        let mut engine = Fake {
            slots,
            under_way: HashMap::new(),
            events: Vec::new(),
        };
        serve(&mut engine, &received);
        assert!(engine.under_way.is_empty(), "requests left under way");
        engine.events
    }

    /// Serves what `send` sends on an engine of `slots` slots, and gives the steps it took.
    fn served(slots: usize, send: impl FnOnce(&Sender<Job>)) -> Vec<Vec<usize>> {
        (served_events(slots, send).into_iter())
            .filter_map(|event| match event {
                Event::Step(slots) => Some(slots),
                Event::Start(_) => None,
            })
            .collect()
    }

    #[test]
    fn speaks_each_request_whole_one_at_a_time_in_the_order_they_came() {
        let mut replies = vec![];
        let steps = served(1, |jobs| {
            replies.push(send(jobs, 3, 2, false));
            replies.push(send(jobs, 12, 5, false));
        });
        let speech: Vec<Speech> = replies.iter().map(|r| replied(r).1.unwrap()).collect();
        assert_eq!(speech[0].samples, [3.0, 1.0, 3.0, 2.0, 3.0, 3.0, -3.0]);
        assert_eq!(speech[0].patches, 3);
        assert_eq!(speech[1].samples, [12.0, 1.0, 12.0, 2.0, -12.0]);
        // The cloning prompt of one patch of voice: its two patches and three tokens around the
        // text.
        let starts = speech
            .iter()
            .map(|s| s.start.map(|s| (s.prompt, s.prefilled)));
        assert_eq!(starts.collect::<Vec<_>>(), [Some((7, 6)), Some((10, 9))]);
        assert_eq!(steps, [[0]; 5]);
    }

    #[test]
    fn requests_take_the_slots_free_as_they_come_and_wait_for_one_when_none_is() {
        let mut replies = vec![];
        let steps = served(2, |jobs| {
            for seed in [2, 4, 1] {
                replies.push(send(jobs, seed, 1, false));
            }
        });
        let patches: Vec<usize> = replies
            .iter()
            .map(|r| replied(r).1.unwrap().patches)
            .collect();
        assert_eq!(patches, [2, 4, 1]);
        // The second starts a step after the first, the third in the first's slot once the first
        // has its last patch.
        assert_eq!(steps, [vec![0], vec![0, 1], vec![0, 1], vec![1], vec![1]]);
    }

    #[test]
    fn a_request_that_fails_frees_its_slot_for_the_next() {
        let mut replies = vec![];
        let steps = served(1, |jobs| {
            for seed in [0, 99, 1] {
                replies.push(send(jobs, seed, 1, false));
            }
        });
        let speech: Vec<_> = replies.iter().map(|r| replied(r).1).collect();
        assert_eq!(
            speech[0].as_ref().unwrap_err().to_string(),
            "no such request"
        );
        assert_eq!(speech[1].as_ref().unwrap_err().to_string(), "not finite");
        assert_eq!(speech[2].as_ref().unwrap().samples, [1.0, 1.0, -1.0]);
        assert_eq!(steps, [[0]; 3]);
    }

    #[test]
    fn a_streamed_request_has_its_audio_as_it_is_decoded_and_then_whole() {
        let mut replies = vec![];
        served(1, |jobs| {
            replies.push(send(jobs, 3, 2, true));
            replies.push(send(jobs, 2, 2, false));
            replies.push(send(jobs, 99, 2, true));
        });
        let (runs, speech) = replied(&replies[0]);
        assert_eq!(
            runs,
            [vec![3.0, 1.0], vec![3.0, 2.0], vec![3.0, 3.0], vec![-3.0]]
        );
        assert_eq!(runs.concat(), speech.unwrap().samples);
        let (runs, speech) = replied(&replies[1]);
        assert!(runs.is_empty());
        assert_eq!(speech.unwrap().samples, [2.0, 1.0, 2.0, 2.0, -2.0]);
        // A stream that fails has what was decoded before it failed, then the failure.
        let (runs, speech) = replied(&replies[2]);
        assert_eq!(runs, [vec![99.0, 1.0]]);
        assert_eq!(speech.unwrap_err().to_string(), "not finite");
    }

    /// Requests that come at once start one between two steps, in the order they came: the
    /// first has its first patch after its own prompt rather than after every one's.
    #[test]
    fn requests_that_come_at_once_start_a_step_apart() {
        let mut replies = vec![];
        let events = served_events(4, |jobs| {
            for seed in [3, 4, 5, 6] {
                replies.push(send(jobs, seed, 1, false));
            }
        });
        use Event::{Start, Step};
        assert_eq!(
            events[..8],
            [
                Start(3),
                Step(vec![0]),
                Start(4),
                Step(vec![0, 1]),
                Start(5),
                Step(vec![0, 1, 2]),
                // The first has had its last patch, and the fourth takes its slot.
                Start(6),
                Step(vec![0, 1, 2]),
            ]
        );
        for (reply, seed) in replies.iter().zip([3, 4, 5, 6]) {
            assert_eq!(replied(reply).1.unwrap().patches, seed);
        }
    }

    /// The four of the real model's DiT time constants that an f64 rounded once to f16 puts an
    /// ulp below where rounding through f32 does, as half rounds without std, as tgtts builds it:
    /// written as rounding through f32 writes them.
    #[test]
    fn writes_the_time_constants_rounded_through_f32() {
        for (value, bits) in [
            (0x4036_fa00_e1ad_d5dd, 0x4dbf),
            (0xc011_fa00_5296_9974, 0xc47f),
            (0xc004_ba00_794e_d8ca, 0xc12f),
            (0x402c_da00_d74d_552a, 0x4b37),
        ] {
            let value = f64::from_bits(value);
            assert_eq!(graph::local::dit::time_constant(value), bits, "{value:e}");
        }
    }
}
