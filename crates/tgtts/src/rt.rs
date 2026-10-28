//! VoxCPM2 on tts-rs's own Vulkan runtime as the service's [`Synthesizer`]: the device `rt`,
//! built with the feature of the same name, on the `graph` crate's generator, which records the
//! model's work once and submits it a patch at a time, rather than on burn. A thread of its own
//! owns the GPU and generates there on an [`Engine`], the requests in the order they come and as
//! many at once as the engine holds, the generator one. The service talks to it as to the burn
//! devices: the prompt is built on the caller's thread, and the speech comes back whole. Each
//! patch is sampled from noise drawn from the request's seed as burn's are, so that a seed gives
//! the same noise on every device.
//!
//! The generator keeps what a prompt computes of its voice for the next prompts in it, within
//! `prompt_cache_bytes`, so that a request in a voice it keeps runs only its own text and the
//! voice spoken again; it counts the voice's patches' embeddings, kept on the host, beside the
//! keys and values on the GPU.

use std::{
    collections::VecDeque,
    mem,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
#[cfg(feature = "rt")]
use std::{fs, path::Path};

#[cfg(feature = "rt")]
use dsp::RATE;
#[cfg(feature = "rt")]
use graph::{
    generator::{Generator, Settings},
    kernels::Kernels,
};
use tracing::{debug, error, info};
use voices::{Latent, Voices};
use voxcpm::{Config, Prompt, Tokenizer};
#[cfg(feature = "rt")]
use voxcpm::{Noise, Params, Sampling, SeededNoise};

use crate::synth::{self, Speech, Start, Synthesizer};

/// The line the warm-up speaks, as long as a median line.
const WARM_UP_LINE: &str = "Hello there, could someone please come to medbay? I need help.";

/// The patches the warm-up generates: past the audio decoder's first chunk.
const WARM_UP_PATCHES: usize = 9;

/// The positions the generator's caches hold: a prompt of the longest voice, 188 patches twice,
/// and the longest message the game sends, 1024 characters, with the patches after it.
#[cfg(feature = "rt")]
const CAPACITY: usize = 2048;

/// The most positions of a prompt one submission runs: a median voice's prompt at once.
#[cfg(feature = "rt")]
const MAX_ROWS: usize = 512;

/// The requests the generator generates at once.
#[cfg(feature = "rt")]
const SLOTS: usize = 1;

/// A request as the thread that generates takes it.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Prompt,
    /// The voice's latent frames, one after another, which the speech goes on from.
    pub voice: Vec<f32>,
    pub seed: u64,
    /// The most patches it may have.
    pub max_patches: usize,
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
    speech: Sender<Result<Speech, synth::Error>>,
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
        let _ = job.speech.send(speech);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Generates the requests `jobs` brings on `engine` until every sender of them is gone: those
/// that come while its slots are full wait in the order they came, each starting as a slot
/// frees, between two steps of those under way.
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
        for (slot, held) in slots.iter_mut().enumerate() {
            if held.is_some() {
                continue;
            }
            let Some(job) = waiting.pop_front() else {
                break;
            };
            match engine.start(slot, &job.request) {
                Ok(prefilled) => {
                    *held = Some(Generation {
                        job,
                        prefilled,
                        start: None,
                        patches: 0,
                        samples: Vec::new(),
                    });
                }
                Err(e) => {
                    let _ = job.speech.send(Err(e));
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
            jobs,
            thread: Some(thread),
        })
    }

    /// Loads the VoxCPM2 checkpoint in `dir` onto the GPU the `vulkan` crate chooses, with the
    /// kernels compiled into `glsl_kernels` and `mlir_kernels`, to speak with `params`, its
    /// prompt cache keeping up to `prompt_cache_bytes`; refusing one whose audio is not at the
    /// service's rate before it loads the weights.
    ///
    /// # Errors
    ///
    /// If the checkpoint cannot be read or the GPU cannot run it.
    #[cfg(feature = "rt")]
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
            let weights = audiovae::Weights::open(dir.join("audiovae.pth"))?;
            let parameter = |name: &str| {
                weights
                    .floats(&format!("decoder.{name}"))
                    .map_err(|e| graph::Error::Checkpoint(e.to_string()))
            };
            let generator = Generator::new(&ctx, &mut kernels, &dir, &settings, parameter)?;
            Ok(Box::new(Generating {
                ctx,
                noise: (0..generator.slots()).map(|_| None).collect(),
                generator,
                sampling,
                patch,
            }))
        })
    }

    /// `text` in `voice` from `seed`'s noise, up to `max_patches` patches of it.
    fn generate(
        &self,
        text: &str,
        voice: &Latent,
        seed: u64,
        max_patches: usize,
    ) -> Result<Speech, synth::Error> {
        let tokens = self.tokenizer.encode(&format!("{}{text}", voice.text));
        let frames = voice.frames.as_flattened();
        let prompt = Prompt::cloning(&tokens, frames, &self.config)?;
        let (speech, received) = mpsc::channel();
        let job = Job {
            request: Request {
                prompt,
                voice: frames.to_vec(),
                seed,
                max_patches,
            },
            sent: Instant::now(),
            speech,
        };
        let stopped = "the thread VoxCPM2 generates on has stopped";
        self.jobs.send(job).map_err(|_| stopped)?;
        received.recv().map_err(|_| stopped)?
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
        let speech = self.generate(WARM_UP_LINE, &voice, 0, WARM_UP_PATCHES)?;
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
#[cfg(feature = "rt")]
struct Generating {
    ctx: vulkan::Context,
    generator: Generator,
    sampling: Sampling,
    /// The values of a patch, and so of its noise.
    patch: usize,
    /// The noise of each slot's request, none in a free slot.
    noise: Vec<Option<SeededNoise>>,
}

#[cfg(feature = "rt")]
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
                    let stepped = patch.map(|patch| Stepped {
                        audio: generator.take_audio(slot),
                        last: patch.last,
                    });
                    (slot, stepped.map_err(Into::into))
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
#[cfg(feature = "rt")]
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
        self.generate(text, voice, seed, self.max_patches)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Arc,
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
        /// Each step's slots under way, in order.
        steps: Vec<Vec<usize>>,
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
            Ok(request.prompt.len() - 1)
        }

        fn step(&mut self) -> Vec<(usize, Result<Stepped, synth::Error>)> {
            let mut slots: Vec<usize> = self.under_way.keys().copied().collect();
            slots.sort_unstable();
            self.steps.push(slots.clone());
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

    /// Sends a request of `seed` to `jobs`, its prompt of `text` tokens, and gives where its
    /// speech comes back.
    fn send(jobs: &Sender<Job>, seed: u64, text: usize) -> Receiver<Result<Speech, synth::Error>> {
        let config = config();
        let voice = vec![0.5; config.patch_size * config.feat_dim];
        let prompt = Prompt::cloning(&vec![7; text], &voice, &config).unwrap();
        let (speech, received) = mpsc::channel();
        let request = Request {
            prompt,
            voice,
            seed,
            max_patches: 16,
        };
        let sent = Instant::now();
        jobs.send(Job {
            request,
            sent,
            speech,
        })
        .unwrap();
        received
    }

    /// Serves what `send` sends on an engine of `slots` slots, and gives the steps it took.
    fn served(slots: usize, send: impl FnOnce(&Sender<Job>)) -> Vec<Vec<usize>> {
        let (jobs, received) = mpsc::channel();
        send(&jobs);
        drop(jobs);
        let mut engine = Fake {
            slots,
            under_way: HashMap::new(),
            steps: Vec::new(),
        };
        serve(&mut engine, &received);
        assert!(engine.under_way.is_empty(), "requests left under way");
        engine.steps
    }

    #[test]
    fn speaks_each_request_whole_one_at_a_time_in_the_order_they_came() {
        let mut replies = vec![];
        let steps = served(1, |jobs| {
            replies.push(send(jobs, 3, 2));
            replies.push(send(jobs, 12, 5));
        });
        let speech: Vec<Speech> = replies.iter().map(|r| r.recv().unwrap().unwrap()).collect();
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
                replies.push(send(jobs, seed, 1));
            }
        });
        let patches: Vec<usize> = replies
            .iter()
            .map(|r| r.recv().unwrap().unwrap().patches)
            .collect();
        assert_eq!(patches, [2, 4, 1]);
        // The third starts in the first's slot once the first has its last patch.
        assert_eq!(steps, [vec![0, 1], vec![0, 1], vec![0, 1], vec![1]]);
    }

    #[test]
    fn a_request_that_fails_frees_its_slot_for_the_next() {
        let mut replies = vec![];
        let steps = served(1, |jobs| {
            for seed in [0, 99, 1] {
                replies.push(send(jobs, seed, 1));
            }
        });
        let speech: Vec<_> = replies.iter().map(|r| r.recv().unwrap()).collect();
        assert_eq!(
            speech[0].as_ref().unwrap_err().to_string(),
            "no such request"
        );
        assert_eq!(speech[1].as_ref().unwrap_err().to_string(), "not finite");
        assert_eq!(speech[2].as_ref().unwrap().samples, [1.0, 1.0, -1.0]);
        assert_eq!(steps, [[0]; 3]);
    }
}
