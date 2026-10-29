//! The streamed routes, for a player that plays speech as it comes rather than once it is whole:
//! the game makes a job of a message, and each version of it the game asks for streams as Ogg
//! Opus to whoever has its capability.
//!
//! - `POST /stream`, with the game's token, takes the message as JSON: the fields of the audio
//!   routes' query and body, `pitch` as a number, and the versions it wants as `variants`, by
//!   their routes' names: `tts`, `tts-blips`, `tts-radio`, `tts-blips-radio` and
//!   `tts-radio-gibberish`. It answers at once with the job and each variant's stream, `/s/` and
//!   32 hex digits: a capability, which plays to anyone who has it and which nobody can guess,
//!   128 random bits.
//! - `GET /stream/{job}`, with the token, says how far each variant has come and, once it is
//!   done, how long its audio is; with `wait=SECONDS` it waits up to that long, at most 30, for
//!   the whole job to be done.
//! - `GET /s/{token}` plays a stream: its pages so far, then each as it is made, to every player
//!   alike, from the job's start until a minute after its end.
//!
//! Speech the model makes anew streams as it is made: brought to the service's loudness by its
//! voice's [level](crate::levels), where the whole-file routes measure the whole line instead,
//! and held under the ceiling by the same limiter, then through the pitch, the effects and the
//! clicks as there. Everything else is made whole first, as the whole-file routes make it, and
//! streams at once: speech from the cache, in a voice with no level, or through an ffmpeg filter
//! graph; the blips; and the radio, which the old radio service made over the whole speech. The
//! gibberish radio streams the plain radio, as its whole-file route serves it.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use dsp::{
    limiter::Limiter,
    pitch::Shifter,
    quantity::{self, Gain, Ratio},
    radio,
    resample::resample,
    segment::Segment,
};
use effects::{Pipeline, Processor};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tracing::info;

use super::{Failure, MAX_PITCH, Params, Route, Served, Service, Trace, Treatment, blocking, json};
use crate::{metrics::Leveling, opus, replay::Replay, synth, wav::Audio};

/// How long a job's streams can still be played once it is done.
const KEPT: Duration = Duration::from_secs(60);
/// The longest a status request waits for its job.
const LONGEST_WAIT: Duration = Duration::from_secs(30);
/// The bits a second the streams are encoded at, as many as the old frontend's Vorbis.
const BITRATE: u32 = 64000;

/// A message as `POST /stream` takes it.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Message {
    voice: String,
    identifier: String,
    text: String,
    pitch: i32,
    filter: String,
    special_filters: String,
    blip_base: String,
    blip_number: String,
    effects: Option<serde_json::Value>,
    variants: Vec<String>,
    /// Whether the radio of speech made anew streams as the speech does, scaled by its voice's
    /// peaks rather than its own, rather than waiting for the whole: the radio comes sooner but
    /// is not today's radio, as the peaks of a line's speech are not its voice's.
    streamed_radio: bool,
}

impl Message {
    /// The query an audio route would have had.
    fn params(&self) -> Params {
        Params {
            voice: self.voice.clone(),
            identifier: self.identifier.clone(),
            filter: self.filter.clone(),
            pitch: self.pitch.to_string(),
            special_filters: self.special_filters.clone(),
            blip_base: self.blip_base.clone(),
            blip_number: self.blip_number.clone(),
        }
    }

    /// The body an audio route would have had.
    fn body(&self) -> super::Body {
        super::Body {
            text: self.text.clone(),
            raw_text: String::new(),
            gibberish_text: String::new(),
            effects: self.effects.clone(),
        }
    }
}

/// The jobs under way or kept, and their streams by capability.
#[derive(Debug, Default)]
pub(super) struct Jobs {
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    streams: Mutex<HashMap<String, Arc<Replay>>>,
}

impl Jobs {
    fn job(&self, id: &str) -> Option<Arc<Job>> {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.get(id).cloned()
    }

    fn stream(&self, token: &str) -> Option<Arc<Replay>> {
        let streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        streams.get(token).cloned()
    }

    fn remove(&self, id: &str, tokens: &[String]) {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        jobs.remove(id);
        let mut streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        for token in tokens {
            streams.remove(token);
        }
    }
}

/// A job: how far each of its variants has come.
#[derive(Debug)]
pub(super) struct Job {
    variants: watch::Sender<BTreeMap<&'static str, Progress>>,
}

/// How far a variant has come.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
enum Progress {
    /// None of its audio has gone out yet.
    Pending,
    /// Its audio is going out.
    Streaming,
    /// All of it has, so many seconds of it.
    Done { seconds: f64 },
    /// It failed, as its whole-file route would have, with that route's status.
    Failed { status: u16, reason: String },
}

impl Progress {
    fn over(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Failed { .. })
    }
}

impl Job {
    /// Sets how far `route` has come, unless it is over.
    fn set(&self, route: Route, progress: Progress) {
        self.variants
            .send_if_modified(|variants| match variants.get_mut(route.name()) {
                Some(now) if !now.over() && *now != progress => {
                    *now = progress;
                    true
                }
                _ => false,
            });
    }
}

/// 128 random bits from the thread's generator, a CSPRNG, in hex.
fn capability() -> String {
    let bits: u128 = rand::rng().random();
    format!("{bits:032x}")
}

/// Where a variant's audio goes: its encoder and its stream, which ends when it does, however
/// it does.
struct Out {
    /// The routes it is the audio of: the gibberish radio streams the plain radio's.
    routes: Vec<Route>,
    encoder: Option<opus::Stream>,
    replay: Arc<Replay>,
    job: Arc<Job>,
    /// When the job was made, and how long after its first audio went out.
    made: Instant,
    first: Option<Duration>,
    samples: usize,
    energy: f64,
    failed: Option<String>,
}

/// What a variant came to.
struct Ended {
    seconds: f64,
    first: Option<Duration>,
    /// The RMS level of its audio, in dBFS.
    rms_dbfs: f64,
}

impl Out {
    /// Encodes `samples`, mono at [`opus::RATE`], and puts their pages on the stream.
    fn push(&mut self, samples: &[f32]) {
        if samples.is_empty() || self.failed.is_some() {
            return;
        }
        let encoder = self
            .encoder
            .as_mut()
            .expect("encoding until it is finished");
        match encoder.push(samples) {
            Ok(pages) => self.replay.push(pages),
            Err(e) => self.failed = Some(e.to_string()),
        }
        self.samples += samples.len();
        self.energy += samples.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
        if self.first.is_none() {
            self.first = Some(self.made.elapsed());
            for &route in &self.routes {
                self.job.set(route, Progress::Streaming);
            }
        }
    }

    /// Ends the stream with the last pages.
    fn finish(mut self) -> Result<Ended, Failure> {
        if let Some(reason) = self.failed.take() {
            return Err(Failure::internal(reason));
        }
        let encoder = self.encoder.take().expect("encoding until it is finished");
        let pages = encoder.finish().map_err(Failure::internal)?;
        self.replay.push(pages);
        self.replay.end();
        let seconds = self.samples as f64 / f64::from(opus::RATE);
        for &route in &self.routes {
            self.job.set(route, Progress::Done { seconds });
        }
        Ok(Ended {
            seconds,
            first: self.first,
            rms_dbfs: 10.0 * (self.energy / self.samples.max(1) as f64).log10(),
        })
    }
}

impl Drop for Out {
    fn drop(&mut self) {
        self.replay.end();
    }
}

/// `audio` as a stream takes it: mono at [`opus::RATE`], its channels averaged and resampled
/// where they are not, as an ffmpeg filter graph can leave them.
fn mono(audio: &Audio) -> Vec<f32> {
    let channels = usize::from(audio.channels.max(1));
    let samples: Vec<f32> = audio
        .samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    resample(&samples, f64::from(audio.rate), f64::from(opus::RATE))
}

/// What speech the model makes anew goes through as it comes, to its stream and to the radio:
/// the voice's gain, the pitch, the limiter, the effects, then the clicks.
pub(super) struct Chain {
    gain: Ratio,
    shifter: Option<Shifter>,
    limiter: Limiter,
    effects: Option<Pipeline>,
    /// The special filter's start and end clicks, the start going out with the first audio.
    clicks: Option<(Vec<f32>, Vec<f32>)>,
    started: bool,
    out: Option<Out>,
    /// All of it, for the radio made whole, or the radio on a stream of it.
    kept: Option<Vec<f32>>,
    radio: Option<(radio::Stream, Out)>,
    /// The energy of what the clicks go around, and its length.
    energy: f64,
    counted: usize,
}

/// What the end of a chain gave.
struct Finished {
    out: Option<Out>,
    kept: Option<Vec<f32>>,
    radio: Option<Out>,
    leveling: Leveling,
    rms_dbfs: f64,
}

impl Chain {
    /// `samples` in, as the model makes them.
    pub(super) fn push(&mut self, samples: &[f32]) {
        let gained: Vec<f32> = (samples.iter())
            .map(|&s| (f64::from(s) * self.gain.value()) as f32)
            .collect();
        let pitched = match &mut self.shifter {
            Some(shifter) => {
                let mut pitched = Vec::new();
                shifter.process(&gained, &mut pitched);
                pitched
            }
            None => gained,
        };
        let audio = self.limit(&pitched, false);
        self.emit(audio);
    }

    /// Ends the speech with the silence the old frontend put after it, before the pitch, and
    /// gives what is left of every step.
    fn finish(mut self) -> Finished {
        let tail = dsp::segment::Segment::silent(dsp::speech::TAIL, dsp::RATE).samples;
        let tail: Vec<f32> = tail.iter().map(|&s| f32::from(s)).collect();
        let pitched = match self.shifter.take() {
            Some(mut shifter) => {
                let mut pitched = Vec::new();
                shifter.process(&tail, &mut pitched);
                shifter.finish(&mut pitched);
                pitched
            }
            None => tail,
        };
        let audio = self.limit(&pitched, true);
        self.emit(audio);
        if let Some((_, end)) = self.clicks.take() {
            self.give(&end);
        }
        let radio = self.radio.take().map(|(stream, mut out)| {
            let mut radio = Vec::new();
            stream.finish(&mut radio);
            out.push(&Audio::from_segment(&Segment::new(dsp::RATE, radio)).samples);
            out
        });
        let energy = self.energy / self.counted.max(1) as f64;
        Finished {
            leveling: Leveling {
                gain: Gain::of(self.gain),
                reduction: self.limiter.reduction(),
                limited: self.limiter.fraction(),
            },
            rms_dbfs: 10.0 * energy.log10(),
            out: self.out.take(),
            kept: self.kept.take(),
            radio,
        }
    }

    /// `audio` through the limiter and the effects, and what they still hold if `last`.
    fn limit(&mut self, audio: &[f32], last: bool) -> Vec<f32> {
        let input: Vec<f64> = audio.iter().map(|&s| f64::from(s)).collect();
        let mut limited = Vec::new();
        self.limiter.process(&input, &mut limited);
        if last {
            self.limiter.finish(&mut limited);
        }
        let limited: Vec<f32> = limited.iter().map(|&s| s as f32).collect();
        match &mut self.effects {
            Some(effects) => {
                let mut out = Vec::new();
                effects.process(&limited, &mut out);
                if last {
                    effects.finish(&mut out);
                }
                out
            }
            None => limited,
        }
    }

    fn emit(&mut self, mut audio: Vec<f32>) {
        if audio.is_empty() {
            return;
        }
        self.energy += audio.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
        self.counted += audio.len();
        if !self.started {
            self.started = true;
            if let Some((start, _)) = &self.clicks {
                audio.splice(0..0, start.iter().copied());
            }
        }
        self.give(&audio);
    }

    /// `audio` to the speech's stream, and to the radio's.
    fn give(&mut self, audio: &[f32]) {
        if let Some(out) = &mut self.out {
            out.push(audio);
        }
        if let Some(kept) = &mut self.kept {
            kept.extend_from_slice(audio);
        }
        if let Some((stream, out)) = &mut self.radio {
            let speech = Audio {
                rate: dsp::RATE,
                channels: 1,
                samples: audio.to_vec(),
            };
            let mut radio = Vec::new();
            stream.process(&speech.to_pcm().samples, &mut radio);
            out.push(&Audio::from_segment(&Segment::new(dsp::RATE, radio)).samples);
        }
    }
}

/// How a speech or blips job came out: each variant's end, by its routes.
type Outcomes = Vec<(Vec<Route>, Result<Ended, Failure>, Option<f64>)>;

/// The outs of a job's variants.
#[derive(Default)]
struct Outs {
    speech: Option<Out>,
    radio: Option<Out>,
    blips: Option<Out>,
    blips_radio: Option<Out>,
}

pub(super) async fn create(State(service): State<Arc<Service>>, body: Bytes) -> Response {
    let bad = |reason: String| (StatusCode::BAD_REQUEST, reason).into_response();
    let message: Message = match serde_json::from_slice(&body) {
        Ok(message) => message,
        Err(e) => return bad(format!("body: {e}")),
    };
    let mut routes: Vec<Route> = Vec::new();
    for name in &message.variants {
        match Route::named(name) {
            Some(route) if !routes.contains(&route) => routes.push(route),
            Some(_) => {}
            None => return bad(format!("no variant {name:?}")),
        }
    }
    if routes.is_empty() {
        return bad("no variants".to_owned());
    }
    if !service.voices.names().any(|name| name == message.voice) {
        return bad(format!("no voice {:?}", message.voice));
    }
    if message.pitch.abs() > MAX_PITCH {
        return bad(format!("pitch {}", message.pitch));
    }
    if let Err(failure) = service.treatment(&message.params(), &message.body()) {
        return (failure.status, failure.reason).into_response();
    }

    let id = capability();
    let made = Instant::now();
    let progress = routes.iter().map(|r| (r.name(), Progress::Pending));
    let job = Arc::new(Job {
        variants: watch::Sender::new(progress.collect()),
    });
    let mut outs = Outs::default();
    let mut streams = serde_json::Map::new();
    let mut tokens = Vec::new();
    // The gibberish radio is the plain radio's stream under a capability of its own.
    let radio: Vec<Route> = (routes.iter().copied())
        .filter(|r| matches!(r, Route::Radio | Route::RadioGibberish))
        .collect();
    let mut groups: Vec<Vec<Route>> = (routes.iter().copied())
        .filter(|r| !radio.contains(r))
        .map(|r| vec![r])
        .collect();
    if !radio.is_empty() {
        groups.push(radio);
    }
    for group in groups {
        let serial = rand::rng().random();
        let (encoder, headers) = match opus::Stream::new(1, BITRATE, serial) {
            Ok(encoder) => encoder,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        let replay = Arc::new(Replay::default());
        replay.push(headers);
        for &route in &group {
            let token = capability();
            streams.insert(route.name().to_owned(), format!("/s/{token}").into());
            tokens.push(token.clone());
            let mut registered = service
                .jobs
                .streams
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            registered.insert(token, replay.clone());
        }
        let out = Out {
            routes: group.clone(),
            encoder: Some(encoder),
            replay,
            job: job.clone(),
            made,
            first: None,
            samples: 0,
            energy: 0.0,
            failed: None,
        };
        match group[0] {
            Route::Speech => outs.speech = Some(out),
            Route::Blips => outs.blips = Some(out),
            Route::BlipsRadio => outs.blips_radio = Some(out),
            Route::Radio | Route::RadioGibberish => outs.radio = Some(out),
        }
    }
    let jobs = &service.jobs;
    jobs.jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id.clone(), job.clone());
    let reply = serde_json::json!({ "job": id, "streams": streams });
    tokio::spawn(service.clone().run(id, tokens, job, message, outs));
    json(StatusCode::OK, reply)
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct Wait {
    wait: f64,
}

pub(super) async fn status(
    State(service): State<Arc<Service>>,
    Path(id): Path<String>,
    Query(wait): Query<Wait>,
) -> Response {
    let Some(job) = service.jobs.job(&id) else {
        return (StatusCode::NOT_FOUND, "no such job").into_response();
    };
    let wait = Duration::try_from_secs_f64(wait.wait.max(0.0))
        .unwrap_or(LONGEST_WAIT)
        .min(LONGEST_WAIT);
    let mut variants = job.variants.subscribe();
    let over = |v: &BTreeMap<&'static str, Progress>| v.values().all(Progress::over);
    let _ = tokio::time::timeout(wait, variants.wait_for(over)).await;
    let variants = variants.borrow().clone();
    let reply = serde_json::json!({ "done": over(&variants), "variants": variants });
    json(StatusCode::OK, reply)
}

pub(super) async fn play(
    State(service): State<Arc<Service>>,
    Path(token): Path<String>,
) -> Response {
    let Some(replay) = service.jobs.stream(&token) else {
        return (StatusCode::NOT_FOUND, "no such stream").into_response();
    };
    let headers = [
        (
            header::CONTENT_TYPE,
            HeaderValue::from_static("audio/ogg; codecs=opus"),
        ),
        (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        (
            header::ACCESS_CONTROL_ALLOW_ORIGIN,
            HeaderValue::from_static("*"),
        ),
    ];
    (headers, Body::from_stream(replay.reader())).into_response()
}

impl Service {
    /// Makes a job's variants, records each, and forgets the job once its streams have been
    /// kept long enough after.
    async fn run(
        self: Arc<Self>,
        id: String,
        tokens: Vec<String>,
        job: Arc<Job>,
        message: Message,
        outs: Outs,
    ) {
        let time = SystemTime::now();
        let started = Instant::now();
        let Outs {
            speech,
            radio,
            blips,
            blips_radio,
        } = outs;
        let mut speech_trace = Trace::default();
        let mut blips_trace = Trace::default();
        let spoken = async {
            if speech.is_none() && radio.is_none() {
                return Vec::new();
            }
            let routes = |out: &Option<Out>| out.as_ref().map(|o| o.routes.clone());
            let wanted: Vec<Vec<Route>> = [routes(&speech), routes(&radio)]
                .into_iter()
                .flatten()
                .collect();
            match self.speak(&message, speech, radio, &mut speech_trace).await {
                Ok(outcomes) => outcomes,
                Err(failure) => failed(wanted, &failure),
            }
        };
        let blipped = async {
            if blips.is_none() && blips_radio.is_none() {
                return Vec::new();
            }
            let routes = |out: &Option<Out>| out.as_ref().map(|o| o.routes.clone());
            let wanted: Vec<Vec<Route>> = [routes(&blips), routes(&blips_radio)]
                .into_iter()
                .flatten()
                .collect();
            match self
                .blip(&message, blips, blips_radio, &mut blips_trace)
                .await
            {
                Ok(outcomes) => outcomes,
                Err(failure) => failed(wanted, &failure),
            }
        };
        let (spoken, blipped) = tokio::join!(spoken, blipped);
        let params = message.params();
        let length = message.text.chars().count();
        for ((routes, outcome, rms), trace) in (spoken.into_iter().map(|o| (o, &speech_trace)))
            .chain(blipped.into_iter().map(|o| (o, &blips_trace)))
        {
            let mut trace = trace.clone();
            trace.timings.total = started.elapsed();
            trace.rms_dbfs = rms;
            for route in routes {
                let result = match &outcome {
                    Ok(ended) => {
                        info!(
                            route = route.name(),
                            identifier = %message.identifier,
                            first_audio_ms = ended.first.map(ms),
                            "streamed"
                        );
                        Ok(Served {
                            ogg: Vec::new(),
                            seconds: ended.seconds,
                        })
                    }
                    Err(failure) => {
                        job.set(
                            route,
                            Progress::Failed {
                                status: failure.status.as_u16(),
                                reason: failure.reason.clone(),
                            },
                        );
                        Err(Failure::new(failure.status, &failure.reason))
                    }
                };
                self.report(route, &params, time, length, &trace, &result);
            }
        }
        tokio::time::sleep(KEPT).await;
        self.jobs.remove(&id, &tokens);
    }

    /// The speech and its radio: streamed as the model makes it where it can be, else whole.
    async fn speak(
        self: &Arc<Self>,
        message: &Message,
        speech: Option<Out>,
        mut radio: Option<Out>,
        trace: &mut Trace,
    ) -> Result<Outcomes, Failure> {
        let params = message.params();
        let treatment = self.treatment(&params, &message.body())?;
        let voice = &message.voice;
        let text = synth::prepare(&message.text);
        let take = match &text {
            Some(text) => self.cached(voice, text, trace).await?,
            None => None,
        };
        let loudness = self.levels.loudness(voice);
        let mut outcomes = Outcomes::new();
        let audio = match (text, take, loudness, treatment) {
            (
                Some(text),
                None,
                Some(loudness),
                treatment @ (Treatment::Effects(_) | Treatment::None),
            ) => {
                let routes = speech.as_ref().map(|out| out.routes.clone());
                let mut chain = self.chain(&params, message.pitch, loudness, treatment);
                chain.out = speech;
                let peaks = self.levels.radio(voice).filter(|_| message.streamed_radio);
                match (radio.take(), peaks) {
                    (Some(out), Some(peaks)) => {
                        let stream = radio::Stream::new(&self.sounds.radio, peaks, dsp::RATE, 1);
                        chain.radio = Some((stream, out));
                    }
                    (whole, _) => {
                        chain.kept = whole.is_some().then(Vec::new);
                        radio = whole;
                    }
                }
                let (_, chain) = self.synthesized(voice, text, Some(chain), trace).await?;
                let chain = chain.expect("a chain given comes back");
                let finished = blocking(move || chain.finish()).await?;
                trace.leveling = Some(finished.leveling);
                if let (Some(routes), Some(out)) = (routes, finished.out) {
                    outcomes.push((routes, out.finish(), Some(finished.rms_dbfs)));
                }
                if let Some(out) = finished.radio {
                    let routes = out.routes.clone();
                    let ended = out.finish();
                    let rms = ended.as_ref().ok().map(|ended| ended.rms_dbfs);
                    outcomes.push((routes, ended, rms));
                }
                Audio {
                    rate: dsp::RATE,
                    channels: 1,
                    samples: finished.kept.unwrap_or_default(),
                }
            }
            (text, take, _, treatment) => {
                let spoken = match (text, take) {
                    // The old model service's empty WAV.
                    (None, _) => Segment::new(dsp::RATE, Vec::new()),
                    (Some(_), Some(take)) => take,
                    (Some(text), None) => self.synthesized(voice, text, None, trace).await?.0,
                };
                let finished = self.finish(voice, spoken, message.pitch, trace).await?;
                let audio = self.treat(&params, finished, treatment, trace).await?;
                if let Some(mut out) = speech {
                    let routes = out.routes.clone();
                    out.push(&mono(&audio));
                    outcomes.push((routes, out.finish(), trace.rms_dbfs));
                }
                audio
            }
        };
        if let Some(out) = radio {
            outcomes.push(self.radio_out(out, audio, trace).await);
        }
        Ok(outcomes)
    }

    /// The blips and their radio, whole.
    async fn blip(
        self: &Arc<Self>,
        message: &Message,
        blips: Option<Out>,
        radio: Option<Out>,
        trace: &mut Trace,
    ) -> Result<Outcomes, Failure> {
        let params = message.params();
        let treatment = self.treatment(&params, &message.body())?;
        let made = self
            .blips(&params, &message.text, message.pitch, trace)
            .await?;
        let audio = self.treat(&params, made, treatment, trace).await?;
        let mut outcomes = Outcomes::new();
        if let Some(mut out) = blips {
            let routes = out.routes.clone();
            out.push(&mono(&audio));
            outcomes.push((routes, out.finish(), trace.rms_dbfs));
        }
        if let Some(out) = radio {
            outcomes.push(self.radio_out(out, audio, trace).await);
        }
        Ok(outcomes)
    }

    /// `audio` over the radio, whole, to `out`.
    async fn radio_out(
        self: &Arc<Self>,
        mut out: Out,
        audio: Audio,
        trace: &mut Trace,
    ) -> (Vec<Route>, Result<Ended, Failure>, Option<f64>) {
        let routes = out.routes.clone();
        let service = self.clone();
        let made = self
            .cpu(trace, move || {
                radio::radio(&audio.to_pcm(), &service.sounds.radio)
            })
            .await;
        match made {
            Ok(radio) => {
                let audio = Audio::from_segment(&radio);
                out.push(&audio.samples);
                (routes, out.finish(), Some(audio.rms_dbfs()))
            }
            Err(failure) => (routes, Err(failure), None),
        }
    }

    /// The chain speech in a voice of `loudness` streams through, as the request's `params`,
    /// `pitch` and `treatment` have it, to nowhere yet.
    fn chain(
        &self,
        params: &Params,
        pitch: i32,
        loudness: quantity::Loudness,
        treatment: Treatment,
    ) -> Chain {
        let clicked = params.special_filters.split('|').any(|s| s == "radio");
        let effects = match treatment {
            Treatment::Effects(pipeline) => Some(pipeline),
            Treatment::Filter(_) | Treatment::None => None,
        };
        Chain {
            gain: Ratio::of(Gain::between(loudness, self.loudness.target)),
            shifter: (pitch != 0).then(|| Shifter::new(dsp::RATE, f64::from(pitch), 24)),
            limiter: Limiter::new(dsp::RATE, self.loudness.ceiling),
            effects,
            clicks: clicked.then(|| self.sounds.clicks(dsp::RATE, 1, &mut rand::rng())),
            started: false,
            out: None,
            kept: None,
            radio: None,
            energy: 0.0,
            counted: 0,
        }
    }
}

/// Every variant of `routes` failed as `failure`.
fn failed(routes: Vec<Vec<Route>>, failure: &Failure) -> Outcomes {
    (routes.into_iter())
        .map(|routes| {
            let failure = Failure::new(failure.status, &failure.reason);
            (routes, Err(failure), None)
        })
        .collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    /// Two seconds of something like speech: a buzz at a wandering pitch, its harmonics falling
    /// off, in syllables of rising and falling loudness, one of them loud enough over the rest
    /// that bringing the whole to the target needs the limiter. Its harmonics' phases spread as
    /// Schroeder's, so that its peaks stand no further over the rest than a voice's do.
    fn voice() -> Vec<f32> {
        let mut phase = 0.0;
        (0..96000)
            .map(|i| {
                let t = f64::from(i) / 48000.0;
                phase += TAU * (120.0 + 30.0 * (TAU * 0.7 * t).sin()) / 48000.0;
                let syllable = (TAU * 2.5 * t).sin().max(0.0).powi(2);
                let loud = if (0.9..1.2).contains(&t) { 2.0 } else { 1.0 };
                let buzz: f64 = (1..20)
                    .map(|h| {
                        let h = f64::from(h);
                        (phase * h + TAU * h * h / 40.0).sin() / h
                    })
                    .sum();
                (0.02 * loud * syllable * buzz) as f32
            })
            .collect()
    }

    /// The chain on its own, at `gain`, shifting by `pitch`: what it keeps for the radio.
    fn chained(raw: &[f32], gain: f64, pitch: i32, chunk: usize) -> (Vec<f32>, Leveling) {
        let mut chain = Chain {
            gain: Ratio::new(gain),
            shifter: (pitch != 0).then(|| Shifter::new(dsp::RATE, f64::from(pitch), 24)),
            limiter: Limiter::new(dsp::RATE, dsp::loudness::CEILING),
            effects: None,
            clicks: None,
            started: false,
            out: None,
            kept: Some(Vec::new()),
            radio: None,
            energy: 0.0,
            counted: 0,
        };
        for run in raw.chunks(chunk) {
            chain.push(run);
        }
        let finished = chain.finish();
        (finished.kept.unwrap(), finished.leveling)
    }

    /// The whole-file path's speech of `raw`, shifted by `pitch`, and the gain from `raw` to it.
    fn whole(raw: &[f32], pitch: i32) -> (Vec<f32>, f64, f64) {
        let peak = raw.iter().fold(0f32, |m, s| m.max(s.abs()));
        let leveled = synth::level(&synth::to_segment(raw));
        let finished = dsp::speech::finish(&leveled, pitch);
        let (target, most) = (quantity::Loudness::new(-16.0), Gain::new(f64::INFINITY));
        let whole = dsp::loudness::normalize(&finished, target, dsp::loudness::CEILING, most);
        // From the model's samples, the leveling's gain and then the loudness's.
        let leveling = f64::from(leveled.max()) / 32768.0 / f64::from(peak);
        let gain = leveling * Ratio::of(whole.gain).value();
        let samples = (whole.segment.samples.iter())
            .map(|&s| f32::from(s) / 32768.0)
            .collect();
        (samples, gain, whole.reduction)
    }

    fn loudness(samples: &[f32]) -> f64 {
        let samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        dsp::loudness::integrated(&samples, dsp::RATE).value()
    }

    /// At the gain the whole-file path came to, the chain makes the whole-file path's speech:
    /// the limiter is the same on a stream, and the gain is linear. What is left is the rounding
    /// to 16 bits the whole-file path does between its steps and the chain does not: on this
    /// voice, which the limiter takes 9.3 dB off, 63.9 dB under full scale at most and 61.1 dB
    /// under the signal.
    #[test]
    fn streams_the_whole_file_paths_speech_at_its_gain() {
        let raw = voice();
        let (expected, gain, reduction) = whole(&raw, 0);
        assert!(reduction > 6.0, "the limiter took off {reduction} dB");
        for chunk in [7680, 1] {
            let (streamed, _) = chained(&raw, gain, 0, chunk);
            assert_eq!(streamed.len(), expected.len());
            let (mut most, mut noise, mut signal) = (0f64, 0f64, 0f64);
            for (&s, &e) in streamed.iter().zip(&expected) {
                let d = f64::from(s) - f64::from(e);
                most = most.max(d.abs());
                noise += d * d;
                signal += f64::from(e).powi(2);
            }
            let (most, under) = (20.0 * most.log10(), 10.0 * (noise / signal).log10());
            assert!(
                most < -60.0 && under < -58.0,
                "{most:.1} dBFS, {under:.1} dB"
            );
        }
    }

    /// Shifted, the speech is not the whole-file path's sample for sample: the vocoder's phase
    /// in each bin runs on from the angles of the bins before, which in quiet bins come down to
    /// the input's rounding, and the whole-file path shifts its speech rounded to 16 bits at
    /// another level. The phases come out apart, and the waveform with them, but not the
    /// spectrum: only the peaks, so where the limiter holds them down the loudness moves a
    /// little. On this voice it came within 0.29 LU of the whole-file path's, with the loud
    /// syllable from 1.5 to 4 times the rest and 4 to 14 dB limited.
    #[test]
    fn shifted_it_is_as_loud_as_the_whole_file_paths() {
        let raw = voice();
        for pitch in [5, -7] {
            let (expected, gain, _) = whole(&raw, pitch);
            let (streamed, _) = chained(&raw, gain, pitch, 1920);
            assert_eq!(streamed.len(), expected.len());
            let apart = loudness(&streamed) - loudness(&expected);
            assert!(apart.abs() < 0.5, "pitch {pitch}: {apart:+.3} LU");
        }
    }
}
