//! The routes the game's TTS subsystem calls, and what each makes of a request.
//!
//! The game sends GETs with the text in a JSON body and the rest in the query, and plays the Ogg
//! Vorbis it gets back for as long as the `audio-length` header says, in seconds. For a message
//! that is not for one player alone it asks all five audio routes at once and drops the message
//! if any of them fails, so the gibberish radio, with no aligner to time the lost words by,
//! serves the plain radio rather than fail.

mod stream;

use std::{
    fmt, fs, io,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    Router,
    body::Bytes,
    extract::{Query, Request, State},
    http::{HeaderName, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dsp::{loudness::Normalized, radio, resample::resample, segment::Segment};
use effects::{
    Pipeline,
    convolve::{self, Impulses},
    schema::Graph,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, level_filters::LevelFilter, warn};
use voices::Voices;

use self::stream::{Chain, Jobs};
use crate::{
    cache::Cache,
    ffmpeg::{self, Ffmpeg},
    handoff::{Handoff, Missing, Source},
    levels::Levels,
    metrics::{Leveling, Metrics, Row, Timings},
    synth::{self, Start, Synthesizer},
    vorbis,
    wav::{self, Audio},
};

/// The old frontend's "silicon" effect, as effects: its ffmpeg graph upmixed the voice to stereo
/// at -3 dB, convolved each channel with its channel of [`SYNTH_IMPULSE`] scaled by 100 over the
/// impulse's magnitude, mixed that in at 1 to 8, then the same with RoomImpulse.wav at 1 to 10,
/// equalized and limited at -6 dBFS. The room's part was 61 dB under the voice and is left out,
/// its mix's 10/11 folded into the weights, which give the level of the mean of the channels.
const SILICON: &str = r#"[{"nodes": {
    "synth": {"effect": "convolve", "impulse": "synth"},
    "mix": {"effect": "mix", "weights": {"input": 0.5714, "synth": 0.1223}},
    "air": {"effect": "peak", "frequency_hz": 7710, "gain_db": -6, "q": 0.6},
    "rumble": {"effect": "peak", "frequency_hz": 33, "gain_db": -10, "q": 0.44},
    "limit": {"effect": "limiter", "ceiling_db": -6.021}},
  "edges": [["input", "synth"], ["input", "mix"], ["synth", "mix"], ["mix", "air"],
    ["air", "rumble"], ["rumble", "limit"], ["limit", "output"]]}]"#;

const SYNTH_IMPULSE: &str = "SynthImpulse.wav";
const CLICK_ON: &str = "mic_click_on.wav";
const CLICK_OFF: &str = "mic_click_off.wav";
const STATIC: &str = "diffstatic.wav";
/// The clicks the old frontend's "radio" special filter put before speech, one at random.
const STARTS: [&str; 2] = ["on1.wav", "on2.wav"];
/// The clicks it put after.
const ENDS: [&str; 4] = ["off1.wav", "off2.wav", "off3.wav", "off4.wav"];

/// The rate the effects run at.
const RATE: f64 = dsp::RATE as f64;

/// The most quarter tones speech is shifted by either way. The old frontend took any integer,
/// though a few octaves up would only have run it out of memory; the game asks for 24 at most.
const MAX_PITCH: i32 = 48;

/// The audio routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `/tts`: speech.
    Speech,
    /// `/tts-blips`: the blips.
    Blips,
    /// `/tts-radio`: the speech over the radio.
    Radio,
    /// `/tts-radio` with a `gibberish_text`: the speech over the radio with words lost in static.
    RadioGibberish,
    /// `/tts-blips-radio`: the blips over the radio.
    BlipsRadio,
}

impl Route {
    /// The name metrics and logs give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Speech => "tts",
            Self::Blips => "tts-blips",
            Self::Radio => "tts-radio",
            Self::RadioGibberish => "tts-radio-gibberish",
            Self::BlipsRadio => "tts-blips-radio",
        }
    }

    /// The route of `name`, as [`name`](Self::name) gives it.
    pub fn named(name: &str) -> Option<Self> {
        [
            Self::Speech,
            Self::Blips,
            Self::Radio,
            Self::RadioGibberish,
            Self::BlipsRadio,
        ]
        .into_iter()
        .find(|route| route.name() == name)
    }

    /// What it hands over to a radio route.
    fn makes(self) -> Option<Source> {
        match self {
            Self::Speech => Some(Source::Speech),
            Self::Blips => Some(Source::Blips),
            _ => None,
        }
    }
}

/// How much the service does at once, and how long it waits.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Syntheses run at once.
    pub synthesis: usize,
    /// Other work on the CPU, the dsp and ffmpeg, run at once.
    pub cpu: usize,
    /// How long a request waits for a turn at the model before it is refused with 503: by then
    /// its message would be too late to play.
    pub queue_timeout: Duration,
    /// How long a radio route waits for the route that makes the speech or blips it puts on the
    /// radio to begin before it answers 408, as the old radio services did after 10 s.
    pub radio_wait: Duration,
    /// How long a radio route waits in all once that route has begun.
    pub radio_wait_ceiling: Duration,
}

/// The loudness the service brings speech and blips to, before the game's filters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Loudness {
    /// The integrated loudness, in LUFS.
    pub target: f64,
    /// The true peak it is held under, in dBTP.
    pub ceiling: f64,
}

/// What a [`Service`] is made of.
pub struct Setup {
    /// What the game sends as its `Authorization` header.
    pub token: String,
    pub voices: Voices,
    /// The model, or none while it loads, until [`Service::ready`] gives it: until then the
    /// health check answers 503, and so does speech the cache does not hold.
    pub synthesizer: Option<Arc<dyn Synthesizer>>,
    pub cache: Cache,
    pub ffmpeg: Ffmpeg,
    pub sounds: Sounds,
    pub metrics: Option<Metrics>,
    pub logging: Logging,
    pub loudness: Loudness,
    /// How loud each voice comes out of the model, which streamed speech is leveled by: speech
    /// in no voice's level, where no voice has one, is not streamed but served whole.
    pub levels: Levels,
    pub limits: Limits,
    /// The seed of every request's blips, which are drawn from a seed of their own each time
    /// where there is none.
    pub blip_seed: Option<u64>,
}

/// The service: its voices, model, cache and limits, shared by every request.
pub struct Service {
    token: Vec<u8>,
    voices: Voices,
    synthesizer: OnceLock<Arc<dyn Synthesizer>>,
    cache: Cache,
    ffmpeg: Ffmpeg,
    sounds: Sounds,
    metrics: Option<Metrics>,
    logging: Logging,
    loudness: Loudness,
    levels: Levels,
    handoff: Handoff,
    jobs: Jobs,
    model: Arc<Semaphore>,
    cpu: Arc<Semaphore>,
    queue_timeout: Duration,
    blip_seed: Option<u64>,
}

impl Service {
    pub fn new(setup: Setup) -> Arc<Self> {
        Arc::new(Self {
            token: setup.token.into_bytes(),
            voices: setup.voices,
            synthesizer: setup.synthesizer.map_or_else(OnceLock::new, OnceLock::from),
            cache: setup.cache,
            ffmpeg: setup.ffmpeg,
            sounds: setup.sounds,
            metrics: setup.metrics,
            logging: setup.logging,
            loudness: setup.loudness,
            levels: setup.levels,
            handoff: Handoff::new(setup.limits.radio_wait, setup.limits.radio_wait_ceiling),
            jobs: Jobs::default(),
            model: Arc::new(Semaphore::new(setup.limits.synthesis)),
            cpu: Arc::new(Semaphore::new(setup.limits.cpu)),
            queue_timeout: setup.limits.queue_timeout,
            blip_seed: setup.blip_seed,
        })
    }

    /// Where requests' metrics go, if anywhere.
    pub fn metrics(&self) -> Option<&Metrics> {
        self.metrics.as_ref()
    }

    /// Speaks with `synthesizer` from now on, the model the service started without.
    ///
    /// # Panics
    ///
    /// If the service has a model already.
    pub fn ready(&self, synthesizer: Arc<dyn Synthesizer>) {
        if self.synthesizer.set(synthesizer).is_err() {
            panic!("the service has a model already");
        }
    }
}

/// The service's routes.
pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/tts", get(speech))
        .route("/tts-blips", get(blips))
        .route("/tts-radio", get(radio))
        .route("/tts-blips-radio", get(blips_radio))
        .route("/tts-voices", get(voices))
        .route("/pitch-available", get(pitch_available))
        .route("/toggle-logging", get(toggle_logging))
        .route("/stream", post(stream::create))
        .route("/stream/{job}", get(stream::status))
        .route_layer(middleware::from_fn_with_state(service.clone(), authorize))
        .route("/health-check", get(health_check))
        .route("/s/{token}", get(stream::play))
        .with_state(service)
}

async fn authorize(State(service): State<Arc<Service>>, request: Request, next: Next) -> Response {
    let given = request.headers().get(header::AUTHORIZATION);
    let given = given.map_or(&[][..], HeaderValue::as_bytes);
    if bool::from(given.ct_eq(&service.token)) {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "Unauthorized").into_response()
    }
}

/// The query of an audio route, each parameter empty where it is not given.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct Params {
    voice: String,
    identifier: String,
    filter: String,
    pitch: String,
    special_filters: String,
    blip_base: String,
    blip_number: String,
}

/// The JSON body of an audio route.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct Body {
    text: String,
    raw_text: String,
    gibberish_text: String,
    /// The graphs of [`effects::schema`], which the speech or blips go through in place of the
    /// `filter` query's ffmpeg graph when they are given, even as none.
    effects: Option<serde_json::Value>,
}

/// What a request's speech or blips go through before they are encoded.
enum Treatment {
    /// The effects it gives.
    Effects(Pipeline),
    /// The ffmpeg filter graph the game sent before it sent effects.
    Filter(String),
    None,
}

async fn speech(
    State(service): State<Arc<Service>>,
    Query(params): Query<Params>,
    body: Bytes,
) -> Response {
    service.serve(Route::Speech, params, body).await
}

async fn blips(
    State(service): State<Arc<Service>>,
    Query(params): Query<Params>,
    body: Bytes,
) -> Response {
    service.serve(Route::Blips, params, body).await
}

async fn radio(
    State(service): State<Arc<Service>>,
    Query(params): Query<Params>,
    body: Bytes,
) -> Response {
    service.serve(Route::Radio, params, body).await
}

async fn blips_radio(
    State(service): State<Arc<Service>>,
    Query(params): Query<Params>,
    body: Bytes,
) -> Response {
    service.serve(Route::BlipsRadio, params, body).await
}

async fn voices(State(service): State<Arc<Service>>) -> Response {
    let names: Vec<&str> = service.voices.names().collect();
    json(StatusCode::OK, serde_json::json!(names))
}

async fn pitch_available() -> &'static str {
    "Pitch available"
}

async fn health_check(State(service): State<Arc<Service>>) -> Response {
    match service.synthesizer.get() {
        Some(_) => "OK".into_response(),
        None => (StatusCode::SERVICE_UNAVAILABLE, "the model is loading").into_response(),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct LevelParams {
    level: String,
}

async fn toggle_logging(
    State(service): State<Arc<Service>>,
    Query(params): Query<LevelParams>,
) -> Response {
    let level = params.level.to_uppercase();
    let level = (!level.is_empty()).then_some(level);
    match service.logging.toggle(level.as_deref()) {
        Some(name) => json(StatusCode::OK, serde_json::json!({ "api": name })),
        None => {
            let message = format!("Invalid level: {}", level.unwrap_or_default());
            let body = serde_json::json!({ "status": "error", "message": message });
            json(StatusCode::BAD_REQUEST, body)
        }
    }
}

fn json(status: StatusCode, value: serde_json::Value) -> Response {
    let headers = [(header::CONTENT_TYPE, "application/json")];
    (status, headers, value.to_string()).into_response()
}

/// Why a request failed, as the response says and the log records.
#[derive(Debug)]
struct Failure {
    status: StatusCode,
    reason: String,
}

impl Failure {
    fn new(status: StatusCode, reason: impl fmt::Display) -> Self {
        Self {
            status,
            reason: reason.to_string(),
        }
    }

    fn internal(reason: impl fmt::Display) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, reason)
    }
}

/// What an audio route serves.
struct Served {
    ogg: Vec<u8>,
    seconds: f64,
}

/// What a request did, for its metrics and log line.
#[derive(Clone, Default)]
struct Trace {
    timings: Timings,
    cache_hit: Option<bool>,
    cache_takes: Option<usize>,
    patches: Option<usize>,
    start: Option<Start>,
    leveling: Option<Leveling>,
    /// The level of the audio served, as the metrics record it.
    rms_dbfs: Option<f64>,
}

impl Service {
    async fn serve(self: Arc<Self>, route: Route, params: Params, body: Bytes) -> Response {
        // Run to the end even if the game hangs up, so that the speech is cached, the radio
        // routes get what they wait for and the metrics are recorded.
        match tokio::spawn(self.respond(route, params, body)).await {
            Ok(response) => response,
            Err(e) => {
                error!(route = route.name(), "request failed: {e}");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }

    async fn respond(self: Arc<Self>, route: Route, params: Params, body: Bytes) -> Response {
        let time = SystemTime::now();
        let started = Instant::now();
        if let Some(source) = route.makes() {
            self.handoff.begin(source, &params.identifier);
        }
        let mut trace = Trace::default();
        let body: Result<Body, _> = if body.is_empty() {
            Ok(Body::default())
        } else {
            serde_json::from_slice(&body)
        };
        let (route, text_length, result) = match body {
            Err(e) => (
                route,
                0,
                Err(Failure::new(StatusCode::BAD_REQUEST, format!("body: {e}"))),
            ),
            Ok(body) => {
                let route = match route {
                    Route::Radio if !body.gibberish_text.is_empty() => Route::RadioGibberish,
                    route => route,
                };
                let text = match route {
                    Route::RadioGibberish => &body.raw_text,
                    _ => &body.text,
                };
                let length = text.chars().count();
                (
                    route,
                    length,
                    self.handle(route, &params, &body, &mut trace).await,
                )
            }
        };
        if let (Some(source), Err(failure)) = (route.makes(), &result) {
            let status = failure.status.as_u16();
            self.handoff
                .publish(source, &params.identifier, Err(status));
        }
        trace.timings.total = started.elapsed();
        self.report(route, &params, time, text_length, &trace, &result);
        match result {
            Ok(served) => ogg(served),
            Err(failure) => (failure.status, failure.reason).into_response(),
        }
    }

    fn report(
        &self,
        route: Route,
        params: &Params,
        time: SystemTime,
        text_length: usize,
        trace: &Trace,
        result: &Result<Served, Failure>,
    ) {
        let t = &trace.timings;
        let ms = |d: Duration| (d.as_secs_f64() * 1000.0).round() as u64;
        match result {
            Ok(served) => info!(
                route = route.name(),
                identifier = %params.identifier,
                voice = %params.voice,
                cache_hit = trace.cache_hit,
                cache_takes = trace.cache_takes,
                seconds = served.seconds,
                rms_dbfs = trace.rms_dbfs,
                queue_ms = ms(t.queue),
                handoff_ms = ms(t.handoff),
                synthesis_ms = ms(t.synthesis),
                dsp_ms = ms(t.dsp),
                ffmpeg_ms = ms(t.ffmpeg),
                total_ms = ms(t.total),
                gain_db = trace.leveling.map(|l| l.gain),
                limiter_db = trace.leveling.map(|l| l.reduction),
                limited = trace.leveling.map(|l| l.limited),
                "served"
            ),
            Err(failure) => warn!(
                route = route.name(),
                identifier = %params.identifier,
                voice = %params.voice,
                status = failure.status.as_u16(),
                total_ms = ms(t.total),
                "failed: {}",
                failure.reason
            ),
        }
        if let Some(metrics) = &self.metrics {
            metrics.record(Row {
                time,
                route: route.name(),
                voice: params.voice.clone(),
                identifier: params.identifier.clone(),
                status: match result {
                    Ok(_) => StatusCode::OK.as_u16(),
                    Err(failure) => failure.status.as_u16(),
                },
                cache_hit: trace.cache_hit,
                cache_takes: trace.cache_takes,
                text_length,
                audio_seconds: result.as_ref().ok().map(|served| served.seconds),
                rms_dbfs: result.as_ref().ok().and(trace.rms_dbfs),
                timings: trace.timings,
                patches: trace.patches,
                start: trace.start,
                leveling: trace.leveling,
            });
        }
    }

    async fn handle(
        self: &Arc<Self>,
        route: Route,
        params: &Params,
        body: &Body,
        trace: &mut Trace,
    ) -> Result<Served, Failure> {
        // The old model service spoke an unknown voice as no voice in particular, and its blips
        // service failed with 500. The game only asks for voices from /tts-voices.
        if !self.voices.names().any(|name| name == params.voice) {
            let reason = format!("no voice {:?}", params.voice);
            return Err(Failure::new(StatusCode::BAD_REQUEST, reason));
        }
        match route {
            Route::Speech => {
                let pitch = pitch(&params.pitch)?;
                let treatment = self.treatment(params, body)?;
                let speech = self.speech(&params.voice, &body.text, pitch, trace).await?;
                self.deliver(Source::Speech, params, speech, treatment, trace)
                    .await
            }
            Route::Blips => {
                let pitch = pitch(&params.pitch)?;
                let treatment = self.treatment(params, body)?;
                let blips = self.blips(params, &body.text, pitch, trace).await?;
                self.deliver(Source::Blips, params, blips, treatment, trace)
                    .await
            }
            Route::Radio => self.radio(Source::Speech, params, trace).await,
            Route::RadioGibberish => {
                info!(
                    identifier = %params.identifier,
                    "serving the plain radio for gibberish: no aligner to time the lost words"
                );
                self.radio(Source::Speech, params, trace).await
            }
            Route::BlipsRadio => self.radio(Source::Blips, params, trace).await,
        }
    }

    /// The speech of `text` in `voice`, finished as the old frontend finished it and brought to
    /// the service's loudness.
    async fn speech(
        self: &Arc<Self>,
        voice: &str,
        text: &str,
        pitch: i32,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let spoken = match synth::prepare(text) {
            // The old model service's empty WAV.
            None => Segment::new(dsp::RATE, Vec::new()),
            Some(text) => self.spoken(voice, text, trace).await?,
        };
        self.finish(voice, spoken, pitch, trace).await
    }

    /// What the model said in `voice`, finished as the old frontend finished it and brought to
    /// the service's loudness, by no more than the voice's level allows.
    async fn finish(
        &self,
        voice: &str,
        spoken: Segment,
        pitch: i32,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let loudness = self.loudness;
        let most = self.levels.most_gain(voice, loudness.target);
        let normalized = self
            .cpu(trace, move || {
                let finished = dsp::speech::finish(&spoken, pitch);
                dsp::loudness::normalize(&finished, loudness.target, loudness.ceiling, most)
            })
            .await?;
        Ok(leveled(normalized, trace))
    }

    /// What the model says for `text` in `voice`, from the cache once it has said it often
    /// enough, leveled.
    async fn spoken(
        self: &Arc<Self>,
        voice: &str,
        text: String,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        if let Some(speech) = self.cached(voice, &text, trace).await? {
            return Ok(speech);
        }
        let (speech, _) = self.synthesized(voice, text, None, trace).await?;
        Ok(speech)
    }

    /// A take of `text` in `voice` from the cache, once it holds enough of them.
    async fn cached(
        self: &Arc<Self>,
        voice: &str,
        text: &str,
        trace: &mut Trace,
    ) -> Result<Option<Segment>, Failure> {
        let service = self.clone();
        let (voice, text) = (voice.to_owned(), text.to_owned());
        let cached =
            blocking(move || service.cache.lookup(&voice, &text, &mut rand::rng())).await?;
        match cached {
            Ok(lookup) => {
                trace.cache_takes = Some(lookup.takes);
                if let Some(speech) = lookup.take {
                    trace.cache_hit = Some(true);
                    return Ok(Some(speech));
                }
            }
            Err(e) => warn!("reading the cache, speaking anew: {e}"),
        }
        trace.cache_hit = Some(false);
        Ok(None)
    }

    /// What the model says for `text` in `voice` anew, leveled and cached; streamed through
    /// `chain` as it is made where there is one, which comes back with it.
    async fn synthesized(
        self: &Arc<Self>,
        voice: &str,
        text: String,
        chain: Option<Chain>,
        trace: &mut Trace,
    ) -> Result<(Segment, Option<Chain>), Failure> {
        let Some(synthesizer) = self.synthesizer.get().cloned() else {
            return Err(Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the model is loading",
            ));
        };

        let queued = Instant::now();
        let turn = tokio::time::timeout(self.queue_timeout, self.model.clone().acquire_owned());
        let permit = turn
            .await
            .map_err(|_| {
                Failure::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "waited too long for the model",
                )
            })?
            .expect("the semaphore is never closed");
        trace.timings.queue += queued.elapsed();

        let seed = rand::rng().random();
        debug!(voice, seed, "synthesizing");
        let (service, voice) = (self.clone(), voice.to_owned());
        let started = Instant::now();
        let spoken = blocking(move || {
            let _permit = permit;
            let latent = service.voices.latent(&voice).map_err(Failure::internal)?;
            let (speech, chain) = match chain {
                Some(mut chain) => {
                    let speech = synthesizer.stream(&text, &latent, seed, &mut |run| {
                        chain.push(run);
                    });
                    (speech, Some(chain))
                }
                None => (synthesizer.synthesize(&text, &latent, seed), None),
            };
            let speech = speech.map_err(|e| Failure::internal(format!("synthesis: {e}")))?;
            let leveled = synth::level(&synth::to_segment(&speech.samples));
            if let Err(e) = service
                .cache
                .store(&voice, &text, &leveled, &mut rand::rng())
            {
                warn!("caching: {e}");
            }
            Ok((leveled, speech.patches, speech.start, chain))
        })
        .await?;
        trace.timings.synthesis += started.elapsed();
        let (speech, patches, start, chain) = spoken?;
        trace.patches = Some(patches);
        trace.start = start;
        Ok((speech, chain))
    }

    /// The blips of `text` in the request's voice, `pitch` quarter tones up, as the old frontend
    /// had them made: at the request's base and number, `male` and `1` where they are not given.
    /// They are brought to the service's loudness.
    async fn blips(
        self: &Arc<Self>,
        params: &Params,
        text: &str,
        pitch: i32,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let or =
            |value: &str, default: &str| if value.is_empty() { default } else { value }.to_owned();
        let base = or(&params.blip_base, "male");
        let number = or(&params.blip_number, "1");
        let (service, voice, text) = (self.clone(), params.voice.clone(), text.to_owned());
        let loudness = self.loudness;
        let seed = self.blip_seed.unwrap_or_else(|| rand::rng().random());
        self.cpu(trace, move || {
            let blips = service.voices.blips(&voice).map_err(Failure::internal)?;
            let mut draws = Draws(StdRng::seed_from_u64(seed));
            let blips = dsp::blips::message(&text, &blips, &base, &number, pitch, &mut draws)
                .map_err(|e| Failure::internal(format!("blips: {e}")))?;
            Ok(dsp::loudness::normalize(
                &blips,
                loudness.target,
                loudness.ceiling,
                f64::INFINITY,
            ))
        })
        .await?
        .map(|normalized| leveled(normalized, trace))
    }

    /// What the request's speech or blips go through: its effects if it gives them, checked
    /// before anything is made for it, or else as the old frontend did, the game's filter graph
    /// if there is one, stripped of double quotes, and the "silicon" effect only if there is
    /// not.
    fn treatment(&self, params: &Params, body: &Body) -> Result<Treatment, Failure> {
        if let Some(effects) = &body.effects {
            let refuse = |e: effects::Error| Failure::new(StatusCode::BAD_REQUEST, e);
            let graphs = effects::schema::parse(effects).map_err(refuse)?;
            if !graphs.is_empty() {
                let pipeline = Pipeline::new(&graphs, &self.sounds.impulses).map_err(refuse)?;
                return Ok(Treatment::Effects(pipeline));
            }
        } else {
            let graph = params.filter.replace('"', "");
            if !graph.is_empty() {
                return Ok(Treatment::Filter(graph));
            }
        }
        if params.special_filters.split('|').any(|s| s == "silicon") {
            let silicon = Pipeline::new(&self.sounds.silicon, &self.sounds.impulses);
            Ok(Treatment::Effects(
                silicon.expect("the silicon effect is checked when read"),
            ))
        } else {
            Ok(Treatment::None)
        }
    }

    /// `audio` through `treatment` and the request's special filters and encoded, handed over
    /// to the radio routes as it is before the encoding.
    async fn deliver(
        self: &Arc<Self>,
        source: Source,
        params: &Params,
        audio: Segment,
        treatment: Treatment,
        trace: &mut Trace,
    ) -> Result<Served, Failure> {
        let audio = Arc::new(self.treat(params, audio, treatment, trace).await?);
        self.handoff
            .publish(source, &params.identifier, Ok(audio.clone()));
        let ogg = self.encode(trace, audio.clone()).await?;
        Ok(Served {
            ogg,
            seconds: audio.seconds(),
        })
    }

    /// `audio` through `treatment` and the request's special filters: what the game plays, but
    /// for the encoding.
    async fn treat(
        self: &Arc<Self>,
        params: &Params,
        audio: Segment,
        treatment: Treatment,
        trace: &mut Trace,
    ) -> Result<Audio, Failure> {
        let specials: Vec<&str> = params.special_filters.split('|').collect();
        let wav = || wav::write_pcm16(audio.rate, 1, &audio.samples);
        let mut audio = match treatment {
            Treatment::Effects(mut pipeline) => {
                let input = Audio::from_segment(&audio);
                self.cpu(trace, move || {
                    let samples = resample(&input.samples, f64::from(input.rate), RATE);
                    Audio {
                        rate: dsp::RATE,
                        channels: 1,
                        samples: effects::run(&mut pipeline, &samples),
                    }
                })
                .await?
            }
            // A graph that fails is served unfiltered, since failing would drop the whole
            // message.
            Treatment::Filter(graph) => {
                let graph = graph.replace("%SAMPLE_RATE%", &audio.rate.to_string());
                let identifier = &params.identifier;
                if let Some(reason) = ffmpeg::refusal(&graph) {
                    warn!(%identifier, "serving unfiltered, the filter graph has {reason}: {graph}");
                    Audio::from_segment(&audio)
                } else {
                    let filtered = self.ffmpeg.filter(wav(), &graph, &[]);
                    match self.run_ffmpeg(trace, filtered).await {
                        Ok(filtered) => filtered,
                        Err(failure) => {
                            error!(%identifier, "serving unfiltered: {}", failure.reason);
                            Audio::from_segment(&audio)
                        }
                    }
                }
            }
            Treatment::None => Audio::from_segment(&audio),
        };
        trace.rms_dbfs = Some(audio.rms_dbfs());
        if specials.contains(&"radio") {
            let service = self.clone();
            audio = self
                .cpu(trace, move || {
                    service.sounds.clicked(&audio, &mut rand::rng())
                })
                .await?;
        }
        Ok(audio)
    }

    /// `audio` as the Ogg Vorbis the game plays, made at a turn on the CPU.
    async fn encode(&self, trace: &mut Trace, audio: Arc<Audio>) -> Result<Vec<u8>, Failure> {
        self.cpu(trace, move || vorbis::encode(&audio))
            .await?
            .map_err(Failure::internal)
    }

    /// What `source` handed over for the request's identifier, over the radio.
    async fn radio(
        self: &Arc<Self>,
        source: Source,
        params: &Params,
        trace: &mut Trace,
    ) -> Result<Served, Failure> {
        let waited = Instant::now();
        let handed = self.handoff.wait(source, &params.identifier).await;
        trace.timings.handoff += waited.elapsed();
        let audio = handed.map_err(|missing| match missing {
            Missing::TimedOut => {
                Failure::new(StatusCode::REQUEST_TIMEOUT, "nothing to put on the radio")
            }
            Missing::Failed(status) => Failure::new(
                StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                "what was to go on the radio failed",
            ),
        })?;
        let (service, input) = (self.clone(), audio);
        let radio = self
            .cpu(trace, move || {
                radio::radio(&input.to_pcm(), &service.sounds.radio)
            })
            .await?;
        let audio = Arc::new(Audio::from_segment(&radio));
        trace.rms_dbfs = Some(audio.rms_dbfs());
        let ogg = self.encode(trace, audio.clone()).await?;
        Ok(Served {
            ogg,
            seconds: audio.seconds(),
        })
    }

    /// Runs `work` off the async runtime at a turn on the CPU.
    async fn cpu<T: Send + 'static>(
        &self,
        trace: &mut Trace,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, Failure> {
        let queued = Instant::now();
        let permit = self.cpu.clone().acquire_owned().await;
        let permit = permit.expect("the semaphore is never closed");
        trace.timings.queue += queued.elapsed();
        let started = Instant::now();
        let result = blocking(move || {
            let _permit = permit;
            work()
        })
        .await;
        trace.timings.dsp += started.elapsed();
        result
    }

    /// Runs ffmpeg at a turn on the CPU.
    async fn run_ffmpeg<T>(
        &self,
        trace: &mut Trace,
        run: impl Future<Output = Result<T, ffmpeg::Error>>,
    ) -> Result<T, Failure> {
        let queued = Instant::now();
        let permit = self
            .cpu
            .acquire()
            .await
            .expect("the semaphore is never closed");
        trace.timings.queue += queued.elapsed();
        let started = Instant::now();
        let result = run.await;
        drop(permit);
        trace.timings.ffmpeg += started.elapsed();
        result.map_err(Failure::internal)
    }
}

/// The audio `normalized` holds, what it took recorded in `trace`.
fn leveled(normalized: Normalized, trace: &mut Trace) -> Segment {
    trace.leveling = Some(Leveling {
        gain: normalized.gain,
        reduction: normalized.reduction,
        limited: normalized.limited,
    });
    normalized.segment
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Failure> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Failure::internal(format!("worker: {e}")))
}

/// The request's pitch in quarter tones, 0 where it is not given.
fn pitch(pitch: &str) -> Result<i32, Failure> {
    if pitch.is_empty() {
        return Ok(0);
    }
    match pitch.parse::<i32>() {
        Ok(pitch) if pitch.abs() <= MAX_PITCH => Ok(pitch),
        _ => Err(Failure::new(
            StatusCode::BAD_REQUEST,
            format!("pitch {pitch:?}"),
        )),
    }
}

fn ogg(served: Served) -> Response {
    let length =
        HeaderValue::from_str(&served.seconds.to_string()).expect("a number is a header value");
    let headers = [
        (header::CONTENT_TYPE, HeaderValue::from_static("audio/ogg")),
        (
            header::CONTENT_DISPOSITION,
            HeaderValue::from_static("attachment; filename=identifier.ogg"),
        ),
        (HeaderName::from_static("audio-length"), length),
    ];
    (headers, served.ogg).into_response()
}

/// The draws the old services made from Python's `random`, from a generator of the service's.
struct Draws<R>(R);

impl<R: Rng> dsp::Random for Draws<R> {
    fn random(&mut self) -> f64 {
        self.0.random()
    }

    fn below(&mut self, n: usize) -> usize {
        self.0.random_range(0..n)
    }
}

/// The sounds the service puts around and under speech, from the assets directory.
pub struct Sounds {
    impulses: Impulses,
    silicon: Vec<Graph>,
    radio: radio::Assets,
    starts: Vec<Audio>,
    ends: Vec<Audio>,
}

impl Sounds {
    /// Reads the sounds in `dir`.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let read = |name: &str| -> io::Result<Audio> {
            let path = dir.join(name);
            let context = |e: &dyn fmt::Display| format!("{}: {e}", path.display());
            let bytes = fs::read(&path).map_err(|e| io::Error::new(e.kind(), context(&e)))?;
            wav::read(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, context(&e)))
        };
        let synth = read(SYNTH_IMPULSE)?;
        let impulses = Impulses {
            synth: Arc::new(convolve::Response::new(
                &synth.samples,
                usize::from(synth.channels),
                synth.rate,
            )),
        };
        let silicon = serde_json::from_str(SILICON).expect("the silicon effect is JSON");
        let silicon = effects::schema::parse(&silicon).expect("the silicon effect parses");
        if let Err(e) = Pipeline::new(&silicon, &impulses) {
            let reason = format!("the silicon effect with {SYNTH_IMPULSE}: {e}");
            return Err(io::Error::new(io::ErrorKind::InvalidData, reason));
        }
        Ok(Self {
            impulses,
            silicon,
            radio: radio::Assets::new(
                &read(CLICK_ON)?.to_pcm(),
                &read(CLICK_OFF)?.to_pcm(),
                &read(STATIC)?.to_pcm(),
            ),
            starts: STARTS
                .iter()
                .map(|name| read(name))
                .collect::<io::Result<_>>()?,
            ends: ENDS
                .iter()
                .map(|name| read(name))
                .collect::<io::Result<_>>()?,
        })
    }

    /// The radio's clicks and static.
    pub fn radio(&self) -> &radio::Assets {
        &self.radio
    }

    /// `audio` between a start and an end click drawn from `rng`, as the old frontend's "radio"
    /// special filter put them around the Ogg it had encoded, here around the audio before it is
    /// encoded, the clicks matched to its rate and channels.
    fn clicked(&self, audio: &Audio, rng: &mut impl Rng) -> Audio {
        let (start, end) = self.clicks(audio.rate, audio.channels, rng);
        let mut samples = start;
        samples.extend_from_slice(&audio.samples);
        samples.extend(end);
        Audio { samples, ..*audio }
    }

    /// A start and an end click drawn from `rng`, at `rate` and in `channels` channels.
    fn clicks(&self, rate: u32, channels: u16, rng: &mut impl Rng) -> (Vec<f32>, Vec<f32>) {
        let start = &self.starts[rng.random_range(0..self.starts.len())];
        let end = &self.ends[rng.random_range(0..self.ends.len())];
        (matched(start, rate, channels), matched(end, rate, channels))
    }
}

/// `click`'s samples at `rate` and in `channels` channels.
fn matched(click: &Audio, rate: u32, channels: u16) -> Vec<f32> {
    let from = usize::from(click.channels.max(1));
    let mono: Vec<f32> = click
        .samples
        .chunks(from)
        .map(|frame| frame.iter().sum::<f32>() / from as f32)
        .collect();
    let mono = resample(&mono, f64::from(click.rate), f64::from(rate));
    let channels = usize::from(channels.max(1));
    mono.iter()
        .flat_map(|&s| std::iter::repeat_n(s, channels))
        .collect()
}

/// The service's log level, which `/toggle-logging` changes.
pub struct Logging {
    level: Mutex<LevelFilter>,
    set: Box<dyn Fn(LevelFilter) + Send + Sync>,
}

impl Logging {
    /// Logging at `level`, changed through `set`.
    pub fn new(level: LevelFilter, set: impl Fn(LevelFilter) + Send + Sync + 'static) -> Self {
        Self {
            level: Mutex::new(level),
            set: Box::new(set),
        }
    }

    /// Sets the level named `level`, or with none toggles between INFO and DEBUG, as the old
    /// service did. Gives the name of the level it set, or `None` for a name it does not know.
    pub fn toggle(&self, level: Option<&str>) -> Option<&'static str> {
        let mut current = self.level.lock().unwrap_or_else(|e| e.into_inner());
        let level = match level {
            Some(name) => parse_level(name)?,
            None if *current == LevelFilter::INFO => LevelFilter::DEBUG,
            None => LevelFilter::INFO,
        };
        *current = level;
        (self.set)(level);
        Some(level_name(level))
    }
}

/// The level of a name as Python's logging has them, in capitals, or TRACE.
pub fn parse_level(name: &str) -> Option<LevelFilter> {
    Some(match name {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARNING" | "WARN" => LevelFilter::WARN,
        "ERROR" | "CRITICAL" | "FATAL" => LevelFilter::ERROR,
        _ => return None,
    })
}

fn level_name(level: LevelFilter) -> &'static str {
    match level {
        LevelFilter::TRACE => "TRACE",
        LevelFilter::DEBUG => "DEBUG",
        LevelFilter::INFO => "INFO",
        LevelFilter::WARN => "WARNING",
        LevelFilter::ERROR => "ERROR",
        _ => "OFF",
    }
}
