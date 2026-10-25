//! The routes the game's TTS subsystem calls, and what each makes of a request.
//!
//! The game sends GETs with the text in a JSON body and the rest in the query, and plays the Ogg
//! Vorbis it gets back for as long as the `audio-length` header says, in seconds. For a message
//! that is not for one player alone it asks all five audio routes at once and drops the message
//! if any of them fails, so the radio routes fall back to serving what they would otherwise
//! fail on.

use std::{
    collections::HashMap,
    fmt, fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    Router,
    body::Bytes,
    extract::{Query, Request, State},
    http::{HeaderName, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use dsp::{
    radio::{self, Pcm},
    resample::resample,
    segment::Segment,
};
use rand::{Rng, RngExt};
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, level_filters::LevelFilter, warn};
use voices::Voices;

use crate::{
    cache::Cache,
    ffmpeg::{self, Ffmpeg},
    handoff::{Handoff, Missing, Source},
    metrics::{Metrics, Row, Timings},
    synth::{self, Synthesizer},
    wav::{self, Audio},
};

/// The old frontend's "silicon" effect, whose further inputs are [`SYNTH_IMPULSE`] and
/// [`ROOM_IMPULSE`].
const SILICON: &str = "[0] aresample=44100 [re_1]; [re_1] apad=pad_dur=2 [in_1]; \
    [in_1] asplit=2 [in_1_1] [in_1_2]; [in_1_1] [1] afir=dry=10:wet=10 [reverb_1]; \
    [in_1_2] [reverb_1] amix=inputs=2:weights=8 1 [mix_1]; [mix_1] asplit=2 [mix_1_1] [mix_1_2]; \
    [mix_1_1] [2] afir=dry=1:wet=1 [reverb_2]; [mix_1_2] [reverb_2] amix=inputs=2:weights=10 1 [mix_2]; \
    [mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [out]; \
    [out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled";

const SYNTH_IMPULSE: &str = "SynthImpulse.wav";
const ROOM_IMPULSE: &str = "RoomImpulse.wav";
const CLICK_ON: &str = "mic_click_on.wav";
const CLICK_OFF: &str = "mic_click_off.wav";
const STATIC: &str = "diffstatic.wav";
/// The clicks the old frontend's "radio" special filter put before speech, one at random.
const STARTS: [&str; 2] = ["on1.wav", "on2.wav"];
/// The clicks it put after.
const ENDS: [&str; 4] = ["off1.wav", "off2.wav", "off3.wav", "off4.wav"];

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
    /// How long a radio route waits for the speech or blips it puts on the radio before it
    /// answers 408, as the old radio services did after 10 s.
    pub radio_wait: Duration,
}

/// What a [`Service`] is made of.
pub struct Setup {
    /// What the game sends as its `Authorization` header.
    pub token: String,
    pub voices: Voices,
    pub synthesizer: Arc<dyn Synthesizer>,
    pub cache: Cache,
    pub ffmpeg: Ffmpeg,
    pub sounds: Sounds,
    pub metrics: Option<Metrics>,
    pub logging: Logging,
    pub limits: Limits,
}

/// The service: its voices, model, cache and limits, shared by every request.
pub struct Service {
    token: Vec<u8>,
    voices: Voices,
    synthesizer: Arc<dyn Synthesizer>,
    cache: Cache,
    ffmpeg: Ffmpeg,
    sounds: Sounds,
    metrics: Option<Metrics>,
    logging: Logging,
    handoff: Handoff,
    model: Arc<Semaphore>,
    cpu: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl Service {
    pub fn new(setup: Setup) -> Arc<Self> {
        Arc::new(Self {
            token: setup.token.into_bytes(),
            voices: setup.voices,
            synthesizer: setup.synthesizer,
            cache: setup.cache,
            ffmpeg: setup.ffmpeg,
            sounds: setup.sounds,
            metrics: setup.metrics,
            logging: setup.logging,
            handoff: Handoff::new(setup.limits.radio_wait),
            model: Arc::new(Semaphore::new(setup.limits.synthesis)),
            cpu: Arc::new(Semaphore::new(setup.limits.cpu)),
            queue_timeout: setup.limits.queue_timeout,
        })
    }

    /// Where requests' metrics go, if anywhere.
    pub fn metrics(&self) -> Option<&Metrics> {
        self.metrics.as_ref()
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
        .route_layer(middleware::from_fn_with_state(service.clone(), authorize))
        .route("/health-check", get(health_check))
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

async fn health_check() -> &'static str {
    "OK"
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
#[derive(Default)]
struct Trace {
    timings: Timings,
    cache_hit: Option<bool>,
    patches: Option<usize>,
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
                seconds = served.seconds,
                queue_ms = ms(t.queue),
                handoff_ms = ms(t.handoff),
                synthesis_ms = ms(t.synthesis),
                dsp_ms = ms(t.dsp),
                ffmpeg_ms = ms(t.ffmpeg),
                total_ms = ms(t.total),
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
                text_length,
                audio_seconds: result.as_ref().ok().map(|served| served.seconds),
                timings: trace.timings,
                patches: trace.patches,
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
                let speech = self.speech(&params.voice, &body.text, pitch, trace).await?;
                self.deliver(Source::Speech, params, speech, trace).await
            }
            Route::Blips => {
                // The old frontend read the pitch here, and failed on one that was not a number,
                // but never passed it on to the blips.
                pitch(&params.pitch)?;
                let blips = self.blips(params, &body.text, trace).await?;
                self.deliver(Source::Blips, params, blips, trace).await
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

    /// The speech of `text` in `voice`, finished as the old frontend finished it.
    async fn speech(
        self: &Arc<Self>,
        voice: &str,
        text: &str,
        pitch: i32,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let spoken = match synth::prepare(text) {
            // The old model service's empty WAV.
            None => Segment::new(self.synthesizer.sample_rate(), Vec::new()),
            Some(text) => self.spoken(voice, text, trace).await?,
        };
        self.cpu(trace, move || dsp::speech::finish(&spoken, pitch))
            .await
    }

    /// What the model says for `text` in `voice`, from the cache once it has said it often
    /// enough, leveled.
    async fn spoken(
        self: &Arc<Self>,
        voice: &str,
        text: String,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let service = self.clone();
        let (voice, lookup) = (voice.to_owned(), text.clone());
        let cached = blocking(move || {
            let cached = service.cache.lookup(&voice, &lookup, &mut rand::rng());
            (voice, cached)
        })
        .await?;
        let (voice, cached) = cached;
        match cached {
            Ok(Some(speech)) => {
                trace.cache_hit = Some(true);
                return Ok(speech);
            }
            Ok(None) => {}
            Err(e) => warn!("reading the cache, speaking anew: {e}"),
        }
        trace.cache_hit = Some(false);

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
        let service = self.clone();
        let started = Instant::now();
        let spoken = blocking(move || {
            let _permit = permit;
            let latent = service.voices.latent(&voice).map_err(Failure::internal)?;
            let speech = service
                .synthesizer
                .synthesize(&text, &latent, seed)
                .map_err(|e| Failure::internal(format!("synthesis: {e}")))?;
            let rate = service.synthesizer.sample_rate();
            let leveled = synth::level(&synth::to_segment(&speech.samples, rate));
            if let Err(e) = service
                .cache
                .store(&voice, &text, &leveled, &mut rand::rng())
            {
                warn!("caching: {e}");
            }
            Ok((leveled, speech.patches))
        })
        .await?;
        trace.timings.synthesis += started.elapsed();
        let (speech, patches) = spoken?;
        trace.patches = Some(patches);
        Ok(speech)
    }

    /// The blips of `text` in the request's voice, as the old frontend had them made: at the
    /// request's base and number, `male` and `1` where they are not given.
    async fn blips(
        self: &Arc<Self>,
        params: &Params,
        text: &str,
        trace: &mut Trace,
    ) -> Result<Segment, Failure> {
        let or =
            |value: &str, default: &str| if value.is_empty() { default } else { value }.to_owned();
        let base = or(&params.blip_base, "male");
        let number = or(&params.blip_number, "1");
        let (service, voice, text) = (self.clone(), params.voice.clone(), text.to_owned());
        self.cpu(trace, move || {
            let blips = service.voices.blips(&voice).map_err(Failure::internal)?;
            let mut draws = Draws(rand::rng());
            dsp::blips::message(&text, &blips, &base, &number, 0, &mut draws)
                .map_err(|e| Failure::internal(format!("blips: {e}")))
        })
        .await?
    }

    /// `audio` through the request's filter and special filters and encoded, handed over to the
    /// radio routes as it is before the encoding.
    ///
    /// As the old frontend, it runs the game's filter graph if there is one and the "silicon"
    /// effect only if there is not, and strips the graph of double quotes.
    async fn deliver(
        self: &Arc<Self>,
        source: Source,
        params: &Params,
        audio: Segment,
        trace: &mut Trace,
    ) -> Result<Served, Failure> {
        let specials: Vec<&str> = params.special_filters.split('|').collect();
        let graph = params.filter.replace('"', "");
        let graph = graph.replace("%SAMPLE_RATE%", &audio.rate.to_string());
        let wav = || wav::write_pcm16(audio.rate, 1, &audio.samples);
        let mut audio = if !graph.is_empty() {
            self.run_ffmpeg(trace, self.ffmpeg.filter(wav(), &graph, &[]))
                .await?
        } else if specials.contains(&"silicon") {
            let impulses = [&*self.sounds.synth_impulse, &*self.sounds.room_impulse];
            self.run_ffmpeg(trace, self.ffmpeg.filter(wav(), SILICON, &impulses))
                .await?
        } else {
            Audio::from_segment(&audio)
        };
        if specials.contains(&"radio") {
            let service = self.clone();
            audio = self
                .cpu(trace, move || {
                    service.sounds.clicked(&audio, &mut rand::rng())
                })
                .await?;
        }
        let audio = Arc::new(audio);
        self.handoff
            .publish(source, &params.identifier, Ok(audio.clone()));
        let ogg = self.run_ffmpeg(trace, self.ffmpeg.encode(&audio)).await?;
        Ok(Served {
            ogg,
            seconds: audio.seconds(),
        })
    }

    /// What `source` handed over for the request's identifier, over the radio. Where the radio
    /// effect fails, the audio is served without it, since failing would drop the whole message.
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
        let (service, input) = (self.clone(), audio.clone());
        let radio = self
            .cpu(trace, move || {
                let pcm = input.to_pcm();
                radio::radio(&pcm, &service.sounds.radio(pcm.rate))
            })
            .await?;
        let audio = match radio {
            Ok(radio) => Arc::new(Audio::from_segment(&radio)),
            Err(e) => {
                error!(identifier = %params.identifier, "radio effect failed, serving the audio without it: {e}");
                audio
            }
        };
        let ogg = self.run_ffmpeg(trace, self.ffmpeg.encode(&audio)).await?;
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
    synth_impulse: PathBuf,
    room_impulse: PathBuf,
    click_on: Pcm,
    click_off: Pcm,
    noise: Pcm,
    radio: Mutex<HashMap<u32, Arc<radio::Assets>>>,
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
        let impulse = |name: &str| read(name).map(|_| dir.join(name));
        Ok(Self {
            synth_impulse: impulse(SYNTH_IMPULSE)?,
            room_impulse: impulse(ROOM_IMPULSE)?,
            click_on: read(CLICK_ON)?.to_pcm(),
            click_off: read(CLICK_OFF)?.to_pcm(),
            noise: read(STATIC)?.to_pcm(),
            radio: Mutex::default(),
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

    /// The radio's clicks and static, matched once per rate of the speech that asks for them.
    fn radio(&self, rate: u32) -> Arc<radio::Assets> {
        let mut assets = self.radio.lock().unwrap_or_else(|e| e.into_inner());
        let matched = || radio::Assets::new(&self.click_on, &self.click_off, &self.noise);
        assets
            .entry(rate)
            .or_insert_with(|| Arc::new(matched()))
            .clone()
    }

    /// `audio` between a start and an end click drawn from `rng`, as the old frontend's "radio"
    /// special filter put them around the Ogg it had encoded, here around the audio before it is
    /// encoded, the clicks matched to its rate and channels.
    fn clicked(&self, audio: &Audio, rng: &mut impl Rng) -> Audio {
        let start = &self.starts[rng.random_range(0..self.starts.len())];
        let end = &self.ends[rng.random_range(0..self.ends.len())];
        let mut samples = matched(start, audio);
        samples.extend_from_slice(&audio.samples);
        samples.extend(matched(end, audio));
        Audio { samples, ..*audio }
    }
}

/// `click`'s samples at `audio`'s rate and in as many channels.
fn matched(click: &Audio, audio: &Audio) -> Vec<f32> {
    let channels = usize::from(click.channels.max(1));
    let mono: Vec<f32> = click
        .samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    let mono = resample(&mono, f64::from(click.rate), f64::from(audio.rate));
    let channels = usize::from(audio.channels.max(1));
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
