//! The service as the game sees it: each route called as code/controllers/subsystem/tts.dm
//! calls it, with a fake model, real ffmpeg and real dsp.

mod common;

use std::{
    collections::BTreeMap,
    f32::consts::TAU,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use common::{Message, Reply, Server, VOICE, call};
use tgtts::{
    cache::Cache,
    ffmpeg::Ffmpeg,
    metrics::Metrics,
    service::{Limits, Logging, Setup, Sounds},
    synth::{self, Speech, Synthesizer},
    wav,
};
use tracing::level_filters::LevelFilter;
use voices::{BlipSet, Latent, Voices};

/// What the fake model says per character of text.
const FRAMES_PER_CHAR: usize = 1200;
/// The silence the old frontend put after speech.
const TAIL: usize = 12000;

/// A model that says a tone as long as the text, counting what it is asked and how many it
/// is asked at once, and failing on text with "Fail" in it.
#[derive(Default)]
struct Fake {
    texts: Mutex<Vec<String>>,
    gate: Option<Gate>,
    active: AtomicUsize,
    most: AtomicUsize,
}

/// Holds each synthesis until the test releases it, saying when one starts.
struct Gate {
    entered: Mutex<mpsc::Sender<String>>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl Fake {
    fn gated() -> (Self, mpsc::Receiver<String>, mpsc::Sender<()>) {
        let (entered, entries) = mpsc::channel();
        let (release, releases) = mpsc::channel();
        let gate = Gate {
            entered: Mutex::new(entered),
            release: Mutex::new(releases),
        };
        let fake = Self {
            gate: Some(gate),
            ..Self::default()
        };
        (fake, entries, release)
    }

    fn texts(&self) -> Vec<String> {
        self.texts.lock().unwrap().clone()
    }
}

impl Synthesizer for Fake {
    fn synthesize(&self, text: &str, _: &Latent, _: u64) -> Result<Speech, synth::Error> {
        self.texts.lock().unwrap().push(text.to_owned());
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(active, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.entered.lock().unwrap().send(text.to_owned()).unwrap();
            gate.release.lock().unwrap().recv().unwrap();
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        if text.contains("Fail") {
            return Err("the fake fails on request".into());
        }
        let samples = (0..FRAMES_PER_CHAR * text.len())
            .map(|i| 0.5 * (TAU * 220.0 * i as f32 / 48000.0).sin())
            .collect();
        Ok(Speech {
            samples,
            patches: text.len(),
        })
    }
}

fn tone(rate: u32, frames: usize, frequency: f32, amplitude: f32) -> Vec<i16> {
    (0..frames)
        .map(|i| {
            let s = amplitude * (TAU * frequency * i as f32 / rate as f32).sin();
            (s * 32767.0) as i16
        })
        .collect()
}

fn write_wav(path: &Path, rate: u32, channels: u16, samples: &[i16]) {
    std::fs::write(path, wav::write_pcm16(rate, channels, samples)).unwrap();
}

/// Four voices, two of whose names run together as another pair's do.
fn write_voices(dir: &Path) {
    std::fs::create_dir_all(dir.join("latents")).unwrap();
    std::fs::create_dir_all(dir.join("blips")).unwrap();
    let table: BTreeMap<String, String> = [
        (VOICE, "man"),
        ("Woman (Test)", "woman"),
        ("A", "a"),
        ("Ab", "ab"),
    ]
    .into_iter()
    .map(|(name, stem)| (name.to_owned(), stem.to_owned()))
    .collect();
    voices::write_table(dir, &table).unwrap();
    let mut keys: Vec<String> = ('a'..='z').chain('0'..='9').map(String::from).collect();
    keys.extend(["Deska".to_owned(), "Gwah".to_owned()]);
    let mut blips = BlipSet {
        sample_rate: 48000,
        ..BlipSet::default()
    };
    for base in ["male", "female"] {
        for (i, key) in keys.iter().enumerate() {
            let samples = tone(48000, 9600, 200.0 + 20.0 * i as f32, 0.5);
            blips
                .samples
                .insert((base.to_owned(), "1".to_owned(), key.clone()), samples);
        }
    }
    for stem in table.values() {
        let latent = Latent {
            frames: vec![[0.0; voices::LATENT_DIM]; 4],
            text: "Reference text.".to_owned(),
        };
        latent.write(&voices::latent_path(dir, stem)).unwrap();
        blips.write(&voices::blips_path(dir, stem)).unwrap();
    }
}

/// Clicks of lengths that resample to whole frames at 48 kHz, ten seconds of static and short
/// stereo impulses.
fn write_assets(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    write_wav(
        &dir.join("mic_click_on.wav"),
        48000,
        1,
        &tone(48000, 2400, 1000.0, 0.3),
    );
    write_wav(
        &dir.join("mic_click_off.wav"),
        24000,
        1,
        &tone(24000, 1200, 800.0, 0.3),
    );
    let mut state = 12345u32;
    let noise: Vec<i16> = (0..2 * 80000)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i16 / 8
        })
        .collect();
    write_wav(&dir.join("diffstatic.wav"), 8000, 2, &noise);
    for (i, name) in ["on1.wav", "on2.wav"].iter().enumerate() {
        write_wav(
            &dir.join(name),
            24000,
            1,
            &tone(24000, 600 * (i + 1), 1200.0, 0.3),
        );
    }
    for (i, name) in ["off1.wav", "off2.wav", "off3.wav", "off4.wav"]
        .iter()
        .enumerate()
    {
        write_wav(
            &dir.join(name),
            24000,
            1,
            &tone(24000, 120 * (i + 1), 900.0, 0.3),
        );
    }
    let impulse: Vec<i16> = (0..64)
        .flat_map(|i| [16384 >> (i / 4), 8192 >> (i / 4)])
        .collect();
    write_wav(&dir.join("SynthImpulse.wav"), 44100, 2, &impulse);
    write_wav(&dir.join("RoomImpulse.wav"), 44100, 2, &impulse);
}

fn limits() -> Limits {
    Limits {
        synthesis: 4,
        cpu: 4,
        queue_timeout: Duration::from_secs(60),
        radio_wait: Duration::from_secs(60),
    }
}

/// The service with the fake model, and what the tests look at.
struct Harness {
    server: Server,
    _dir: tempfile::TempDir,
    fake: Arc<Fake>,
    cache: PathBuf,
    metrics: PathBuf,
    levels: Arc<Mutex<Vec<LevelFilter>>>,
}

fn harness(fake: Fake, limits: Limits) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let (voices, assets) = (dir.path().join("voices"), dir.path().join("assets"));
    write_voices(&voices);
    write_assets(&assets);
    let (cache, metrics) = (dir.path().join("cache"), dir.path().join("metrics.sqlite"));
    let levels = Arc::new(Mutex::new(Vec::new()));
    let set = levels.clone();
    let fake = Arc::new(fake);
    let server = common::serve(Setup {
        token: String::new(),
        voices: Voices::open(&voices).unwrap(),
        synthesizer: fake.clone(),
        cache: Cache::open(&cache).unwrap(),
        ffmpeg: Ffmpeg::new("ffmpeg"),
        sounds: Sounds::open(&assets).unwrap(),
        metrics: Some(Metrics::open(&metrics, 1024).unwrap()),
        logging: Logging::new(LevelFilter::INFO, move |level| {
            set.lock().unwrap().push(level);
        }),
        limits,
    });
    Harness {
        server,
        _dir: dir,
        fake,
        cache,
        metrics,
        levels,
    }
}

impl Deref for Harness {
    type Target = Server;

    fn deref(&self) -> &Server {
        &self.server
    }
}

impl Harness {
    fn rows(&self) -> Vec<Row> {
        self.service.metrics().unwrap().flush();
        let connection = rusqlite::Connection::open(&self.metrics).unwrap();
        let mut select = connection
            .prepare(
                "SELECT route, voice, identifier, status, cache_hit, text_length, audio_seconds, \
                 patches FROM requests ORDER BY id",
            )
            .unwrap();
        select
            .query_map([], |r| {
                Ok(Row {
                    route: r.get(0)?,
                    voice: r.get(1)?,
                    identifier: r.get(2)?,
                    status: r.get(3)?,
                    cache_hit: r.get(4)?,
                    text_length: r.get(5)?,
                    audio_seconds: r.get(6)?,
                    patches: r.get(7)?,
                })
            })
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }
}

#[derive(Debug, PartialEq)]
struct Row {
    route: String,
    voice: String,
    identifier: String,
    status: u16,
    cache_hit: Option<bool>,
    text_length: i64,
    audio_seconds: Option<f64>,
    patches: Option<i64>,
}

/// The length of `frames` of speech over the radio, between the test's clicks.
fn radio_frames(frames: usize) -> usize {
    common::radio_frames(frames, 2400, 2400)
}

#[test]
fn serves_what_the_game_asks_for_at_startup() {
    let h = harness(Fake::default(), limits());
    let voices = h.get("/tts-voices", "");
    assert_eq!(voices.status, 200);
    assert_eq!(voices.text(), r#"["A","Ab","Man (Test)","Woman (Test)"]"#);
    let pitch = h.get("/pitch-available", "");
    assert_eq!((pitch.status, pitch.text()), (200, "Pitch available"));
    let health = call(&format!("{}/health-check", h.base), None, "");
    assert_eq!((health.status, health.text()), (200, "OK"));
}

#[test]
fn every_route_but_the_health_check_wants_the_token() {
    let h = harness(Fake::default(), limits());
    let message = Message::default();
    let mut paths: Vec<String> = ["tts", "tts-blips", "tts-radio", "tts-blips-radio"]
        .iter()
        .map(|route| message.query(route))
        .collect();
    paths.extend(["/tts-voices", "/pitch-available", "/toggle-logging"].map(String::from));
    let wrong = [
        None,
        Some(""),
        Some("s3cret"),
        Some("s3cret-token2"),
        Some("S3CRET-TOKEN"),
    ];
    for path in &paths {
        for token in wrong {
            let reply = call(&format!("{}{path}", h.base), token, &message.body());
            assert_eq!(reply.status, 401, "{path} with {token:?}");
        }
    }
    assert_eq!(h.fake.texts(), Vec::<String>::new());
    assert_eq!(*h.levels.lock().unwrap(), Vec::<LevelFilter>::new());
    assert_eq!(h.rows(), Vec::<Row>::new());
}

#[test]
fn speaks_as_the_game_asks() {
    let h = harness(Fake::default(), limits());
    let reply = Message::default().speech(&h);
    reply.assert_ogg(48000);
    assert_eq!(reply.length.as_deref(), Some("0.55"));
    assert_eq!(reply.frames(48000), 12 * FRAMES_PER_CHAR + TAIL);
    assert_eq!(h.fake.texts(), ["Hello there."]);

    let shifted = Message {
        pitch: "4",
        ..Message::default()
    }
    .speech(&h);
    shifted.assert_ogg(48000);
    assert_eq!(shifted.length.as_deref(), Some("0.55"));
    let unpitched = Message {
        pitch: "",
        ..Message::default()
    }
    .speech(&h);
    unpitched.assert_ogg(48000);
    for pitch in ["abc", "1.5", "49", "-49"] {
        let reply = Message {
            pitch,
            ..Message::default()
        }
        .speech(&h);
        assert_eq!(reply.status, 400, "pitch {pitch}");
    }
}

#[test]
fn speaks_any_letter_or_digit_and_makes_silence_of_the_rest() {
    let h = harness(Fake::default(), limits());
    let lower = Message {
        text: "hello there",
        ..Message::default()
    }
    .speech(&h);
    lower.assert_ogg(48000);
    assert_eq!(lower.frames(48000), 13 * FRAMES_PER_CHAR + TAIL);
    let silent = Message {
        text: "...!",
        ..Message::default()
    }
    .speech(&h);
    silent.assert_ogg(48000);
    assert_eq!(silent.length.as_deref(), Some("0.25"));
    assert_eq!(h.fake.texts(), ["Hello there. "]);
}

#[test]
fn runs_the_games_filters() {
    let h = harness(Fake::default(), limits());
    let gas_mask = Message {
        filter: "lowpass=f=750,volume=2",
        ..Message::default()
    }
    .speech(&h);
    gas_mask.assert_ogg(48000);
    assert_eq!(gas_mask.frames(48000), 12 * FRAMES_PER_CHAR + TAIL);
    let slowed = Message {
        filter: "tremolo=f=10:d=0.8,rubberband=tempo=0.5",
        ..Message::default()
    }
    .speech(&h);
    slowed.assert_ogg(48000);
    assert!(
        slowed.frames(48000) > 2 * 12 * FRAMES_PER_CHAR,
        "{slowed:?}"
    );
    let filters = [
        "acrusher=samples=9:level_out=7,aecho=delays=100:decays=0.4,aemphasis=type=emi,\
         crystalizer=i=6,acontrast=60,rubberband=pitch=0.9",
        "alimiter=0.9,acompressor=threshold=0.2:ratio=20:attack=10:release=50:makeup=2,\
         highpass=f=1000",
        "[0:a] asetrate=%SAMPLE_RATE%*0.7,aresample=16000,atempo=1/0.7,lowshelf=g=-20:f=500,\
         highpass=f=500,aphaser=in_gain=1:out_gain=1:delay=3.0:decay=0.4:speed=0.5:type=t [out]; \
         [out]atempo=1.2,volume=15dB [final]; anoisesrc=a=0.01:d=60 [noise]; \
         [final][noise] amix=duration=shortest",
        "rubberband=pitch='ifnot(0,2-(0+if(1,4))/16,1)':formant=preserved,\
         highpass=f=1000:t=s:w=24,equalizer=f=1200:g=15,equalizer=f=4350:g=-15,\
         highshelf=f=870:g=1,afftfilt=real='st(0,(b+0.5)/nb*sr);\
         st(1,3000+1500*sin(9.3*2*PI*pts));st(2,ld(0)/ld(1));\
         re*(1-ld(2)^2+2*gauss(log(ld(2)+1)))':imag='st(0,(b+0.5)/nb*sr);\
         st(1,3000+1500*sin(9.3*2*PI*pts));st(2,ld(0)/ld(1));\
         im*(1-ld(2)^2+2*gauss(log(ld(2)+1)))':win_size=1024",
    ];
    for filter in filters {
        let reply = Message {
            filter,
            ..Message::default()
        }
        .speech(&h);
        assert_eq!(reply.status, 200, "{filter}: {}", reply.text());
        assert!(reply.frames(48000) > 0);
    }
    let broken = Message {
        filter: "nosuchfilter",
        ..Message::default()
    }
    .speech(&h);
    assert_eq!(broken.status, 500);
}

#[test]
fn splits_the_special_filters() {
    let h = harness(Fake::default(), limits());
    let speech = 12 * FRAMES_PER_CHAR + TAIL;
    let special = |special_filters| {
        Message {
            special_filters,
            ..Message::default()
        }
        .speech(&h)
    };
    let silicon = special("silicon");
    silicon.assert_ogg(44100);
    assert!(silicon.frames(44100) >= speech * 44100 / 48000 + 2 * 44100);
    // The old frontend tested /tts's special filters as a string, for "silicon" in it.
    let not_silicon = special("notsilicon");
    not_silicon.assert_ogg(48000);
    assert_eq!(not_silicon.frames(48000), speech);
    special("radio|silicon").assert_ogg(44100);

    let starts = [1200, 2400];
    let ends = [240, 480, 720, 960];
    for _ in 0..4 {
        let radio = special("radio").frames(48000) - speech;
        assert!(
            starts.iter().any(|s| ends.iter().any(|e| s + e == radio)),
            "{radio} frames of clicks"
        );
    }
    let blips = Message {
        special_filters: "radio",
        ..Message::default()
    }
    .blips(&h);
    blips.assert_ogg(48000);
}

#[test]
fn makes_blips_as_the_game_asks() {
    let h = harness(Fake::default(), limits());
    let male = Message::default().blips(&h);
    male.assert_ogg(48000);
    let female = Message {
        blip_base: "female",
        ..Message::default()
    }
    .blips(&h);
    female.assert_ogg(48000);
    assert_eq!(female.length, male.length);
    let defaults = Message {
        blip_base: "",
        blip_number: "",
        ..Message::default()
    }
    .blips(&h);
    assert_eq!(defaults.length, male.length);
    let missing = Message {
        blip_number: "9",
        ..Message::default()
    }
    .blips(&h);
    assert_eq!(missing.status, 500);
    assert_eq!(h.fake.texts(), Vec::<String>::new());
}

/// The cycles in `samples`: their swings from over a tenth of their peak to under its negative.
fn cycles(samples: &[f32]) -> usize {
    let threshold = 0.1 * samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    let (mut high, mut count) = (false, 0);
    for &s in samples {
        if s > threshold && !high {
            (high, count) = (true, count + 1);
        } else if s < -threshold {
            high = false;
        }
    }
    count
}

#[test]
fn shifts_the_blips_by_the_pitch() {
    let h = harness(Fake::default(), limits());
    // Lowercase letters, which the blips detune by no more than 0.6 semitones either way.
    let blips = |pitch| {
        let reply = Message {
            pitch,
            text: "hello there",
            ..Message::default()
        }
        .blips(&h);
        reply.assert_ogg(48000);
        common::decode(&reply.body).samples
    };
    let (level, octave_up) = (blips("0"), blips("24"));
    let ratio = cycles(&octave_up) as f64 / cycles(&level) as f64;
    assert!((1.8..2.2).contains(&ratio), "{ratio} times the cycles");
}

#[test]
fn serves_a_message_for_everyone_whole() {
    let h = harness(Fake::default(), limits());
    let [speech, blips, radio, blips_radio, gibberish] = Message::default().all(&h);
    for reply in [&speech, &blips, &radio, &blips_radio, &gibberish] {
        reply.assert_ogg(48000);
    }
    assert_eq!(radio.frames(48000), radio_frames(speech.frames(48000)));
    assert_eq!(gibberish.length, radio.length);
    assert_eq!(blips_radio.frames(48000), radio_frames(blips.frames(48000)));
    assert_eq!(h.fake.texts(), ["Hello there."]);
}

#[test]
fn the_radio_waits_for_the_speech() {
    let (fake, entered, release) = Fake::gated();
    let h = harness(fake, limits());
    let message = Message::default();
    thread::scope(|s| {
        let speech = s.spawn(|| message.speech(&h));
        assert_eq!(entered.recv().unwrap(), "Hello there.");
        let radio = s.spawn(|| message.radio(&h));
        let gibberish = s.spawn(|| message.gibberish(&h));
        release.send(()).unwrap();
        let speech = speech.join().unwrap();
        speech.assert_ogg(48000);
        for reply in [radio.join().unwrap(), gibberish.join().unwrap()] {
            reply.assert_ogg(48000);
            assert_eq!(reply.frames(48000), radio_frames(speech.frames(48000)));
        }
    });
}

#[test]
fn the_radio_gives_up_after_its_wait() {
    let h = harness(
        Fake::default(),
        Limits {
            radio_wait: Duration::from_millis(50),
            ..limits()
        },
    );
    let message = Message::default();
    assert_eq!(message.radio(&h).status, 408);
    assert_eq!(message.blips_radio(&h).status, 408);
    assert_eq!(message.gibberish(&h).status, 408);
}

#[test]
fn the_radio_fails_with_what_it_waits_for() {
    let h = harness(Fake::default(), limits());
    let message = Message {
        text: "Fail now.",
        ..Message::default()
    };
    let [speech, blips, radio, blips_radio, gibberish] = message.all(&h);
    assert_eq!(speech.status, 500);
    assert_eq!((radio.status, gibberish.status), (500, 500));
    blips.assert_ogg(48000);
    blips_radio.assert_ogg(48000);
}

#[test]
fn refuses_an_unknown_voice_everywhere() {
    let h = harness(Fake::default(), limits());
    let message = Message {
        speaker: "Man (Nobody)",
        ..Message::default()
    };
    for reply in message.all(&h) {
        assert_eq!(reply.status, 400);
    }
    assert_eq!(h.fake.texts(), Vec::<String>::new());
}

#[test]
fn caches_five_takes_then_reuses_them() {
    let h = harness(Fake::default(), limits());
    let message = Message::default();
    for _ in 0..7 {
        message.speech(&h).assert_ogg(48000);
    }
    assert_eq!(h.fake.texts(), ["Hello there."; 5]);
    let hits: Vec<_> = h.rows().iter().map(|row| row.cache_hit).collect();
    let (miss, hit) = (Some(false), Some(true));
    assert_eq!(hits, [miss, miss, miss, miss, miss, hit, hit]);
    let entries = std::fs::read_dir(&h.cache).unwrap().count();
    assert_eq!(entries, 1);

    // Voices and texts that run together alike are kept apart.
    for (speaker, text) in [("A", "bc"), ("Ab", "c")] {
        Message {
            speaker,
            text,
            ..Message::default()
        }
        .speech(&h)
        .assert_ogg(48000);
    }
    assert_eq!(h.fake.texts()[5..], ["Bc. ", "C. "]);
    assert_eq!(std::fs::read_dir(&h.cache).unwrap().count(), 3);
}

#[test]
fn bounds_synthesis_and_refuses_a_long_wait() {
    let (fake, entered, release) = Fake::gated();
    let h = harness(
        fake,
        Limits {
            synthesis: 2,
            queue_timeout: Duration::from_millis(200),
            ..limits()
        },
    );
    let message = |text| Message {
        text,
        ..Message::default()
    };
    thread::scope(|s| {
        let first = s.spawn(|| message("One.").speech(&h));
        let second = s.spawn(|| message("Two.").speech(&h));
        let mut started = [entered.recv().unwrap(), entered.recv().unwrap()];
        started.sort();
        assert_eq!(started, ["One.", "Two."]);
        // Both turns are taken until the test lets the syntheses go.
        assert_eq!(message("Three.").speech(&h).status, 503);
        release.send(()).unwrap();
        release.send(()).unwrap();
        first.join().unwrap().assert_ogg(48000);
        second.join().unwrap().assert_ogg(48000);
    });
    release.send(()).unwrap();
    message("Four.").speech(&h).assert_ogg(48000);
    assert_eq!(entered.recv().unwrap(), "Four.");
    assert_eq!(h.fake.most.load(Ordering::SeqCst), 2);
    let statuses: Vec<_> = h.rows().iter().map(|row| row.status).collect();
    assert_eq!(statuses.iter().filter(|&&s| s == 503).count(), 1);
}

#[test]
fn toggles_logging_as_the_old_service() {
    let h = harness(Fake::default(), limits());
    let toggle = |query: &str| {
        let reply = h.get(&format!("/toggle-logging{query}"), "");
        let json: serde_json::Value = serde_json::from_slice(&reply.body).unwrap();
        (reply.status, json)
    };
    use serde_json::json;
    assert_eq!(toggle(""), (200, json!({ "api": "DEBUG" })));
    assert_eq!(toggle(""), (200, json!({ "api": "INFO" })));
    assert_eq!(toggle("?level=warning"), (200, json!({ "api": "WARNING" })));
    assert_eq!(toggle(""), (200, json!({ "api": "INFO" })));
    assert_eq!(
        toggle("?level=loud"),
        (
            400,
            json!({ "status": "error", "message": "Invalid level: LOUD" })
        )
    );
    use LevelFilter as L;
    assert_eq!(
        *h.levels.lock().unwrap(),
        [L::DEBUG, L::INFO, L::WARN, L::INFO]
    );
}

#[test]
fn records_metrics_without_the_text() {
    let h = harness(Fake::default(), limits());
    let message = Message {
        identifier: "metrics.42",
        text: "Zebra quartz nightingale.",
        ..Message::default()
    };
    let [speech, blips, radio, blips_radio, gibberish] = message.all(&h);
    let unknown = Message {
        speaker: "Nobody",
        ..message.clone()
    }
    .speech(&h);
    assert_eq!(unknown.status, 400);
    let mut rows = h.rows();
    rows[..5].sort_by_key(|row| row.route.clone());
    let row = |route: &str, reply: &Reply, cache_hit, patches| Row {
        route: route.to_owned(),
        voice: VOICE.to_owned(),
        identifier: "metrics.42".to_owned(),
        status: reply.status,
        cache_hit,
        text_length: 25,
        audio_seconds: Some(reply.length.as_deref().unwrap().parse().unwrap()),
        patches,
    };
    assert_eq!(
        rows,
        [
            row("tts", &speech, Some(false), Some(25)),
            row("tts-blips", &blips, None, None),
            row("tts-blips-radio", &blips_radio, None, None),
            row("tts-radio", &radio, None, None),
            row("tts-radio-gibberish", &gibberish, None, None),
            Row {
                voice: "Nobody".to_owned(),
                status: 400,
                cache_hit: None,
                audio_seconds: None,
                patches: None,
                ..row("tts", &speech, None, None)
            },
        ]
    );
    let Harness { server, _dir, .. } = h;
    drop(server.runtime);
    let service = Arc::into_inner(server.service).expect("the server has stopped");
    drop(service);
    for entry in std::fs::read_dir(_dir.path()).unwrap() {
        let path = entry.unwrap().path();
        if path.to_string_lossy().contains("metrics.sqlite") {
            let bytes = std::fs::read(&path).unwrap();
            for word in ["Zebra", "quartz", "nightingale"] {
                let found = bytes.windows(word.len()).any(|w| w == word.as_bytes());
                assert!(!found, "{word} in {}", path.display());
            }
        }
    }
}
