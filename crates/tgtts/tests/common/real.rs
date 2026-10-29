//! The service on the real model, voices and assets, which `VOXCPM2_DIR`, `VOICES_DIR` and
//! `ASSETS_DIR` name.

use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use tgtts::{
    cache::Cache,
    ffmpeg::Ffmpeg,
    levels::Levels,
    service::{Limits, Logging, Loudness, Setup, Sounds},
    synth::Synthesizer,
    wav,
};
use tracing::level_filters::LevelFilter;
use voices::Voices;
use voxcpm_core::Params;

use super::{Message, call};

pub fn dir(variable: &str) -> PathBuf {
    let dir = env::var_os(variable).unwrap_or_else(|| panic!("{variable} is not set"));
    PathBuf::from(dir)
}

/// What the model generates with: a short cap keeps a test to a few patches.
pub fn params() -> Params {
    Params {
        max_patches: 12,
        ..Params::default()
    }
}

/// The frames the clip `name` in `assets` takes at 48 kHz.
fn frames_at_48k(assets: &Path, name: &str) -> usize {
    let clip = wav::read(&std::fs::read(assets.join(name)).unwrap()).unwrap();
    super::resampled(clip.frames(), clip.rate, 48000)
}

/// Serves `model`, which `prepare` readies with the voices while the health check answers 503,
/// and asks for a message for everyone, as the game asks for it.
pub fn serves_a_message<S: Synthesizer>(model: S, prepare: impl FnOnce(&S, &Voices)) {
    let (voices, assets) = (dir("VOICES_DIR"), dir("ASSETS_DIR"));
    let table = Voices::open(&voices).unwrap();
    let speaker = table.names().next().unwrap().to_owned();
    let cache = tempfile::tempdir().unwrap();
    let server = super::serve(Setup {
        token: String::new(),
        voices: Voices::open(&voices).unwrap(),
        synthesizer: None,
        cache: Cache::open(cache.path()).unwrap(),
        ffmpeg: Ffmpeg::new("ffmpeg"),
        sounds: Sounds::open(&assets).unwrap(),
        metrics: None,
        logging: Logging::new(LevelFilter::INFO, |_| {}),
        loudness: Loudness {
            target: -16.0,
            ceiling: dsp::loudness::CEILING,
        },
        levels: Levels::default(),
        limits: Limits {
            synthesis: 1,
            cpu: 4,
            queue_timeout: Duration::from_secs(3600),
            radio_wait: Duration::from_secs(3600),
            radio_wait_ceiling: Duration::from_secs(3600),
        },
        blip_seed: None,
    });
    let health = || call(&format!("{}/health-check", server.base), None, "").status;
    assert_eq!(health(), 503);
    prepare(&model, &table);
    server.service.ready(Arc::new(model));
    assert_eq!(health(), 200);

    let message = Message {
        speaker: &speaker,
        text: "Hello, crew.",
        ..Message::default()
    };
    let [speech, blips, radio, blips_radio, gibberish] = message.all(&server);
    for reply in [&speech, &blips, &radio, &blips_radio, &gibberish] {
        reply.assert_ogg(48000);
    }
    let tail = 12000;
    assert!(speech.frames(48000) > tail, "no speech before the tail");
    let on = frames_at_48k(&assets, "mic_click_on.wav");
    let off = frames_at_48k(&assets, "mic_click_off.wav");
    let radio_frames = |frames| super::radio_frames(frames, on, off);
    assert_eq!(radio.frames(48000), radio_frames(speech.frames(48000)));
    assert_eq!(gibberish.length, radio.length);
    assert_eq!(blips_radio.frames(48000), radio_frames(blips.frames(48000)));

    let silicon = Message {
        special_filters: "silicon",
        ..message.clone()
    }
    .speech(&server);
    silicon.assert_ogg(48000);
}
