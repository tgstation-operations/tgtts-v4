//! The service end to end on the real model, voices and assets: a message for everyone, as the
//! game asks for it.

mod common;

use std::{
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use burn::tensor::{Device, FloatDType};
use common::Message;
use tgtts::{
    cache::Cache,
    ffmpeg::Ffmpeg,
    model::VoxCpm,
    service::{Limits, Logging, Loudness, Setup, Sounds},
    wav,
};
use tracing::level_filters::LevelFilter;
use voices::Voices;
use voxcpm::{Dtypes, Params};

fn dir(variable: &str) -> PathBuf {
    let dir = env::var_os(variable).unwrap_or_else(|| panic!("{variable} is not set"));
    PathBuf::from(dir)
}

/// The frames the clip `name` in `assets` takes at 48 kHz.
fn frames_at_48k(assets: &Path, name: &str) -> usize {
    let clip = wav::read(&std::fs::read(assets.join(name)).unwrap()).unwrap();
    common::resampled(clip.frames(), clip.rate, 48000)
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn serves_a_message_on_the_real_model() {
    let (checkpoint, voices, assets) = (dir("VOXCPM2_DIR"), dir("VOICES_DIR"), dir("ASSETS_DIR"));
    let f32 = FloatDType::F32;
    let dtypes = Dtypes {
        lm: f32,
        local_encoder: f32,
        local_dit: f32,
        audio_decoder: f32,
    };
    // A short cap keeps the test to a few patches.
    let params = Params {
        max_patches: 12,
        ..Params::default()
    };
    let model = VoxCpm::load(&checkpoint, &dtypes, &Device::flex(), params).unwrap();
    let voices = Voices::open(&voices).unwrap();
    let speaker = voices.names().next().unwrap().to_owned();
    let cache = tempfile::tempdir().unwrap();
    let server = common::serve(Setup {
        token: String::new(),
        voices,
        synthesizer: Arc::new(model),
        cache: Cache::open(cache.path()).unwrap(),
        ffmpeg: Ffmpeg::new("ffmpeg"),
        sounds: Sounds::open(&assets).unwrap(),
        metrics: None,
        logging: Logging::new(LevelFilter::INFO, |_| {}),
        loudness: Loudness {
            target: -16.0,
            ceiling: dsp::loudness::CEILING,
        },
        limits: Limits {
            synthesis: 1,
            cpu: 4,
            queue_timeout: Duration::from_secs(3600),
            radio_wait: Duration::from_secs(3600),
        },
        blip_seed: None,
    });
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
    let radio_frames = |frames| common::radio_frames(frames, on, off);
    assert_eq!(radio.frames(48000), radio_frames(speech.frames(48000)));
    assert_eq!(gibberish.length, radio.length);
    assert_eq!(blips_radio.frames(48000), radio_frames(blips.frames(48000)));

    let silicon = Message {
        special_filters: "silicon",
        ..message.clone()
    }
    .speech(&server);
    silicon.assert_ogg(44100);
}
