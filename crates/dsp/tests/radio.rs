//! The radio against the old radio service, on made-up clicks, static and speech.

mod common;

use std::{env, fs};

use common::{Tensors, assert_pcm_close, read_wav};
use dsp::{
    float::to_pcm,
    radio::{self, Assets, Pcm},
};

/// The radio's steps are numpy's and scipy's, but the filter's coefficients and numpy's tanh
/// differ from theirs in the last bits: the 16-bit result came out the same but for samples 1
/// apart, at most 6 of 12297.
const MAX_DIFFERENCE: i32 = 1;
const MAX_DIFFERING: f64 = 0.001;

fn clip(fx: &Tensors, stem: &str) -> Pcm {
    Pcm {
        rate: fx.int(&format!("{stem}/rate")) as u32,
        channels: fx.int(&format!("{stem}/channels")) as u16,
        samples: fx.pcm(&format!("{stem}/samples")),
    }
}

const ASSETS: [&str; 3] = ["mic_click_on", "mic_click_off", "diffstatic"];

#[test]
fn matches_the_old_service() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    check(&fx, ASSETS.map(|s| clip(&fx, s)));
}

/// Checks the radio on `fx`'s speech, with the clicks and static `assets`, against the old
/// service's, and the assets as it matched them to 48 kHz, to the bit.
fn check(fx: &Tensors, [on, off, noise]: [Pcm; 3]) {
    let assets = Assets::new(&on, &off, &noise);
    let sounds = [assets.click_on(), assets.click_off(), assets.noise()];
    for (stem, sound) in ASSETS.iter().zip(sounds) {
        let matched = format!("matched/{stem}");
        assert_eq!(
            sound.len() as i64,
            fx.int(&format!("{matched}/len")),
            "{matched}"
        );
        let bytes: Vec<u8> = sound.iter().flat_map(|s| s.to_le_bytes()).collect();
        let crc = i64::from(crc32fast::hash(&bytes));
        assert_eq!(
            crc,
            fx.int(&format!("{matched}/crc32")),
            "{matched} differs"
        );
    }
    for case in 0..fx.int("cases") {
        let name = case.to_string();
        let speech = Pcm {
            rate: fx.int(&format!("{name}/rate")) as u32,
            channels: fx.int(&format!("{name}/channels")) as u16,
            samples: fx.pcm(&format!("{name}/input")),
        };
        let actual = radio::radio(&speech, &assets);
        assert_eq!(actual.rate, 48000);
        let expected = fx.pcm(&format!("{name}/output"));
        assert_pcm_close(
            &name,
            &actual.samples,
            &expected,
            MAX_DIFFERENCE,
            MAX_DIFFERING,
        );
    }
}

#[test]
fn averages_any_number_of_channels() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    let [on, off, noise] = ASSETS.map(|s| clip(&fx, s));
    let assets = Assets::new(&on, &off, &noise);
    let mono = Pcm {
        rate: 48000,
        channels: 1,
        samples: fx.pcm("0/input"),
    };
    let tripled = Pcm {
        channels: 3,
        samples: mono.samples.iter().flat_map(|&s| [s; 3]).collect(),
        ..mono.clone()
    };
    assert_eq!(
        radio::radio(&tripled, &assets),
        radio::radio(&mono, &assets)
    );
}

#[test]
fn loops_the_static_under_speech_longer_than_it() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    let [on, off, noise] = ASSETS.map(|s| clip(&fx, s));
    let assets = Assets::new(&on, &off, &noise);
    let speech = Pcm {
        rate: 48000,
        channels: 1,
        samples: fx.pcm("0/input").repeat(8),
    };
    assert!(speech.samples.len() > assets.noise().len());
    let radio = radio::radio(&speech, &assets);
    let tail = 4800;
    let around = assets.click_on().len() + tail + assets.click_off().len();
    assert_eq!(radio.samples.len(), around + speech.samples.len());
}

/// The radio of `len` samples of silence at 48 kHz, which narrowing leaves silent: the static at
/// 0.2 under it and on for 0.1 s after, between the clicks.
fn around_silence(assets: &Assets, len: usize) -> Vec<i16> {
    let noise = assets.noise();
    let mut expected = assets.click_on().to_vec();
    expected.extend(noise[..len].iter().map(|n| n * 0.2));
    expected.extend(&noise[len..len + 4800]);
    expected.extend(assets.click_off());
    let peak = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
    let expected: Vec<f64> = expected.iter().map(|v| v / peak * 0.9).collect();
    to_pcm(&expected)
}

fn silence(len: usize) -> Pcm {
    Pcm {
        rate: 48000,
        channels: 1,
        samples: vec![0; len],
    }
}

#[test]
fn the_static_goes_on_after_the_speech() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    let [on, off, noise] = ASSETS.map(|s| clip(&fx, s));
    let assets = Assets::new(&on, &off, &noise);
    let radio = radio::radio(&silence(2400), &assets);
    assert_eq!(radio.samples, around_silence(&assets, 2400));
}

#[test]
fn puts_the_radio_around_no_speech() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    let [on, off, noise] = ASSETS.map(|s| clip(&fx, s));
    let assets = Assets::new(&on, &off, &noise);
    let radio = radio::radio(&silence(0), &assets);
    assert_eq!(radio.samples, around_silence(&assets, 0));
}

/// Making the golden needs the real assets, so a run of every generator leaves it as it is: this
/// refuses one that such a run has left behind.
#[test]
fn real_assets_golden_was_made_as_the_fixtures_were() {
    let (golden, fixtures) = (
        testkit::fixture!("dsp_radio_assets"),
        testkit::fixture!("dsp_radio"),
    );
    let regenerate = "with ASSETS_DIR set, regenerate it: nix run .#fixtures dsp_radio_assets";
    for key in ["librosa", "numpy", "scipy", "soundfile", "torch"] {
        assert_eq!(
            golden.metadata()[key],
            fixtures.metadata()[key],
            "the golden's {key} differs from the fixtures'; {regenerate}"
        );
    }
    let generators = ["dsp_radio_assets.py", "dsp_radio.py"].map(|name| {
        fs::read(format!(
            "{}/../../fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    });
    let crc32 = format!("{:08x}", crc32fast::hash(&generators.concat()));
    assert_eq!(
        golden.metadata().get("generator_crc32"),
        Some(&crc32),
        "the generators changed after making the golden; {regenerate}"
    );
}

#[test]
#[ignore = "needs ASSETS_DIR to name the old service's assets, with mic_click_on.wav, \
            mic_click_off.wav and diffstatic.wav"]
fn the_real_assets_match_the_old_service() {
    let Some(dir) = env::var_os("ASSETS_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("ASSETS_DIR is not set");
    };
    let dir = std::path::absolute(dir).unwrap();
    let assets = ASSETS.map(|stem| {
        let (rate, channels, samples) = read_wav(&dir.join(format!("{stem}.wav")));
        Pcm {
            rate,
            channels,
            samples,
        }
    });
    check(
        &Tensors::new(&testkit::fixture!("dsp_radio_assets")),
        assets,
    );
}

/// Given the peaks the whole radio scales by, the radio on a stream is the whole radio to the
/// bit, however the speech comes and at whatever rate, silence and nothing included.
#[test]
fn a_stream_at_the_wholes_peaks_is_the_whole() {
    let fx = Tensors::new(&testkit::fixture!("dsp_radio"));
    let [on, off, noise] = ASSETS.map(|s| clip(&fx, s));
    let assets = Assets::new(&on, &off, &noise);
    let mut speeches: Vec<Pcm> = (0..fx.int("cases"))
        .map(|case| Pcm {
            rate: fx.int(&format!("{case}/rate")) as u32,
            channels: fx.int(&format!("{case}/channels")) as u16,
            samples: fx.pcm(&format!("{case}/input")),
        })
        .collect();
    let voice: Vec<i16> = (0..30000)
        .map(|i| ((f64::from(i) * 0.031).sin() * (f64::from(i) * 0.0007).sin() * 20000.0) as i16)
        .collect();
    for (rate, channels) in [(48000, 1), (44100, 2), (8000, 1)] {
        let samples = voice
            .iter()
            .flat_map(|&s| std::iter::repeat_n(s, channels))
            .collect();
        speeches.push(Pcm {
            rate,
            channels: channels as u16,
            samples,
        });
    }
    speeches.push(silence(2400));
    speeches.push(silence(0));
    for speech in &speeches {
        let whole = radio::radio(speech, &assets).samples;
        let peaks = radio::peaks(speech, &assets);
        let channels = usize::from(speech.channels);
        for frames in [1, 480, 7680] {
            let mut stream = radio::Stream::new(&assets, peaks, speech.rate, speech.channels);
            let mut streamed = Vec::new();
            for chunk in speech.samples.chunks(frames * channels) {
                stream.process(chunk, &mut streamed);
            }
            stream.finish(&mut streamed);
            assert!(
                streamed == whole,
                "{} Hz in {} channels, {} frames, in chunks of {frames}",
                speech.rate,
                speech.channels,
                speech.samples.len() / channels
            );
        }
    }
}
