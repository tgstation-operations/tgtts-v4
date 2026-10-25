//! The blips against the old blips service and its frontend, restated with the fixes the port
//! makes, on a made-up voice and a real one.

mod common;

use std::{collections::BTreeMap, env, fs, path::PathBuf};

use common::{Tensors, first_difference};
use dsp::{
    Random,
    blips::{self, Error},
    float::{from_pcm, to_pcm},
    resample::resample,
    segment::{self, Segment},
};
use voices::{BlipSet, Blips, Voices};

/// The made-up voice's rate.
const RATE: u32 = 16000;
const KEYS: [&str; 10] = ["a", "b", "c", "h", "l", "o", "1", "2", "Deska", "Gwah"];
const CROSSFADE_TOO_LONG: i64 = 1;

/// Python's draws, replayed in order.
struct Replay(Vec<f64>, usize);

impl Random for Replay {
    fn random(&mut self) -> f64 {
        let draw = self.0[self.1];
        self.1 += 1;
        draw
    }

    fn below(&mut self, _: usize) -> usize {
        unreachable!("the blips draw no integers")
    }
}

/// Draws of 0.5, which detune nothing.
struct Still;

impl Random for Still {
    fn random(&mut self) -> f64 {
        0.5
    }

    fn below(&mut self, _: usize) -> usize {
        unreachable!("the blips draw no integers")
    }
}

/// The made-up voice's samples, by key.
fn samples(fx: &Tensors) -> BTreeMap<&'static str, Vec<i16>> {
    KEYS.iter()
        .map(|&key| (key, fx.pcm(&format!("voice/{key}"))))
        .collect()
}

/// A voice, as a blips file the voices crate opens.
struct Voice(PathBuf, Blips);

impl Voice {
    /// The made-up voice.
    fn new(fx: &Tensors, test: &str) -> Self {
        Self::of(test, RATE, samples(fx))
    }

    /// The made-up voice resampled to the service's rate.
    fn at_service_rate(fx: &Tensors, test: &str) -> Self {
        let resampled = samples(fx).into_iter().map(|(key, sample)| {
            let sample = resample(&from_pcm(&sample), f64::from(RATE), f64::from(dsp::RATE));
            (key, to_pcm(&sample))
        });
        Self::of(test, dsp::RATE, resampled.collect())
    }

    /// The voice of `samples` at `rate`.
    fn of(test: &str, rate: u32, samples: BTreeMap<&str, Vec<i16>>) -> Self {
        let samples = samples
            .into_iter()
            .map(|(key, sample)| (("male".to_owned(), "1".to_owned(), key.to_owned()), sample))
            .collect();
        let set = BlipSet {
            sample_rate: rate,
            samples,
        };
        let path = std::env::temp_dir().join(format!("dsp-{test}-{}.blips", std::process::id()));
        set.write(&path).unwrap();
        let blips = Blips::open(&path).unwrap();
        Self(path, blips)
    }
}

impl Drop for Voice {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn texts(fx: &testkit::Fixture, key: &str) -> Vec<String> {
    fx.metadata()[key].split('\n').map(str::to_owned).collect()
}

/// Checks `actual` against what the old service made of the case `name`: where it shifted no
/// pitch, to the bit; else within what the pitch shift's rounding moves samples by.
#[track_caller]
fn check(fx: &Tensors, name: &str, actual: Result<Segment, Error>, random: &Replay) {
    assert_eq!(random.1, random.0.len(), "{name}: draws taken");
    let exact = fx.int(&format!("{name}/shifts")) == 0;
    let error = fx.int(&format!("{name}/error"));
    let actual = match (actual, error) {
        (Ok(actual), 0) => actual,
        (Err(Error::Segment(segment::Error::CrossfadeTooLong { .. })), CROSSFADE_TOO_LONG) => {
            return;
        }
        (actual, error) => panic!("{name}: {actual:?} where the service gave error {error}"),
    };
    if fx.contains(&format!("{name}/output_crc32")) {
        assert!(exact, "{name}: a checksum of shifted blips");
        let bytes: Vec<u8> = actual
            .samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(
            actual.samples.len() as i64,
            fx.int(&format!("{name}/output_len")),
            "{name}"
        );
        let crc = i64::from(crc32fast::hash(&bytes));
        assert_eq!(
            crc,
            fx.int(&format!("{name}/output_crc32")),
            "{name}: samples differ"
        );
        return;
    }
    let expected = fx.pcm(&format!("{name}/output"));
    if exact {
        let diff = first_difference(&actual.samples, &expected);
        assert!(actual.samples == expected, "{name}: {diff}");
        return;
    }
    assert_eq!(actual.samples.len(), expected.len(), "{name}");
    let differences: Vec<i32> = (actual.samples.iter().zip(&expected))
        .map(|(a, e)| (i32::from(*a) - i32::from(*e)).abs())
        .collect();
    let worst = differences.iter().max();
    assert!(
        worst <= Some(&MAX_DIFFERENCE),
        "{name}: samples {worst:?} apart"
    );
    let differing = differences.iter().filter(|&&d| d != 0).count();
    let fraction = differing as f64 / expected.len() as f64;
    assert!(
        fraction <= MAX_DIFFERING,
        "{name}: {differing} samples differ"
    );
}

/// The pitch shift's samples that truncate to a 16-bit value one off, scaled by the gain that
/// brings the blips to -25 dBFS, made samples of the shifted cases up to 2 apart here and 3 in the
/// real voice.
const MAX_DIFFERENCE: i32 = 4;
/// Of their samples, at most 0.17% differed.
const MAX_DIFFERING: f64 = 0.01;

#[test]
fn sentences_match_the_old_service() {
    let fixture = testkit::fixture!("dsp_blips");
    let fx = Tensors::new(&fixture);
    let voice = Voice::new(&fx, "sentences");
    for (case, text) in texts(&fixture, "texts").iter().enumerate() {
        let name = case.to_string();
        let mut random = Replay(fx.f64s(&format!("{name}/draws")), 0);
        let pitch = fx.int(&format!("{name}/pitch")) as i32;
        let actual = blips::sentence(text, &voice.1, "male", "1", pitch, &mut random);
        check(&fx, &name, actual, &random);
    }
}

#[test]
fn lowercase_text_sounds() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::new(&fx, "lowercase");
    let blips = |text| blips::sentence(text, &voice.1, "male", "1", 0, &mut Still).unwrap();
    let lowercase = blips("hob");
    assert!(!lowercase.samples.is_empty());
    // A capital at an odd index makes no sound of its own.
    assert_eq!(lowercase, blips("hobA"));
}

#[test]
fn a_sentence_starts_at_its_first_sound() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::new(&fx, "first-sound");
    let blips = |text| blips::sentence(text, &voice.1, "male", "1", 0, &mut Still).unwrap();
    assert_eq!(blips("'Hob"), blips("o"));
}

#[test]
fn crossfades_are_no_longer_than_either_side() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::new(&fx, "crossfades");
    // The made-up Gwah, stripped of its silence, is shorter than a crossfade.
    for text in ["Ha!", "!.a"] {
        let blips = blips::sentence(text, &voice.1, "male", "1", 0, &mut Still);
        assert!(blips.is_ok_and(|b| !b.samples.is_empty()), "{text}");
    }
}

#[test]
fn messages_need_a_voice_at_the_service_rate() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::new(&fx, "rate");
    let message = blips::message("Hob.", &voice.1, "male", "1", 0, &mut Still);
    let samples = message.as_ref().map(|m| m.samples.len());
    assert!(matches!(message, Err(Error::Rate(RATE))), "{samples:?}");
}

#[test]
fn messages_take_the_pitch() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::at_service_rate(&fx, "pitch");
    let message = blips::message("Hob.", &voice.1, "male", "1", 5, &mut Still).unwrap();
    let mut expected = blips::sentence("Hob.", &voice.1, "male", "1", 5, &mut Still).unwrap();
    expected
        .samples
        .extend(Segment::silent(250, dsp::RATE).samples);
    assert_eq!(message, expected);
}

#[test]
fn silent_sentences_take_no_time() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let voice = Voice::at_service_rate(&fx, "silent");
    let message = |text| blips::message(text, &voice.1, "male", "1", 0, &mut Still).unwrap();
    assert!(message("...").samples.is_empty());
    // pysbd makes three sentences of this, the second without a sound.
    let mut expected = Vec::new();
    for text in ["Hob.", "b!"] {
        let sentence = blips::sentence(text, &voice.1, "male", "1", 0, &mut Still).unwrap();
        expected.extend(sentence.samples);
        expected.extend(Segment::silent(250, dsp::RATE).samples);
    }
    assert_eq!(message("Hob. - . b!").samples, expected);
}

/// A blip loud from its first sample to its last, `ms` long, which stripping leaves whole.
fn loud(ms: u32) -> Vec<i16> {
    (0..RATE * ms / 1000)
        .map(|i| if i % 2 == 0 { 8000 } else { -8000 })
        .collect()
}

#[test]
fn question_marks_are_faded() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let mut samples = samples(&fx);
    samples.insert("Deska", loud(100));
    let voice = Voice::of("faded", RATE, samples);
    // The b, at an odd index, makes no sound.
    let blips = blips::sentence("?b", &voice.1, "male", "1", 0, &mut Still).unwrap();
    let peak = blips.max() as i16;
    let ends = [blips.samples[0], blips.samples[blips.samples.len() - 1]];
    assert!(
        ends.iter().all(|s| s.abs() < peak / 10),
        "{ends:?} of {peak}"
    );
}

#[test]
fn blips_shorter_than_their_fades_keep_their_length() {
    let fx = Tensors::new(&testkit::fixture!("dsp_blips"));
    let mut samples = samples(&fx);
    samples.insert("Deska", loud(5));
    let voice = Voice::of("short", RATE, samples);
    let blips = blips::sentence("?b", &voice.1, "male", "1", 0, &mut Still).unwrap();
    assert_eq!(blips.samples.len(), loud(5).len());
}

/// Making the golden needs a real voice, so a run of every generator leaves it as it is: this
/// refuses one that such a run has left behind.
#[test]
fn real_voice_golden_was_made_as_the_fixtures_were() {
    let (golden, fixtures) = (
        testkit::fixture!("dsp_blips_voice"),
        testkit::fixture!("dsp_blips"),
    );
    let regenerate = "with VOICES_DIR set, regenerate it: nix run .#fixtures dsp_blips_voice";
    for key in ["librosa", "pydub", "pysbd", "torch"] {
        assert_eq!(
            golden.metadata()[key],
            fixtures.metadata()[key],
            "the golden's {key} differs from the fixtures'; {regenerate}"
        );
    }
    let generators = ["dsp_blips_voice.py", "dsp_blips.py"].map(|name| {
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
#[ignore = "needs VOICES_DIR to name a voices directory with Man (Ace Detective)'s blips"]
fn blips_in_a_real_voice_match_the_old_service() {
    let Some(dir) = env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set");
    };
    let fixture = testkit::fixture!("dsp_blips_voice");
    let fx = Tensors::new(&fixture);
    let voice = &fixture.metadata()["voice"];
    let blips = Voices::open(std::path::absolute(dir).unwrap())
        .unwrap()
        .blips(voice)
        .unwrap();
    for (case, text) in texts(&fixture, "texts").iter().enumerate() {
        let name = case.to_string();
        let mut random = Replay(fx.f64s(&format!("{name}/draws")), 0);
        let pitch = fx.int(&format!("{name}/pitch")) as i32;
        let actual = blips::sentence(text, &blips, "male", "1", pitch, &mut random);
        check(&fx, &name, actual, &random);
    }
    for (case, text) in texts(&fixture, "messages").iter().enumerate() {
        let name = format!("message/{case}");
        let mut random = Replay(fx.f64s(&format!("{name}/draws")), 0);
        let pitch = fx.int(&format!("{name}/pitch")) as i32;
        let actual = blips::message(text, &blips, "male", "1", pitch, &mut random);
        check(&fx, &name, actual, &random);
    }
}
