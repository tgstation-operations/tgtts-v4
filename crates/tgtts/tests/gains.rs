//! The whole-file path's leveling on the speech a gains run kept, each voice's first line as the
//! model said it, at the voices' levels the run measured.

mod common;

use std::fs;

use common::real;
use dsp::{
    loudness::{CEILING, FLOOR, integrated, normalize},
    segment::Segment,
};
use tgtts::{levels::Levels, synth, wav};

const TARGET: f64 = -16.0;

/// Each kept line, by its voice, as `tgtts-gains --keep` named it.
fn kept() -> Vec<(String, Segment)> {
    let table = real::dir("VOICE_LEVELS_FILE");
    let json: serde_json::Value = serde_json::from_slice(&fs::read(table).unwrap()).unwrap();
    let dir = real::dir("GAINS_KEPT_DIR");
    let lines: Vec<_> = (json.as_object().unwrap().keys())
        .map(|voice| {
            let file: String = (voice.chars())
                .map(|c| match c {
                    'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
                    _ => '_',
                })
                .collect();
            let audio = wav::read(&fs::read(dir.join(format!("{file}.wav"))).unwrap()).unwrap();
            let said = Segment::new(audio.rate, audio.to_pcm().samples);
            (voice.clone(), said)
        })
        .collect();
    assert!(!lines.is_empty());
    lines
}

/// The cap leaves every line the whole-file path raises by no more than it as it was, and holds
/// the rest to it. Of the 600 lines the first gains run kept, it held one: a patch of Man
/// (Druid)'s, which the path raised by 63 dB for a voice it raises by 10, its limiter taking 64
/// off: the limiter's release is longer than the line, which no gain then makes louder.
#[test]
#[ignore = "needs VOICE_LEVELS_FILE to name a gains run's table and GAINS_KEPT_DIR the lines it \
            kept"]
fn holds_the_gain_of_a_line_to_its_voices_level_and_the_margin() {
    let levels = Levels::read(&real::dir("VOICE_LEVELS_FILE")).unwrap();
    let mut held = 0;
    for (voice, said) in kept() {
        let finished = dsp::speech::finish(&synth::level(&said), 0);
        let most = levels.most_gain(&voice, TARGET);
        let free = normalize(&finished, TARGET, CEILING, f64::INFINITY);
        let capped = normalize(&finished, TARGET, CEILING, most);
        if free.gain <= most {
            assert_eq!(capped, free, "{voice}");
        } else {
            assert_eq!(capped.gain, most, "{voice}");
            held += 1;
        }
    }
    assert!(held > 0, "no line held");
}

/// A line the model said under the floor is served as it said it, where the cache would have
/// raised its peak to full scale and the loudness raised it the rest of the way to the target.
#[test]
#[ignore = "needs VOICE_LEVELS_FILE to name a gains run's table and GAINS_KEPT_DIR the lines it \
            kept"]
fn serves_a_line_said_under_the_floor_as_it_was_said() {
    let levels = Levels::read(&real::dir("VOICE_LEVELS_FILE")).unwrap();
    for (voice, said) in kept() {
        let x = |s: &Segment| -> Vec<f64> {
            let finished = dsp::speech::finish(s, 0);
            (finished.samples.iter())
                .map(|&s| f64::from(s) / 32768.0)
                .collect()
        };
        let loudness = integrated(&x(&said), said.rate);
        let down = 10f64.powf((FLOOR - 1.0 - loudness) / 20.0);
        let quiet = said
            .samples
            .iter()
            .map(|&s| (f64::from(s) * down).round() as i16);
        let quiet = Segment::new(said.rate, quiet.collect());
        let take = synth::level(&quiet);
        assert_eq!(take, quiet, "{voice}");
        let finished = dsp::speech::finish(&take, 0);
        let most = levels.most_gain(&voice, TARGET);
        let served = normalize(&finished, TARGET, CEILING, most);
        assert_eq!(served.segment, finished, "{voice}");
    }
}
