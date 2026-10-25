//! The gibberish radio's masking, spans and ducking against the old radio-gibberish service's.

mod common;

use common::{Tensors, assert_pcm_close};
use dsp::{
    Random, gibberish,
    radio::{Assets, Pcm},
};
use serde_json::Value;

/// As the radio's, the ducking being float64 as numpy's is: at most 2 of 17762 samples came out
/// one apart.
const MAX_DIFFERENCE: i32 = 1;
const MAX_DIFFERING: f64 = 0.001;

/// Python's draws, replayed in order: each a kind, 0 for `random()` and 1 for `_randbelow`, its
/// bound, and its value, a double's bits for `random()`.
struct Replay(Vec<Vec<i64>>, usize);

impl Replay {
    fn take(&mut self, kind: i64, bound: i64) -> i64 {
        let draw = &self.0[self.1];
        assert_eq!((draw[0], draw[1]), (kind, bound), "draw {}", self.1);
        self.1 += 1;
        draw[2]
    }
}

impl Random for Replay {
    fn random(&mut self) -> f64 {
        f64::from_bits(self.take(0, 0) as u64)
    }

    fn below(&mut self, n: usize) -> usize {
        self.take(1, n as i64) as usize
    }
}

fn pairs(fx: &Tensors, name: &str) -> Vec<(f64, f64)> {
    let values = if fx.contains(name) {
        fx.f64s(name)
    } else {
        Vec::new()
    };
    values.chunks(2).map(|p| (p[0], p[1])).collect()
}

#[test]
fn masks_and_finds_spans_as_the_old_service() {
    let fixture = testkit::fixture!("dsp_gibberish");
    let fx = Tensors::new(&fixture);
    let cases: Vec<Value> = serde_json::from_str(&fixture.metadata()["cases"]).unwrap();
    for (i, case) in cases.iter().enumerate() {
        let mut random = Replay(fx.int_rows(&format!("mask/{i}/draws")), 0);
        let text = case["text"].as_str().unwrap();
        let masked = gibberish::mask(text, &mut random);
        assert_eq!(masked, case["masked"].as_str().unwrap(), "{i}: mask");
        assert_eq!(random.1, random.0.len(), "{i}: draws taken");
        let letters = pairs(&fx, &format!("mask/{i}/letters"));
        let spans = gibberish::corrupted_spans(text, &masked, &letters);
        let bits = |spans: &[(f64, f64)]| -> Vec<(u64, u64)> {
            spans
                .iter()
                .map(|(s, e)| (s.to_bits(), e.to_bits()))
                .collect()
        };
        assert_eq!(
            bits(&spans),
            bits(&pairs(&fx, &format!("mask/{i}/spans"))),
            "{i}: spans"
        );
    }
}

/// The fixture's clicks and static.
fn assets(fx: &Tensors) -> Assets {
    let clip = |stem: &str| Pcm {
        rate: fx.int(&format!("{stem}/rate")) as u32,
        channels: fx.int(&format!("{stem}/channels")) as u16,
        samples: fx.pcm(&format!("{stem}/samples")),
    };
    let [on, off, noise] = ["mic_click_on", "mic_click_off", "diffstatic"].map(clip);
    Assets::new(&on, &off, &noise)
}

fn speech(fx: &Tensors, name: &str) -> Pcm {
    Pcm {
        rate: fx.int(&format!("{name}/rate")) as u32,
        channels: 1,
        samples: fx.pcm(&format!("{name}/input")),
    }
}

#[test]
fn ducks_as_the_old_service() {
    let fixture = testkit::fixture!("dsp_gibberish");
    let fx = Tensors::new(&fixture);
    let assets = assets(&fx);
    let radios: usize = fixture.metadata()["radios"].parse().unwrap();
    for case in 0..radios {
        let name = format!("radio/{case}");
        let spans = pairs(&fx, &format!("{name}/spans"));
        let actual = gibberish::gibberish(&speech(&fx, &name), &assets, &spans);
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
fn spans_whose_margins_overlap_duck_as_one() {
    let fx = Tensors::new(&testkit::fixture!("dsp_gibberish"));
    let (assets, speech) = (assets(&fx), speech(&fx, "radio/0"));
    // The first span's 80 ms after it meets the second's 60 ms before it.
    let apart = [(0.02, 0.05), (0.13, 0.16)];
    assert_eq!(
        gibberish::gibberish(&speech, &assets, &apart),
        gibberish::gibberish(&speech, &assets, &[(0.02, 0.16)])
    );
}

#[test]
fn spans_past_the_speech_duck_nothing() {
    let fx = Tensors::new(&testkit::fixture!("dsp_gibberish"));
    let (assets, speech) = (assets(&fx), speech(&fx, "radio/2"));
    let within = (0.02, 0.03);
    let past = (speech.samples.len() as f64 / 48000.0 + 0.1, 1.0);
    assert_eq!(
        gibberish::gibberish(&speech, &assets, &[within, past]),
        gibberish::gibberish(&speech, &assets, &[within])
    );
}
