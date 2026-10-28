//! The finishing of speech against the old frontend's.

mod common;

use std::f64::consts::TAU;

use common::{Tensors, assert_pcm_close};
use dsp::{float::from_pcm, pitch::pitch_shift, segment::Segment, speech};

/// The pitch shift's: a sample one off now and then, at most 7 of 19200 here.
const MAX_DIFFERENCE: i32 = 1;
const MAX_DIFFERING: f64 = 0.01;

#[test]
fn finishes_as_the_old_frontend() {
    let fx = Tensors::new(&testkit::fixture!("dsp_speech"));
    let input = Segment::new(48000, fx.pcm("input"));
    for pitch in fx.ints("pitches") {
        let actual = speech::finish(&input, pitch as i32);
        if pitch == 0 {
            let bytes: Vec<u8> = actual
                .samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            assert_eq!(actual.samples.len() as i64, fx.int("0/output_len"));
            assert_eq!(i64::from(crc32fast::hash(&bytes)), fx.int("0/output_crc32"));
        } else {
            let expected = fx.pcm(&format!("{pitch}/output"));
            let name = format!("pitch {pitch}");
            let shifted = shifted(&input, pitch);
            let peak = f64::from(shifted.iter().fold(0f32, |m, s| m.max(s.abs())));
            if peak <= 1.0 {
                assert_pcm_close(
                    &name,
                    &actual.samples,
                    &expected,
                    MAX_DIFFERENCE,
                    MAX_DIFFERING,
                );
            } else {
                // The old frontend clipped a shift past full scale where the port brings it down
                // by its peak: what the old frontend left whole is the port's brought back up.
                let whole = |e: &i16| *e != i16::MAX && *e != i16::MIN;
                assert!(!expected.iter().all(whole), "{name}: unclipped");
                for (&a, &e) in actual.samples.iter().zip(&expected) {
                    let apart = f64::from(a) * peak - f64::from(e);
                    let near = apart.abs() <= f64::from(MAX_DIFFERENCE) + peak;
                    assert!(near || !whole(&e), "{name}: {a} for {e}");
                }
            }
        }
    }
}

/// Two seconds of a buzz at a wandering pitch, its harmonics' phases spread as Schroeder's, so
/// that its peaks stand little over the rest, at the level the cache keeps speech at.
fn buzz() -> Segment {
    let mut phase = 0.0;
    let x = (0..96000).map(|i| {
        let t = f64::from(i) / 48000.0;
        phase += TAU * (120.0 + 30.0 * (TAU * 0.7 * t).sin()) / 48000.0;
        let buzz: f64 = (1..20)
            .map(|h| {
                let h = f64::from(h);
                (phase * h + TAU * h * h / 40.0).sin() / h.sqrt()
            })
            .sum();
        (8000.0 * (0.6 + 0.4 * (TAU * 2.5 * t).sin()) * buzz) as i16
    });
    Segment::new(48000, x.collect()).normalize(0.1)
}

/// `speech` with the silence after it shifted by `pitch` quarter tones, as floats.
fn shifted(speech: &Segment, pitch: i64) -> Vec<f32> {
    let mut x = from_pcm(&speech.samples);
    x.extend(from_pcm(
        &Segment::silent(speech::TAIL, speech.rate).samples,
    ));
    pitch_shift(&x, speech.rate, pitch as f64, 24)
}

/// The shift spreads the buzz's phases anew, which takes its peaks past full scale, where the old
/// frontend's 16 bits clipped them: the shift comes out whole, brought down to fit.
#[test]
fn brings_a_shift_past_full_scale_down_whole() {
    let take = buzz();
    let shifted = shifted(&take, -7);
    let peak = shifted.iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(peak > 1.1, "{peak}");
    let finished = speech::finish(&take, -7);
    let scale = 32767.0 / f64::from(peak);
    for (&out, &s) in finished.samples.iter().zip(&shifted) {
        let apart = f64::from(out) - f64::from(s) * scale;
        assert!(apart.abs() <= 1.0, "{out} for {s}");
    }
}
