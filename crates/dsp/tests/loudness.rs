//! The loudness meter on EBU Tech 3341's test signals in one channel, against the readings the
//! document requires and those of ffmpeg's ebur128 filter.

mod common;

use std::f64::consts::PI;

use common::Tensors;
use dsp::{
    loudness::{CEILING, integrated, normalize, true_peak},
    segment::Segment,
};

const RATE: u32 = 48000;

/// ffmpeg gates through a histogram of 0.01 LU bins, which floors its integrated loudness: it
/// came out 0.0060 to 0.0068 LU under this meter's.
const FFMPEG_LOUDNESS: f64 = 0.01;
/// ffmpeg oversamples to 192 kHz through its own resampler, where the example filter of BS.1770-4
/// droops towards the top of the band, by 0.29 dB at a sixth of the rate in case 17; the other
/// cases came out within 0.08 dB of ffmpeg's.
const FFMPEG_TRUE_PEAK: f64 = 0.3;

fn cases(fixture: &testkit::Fixture, key: &str) -> Vec<String> {
    fixture.metadata()[key]
        .split(',')
        .map(str::to_owned)
        .collect()
}

/// A 1 kHz sine at each of `levels`, in dBFS, for each of `seconds`, its phase running on.
fn tone(levels: &[f64], seconds: &[f64]) -> Vec<f64> {
    let amplitudes = levels.iter().zip(seconds).flat_map(|(level, s)| {
        let len = (s * f64::from(RATE)).round() as usize;
        std::iter::repeat_n(10f64.powf(level / 20.0), len)
    });
    amplitudes
        .enumerate()
        .map(|(n, a)| a * (2.0 * PI * 1000.0 * n as f64 / f64::from(RATE)).sin())
        .collect()
}

#[test]
fn integrated_loudness_meets_tech_3341() {
    let fixture = testkit::fixture!("dsp_loudness");
    let fx = Tensors::new(&fixture);
    for case in cases(&fixture, "tones") {
        let name = |what: &str| format!("tone/{case}/{what}");
        let x = tone(&fx.f64s(&name("levels")), &fx.f64s(&name("seconds")));
        let loudness = integrated(&x, RATE);
        let expected = fx.f64s(&name("expected"))[0];
        assert!(
            (loudness - expected).abs() <= 0.1,
            "case {case}: {loudness} LUFS, not {expected}"
        );
        let ffmpeg = fx.f64s(&name("ffmpeg"))[0];
        assert!(
            (loudness - ffmpeg).abs() <= FFMPEG_LOUDNESS,
            "case {case}: {loudness} LUFS, ffmpeg's {ffmpeg}"
        );
    }
}

#[test]
fn true_peak_meets_tech_3341() {
    let fixture = testkit::fixture!("dsp_loudness");
    let fx = Tensors::new(&fixture);
    for case in cases(&fixture, "peaks") {
        let name = |what: &str| format!("peak/{case}/{what}");
        let x: Vec<f64> = fx
            .f32s(&name("samples"))
            .into_iter()
            .map(f64::from)
            .collect();
        let peak = true_peak(&x);
        let expected = fx.f64s(&name("expected"))[0];
        assert!(
            (expected - 0.4..=expected + 0.2).contains(&peak),
            "case {case}: {peak} dBTP, not {expected}"
        );
        let ffmpeg = fx.f64s(&name("ffmpeg"))[0];
        assert!(
            (peak - ffmpeg).abs() <= FFMPEG_TRUE_PEAK,
            "case {case}: {peak} dBTP, ffmpeg's {ffmpeg}"
        );
    }
}

#[test]
fn clips_shorter_than_a_block_read_as_the_sound_held() {
    let held = integrated(&tone(&[-23.0], &[20.0]), RATE);
    for seconds in [0.39, 0.1] {
        let short = integrated(&tone(&[-23.0], &[seconds]), RATE);
        assert!(
            (short - held).abs() < 0.05,
            "{seconds} s: {short} for {held}"
        );
    }
}

#[test]
fn silence_has_no_loudness_or_peak() {
    for len in [0, 100, 48000] {
        assert_eq!(integrated(&vec![0.0; len], RATE), f64::NEG_INFINITY);
        assert_eq!(true_peak(&vec![0.0; len]), f64::NEG_INFINITY);
    }
    // A tone under the absolute gate is gated away as silence is.
    assert_eq!(integrated(&tone(&[-80.0], &[1.0]), RATE), f64::NEG_INFINITY);
}

/// `x` in 16 bits.
fn pcm(x: &[f64]) -> Segment {
    Segment::new(
        RATE,
        x.iter().map(|s| (s * 32768.0).round() as i16).collect(),
    )
}

fn measured(segment: &Segment) -> Vec<f64> {
    (segment.samples.iter())
        .map(|&s| f64::from(s) / 32768.0)
        .collect()
}

#[test]
fn normalizes_to_the_target() {
    for seconds in [3.0, 0.2] {
        let quiet = pcm(&tone(&[-40.0], &[seconds]));
        let loudness = integrated(&measured(&normalize(&quiet, -20.0, CEILING)), RATE);
        assert!((loudness + 20.0).abs() < 0.01, "{seconds} s: {loudness}");
    }
}

#[test]
fn the_ceiling_holds_the_gain_down() {
    // A quiet tone with one loud click in it, whose loudness the click hardly moves.
    let mut x = tone(&[-40.0], &[3.0]);
    x[RATE as usize] = 0.3;
    let normalized = measured(&normalize(&pcm(&x), -20.0, CEILING));
    let peak = true_peak(&normalized);
    assert!((peak - CEILING).abs() < 0.01, "{peak} dBTP");
    assert!(integrated(&normalized, RATE) < -25.0);
}

#[test]
fn leaves_silence_as_it_is() {
    for x in [vec![0.0; 48000], tone(&[-80.0], &[1.0])] {
        let silence = pcm(&x);
        assert_eq!(normalize(&silence, -20.0, CEILING), silence);
    }
}
