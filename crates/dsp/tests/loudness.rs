//! The loudness meter on EBU Tech 3341's test signals in one channel, against the readings the
//! document requires and those of ffmpeg's ebur128 filter.

mod common;

use std::f64::consts::PI;

use common::Tensors;
use dsp::{
    loudness::{CEILING, FLOOR, integrated, normalize, true_peak},
    quantity::{Gain, Loudness, SampleRate},
    segment::Segment,
};

const RATE: SampleRate = SampleRate::new(48000);

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
        let loudness = integrated(&x, RATE).value();
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
        let peak = true_peak(&x).value();
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
    let held = integrated(&tone(&[-23.0], &[20.0]), RATE).value();
    for seconds in [0.39, 0.1] {
        let short = integrated(&tone(&[-23.0], &[seconds]), RATE).value();
        assert!(
            (short - held).abs() < 0.05,
            "{seconds} s: {short} for {held}"
        );
    }
}

#[test]
fn silence_has_no_loudness_or_peak() {
    for len in [0, 100, 48000] {
        assert_eq!(integrated(&vec![0.0; len], RATE).value(), f64::NEG_INFINITY);
        assert_eq!(true_peak(&vec![0.0; len]).value(), f64::NEG_INFINITY);
    }
    // A tone under the absolute gate is gated away as silence is.
    assert_eq!(
        integrated(&tone(&[-80.0], &[1.0]), RATE).value(),
        f64::NEG_INFINITY
    );
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
        let normalized = normalize(
            &quiet,
            Loudness::new(-20.0),
            CEILING,
            Gain::new(f64::INFINITY),
        );
        let loudness = integrated(&measured(&normalized.segment), RATE).value();
        assert!((loudness + 20.0).abs() < 0.01, "{seconds} s: {loudness}");
        assert_eq!((normalized.reduction, normalized.limited), (0.0, 0.0));
    }
}

/// A quiet tone with loud clicks in it, every `every` seconds.
fn clicked(every: f64) -> Vec<f64> {
    let mut x = tone(&[-40.0], &[3.0]);
    for click in (0..)
        .map(|i| f64::from(i) * every + 0.05)
        .take_while(|&t| t < 3.0)
    {
        x[(click * f64::from(RATE)) as usize] = 0.3;
    }
    x
}

#[test]
fn limits_the_peaks_to_reach_the_target() {
    // One click, and then so many that the limiter is never done with them, which takes the
    // loudness down and the gain back up.
    for every in [10.0, 0.05] {
        let normalized = normalize(
            &pcm(&clicked(every)),
            Loudness::new(-20.0),
            CEILING,
            Gain::new(f64::INFINITY),
        );
        let x = measured(&normalized.segment);
        let (loudness, peak) = (integrated(&x, RATE).value(), true_peak(&x).value());
        assert!((loudness + 20.0).abs() < 0.1, "{every} s: {loudness} LUFS");
        assert!(peak <= CEILING.value(), "{every} s: {peak} dBTP");
        assert!(normalized.reduction > 12.0, "{every} s: {normalized:?}");
        assert!(normalized.limited > 0.0);
    }
}

#[test]
fn leaves_silence_as_it_is() {
    for x in [vec![0.0; 48000], tone(&[-80.0], &[1.0])] {
        let silence = pcm(&x);
        let normalized = normalize(
            &silence,
            Loudness::new(-20.0),
            CEILING,
            Gain::new(f64::INFINITY),
        );
        assert_eq!(normalized.segment, silence);
        assert_eq!(
            (
                normalized.gain.value(),
                normalized.reduction,
                normalized.limited
            ),
            (0.0, 0.0, 0.0)
        );
    }
}

#[test]
fn leaves_what_is_quieter_than_the_floor_as_it_is() {
    // A tone 3 LU under the floor, and a patch of noise a few dozen steps of 16 bits high, with
    // the silence the service puts after speech, as a model gives for a line it hardly said.
    let mut noise: Vec<f64> = (0..7680)
        .map(|i| f64::from((i * 7919) % 97 - 48) / 32768.0)
        .collect();
    noise.resize(19680, 0.0);
    for x in [tone(&[FLOOR.value()], &[1.0]), noise] {
        let near = pcm(&x);
        let loudness = integrated(&measured(&near), RATE).value();
        assert!(
            (-70.0..FLOOR.value()).contains(&loudness),
            "{loudness} LUFS"
        );
        let normalized = normalize(
            &near,
            Loudness::new(-16.0),
            CEILING,
            Gain::new(f64::INFINITY),
        );
        assert_eq!(normalized.segment, near);
        assert_eq!(normalized.gain.value(), 0.0);
    }
}

#[test]
fn raises_by_no_more_than_the_most() {
    let quiet = pcm(&tone(&[-50.0], &[1.0]));
    let normalized = normalize(&quiet, Loudness::new(-16.0), CEILING, Gain::new(12.0));
    let loudness = integrated(&measured(&normalized.segment), RATE).value();
    assert_eq!(normalized.gain.value(), 12.0);
    assert!((loudness + 41.0).abs() < 0.01, "{loudness} LUFS");

    // Clicks in silence the limiter holds down, so that no gain brings them to the target: the
    // gain runs on to hundreds of decibels unless it is held.
    let mut clicks = vec![0.0; 19680];
    for at in [300, 2000, 5100, 7000] {
        clicks[at] = 0.99;
    }
    let clicks = pcm(&clicks);
    let held = normalize(&clicks, Loudness::new(-16.0), CEILING, Gain::new(24.0));
    assert_eq!(held.gain.value(), 24.0);
    assert!(held.reduction < 26.0, "{held:?}");
    let unheld = normalize(
        &clicks,
        Loudness::new(-16.0),
        CEILING,
        Gain::new(f64::INFINITY),
    );
    assert!(unheld.gain.value() > 100.0, "{} dB", unheld.gain.value());
}
