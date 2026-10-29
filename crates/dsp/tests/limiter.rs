//! The true-peak limiter on EBU Tech 3341's true-peak signals raised over the ceiling, and on a
//! tone with a click in it.

mod common;

use std::f64::consts::PI;

use common::Tensors;
use dsp::{
    limiter::{Limiter, limit},
    loudness::{CEILING, true_peak},
    quantity::SampleRate,
};

const RATE: SampleRate = SampleRate::new(48000);

/// How far over the ceiling the waveform's peak may come out of the limiter, by the reading below:
/// the limiter's own interpolator, of 32 taps a phase, reads the bursts of cases 21 and 23, whose
/// content reaches the top of the band, 0.014 dB under it. Everywhere else it comes out under.
const WAVEFORM_OVER: f64 = 0.02;

/// The waveform's peak, in dBFS, sixteen points to a sample through a long windowed sinc: a
/// reading apart from the limiter's own, nearer the waveform between the samples.
fn waveform_peak(x: &[f64]) -> f64 {
    const HALF: i64 = 64;
    const POINTS: usize = 16;
    let i0 = |x: f64| {
        let (mut sum, mut term) = (1.0, 1.0);
        for k in 1..60 {
            term *= (x / (2.0 * f64::from(k))).powi(2);
            sum += term;
        }
        sum
    };
    let beta = 10.0;
    let mut peak = x.iter().fold(0f64, |m, s| m.max(s.abs()));
    for k in 1..POINTS {
        let at = k as f64 / POINTS as f64;
        let taps: Vec<(i64, f64)> = (1 - HALF..=HALF)
            .map(|o| {
                let t = at - o as f64;
                let window =
                    i0(beta * (1.0 - (t / HALF as f64).powi(2)).max(0.0).sqrt()) / i0(beta);
                (o, (PI * t).sin() / (PI * t) * window)
            })
            .collect();
        for i in 0..x.len() as i64 {
            let y: f64 = (taps.iter())
                .filter_map(|&(o, h)| Some(h * x.get(usize::try_from(i + o).ok()?)?))
                .sum();
            peak = peak.max(y.abs());
        }
    }
    20.0 * peak.log10()
}

fn tone(level: f64, seconds: f64) -> Vec<f64> {
    let amplitude = 10f64.powf(level / 20.0);
    (0..(seconds * f64::from(RATE)) as usize)
        .map(|n| amplitude * (2.0 * PI * 1000.0 * n as f64 / f64::from(RATE)).sin())
        .collect()
}

#[test]
fn holds_tech_3341_peaks_under_the_ceiling() {
    let fixture = testkit::fixture!("dsp_loudness");
    let fx = Tensors::new(&fixture);
    for case in fixture.metadata()["peaks"].split(',') {
        // 12 dB louder, which takes every case well over the ceiling.
        let x: Vec<f64> = (fx.f32s(&format!("peak/{case}/samples")).into_iter())
            .map(|s| 4.0 * f64::from(s))
            .collect();
        assert!(true_peak(&x).value() > CEILING.value() + 6.0, "case {case}");
        let limited = limit(&x, RATE, CEILING).samples;
        let (peak, waveform) = (true_peak(&limited).value(), waveform_peak(&limited));
        assert!(peak <= CEILING.value(), "case {case}: {peak} dBTP");
        assert!(
            waveform <= CEILING.value() + WAVEFORM_OVER,
            "case {case}: the waveform at {waveform} dBFS"
        );
    }
}

#[test]
fn leaves_what_is_under_the_ceiling_alone() {
    let x = tone(-3.0, 1.0);
    let limited = limit(&x, RATE, CEILING);
    assert_eq!(limited.samples, x);
    assert_eq!((limited.reduction, limited.fraction), (0.0, 0.0));
}

#[test]
fn takes_down_only_what_a_peak_needs() {
    let mut x = tone(-20.0, 3.0);
    let click = RATE.value() as usize;
    x[click] = 1.5;
    let limited = limit(&x, RATE, CEILING);
    let over = true_peak(&x).value() - CEILING.value();
    assert!(
        (over..over + 0.01).contains(&limited.reduction),
        "{} dB for {over}",
        limited.reduction
    );
    // Untouched until a lookahead and the interpolators' reach before the click.
    let before = click - 240 - 16 - 1;
    assert_eq!(limited.samples[..before], x[..before]);
    assert_ne!(limited.samples[before + 1], x[before + 1]);
    // And from where the gain is back to unity, 0.7 s on.
    let after = click + (0.7 * f64::from(RATE)) as usize;
    assert_eq!(limited.samples[after..], x[after..]);
    let lowered = (limited.fraction * x.len() as f64).round() as usize;
    assert!(
        (before..after).len() >= lowered,
        "{lowered} samples lowered"
    );
}

#[test]
fn gives_the_same_for_any_chunks() {
    // A tone with clicks in it, quiet and loud, and a draw of chunk lengths.
    let mut x = tone(-6.0, 2.0);
    for (i, click) in [(1000, 2.0), (30000, -1.2), (30010, 3.0), (60000, 0.95)] {
        x[i] = click;
    }
    let whole = limit(&x, RATE, CEILING);
    let mut state = 7u64;
    for _ in 0..5 {
        let mut limiter = Limiter::new(RATE, CEILING);
        let mut out = Vec::new();
        let mut rest = &x[..];
        while !rest.is_empty() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = ((state >> 33) as usize % 3000).min(rest.len());
            let before = out.len();
            limiter.process(&rest[..len], &mut out);
            assert!(out.len() - before <= len);
            rest = &rest[len..];
        }
        assert_eq!(x.len() - out.len(), limiter.latency());
        limiter.finish(&mut out);
        assert!(out == whole.samples, "chunked");
        assert_eq!(
            (limiter.reduction(), limiter.fraction()),
            (whole.reduction, whole.fraction)
        );
    }
}
