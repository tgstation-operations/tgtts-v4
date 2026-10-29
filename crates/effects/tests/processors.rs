//! Each effect on signals whose answer is known.

use std::{f64::consts::TAU, sync::Arc};

use dsp::quantity::{self, Level, SampleRate, TruePeak};
use effects::{
    Pipeline, Processor,
    convolve::{Impulses, Response},
    delay::{Echo, Flanger, Vibrato},
    dynamics::{Compressor, Limiter},
    filter::{Biquad, Shape},
    level::{Gain, Noise, Tremolo},
    run,
    schema::{self, Graph, Method, Waveform},
    shape::{Clip, Crush, Saturate},
    spectral::Wobble,
    time::{self, Pitch, Speed},
};

const RATE: f64 = 48000.0;

fn graph(json: &str) -> Vec<Graph> {
    schema::parse(&serde_json::from_str(json).unwrap()).unwrap()
}

/// A made-up impulse: a fifth of a second of decaying noise at 44.1 kHz, in stereo.
fn impulses() -> Impulses {
    let mut state = 1u32;
    let samples: Vec<f32> = (0..8820 * 2)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
            noise * (-(i / 2) as f32 / 2000.0).exp()
        })
        .collect();
    Impulses {
        synth: Arc::new(Response::new(&samples, 2, SampleRate::new(44100))),
    }
}

fn sine(frequency: f64, amplitude: f64, seconds: f64) -> Vec<f32> {
    (0..(seconds * RATE) as usize)
        .map(|i| (amplitude * (TAU * frequency * i as f64 / RATE).sin()) as f32)
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// The gain, in dB, `processor` gives a sine at `frequency` once it has settled.
fn gain_at(processor: &mut dyn Processor, frequency: f64) -> f64 {
    let x = sine(frequency, 0.1, 1.0);
    let y = run(processor, &x);
    db(rms(&y[24000..48000]) / rms(&x[24000..48000]))
}

/// The frequency of the strongest component of `x`, by its spectrum's peak, refined by a
/// parabola through the bins either side.
fn frequency(x: &[f32]) -> f64 {
    let n = x.len().next_power_of_two();
    let mut buffer: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
        .map(|i| {
            let w = 0.5 - 0.5 * (TAU * i as f64 / x.len() as f64).cos();
            let s = x.get(i).map_or(0.0, |&s| f64::from(s) * w);
            rustfft::num_complex::Complex::new(s, 0.0)
        })
        .collect();
    rustfft::FftPlanner::new()
        .plan_fft_forward(n)
        .process(&mut buffer);
    let power: Vec<f64> = buffer[..n / 2].iter().map(|c| c.norm().ln()).collect();
    let k = (1..n / 2 - 1)
        .max_by(|&a, &b| power[a].total_cmp(&power[b]))
        .unwrap();
    let (a, b, c) = (power[k - 1], power[k], power[k + 1]);
    (k as f64 + 0.5 * (a - c) / (a - 2.0 * b + c)) * RATE / n as f64
}

#[track_caller]
fn near(value: f64, expected: f64, tolerance: f64) {
    assert!(
        (value - expected).abs() <= tolerance,
        "{value} is not within {tolerance} of {expected}"
    );
}

#[test]
fn biquads_follow_the_cookbook() {
    let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
    near(
        gain_at(&mut Biquad::new(Shape::Lowpass, 1000.0, sqrt_half), 1000.0),
        -3.01,
        0.05,
    );
    near(
        gain_at(&mut Biquad::new(Shape::Lowpass, 1000.0, sqrt_half), 100.0),
        0.0,
        0.05,
    );
    assert!(gain_at(&mut Biquad::new(Shape::Lowpass, 1000.0, sqrt_half), 4000.0) < -24.0);
    near(
        gain_at(&mut Biquad::new(Shape::Highpass, 1000.0, sqrt_half), 1000.0),
        -3.01,
        0.05,
    );
    near(
        gain_at(
            &mut Biquad::new(Shape::Peak(quantity::Gain::new(12.0)), 2000.0, 1.0),
            2000.0,
        ),
        12.0,
        0.05,
    );
    near(
        gain_at(
            &mut Biquad::new(Shape::Lowshelf(quantity::Gain::new(-10.0)), 500.0, 0.5),
            30.0,
        ),
        -10.0,
        0.1,
    );
    near(
        gain_at(
            &mut Biquad::new(Shape::Highshelf(quantity::Gain::new(6.0)), 2000.0, 0.7),
            18000.0,
        ),
        6.0,
        0.2,
    );
    near(
        gain_at(
            &mut Biquad::new(Shape::Peak(quantity::Gain::new(-10.0)), 33.0, 0.44),
            33.0,
        ),
        -10.0,
        0.1,
    );
}

#[test]
fn wobble_lifts_the_lows_and_boosts_the_highs() {
    // With no swing, ffmpeg's curve: 1.8 times well under the cutoff, nothing at 1.27 times
    // it and 15 times at four times.
    let still = |frequency| gain_at(&mut Wobble::new(1500.0, 0.0, 37.5), frequency);
    near(still(100.0), db(1.79), 0.1);
    assert!(still(1500.0 * 1.27) < -20.0);
    near(still(6000.0), db(14.8), 0.3);
    // A tone between the cutoff's ends comes and goes as the cutoff swings past it.
    let y = run(
        &mut Wobble::new(1500.0, 750.0, 37.5),
        &sine(1900.0, 0.1, 0.2),
    );
    let levels: Vec<f64> = (0..32)
        .map(|i| db(rms(&y[4800 + i * 64..4800 + i * 64 + 64]) / 0.1))
        .collect();
    let (low, high) = levels
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(high - low > 15.0, "{levels:?}");
}

#[test]
fn levels() {
    near(
        gain_at(&mut Gain::new(quantity::Gain::new(-6.0)), 440.0),
        -6.0,
        1e-4,
    );
    let y = run(&mut Tremolo::new(10.0, 0.8), &vec![1.0; 48000]);
    near(f64::from(y[0]), 1.0, 1e-6);
    near(f64::from(y[2400]), 0.2, 1e-4);
    let y = run(&mut Noise::new(Level::new(-40.0)), &vec![0.0; 48000]);
    near(db(rms(&y)), -40.0, 0.1);
    assert!(y.iter().all(|s| s.abs() < 0.0174));
    // acontrast's curve, folding back over full scale.
    let shaped = run(&mut Saturate::new(0.3), &[0.1, 0.5, 1.0, 1.5, 2.0]);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    for (y, expected) in shaped.iter().zip([0.18, half, 1.0, half, 0.0]) {
        near(f64::from(*y), expected, 2e-3);
    }
    assert_eq!(
        run(&mut Clip::new(Level::new(-6.0)), &[0.2, 0.9, -1.5]),
        [0.2, 0.5011872, -0.5011872]
    );
}

#[test]
fn crush_holds_each_sample_at_its_rate() {
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    assert_eq!(
        run(&mut Crush::new(24000.0, 1.0), &x),
        [0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0]
    );
    assert_eq!(
        run(&mut Crush::new(16000.0, 0.5), &x),
        [0.0, 0.5, 1.0, 3.0, 3.5, 4.0, 6.0, 6.5]
    );
}

#[test]
fn compressor_follows_its_curve() {
    // On a constant, the level it detects is the constant's.
    let gain = |level: f64, knee: f64| {
        let x = vec![10f64.powf(level / 20.0) as f32; 24000];
        let mut compressor = Compressor::new(
            Level::new(-20.0),
            4.0,
            5.0,
            50.0,
            quantity::Gain::new(3.0),
            quantity::Gain::new(knee),
        );
        let y = run(&mut compressor, &x);
        db(f64::from(y[23999] / x[23999]))
    };
    near(gain(-30.0, 0.0), 3.0, 1e-3);
    near(gain(-6.0, 0.0), 3.0 - 0.75 * 14.0, 1e-3);
    // At the threshold, a knee of 10 dB takes off (1/4 - 1) * 5^2 / 20.
    near(gain(-20.0, 10.0), 3.0 - 0.9375, 1e-3);
    near(gain(-26.0, 10.0), 3.0, 1e-3);
}

#[test]
fn limiter_holds_the_ceiling_and_keeps_the_length() {
    let x = sine(440.0, 2.0, 0.5);
    let y = run(&mut Limiter::new(TruePeak::new(-6.0)), &x);
    assert_eq!(y.len(), x.len());
    let peak = y.iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(peak <= 10f32.powf(-6.0 / 20.0), "{peak}");
}

#[test]
fn echo_adds_its_taps() {
    let mut impulse = vec![0.0; 10];
    impulse[0] = 1.0;
    let mut echo = Echo::new([(0.0, 0.5), (1000.0 / RATE * 3.0, 0.25)]);
    let y = run(&mut echo, &impulse);
    assert_eq!(y.len(), 13);
    assert_eq!(&y[..5], [0.5, 0.0, 0.0, 0.25, 0.0]);
}

#[test]
fn flanger_feeds_back_its_output() {
    let mut impulse = vec![0.0; 4];
    impulse[0] = 1.0;
    // The sweep starts at its shortest, a sample.
    let y = run(&mut Flanger::new(3.0, 0.5, 0.5, Waveform::Sine), &impulse);
    for (y, expected) in y.iter().zip([1.0, 0.5, 0.25, 0.125]) {
        near(f64::from(*y), expected, 1e-5);
    }
}

#[test]
fn vibrato_swings_the_pitch_and_keeps_the_length() {
    let x = sine(1000.0, 0.5, 1.0);
    let y = run(&mut Vibrato::new(5.0, 1.0), &x);
    assert_eq!(y.len(), x.len());
    // The pitch is highest a quarter of a swing in, and lowest three quarters in.
    let high = frequency(&y[2400 - 600..2400 + 600]);
    let low = frequency(&y[7200 - 600..7200 + 600]);
    near(12.0 * (high / 1000.0).log2(), 1.0, 0.1);
    near(12.0 * (low / 1000.0).log2(), -1.0, 0.1);
}

#[test]
fn tempo_keeps_the_pitch() {
    // The vocoder comes down 1.7 dB, as rubberband does, once past a quarter octave.
    for (method, level) in [(Method::Vocoder, -1.7), (Method::Splice, 0.0)] {
        for ratio in [0.5, 0.8, 1.2, 2.0] {
            let x = sine(220.0, 0.5, 1.0);
            let y = run(&mut *time::tempo(ratio, method), &x);
            assert_eq!(y.len(), (48000.0 / ratio).round() as usize, "{ratio}");
            near(frequency(&y[4800..y.len() - 4800]), 220.0, 0.5);
            near(db(rms(&y[4800..y.len() - 4800]) / rms(&x)), level, 0.3);
        }
    }
}

#[test]
fn speed_moves_pitch_and_tempo_together() {
    for ratio in [0.5, 0.99, 1.5, 4.0] {
        let x = sine(1000.0, 0.5, 1.0);
        let y = run(&mut Speed::new(ratio), &x);
        assert_eq!(y.len(), (48000.0 / ratio).round() as usize, "{ratio}");
        near(frequency(&y[200..y.len() - 200]), 1000.0 * ratio, 0.5);
        near(db(rms(&y[200..y.len() - 200]) / rms(&x)), 0.0, 0.1);
    }
    // Above the new Nyquist frequency, folded back, nearly nothing.
    let y = run(&mut Speed::new(2.0), &sine(15000.0, 0.5, 1.0));
    assert!(db(rms(&y[200..y.len() - 200]) / 0.5) < -60.0);
}

#[test]
fn pitch_keeps_the_length() {
    let cases = [(7.0, false), (-12.0, false), (12.0, true), (-5.0, true)];
    for ((semitones, keep_formants), method) in cases
        .into_iter()
        .zip([Method::Vocoder, Method::Splice].into_iter().cycle())
    {
        let x = sine(220.0, 0.5, 1.0);
        let y = run(&mut Pitch::new(semitones, keep_formants, method), &x);
        assert_eq!(y.len(), x.len(), "{semitones}");
        let shifted = 220.0 * 2f64.powf(semitones / 12.0);
        near(frequency(&y[4800..43200]), shifted, 1.0);
    }
}

#[test]
fn convolution_is_convolution() {
    let impulses = impulses();
    let response = &impulses.synth;
    let mut unit = vec![0.0; 5000];
    unit[0] = 1.0;
    let graph = graph(r#"[[{"effect": "convolve", "impulse": "synth"}]]"#);
    let mut pipeline = Pipeline::new(&graph, &impulses).unwrap();
    let h = run(&mut pipeline, &unit);
    assert_eq!(h.len(), unit.len() + response.len() - 1);
    let energy: f64 = h.iter().map(|&s| f64::from(s).powi(2)).sum();
    near(energy, 1.0, 1e-4);

    let x: Vec<f32> = (0..3000)
        .map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let y = run(&mut Pipeline::new(&graph, &impulses).unwrap(), &x);
    for n in [0, 1, 1023, 1024, 2500, 9000] {
        let expected: f64 = (0..=n.min(x.len() - 1))
            .filter(|&k| n - k < response.len())
            .map(|k| f64::from(x[k]) * f64::from(h[n - k]))
            .sum();
        near(f64::from(y[n]), expected, 1e-4);
    }
}

#[test]
fn branches_line_up_whatever_each_holds_back() {
    // The limiter leaves audio under its ceiling as it is, a few hundred samples late; mixed
    // against the input it delays, the two cancel.
    let graph = graph(
        r#"[{"nodes": {"late": {"effect": "limiter", "ceiling_db": 0},
              "mix": {"effect": "mix", "weights": {"late": 1, "input": -1}}},
             "edges": [["input", "late"], ["input", "mix"], ["late", "mix"], ["mix", "output"]]}]"#,
    );
    let mut pipeline = Pipeline::new(&graph, &impulses()).unwrap();
    assert!(pipeline.latency() > 100);
    let x = sine(440.0, 0.5, 0.5);
    let y = run(&mut pipeline, &x);
    assert_eq!(y.len(), x.len());
    assert!(y.iter().all(|s| s.abs() < 1e-6));
}
