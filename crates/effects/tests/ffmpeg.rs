//! The effects against what ffmpeg makes of the game's filters, which fixtures/effects.py saved:
//! a loose guard on how they sound, not a check of the samples.

use std::sync::Arc;

use effects::{
    Pipeline, Processor,
    convolve::{Impulses, Response},
    run, schema,
};
use rustfft::{FftPlanner, num_complex::Complex};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

/// The most each case's log-mel spectrum may differ from ffmpeg's, in dB: 1.3 times what it
/// measured and 0.25 dB, rounded up to a tenth. Each is under what the unfiltered input measures,
/// which the test checks, so an effect that did nothing would fail it. A vibrato moves the
/// harmonics less than a band is wide, so its case is left to [`PITCH_THRESHOLDS`].
const THRESHOLDS: [(&str, Option<f32>); 20] = [
    ("synthetic", Some(1.7)),
    ("tram", Some(1.6)),
    ("fishing", Some(3.5)),
    ("gas_mask", Some(0.3)),
    ("megaphone", Some(2.8)),
    ("megaphone_emagged", Some(2.8)),
    ("hailer", Some(4.4)),
    ("lizard_tongue", Some(1.6)),
    ("alien_tongue", Some(2.7)),
    ("ethereal_tongue", Some(0.3)),
    ("snail_tongue", Some(1.3)),
    ("stutter", Some(2.4)),
    ("cult_slurring", Some(2.5)),
    ("poly", Some(2.8)),
    ("fish_tongue", Some(5.3)),
    ("compressor", Some(0.5)),
    ("flanger", Some(0.7)),
    ("vibrato", None),
    ("wobble", Some(1.2)),
    ("silicon", Some(1.3)),
];

/// The most the pitch of the cases with a vibrato may differ from ffmpeg's, in cents, where
/// log-mel bands are too coarse to tell a vibrato from none: 1.3 times what it measured and 5
/// cents, rounded up, each under what the unfiltered input measures.
const PITCH_THRESHOLDS: [(&str, f32); 2] = [("vibrato", 14.0), ("cult_slurring", 31.0)];

// The log-mel spectrum, as fixtures/effects.py computes it.
const RATE: f64 = 48000.0;
const FRAME: usize = 1024;
const HOP: usize = 480;
const BANDS: usize = 40;
const LOW: f64 = 60.0;
const HIGH: f64 = 16000.0;
const FLOOR: f64 = 1e-10;
/// How far below the reference's loudest the comparison clips both spectra, in dB: quieter than
/// that is silence to it.
const RANGE: f32 = 70.0;
/// The frames the comparison reads: those whose loudest band is within this of the reference's
/// loudest, in dB.
const ACTIVE: f32 = 50.0;
/// How far either way the comparison slides one spectrum along the other, in frames: 100 ms.
const LAG: isize = 10;
// The pitch tracker: frames of 40 ms, every HOP, and the normalized autocorrelation over which a
// frame is voiced.
const PITCH_FRAME: usize = 1920;
const VOICED: f64 = 0.6;
const NEAR: f64 = 0.9;

fn threshold(case: &str) -> Option<f32> {
    THRESHOLDS
        .iter()
        .find(|(name, _)| *name == case)
        .unwrap_or_else(|| panic!("no threshold for {case}"))
        .1
}

#[test]
fn pitch_moves_as_ffmpegs_does() {
    let fixture = Fixture::open();
    let input = fixture.input();
    let impulses = fixture.impulses();
    let mut failed = Vec::new();
    for (name, threshold) in PITCH_THRESHOLDS {
        let case = fixture.cases.iter().find(|c| c.name == name).expect(name);
        let reference = fixture.reference_pitch(name);
        let mut pipeline = case.pipeline(&impulses);
        let ours = pitch_distance(&pitch(&run(&mut pipeline, &input)), &reference);
        let unfiltered = pitch_distance(&pitch(&input), &reference);
        eprintln!(
            "{name}: {ours:.1} cents from ffmpeg, the unfiltered input {unfiltered:.1}, threshold \
             {threshold}"
        );
        if ours > threshold || unfiltered <= threshold {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The least correlation between the fish tongue's filter and ffmpeg's, at the lag that brings
/// them nearest: 0.997 was measured, and the unfiltered input, a still cutoff and one swinging at
/// the 9.3 Hz the game meant come to 0.23, 0.93 and 0.83.
const WOBBLE_CORRELATION: f64 = 0.97;
/// How far ffmpeg's afftfilt may lag, in samples: it gives its output three quarters of a frame
/// late.
const WOBBLE_LAG: usize = 2048;

#[test]
fn wobble_follows_ffmpegs_samples() {
    let fixture = Fixture::open();
    let input = fixture.input();
    let (_, reference) = fixture.tensor("wobble/samples");
    let case = fixture.cases.iter().find(|c| c.name == "wobble").unwrap();
    let ours = run(&mut case.pipeline(&fixture.impulses()), &input);
    let correlation = |y: &[f32]| {
        (0..=WOBBLE_LAG)
            .map(|lag| {
                let (a, b) = (&y[..reference.len() - lag], &reference[lag..]);
                let dot = |a: &[f32], b: &[f32]| -> f64 {
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum()
                };
                dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
            })
            .fold(f64::MIN, f64::max)
    };
    let (ours, unfiltered) = (correlation(&ours), correlation(&input));
    eprintln!("wobble: correlation {ours:.4}, the unfiltered input {unfiltered:.4}");
    assert!(ours >= WOBBLE_CORRELATION && unfiltered < WOBBLE_CORRELATION);
}

#[test]
fn every_graph_sounds_like_ffmpegs() {
    let fixture = Fixture::open();
    let input = fixture.input();
    let impulses = fixture.impulses();
    let mut failed = Vec::new();
    for case in &fixture.cases {
        let Some(threshold) = threshold(&case.name) else {
            continue;
        };
        let reference = fixture.reference(&case.name);
        let mut pipeline = case.pipeline(&impulses);
        let ours = distance(&log_mel(&run(&mut pipeline, &input)), &reference);
        let unfiltered = distance(&log_mel(&input), &reference);
        eprintln!(
            "{}: {ours:.2} dB from ffmpeg, the unfiltered input {unfiltered:.2}, threshold {threshold}",
            case.name
        );
        if ours > threshold || unfiltered <= threshold {
            failed.push(&case.name);
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// Each graph gives the same samples however its input is cut into chunks.
#[test]
fn chunks_change_nothing() {
    let fixture = Fixture::open();
    let input = fixture.input();
    let impulses = fixture.impulses();
    for case in &fixture.cases {
        let whole = run(&mut case.pipeline(&impulses), &input);
        let mut pipeline = case.pipeline(&impulses);
        let mut chunked = Vec::new();
        let mut rest = &input[..];
        let mut size = 1;
        while !rest.is_empty() {
            let (chunk, after) = rest.split_at(size.min(rest.len()));
            pipeline.process(chunk, &mut chunked);
            rest = after;
            size = size * 7 % 4093 + 1;
        }
        pipeline.finish(&mut chunked);
        assert!(whole == chunked, "{}", case.name);
    }
}

/// A game graph fixtures/effects.py ran through ffmpeg, by the effects the game sends in its
/// place.
#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    effects: serde_json::Value,
}

impl Case {
    fn pipeline(&self, impulses: &Impulses) -> Pipeline {
        Pipeline::new(&schema::parse(&self.effects).unwrap(), impulses).unwrap()
    }
}

/// The fixture's tensors and cases.
struct Fixture {
    bytes: Vec<u8>,
    cases: Vec<Case>,
    synth_rate: u32,
}

impl Fixture {
    fn open() -> Self {
        let fixture = testkit::fixture!("effects");
        let cases = serde_json::from_str(&fixture.metadata()["cases"]).expect("cases");
        let synth_rate = fixture.metadata()["synth_rate"].parse().expect("a rate");
        Self {
            bytes: std::fs::read(fixture.path()).expect("testkit read it"),
            cases,
            synth_rate,
        }
    }

    /// The tensor `name`, widened to f32, and its shape.
    #[track_caller]
    fn tensor(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let tensors = SafeTensors::deserialize(&self.bytes).expect("testkit checked it");
        let view = tensors
            .tensor(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let data = view.data();
        let values = match view.dtype() {
            Dtype::F32 => data
                .as_chunks()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::F16 => data
                .as_chunks()
                .0
                .iter()
                .map(|b| f16(u16::from_le_bytes(*b)))
                .collect(),
            dtype => panic!("{name}: {dtype:?}"),
        };
        (view.shape().to_vec(), values)
    }

    /// The made-up speech the graphs ran on.
    fn input(&self) -> Vec<f32> {
        self.tensor("input").1
    }

    /// What ffmpeg made of it for `case`, as frames of bands.
    fn reference(&self, case: &str) -> Vec<Vec<f32>> {
        let (shape, values) = self.tensor(&format!("{case}/mel"));
        values.chunks(shape[1]).map(<[f32]>::to_vec).collect()
    }

    /// The pitch of what ffmpeg made of it for `case`, by frame.
    fn reference_pitch(&self, case: &str) -> Vec<Option<f32>> {
        let (_, values) = self.tensor(&format!("{case}/pitch"));
        values.iter().map(|&f| (f > 0.0).then_some(f)).collect()
    }

    /// The made-up impulses ffmpeg convolved with.
    fn impulses(&self) -> Impulses {
        let (_, synth) = self.tensor("synth");
        Impulses {
            synth: Arc::new(Response::new(&synth, 2, self.synth_rate)),
        }
    }
}

/// An IEEE half as f32.
fn f16(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// HTK triangles, peaking at 1, over the bins of a FRAME-point transform.
fn mel_filters() -> Vec<Vec<f64>> {
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (low, high) = (mel(LOW), mel(HIGH));
    let edges: Vec<f64> = (0..BANDS + 2)
        .map(|i| hz(low + (high - low) * i as f64 / (BANDS + 1) as f64))
        .collect();
    (0..BANDS)
        .map(|b| {
            let (low, mid, high) = (edges[b], edges[b + 1], edges[b + 2]);
            (0..=FRAME / 2)
                .map(|k| {
                    let f = k as f64 * RATE / FRAME as f64;
                    ((f - low) / (mid - low))
                        .min((high - f) / (high - mid))
                        .max(0.0)
                })
                .collect()
        })
        .collect()
}

/// The log-mel spectrum of `y`, frames by bands, in dB, as fixtures/effects.py computes it.
fn log_mel(y: &[f32]) -> Vec<Vec<f32>> {
    let filters = mel_filters();
    let fft = FftPlanner::new().plan_fft_forward(FRAME);
    let window: Vec<f64> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
        .collect();
    let frames = 1 + y.len().saturating_sub(FRAME) / HOP;
    (0..frames)
        .map(|t| {
            let mut buffer: Vec<Complex<f64>> = (0..FRAME)
                .map(|i| {
                    let x = y.get(t * HOP + i).copied().unwrap_or(0.0);
                    Complex::new(window[i] * f64::from(x), 0.0)
                })
                .collect();
            fft.process(&mut buffer);
            filters
                .iter()
                .map(|filter| {
                    let power: f64 = filter
                        .iter()
                        .zip(&buffer)
                        .map(|(w, c)| w * c.norm_sqr())
                        .sum();
                    (10.0 * power.max(FLOOR).log10()) as f32
                })
                .collect()
        })
        .collect()
}

/// How far apart two log-mel spectra are: the mean difference in dB over the bands of the
/// reference's audible frames, both clipped to the reference's range, at the lag that brings
/// them nearest. A frame `ours` lacks counts as silence.
fn distance(ours: &[Vec<f32>], reference: &[Vec<f32>]) -> f32 {
    let top = reference.iter().flatten().copied().fold(f32::MIN, f32::max);
    let floor = top - RANGE;
    let active: Vec<usize> = (0..reference.len())
        .filter(|&t| reference[t].iter().copied().fold(f32::MIN, f32::max) > top - ACTIVE)
        .collect();
    let silent = vec![floor; BANDS];
    (-LAG..=LAG)
        .map(|lag| {
            let sum: f32 = active
                .iter()
                .map(|&t| {
                    let ours = usize::try_from(t as isize + lag)
                        .ok()
                        .and_then(|u| ours.get(u))
                        .unwrap_or(&silent);
                    ours.iter()
                        .zip(&reference[t])
                        .map(|(&a, &b)| (a.max(floor) - b.max(floor)).abs())
                        .sum::<f32>()
                })
                .sum();
            sum / (active.len() * BANDS) as f32
        })
        .fold(f32::MAX, f32::min)
}

/// The pitch of each frame of `y`, in Hz, where it is voiced, as fixtures/effects.py tracks
/// it: the shortest lag between 50 and 500 Hz where the frame's normalized autocorrelation peaks
/// within [`NEAR`] of its highest, refined by a parabola, where that highest is over [`VOICED`].
fn pitch(y: &[f32]) -> Vec<Option<f32>> {
    let size = 2 * PITCH_FRAME;
    let mut planner = FftPlanner::new();
    let (fft, ifft) = (
        planner.plan_fft_forward(size),
        planner.plan_fft_inverse(size),
    );
    let frames = 1 + y.len().saturating_sub(PITCH_FRAME) / HOP;
    let (shortest, longest) = ((RATE / 500.0) as usize, (RATE / 50.0) as usize);
    (0..frames)
        .map(|t| {
            let mut buffer: Vec<Complex<f64>> = (0..size)
                .map(|i| {
                    let x = (i < PITCH_FRAME)
                        .then(|| y.get(t * HOP + i).copied())
                        .flatten()
                        .unwrap_or(0.0);
                    Complex::new(f64::from(x), 0.0)
                })
                .collect();
            fft.process(&mut buffer);
            for c in &mut buffer {
                *c = Complex::new(c.norm_sqr(), 0.0);
            }
            ifft.process(&mut buffer);
            let r: Vec<f64> = buffer.iter().map(|c| c.re).collect();
            if r[0] <= 0.0 {
                return None;
            }
            let normalized =
                |lag: usize| r[lag] / r[0] * PITCH_FRAME as f64 / (PITCH_FRAME - lag) as f64;
            let best = (shortest..=longest)
                .map(normalized)
                .fold(f64::MIN, f64::max);
            if best < VOICED {
                return None;
            }
            // The shortest lag near the best, whose multiples would also be near it.
            let mut lag = (shortest..=longest).find(|&lag| normalized(lag) >= NEAR * best)?;
            while lag < longest && normalized(lag + 1) > normalized(lag) {
                lag += 1;
            }
            if lag == shortest || lag == longest {
                return None;
            }
            let (a, b, c) = (normalized(lag - 1), normalized(lag), normalized(lag + 1));
            let at = lag as f64 + 0.5 * (a - c) / (a - 2.0 * b + c);
            Some((RATE / at) as f32)
        })
        .collect()
}

/// How far apart two pitch tracks are: the median difference in cents over the frames voiced in
/// both, which a tracker's octave slips do not move, at the lag that brings them nearest.
fn pitch_distance(ours: &[Option<f32>], reference: &[Option<f32>]) -> f32 {
    (-LAG..=LAG)
        .filter_map(|lag| {
            let mut cents: Vec<f32> = (0..reference.len())
                .filter_map(|t| {
                    let ours = usize::try_from(t as isize + lag)
                        .ok()
                        .and_then(|u| ours.get(u))?;
                    Some(1200.0 * ((*ours)? / reference[t]?).log2().abs())
                })
                .collect();
            cents.sort_by(f32::total_cmp);
            cents.get(cents.len() / 2).copied()
        })
        .fold(f32::MAX, f32::min)
}
