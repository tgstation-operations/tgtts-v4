//! A true-peak limiter: audio held under a ceiling by a gain that comes down ahead of each peak
//! that would cross it and goes back up after, the peaks found between the samples as well as on
//! them. It works on a stream, a chunk at a time, behind it by a fixed latency.

use std::{collections::VecDeque, f64::consts::PI};

use crate::loudness::INTERPOLATOR;

/// How long before a peak the gain starts to come down, in seconds: long enough that the gain's
/// fall spreads over several periods of a voice, which keeps it from distorting the waveform, and
/// short enough that the dip before a plosive is masked by it.
const LOOKAHEAD: f64 = 0.005;
/// The time constant of the gain's return after a peak, in seconds: some periods of a low voice,
/// so that the gain does not follow the waveform, and shorter than a syllable, so that one loud
/// syllable does not hold down the next.
const RELEASE: f64 = 0.08;
/// How far under the ceiling the gain aims, in full scale: more than rounding the output to 16
/// bits can move a peak by.
const MARGIN: f64 = 8.0 / 32768.0;
/// Half the length of the interpolator that finds the waveform between the samples, in samples:
/// farther than BS.1770-4's of 12 taps a phase reaches, so that a gain held over it covers both.
const HALF: usize = 16;
/// The points it finds between each two samples.
const BETWEEN: usize = 7;
/// A gain within this of unity, in dB, is unity: the release reaches it, where on its own it
/// would only come nearer, by a step of a hundredth of a percent.
const UNITY: f64 = 1e-3;

/// `samples`, at `rate` Hz and full scale at 1, through a [`Limiter`] whole.
pub fn limit(samples: &[f64], rate: u32, ceiling: f64) -> Limited {
    let mut limiter = Limiter::new(rate, ceiling);
    let mut out = Vec::with_capacity(samples.len());
    limiter.process(samples, &mut out);
    limiter.finish(&mut out);
    Limited {
        samples: out,
        reduction: limiter.reduction(),
        fraction: limiter.fraction(),
    }
}

/// What [`limit`] made of samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Limited {
    /// The samples.
    pub samples: Vec<f64>,
    /// The most the gain came down, in dB: 0 where nothing was limited.
    pub reduction: f64,
    /// The fraction of the samples whose gain came down.
    pub fraction: f64,
}

/// A limiter holding the true peak of a stream of audio at or under a ceiling, as BS.1770-4's
/// meter reads it and as an interpolation eight times over, with the parabola through each turn,
/// reads the waveform between the samples.
///
/// Each peak over the ceiling sets the gain it needs over the samples it comes from and a
/// lookahead before them; the gain comes back up after it with the release, and is smoothed by
/// two moving averages that together span the lookahead, so that it is never above what a peak in
/// reach needs, and is exactly unity wherever no peak is, which leaves audio under the ceiling as
/// it is. Its output is the same for any chunking of its input.
#[derive(Clone, Debug)]
pub struct Limiter {
    target: f64,
    fall: f64,
    reach: usize,
    latency: usize,
    taps: Vec<[f64; 2 * HALF]>,
    /// The input from the oldest sample still needed, the latency before the newest.
    input: Vec<f64>,
    /// The index of the input's first sample in the stream, those before the stream's first
    /// being zeros.
    start: i64,
    /// The index of the next sample in, past the stream's end once it is finished.
    pushed: i64,
    /// The stream's samples in.
    taken: i64,
    /// The last point of the waveform before the next sample to be read.
    before: f64,
    /// The gain each position needs, in dB, kept while it holds the gain down: (position, gain).
    needed: VecDeque<(i64, f64)>,
    level: f64,
    averages: [Average; 2],
    most: f64,
    lowered: u64,
    given: u64,
}

impl Limiter {
    /// A limiter for audio at `rate` Hz, full scale at 1, holding its true peak at or under
    /// `ceiling` dBTP.
    pub fn new(rate: u32, ceiling: f64) -> Self {
        let lookahead = (LOOKAHEAD * f64::from(rate)).round() as usize;
        let smoothing = lookahead / 2;
        // Held over the interpolators' reach each side of a peak too, the gain is flat over all
        // the samples the peak comes from, so it scales the peak as it scales them.
        let reach = lookahead + HALF;
        let latency = reach + HALF;
        let taps = (1..=BETWEEN)
            .map(|k| {
                let at = k as f64 / (BETWEEN + 1) as f64;
                std::array::from_fn(|j| {
                    let t = at + HALF as f64 - 1.0 - j as f64;
                    sinc(t) * kaiser(t / HALF as f64)
                })
            })
            .collect();
        Self {
            target: 10f64.powf(ceiling / 20.0) - MARGIN,
            fall: (-1.0 / (RELEASE * f64::from(rate))).exp(),
            reach,
            latency,
            taps,
            input: vec![0.0; latency],
            start: -(latency as i64),
            pushed: 0,
            taken: 0,
            before: 0.0,
            needed: VecDeque::new(),
            level: 0.0,
            averages: [Average::new(smoothing), Average::new(smoothing)],
            most: 0.0,
            lowered: 0,
            given: 0,
        }
    }

    /// How many samples its output lags its input by.
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Takes `input` and appends to `output` the samples it has finished, which end the latency
    /// before the last it has taken.
    pub fn process(&mut self, input: &[f64], output: &mut Vec<f64>) {
        for &x in input {
            self.step(x, true, output);
        }
        // Keeps what the next samples need of the input, from the latency before the newest.
        let spent = self.input.len() - self.latency;
        self.input.drain(..spent);
        self.start += spent as i64;
    }

    /// Appends to `output` the samples it still holds, as the stream ends in silence.
    pub fn finish(&mut self, output: &mut Vec<f64>) {
        for _ in 0..self.latency {
            self.step(0.0, false, output);
        }
    }

    /// The most its gain has come down so far, in dB.
    pub fn reduction(&self) -> f64 {
        self.most
    }

    /// The fraction of the samples it has given whose gain came down.
    pub fn fraction(&self) -> f64 {
        self.lowered as f64 / self.given.max(1) as f64
    }

    /// Takes the next sample, `real` where it is of the stream, and gives the one the latency
    /// before it, where that one is of the stream.
    fn step(&mut self, x: f64, real: bool, output: &mut Vec<f64>) {
        let newest = self.pushed;
        self.pushed += 1;
        self.taken += i64::from(real);
        self.input.push(x);
        let start = self.start;
        let at = |i: i64| (i - start) as usize;
        // The position whose waveform the newest sample completes.
        let position = newest - HALF as i64;
        let peak = self.peak(at(position));
        let needed = if peak > self.target {
            20.0 * (self.target / peak).log10()
        } else {
            0.0
        };
        while self.needed.back().is_some_and(|&(_, n)| n >= needed) {
            self.needed.pop_back();
        }
        self.needed.push_back((position, needed));
        // The gain of the sample the latency back is held to the least needed from the
        // interpolators' reach before it to the reach and the lookahead after it.
        let sample = position - self.reach as i64;
        while self
            .needed
            .front()
            .is_some_and(|&(p, _)| p < sample - HALF as i64)
        {
            self.needed.pop_front();
        }
        let held = self.needed[0].1;
        self.level *= self.fall;
        if self.level > -UNITY {
            self.level = 0.0;
        }
        self.level = self.level.min(held);
        let [first, second] = &mut self.averages;
        let gain = second.next(first.next(self.level));
        if (0..self.taken).contains(&sample) {
            output.push(self.input[at(sample)] * 10f64.powf(gain / 20.0));
            self.given += 1;
            self.most = self.most.max(-gain);
            self.lowered += u64::from(gain < 0.0);
        }
    }

    /// The largest magnitude of the input from the sample at `i` to the next: the sample's own,
    /// those of the points BS.1770-4's interpolation puts between them, and that of the waveform,
    /// from a longer interpolation's points and the parabola through each turn among them.
    fn peak(&mut self, i: usize) -> f64 {
        let x = &self.input;
        // The output of BS.1770-4's filter that falls between the sample and the next, 5.875
        // samples after the input it ends with.
        let annex = &x[i + 6 - 11..=i + 6];
        let mut peak = (INTERPOLATOR.iter())
            .map(|phase| {
                let y: f64 = phase
                    .iter()
                    .zip(annex.iter().rev())
                    .map(|(h, s)| h * s)
                    .sum();
                y.abs()
            })
            .fold(x[i].abs(), f64::max);
        let window = &x[i + 1 - HALF..=i + HALF];
        let mut points = [0.0; BETWEEN + 3];
        points[0] = self.before;
        points[1] = x[i];
        for (point, phase) in points[2..].iter_mut().zip(&self.taps) {
            *point = phase.iter().zip(window).map(|(h, s)| h * s).sum();
        }
        points[BETWEEN + 2] = x[i + 1];
        self.before = points[BETWEEN + 1];
        for three in points.windows(3) {
            peak = peak.max(turn(three[0], three[1], three[2]));
        }
        peak
    }
}

/// The magnitude of the parabola through three evenly spaced points at its turn, where the
/// middle one is a turn of the three, and the middle one's own where it is not.
fn turn(a: f64, b: f64, c: f64) -> f64 {
    let curvature = a - 2.0 * b + c;
    if (b - a) * (c - b) <= 0.0 && curvature != 0.0 {
        (b - (a - c).powi(2) / (8.0 * curvature)).abs()
    } else {
        b.abs()
    }
}

/// A moving average over a window of samples, those before the first taken as 0, and exactly 0
/// where they all are.
#[derive(Clone, Debug)]
struct Average {
    window: VecDeque<f64>,
    sum: f64,
    nonzero: usize,
}

impl Average {
    fn new(len: usize) -> Self {
        Self {
            window: VecDeque::from(vec![0.0; len]),
            sum: 0.0,
            nonzero: 0,
        }
    }

    fn next(&mut self, x: f64) -> f64 {
        let old = self.window.pop_front().expect("the window is never empty");
        self.window.push_back(x);
        self.sum += x - old;
        self.nonzero = self.nonzero + usize::from(x != 0.0) - usize::from(old != 0.0);
        if self.nonzero == 0 {
            self.sum = 0.0;
        }
        self.sum / self.window.len() as f64
    }
}

fn sinc(t: f64) -> f64 {
    if t == 0.0 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    }
}

/// The Kaiser window at `x` in [-1, 1], at a beta that keeps the interpolator's error far under
/// the differences it is to find.
fn kaiser(x: f64) -> f64 {
    const BETA: f64 = 8.0;
    bessel_i0(BETA * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(BETA)
}

/// The modified Bessel function of the first kind and order 0, by its series.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..50 {
        term *= (x / (2.0 * f64::from(k))).powi(2);
        sum += term;
    }
    sum
}
