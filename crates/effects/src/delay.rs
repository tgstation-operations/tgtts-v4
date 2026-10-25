//! Effects made of delay lines: a multi-tap echo, a vibrato and a flanger.

use std::f64::consts::{PI, TAU};

use crate::{Processor, RATE, schema::Waveform};

/// The last samples of a stream, read back at a delay.
#[derive(Clone, Debug)]
struct Line {
    samples: Vec<f64>,
    mask: usize,
    /// Where the next sample goes.
    at: usize,
}

impl Line {
    /// A line that reads back at least `longest` samples.
    fn new(longest: usize) -> Self {
        let size = (longest + 4).next_power_of_two();
        Self {
            samples: vec![0.0; size],
            mask: size - 1,
            at: 0,
        }
    }

    fn push(&mut self, x: f64) {
        self.samples[self.at] = x;
        self.at = (self.at + 1) & self.mask;
    }

    /// The sample `delay` before the next one, 1 being the last pushed.
    fn get(&self, delay: usize) -> f64 {
        self.samples[self.at.wrapping_sub(delay) & self.mask]
    }

    /// The stream `delay` samples before the next one, between the samples by a cubic Hermite
    /// spline, `delay` being at least 2.
    fn read(&self, delay: f64) -> f64 {
        let whole = delay.floor();
        let t = delay - whole;
        let d = whole as usize;
        // Going back in time: y1 is at `d`, y2 one sample older, y0 one newer.
        let (y0, y1, y2, y3) = (
            self.get(d - 1),
            self.get(d),
            self.get(d + 1),
            self.get(d + 2),
        );
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
        ((c3 * t + c2) * t + c1) * t + y1
    }
}

/// The sum of delayed copies of the signal, its output as long as its input and the longest
/// delay.
#[derive(Clone, Debug)]
pub struct Echo {
    taps: Vec<(usize, f64)>,
    line: Line,
    longest: usize,
}

impl Echo {
    /// Copies at `(delay in ms, weight)`.
    pub fn new(taps: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let taps: Vec<_> = taps
            .into_iter()
            .map(|(ms, weight)| ((ms / 1000.0 * RATE).round() as usize, weight))
            .collect();
        let longest = taps.iter().map(|&(d, _)| d).max().unwrap_or(0);
        Self {
            taps,
            line: Line::new(longest + 1),
            longest,
        }
    }
}

impl Processor for Echo {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            self.line.push(f64::from(x));
            let y: f64 = self
                .taps
                .iter()
                .map(|&(d, w)| w * self.line.get(d + 1))
                .sum();
            y as f32
        }));
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        self.process(&vec![0.0; self.longest], output);
    }
}

/// A vibrato: the signal read back at a delay that swings sinusoidally, the pitch going up
/// first. Its output is put back by the delay's middle, so that it starts with its input.
#[derive(Clone, Debug)]
pub struct Vibrato {
    swing: f64,
    step: f64,
    phase: f64,
    line: Line,
    /// The delay's least, in samples, which the interpolation needs.
    least: f64,
    /// The samples of output still to drop: the delay's middle.
    skip: usize,
    lag: usize,
}

impl Vibrato {
    /// Swinging `rate` times a second by up to `depth` semitones either way.
    pub fn new(rate: f64, depth: f64) -> Self {
        let step = TAU * rate / RATE;
        // The delay d(n) = swing * (1 + cos) changes the rate the signal is read at by -d'(n),
        // at most swing * step.
        let swing = (2f64.powf(depth / 12.0) - 1.0) / step;
        let least = 2.0;
        let lag = (least + swing).round() as usize;
        Self {
            swing,
            step,
            phase: 0.0,
            line: Line::new((least + 2.0 * swing).ceil() as usize + 2),
            least,
            skip: lag,
            lag,
        }
    }
}

impl Processor for Vibrato {
    fn latency(&self) -> usize {
        self.lag
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        for &x in input {
            self.line.push(f64::from(x));
            let delay = self.least + self.swing * (1.0 + self.phase.cos());
            self.phase = (self.phase + self.step) % TAU;
            let y = self.line.read(delay);
            if self.skip > 0 {
                self.skip -= 1;
            } else {
                output.push(y as f32);
            }
        }
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let held = self.lag - self.skip;
        self.process(&vec![0.0; held], output);
    }
}

/// A flanger as SoX's "phaser" is one: `y = x + feedback * y` delayed, the delay sweeping from a
/// sample up to its longest and back.
#[derive(Clone, Debug)]
pub struct Flanger {
    longest: f64,
    feedback: f64,
    waveform: Waveform,
    step: f64,
    phase: f64,
    line: Line,
}

impl Flanger {
    /// Sweeping up to `delay` ms and back `rate` times a second along `waveform`, feeding back
    /// `feedback`.
    pub fn new(delay: f64, rate: f64, feedback: f64, waveform: Waveform) -> Self {
        let longest = (delay / 1000.0 * RATE).max(1.0);
        Self {
            longest,
            feedback,
            waveform,
            step: TAU * rate / RATE,
            phase: 0.0,
            line: Line::new(longest.ceil() as usize + 2),
        }
    }

    /// Where the sweep is, from 0 at its shortest to 1 at its longest, starting at 0.
    fn sweep(&self) -> f64 {
        match self.waveform {
            Waveform::Sine => (1.0 - self.phase.cos()) / 2.0,
            Waveform::Triangle => 1.0 - (self.phase - PI).abs() / PI,
        }
    }
}

impl Processor for Flanger {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            let delay = 1.0 + (self.longest - 1.0) * self.sweep();
            self.phase = (self.phase + self.step) % TAU;
            // Between the samples linearly: the newest a sample back, where the spline would
            // need the output being made.
            let whole = delay.floor();
            let t = delay - whole;
            let d = whole as usize;
            let delayed = (1.0 - t) * self.line.get(d) + t * self.line.get(d + 1);
            let y = f64::from(x) + self.feedback * delayed;
            self.line.push(y);
            y as f32
        }));
    }
}
