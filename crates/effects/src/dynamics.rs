//! Dynamics: a compressor, and [`dsp::limiter`]'s true-peak limiter.

use dsp::limiter;

use crate::{Processor, RATE};

/// A feedforward compressor on the RMS level, with a soft knee, as Giannoulis, Massberg and
/// Reiss's "Digital Dynamic Range Compressor Design" (JAES, 2012) lays one out: a level detector
/// that rises with the attack and falls with the release, then a static curve in dB.
#[derive(Clone, Debug)]
pub struct Compressor {
    threshold: f64,
    slope: f64,
    knee: f64,
    makeup: f64,
    attack: f64,
    release: f64,
    power: f64,
}

impl Compressor {
    /// Compression by `ratio` above `threshold` dBFS, over a knee of `knee` dB, following the
    /// level within `attack` and `release` ms, with `makeup` dB after.
    pub fn new(
        threshold: f64,
        ratio: f64,
        attack: f64,
        release: f64,
        makeup: f64,
        knee: f64,
    ) -> Self {
        let coefficient = |ms: f64| (-1.0 / (ms / 1000.0 * RATE)).exp();
        Self {
            threshold,
            slope: 1.0 / ratio - 1.0,
            knee,
            makeup,
            attack: coefficient(attack),
            release: coefficient(release),
            power: 0.0,
        }
    }

    /// The gain in dB for a level of `level` dBFS.
    fn gain(&self, level: f64) -> f64 {
        let over = level - self.threshold;
        let reduction = if 2.0 * over <= -self.knee {
            0.0
        } else if 2.0 * over < self.knee {
            self.slope * (over + self.knee / 2.0).powi(2) / (2.0 * self.knee)
        } else {
            self.slope * over
        };
        reduction + self.makeup
    }
}

impl Processor for Compressor {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            let x = f64::from(x);
            let coefficient = if x * x > self.power {
                self.attack
            } else {
                self.release
            };
            self.power = coefficient * self.power + (1.0 - coefficient) * x * x;
            let level = 10.0 * self.power.max(1e-20).log10();
            (x * 10f64.powf(self.gain(level) / 20.0)) as f32
        }));
    }
}

/// [`dsp::limiter::Limiter`] on f32.
#[derive(Clone, Debug)]
pub struct Limiter {
    limiter: limiter::Limiter,
    buffer: Vec<f64>,
}

impl Limiter {
    /// A limiter holding the true peak at or under `ceiling` dBTP.
    pub fn new(ceiling: f64) -> Self {
        Self {
            limiter: limiter::Limiter::new(dsp::RATE, ceiling),
            buffer: Vec::new(),
        }
    }

    fn give(&mut self, output: &mut Vec<f32>) {
        output.extend(self.buffer.drain(..).map(|y| y as f32));
    }
}

impl Processor for Limiter {
    fn latency(&self) -> usize {
        self.limiter.latency()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let input: Vec<f64> = input.iter().map(|&x| f64::from(x)).collect();
        self.limiter.process(&input, &mut self.buffer);
        self.give(output);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        self.limiter.finish(&mut self.buffer);
        self.give(output);
    }
}
