//! Distortion: soft clipping and a sample-rate crusher.

use crate::{Processor, RATE};

/// Soft clipping, `tanh(drive * x)`.
#[derive(Clone, Debug)]
pub struct Saturate(f32);

impl Saturate {
    /// Clipping driven by `db`.
    pub fn new(db: f64) -> Self {
        Self(10f64.powf(db / 20.0) as f32)
    }
}

impl Processor for Saturate {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| (self.0 * x).tanh()));
    }
}

/// A sample-rate crusher: a sample and hold, mixed with the signal as it was.
#[derive(Clone, Debug)]
pub struct Crush {
    step: f64,
    mix: f64,
    clock: f64,
    held: f64,
}

impl Crush {
    /// Held at `rate` Hz, `mix` of it in the output.
    pub fn new(rate: f64, mix: f64) -> Self {
        Self {
            step: rate / RATE,
            mix,
            // So that the first sample is held.
            clock: 1.0,
            held: 0.0,
        }
    }
}

impl Processor for Crush {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            let x = f64::from(x);
            if self.clock >= 1.0 {
                self.clock -= 1.0;
                self.held = x;
            }
            self.clock += self.step;
            ((1.0 - self.mix) * x + self.mix * self.held) as f32
        }));
    }
}
