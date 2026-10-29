//! Distortion: a waveshaper, clipping and a sample-rate crusher.

use std::f64::consts::{PI, TAU};

use crate::{Processor, RATE};

/// A sine waveshaper, as ffmpeg's acontrast: `sin(pi/2 x + amount * 2/15 * sin(2 pi x))`, which
/// leaves full scale at full scale, brings quiet sounds up by 4 to 8 dB, more with more of
/// `amount`, and folds whatever is over full scale back down.
#[derive(Clone, Debug)]
pub struct Saturate(f64);

impl Saturate {
    /// Shaping by `amount`, from 0 to 1.
    pub fn new(amount: f64) -> Self {
        Self(amount * 2.0 / 15.0)
    }
}

impl Processor for Saturate {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            let x = f64::from(x);
            (PI / 2.0 * x + self.0 * (TAU * x).sin()).sin() as f32
        }));
    }
}

/// Hard clipping at a ceiling.
#[derive(Clone, Debug)]
pub struct Clip(f32);

impl Clip {
    /// Clipping at `ceiling` dBFS.
    pub fn new(ceiling: f64) -> Self {
        Self(10f64.powf(ceiling / 20.0) as f32)
    }
}

impl Processor for Clip {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| x.clamp(-self.0, self.0)));
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
