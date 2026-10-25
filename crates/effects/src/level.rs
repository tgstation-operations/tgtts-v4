//! Effects on the level alone: a gain, a tremolo and added noise.

use std::f64::consts::TAU;

use crate::{Processor, RATE};

/// A fixed gain.
#[derive(Clone, Debug)]
pub struct Gain(f32);

impl Gain {
    /// A gain of `db`.
    pub fn new(db: f64) -> Self {
        Self(10f64.powf(db / 20.0) as f32)
    }
}

impl Processor for Gain {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| x * self.0));
    }
}

/// A gain swinging sinusoidally between `1 - depth` and 1, starting at 1.
#[derive(Clone, Debug)]
pub struct Tremolo {
    depth: f64,
    step: f64,
    phase: f64,
}

impl Tremolo {
    /// Swinging `rate` times a second down by `depth`.
    pub fn new(rate: f64, depth: f64) -> Self {
        Self {
            depth,
            step: TAU * rate / RATE,
            phase: 0.0,
        }
    }
}

impl Processor for Tremolo {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| {
            let gain = 1.0 - self.depth * (1.0 - self.phase.cos()) / 2.0;
            self.phase = (self.phase + self.step) % TAU;
            (f64::from(x) * gain) as f32
        }));
    }
}

/// White noise added to the signal, uniform, from a generator seeded the same each time, so that
/// the same request comes out the same.
#[derive(Clone, Debug)]
pub struct Noise {
    amplitude: f64,
    state: u64,
}

impl Noise {
    /// Noise at `level` dBFS RMS.
    pub fn new(level: f64) -> Self {
        Self {
            // Uniform noise over [-a, a] has an RMS of a / sqrt(3).
            amplitude: 3f64.sqrt() * 10f64.powf(level / 20.0),
            state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// A draw from [-1, 1), by xorshift64*.
    fn draw(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        bits as f64 / (1u64 << 52) as f64 - 1.0
    }
}

impl Processor for Noise {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(
            input
                .iter()
                .map(|&x| (f64::from(x) + self.amplitude * self.draw()) as f32),
        );
    }
}
