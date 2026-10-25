//! Filters: the RBJ cookbook's biquads.

use std::f64::consts::TAU;

use crate::{Processor, RATE};

/// A biquad's shape, as Robert Bristow-Johnson's "Cookbook formulae for audio EQ biquad filter
/// coefficients" names them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Second-order lowpass.
    Lowpass,
    /// Second-order highpass.
    Highpass,
    /// Peaking equalizer, with its gain in dB.
    Peak(f64),
    /// Low shelf, with its gain in dB.
    Lowshelf(f64),
    /// High shelf, with its gain in dB.
    Highshelf(f64),
}

/// A biquad, in transposed direct form II, kept in f64: at a few tens of Hz an f32 state rounds
/// the low frequencies into noise.
#[derive(Clone, Debug)]
pub struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    state: [f64; 2],
}

impl Biquad {
    /// The cookbook's `shape` at `frequency` Hz with its `q`.
    pub fn new(shape: Shape, frequency: f64, q: f64) -> Self {
        let w0 = TAU * frequency / RATE;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let amplitude = |gain: f64| 10f64.powf(gain / 40.0);
        let (b, a) = match shape {
            Shape::Lowpass => (
                [(1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0],
                [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
            ),
            Shape::Highpass => (
                [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0],
                [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
            ),
            Shape::Peak(gain) => {
                let a = amplitude(gain);
                (
                    [1.0 + alpha * a, -2.0 * cos, 1.0 - alpha * a],
                    [1.0 + alpha / a, -2.0 * cos, 1.0 - alpha / a],
                )
            }
            Shape::Lowshelf(gain) => {
                let a = amplitude(gain);
                let root = 2.0 * a.sqrt() * alpha;
                (
                    [
                        a * ((a + 1.0) - (a - 1.0) * cos + root),
                        2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
                        a * ((a + 1.0) - (a - 1.0) * cos - root),
                    ],
                    [
                        (a + 1.0) + (a - 1.0) * cos + root,
                        -2.0 * ((a - 1.0) + (a + 1.0) * cos),
                        (a + 1.0) + (a - 1.0) * cos - root,
                    ],
                )
            }
            Shape::Highshelf(gain) => {
                let a = amplitude(gain);
                let root = 2.0 * a.sqrt() * alpha;
                (
                    [
                        a * ((a + 1.0) + (a - 1.0) * cos + root),
                        -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                        a * ((a + 1.0) + (a - 1.0) * cos - root),
                    ],
                    [
                        (a + 1.0) - (a - 1.0) * cos + root,
                        2.0 * ((a - 1.0) - (a + 1.0) * cos),
                        (a + 1.0) - (a - 1.0) * cos - root,
                    ],
                )
            }
        };
        Self {
            b: b.map(|b| b / a[0]),
            a: [a[1] / a[0], a[2] / a[0]],
            state: [0.0; 2],
        }
    }

    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.state[0];
        self.state[0] = self.b[1] * x - self.a[0] * y + self.state[1];
        self.state[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

impl Processor for Biquad {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.extend(input.iter().map(|&x| self.tick(f64::from(x)) as f32));
    }
}
