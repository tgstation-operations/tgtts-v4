//! The voice effects the game asks for, run in the service: graphs of effects from
//! [`schema`], over mono audio at [`dsp::RATE`] in full scale at 1.
//!
//! Each effect is a [`Processor`] on a stream, so that the same code runs on a whole clip, as one
//! chunk, and on speech as it is made.

pub mod convolve;
pub mod delay;
pub mod dynamics;
pub mod filter;
pub mod graph;
pub mod level;
pub mod schema;
pub mod shape;
pub mod spectral;
pub mod time;

pub use graph::{Error, Pipeline};

/// The rate everything here runs at, in Hz.
const RATE: f64 = dsp::RATE.value() as f64;

/// An effect on a stream of audio, given a chunk at a time.
///
/// Its output starts where its input does: what it holds back to make it is its latency, and it
/// gives it once it has the input after. An effect that changes the duration gives the output
/// for the input it has taken as far as it can.
pub trait Processor: Send {
    /// The most samples of input it holds without giving the output they make.
    fn latency(&self) -> usize {
        0
    }

    /// Takes `input` and appends to `output` what it has finished.
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>);

    /// Appends to `output` what it still holds as the stream ends, its tail included.
    fn finish(&mut self, output: &mut Vec<f32>) {
        let _ = output;
    }
}

/// `processor` over the whole of `input`.
pub fn run(processor: &mut dyn Processor, input: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    processor.process(input, &mut output);
    processor.finish(&mut output);
    output
}

/// Processors one after another.
pub struct Series(pub Vec<Box<dyn Processor>>);

impl Processor for Series {
    fn latency(&self) -> usize {
        self.0.iter().map(|p| p.latency()).sum()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let mut chunk = input.to_vec();
        for processor in &mut self.0 {
            let mut next = Vec::with_capacity(chunk.len());
            processor.process(&chunk, &mut next);
            chunk = next;
        }
        output.extend(chunk);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let mut chunk = Vec::new();
        for processor in &mut self.0 {
            let mut next = Vec::with_capacity(chunk.len());
            processor.process(&chunk, &mut next);
            processor.finish(&mut next);
            chunk = next;
        }
        output.extend(chunk);
    }
}
