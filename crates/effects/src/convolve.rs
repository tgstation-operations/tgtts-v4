//! Convolution with an impulse response, by uniformly partitioned overlap-save: the impulse cut
//! into blocks, each block's spectrum multiplied with the spectrum of the input as many blocks
//! back, so that a long impulse costs a block's FFT and a sum of products per block.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{Processor, schema::Impulse};

/// The block, in samples: the convolution's latency.
const BLOCK: usize = 1024;
const SIZE: usize = 2 * BLOCK;
const BINS: usize = BLOCK + 1;

/// An impulse response, ready to convolve with.
#[derive(Clone)]
pub struct Response {
    /// The spectrum of each block of the impulse, the block in the first half of the transform.
    blocks: Vec<Vec<Complex<f64>>>,
    len: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Response {
    /// The impulse response `samples`, interleaved in `channels` at `rate` Hz: their mean, at
    /// [`dsp::RATE`], scaled to unit energy.
    ///
    /// # Panics
    ///
    /// If `channels` is 0 or the impulse is silent.
    pub fn new(samples: &[f32], channels: usize, rate: u32) -> Self {
        assert!(channels > 0, "an impulse of no channels");
        let mono: Vec<f64> = samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().map(|&s| f64::from(s)).sum::<f64>() / channels as f64)
            .collect();
        let mut impulse = dsp::resample::resample(&mono, f64::from(rate), f64::from(dsp::RATE));
        let energy = impulse.iter().map(|s| s * s).sum::<f64>().sqrt();
        assert!(energy > 0.0, "a silent impulse");
        impulse.iter_mut().for_each(|s| *s /= energy);
        Self::from_impulse(&impulse)
    }

    fn from_impulse(impulse: &[f64]) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(SIZE);
        let inverse = planner.plan_fft_inverse(SIZE);
        let blocks = impulse
            .chunks(BLOCK)
            .map(|block| {
                let mut buffer = vec![Complex::default(); SIZE];
                for (b, &s) in buffer.iter_mut().zip(block) {
                    b.re = s;
                }
                forward.process(&mut buffer);
                buffer.truncate(BINS);
                buffer
            })
            .collect();
        Self {
            blocks,
            len: impulse.len(),
            forward,
            inverse,
        }
    }

    /// Its length, in samples.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has no samples, which [`Response::new`] never gives.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The impulse responses the service has, by [`Impulse`].
#[derive(Clone)]
pub struct Impulses {
    /// SynthImpulse.wav's.
    pub synth: Arc<Response>,
}

impl Impulses {
    /// The response `impulse` names.
    pub fn get(&self, impulse: Impulse) -> &Arc<Response> {
        match impulse {
            Impulse::Synth => &self.synth,
        }
    }
}

/// The signal convolved with a [`Response`], its output as long as its input and the response
/// less one sample.
pub struct Convolve {
    response: Arc<Response>,
    /// The spectra of the input's last blocks, newest first.
    history: Vec<Vec<Complex<f64>>>,
    newest: usize,
    /// The block before the one being gathered, and that one.
    previous: Vec<f64>,
    gathering: Vec<f64>,
    /// The output still owed after the stream: the input's length and the response's tail.
    stream: usize,
    given: usize,
    buffer: Vec<Complex<f64>>,
    sum: Vec<Complex<f64>>,
}

impl Convolve {
    /// Convolving with `response`.
    pub fn new(response: Arc<Response>) -> Self {
        let blocks = response.blocks.len();
        Self {
            response,
            history: vec![vec![Complex::default(); BINS]; blocks],
            newest: 0,
            previous: vec![0.0; BLOCK],
            gathering: Vec::with_capacity(BLOCK),
            stream: 0,
            given: 0,
            buffer: vec![Complex::default(); SIZE],
            sum: vec![Complex::default(); BINS],
        }
    }

    /// The output of the block just gathered.
    fn block(&mut self, output: &mut Vec<f32>, limit: usize) {
        for (b, &s) in self
            .buffer
            .iter_mut()
            .zip(self.previous.iter().chain(&self.gathering))
        {
            *b = Complex::new(s, 0.0);
        }
        self.response.forward.process(&mut self.buffer);
        let count = self.history.len();
        self.newest = (self.newest + count - 1) % count;
        self.history[self.newest].copy_from_slice(&self.buffer[..BINS]);
        self.sum.fill(Complex::default());
        for (j, block) in self.response.blocks.iter().enumerate() {
            let input = &self.history[(self.newest + j) % count];
            for ((s, &x), &h) in self.sum.iter_mut().zip(input).zip(block) {
                *s += x * h;
            }
        }
        self.buffer[..BINS].copy_from_slice(&self.sum);
        for k in 1..BLOCK {
            self.buffer[SIZE - k] = self.sum[k].conj();
        }
        self.response.inverse.process(&mut self.buffer);
        // Overlap-save: the second half is the circular convolution's uncorrupted part.
        let take = BLOCK.min(limit - self.given);
        output.extend(
            self.buffer[BLOCK..BLOCK + take]
                .iter()
                .map(|c| (c.re / SIZE as f64) as f32),
        );
        self.given += take;
        std::mem::swap(&mut self.previous, &mut self.gathering);
        self.gathering.clear();
    }
}

impl Processor for Convolve {
    fn latency(&self) -> usize {
        BLOCK - 1
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.stream += input.len();
        for &x in input {
            self.gathering.push(f64::from(x));
            if self.gathering.len() == BLOCK {
                self.block(output, usize::MAX);
            }
        }
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let total = self.stream + self.response.len - 1;
        while self.given < total {
            self.gathering.resize(BLOCK, 0.0);
            self.block(output, total);
        }
    }
}
