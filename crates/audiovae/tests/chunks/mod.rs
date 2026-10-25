use audiovae::Decoder;
use burn::tensor::{Device, Tensor, TensorData};

/// A seeded generator, for draws a rerun repeats.
pub struct Draws(u64);

impl Draws {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Lengths of 1 to `longest` frames, the last cut short, that add up to `frames`.
    pub fn lengths(&mut self, frames: usize, longest: usize) -> Vec<usize> {
        let mut lengths = vec![];
        let mut left = frames;
        while left > 0 {
            let length = (1 + self.next() as usize % longest).min(left);
            lengths.push(length);
            left -= length;
        }
        lengths
    }

    /// Latents of `shape`, uniform in (-2, 2).
    pub fn latents(&mut self, shape: [usize; 3], device: &Device) -> Tensor<3> {
        let values = (0..shape.iter().product())
            .map(|_| self.next() as f32 / (1u64 << 31) as f32 * 4.0 - 2.0)
            .collect();
        Tensor::from_data(TensorData::new(values, shape), device)
    }
}

/// `latents` `[batch, latent_dim, frames]` pushed through a stream in chunks of `lengths`
/// frames, which add up to `frames`, and the chunks' audio joined.
pub fn streamed(decoder: &Decoder, latents: Tensor<3>, lengths: &[usize]) -> Tensor<3> {
    let mut stream = decoder.stream();
    let mut start = 0;
    let audio = lengths.iter().map(|&length| {
        let chunk = latents.clone().narrow(2, start, length);
        start += length;
        stream.push(chunk)
    });
    Tensor::cat(audio.collect(), 2)
}
