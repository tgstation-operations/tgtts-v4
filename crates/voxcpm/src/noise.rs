use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

/// Where the standard normal noise each patch is sampled from comes from.
pub trait Noise {
    /// The next patch's noise, `len` values in the layout upstream draws it in,
    /// `[feat_dim, patch_size]`.
    fn next(&mut self, len: usize) -> Vec<f32>;
}

/// Noise drawn from a seeded generator, the same for the same seed.
#[derive(Debug)]
pub struct SeededNoise(StdRng);

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self(StdRng::seed_from_u64(seed))
    }
}

impl Noise for SeededNoise {
    fn next(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| StandardNormal.sample(&mut self.0))
            .collect()
    }
}
