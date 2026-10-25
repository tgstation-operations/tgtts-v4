use std::collections::VecDeque;

use burn::tensor::Tensor;
use voxcpm::Noise;

/// Upstream's noise, patch by patch.
pub struct Recorded(VecDeque<Vec<f32>>);

impl Recorded {
    /// The noise of each patch, `[patches, feat_dim, patch_size]`.
    pub fn new(noise: Tensor<3>) -> Self {
        let patches = noise.dims()[0];
        let patch = |i| noise.clone().narrow(0, i, 1).into_data().iter().collect();
        Self((0..patches).map(patch).collect())
    }
}

impl Noise for Recorded {
    fn next(&mut self, len: usize) -> Vec<f32> {
        let noise = self.0.pop_front().expect("upstream drew no more noise");
        assert_eq!(noise.len(), len);
        noise
    }
}
