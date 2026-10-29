use std::num::NonZeroUsize;

/// What a request asks for beyond its text and voice. The default is what the Python service
/// asked for, decoded a second at a time.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub sampling: Sampling,
    /// The most patches to generate, nanovllm-voxcpm's `max_generate_length`.
    pub max_patches: usize,
    /// The most latent frames the audio VAE decodes in one call, so that no call holds a GPU
    /// long enough for its driver to reset it. The audio is the same, to rounding, however the
    /// frames are split.
    pub decode_chunk_frames: NonZeroUsize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sampling: Sampling::default(),
            max_patches: 256,
            decode_chunk_frames: NonZeroUsize::new(25).expect("not zero"),
        }
    }
}

/// How a patch is sampled. The default is what the Python service asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    /// Euler steps from noise to the patch, upstream's `inference_timesteps`.
    pub timesteps: usize,
    /// The scale of the noise the flow starts from.
    pub temperature: f32,
    /// The classifier-free guidance scale.
    pub cfg_value: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            timesteps: 10,
            temperature: 1.0,
            cfg_value: 2.0,
        }
    }
}
