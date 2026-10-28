use serde::Deserialize;

/// The `audio_vae_config` of a VoxCPM2 `config.json`.
///
/// Fields the JSON leaves out take the defaults of upstream's `AudioVAEConfig`, and a field
/// upstream does not have is refused rather than ignored. Building the `audiovae` crate's
/// `Encoder` or `Decoder` refuses the values it does not implement.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AudioVaeConfig {
    /// Channels of the encoder's first convolution, doubled by each of its blocks.
    pub encoder_dim: usize,
    /// The encoder's downsampling strides; their product is the input samples per latent frame.
    pub encoder_rates: Vec<usize>,
    /// Channels of a latent frame.
    pub latent_dim: usize,
    /// Channels of the decoder's first block, halved by each block.
    pub decoder_dim: usize,
    /// The decoder's upsampling strides; their product is the output samples per latent frame.
    pub decoder_rates: Vec<usize>,
    /// Whether the residual units' dilated convolutions are depthwise. Only `true` is
    /// implemented.
    pub depthwise: bool,
    /// The sample rate the encoder takes.
    pub sample_rate: u32,
    /// The sample rate the decoder writes, which chooses its sample-rate bucket.
    pub out_sample_rate: u32,
    /// Whether the decoder adds noise after each upsampling. Only `false` is implemented.
    pub use_noise_block: bool,
    /// Ascending sample rates that split output rates into buckets, each with its own scale
    /// and bias in every decoder block. Only a list is implemented, not `null`.
    pub sr_bin_boundaries: Option<Vec<u32>>,
    /// How a bucket conditions a decoder block. Only `scale_bias` and `scale_bias_init` are
    /// implemented; they differ only in how upstream initializes them for training.
    pub cond_type: String,
    /// Width of the bucket embedding for the `concat` condition type, unused by the others.
    pub cond_dim: usize,
    /// Whether a condition is followed by its own Snake and convolution. Only `false` is
    /// implemented.
    pub cond_out_layer: bool,
}

impl Default for AudioVaeConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 128,
            encoder_rates: vec![2, 5, 8, 8],
            latent_dim: 64,
            decoder_dim: 2048,
            decoder_rates: vec![8, 6, 5, 2, 2, 2],
            depthwise: true,
            sample_rate: 16000,
            out_sample_rate: 48000,
            use_noise_block: false,
            sr_bin_boundaries: Some(vec![20000, 30000, 40000]),
            cond_type: "scale_bias".into(),
            cond_dim: 128,
            cond_out_layer: false,
        }
    }
}
