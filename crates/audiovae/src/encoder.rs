use burn::{
    module::Module,
    nn::conv::{Conv1d, Conv1dConfig},
    tensor::{DType, Device, FloatDType, Tensor, ops::PadMode},
};

use crate::{
    AudioVaeConfig, Error, Weights,
    layers::{Layer, ResidualUnit, Snake1d, causal, sequential},
};

/// The encoder half of the audio VAE: audio at the config's `sample_rate` to latent frames,
/// `encoder_rates.product()` samples per frame.
///
/// Only the mean of the latent distribution is computed, which is what upstream encodes
/// with; the log-variance head is not loaded.
#[derive(Module, Debug)]
pub struct Encoder {
    block: Vec<Layer>,
    fc_mu: Conv1d,
    hop: usize,
    #[module(skip)]
    dtype: DType,
}

impl Encoder {
    /// The encoder `config` describes, with the weights under `encoder.` in `weights`, held and
    /// computed in `dtype`. Weight-normed convolutions are folded into plain ones as they load.
    ///
    /// # Errors
    ///
    /// If the config asks for something not implemented, or the weights do not fit it: a
    /// missing or unexpected tensor, or a shape that differs.
    pub fn load(
        config: &AudioVaeConfig,
        weights: &Weights,
        dtype: FloatDType,
        device: &Device,
    ) -> Result<Self, Error> {
        config.check_encoder()?;
        let encoder = Self::new(config, dtype, device);
        weights.load("encoder", encoder, dtype, &["fc_logvar"])
    }

    fn new(config: &AudioVaeConfig, dtype: FloatDType, device: &Device) -> Self {
        let mut dim = config.encoder_dim;
        let first = causal(Conv1dConfig::new(1, dim, 7), 3, 0);
        let mut block = vec![Layer::Conv(first.init(device))];
        for &stride in &config.encoder_rates {
            dim *= 2;
            let groups = dim / 2;
            block.push(Layer::EncoderBlock(EncoderBlock::new(
                dim, stride, groups, device,
            )));
        }
        let fc_mu = causal(Conv1dConfig::new(dim, config.latent_dim, 3), 1, 0).init(device);
        Self {
            block,
            fc_mu,
            hop: config.encoder_rates.iter().product(),
            dtype: dtype.into(),
        }
    }

    /// Encodes audio `[batch, 1, samples]` into latents `[batch, latent_dim, frames]`, in the
    /// dtype the weights were loaded in, which the audio is cast to.
    ///
    /// The audio is padded with silence at the end to whole frames, as upstream pads it.
    pub fn encode(&self, audio: Tensor<3>) -> Tensor<3> {
        let samples = audio.dims()[2];
        let padding = samples.next_multiple_of(self.hop) - samples;
        let audio = audio
            .cast(self.dtype)
            .pad([(0, padding)], PadMode::Constant(0.0));
        self.fc_mu.forward(sequential(&self.block, audio))
    }
}

/// Three residual units of dilation 1, 3 and 9, Snake, and a causal convolution that
/// downsamples by `stride` and doubles the channels.
#[derive(Module, Debug)]
pub(crate) struct EncoderBlock {
    block: Vec<Layer>,
}

impl EncoderBlock {
    fn new(output_dim: usize, stride: usize, groups: usize, device: &Device) -> Self {
        let input_dim = output_dim / 2;
        let downsample = Conv1dConfig::new(input_dim, output_dim, 2 * stride).with_stride(stride);
        let mut block = [1, 3, 9]
            .map(|dilation| Layer::Residual(ResidualUnit::new(input_dim, dilation, groups, device)))
            .to_vec();
        block.extend([
            Layer::Snake(Snake1d::new(input_dim, device)),
            Layer::Conv(causal(downsample, stride.div_ceil(2), stride % 2).init(device)),
        ]);
        Self { block }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        sequential(&self.block, x)
    }
}
