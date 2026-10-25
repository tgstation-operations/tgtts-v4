use burn::{
    module::Module,
    nn::{Tanh, conv::Conv1dConfig},
    tensor::{DType, Device, FloatDType, Int, Tensor},
};

use crate::{
    AudioVaeConfig, Error, Weights,
    layers::{
        CausalConvTranspose1d, Layer, ResidualUnit, SampleRateCondition, Snake1d, causal,
        sequential,
    },
};

/// The decoder half of the audio VAE: latent frames to audio at the config's
/// `out_sample_rate`, `decoder_rates.product()` samples per frame.
///
/// It is causal: each sample depends on the frames up to its own and none after.
#[derive(Module, Debug)]
pub struct Decoder {
    model: Vec<Layer>,
    /// The condition applied before each layer of `model`, upstream's `sr_cond_model`.
    sr_cond_model: Vec<Option<SampleRateCondition>>,
    bucket: usize,
    #[module(skip)]
    dtype: DType,
}

impl Decoder {
    /// The decoder `config` describes, with the weights under `decoder.` in `weights`, held and
    /// computed in `dtype`. Weight-normed convolutions are folded into plain ones as they load.
    ///
    /// # Errors
    ///
    /// If the config asks for something not implemented, or the weights do not fit it: a
    /// missing or unexpected tensor, a shape that differs, or bucket boundaries other than the
    /// config's.
    pub fn load(
        config: &AudioVaeConfig,
        weights: &Weights,
        dtype: FloatDType,
        device: &Device,
    ) -> Result<Self, Error> {
        let boundaries = config.check_decoder()?;
        let stored = weights.ints("decoder.sr_bin_boundaries")?;
        if !stored
            .iter()
            .copied()
            .eq(boundaries.iter().copied().map(i64::from))
        {
            return Err(Error::Weights(format!(
                "decoder.sr_bin_boundaries: {stored:?}, but the config has {boundaries:?}"
            )));
        }
        let decoder = Self::new(config, boundaries, dtype, device);
        weights.load("decoder", decoder, dtype, &["sr_bin_boundaries"])
    }

    fn new(
        config: &AudioVaeConfig,
        boundaries: &[u32],
        dtype: FloatDType,
        device: &Device,
    ) -> Self {
        let (latent, channels) = (config.latent_dim, config.decoder_dim);
        let depthwise = Conv1dConfig::new(latent, latent, 7).with_groups(latent);
        let mut model = vec![
            Layer::Conv(causal(depthwise, 3, 0).init(device)),
            Layer::Conv(Conv1dConfig::new(latent, channels, 1).init(device)),
        ];
        let mut sr_cond_model = vec![None, None];
        let buckets = boundaries.len() + 1;
        for (i, &stride) in config.decoder_rates.iter().enumerate() {
            let (input_dim, output_dim) = (channels >> i, channels >> (i + 1));
            let block = DecoderBlock::new(input_dim, output_dim, stride, output_dim, device);
            model.push(Layer::DecoderBlock(block));
            sr_cond_model.push(Some(SampleRateCondition::new(input_dim, buckets, device)));
        }
        let output_dim = channels >> config.decoder_rates.len();
        model.extend([
            Layer::Snake(Snake1d::new(output_dim, device)),
            Layer::Conv(causal(Conv1dConfig::new(output_dim, 1, 7), 3, 0).init(device)),
            Layer::Tanh(Tanh),
        ]);
        sr_cond_model.resize_with(model.len(), || None);
        Self {
            model,
            sr_cond_model,
            bucket: bucket(boundaries, config.out_sample_rate),
            dtype: dtype.into(),
        }
    }

    /// Decodes latents `[batch, latent_dim, frames]` into audio `[batch, 1, samples]`, in the
    /// dtype the weights were loaded in, which the latents are cast to.
    pub fn decode(&self, latents: Tensor<3>) -> Tensor<3> {
        let [batch, ..] = latents.dims();
        let buckets = Tensor::full([batch], self.bucket as i64, &latents.device());
        self.forward(latents.cast(self.dtype), buckets)
    }

    fn forward(&self, x: Tensor<3>, buckets: Tensor<1, Int>) -> Tensor<3> {
        let layers = self.model.iter().zip(&self.sr_cond_model);
        layers.fold(x, |x, (layer, condition)| {
            let x = match condition {
                Some(condition) => condition.forward(x, buckets.clone()),
                None => x,
            };
            layer.forward(x)
        })
    }
}

/// The bucket of `sample_rate` among ascending `boundaries`: how many lie below it, as
/// upstream's `torch.bucketize` counts them.
fn bucket(boundaries: &[u32], sample_rate: u32) -> usize {
    boundaries.partition_point(|&boundary| boundary < sample_rate)
}

/// Snake, a causal transposed convolution upsampling by `stride`, and three residual units of
/// dilation 1, 3 and 9.
#[derive(Module, Debug)]
pub(crate) struct DecoderBlock {
    block: Vec<Layer>,
}

impl DecoderBlock {
    fn new(
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
        device: &Device,
    ) -> Self {
        let upsample = CausalConvTranspose1d::new(
            [input_dim, output_dim],
            stride,
            stride.div_ceil(2),
            stride % 2,
            device,
        );
        let mut block = vec![
            Layer::Snake(Snake1d::new(input_dim, device)),
            Layer::ConvTranspose(upsample),
        ];
        block.extend([1, 3, 9].map(|dilation| {
            Layer::Residual(ResidualUnit::new(output_dim, dilation, groups, device))
        }));
        Self { block }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        sequential(&self.block, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_counts_boundaries_below_the_rate() {
        let boundaries = [20000, 30000, 40000];
        let buckets = [16000, 20000, 20001, 30000, 44100, 48000]
            .map(|sample_rate| bucket(&boundaries, sample_rate));
        assert_eq!(buckets, [0, 0, 1, 1, 3, 3]);
    }
}
