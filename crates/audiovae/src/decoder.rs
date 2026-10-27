use std::mem;

use burn::{
    module::Module,
    nn::{Tanh, conv::Conv1dConfig},
    tensor::{DType, Device, FloatDType, Int, Tensor},
};

use crate::{
    AudioVaeConfig, Error, Weights,
    layers::{
        Carry, CausalConvTranspose1d, Layer, ResidualUnit, SampleRateCondition, Snake1d, causal,
        sequential, sequential_push,
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
    latent_dim: usize,
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
            latent_dim: latent,
            dtype: dtype.into(),
        }
    }

    /// Decodes latents `[batch, latent_dim, frames]` into audio `[batch, 1, samples]`, in the
    /// dtype the weights were loaded in, which the latents are cast to.
    pub fn decode(&self, latents: Tensor<3>) -> Tensor<3> {
        let buckets = self.buckets(&latents);
        self.forward(latents.cast(self.dtype), buckets, Layer::forward)
    }

    /// A stream to decode latents through a chunk at a time, as they come.
    pub fn stream(&self) -> DecodeStream<'_> {
        DecodeStream {
            decoder: self,
            kept: vec![],
        }
    }

    fn buckets(&self, latents: &Tensor<3>) -> Tensor<1, Int> {
        let [batch, ..] = latents.dims();
        Tensor::full([batch], self.bucket as i64, &latents.device())
    }

    fn forward(
        &self,
        x: Tensor<3>,
        buckets: Tensor<1, Int>,
        mut run: impl FnMut(&Layer, Tensor<3>) -> Tensor<3>,
    ) -> Tensor<3> {
        let layers = self.model.iter().zip(&self.sr_cond_model);
        layers.fold(x, |x, (layer, condition)| {
            let x = match condition {
                Some(condition) => condition.forward(x, buckets.clone()),
                None => x,
            };
            run(layer, x)
        })
    }
}

/// Latents decoded a chunk at a time into the audio [`Decoder::decode`] gives them whole, to
/// rounding, whatever the chunks.
///
/// The convolutions are the only layers that reach back in time, so a chunk's audio depends
/// on the chunks before only through the last inputs of each, as many as its kernel reaches
/// back. The stream keeps those from one chunk to the next, in the decoder's dtype on its
/// device: for VoxCPM2's decoder, 161856 values a batch item, 632 KiB in f32. The decoder is
/// causal, so a chunk's audio is whole when it is pushed and nothing is left to flush.
#[derive(Debug)]
pub struct DecodeStream<'a> {
    decoder: &'a Decoder,
    kept: Vec<Tensor<3>>,
}

impl DecodeStream<'_> {
    /// Decodes the next latents `[batch, latent_dim, frames]`, at least a frame, into their
    /// audio `[batch, 1, samples]`, as [`Decoder::decode`] does. Every chunk must have the
    /// batch of the first.
    pub fn push(&mut self, latents: Tensor<3>) -> Tensor<3> {
        Self::push_together(&mut [self], latents)
    }

    /// Decodes the next latents of each of `streams` at once, as each [`push`](Self::push)
    /// would alone: `latents` `[streams, latent_dim, frames]`, a row each, into their audio
    /// `[streams, 1, samples]`. A stream may be at its first chunk beside others past theirs.
    ///
    /// # Panics
    ///
    /// If there is no stream, the streams are of other decoders, or `latents` has another
    /// number of rows than there are streams, where there are several.
    pub fn push_together(streams: &mut [&mut Self], latents: Tensor<3>) -> Tensor<3> {
        let decoder = streams.first().expect("a stream to decode").decoder;
        assert!(
            streams
                .iter()
                .all(|stream| std::ptr::eq(stream.decoder, decoder)),
            "streams of other decoders"
        );
        let rows = latents.dims()[0];
        assert!(
            streams.len() == 1 || rows == streams.len(),
            "{rows} rows of latents for {} streams",
            streams.len()
        );
        let buckets = decoder.buckets(&latents);
        let kept = streams.iter_mut().map(|stream| mem::take(&mut stream.kept));
        let mut carry = Carry::new(kept.collect());
        let audio = decoder.forward(latents.cast(decoder.dtype), buckets, |layer, x| {
            layer.push(x, &mut carry)
        });
        for (stream, kept) in streams.iter_mut().zip(carry.into_kept()) {
            stream.kept = kept;
        }
        audio
    }
}

/// Streams of latents decoded a chunk at a time, as [`DecodeStream`]s are, each carrying what it
/// keeps from chunk to chunk in a slot of tensors that stay where they are: what a graph of
/// [`push`](Self::push) captured once replays over, whichever slots it is given.
#[derive(Clone, Debug)]
pub struct DecodeSlots {
    decoder: Decoder,
    /// What each layer that reaches back keeps of every slot, `[slots, channels, length]`.
    kept: Vec<Tensor<3>>,
}

impl DecodeSlots {
    /// `slots` slots of streams through `decoder`, each as a stream holds them before its first
    /// chunk.
    pub fn new(decoder: &Decoder, slots: usize, device: &Device) -> Self {
        let mut stream = decoder.stream();
        stream.push(Tensor::zeros(
            [1, decoder.latent_dim, 1],
            (device, decoder.dtype),
        ));
        let fresh = |kept: &Tensor<3>| {
            let [_, channels, length] = kept.dims();
            Tensor::zeros([slots, channels, length], (device, kept.dtype()))
        };
        Self {
            decoder: decoder.clone(),
            kept: stream.kept.iter().map(fresh).collect(),
        }
    }

    /// Makes `slot` a stream before its first chunk.
    pub fn clear(&mut self, slot: usize) {
        for kept in &mut self.kept {
            let [_, channels, length] = kept.dims();
            let options = (&kept.device(), kept.dtype());
            let at = [slot..slot + 1, 0..channels, 0..length];
            let zeros = Tensor::zeros([1, channels, length], options);
            kept.inplace(|kept| kept.slice_assign(at, zeros));
        }
    }

    /// Decodes the next latents of the streams in the slots `at` `[rows]`, `[rows, latent_dim,
    /// frames]` a row each, into their audio `[rows, 1, samples]`, as each stream's
    /// [`DecodeStream::push`] would. Every operation on the device, nothing read back, so that a
    /// graph of it replays as it ran. Rows of the same slot write it in turn.
    pub fn push(&mut self, at: Tensor<1, Int>, latents: Tensor<3>) -> Tensor<3> {
        let decoder = &self.decoder;
        let buckets = decoder.buckets(&latents);
        let mut carry = Carry::Slots {
            kept: &mut self.kept,
            at: Box::new(at),
            layer: 0,
        };
        decoder.forward(latents.cast(decoder.dtype), buckets, |layer, x| {
            layer.push(x, &mut carry)
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

    pub(crate) fn push(&self, x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
        sequential_push(&self.block, x, carry)
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
