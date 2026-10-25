use std::vec;

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Embedding, EmbeddingConfig, PaddingConfig1d, Tanh,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Device, Int, Tensor,
        module::{conv_transpose1d, conv1d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode},
    },
};

use crate::{decoder::DecoderBlock, encoder::EncoderBlock};

/// An element of one of upstream's `nn.Sequential`s, which are mirrored as `Vec<Layer>` so that
/// every parameter keeps its state_dict name: loading skips the variant names.
#[derive(Module, Debug)]
pub(crate) enum Layer {
    Snake(Snake1d),
    Conv(Conv1d),
    ConvTranspose(CausalConvTranspose1d),
    Residual(ResidualUnit),
    EncoderBlock(EncoderBlock),
    DecoderBlock(DecoderBlock),
    Tanh(Tanh),
}

impl Layer {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Snake(snake) => snake.forward(x),
            Self::Conv(conv) => conv.forward(x),
            Self::ConvTranspose(conv) => conv.forward(x),
            Self::Residual(unit) => unit.forward(x),
            Self::EncoderBlock(block) => block.forward(x),
            Self::DecoderBlock(block) => block.forward(x),
            Self::Tanh(tanh) => tanh.forward(x),
        }
    }

    /// As `forward`, over the next chunk of a sequence whose earlier chunks went through
    /// `carry`.
    pub(crate) fn push(&self, x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
        match self {
            Self::Conv(conv) => push_conv(conv, x, carry),
            Self::ConvTranspose(conv) => conv.push(x, carry),
            Self::Residual(unit) => unit.push(x, carry),
            Self::DecoderBlock(block) => block.push(x, carry),
            Self::EncoderBlock(_) => unreachable!("the encoder is not streamed"),
            Self::Snake(_) | Self::Tanh(_) => self.forward(x),
        }
    }
}

pub(crate) fn sequential(layers: &[Layer], x: Tensor<3>) -> Tensor<3> {
    layers.iter().fold(x, |x, layer| layer.forward(x))
}

pub(crate) fn sequential_push(layers: &[Layer], x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
    layers.iter().fold(x, |x, layer| layer.push(x, carry))
}

/// What the layers that reach back across a chunk boundary carry from one chunk to the next:
/// each one's last inputs, in the order the layers run.
#[derive(Debug)]
pub(crate) struct Carry {
    last: vec::IntoIter<Tensor<3>>,
    next: Vec<Tensor<3>>,
}

impl Carry {
    /// The carry out of the chunk before, `kept` as [`into_kept`](Self::into_kept) gave it, or
    /// nothing before the first.
    pub(crate) fn new(kept: Vec<Tensor<3>>) -> Self {
        Self {
            last: kept.into_iter(),
            next: vec![],
        }
    }

    /// What the next chunk takes over.
    pub(crate) fn into_kept(self) -> Vec<Tensor<3>> {
        self.next
    }

    /// `x` after the `length` inputs before it: those the next layer to run kept from the
    /// chunk before, or zeros in the first chunk, as the layer pads a whole sequence. The last
    /// `length` of the result are kept for that layer's next chunk.
    fn extend(&mut self, x: Tensor<3>, length: usize) -> Tensor<3> {
        let x = match self.last.next() {
            Some(kept) => Tensor::cat(vec![kept, x], 2),
            None => x.pad([(length, 0)], PadMode::Constant(0.0)),
        };
        let total = x.dims()[2];
        self.next.push(x.clone().narrow(2, total - length, length));
        x
    }
}

/// A convolution of stride 1, padded on the left as [`causal`] pads it, or not at all, over
/// the next chunk: after the inputs from the chunk before that its kernel reaches back to.
fn push_conv(conv: &Conv1d, x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
    let reach = (conv.kernel_size - 1) * conv.dilation;
    if reach == 0 {
        return conv.forward(x);
    }
    let options = ConvOptions::new([conv.stride], [0], [conv.dilation], conv.groups);
    let bias = conv.bias.as_ref().map(Param::val);
    conv1d(carry.extend(x, reach), conv.weight.val(), bias, options)
}

/// Upstream's `CausalConv1d`: a convolution padded by `2 * padding - output_padding` on the
/// left only, so that no output sees a later input.
pub(crate) fn causal(config: Conv1dConfig, padding: usize, output_padding: usize) -> Conv1dConfig {
    config.with_padding(PaddingConfig1d::Explicit(2 * padding - output_padding, 0))
}

/// Upstream's `CausalTransposeConv1d`: a transposed convolution without padding whose last
/// `2 * padding - output_padding` outputs are dropped, so that no output sees a later input.
#[derive(Module, Debug)]
pub(crate) struct CausalConvTranspose1d {
    /// `[channels_in, channels_out, kernel_size]`, as torch lays it out.
    weight: Param<Tensor<3>>,
    bias: Param<Tensor<1>>,
    stride: usize,
    trim: usize,
}

impl CausalConvTranspose1d {
    pub(crate) fn new(
        channels: [usize; 2],
        stride: usize,
        padding: usize,
        output_padding: usize,
        device: &Device,
    ) -> Self {
        let [channels_in, channels_out] = channels;
        let kernel_size = 2 * stride;
        Self {
            weight: Initializer::Zeros.init([channels_in, channels_out, kernel_size], device),
            bias: Initializer::Zeros.init([channels_out], device),
            stride,
            trim: 2 * padding - output_padding,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let options = ConvTransposeOptions::new([self.stride], [0], [0], [1], 1);
        let y = conv_transpose1d(x, self.weight.val(), Some(self.bias.val()), options);
        let length = y.dims()[2] - self.trim;
        y.narrow(2, 0, length)
    }

    /// As `forward`, over the next chunk. Its kernel is twice its stride, so that each output
    /// takes in the input step it falls in and the one before, the last of the chunk before
    /// at the boundary.
    pub(crate) fn push(&self, x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
        let steps = x.dims()[2];
        let y = self.forward(carry.extend(x, 1));
        y.narrow(2, self.stride, steps * self.stride)
    }
}

/// Snake activation, `x + sin(alpha x)^2 / alpha` with a learned `alpha` per channel.
#[derive(Module, Debug)]
pub(crate) struct Snake1d {
    /// `[1, channels, 1]`.
    alpha: Param<Tensor<3>>,
}

impl Snake1d {
    pub(crate) fn new(channels: usize, device: &Device) -> Self {
        Self {
            alpha: Initializer::Ones.init([1, channels, 1], device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let alpha = self.alpha.val();
        let sin = (alpha.clone() * x.clone()).sin();
        x + alpha.add_scalar(1e-9).recip() * sin.square()
    }
}

/// Snake, a dilated causal convolution of kernel 7, Snake and a pointwise convolution, added to
/// the input.
#[derive(Module, Debug)]
pub(crate) struct ResidualUnit {
    block: Vec<Layer>,
}

impl ResidualUnit {
    pub(crate) fn new(dim: usize, dilation: usize, groups: usize, device: &Device) -> Self {
        let kernel_size = 7;
        let padding = (kernel_size - 1) * dilation / 2;
        let dilated = Conv1dConfig::new(dim, dim, kernel_size)
            .with_dilation(dilation)
            .with_groups(groups);
        Self {
            block: vec![
                Layer::Snake(Snake1d::new(dim, device)),
                Layer::Conv(causal(dilated, padding, 0).init(device)),
                Layer::Snake(Snake1d::new(dim, device)),
                Layer::Conv(Conv1dConfig::new(dim, dim, 1).init(device)),
            ],
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x.clone() + sequential(&self.block, x)
    }

    pub(crate) fn push(&self, x: Tensor<3>, carry: &mut Carry) -> Tensor<3> {
        x.clone() + sequential_push(&self.block, x, carry)
    }
}

/// Upstream's `SampleRateConditionLayer` of type `scale_bias`: a scale and a bias per channel,
/// looked up by each batch item's sample-rate bucket.
#[derive(Module, Debug)]
pub(crate) struct SampleRateCondition {
    scale_embed: Embedding,
    bias_embed: Embedding,
}

impl SampleRateCondition {
    pub(crate) fn new(channels: usize, buckets: usize, device: &Device) -> Self {
        let embedding = EmbeddingConfig::new(buckets, channels);
        Self {
            scale_embed: embedding.init(device),
            bias_embed: embedding.init(device),
        }
    }

    /// `buckets` holds one bucket per batch item.
    pub(crate) fn forward(&self, x: Tensor<3>, buckets: Tensor<1, Int>) -> Tensor<3> {
        let per_item = |embedding: &Embedding| -> Tensor<3> {
            embedding
                .weight
                .val()
                .select(0, buckets.clone())
                .unsqueeze_dim(2)
        };
        x * per_item(&self.scale_embed) + per_item(&self.bias_embed)
    }
}

#[cfg(test)]
mod tests {
    use burn::{store::SafetensorsStore, tensor::FloatDType};
    use testkit::{Fixture, assert_close, device, fixture};

    use super::*;
    use crate::Weights;

    fn open() -> (Fixture, Weights) {
        let fixture = fixture!("audiovae");
        let mut store = SafetensorsStore::from_file(fixture.path());
        let weights = Weights::from_store(&mut store).unwrap();
        (fixture, weights)
    }

    fn check<M: Module>(name: &str, module: M, forward: impl Fn(&M, Tensor<3>) -> Tensor<3>) {
        let device = device();
        let (fixture, weights) = open();
        let module = weights.load(name, module, FloatDType::F32, &[]).unwrap();
        let x = fixture.tensor(&format!("input.{name}.x"), &device);
        let expected = fixture.tensor(&format!("output.{name}"), &device);
        // A few ulps: sums taken in another order, and the weight-norm fold's f64 norm.
        assert_close(forward(&module, x), expected, 1e-5, 1e-5);
    }

    #[test]
    fn causal_conv_pads_on_the_left_by_its_stride() {
        let config = Conv1dConfig::new(3, 4, 10).with_stride(5);
        let conv = causal(config, 3, 1).init(&device());
        check("conv", conv, |conv, x| conv.forward(x));
    }

    #[test]
    fn causal_transposed_conv_trims_the_right_by_an_odd_stride() {
        let conv = CausalConvTranspose1d::new([4, 3], 5, 3, 1, &device());
        check("conv_transpose_odd", conv, CausalConvTranspose1d::forward);
    }

    #[test]
    fn causal_transposed_conv_trims_the_right_by_an_even_stride() {
        let conv = CausalConvTranspose1d::new([4, 3], 2, 1, 0, &device());
        check("conv_transpose_even", conv, CausalConvTranspose1d::forward);
    }

    #[test]
    fn snake_takes_negative_and_near_zero_alphas() {
        check("snake", Snake1d::new(4, &device()), Snake1d::forward);
    }

    #[test]
    fn residual_unit_dilates_its_depthwise_conv() {
        let unit = ResidualUnit::new(4, 9, 4, &device());
        check("residual", unit, ResidualUnit::forward);
    }

    #[test]
    fn condition_scales_and_shifts_each_item_by_its_bucket() {
        let device = device();
        let (fixture, weights) = open();
        let condition = SampleRateCondition::new(4, 4, &device);
        let condition = weights
            .load("sr_cond", condition, FloatDType::F32, &[])
            .unwrap();
        let buckets = weights.ints("input.sr_cond.buckets").unwrap();
        let buckets = Tensor::from_data(buckets.as_slice(), &device);
        let x = fixture.tensor("input.sr_cond.x", &device);
        let expected = fixture.tensor("output.sr_cond", &device);
        assert_close(condition.forward(x, buckets), expected, 0.0, 0.0);
    }
}
