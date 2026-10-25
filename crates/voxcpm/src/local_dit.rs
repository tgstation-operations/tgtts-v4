use std::path::Path;

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{DType, Device, FloatDType, Int, Tensor, activation::silu},
};
use minicpm4::Mask;

use crate::{Config, Error, weights};

/// VoxCPM2's local DiT, upstream's `VoxCPMLocDiT` of `local_dit_v2.py`: the velocity of the
/// flow at a noisy patch, from a bidirectional MiniCPM-4 stack over the LMs' hidden states, the
/// time, the previous patch and the noisy one.
#[derive(Debug)]
pub struct LocalDit {
    head: Head,
    decoder: minicpm4::Model,
    hidden: usize,
    dtype: DType,
}

#[derive(Module, Debug)]
struct Head {
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
}

#[derive(Module, Debug)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(hidden: usize, device: &Device) -> Self {
        Self {
            linear_1: LinearConfig::new(hidden, hidden).init(device),
            linear_2: LinearConfig::new(hidden, hidden).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.linear_2.forward(silu(self.linear_1.forward(x)))
    }
}

impl LocalDit {
    /// Loads the local DiT `config` describes from the safetensors file at `path`, under the
    /// checkpoint's names, in `dtype` on `device`.
    ///
    /// # Errors
    ///
    /// If the stack asks for what MiniCPM-4 does not implement, or the file does not hold
    /// every weight at the shape `config` gives it.
    pub fn load(
        config: &Config,
        path: &Path,
        dtype: FloatDType,
        device: &Device,
    ) -> Result<Self, Error> {
        let stack = config.local_dit();
        let prefix = "feat_decoder.estimator.decoder.";
        let decoder =
            minicpm4::Model::load(&stack, path, prefix, dtype, device).map_err(Error::Stack)?;
        let (channels, hidden) = (config.feat_dim, stack.hidden_size);
        let head = Head {
            in_proj: LinearConfig::new(channels, hidden).init(device),
            cond_proj: LinearConfig::new(channels, hidden).init(device),
            out_proj: LinearConfig::new(hidden, channels).init(device),
            time_mlp: TimestepEmbedding::new(hidden, device),
            delta_time_mlp: TimestepEmbedding::new(hidden, device),
        };
        Ok(Self {
            head: weights::load(head, path, "feat_decoder.estimator.", dtype)?,
            decoder,
            hidden,
            dtype: dtype.into(),
        })
    }

    /// The velocity `[n, feat_dim, len]` at noisy patches `x` `[n, feat_dim, len]` and times
    /// `t` `[n]`, given the LMs' hidden states `mu` `[n, k * hidden]`, which become `k` tokens,
    /// the previous patches `cond` `[n, feat_dim, cond_len]` and the step lengths `dt` `[n]`.
    /// It is computed in the dtype the DiT was loaded in, which the inputs are cast to.
    ///
    /// # Panics
    ///
    /// If `mu` is not a whole number of tokens.
    pub fn forward(
        &self,
        x: Tensor<3>,
        mu: Tensor<2>,
        t: Tensor<1>,
        cond: Tensor<3>,
        dt: Tensor<1>,
    ) -> Tensor<3> {
        let project =
            |proj: &Linear, x: Tensor<3>| proj.forward(x.cast(self.dtype).swap_dims(1, 2));
        let x = project(&self.head.in_proj, x);
        let cond = project(&self.head.cond_proj, cond);
        let [n, len, hidden] = x.dims();
        let [_, prefix, _] = cond.dims();
        let t = self.head.time_mlp.forward(self.time_embedding(t))
            + self.head.delta_time_mlp.forward(self.time_embedding(dt));
        let [_, width] = mu.dims();
        assert_eq!(
            width % hidden,
            0,
            "mu of width {width} for tokens of {hidden}"
        );
        let mu = mu.cast(self.dtype).reshape([n, width / hidden, hidden]);
        let tokens = width / hidden;
        let x = Tensor::cat(vec![mu, t.unsqueeze_dim(1), cond, x], 1);
        let out = self.decoder.forward(x, Mask::Bidirectional);
        let out = out.narrow(1, tokens + 1 + prefix, len);
        self.head.out_proj.forward(out).swap_dims(1, 2)
    }

    /// Upstream's `SinusoidalPosEmb` at scale 1000, in the order its operations round.
    fn time_embedding(&self, t: Tensor<1>) -> Tensor<2> {
        let half = self.hidden / 2;
        let rate = 10000f64.ln() / (half - 1) as f64;
        let device = t.device();
        let frequencies = Tensor::<1, Int>::arange(0..half as i64, &device)
            .float()
            .cast(self.dtype)
            .mul_scalar(-rate)
            .exp();
        let t = t.cast(self.dtype).mul_scalar(1000).unsqueeze_dim::<2>(1);
        let angles = t * frequencies.unsqueeze_dim(0);
        Tensor::cat(vec![angles.clone().sin(), angles.cos()], 1)
    }
}
