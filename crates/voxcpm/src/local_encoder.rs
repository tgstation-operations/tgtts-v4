use std::path::Path;

use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{DType, Device, FloatDType, Tensor},
};
use minicpm4::Mask;

use crate::{Config, Error, weights};

/// VoxCPM2's local encoder, upstream's `VoxCPMLocEnc`: it embeds a patch of latent frames for
/// the LMs, as the output at a learned token that leads the patch through a bidirectional
/// MiniCPM-4 stack.
#[derive(Debug)]
pub struct LocalEncoder {
    head: Head,
    encoder: minicpm4::Model,
    dtype: DType,
}

#[derive(Module, Debug)]
struct Head {
    special_token: Param<Tensor<4>>,
    in_proj: Linear,
}

impl LocalEncoder {
    /// Loads the local encoder `config` describes from the safetensors file at `path`, under
    /// the checkpoint's names, in `dtype` on `device`.
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
        let stack = config.local_encoder();
        let prefix = "feat_encoder.encoder.";
        let encoder =
            minicpm4::Model::load(&stack, path, prefix, dtype, device).map_err(Error::Stack)?;
        let hidden = stack.hidden_size;
        let head = Head {
            special_token: Initializer::Zeros.init([1, 1, 1, hidden], device),
            in_proj: LinearConfig::new(config.feat_dim, hidden).init(device),
        };
        Ok(Self {
            head: weights::load(head, path, "feat_encoder.", dtype)?,
            encoder,
            dtype: dtype.into(),
        })
    }

    /// Embeds patches `[batch, len, patch_size, feat_dim]` as `[batch, len, hidden]`, in the
    /// dtype the encoder was loaded in, which the patches are cast to.
    pub fn forward(&self, patches: Tensor<4>) -> Tensor<3> {
        let [batch, len, patch_size, _] = patches.dims();
        let x = self.head.in_proj.forward(patches.cast(self.dtype));
        let [.., hidden] = x.dims();
        let special = self
            .head
            .special_token
            .val()
            .expand([batch, len, 1, hidden]);
        let x = Tensor::cat(vec![special, x], 2).reshape([batch * len, patch_size + 1, hidden]);
        let out = self.encoder.forward(x, Mask::Bidirectional);
        out.narrow(1, 0, 1).reshape([batch, len, hidden])
    }
}
