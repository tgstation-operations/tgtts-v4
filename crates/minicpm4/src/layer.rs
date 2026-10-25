use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{
        Bool, Device, FloatDType, Tensor, activation::silu, module::attention,
        ops::AttentionModuleOptions,
    },
};

use crate::{Config, cache::LayerCache, rope::Rotation};

/// Upstream's `MiniCPMRMSNorm`, which takes the mean square in f32.
#[derive(Module, Debug)]
pub(crate) struct RmsNorm {
    weight: Param<Tensor<1>>,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn init(size: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Initializer::Ones.init([size], device),
            eps,
        }
    }

    pub(crate) fn weight(&self) -> Tensor<1> {
        self.weight.val()
    }

    pub(crate) fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        let x = x.cast(FloatDType::F32);
        let variance = x.clone().square().mean_dim(D - 1);
        let normed = x * variance.add_scalar(self.eps).sqrt().recip();
        normed.cast(dtype) * self.weight.val().unsqueeze()
    }
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn init(config: &Config, device: &Device) -> Self {
        let (hidden, head_dim) = (config.hidden_size, config.head_dim());
        let (heads, kv_heads) = (config.num_attention_heads, config.num_key_value_heads);
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        Self {
            q_proj: linear(hidden, heads * head_dim),
            k_proj: linear(hidden, kv_heads * head_dim),
            v_proj: linear(hidden, kv_heads * head_dim),
            o_proj: linear(heads * head_dim, hidden),
            heads,
            kv_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        rotation: Option<&Rotation>,
        mask: Option<&Tensor<4, Bool>>,
        cache: Option<(&mut LayerCache, usize)>,
    ) -> Tensor<3> {
        let [batch, len, _] = x.dims();
        let project = |proj: &Linear, heads| {
            proj.forward(x.clone())
                .reshape([batch, len, heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let mut q = project(&self.q_proj, self.heads);
        let mut k = project(&self.k_proj, self.kv_heads);
        let v = project(&self.v_proj, self.kv_heads);
        if let Some(rotation) = rotation {
            q = rotation.apply(q);
            k = rotation.apply(k);
        }
        let (k, v) = match cache {
            Some((cache, start)) => cache.append(start, k, v),
            None => (k, v),
        };
        // Each key-value head serves a run of consecutive query heads, as torch's enable_gqa
        // has it: stacking a run's queries along the sequence lets them share it uncopied.
        let groups = self.heads / self.kv_heads;
        let q = q.reshape([batch, self.kv_heads, groups * len, self.head_dim]);
        let options = AttentionModuleOptions::default();
        let out = attention(q, k, v, mask.cloned(), None, options);
        let out = out
            .reshape([batch, self.heads, len, self.head_dim])
            .swap_dims(1, 2);
        self.o_proj
            .forward(out.reshape([batch, len, self.heads * self.head_dim]))
    }
}

#[derive(Module, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn init(config: &Config, device: &Device) -> Self {
        let (hidden, inner) = (config.hidden_size, config.intermediate_size);
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        Self {
            gate_proj: linear(hidden, inner),
            up_proj: linear(hidden, inner),
            down_proj: linear(inner, hidden),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

#[derive(Module, Debug)]
pub(crate) struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub(crate) fn init(config: &Config, device: &Device) -> Self {
        let norm = || RmsNorm::init(config.hidden_size, config.rms_norm_eps, device);
        Self {
            self_attn: Attention::init(config, device),
            mlp: Mlp::init(config, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
        }
    }

    pub(crate) fn forward(
        &self,
        x: Tensor<3>,
        rotation: Option<&Rotation>,
        mask: Option<&Tensor<4, Bool>>,
        cache: Option<(&mut LayerCache, usize)>,
    ) -> Tensor<3> {
        let normed = self.input_layernorm.forward(x.clone());
        let attended = self.self_attn.forward(normed, rotation, mask, cache);
        let x = x + attended;
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}
