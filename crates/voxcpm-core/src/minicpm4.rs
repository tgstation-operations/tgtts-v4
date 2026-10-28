//! The configuration of a MiniCPM-4 stack, as the `minicpm4` crate loads it.

use serde::Deserialize;

/// A stack's configuration, in the form of upstream's `MiniCPM4Config`: VoxCPM2's
/// `config.json` gives the base LM's as `lm_config`, and the other stacks are derived from it
/// by [`residual_lm`](Self::residual_lm) and [`local`](Self::local).
///
/// A field upstream does not know is refused rather than ignored, and loading refuses values
/// VoxCPM2 does not use: muP scaling, and RoPE other than LongRoPE at its native length.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Read by nothing here: the tokenizer's, carried for upstream's schema.
    pub bos_token_id: u32,
    /// Read by nothing here: the tokenizer's, carried for upstream's schema.
    pub eos_token_id: u32,
    pub hidden_size: usize,
    /// The width of the MLP's hidden layer.
    pub intermediate_size: usize,
    /// The positions RoPE has rotations for.
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    /// Fewer than the query heads for grouped-query attention, each serving an equal share.
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    /// The token embedding's size, or 0 for a stack that is fed embeddings and has none.
    pub vocab_size: usize,
    /// Whether muP scales the residual branches and the embedding, which upstream assumes
    /// when the field is missing and VoxCPM2 turns off.
    #[serde(default = "upstream_use_mup")]
    pub use_mup: bool,
    /// muP's embedding scale, read only under `use_mup`.
    pub scale_emb: f64,
    /// muP's base width, read only under `use_mup`.
    pub dim_model_base: usize,
    /// muP's residual scale, read only under `use_mup`.
    pub scale_depth: f64,
    pub rope_theta: f64,
    /// The width of one attention head, if not the hidden size split among the heads.
    #[serde(default)]
    pub kv_channels: Option<usize>,
    /// Whether attention goes without RoPE.
    #[serde(default)]
    pub no_rope: bool,
}

fn upstream_use_mup() -> bool {
    true
}

/// `rope_scaling`: LongRoPE's per-frequency factors.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RopeScaling {
    /// `"longrope"`, the only scheme upstream implements whatever this says.
    #[serde(rename = "type")]
    pub kind: String,
    /// The divisors of each frequency past `original_max_position_embeddings`.
    pub long_factor: Vec<f64>,
    /// The divisors of each frequency within `original_max_position_embeddings`.
    pub short_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
}

/// What VoxCPM2's `encoder_config` and `dit_config` say of the local encoder and the local
/// DiT's stacks, which [`Config::local`] derives from the base LM's.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalShape {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: Option<usize>,
}

impl Config {
    /// The width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.kv_channels
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// VoxCPM2's residual LM, from the base LM's config and `config.json`'s
    /// `residual_lm_num_layers` and `residual_lm_no_rope`.
    pub fn residual_lm(&self, num_hidden_layers: usize, no_rope: bool) -> Self {
        Self {
            num_hidden_layers,
            vocab_size: 0,
            no_rope,
            ..self.clone()
        }
    }

    /// VoxCPM2's local encoder or local DiT stack, from the base LM's config and the shape
    /// `config.json` gives it. Head count and head width are the shape's, whatever the
    /// hidden size.
    pub fn local(&self, shape: &LocalShape) -> Self {
        Self {
            hidden_size: shape.hidden_dim,
            intermediate_size: shape.ffn_dim,
            num_attention_heads: shape.num_heads,
            num_hidden_layers: shape.num_layers,
            kv_channels: shape.kv_channels,
            vocab_size: 0,
            ..self.clone()
        }
    }
}
