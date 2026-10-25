use audiovae::AudioVaeConfig;
use minicpm4::LocalShape;
use serde::Deserialize;

use crate::Error;

/// A VoxCPM2 checkpoint's `config.json`, in the form of upstream's `VoxCPMConfig`.
///
/// Fields the JSON leaves out take upstream's defaults. A field upstream does not have is
/// refused rather than ignored, but for the two the checkpoint carries beyond upstream's schema,
/// `architecture` and `dit_config.mean_mode`. [`from_json`](Self::from_json) refuses what this
/// crate does not implement.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// What the checkpoint says it is, which upstream does not read. Only `"voxcpm2"` is
    /// implemented.
    #[serde(default)]
    pub architecture: Option<String>,
    /// The base LM's, from which the other three stacks are derived.
    pub lm_config: minicpm4::Config,
    /// Latent frames to a patch, the unit the LMs step in.
    #[serde(default = "defaults::patch_size")]
    pub patch_size: usize,
    /// Channels of a latent frame, which have to be the audio VAE's.
    #[serde(default = "defaults::feat_dim")]
    pub feat_dim: usize,
    #[serde(default = "defaults::residual_lm_num_layers")]
    pub residual_lm_num_layers: usize,
    #[serde(default)]
    pub residual_lm_no_rope: bool,
    /// Width of the scalar quantizer's bottleneck.
    #[serde(default = "defaults::scalar_quantization_latent_dim")]
    pub scalar_quantization_latent_dim: usize,
    /// The scalar quantizer rounds each bottleneck value to a multiple of its reciprocal.
    #[serde(default = "defaults::scalar_quantization_scale")]
    pub scalar_quantization_scale: u32,
    pub encoder_config: EncoderConfig,
    pub dit_config: DitConfig,
    /// The audio VAE's, upstream's defaults where it is null or missing.
    #[serde(default)]
    pub audio_vae_config: Option<AudioVaeConfig>,
    /// Read by nothing here: the positions upstream allocates its KV caches for.
    #[serde(default = "defaults::max_length")]
    pub max_length: usize,
    /// Read by nothing here: where upstream runs the model.
    #[serde(default = "defaults::device")]
    pub device: String,
    /// Read by nothing here: the dtype upstream runs the model in.
    #[serde(default = "defaults::dtype")]
    pub dtype: String,
}

/// `encoder_config`: the shape of the local encoder's stack.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EncoderConfig {
    #[serde(default = "defaults::hidden_dim")]
    pub hidden_dim: usize,
    #[serde(default = "defaults::ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "defaults::num_heads")]
    pub num_heads: usize,
    #[serde(default = "defaults::num_layers")]
    pub num_layers: usize,
    #[serde(default)]
    pub kv_channels: Option<usize>,
}

/// `dit_config`: the shape of the local DiT's stack, and how it is sampled.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DitConfig {
    #[serde(default = "defaults::hidden_dim")]
    pub hidden_dim: usize,
    #[serde(default = "defaults::ffn_dim")]
    pub ffn_dim: usize,
    #[serde(default = "defaults::num_heads")]
    pub num_heads: usize,
    #[serde(default = "defaults::num_layers")]
    pub num_layers: usize,
    #[serde(default)]
    pub kv_channels: Option<usize>,
    /// Whether the DiT predicts the mean velocity over a step rather than the velocity at its
    /// start. Only `false` is implemented.
    #[serde(default)]
    pub dit_mean_mode: bool,
    /// The checkpoint's name for `dit_mean_mode`, under which upstream ignores it. Only `false`
    /// is implemented.
    #[serde(default)]
    pub mean_mode: bool,
    pub cfm_config: CfmConfig,
}

/// `cfm_config`: the flow matching the local DiT was trained with, most of which inference
/// does not read.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CfmConfig {
    /// Read by nothing here, nor by upstream's sampler.
    pub sigma_min: f64,
    /// The ODE solver. Only `"euler"` is implemented, the only one upstream has whatever this
    /// says.
    pub solver: String,
    /// Read by nothing here: training's.
    pub t_scheduler: String,
    /// Read by nothing here: training's.
    pub training_cfg_rate: f64,
    /// Read by nothing here, nor by upstream's sampler, which is given its guidance scale.
    pub inference_cfg_rate: f64,
    /// Read by nothing here: training's.
    pub reg_loss_type: String,
    /// Read by nothing here: training's.
    pub ratio_r_neq_t_range: (f64, f64),
    /// Read by nothing here: training's.
    pub noise_cond_prob_range: (f64, f64),
    /// Read by nothing here: training's.
    pub noise_cond_scale: f64,
}

impl Default for CfmConfig {
    fn default() -> Self {
        Self {
            sigma_min: 1e-6,
            solver: "euler".into(),
            t_scheduler: "log-norm".into(),
            training_cfg_rate: 0.1,
            inference_cfg_rate: 1.0,
            reg_loss_type: "l1".into(),
            ratio_r_neq_t_range: (0.25, 0.75),
            noise_cond_prob_range: (0.0, 0.0),
            noise_cond_scale: 0.0,
        }
    }
}

mod defaults {
    pub(super) fn patch_size() -> usize {
        4
    }
    pub(super) fn feat_dim() -> usize {
        64
    }
    pub(super) fn residual_lm_num_layers() -> usize {
        8
    }
    pub(super) fn scalar_quantization_latent_dim() -> usize {
        512
    }
    pub(super) fn scalar_quantization_scale() -> u32 {
        9
    }
    pub(super) fn max_length() -> usize {
        8192
    }
    pub(super) fn device() -> String {
        "cuda".into()
    }
    pub(super) fn dtype() -> String {
        "bfloat16".into()
    }
    pub(super) fn hidden_dim() -> usize {
        1024
    }
    pub(super) fn ffn_dim() -> usize {
        4096
    }
    pub(super) fn num_heads() -> usize {
        16
    }
    pub(super) fn num_layers() -> usize {
        4
    }
}

impl Config {
    /// Parses a `config.json`, refusing what this crate does not implement: another
    /// architecture, mean mode, a solver other than Euler, or latent frames other than the
    /// audio VAE's. What the MiniCPM-4 stacks do not implement is refused as they load.
    ///
    /// # Errors
    ///
    /// If the JSON is not a config upstream would read, or asks for any of those.
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let config: Self = serde_json::from_str(json).map_err(Error::Config)?;
        config.check()?;
        Ok(config)
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        let unsupported = |what: String| Err(Error::Unsupported(what));
        if let Some(architecture) = &self.architecture
            && architecture != "voxcpm2"
        {
            return unsupported(format!("architecture {architecture:?}"));
        }
        let dit = &self.dit_config;
        if dit.dit_mean_mode {
            return unsupported("dit_config.dit_mean_mode: true".into());
        }
        if dit.mean_mode {
            return unsupported("dit_config.mean_mode: true".into());
        }
        if dit.cfm_config.solver != "euler" {
            return unsupported(format!("solver {:?}", dit.cfm_config.solver));
        }
        let latent_dim = self.audio_vae().latent_dim;
        if latent_dim != self.feat_dim {
            return unsupported(format!(
                "feat_dim {} where the audio VAE's latents have {latent_dim} channels",
                self.feat_dim
            ));
        }
        Ok(())
    }

    /// The residual LM's stack, derived from the base LM's as upstream derives it.
    pub fn residual_lm(&self) -> minicpm4::Config {
        self.lm_config
            .residual_lm(self.residual_lm_num_layers, self.residual_lm_no_rope)
    }

    /// The local encoder's stack, derived from the base LM's as upstream derives it.
    pub fn local_encoder(&self) -> minicpm4::Config {
        let encoder = &self.encoder_config;
        self.lm_config.local(&LocalShape {
            hidden_dim: encoder.hidden_dim,
            ffn_dim: encoder.ffn_dim,
            num_heads: encoder.num_heads,
            num_layers: encoder.num_layers,
            kv_channels: encoder.kv_channels,
        })
    }

    /// The local DiT's stack, derived from the base LM's as upstream derives it.
    pub fn local_dit(&self) -> minicpm4::Config {
        let dit = &self.dit_config;
        self.lm_config.local(&LocalShape {
            hidden_dim: dit.hidden_dim,
            ffn_dim: dit.ffn_dim,
            num_heads: dit.num_heads,
            num_layers: dit.num_layers,
            kv_channels: dit.kv_channels,
        })
    }

    /// The audio VAE's config.
    pub fn audio_vae(&self) -> AudioVaeConfig {
        self.audio_vae_config.clone().unwrap_or_default()
    }
}
