use std::path::Path;

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, DType, Device, FloatDType, Int, Tensor, TensorData, activation::silu},
};
use minicpm4::KvCache;

use crate::{Cfm, Config, Error, LocalEncoder, Noise, Prompt, Sampling, weights};

/// The dtype each part of the model is held and computed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dtypes {
    /// The base and residual LMs, and the projections and heads around them.
    pub lm: FloatDType,
    pub local_encoder: FloatDType,
    pub local_dit: FloatDType,
    /// The audio VAE's decoder, which [`TextToSpeech`](crate::TextToSpeech) loads beside the model.
    pub audio_decoder: FloatDType,
}

/// VoxCPM2 short of its audio VAE: from a prompt of text and audio latents to the latents that
/// follow, as upstream's `VoxCPM2Model._inference` computes them and nanovllm-voxcpm stops.
#[derive(Debug)]
pub struct Model {
    base_lm: minicpm4::Model,
    residual_lm: minicpm4::Model,
    encoder: LocalEncoder,
    cfm: Cfm,
    heads: Heads,
    hidden: usize,
    patch_size: usize,
    feat_dim: usize,
    max_positions: usize,
    lm_dtype: DType,
    device: Device,
}

/// The layers between the stacks, under their names in the checkpoint.
#[derive(Module, Debug)]
struct Heads {
    fsq_layer: ScalarQuantizer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
}

/// Upstream's `ScalarQuantizationLayer` at inference: a bottleneck whose values, squashed into
/// (-1, 1), are rounded to multiples of `1 / scale`.
#[derive(Module, Debug)]
struct ScalarQuantizer {
    in_proj: Linear,
    out_proj: Linear,
    scale: u32,
}

impl ScalarQuantizer {
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let scale = f64::from(self.scale);
        let levels = self.in_proj.forward(x).tanh().mul_scalar(scale).round();
        self.out_proj.forward(levels.div_scalar(scale))
    }
}

impl Model {
    /// Loads the model `config` describes from the safetensors file at `path`, under the
    /// checkpoint's names, each part in its dtype from `dtypes`, on `device`.
    ///
    /// # Errors
    ///
    /// If the config asks for what is not implemented, or the file does not hold every weight
    /// at the shape `config` gives it.
    pub fn load(
        config: &Config,
        path: &Path,
        dtypes: &Dtypes,
        device: &Device,
    ) -> Result<Self, Error> {
        config.check()?;
        let lm = &config.lm_config;
        let stack = |config, prefix| {
            minicpm4::Model::load(config, path, prefix, dtypes.lm, device).map_err(Error::Stack)
        };
        let (hidden, latent) = (lm.hidden_size, config.scalar_quantization_latent_dim);
        let dit_hidden = config.dit_config.hidden_dim;
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).init(device);
        let heads = Heads {
            fsq_layer: ScalarQuantizer {
                in_proj: linear(hidden, latent),
                out_proj: linear(latent, hidden),
                scale: config.scalar_quantization_scale,
            },
            enc_to_lm_proj: linear(config.encoder_config.hidden_dim, hidden),
            lm_to_dit_proj: linear(hidden, dit_hidden),
            res_to_dit_proj: linear(hidden, dit_hidden),
            fusion_concat_proj: linear(2 * hidden, hidden),
            stop_proj: linear(hidden, hidden),
            stop_head: LinearConfig::new(hidden, 2).with_bias(false).init(device),
        };
        Ok(Self {
            base_lm: stack(lm, "base_lm.")?,
            residual_lm: stack(&config.residual_lm(), "residual_lm.")?,
            encoder: LocalEncoder::load(config, path, dtypes.local_encoder, device)?,
            cfm: Cfm::load(config, path, dtypes.local_dit, device)?,
            heads: weights::load(heads, path, "", dtypes.lm)?,
            hidden,
            patch_size: config.patch_size,
            feat_dim: config.feat_dim,
            max_positions: lm.max_position_embeddings,
            lm_dtype: dtypes.lm.into(),
            device: device.clone(),
        })
    }

    /// Runs `prompt` through the LMs, ready to sample up to `max_patches` patches after it.
    ///
    /// # Errors
    ///
    /// If the prompt is empty, no patch is asked for, or the prompt and the patches would run
    /// past the positions the base LM has.
    pub fn start(&self, prompt: &Prompt, max_patches: usize) -> Result<Generation<'_>, Error> {
        let len = prompt.len();
        let capacity = len + max_patches;
        if len == 0 || max_patches == 0 {
            return Err(Error::Request(format!(
                "{len} positions of prompt and {max_patches} patches after it"
            )));
        }
        if capacity > self.max_positions {
            return Err(Error::Request(format!(
                "{len} positions of prompt and {max_patches} patches after it are past the {} \
                 positions the LM has",
                self.max_positions
            )));
        }
        let (device, hidden) = (&self.device, self.hidden);
        let tokens = prompt
            .tokens
            .iter()
            .map(|&token| i64::from(token))
            .collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, len]), device);
        let shape = [1, len, self.patch_size, self.feat_dim];
        let patches =
            Tensor::<4>::from_data(TensorData::new(prompt.patches.clone(), shape), device);
        let audio = TensorData::new(prompt.audio.clone(), [1, len, 1]);
        let audio = Tensor::<3, Bool>::from_data(audio, device).expand([1, len, hidden]);

        let heads = &self.heads;
        let audio_embeds = self.embed_patches(patches.clone());
        let embeds = self
            .base_lm
            .embed(tokens)
            .mask_where(audio.clone(), audio_embeds.clone());
        let mut base_cache = self.base_lm.cache(1, capacity);
        let outputs = self.base_lm.forward_cached(embeds, &mut base_cache);
        let quantized = heads.fsq_layer.forward(outputs.clone());
        let outputs = outputs.mask_where(audio.clone(), quantized);
        let audio_embeds = audio_embeds.mask_fill(audio.bool_not(), 0.0);
        let residual_inputs = heads
            .fusion_concat_proj
            .forward(Tensor::cat(vec![outputs.clone(), audio_embeds], 2));
        let mut residual_cache = self.residual_lm.cache(1, capacity);
        let residual = self
            .residual_lm
            .forward_cached(residual_inputs, &mut residual_cache);
        let last = |x: Tensor<3>| x.narrow(1, len - 1, 1).reshape([1, hidden]);
        Ok(Generation {
            model: self,
            base_cache,
            residual_cache,
            lm_hidden: last(outputs),
            residual_hidden: last(residual),
            cond: patches
                .narrow(1, len - 1, 1)
                .reshape([1, self.patch_size, self.feat_dim]),
        })
    }

    /// The patches `[n, patch_size, feat_dim]` that follow `prompt`, each sampled from the
    /// next of `noise`, as nanovllm-voxcpm stops: after the first patch at which the stop head
    /// says to, or at `max_patches`.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start).
    pub fn generate(
        &self,
        prompt: &Prompt,
        sampling: &Sampling,
        max_patches: usize,
        noise: &mut impl Noise,
    ) -> Result<Tensor<3>, Error> {
        let patches = self.patches(prompt, sampling, max_patches, noise)?;
        Ok(Tensor::cat(patches.collect(), 0))
    }

    /// The patches [`generate`](Self::generate) gives, each sampled only as the iterator is
    /// advanced, for a caller to decode and send on while the next is sampled.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start).
    pub fn patches<'a, N: Noise>(
        &'a self,
        prompt: &Prompt,
        sampling: &'a Sampling,
        max_patches: usize,
        noise: &'a mut N,
    ) -> Result<Patches<'a, N>, Error> {
        Ok(Patches {
            generation: self.start(prompt, max_patches)?,
            sampling,
            noise,
            left: max_patches,
            last: None,
        })
    }

    /// The next of `noise` as a tensor `[1, feat_dim, patch_size]`.
    pub fn noise(&self, noise: &mut impl Noise) -> Tensor<3> {
        let shape = [1, self.feat_dim, self.patch_size];
        let values = noise.next(self.feat_dim * self.patch_size);
        Tensor::from_data(TensorData::new(values, shape), &self.device)
    }

    /// The LMs' embeddings of patches `[1, len, patch_size, feat_dim]`.
    fn embed_patches(&self, patches: Tensor<4>) -> Tensor<3> {
        let embeds = self.encoder.forward(patches).cast(self.lm_dtype);
        self.heads.enc_to_lm_proj.forward(embeds)
    }
}

/// A generation under way: the LMs' caches and last hidden states, and the last patch.
#[derive(Debug)]
pub struct Generation<'a> {
    model: &'a Model,
    base_cache: KvCache,
    residual_cache: KvCache,
    lm_hidden: Tensor<2>,
    residual_hidden: Tensor<2>,
    cond: Tensor<3>,
}

/// The patches of a generation, each `[1, patch_size, feat_dim]` in f32, sampled one at a time
/// as the iterator is advanced: [`Model::patches`].
#[derive(Debug)]
pub struct Patches<'a, N> {
    generation: Generation<'a>,
    sampling: &'a Sampling,
    noise: &'a mut N,
    /// How many more patches may follow, none once the stop head has said to stop.
    left: usize,
    /// The patch given last, which the LMs take in only before the next is sampled, so that
    /// the caller has it sooner.
    last: Option<Tensor<3>>,
}

impl<N: Noise> Iterator for Patches<'_, N> {
    type Item = Tensor<3>;

    fn next(&mut self) -> Option<Tensor<3>> {
        if self.left == 0 {
            return None;
        }
        if let Some(patch) = self.last.take() {
            self.generation.advance(patch);
        }
        let noise = self.generation.model.noise(self.noise);
        let step = self.generation.sample(noise, self.sampling);
        self.left = if step.stops() { 0 } else { self.left - 1 };
        self.last = Some(step.patch.clone());
        Some(step.patch)
    }
}

/// A patch sampled, and what the stop head said of the state it was sampled in.
#[derive(Debug)]
pub struct Step {
    /// The patch `[1, patch_size, feat_dim]`, in f32.
    pub patch: Tensor<3>,
    /// The stop head's logits `[1, 2]`, of going on and of stopping after the patch.
    pub stop_logits: Tensor<2>,
}

impl Step {
    /// Whether generation stops after this patch.
    pub fn stops(&self) -> bool {
        let stop = self.stop_logits.clone().argmax(1).into_data();
        stop.iter::<i64>().next() == Some(1)
    }
}

impl Generation<'_> {
    /// Samples the next patch from `noise` `[1, feat_dim, patch_size]`.
    pub fn sample(&self, noise: Tensor<3>, sampling: &Sampling) -> Step {
        let heads = &self.model.heads;
        let mu = Tensor::cat(
            vec![
                heads.lm_to_dit_proj.forward(self.lm_hidden.clone()),
                heads.res_to_dit_proj.forward(self.residual_hidden.clone()),
            ],
            1,
        );
        let cond = self.cond.clone().swap_dims(1, 2);
        let patch = self.model.cfm.sample(mu, cond, noise, sampling);
        let stop = silu(heads.stop_proj.forward(self.lm_hidden.clone()));
        Step {
            patch: patch.swap_dims(1, 2),
            stop_logits: heads.stop_head.forward(stop),
        }
    }

    /// Feeds `patch` `[1, patch_size, feat_dim]` to the LMs as the next position.
    pub fn advance(&mut self, patch: Tensor<3>) {
        let (model, heads) = (self.model, &self.model.heads);
        let embed = model.embed_patches(patch.clone().unsqueeze_dim(1));
        let outputs = model
            .base_lm
            .forward_cached(embed.clone(), &mut self.base_cache);
        let outputs = heads.fsq_layer.forward(outputs);
        let residual_inputs = heads
            .fusion_concat_proj
            .forward(Tensor::cat(vec![outputs.clone(), embed], 2));
        let residual = model
            .residual_lm
            .forward_cached(residual_inputs, &mut self.residual_cache);
        self.lm_hidden = outputs.reshape([1, model.hidden]);
        self.residual_hidden = residual.reshape([1, model.hidden]);
        self.cond = patch;
    }
}
