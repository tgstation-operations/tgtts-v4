use std::{
    path::Path,
    thread::{self, ThreadId},
};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, DType, Device, FloatDType, Int, Tensor, TensorData, activation::silu},
};
use minicpm4::KvCache;

use crate::{
    Cfm, Config, Error, GraphCounts, LocalEncoder, Noise, Prompt, Sampling, Stepper, Stepping,
    graph::Graphs, prefill::PrefillGraphs, weights,
};

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
///
/// A clone shares its weights.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) base_lm: minicpm4::Model,
    pub(crate) residual_lm: minicpm4::Model,
    encoder: LocalEncoder,
    cfm: Cfm,
    heads: Heads,
    pub(crate) hidden: usize,
    pub(crate) patch_size: usize,
    pub(crate) feat_dim: usize,
    pub(crate) max_positions: usize,
    pub(crate) lm_dtype: DType,
    pub(crate) device: Device,
    stepping: Stepping,
    pub(crate) graphs: Graphs,
    pub(crate) prefill_graphs: PrefillGraphs,
    /// The thread the weights were loaded on, whose cubecl stream holds them.
    pub(crate) loaded_on: ThreadId,
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
            stepping: Stepping::Eager,
            graphs: Graphs::default(),
            prefill_graphs: PrefillGraphs::new(Tensor::<1>::zeros([1], device).can_mut()),
            loaded_on: thread::current().id(),
        })
    }

    /// How [`patches`](Self::patches) computes each patch: eagerly by default.
    pub fn stepping(&self) -> Stepping {
        self.stepping
    }

    /// How many graphs of its step it has captured and destroyed.
    pub fn graph_counts(&self) -> GraphCounts {
        let mut counts = self.graphs.counts();
        counts.captured += self.prefill_graphs.captured();
        counts
    }

    /// Has [`warm_prefill_graphs`](Self::warm_prefill_graphs) capture graphs of the runs of
    /// prompts that join together of each of `buckets`, rows and positions run, each rounded up
    /// to a power of two.
    pub fn set_prefill_graphs(&mut self, buckets: &[(usize, usize)]) {
        let pow2 =
            |&(rows, run): &(usize, usize)| (rows.next_power_of_two(), run.next_power_of_two());
        self.prefill_graphs.buckets = buckets.iter().map(pow2).collect();
    }

    /// Has [`patches`](Self::patches) compute each patch as `stepping` says. Steps are captured
    /// only on the thread the model loaded on.
    pub fn set_stepping(&mut self, stepping: Stepping) {
        self.stepping = stepping;
    }

    /// Runs `prompt` through the LMs, ready to sample up to `max_patches` patches after it.
    ///
    /// # Errors
    ///
    /// If the prompt is empty, no patch is asked for, or the prompt and the patches would run
    /// past the positions the base LM has.
    pub fn start(&self, prompt: &Prompt, max_patches: usize) -> Result<Generation<'_>, Error> {
        let capacity = self.capacity(prompt, max_patches)?;
        let (device, len) = (&self.device, prompt.len());
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
        let audio = Tensor::<3, Bool>::from_data(audio, device);

        let audio_embeds = self.embed_patches(patches);
        let mut base_cache = self.base_lm.cache(1, capacity);
        let mut residual_cache = self.residual_lm.cache(1, capacity);
        let (outputs, residual) = self.run_prompt(
            tokens,
            audio,
            audio_embeds,
            (&mut base_cache, &mut residual_cache),
        );
        Ok(self.generation((base_cache, residual_cache), (outputs, residual), prompt))
    }

    /// The capacity of the caches a graph of the step holds a generation from a prompt of
    /// `len` positions in, with room for `max_patches` patches after it: the first power of two
    /// they fit in, or the positions the LM has.
    pub fn graph_capacity(&self, len: usize, max_patches: usize) -> usize {
        (len + max_patches)
            .next_power_of_two()
            .min(self.max_positions)
    }

    /// The positions the LMs' caches need for `prompt` and `max_patches` patches after it.
    pub(crate) fn capacity(&self, prompt: &Prompt, max_patches: usize) -> Result<usize, Error> {
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
        Ok(capacity)
    }

    /// Runs positions of prompts through the LMs after those in each row of `caches`: their
    /// `tokens` `[rows, len]`, whether each is `audio` `[rows, len, 1]`, and the embeddings of
    /// their patches `[rows, len, hidden]`, which only audio positions take. Gives the base LM's
    /// outputs, quantized where audio, and the residual LM's.
    pub(crate) fn run_prompt(
        &self,
        tokens: Tensor<2, Int>,
        audio: Tensor<3, Bool>,
        audio_embeds: Tensor<3>,
        caches: (&mut KvCache, &mut KvCache),
    ) -> (Tensor<3>, Tensor<3>) {
        let run = minicpm4::Model::forward_cached;
        self.run_prompt_with(tokens, audio, audio_embeds, caches, run)
    }

    /// Runs positions of prompts through the LMs as [`run_prompt`](Self::run_prompt) does, at
    /// `positions` `[rows, len]` given on the device, as [`minicpm4::Model::forward_run`] runs
    /// them.
    pub(crate) fn run_prompt_at(
        &self,
        (tokens, audio, audio_embeds): (Tensor<2, Int>, Tensor<3, Bool>, Tensor<3>),
        caches: (&mut KvCache, &mut KvCache),
        positions: Tensor<2, Int>,
    ) -> (Tensor<3>, Tensor<3>) {
        let run = |lm: &minicpm4::Model, x, cache: &mut KvCache| {
            lm.forward_run(x, cache, positions.clone())
        };
        self.run_prompt_with(tokens, audio, audio_embeds, caches, run)
    }

    fn run_prompt_with(
        &self,
        tokens: Tensor<2, Int>,
        audio: Tensor<3, Bool>,
        audio_embeds: Tensor<3>,
        (base_cache, residual_cache): (&mut KvCache, &mut KvCache),
        run: impl Fn(&minicpm4::Model, Tensor<3>, &mut KvCache) -> Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [rows, len] = tokens.dims();
        let audio = audio.expand([rows, len, self.hidden]);
        let heads = &self.heads;
        let embeds = self
            .base_lm
            .embed(tokens)
            .mask_where(audio.clone(), audio_embeds.clone());
        let outputs = run(&self.base_lm, embeds, base_cache);
        let quantized = heads.fsq_layer.forward(outputs.clone());
        let outputs = outputs.mask_where(audio.clone(), quantized);
        let audio_embeds = audio_embeds.mask_fill(audio.bool_not(), 0.0);
        let residual_inputs = heads
            .fusion_concat_proj
            .forward(Tensor::cat(vec![outputs.clone(), audio_embeds], 2));
        let residual = run(&self.residual_lm, residual_inputs, residual_cache);
        (outputs, residual)
    }

    /// The generation that goes on from `prompt`, given the LMs' caches over it and their
    /// outputs over its last positions.
    pub(crate) fn generation(
        &self,
        (base_cache, residual_cache): (KvCache, KvCache),
        (outputs, residual): (Tensor<3>, Tensor<3>),
        prompt: &Prompt,
    ) -> Generation<'_> {
        let hidden = self.hidden;
        let last = |x: Tensor<3>| {
            let [_, len, _] = x.dims();
            x.narrow(1, len - 1, 1).reshape([1, hidden])
        };
        let patch = self.patch_size * self.feat_dim;
        let cond = prompt.patches[prompt.patches.len() - patch..].to_vec();
        let cond = TensorData::new(cond, [1, self.patch_size, self.feat_dim]);
        Generation {
            model: self,
            capacity: base_cache.capacity(),
            base_cache,
            residual_cache,
            lm_hidden: last(outputs),
            residual_hidden: last(residual),
            cond: Tensor::from_data(cond, &self.device),
        }
    }

    /// The patches `[n, patch_size, feat_dim]` that follow `prompt`, each sampled from the
    /// next of `noise`, as nanovllm-voxcpm stops: after the first patch at which the stop head
    /// says to, or at `max_patches`.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start), and if a patch comes out NaN or infinite.
    pub fn generate(
        &self,
        prompt: &Prompt,
        sampling: &Sampling,
        max_patches: usize,
        noise: &mut impl Noise,
    ) -> Result<Tensor<3>, Error> {
        let patches = self.patches(prompt, sampling, max_patches, noise)?;
        Ok(Tensor::cat(patches.collect::<Result<_, _>>()?, 0))
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
        let generation = self.start(prompt, max_patches)?;
        let generating = match self.stepping {
            Stepping::Eager => Generating::Eager(Box::new(generation)),
            stepping => {
                let stepper = generation.stepper(sampling, stepping, &mut *noise)?;
                Generating::Stable(Box::new(stepper))
            }
        };
        Ok(Patches {
            model: self,
            generating,
            sampling,
            noise,
            left: max_patches,
            given: 0,
            last: None,
        })
    }

    /// The next of `noise` as a tensor `[1, feat_dim, patch_size]`.
    pub fn noise(&self, noise: &mut impl Noise) -> Tensor<3> {
        let shape = [1, self.feat_dim, self.patch_size];
        let values = noise.next(self.feat_dim * self.patch_size);
        Tensor::from_data(TensorData::new(values, shape), &self.device)
    }

    /// The LMs' embeddings of patches `[batch, len, patch_size, feat_dim]`.
    pub(crate) fn embed_patches(&self, patches: Tensor<4>) -> Tensor<3> {
        let embeds = self.encoder.forward(patches).cast(self.lm_dtype);
        self.heads.enc_to_lm_proj.forward(embeds)
    }
}

/// A generation under way: the LMs' caches and last hidden states, and the last patch.
#[derive(Debug)]
pub struct Generation<'a> {
    pub(crate) model: &'a Model,
    pub(crate) base_cache: KvCache,
    pub(crate) residual_cache: KvCache,
    /// The positions it may hold, which a cache it shares with others may have more of.
    pub(crate) capacity: usize,
    pub(crate) lm_hidden: Tensor<2>,
    pub(crate) residual_hidden: Tensor<2>,
    pub(crate) cond: Tensor<3>,
}

/// The patches of a generation, each `[1, patch_size, feat_dim]` in f32, sampled one at a time
/// as the iterator is advanced: [`Model::patches`]. One that comes out NaN or infinite ends it,
/// as an error.
#[derive(Debug)]
pub struct Patches<'a, N> {
    model: &'a Model,
    generating: Generating<'a>,
    sampling: &'a Sampling,
    noise: &'a mut N,
    /// How many more patches may follow, none once the stop head has said to stop.
    left: usize,
    given: usize,
    /// The patch given last, which the eager step's LMs take in only before the next is
    /// sampled, so that the caller has it sooner.
    last: Option<Tensor<3>>,
}

#[derive(Debug)]
enum Generating<'a> {
    Eager(Box<Generation<'a>>),
    Stable(Box<Stepper<'a>>),
}

impl<N: Noise> Iterator for Patches<'_, N> {
    type Item = Result<Tensor<3>, Error>;

    fn next(&mut self) -> Option<Result<Tensor<3>, Error>> {
        if self.left == 0 {
            return None;
        }
        let step = match &mut self.generating {
            Generating::Eager(generation) => {
                if let Some(patch) = self.last.take() {
                    generation.advance(patch);
                }
                let noise = self.model.noise(self.noise);
                let step = generation.sample(noise, self.sampling);
                self.last = Some(step.patch.clone());
                step
            }
            Generating::Stable(stepper) => stepper.step(),
        };
        self.given += 1;
        if !step.is_finite() {
            self.left = 0;
            return Some(Err(Error::NotFinite(self.given - 1)));
        }
        self.left = if step.stops() { 0 } else { self.left - 1 };
        Some(Ok(step.patch))
    }
}

/// A patch sampled, and what the stop head said of the state it was sampled in.
#[derive(Clone, Debug)]
pub struct Step {
    /// The patch `[1, patch_size, feat_dim]`, in f32.
    pub patch: Tensor<3>,
    /// The stop head's logits `[1, 2]`, of going on and of stopping after the patch.
    pub stop_logits: Tensor<2>,
}

impl Step {
    /// Whether the patch and the stop head's logits are all finite.
    pub fn is_finite(&self) -> bool {
        let finite = |data: TensorData| data.iter::<f32>().all(f32::is_finite);
        finite(self.patch.clone().into_data()) && finite(self.stop_logits.clone().into_data())
    }

    /// Whether generation stops after this patch.
    pub fn stops(&self) -> bool {
        let stop = self.stop_logits.clone().argmax(1).into_data();
        stop.iter::<i64>().next() == Some(1)
    }
}

impl Generation<'_> {
    /// Samples the next patch from `noise` `[1, feat_dim, patch_size]`.
    pub fn sample(&self, noise: Tensor<3>, sampling: &Sampling) -> Step {
        self.model.sample_after(
            self.lm_hidden.clone(),
            self.residual_hidden.clone(),
            self.cond.clone(),
            |cfm, mu, cond| cfm.sample(mu, cond, noise, sampling),
        )
    }

    /// Feeds `patch` `[1, patch_size, feat_dim]` to the LMs as the next position.
    pub fn advance(&mut self, patch: Tensor<3>) {
        let embed = self.model.embed_patches(patch.clone().unsqueeze_dim(1));
        let caches = (&mut self.base_cache, &mut self.residual_cache);
        (self.lm_hidden, self.residual_hidden) =
            self.model
                .advance_lms(embed, caches, minicpm4::Model::forward_cached);
        self.cond = patch;
    }
}

impl Model {
    /// A patch for each row of the LMs' last hidden states `[rows, hidden]` after the patches
    /// `cond` `[rows, patch_size, feat_dim]` before them, which `sample` samples given the
    /// DiT's conditions, `mu` and `cond` `[rows, feat_dim, patch_size]`.
    pub(crate) fn sample_after(
        &self,
        lm_hidden: Tensor<2>,
        residual_hidden: Tensor<2>,
        cond: Tensor<3>,
        sample: impl FnOnce(&Cfm, Tensor<2>, Tensor<3>) -> Tensor<3>,
    ) -> Step {
        let heads = &self.heads;
        let mu = Tensor::cat(
            vec![
                heads.lm_to_dit_proj.forward(lm_hidden.clone()),
                heads.res_to_dit_proj.forward(residual_hidden),
            ],
            1,
        );
        let patch = sample(&self.cfm, mu, cond.swap_dims(1, 2));
        let stop = silu(heads.stop_proj.forward(lm_hidden));
        Step {
            patch: patch.swap_dims(1, 2),
            stop_logits: heads.stop_head.forward(stop),
        }
    }

    /// Runs the LMs a position further over each row of `caches`, from the embedding of its
    /// patch `[rows, 1, hidden]`, each stack as `step` runs it, and gives their last hidden
    /// states `[rows, hidden]`.
    pub(crate) fn advance_lms(
        &self,
        embeds: Tensor<3>,
        (base_cache, residual_cache): (&mut KvCache, &mut KvCache),
        step: impl Fn(&minicpm4::Model, Tensor<3>, &mut KvCache) -> Tensor<3>,
    ) -> (Tensor<2>, Tensor<2>) {
        let heads = &self.heads;
        let [rows, ..] = embeds.dims();
        let outputs = step(&self.base_lm, embeds.clone(), base_cache);
        let outputs = heads.fsq_layer.forward(outputs);
        let residual_inputs = heads
            .fusion_concat_proj
            .forward(Tensor::cat(vec![outputs.clone(), embeds], 2));
        let residual = step(&self.residual_lm, residual_inputs, residual_cache);
        (
            outputs.reshape([rows, self.hidden]),
            residual.reshape([rows, self.hidden]),
        )
    }
}
