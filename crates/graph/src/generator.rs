//! VoxCPM2's text-to-speech at batch 1 on the GPU, from a prompt to audio, one request at a time:
//! the prompt run into the patch step's caches ([`prompt`](crate::prompt)), patches sampled a
//! submission each ([`patch`](crate::patch)) until the stop head says to stop after one or the
//! request's most are sampled, as the service stops, and their latent frames decoded by the audio
//! VAE ([`vae`](crate::vae)) after the voice's last frames, as the service primes its decoder,
//! in chunks of a fixed number of frames as they come.

use std::{fs, mem, path::Path};

use serde::Deserialize;
use vulkan::Context;

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode,
    kernels::Kernels,
    lm_side,
    local::{self, PatchConfig},
    patch::{PatchStep, patch_graph, transposed},
    prompt::{Prompt, Prompter},
    vae::{Parameter, VaeConfig, VaeDecoder, VaeShape},
};

/// The voice's last frames the audio VAE is primed with before a speech, nanovllm-voxcpm's
/// `n_decode_pad_frames`.
const CONTEXT_FRAMES: usize = 12;

/// What the generator is recorded for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The positions the LMs' caches hold, a multiple of 32: a request's prompt and its patches.
    pub capacity: usize,
    /// The most positions of a prompt one submission runs, a multiple of 128.
    pub max_rows: usize,
    /// The sampler's Euler steps.
    pub timesteps: usize,
    /// The most latent frames the audio VAE decodes in one submission, and those it waits for
    /// before it decodes any.
    pub chunk_frames: usize,
    /// Whether each submission is timed per dispatch.
    pub timed: bool,
}

/// The parts of VoxCPM2's `config.json` the generator reads.
#[derive(Deserialize)]
struct AudioVae {
    audio_vae_config: VaeConfig,
}

/// A patch the generator sampled.
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    /// Its latent frames, `[patch_size][feat_dim]`.
    pub frames: Vec<f32>,
    /// The stop head's logits it was sampled beside, of going on and of stopping after it.
    pub stop_logits: [f32; 2],
    /// Whether it is the request's last.
    pub last: bool,
}

/// A request under way.
#[derive(Debug)]
struct Request {
    /// The position the next patch's step runs the patch before at.
    position: usize,
    /// The patch before, `[feat_dim][patch_size]`, as the patch step takes it.
    before: Vec<f32>,
    /// The patches it may still sample.
    left: usize,
    sampled: usize,
    temperature: f32,
    cfg: f32,
    /// Latent frames waiting to be decoded, `[frames][feat_dim]`.
    frames: Vec<f32>,
    /// The voice's frames still at the start of the frames, whose audio is not the speech's.
    context: usize,
    /// Audio decoded and not yet taken.
    audio: Vec<f32>,
}

/// VoxCPM2 on the GPU, loaded once and generating one request at a time.
#[derive(Debug)]
pub struct Generator {
    step: PatchStep,
    prompter: Prompter,
    vae: VaeDecoder,
    checkpoint: Checkpoint,
    settings: Settings,
    patch_size: usize,
    feat_dim: usize,
    request: Option<Request>,
}

impl Generator {
    /// Records the generator `settings` describe for the checkpoint in `dir`, its kernels cut as
    /// the device suits, and loads the weights: the model's from its `model.safetensors`, the
    /// audio VAE decoder's as `parameter` gives each by its name under the decoder, weight norm
    /// folded, as the `audiovae` crate's `Weights::floats` reads them from its `audiovae.pth`.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        dir: &Path,
        settings: &Settings,
        parameter: impl FnMut(&str) -> Result<Parameter, Error>,
    ) -> Result<Self, Error> {
        let path = dir.join("config.json");
        let json = fs::read_to_string(&path)
            .map_err(|e| Error::Checkpoint(format!("{}: {e}", path.display())))?;
        let parse = |e: serde_json::Error| Error::Checkpoint(format!("{}: {e}", path.display()));
        let side_config: lm_side::Config = serde_json::from_str(&json).map_err(parse)?;
        let config: PatchConfig = serde_json::from_str(&json).map_err(parse)?;
        let AudioVae { audio_vae_config } = serde_json::from_str(&json).map_err(parse)?;
        let report = ctx.report();
        let lm = decode::Tuning::for_device(report)?;
        let local = local::Tuning::for_device(report)?;
        let shape = lm_side::LmSideShape::new(&side_config, settings.capacity)?;
        let graph = patch_graph(&shape, &config, settings.timesteps, &lm, &local)?;
        let mut step = PatchStep::new(ctx, kernels, graph, settings.timed)?;
        let checkpoint = Checkpoint::open(dir.join("model.safetensors"))?;
        step.load(&checkpoint)?;
        let prompter = Prompter::new(
            ctx,
            kernels,
            &mut step,
            settings.max_rows,
            &lm,
            settings.timed,
        )?;
        let vae_shape = VaeShape::new(&audio_vae_config, settings.chunk_frames)?;
        let mut vae = VaeDecoder::new(ctx, kernels, &vae_shape, settings.timed)?;
        vae.load(parameter)?;
        Ok(Generator {
            step,
            prompter,
            vae,
            checkpoint,
            settings: settings.clone(),
            patch_size: config.patch_size,
            feat_dim: config.feat_dim,
            request: None,
        })
    }

    /// Starts a request, ending any under way: runs `prompt`, which ends in audio, into the
    /// caches, to sample up to `max_patches` patches after it at `temperature` and guidance
    /// scale `cfg`, whose audio goes on from `voice`, the voice's latent frames.
    pub fn start(
        &mut self,
        prompt: Prompt<'_>,
        voice: &[f32],
        max_patches: usize,
        temperature: f32,
        cfg: f32,
    ) -> Result<(), Error> {
        self.request = None;
        let (len, capacity) = (prompt.len(), self.settings.capacity);
        let patch = self.patch_size * self.feat_dim;
        if len == 0 || max_patches == 0 || len - 1 + max_patches > capacity {
            return Err(Error::Shape(format!(
                "{len} positions of prompt and {max_patches} patches after it in a cache of \
                 {capacity}"
            )));
        }
        if !voice.len().is_multiple_of(self.feat_dim) || prompt.patches.len() != len * patch {
            return Err(Error::Shape(format!(
                "a voice of {} values and a prompt of {} values of patches in frames of {}",
                voice.len(),
                prompt.patches.len(),
                self.feat_dim
            )));
        }
        self.prompter
            .run(&mut self.step, &self.checkpoint, prompt)?;
        self.vae.reset()?;
        let context = (voice.len() / self.feat_dim).min(CONTEXT_FRAMES);
        let last = &prompt.patches[(len - 1) * patch..];
        self.request = Some(Request {
            position: len - 1,
            before: transposed(last, self.patch_size),
            left: max_patches,
            sampled: 0,
            temperature,
            cfg,
            frames: voice[voice.len() - context * self.feat_dim..].to_vec(),
            context,
            audio: Vec::new(),
        });
        Ok(())
    }

    /// Has the next patch sampled after `frames`, `[patch_size][feat_dim]`, rather than the
    /// patch sampled last: for checking a generation against another's, each patch sampled
    /// after the other's patch before, as teacher forcing does.
    pub fn sample_after(&mut self, frames: &[f32]) -> Result<(), Error> {
        let request = self.request.as_mut().ok_or_else(no_request)?;
        if frames.len() != request.before.len() {
            return Err(Error::Shape(format!("a patch of {} values", frames.len())));
        }
        request.before = transposed(frames, self.patch_size);
        Ok(())
    }

    /// Samples the request's next patch from `noise`, standard normal values
    /// `[feat_dim][patch_size]`, and decodes the frames waiting while a chunk's worth of them
    /// waits; none once the request has ended.
    pub fn next_patch(&mut self, noise: &[f32]) -> Result<Option<Patch>, Error> {
        let request = self.request.as_mut().ok_or_else(no_request)?;
        if request.left == 0 {
            return Ok(None);
        }
        if noise.len() != request.before.len() {
            return Err(Error::Shape(format!("noise of {} values", noise.len())));
        }
        let sampled = self.step.run(
            request.position,
            &request.before,
            noise,
            request.temperature,
            request.cfg,
        )?;
        let finite = |v: &[f32]| v.iter().all(|v| v.is_finite());
        if !finite(&sampled.patch) || !finite(&sampled.stop_logits) {
            request.left = 0;
            return Err(Error::NotFinite(request.sampled));
        }
        request.sampled += 1;
        request.left = if sampled.stops() { 0 } else { request.left - 1 };
        request.position += 1;
        let frames = transposed(&sampled.patch, self.feat_dim);
        request.frames.extend_from_slice(&frames);
        request.before = sampled.patch;
        let chunk = self.settings.chunk_frames * self.feat_dim;
        while request.frames.len() >= chunk {
            let rest = request.frames.split_off(chunk);
            let frames = mem::replace(&mut request.frames, rest);
            decode(&mut self.vae, request, &frames)?;
        }
        Ok(Some(Patch {
            frames,
            stop_logits: sampled.stop_logits,
            last: request.left == 0,
        }))
    }

    /// The audio decoded since it was last taken, mono samples at the audio VAE's rate.
    pub fn take_audio(&mut self) -> Vec<f32> {
        self.request
            .as_mut()
            .map_or_else(Vec::new, |request| mem::take(&mut request.audio))
    }

    /// Ends the request, decoding the frames still waiting, and gives the audio not yet taken.
    pub fn finish(&mut self) -> Result<Vec<f32>, Error> {
        let mut request = self.request.take().ok_or_else(no_request)?;
        let frames = mem::take(&mut request.frames);
        decode(&mut self.vae, &mut request, &frames)?;
        Ok(request.audio)
    }
}

fn no_request() -> Error {
    Error::Shape("no request under way".into())
}

/// `frames`, `[frames][feat_dim]`, through `vae`'s stream into `request`'s audio, the voice's
/// frames' audio left out.
fn decode(vae: &mut VaeDecoder, request: &mut Request, frames: &[f32]) -> Result<(), Error> {
    if frames.is_empty() {
        return Ok(());
    }
    let audio = vae.push(frames)?;
    let voice = request.context.min(frames.len() / vae.shape().latent);
    request.context -= voice;
    request
        .audio
        .extend_from_slice(&audio[voice * vae.shape().hop()..]);
    Ok(())
}
