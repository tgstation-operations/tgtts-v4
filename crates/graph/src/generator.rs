//! VoxCPM2's text-to-speech on the GPU, from a prompt to audio, several requests at once, each in
//! a slot of its own. A request is admitted between two steps: its prompt run into its slot of the
//! patch step's caches ([`prompt`](crate::prompt)) from what a prompt cache keeps of its voice
//! ([`PromptCache`]). Each step samples a patch of every request under way in one submission
//! ([`patch`](crate::patch)), on the program of the fewest rows that holds them, until the stop
//! head says to stop after one or the request's most are sampled, as the service stops. A
//! request's latent frames are decoded by the audio VAE ([`vae`](crate::vae)) in a stream of its
//! own after the voice's last frames, as the service primes its decoder, in chunks of a fixed
//! number of frames as they come. A request comes out bit for bit as it does alone, whatever
//! runs beside it.

use std::{fs, iter, mem, path::Path};

use half::f16;
use serde::Deserialize;
use vulkan::{Context, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{self, Batch, Row},
    graph::Kind,
    kernels::Kernels,
    lm_side,
    local::{self, PatchConfig},
    patch::{PatchRow, PatchStep, Sampled, patch_batch_graph, transposed},
    prompt::{Prompt, PromptCache, Prompter},
    vae::{Parameter, VaeConfig, VaeDecoder, VaeShape},
};

/// The voice's last frames the audio VAE is primed with before a speech, nanovllm-voxcpm's
/// `n_decode_pad_frames`.
const CONTEXT_FRAMES: usize = 12;

/// What the generator is recorded for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The positions each slot of the LMs' caches holds, a multiple of 32: a request's prompt and
    /// its patches.
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
    /// The bytes the prompt cache keeps of the voices.
    pub prompt_cache_bytes: usize,
    /// The subgroup size the kernels run at, the device's widest they are written for when
    /// `None`.
    pub subgroup: Option<u32>,
    /// The requests it generates at once.
    pub slots: usize,
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

/// What a step gave a request: its patch, or why it has none, after which it has ended.
pub type Outcome = Result<Patch, Error>;

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

/// VoxCPM2 on the GPU, loaded once and generating requests in its slots.
#[derive(Debug)]
pub struct Generator {
    step: PatchStep,
    prompter: Prompter,
    cache: PromptCache,
    vae: VaeDecoder,
    checkpoint: Checkpoint,
    settings: Settings,
    patch_size: usize,
    feat_dim: usize,
    /// The request in each slot.
    requests: Vec<Option<Request>>,
    /// The row each slot's request ran in at the last step, none for one that did not run there.
    rows: Vec<Option<usize>>,
}

/// The rows of the patch step's programs for `slots` slots: each power of two below it, then
/// `slots`.
fn program_rows(slots: usize) -> Vec<usize> {
    let powers = iter::successors(Some(1), |rows| Some(rows * 2));
    powers
        .take_while(|&rows| rows < slots)
        .chain([slots])
        .collect()
}

impl Generator {
    /// Records the generator `settings` describe for the checkpoint in `dir`, its kernels cut as
    /// the device suits, and loads the weights: the model's from its `model.safetensors`, the
    /// audio VAE decoder's as `parameter` gives each by its name under the decoder, weight norm
    /// folded, as [`Pth::floats`](crate::pth::Pth::floats) reads them from its `audiovae.pth`.
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
        let subgroup = match settings.subgroup {
            Some(subgroup) => subgroup,
            None => decode::Tuning::for_device(report)?.subgroup,
        };
        let lm = decode::Tuning::for_subgroup(subgroup);
        let local = local::Tuning::for_subgroup(subgroup).fit(report)?;
        let shape = lm_side::LmSideShape::new(&side_config, settings.capacity)?;
        let slots = settings.slots;
        let graphs = program_rows(slots)
            .into_iter()
            .map(|rows| {
                let lm = decode::Tuning::for_batch(subgroup, rows);
                let batch = Batch { rows, slots };
                let mut graph =
                    patch_batch_graph(&shape, &config, settings.timesteps, &lm, &local, batch)?;
                // For the host to read, out of the arena, whose later ops would reuse its bytes.
                let levels = graph.side_ids[graph.side.levels.0];
                graph.graph.tensors[levels.0].kind = Kind::Output;
                Ok(graph)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut step = PatchStep::with_rows(ctx, kernels, graphs, settings.timed)?;
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
        let vae_shape = VaeShape::new(&audio_vae_config, settings.chunk_frames, slots)?;
        let mut vae = VaeDecoder::new(ctx, kernels, &vae_shape, settings.timed)?;
        vae.load(parameter)?;
        Ok(Generator {
            step,
            prompter,
            cache: PromptCache::new(ctx, settings.prompt_cache_bytes),
            vae,
            checkpoint,
            settings: settings.clone(),
            patch_size: config.patch_size,
            feat_dim: config.feat_dim,
            requests: (0..slots).map(|_| None).collect(),
            rows: vec![None; slots],
        })
    }

    /// The requests it generates at once.
    pub fn slots(&self) -> usize {
        self.requests.len()
    }

    /// Starts a request in slot `slot`, ending any under way there: runs `prompt`, which ends in
    /// audio, into the slot of the caches, to sample up to `max_patches` patches after it at
    /// `temperature` and guidance scale `cfg`, whose audio goes on from `voice`, the voice's
    /// latent frames. The request's first patch is sampled at the next [`step`](Self::step).
    /// Gives the positions of the prompt that run through the LMs, the rest coming from what the
    /// prompt cache keeps of the voice.
    pub fn start(
        &mut self,
        slot: usize,
        prompt: Prompt<'_>,
        voice: &[f32],
        max_patches: usize,
        temperature: f32,
        cfg: f32,
    ) -> Result<usize, Error> {
        let held = self
            .requests
            .get_mut(slot)
            .ok_or_else(|| Error::Shape(format!("slot {slot} of {}", self.settings.slots)))?;
        *held = None;
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
        let prefilled = self.prompter.run_cached(
            &mut self.step,
            slot,
            &self.checkpoint,
            prompt,
            voice,
            &mut self.cache,
        )?;
        self.vae.reset(slot)?;
        self.rows[slot] = None;
        let context = (voice.len() / self.feat_dim).min(CONTEXT_FRAMES);
        let last = &prompt.patches[(len - 1) * patch..];
        self.requests[slot] = Some(Request {
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
        Ok(prefilled)
    }

    /// The request in `slot`.
    fn request(&mut self, slot: usize) -> Result<&mut Request, Error> {
        let request = self.requests.get_mut(slot).and_then(Option::as_mut);
        request.ok_or_else(|| no_request(slot))
    }

    /// Has the next patch of the request in `slot` sampled after `frames`,
    /// `[patch_size][feat_dim]`, rather than the patch sampled last: for checking a generation
    /// against another's, each patch sampled after the other's patch before, as teacher forcing
    /// does.
    pub fn sample_after(&mut self, slot: usize, frames: &[f32]) -> Result<(), Error> {
        let patch_size = self.patch_size;
        let request = self.request(slot)?;
        if frames.len() != request.before.len() {
            return Err(Error::Shape(format!("a patch of {} values", frames.len())));
        }
        request.before = transposed(frames, patch_size);
        Ok(())
    }

    /// Samples the next patch of each request under way, all in one submission, each from the
    /// noise `noise` gives for its slot, standard normal values `[feat_dim][patch_size]`; and
    /// decodes each one's frames waiting while a chunk's worth of them waits. Gives what it gave
    /// each, by its slot. A request whose last patch has been sampled is no longer under way, and
    /// waits in its slot to be [`finish`](Self::finish)ed.
    pub fn step(
        &mut self,
        mut noise: impl FnMut(usize) -> Vec<f32>,
    ) -> Result<Vec<(usize, Outcome)>, Error> {
        let under_way: Vec<usize> = (0..self.requests.len())
            .filter(|&slot| self.requests[slot].as_ref().is_some_and(|r| r.left > 0))
            .collect();
        if under_way.is_empty() {
            return Ok(Vec::new());
        }
        let noises: Vec<Vec<f32>> = under_way.iter().map(|&slot| noise(slot)).collect();
        let rows: Vec<Option<PatchRow<'_>>> = under_way
            .iter()
            .zip(&noises)
            .map(|(&slot, noise)| {
                let request = self.requests[slot].as_ref().expect("under way");
                Some(PatchRow {
                    at: Row {
                        position: request.position,
                        slot,
                    },
                    cond: &request.before,
                    noise,
                    temperature: request.temperature,
                    cfg: request.cfg,
                })
            })
            .collect();
        let sampled = self.step.run_rows(&rows)?;
        self.rows.fill(None);
        for (row, &slot) in under_way.iter().enumerate() {
            self.rows[slot] = Some(row);
        }
        let chunk = self.settings.chunk_frames * self.feat_dim;
        let mut patches = Vec::with_capacity(under_way.len());
        for (slot, sampled) in under_way.into_iter().zip(sampled) {
            let request = self.requests[slot].as_mut().expect("under way");
            let sampled = sampled.expect("a row that runs");
            let patch = advance(&mut self.vae, slot, request, sampled, self.feat_dim, chunk);
            if patch.is_err() {
                request.left = 0;
            }
            patches.push((slot, patch));
        }
        Ok(patches)
    }

    /// Decodes the frames waiting of the request in `slot` now rather than once a chunk's worth
    /// of them waits: the audio sooner, for a submission of the audio VAE on fewer frames than
    /// it could take.
    pub fn decode_waiting(&mut self, slot: usize) -> Result<(), Error> {
        let request = self.requests.get_mut(slot).and_then(Option::as_mut);
        let request = request.ok_or_else(|| no_request(slot))?;
        let frames = mem::take(&mut request.frames);
        decode(&mut self.vae, slot, request, &frames)
    }

    /// The scalar quantizer's levels the last patch of the request in `slot` was sampled after,
    /// integers: for checking a generation against another's, as
    /// [`sample_after`](Self::sample_after) is. Generating reads them nowhere else, yet each step
    /// writes them into memory the host maps and reads them back from there, 1 KiB a row: they
    /// are an output of each program of the patch step, as a program's state would be the
    /// others' too.
    pub fn levels(&self, slot: usize) -> Result<Vec<f32>, Error> {
        let ran = self.requests.get(slot).and_then(Option::as_ref);
        let row = ran
            .and_then(|_| self.rows[slot])
            .ok_or_else(|| no_request(slot))?;
        let graph = self.step.last_run();
        let tensor = graph.side_ids[graph.side.levels.0];
        let halves: Vec<u16> = self.step.graph_step().read(tensor)?;
        let shape = &graph.side.shape;
        let unit = shape.levels as f32;
        let row = &halves[row * shape.latent..(row + 1) * shape.latent];
        Ok(row
            .iter()
            .map(|&h| (f16::from_bits(h).to_f32() * unit).round())
            .collect())
    }

    /// What the generator keeps of the voices.
    pub fn prompt_cache(&self) -> &PromptCache {
        &self.cache
    }

    /// The audio of the request in `slot` decoded since it was last taken, mono samples at the
    /// audio VAE's rate.
    pub fn take_audio(&mut self, slot: usize) -> Vec<f32> {
        let request = self.requests.get_mut(slot).and_then(Option::as_mut);
        request.map_or_else(Vec::new, |request| mem::take(&mut request.audio))
    }

    /// Each dispatch's label and GPU time in the last run over the LMs' caches, when recorded
    /// timed: the prompt's last run after [`start`](Self::start), the patch step's after
    /// [`step`](Self::step).
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    /// The GPU time of the last run over the LMs' caches, as [`timings`](Self::timings) says
    /// which, from before its first dispatch to after its last.
    pub fn elapsed(&self) -> Result<Timing, Error> {
        self.step.graph_step().elapsed()
    }

    /// Each dispatch's label and GPU time in the audio VAE's last chunk, when recorded timed.
    pub fn audio_timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.vae.timings()
    }

    /// Ends the request in `slot`, decoding the frames still waiting, and gives the audio not yet
    /// taken.
    pub fn finish(&mut self, slot: usize) -> Result<Vec<f32>, Error> {
        let request = self.requests.get_mut(slot).and_then(Option::take);
        let mut request = request.ok_or_else(|| no_request(slot))?;
        let frames = mem::take(&mut request.frames);
        decode(&mut self.vae, slot, &mut request, &frames)?;
        Ok(request.audio)
    }
}

fn no_request(slot: usize) -> Error {
    Error::Shape(format!("no request under way in slot {slot}"))
}

/// Takes `sampled`, the patch step's for the request in `slot`, as the request's next patch,
/// and decodes its frames waiting while a chunk's worth of them, `chunk` values, waits.
fn advance(
    vae: &mut VaeDecoder,
    slot: usize,
    request: &mut Request,
    sampled: Sampled,
    feat_dim: usize,
    chunk: usize,
) -> Result<Patch, Error> {
    let finite = |v: &[f32]| v.iter().all(|v| v.is_finite());
    if !finite(&sampled.patch) || !finite(&sampled.stop_logits) {
        return Err(Error::NotFinite(request.sampled));
    }
    request.sampled += 1;
    request.left = if sampled.stops() { 0 } else { request.left - 1 };
    request.position += 1;
    let frames = transposed(&sampled.patch, feat_dim);
    request.frames.extend_from_slice(&frames);
    request.before = sampled.patch;
    while request.frames.len() >= chunk {
        let rest = request.frames.split_off(chunk);
        let frames = mem::replace(&mut request.frames, rest);
        decode(vae, slot, request, &frames)?;
    }
    Ok(Patch {
        frames,
        stop_logits: sampled.stop_logits,
        last: request.left == 0,
    })
}

/// `frames`, `[frames][feat_dim]`, through `vae`'s stream `slot` into `request`'s audio, the
/// voice's frames' audio left out.
fn decode(
    vae: &mut VaeDecoder,
    slot: usize,
    request: &mut Request,
    frames: &[f32],
) -> Result<(), Error> {
    if frames.is_empty() {
        return Ok(());
    }
    let audio = vae.push(slot, frames)?;
    let voice = request.context.min(frames.len() / vae.shape().latent);
    request.context -= voice;
    request
        .audio
        .extend_from_slice(&audio[voice * vae.shape().hop()..]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_programs_are_the_powers_of_two_below_the_slots_and_the_slots() {
        assert_eq!(program_rows(1), [1]);
        assert_eq!(program_rows(2), [1, 2]);
        assert_eq!(program_rows(12), [1, 2, 4, 8, 12]);
        assert_eq!(program_rows(16), [1, 2, 4, 8, 16]);
    }
}
