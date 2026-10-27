//! A patch of VoxCPM2's generation at batch 1 as one recorded step: the local encoder embeds the
//! patch before, the LM side runs it through both LMs at the next position and gives the DiT's
//! condition and the stop logits, and the sampler samples the patch from noise. The host writes
//! the patch before, the noise and the position, and reads the patch and the stop logits.
//!
//! The stop logits come out beside the patch they were sampled with, as voxcpm's `Step` gives
//! them, so the submission that samples a patch says whether it is the last, and nothing runs
//! past it. The step for the first patch after a prompt that ends in audio runs the prompt's last
//! position too, its patch as the patch before: the prompt's other positions are what a prefill
//! leaves in the caches.

use vulkan::{Context, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode,
    graph::{Graph, Kind, TensorId},
    kernels::Kernels,
    lm_side::{LmSideGraph, LmSideShape, lm_side_graph, load_lm_side, splice},
    local::{
        self, PatchConfig,
        dit::{SamplerGraph, load_sampler, sampler_graph},
        encoder::{EncoderGraph, encoder_graph, load_encoder},
    },
    step::Step,
};

/// The patch step's graph, the graphs spliced into it with where each of their tensors went, and
/// the tensors the host writes and reads.
#[derive(Clone, Debug)]
pub struct PatchGraph {
    pub graph: Graph,
    pub encoder: EncoderGraph,
    pub encoder_ids: Vec<TensorId>,
    pub side: LmSideGraph,
    pub side_ids: Vec<TensorId>,
    pub sampler: SamplerGraph,
    pub sampler_ids: Vec<TensorId>,
    /// The patch before, `[feat_dim][patch_size]` in f32, as the sampler gives a patch.
    pub cond: TensorId,
    /// The position the patch before goes to in both LMs' caches, a u32.
    pub position: TensorId,
    /// The noise the patch is sampled from, laid out as `cond`.
    pub noise: TensorId,
    /// The temperature, then the guidance scale, in f32.
    pub sampling: TensorId,
    /// The patch, laid out as `cond`.
    pub patch: TensorId,
    /// The stop head's logits `[2]` f16, of going on and of stopping after the patch.
    pub stop_logits: TensorId,
}

/// The patch step of the LM side `side` and the local stacks of `config`, the sampler in
/// `timesteps` steps, the LM side's kernels cut as `lm` says and the local stacks' as `local`.
pub fn patch_graph(
    side: &LmSideShape,
    config: &PatchConfig,
    timesteps: usize,
    lm: &decode::Tuning,
    local: &local::Tuning,
) -> Result<PatchGraph, Error> {
    let encoder = encoder_graph(config, local)?;
    let side = lm_side_graph(side, lm)?;
    let sampler = sampler_graph(config, timesteps, local)?;
    let mut g = Graph::default();
    let bytes = |graph: &Graph, t: TensorId| graph.tensors[t.0].bytes;
    let cond = g.tensor("cond", bytes(&sampler.graph, sampler.cond), Kind::Input);
    let position = g.tensor("position", bytes(&side.graph, side.position), Kind::Param);
    let noise = g.tensor("noise", bytes(&sampler.graph, sampler.noise), Kind::Input);
    let sampling = g.tensor(
        "sampling",
        bytes(&sampler.graph, sampler.sampling),
        Kind::Param,
    );
    let patch = g.tensor("patch", bytes(&sampler.graph, sampler.patch), Kind::Output);
    let stop_logits = g.tensor(
        "stop_logits",
        bytes(&side.graph, side.stop_logits),
        Kind::Output,
    );
    let enc = g.tensor("enc", bytes(&side.graph, side.enc), Kind::Intermediate);
    let mu = g.tensor("mu", bytes(&side.graph, side.mu), Kind::Intermediate);
    let encoder_ids = splice(
        &mut g,
        &encoder.graph,
        "encoder.",
        &[(encoder.patch, cond), (encoder.out, enc)],
    );
    let side_ids = splice(
        &mut g,
        &side.graph,
        "",
        &[
            (side.enc, enc),
            (side.position, position),
            (side.mu, mu),
            (side.stop_logits, stop_logits),
        ],
    );
    let sampler_ids = splice(
        &mut g,
        &sampler.graph,
        "sampler.",
        &[
            (sampler.mu, mu),
            (sampler.cond, cond),
            (sampler.noise, noise),
            (sampler.sampling, sampling),
            (sampler.patch, patch),
        ],
    );
    Ok(PatchGraph {
        graph: g,
        encoder,
        encoder_ids,
        side,
        side_ids,
        sampler,
        sampler_ids,
        cond,
        position,
        noise,
        sampling,
        patch,
        stop_logits,
    })
}

/// What a patch step gives.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    /// The patch, `[feat_dim][patch_size]`.
    pub patch: Vec<f32>,
    /// The stop head's logits, of going on and of stopping after the patch.
    pub stop_logits: [f32; 2],
}

impl Sampled {
    /// Whether generation stops after this patch: the stop head's argmax, the first on a tie.
    pub fn stops(&self) -> bool {
        self.stop_logits[1] > self.stop_logits[0]
    }
}

/// VoxCPM2's patch step on the GPU, with its weights and both LMs' KV caches.
#[derive(Debug)]
pub struct PatchStep {
    step: Step,
    tensors: PatchGraph,
    capacity: usize,
}

impl PatchStep {
    /// Records `tensors`, a graph [`patch_graph`] gives, with the LMs' rotation tables written
    /// and every weight zero until [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        tensors: PatchGraph,
        timed: bool,
    ) -> Result<Self, Error> {
        let mut step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        let base = &tensors.side.shape.base;
        let (cos, sin) = base.rope_tables();
        let at = |t: TensorId| tensors.side_ids[t.0];
        step.write(at(tensors.side.cos), &cos)?;
        step.write(at(tensors.side.sin), &sin)?;
        Ok(PatchStep {
            capacity: base.capacity,
            step,
            tensors,
        })
    }

    /// Writes the weights of the LM side, the DiT and the local encoder from VoxCPM2's
    /// checkpoint, converted to f16, and the sampler's time tokens.
    pub fn load(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let t = &self.tensors;
        let (step, g) = (&mut self.step, &t.graph);
        load_encoder(step, g, &t.encoder, &t.encoder_ids, checkpoint)?;
        load_lm_side(step, g, &t.side, &t.side_ids, checkpoint)?;
        load_sampler(step, g, &t.sampler, &t.sampler_ids, checkpoint)
    }

    /// Runs `cond`, the patch before, through both LMs at `position`, writing their keys and
    /// values there, and samples the patch from `noise` at `temperature` and guidance scale
    /// `cfg`; `cond` and `noise` are laid out as the module's documentation says.
    pub fn run(
        &mut self,
        position: usize,
        cond: &[f32],
        noise: &[f32],
        temperature: f32,
        cfg: f32,
    ) -> Result<Sampled, Error> {
        if position >= self.capacity {
            return Err(Error::Shape(format!(
                "position {position} in a cache of {}",
                self.capacity
            )));
        }
        let t = &self.tensors;
        self.step.write(t.cond, cond)?;
        self.step.write(t.noise, noise)?;
        self.step.write(t.position, &[position as u32])?;
        self.step.write(t.sampling, &[temperature, cfg])?;
        self.step.submit()?;
        let logits: Vec<u16> = self.step.read(t.stop_logits)?;
        let logit = |i: usize| half::f16::from_bits(logits[i]).to_f32();
        Ok(Sampled {
            patch: self.step.read(t.patch)?,
            stop_logits: [logit(0), logit(1)],
        })
    }

    /// The positions the caches hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The step's tensors, and the step, for programs over its weights and caches to be added to
    /// it ([`Step::add`]) and run.
    pub fn parts_mut(&mut self) -> (&PatchGraph, &mut Step) {
        (&self.tensors, &mut self.step)
    }

    /// Each dispatch's label and GPU time in the last run, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    /// The recorded step's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        decode::{SUBGROUPS, Shape},
        graph::{barriers, plan},
        local::{DitConfig, StackConfig},
    };

    fn stack(layers: usize, rope: bool) -> Shape {
        Shape {
            hidden: 2048,
            inner: 6144,
            heads: 16,
            kv_heads: 2,
            head_dim: 128,
            layers,
            capacity: 256,
            eps: 1e-5,
            rope_theta: 10_000.0,
            rope_factors: vec![1.0; 64],
            rope,
        }
    }

    fn side() -> LmSideShape {
        LmSideShape {
            base: stack(2, true),
            residual: stack(1, false),
            encoder: 1024,
            dit: 1024,
            latent: 512,
            levels: 9,
        }
    }

    fn config() -> PatchConfig {
        let local = StackConfig {
            hidden_dim: 1024,
            ffn_dim: 4096,
            num_heads: 16,
            num_layers: 2,
            kv_channels: Some(128),
        };
        let lm = serde_json::json!({
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "kv_channels": 128,
            "max_position_embeddings": 32768,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": vec![1.0; 64],
                "original_max_position_embeddings": 32768,
            },
            "use_mup": false,
        });
        PatchConfig {
            lm_config: serde_json::from_value(lm).unwrap(),
            patch_size: 4,
            feat_dim: 64,
            encoder_config: local.clone(),
            dit_config: DitConfig {
                stack: local,
                mean_mode: false,
                dit_mean_mode: false,
            },
        }
    }

    /// A patch step of two layers of the base LM, one of the residual LM and two of each local
    /// stack, at VoxCPM2's widths.
    pub(crate) fn graph(subgroup: u32) -> PatchGraph {
        let lm = decode::Tuning::for_subgroup(subgroup);
        let local = local::Tuning::for_subgroup(subgroup);
        patch_graph(&side(), &config(), 10, &lm, &local).unwrap()
    }

    /// The ops that bind `t`, by index, and whether each writes it.
    fn uses(g: &Graph, t: TensorId) -> Vec<(usize, bool)> {
        let mut uses = Vec::new();
        for (i, op) in g.ops.iter().enumerate() {
            let access = op.pipeline.kernel.access.iter();
            for (&bound, access) in op.bindings.iter().zip(access) {
                if bound == t {
                    uses.push((i, *access == crate::kernels::Access::Write));
                }
            }
        }
        uses
    }

    #[test]
    fn the_encoder_feeds_the_lm_side_which_conditions_the_sampler() {
        for subgroup in SUBGROUPS {
            let p = graph(subgroup);
            let g = &p.graph;
            let count = |prefix: &str| {
                g.ops
                    .iter()
                    .filter(|op| op.label.starts_with(prefix))
                    .count()
            };
            let (encoder, sampler) = (count("encoder."), count("sampler."));
            assert_eq!(encoder, p.encoder.graph.ops.len());
            assert_eq!(sampler, p.sampler.graph.ops.len());
            assert_eq!(g.ops.len(), encoder + p.side.graph.ops.len() + sampler);
            assert!(
                g.ops[..encoder]
                    .iter()
                    .all(|op| op.label.starts_with("encoder."))
            );
            assert!(
                g.ops[g.ops.len() - sampler..]
                    .iter()
                    .all(|op| op.label.starts_with("sampler."))
            );
            // The patch before is read by the encoder's first op and the sampler's condition, and
            // written by the host alone; the patch is written by the sampler alone.
            let cond = uses(g, p.cond);
            assert!(cond.iter().all(|&(_, writes)| !writes), "{cond:?}");
            assert_eq!(cond.first().map(|u| u.0), Some(0));
            assert!(cond.iter().any(|&(i, _)| i >= g.ops.len() - sampler));
            let patch = uses(g, p.patch);
            assert!(patch.iter().all(|&(i, _)| i >= g.ops.len() - sampler));
            // The condition is the LM side's, ready before the sampler reads it.
            let mu = p.side_ids[p.side.mu.0];
            assert_eq!(mu, p.sampler_ids[p.sampler.mu.0]);
            let mu_uses = uses(g, mu);
            let last_write = mu_uses.iter().filter(|u| u.1).map(|u| u.0).max().unwrap();
            let first_read = mu_uses.iter().filter(|u| !u.1).map(|u| u.0).min().unwrap();
            assert!(last_write < first_read, "{mu_uses:?}");
            let needs_barrier = barriers(g, &plan(g));
            assert!(needs_barrier[last_write + 1..=first_read].contains(&true));
            let enc = p.encoder_ids[p.encoder.out.0];
            assert_eq!(enc, p.side_ids[p.side.enc.0]);
            assert_eq!(g.tensors[enc.0].kind, Kind::Intermediate);
        }
    }
}
