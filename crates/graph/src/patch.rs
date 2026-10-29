//! A patch of VoxCPM2's generation as one recorded step: the local encoder embeds the patch
//! before, the LM side runs it through both LMs at the next position and gives the DiT's
//! condition and the stop logits, and the sampler samples the patch from noise. The host writes
//! the patch before, the noise and the position, and reads the patch and the stop logits.
//!
//! The stop logits come out beside the patch they were sampled with, as voxcpm's `Step` gives
//! them, so the submission that samples a patch says whether it is the last, and nothing runs
//! past it. The step for the first patch after a prompt that ends in audio runs the prompt's last
//! position too, its patch as the patch before: the prompt's other positions are what a prefill
//! leaves in the caches.
//!
//! A step runs a [`Batch`] of rows, each a request at its own position in its own slot of both
//! LMs' caches, a row of each tensor the host writes and reads; a row comes out bit for bit as it
//! does alone. A [`PatchStep`] holds a program for each of several numbers of rows over one set of
//! weights and caches, and runs the fewest rows that hold the requests under way.

use std::mem;

use vulkan::{Context, F16, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{self, Batch, Row},
    graph::{Graph, Kind, TensorId},
    kernels::Kernels,
    lm_side::{LmSideGraph, LmSideShape, lm_side_batch_graph, load_lm_side, splice},
    local::{
        self, PatchConfig,
        dit::{SamplerGraph, load_sampler, sampler_graph},
        encoder::{EncoderGraph, encoder_graph, load_encoder},
    },
    step::{Program, Step},
};

/// The patch step's graph, the graphs spliced into it with where each of their tensors went, and
/// the tensors the host writes and reads, a row of each for every row of the batch.
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
    /// Where the patch before goes in both LMs' caches, [`Batch::words`] u32: at batch 1, the one
    /// position.
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

impl PatchGraph {
    pub fn batch(&self) -> Batch {
        self.side.batch
    }

    /// The graph with each of its tensors where `ids` puts it, in `graph`.
    fn moved(self, graph: Graph, ids: &[TensorId]) -> Self {
        let at = |t: TensorId| ids[t.0];
        let all = |ids: Vec<TensorId>| ids.into_iter().map(at).collect();
        PatchGraph {
            graph,
            encoder_ids: all(self.encoder_ids),
            side_ids: all(self.side_ids),
            sampler_ids: all(self.sampler_ids),
            cond: at(self.cond),
            position: at(self.position),
            noise: at(self.noise),
            sampling: at(self.sampling),
            patch: at(self.patch),
            stop_logits: at(self.stop_logits),
            ..self
        }
    }
}

/// The patch step of the LM side `side` and the local stacks of `config` at batch 1, the sampler
/// in `timesteps` steps, the LM side's kernels cut as `lm` says and the local stacks' as `local`.
pub fn patch_graph(
    side: &LmSideShape,
    config: &PatchConfig,
    timesteps: usize,
    lm: &decode::Tuning,
    local: &local::Tuning,
) -> Result<PatchGraph, Error> {
    patch_batch_graph(side, config, timesteps, lm, local, Batch::ONE)
}

/// The patch step of [`patch_graph`] for `batch`.
pub fn patch_batch_graph(
    side: &LmSideShape,
    config: &PatchConfig,
    timesteps: usize,
    lm: &decode::Tuning,
    local: &local::Tuning,
    batch: Batch,
) -> Result<PatchGraph, Error> {
    let encoder = encoder_graph(config, batch.rows, local)?;
    let side = lm_side_batch_graph(side, lm, batch)?;
    let sampler = sampler_graph(config, timesteps, batch.rows, local)?;
    let mut g = Graph::default();
    let bytes = |graph: &Graph, t: TensorId| graph.tensors[t.0].bytes;
    let cond = g.untyped("cond", bytes(&sampler.graph, sampler.cond), Kind::Input);
    let position = g.untyped("position", bytes(&side.graph, side.position), Kind::Param);
    let noise = g.untyped("noise", bytes(&sampler.graph, sampler.noise), Kind::Input);
    let sampling = g.untyped(
        "sampling",
        bytes(&sampler.graph, sampler.sampling),
        Kind::Param,
    );
    let patch = g.untyped("patch", bytes(&sampler.graph, sampler.patch), Kind::Output);
    let stop_logits = g.untyped(
        "stop_logits",
        bytes(&side.graph, side.stop_logits),
        Kind::Output,
    );
    let enc = g.untyped("enc", bytes(&side.graph, side.enc), Kind::Intermediate);
    let mu = g.untyped("mu", bytes(&side.graph, side.mu), Kind::Intermediate);
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

/// `rest` as programs over `first`'s weights and caches, and `first` as one beside them: each
/// graph over `first`'s tensors and then the others' own in turn, a program's constants and state
/// taken as `first`'s, which it must have alike and in the same order, and the rest its own.
fn shared(first: PatchGraph, rest: Vec<PatchGraph>) -> Result<Vec<PatchGraph>, Error> {
    let kept = |g: &Graph| -> Vec<TensorId> {
        let tensors = g.tensors.iter().enumerate();
        let kept = tensors.filter(|(_, t)| matches!(t.kind, Kind::Constant | Kind::State));
        kept.map(|(i, _)| TensorId(i)).collect()
    };
    let weights = kept(&first.graph);
    let mut tensors = Graph {
        tensors: first.graph.tensors.clone(),
        ops: Vec::new(),
    };
    let mut programs = Vec::with_capacity(rest.len());
    for p in rest {
        let theirs = kept(&p.graph);
        let alike = |(a, b): (&TensorId, &TensorId)| {
            let (a, b) = (&p.graph.tensors[a.0], &tensors.tensors[b.0]);
            (&a.name, a.bytes, a.kind) == (&b.name, b.bytes, b.kind)
        };
        if theirs.len() != weights.len() || !theirs.iter().zip(&weights).all(alike) {
            return Err(Error::Shape(format!(
                "a program of {} rows over other weights or caches",
                p.batch().rows
            )));
        }
        let pairs: Vec<(TensorId, TensorId)> = theirs.into_iter().zip(weights.clone()).collect();
        let prefix = format!("rows{}.", p.batch().rows);
        let ids = splice(&mut tensors, &p.graph, &prefix, &pairs);
        let ops = mem::take(&mut tensors.ops);
        programs.push((p, ids, ops));
    }
    let over = |ops: Vec<_>| Graph {
        tensors: tensors.tensors.clone(),
        ops,
    };
    let ids: Vec<TensorId> = (0..first.graph.tensors.len()).map(TensorId).collect();
    let ops = first.graph.ops.clone();
    let mut all = vec![first.moved(over(ops), &ids)];
    for (p, ids, ops) in programs {
        all.push(p.moved(over(ops), &ids));
    }
    Ok(all)
}

/// `[rows][columns]` as `[columns][rows]`: a patch's frames, `[patch_size][feat_dim]`, laid out
/// as the patch step takes a patch when `rows` is `patch_size`, and back when it is `feat_dim`.
pub(crate) fn transposed(values: &[f32], rows: usize) -> Vec<f32> {
    let columns = values.len() / rows;
    (0..columns)
        .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
        .collect()
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

/// A request's row of a patch step: where it runs, and what its patch is sampled from.
#[derive(Clone, Copy, Debug)]
pub struct PatchRow<'a> {
    /// The position its patch before goes to, in its slot of the caches.
    pub at: Row,
    /// The patch before and the noise, laid out as the module's documentation says.
    pub cond: &'a [f32],
    pub noise: &'a [f32],
    pub temperature: f32,
    /// The guidance scale.
    pub cfg: f32,
}

/// VoxCPM2's patch step on the GPU, with its weights and both LMs' KV caches: a program for each
/// of several numbers of rows over them.
#[derive(Debug)]
pub struct PatchStep {
    step: Step,
    /// Fewest rows first, each graph over every tensor the step holds.
    programs: Vec<(Program, PatchGraph)>,
    /// The program [`run_rows`](Self::run_rows) ran last.
    last: usize,
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
        Self::with_rows(ctx, kernels, vec![tensors], timed)
    }

    /// Records `graphs`, each a graph [`patch_batch_graph`] gives for a number of rows of its
    /// own over caches of as many slots as the others', as programs over one set of weights and
    /// caches, as [`new`](Self::new) records one.
    pub fn with_rows(
        ctx: &Context,
        kernels: &mut Kernels,
        mut graphs: Vec<PatchGraph>,
        timed: bool,
    ) -> Result<Self, Error> {
        graphs.sort_by_key(|g| g.batch().rows);
        let rows: Vec<usize> = graphs.iter().map(|g| g.batch().rows).collect();
        let slots = graphs.first().map(|g| g.batch().slots);
        if slots.is_none()
            || rows.windows(2).any(|w| w[0] == w[1])
            || graphs.iter().any(|g| Some(g.batch().slots) != slots)
        {
            return Err(Error::Shape(format!(
                "programs of {rows:?} rows over caches of {:?} slots",
                graphs.iter().map(|g| g.batch().slots).collect::<Vec<_>>()
            )));
        }
        let rest = graphs.split_off(1);
        let first = graphs.remove(0);
        let mut step = Step::new(ctx, kernels, &first.graph, timed)?;
        let all = shared(first, rest)?;
        let added: Vec<&Graph> = all[1..].iter().map(|p| &p.graph).collect();
        let programs = step.add(ctx, kernels, &added, timed)?;
        let programs: Vec<(Program, PatchGraph)> = std::iter::once(step.first())
            .chain(programs)
            .zip(all)
            .collect();
        let tensors = &programs[0].1;
        let base = &tensors.side.shape.base;
        let (cos, sin) = base.rope_tables();
        let at = |t: TensorId| tensors.side_ids[t.0];
        step.write(at(tensors.side.cos), &cos)?;
        step.write(at(tensors.side.sin), &sin)?;
        Ok(PatchStep {
            capacity: base.capacity,
            step,
            programs,
            last: 0,
        })
    }

    /// Writes the weights of the LM side, the DiT and the local encoder from VoxCPM2's
    /// checkpoint, converted to f16, and the sampler's time tokens.
    pub fn load(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let t = &self.programs[0].1;
        let (step, g) = (&mut self.step, &t.graph);
        load_encoder(step, g, &t.encoder, &t.encoder_ids, checkpoint)?;
        load_lm_side(step, g, &t.side, &t.side_ids, checkpoint)?;
        load_sampler(step, g, &t.sampler, &t.sampler_ids, checkpoint)
    }

    /// Runs `cond`, the patch before, through both LMs at `position` of the caches' first slot,
    /// writing their keys and values there, and samples the patch from `noise` at `temperature`
    /// and guidance scale `cfg`; `cond` and `noise` are laid out as the module's documentation
    /// says.
    pub fn run(
        &mut self,
        position: usize,
        cond: &[f32],
        noise: &[f32],
        temperature: f32,
        cfg: f32,
    ) -> Result<Sampled, Error> {
        let row = PatchRow {
            at: Row { position, slot: 0 },
            cond,
            noise,
            temperature,
            cfg,
        };
        let mut sampled = self.run_rows(&[Some(row)])?;
        Ok(sampled.remove(0).expect("a row that runs"))
    }

    /// Runs `rows`, each a request's or spare, as [`run`](Self::run) runs one, each in its own
    /// slot: on the program of the fewest rows that holds them, the rows past them spare. Gives
    /// what each row sampled, none for a spare row.
    pub fn run_rows(
        &mut self,
        rows: &[Option<PatchRow<'_>>],
    ) -> Result<Vec<Option<Sampled>>, Error> {
        let Some(index) = self
            .programs
            .iter()
            .position(|(_, t)| t.batch().rows >= rows.len())
        else {
            return Err(Error::Shape(format!(
                "{} rows, of at most {}",
                rows.len(),
                self.rows().last().copied().unwrap_or(0)
            )));
        };
        let (program, t) = &self.programs[index];
        let batch = t.batch();
        let len = t.sampler.feat_dim * t.sampler.patch_size;
        let mut placed = vec![None; batch.rows];
        let mut cond = vec![0.0; batch.rows * len];
        let mut noise = vec![0.0; batch.rows * len];
        let mut sampling = vec![0.0; 2 * batch.rows];
        for (b, row) in rows.iter().enumerate() {
            let Some(row) = row else {
                continue;
            };
            if row.cond.len() != len || row.noise.len() != len {
                return Err(Error::Shape(format!(
                    "row {b}: a patch before of {} values and noise of {}, of {len}",
                    row.cond.len(),
                    row.noise.len()
                )));
            }
            placed[b] = Some(row.at);
            cond[b * len..(b + 1) * len].copy_from_slice(row.cond);
            noise[b * len..(b + 1) * len].copy_from_slice(row.noise);
            sampling[2 * b..2 * b + 2].copy_from_slice(&[row.temperature, row.cfg]);
        }
        let words = batch.placement(&placed, self.capacity)?;
        self.step.write(t.cond, &cond)?;
        self.step.write(t.noise, &noise)?;
        self.step.write(t.position, &words)?;
        self.step.write(t.sampling, &sampling)?;
        self.step.run(*program)?;
        self.last = index;
        let logits: Vec<F16> = self.step.read(t.stop_logits)?;
        let patches: Vec<f32> = self.step.read(t.patch)?;
        let logit = |i: usize| logits[i].to_f32();
        Ok(rows
            .iter()
            .enumerate()
            .map(|(b, row)| {
                row.map(|_| Sampled {
                    patch: patches[b * len..(b + 1) * len].to_vec(),
                    stop_logits: [logit(2 * b), logit(2 * b + 1)],
                })
            })
            .collect())
    }

    /// The positions each slot of the caches holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The slots of the caches.
    pub fn slots(&self) -> usize {
        self.programs[0].1.batch().slots
    }

    /// The numbers of rows it has a program for, fewest first.
    pub fn rows(&self) -> Vec<usize> {
        self.programs.iter().map(|(_, t)| t.batch().rows).collect()
    }

    /// The tensors of the program [`run_rows`](Self::run_rows) ran last, over every tensor of the
    /// step.
    pub fn last_run(&self) -> &PatchGraph {
        &self.programs[self.last].1
    }

    /// The tensors of the program of the fewest rows, over every tensor of the step, and the
    /// step, for programs over its weights and caches to be added to it ([`Step::add`]) and run.
    pub fn parts_mut(&mut self) -> (&PatchGraph, &mut Step) {
        (&self.programs[0].1, &mut self.step)
    }

    /// Each dispatch's label and GPU time in the last run, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    /// The recorded step's dispatches, barriers and arena: those of the program of the fewest
    /// rows.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

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
            let access = op.pipeline.kernel.bindings.iter().map(|b| b.access);
            for (&bound, access) in op.bindings.iter().zip(access) {
                if bound == t {
                    uses.push((i, access == crate::kernels::Access::Write));
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

    /// [`graph`] for `rows` rows over caches of `slots` slots, the LM side cut as a batch of as
    /// many rows is.
    pub(crate) fn batch_graph(subgroup: u32, rows: usize, slots: usize) -> PatchGraph {
        let lm = decode::Tuning::for_batch(subgroup, rows);
        let local = local::Tuning::for_subgroup(subgroup);
        let batch = Batch { rows, slots };
        patch_batch_graph(&side(), &config(), 10, &lm, &local, batch).unwrap()
    }

    #[test]
    fn programs_share_the_weights_and_caches_and_keep_their_rows_to_themselves() {
        let rows = [1, 4, 16];
        let graphs: Vec<PatchGraph> = rows.iter().map(|&r| batch_graph(32, r, 16)).collect();
        let held = graphs[0].graph.tensors.len();
        let all = shared(graphs[0].clone(), graphs[1..].to_vec()).unwrap();
        let tensors = &all[0].graph.tensors;
        let at = |p: &PatchGraph, t: TensorId| p.side_ids[t.0];
        let mut owner: HashMap<TensorId, usize> = HashMap::new();
        for (i, p) in all.iter().enumerate() {
            assert_eq!(p.batch().rows, rows[i]);
            assert_eq!(p.graph.tensors.len(), tensors.len());
            for op in &p.graph.ops {
                for &t in &op.bindings {
                    let name = &tensors[t.0].name;
                    if matches!(tensors[t.0].kind, Kind::Constant | Kind::State) {
                        assert!(
                            t.0 < held,
                            "{} of {} rows: its own {name}",
                            op.label,
                            rows[i]
                        );
                        continue;
                    }
                    let first = *owner.entry(t).or_insert(i);
                    assert_eq!(first, i, "{} of {} rows: {name}", op.label, rows[i]);
                    assert_eq!(
                        t.0 < held,
                        i == 0,
                        "{} of {} rows: {name}",
                        op.label,
                        rows[i]
                    );
                }
            }
            let bytes = |t: TensorId| tensors[t.0].bytes;
            assert_eq!(bytes(p.cond), (rows[i] * 64 * 4 * 4) as u64);
            assert_eq!(bytes(p.position), (2 * rows[i] * 4) as u64);
            let (head, cache) = (p.side.heads.stop_head.weight, p.side.base.layers[0].k_cache);
            let first = &all[0];
            assert_eq!(at(p, head), at(first, first.side.heads.stop_head.weight));
            assert_eq!(at(p, cache), at(first, first.side.base.layers[0].k_cache));
        }
        let labels = |g: &Graph| g.ops.iter().map(|op| op.label.clone()).collect::<Vec<_>>();
        assert_eq!(labels(&all[0].graph), labels(&graphs[0].graph));
        // A program over caches of other slots is refused.
        let other = batch_graph(32, 4, 8);
        assert!(matches!(
            shared(graphs[0].clone(), vec![other]),
            Err(Error::Shape(_))
        ));
    }
}
