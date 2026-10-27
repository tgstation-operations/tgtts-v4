//! A prompt run into the patch step's caches at batch 1, as voxcpm's `run_prompt` runs it, all
//! but its last position, which the step of the first patch after it runs as the patch before
//! ([`patch`](crate::patch)): a prompt that ends in audio, as the service's do.
//!
//! A position is a text token or a patch of audio latents. A token's embedding is its row of the
//! base LM's embedding table, read from the checkpoint on the host: the table is 73448 rows of
//! 2048 values, 300 MB in f16, of which a prompt reads a few dozen rows, so a gather on the GPU
//! would keep it resident to save a copy of 4 KiB a row. A patch's embedding is the local
//! encoder's and enc_to_lm_proj's, a submission for each distinct patch of the prompt: the
//! service's prompts hold their voice's patches twice.
//!
//! Then a submission for each run of up to `max_rows` positions: the base LM over the
//! embeddings, the scalar quantizer over its outputs at the audio positions, fusion_concat_proj
//! of those beside the audio embeddings, zero at text, and the residual LM over that, both LMs
//! writing their keys and values into the patch step's caches. The layers between the LMs read
//! each weight once for a block of rows (`linear_rows.comp`). Both are programs over the patch
//! step's memory ([`Step::add`](crate::step::Step::add)).

use std::{collections::HashMap, iter};

use vulkan::{Context, SpecValue};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{LayerTensors, Tuning},
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::{
        Access::{Read, Write},
        KernelInfo, Kernels,
    },
    lm_side::{Act, Linear, Stack, append_linear, splice},
    patch::{PatchGraph, PatchStep, transposed},
    prefill::{StackTensors, TILE, append_prefill},
    step::Program,
};

/// `linear_rows.comp`: run, x0, x1, W, bias -> out.
pub static LINEAR_ROWS: KernelInfo = KernelInfo {
    name: "linear_rows",
    access: &[Read, Read, Read, Read, Read, Write],
};

/// `select_rows.comp`: run, flags, a, b -> out.
pub static SELECT_ROWS: KernelInfo = KernelInfo {
    name: "select_rows",
    access: &[Read, Read, Read, Read, Write],
};

/// The width of the rows kernels' workgroups.
const WORKGROUP: u32 = 256;

/// The rows a workgroup of linear_rows takes, and the columns of them it stages at a time.
const BLOCK: usize = 32;
const STAGED: usize = 512;

/// The lanes linear_rows sums a row of W with, fixed in gemm.glsl.
const CLUSTER: usize = 16;

/// The base LM's embedding table in the checkpoint.
const EMBED_TOKENS: &str = "base_lm.embed_tokens.weight";

/// The programs a prompt runs over the patch step's memory, all over the same tensors.
#[derive(Clone, Debug)]
pub struct PromptGraphs {
    /// Embeds a patch.
    pub embed: Graph,
    /// `programs[i]` runs `(i + 1) TILE` positions.
    pub programs: Vec<Graph>,
    pub tensors: PromptTensors,
}

/// The tensors of a prompt's programs that the host writes and reads.
#[derive(Clone, Copy, Debug)]
pub struct PromptTensors {
    pub max_rows: usize,
    /// The patch to embed, laid out as the patch step's `cond`, and its embedding `[hidden]` f16.
    pub patch: TensorId,
    pub embedding: TensorId,
    /// The first position and the length of the run, two u32.
    pub run: TensorId,
    /// The positions' embeddings, `[max_rows][hidden]` f16.
    pub x: TensorId,
    /// The audio positions' embeddings, laid out as `x`, zero at text.
    pub audio_x: TensorId,
    /// Whether each position is audio, a u32 each.
    pub audio: TensorId,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The stack `stack` of the LM side, where `at` puts its tensors in the patch step's graph.
fn stack_layers(stack: &Stack, at: impl Fn(TensorId) -> TensorId) -> (Vec<LayerTensors>, TensorId) {
    (
        stack.layers.iter().map(|l| l.map(&at)).collect(),
        at(stack.norm),
    )
}

/// The prompt's programs over `patch`'s memory, for runs of up to `max_rows` positions, their
/// kernels at `tuning`'s subgroup size.
pub fn prompt_graphs(
    patch: &PatchGraph,
    max_rows: usize,
    tuning: &Tuning,
) -> Result<PromptGraphs, Error> {
    let side = &patch.side;
    let s = &side.shape;
    let hidden = s.base.hidden;
    let at = |t: TensorId| patch.side_ids[t.0];
    let layer = |l: Linear| Linear {
        weight: at(l.weight),
        bias: at(l.bias),
    };
    let mut g = patch.graph.clone();
    g.ops.clear();

    let encoder = &patch.encoder;
    let bytes = |graph: &Graph, t: TensorId| graph.tensors[t.0].bytes;
    let input = g.tensor(
        "embed.patch",
        bytes(&encoder.graph, encoder.patch),
        Kind::Input,
    );
    let enc = g.tensor(
        "embed.enc",
        bytes(&encoder.graph, encoder.out),
        Kind::Intermediate,
    );
    let embedding = g.tensor("embed.out", 2 * hidden as u64, Kind::Output);
    let weights = encoder.graph.tensors.iter().enumerate();
    let shared: Vec<(TensorId, TensorId)> = weights
        .filter(|(_, t)| t.kind == Kind::Constant)
        .map(|(i, _)| (TensorId(i), patch.encoder_ids[i]))
        .chain([(encoder.patch, input), (encoder.out, enc)])
        .collect();
    splice(&mut g, &encoder.graph, "embed.", &shared);
    append_linear(
        &mut g,
        "embed.enc_to_lm_proj",
        layer(side.heads.enc_to_lm_proj),
        &[enc],
        Act::None,
        (embedding, 0),
        tuning,
        s.levels,
    )?;
    let embed_ops = g.ops.len();

    let rows_of = |n: usize| (max_rows * n * 2) as u64;
    let run = g.tensor("prompt.run", 8, Kind::Param);
    let x = g.tensor("prompt.x", rows_of(hidden), Kind::Input);
    let audio_x = g.tensor("prompt.audio_x", rows_of(hidden), Kind::Input);
    let audio = g.tensor("prompt.audio", (max_rows * 4) as u64, Kind::Input);
    let mut between = |name: &str, n: usize| g.tensor(name, rows_of(n), Kind::Intermediate);
    let base_out = between("prompt.base_lm.out", hidden);
    let levels = between("prompt.fsq_layer.levels", s.latent);
    let quantized = between("prompt.fsq_layer.out", hidden);
    let lm_hidden = between("prompt.lm_hidden", hidden);
    let residual_in = between("prompt.residual_lm.in", hidden);
    let residual_out = between("prompt.residual_lm.out", hidden);
    let (base_layers, base_norm) = stack_layers(&side.base, at);
    let (residual_layers, residual_norm) = stack_layers(&side.residual, at);
    let stack = |shape, layers, norm| StackTensors {
        shape,
        layers,
        norm,
        cos: at(side.cos),
        sin: at(side.sin),
    };
    let base = stack(&s.base, &base_layers[..], base_norm);
    let residual = stack(&s.residual, &residual_layers[..], residual_norm);

    let subgroup = tuning.subgroup;
    let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
        kernel,
        constants: [
            (0, SpecValue::U32(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
        ]
        .iter()
        .chain(constants)
        .copied()
        .collect(),
        subgroup_size: Some(subgroup),
    };
    let check = |ok: bool, why: String| if ok { Ok(()) } else { Err(Error::Shape(why)) };
    // `layer` over each row of the inputs `x` joined, into `out`.
    let rows_linear = |g: &mut Graph,
                       label: &str,
                       layer: Linear,
                       x: &[TensorId],
                       act: Act,
                       out: TensorId,
                       rows: usize|
     -> Result<(), Error> {
        let width = |t: TensorId| g.tensors[t.0].bytes as usize / 2 / max_rows;
        let (k0, k1) = (width(x[0]), x.get(1).map_or(0, |&t| width(t)));
        let n = width(out);
        let k = k0 + k1;
        let staged = STAGED.min(k);
        check(
            k0.is_multiple_of(8)
                && k.is_multiple_of(staged)
                && staged.is_multiple_of(8 * CLUSTER)
                && n.is_multiple_of(WORKGROUP as usize / CLUSTER)
                && rows.is_multiple_of(BLOCK),
            format!("{label}: {k0} and {k1} columns into {n} over {rows} rows"),
        )?;
        let pipeline = pipeline(
            &LINEAR_ROWS,
            &[
                (2, u(k0)),
                (3, u(k1)),
                (4, u(staged)),
                (5, u(BLOCK)),
                (6, u(n)),
                (7, SpecValue::U32(act as u32)),
                (8, SpecValue::F32(s.levels as f32)),
            ],
        );
        let groups = [
            (n / (WORKGROUP as usize / CLUSTER)) as u32,
            (rows / BLOCK) as u32,
            1,
        ];
        let bindings = [run, x[0], x[x.len() - 1], layer.weight, layer.bias, out];
        g.op(label, pipeline, &bindings, groups);
        Ok(())
    };
    let select = pipeline(&SELECT_ROWS, &[(2, u(hidden))]);

    let mut programs = Vec::with_capacity(max_rows / TILE);
    for tiles in 1..=max_rows / TILE {
        let rows = tiles * TILE;
        let mut p = g.clone();
        let io = |x, out| [run, x, out];
        append_prefill(
            &mut p,
            "prompt.base_lm",
            base,
            rows,
            max_rows,
            tuning,
            io(x, base_out),
        )?;
        let h = &side.heads;
        rows_linear(
            &mut p,
            "fsq_layer.in_proj",
            layer(h.fsq_in_proj),
            &[base_out],
            Act::Levels,
            levels,
            rows,
        )?;
        rows_linear(
            &mut p,
            "fsq_layer.out_proj",
            layer(h.fsq_out_proj),
            &[levels],
            Act::None,
            quantized,
            rows,
        )?;
        p.op(
            "select_rows",
            select.clone(),
            &[run, audio, quantized, base_out, lm_hidden],
            [rows as u32, 1, 1],
        );
        rows_linear(
            &mut p,
            "fusion_concat_proj",
            layer(h.fusion_concat_proj),
            &[lm_hidden, audio_x],
            Act::None,
            residual_in,
            rows,
        )?;
        append_prefill(
            &mut p,
            "prompt.residual_lm",
            residual,
            rows,
            max_rows,
            tuning,
            io(residual_in, residual_out),
        )?;
        p.ops.drain(..embed_ops);
        programs.push(p);
    }
    let Some(first) = programs.first() else {
        return Err(Error::Shape(format!("programs of up to {max_rows} rows")));
    };
    let mut embed = first.clone();
    embed.ops = g.ops;
    Ok(PromptGraphs {
        embed,
        programs,
        tensors: PromptTensors {
            max_rows,
            patch: input,
            embedding,
            run,
            x,
            audio_x,
            audio,
        },
    })
}

/// A prompt's positions, laid out as voxcpm's `Prompt` lays them out.
#[derive(Clone, Copy, Debug)]
pub struct Prompt<'a> {
    /// Each position's token, read where it is text.
    pub tokens: &'a [u32],
    /// Each position's patch, `[patch_size][feat_dim]`, read where it is audio.
    pub patches: &'a [f32],
    /// Whether each position is audio.
    pub audio: &'a [bool],
}

impl Prompt<'_> {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// The prompt's programs on the GPU, recorded over a patch step's memory.
#[derive(Debug)]
pub struct Prompter {
    embed: Program,
    /// `programs[i]` runs `(i + 1) TILE` positions.
    programs: Vec<Program>,
    tensors: PromptTensors,
    hidden: usize,
    patch_size: usize,
}

impl Prompter {
    /// Records the prompt's programs over `step`'s memory, for runs of up to `max_rows`
    /// positions at once, their kernels at `tuning`'s subgroup size; timed per dispatch when
    /// `timed`.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        step: &mut PatchStep,
        max_rows: usize,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let (graph, recorded) = step.parts_mut();
        let graphs = prompt_graphs(graph, max_rows, tuning)?;
        let hidden = graph.side.shape.base.hidden;
        let patch_size = graph.sampler.patch_size;
        let all: Vec<&Graph> = iter::once(&graphs.embed).chain(&graphs.programs).collect();
        let mut programs = recorded.add(ctx, kernels, &all, timed)?;
        let embed = programs.remove(0);
        Ok(Prompter {
            embed,
            programs,
            tensors: graphs.tensors,
            hidden,
            patch_size,
        })
    }

    /// The most positions one submission runs.
    pub fn max_rows(&self) -> usize {
        self.tensors.max_rows
    }

    /// The embedding of `patch`, `[patch_size][feat_dim]`, as the base LM takes it, as f16 bits.
    pub fn embed(&self, step: &mut PatchStep, patch: &[f32]) -> Result<Vec<u16>, Error> {
        let (_, recorded) = step.parts_mut();
        let cond = transposed(patch, self.patch_size);
        recorded.write(self.tensors.patch, &cond)?;
        recorded.run(self.embed)?;
        recorded.read(self.tensors.embedding)
    }

    /// Runs `prompt` but its last position into `step`'s caches from position 0, the tokens'
    /// embeddings read from `checkpoint`. The step of the first patch after it runs the last,
    /// whose patch it takes as the patch before.
    pub fn run(
        &self,
        step: &mut PatchStep,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
    ) -> Result<(), Error> {
        let (x, audio_x) = self.embeddings(step, checkpoint, prompt)?;
        self.run_rows(step, 0, &x, &audio_x, &prompt.audio[..prompt.len() - 1])
    }

    /// The embeddings of `prompt`'s positions but its last, `[len - 1][hidden]` as f16 bits, and
    /// those of its audio positions, zero at text: a token's read from `checkpoint`, a patch's
    /// from the GPU, once for each distinct patch.
    pub fn embeddings(
        &self,
        step: &mut PatchStep,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
    ) -> Result<(Vec<u16>, Vec<u16>), Error> {
        let len = prompt.len();
        let patch = prompt.patches.len() / len.max(1);
        if len == 0
            || prompt.audio.len() != len
            || prompt.patches.len() != len * patch
            || !prompt.audio[len - 1]
        {
            return Err(Error::Shape(format!(
                "a prompt of {len} tokens, {} values of patches and {} audio flags, which must \
                 end in audio",
                prompt.patches.len(),
                prompt.audio.len()
            )));
        }
        let (ran, hidden) = (len - 1, self.hidden);
        let mut x = vec![0u16; ran * hidden];
        let mut audio_x = vec![0u16; ran * hidden];
        let text: Vec<usize> = (0..ran).filter(|&p| !prompt.audio[p]).collect();
        let tokens: Vec<usize> = text.iter().map(|&p| prompt.tokens[p] as usize).collect();
        let rows = checkpoint.gather_f16(EMBED_TOKENS, &tokens)?;
        for (&p, row) in text.iter().zip(rows.chunks(hidden)) {
            x[p * hidden..(p + 1) * hidden].copy_from_slice(row);
        }
        let mut embedded: HashMap<Vec<u32>, Vec<u16>> = HashMap::new();
        for p in (0..ran).filter(|&p| prompt.audio[p]) {
            let values = &prompt.patches[p * patch..(p + 1) * patch];
            let key: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let row = match embedded.get(&key) {
                Some(row) => row,
                None => {
                    let row = self.embed(step, values)?;
                    embedded.entry(key).or_insert(row)
                }
            };
            x[p * hidden..(p + 1) * hidden].copy_from_slice(row);
            audio_x[p * hidden..(p + 1) * hidden].copy_from_slice(row);
        }
        Ok((x, audio_x))
    }

    /// Runs the positions from `start` on whose embeddings are `x`, `[len][hidden]` as f16 bits,
    /// those of audio as `audio` says, through both LMs into `step`'s caches: `audio_x` holds the
    /// embeddings of the audio positions, zero at text. A run of more than
    /// [`max_rows`](Self::max_rows) positions runs in pieces.
    pub fn run_rows(
        &self,
        step: &mut PatchStep,
        start: usize,
        x: &[u16],
        audio_x: &[u16],
        audio: &[bool],
    ) -> Result<(), Error> {
        let (hidden, max_rows) = (self.hidden, self.tensors.max_rows);
        let len = audio.len();
        if x.len() != len * hidden || audio_x.len() != x.len() || start + len > step.capacity() {
            return Err(Error::Shape(format!(
                "{} and {} values for {len} positions from {start} in a cache of {}",
                x.len(),
                audio_x.len(),
                step.capacity()
            )));
        }
        let (_, recorded) = step.parts_mut();
        let t = &self.tensors;
        let mut padded = vec![0u16; max_rows * hidden];
        let mut padded_audio = vec![0u16; max_rows * hidden];
        let mut flags = vec![0u32; max_rows];
        for first in (0..len).step_by(max_rows) {
            let rows = (len - first).min(max_rows);
            let piece = first * hidden..(first + rows) * hidden;
            padded[..rows * hidden].copy_from_slice(&x[piece.clone()]);
            padded_audio[..rows * hidden].copy_from_slice(&audio_x[piece]);
            let audio = &audio[first..first + rows];
            for (flag, &a) in flags.iter_mut().zip(audio) {
                *flag = u32::from(a);
            }
            recorded.write(t.run, &[(start + first) as u32, rows as u32])?;
            recorded.write(t.x, &padded)?;
            recorded.write(t.audio_x, &padded_audio)?;
            recorded.write(t.audio, &flags)?;
            recorded.run(self.programs[rows.div_ceil(TILE) - 1])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::Access, patch};

    fn constant(op: &crate::graph::Op, id: u32) -> usize {
        match op.pipeline.constants.iter().find(|c| c.0 == id) {
            Some(&(_, SpecValue::U32(v))) => v as usize,
            other => panic!("{}: constant {id} is {other:?}", op.label),
        }
    }

    #[test]
    fn the_programs_run_over_the_patch_steps_weights_and_caches_and_tensors_of_their_own() {
        for subgroup in crate::decode::SUBGROUPS {
            let patch = patch::tests::graph(subgroup);
            let p = prompt_graphs(&patch, 2 * TILE, &Tuning::for_subgroup(subgroup)).unwrap();
            let shape = |g: &Graph| -> Vec<(u64, Kind)> {
                g.tensors.iter().map(|t| (t.bytes, t.kind)).collect()
            };
            assert_eq!(p.programs.len(), 2);
            assert!(p.programs.iter().all(|g| shape(g) == shape(&p.embed)));
            assert_eq!(p.embed.ops.len(), patch.encoder.graph.ops.len() + 1);
            let held = &patch.graph.tensors;
            for g in iter::once(&p.embed).chain(&p.programs) {
                for op in &g.ops {
                    let theirs = op.bindings.iter().filter(|t| t.0 < held.len());
                    let arena = theirs.filter(|t| held[t.0].kind.in_arena());
                    assert_eq!(
                        arena.count(),
                        0,
                        "{} binds the patch step's arena",
                        op.label
                    );
                }
            }
            let caches = |stack: &Stack| -> Vec<TensorId> {
                let at = |t: TensorId| patch.side_ids[t.0];
                stack.layers.iter().map(|l| at(l.k_cache)).collect()
            };
            let mut keys = caches(&patch.side.base);
            keys.extend(caches(&patch.side.residual));
            for (tiles, g) in p.programs.iter().enumerate() {
                let rows = (tiles + 1) * TILE;
                let written: Vec<TensorId> = g
                    .ops
                    .iter()
                    .flat_map(|op| op.bindings.iter().zip(op.pipeline.kernel.access))
                    .filter(|&(t, access)| *access == Access::Write && keys.contains(t))
                    .map(|(&t, _)| t)
                    .collect();
                assert_eq!(written, keys, "{rows} rows");
                let linear = g
                    .ops
                    .iter()
                    .filter(|op| op.pipeline.kernel.name == "linear_rows");
                let mut layers = 0;
                for op in linear {
                    let n = constant(op, 6);
                    assert_eq!(op.groups[0] as usize * WORKGROUP as usize / CLUSTER, n);
                    assert_eq!(op.groups[1] as usize * BLOCK, rows, "{}", op.label);
                    layers += 1;
                }
                assert_eq!(layers, 3);
            }
        }
    }
}
