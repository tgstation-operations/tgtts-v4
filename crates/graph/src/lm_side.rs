//! VoxCPM2's LM side of a patch, as one recorded step: from the local encoder's output for the
//! last patch to what the next is sampled with, as upstream's `VoxCPM2Model` computes it.
//! enc_to_lm_proj embeds the patch for the base LM, whose output the scalar quantizer rounds;
//! fusion_concat_proj joins that with the embedding for the residual LM, which goes without RoPE;
//! the DiT's condition is lm_to_dit_proj of the quantized output beside res_to_dit_proj of the
//! residual LM's, and the stop logits stop_head of SiLU(stop_proj) of the quantized output. Both
//! stacks are [`decode`](crate::decode)'s step, their caches written at the position the host
//! gives. A step runs a [`Batch`] of rows, each a request at its own position in its own slot of
//! both caches, every layer reading its weights once for all of them; a row comes out bit for bit
//! as it does alone.

use serde::Deserialize;
use vulkan::{Context, SpecValue, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{
        Batch, DecodeGraph, LayerTensors, LmConfig, Row, Shape, Tuning, decode_batch_graph,
        load_stack,
    },
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::{
        Access::{Read, Write},
        KernelInfo, Kernels,
    },
    local::identity,
    step::Step,
};

/// `gemv_linear.comp`: x0, x1, W, bias -> out.
pub static GEMV_LINEAR: KernelInfo = KernelInfo {
    name: "gemv_linear",
    access: &[Read, Read, Read, Read, Write],
};

/// What the LM side reads of VoxCPM2's `config.json`; the rest is ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub lm_config: LmConfig,
    pub residual_lm_num_layers: usize,
    pub residual_lm_no_rope: bool,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: u32,
    pub encoder_config: Width,
    pub dit_config: Width,
}

/// The width of the local encoder's or the local DiT's stack.
#[derive(Clone, Debug, Deserialize)]
pub struct Width {
    pub hidden_dim: usize,
}

/// The shapes the LM side is recorded at.
#[derive(Clone, Debug, PartialEq)]
pub struct LmSideShape {
    pub base: Shape,
    pub residual: Shape,
    /// The local encoder's width, the step's input.
    pub encoder: usize,
    /// The local DiT's width, half the condition's.
    pub dit: usize,
    /// The scalar quantizer's bottleneck.
    pub latent: usize,
    /// The levels to a unit the scalar quantizer rounds to, upstream's
    /// `scalar_quantization_scale`.
    pub levels: u32,
}

impl LmSideShape {
    /// The LM side `config` describes, both LMs with caches of `capacity` positions.
    pub fn new(config: &Config, capacity: usize) -> Result<Self, Error> {
        let lm = &config.lm_config;
        let residual = LmConfig {
            num_hidden_layers: config.residual_lm_num_layers,
            no_rope: config.residual_lm_no_rope,
            ..lm.clone()
        };
        Ok(LmSideShape {
            base: Shape::new(lm, lm.num_hidden_layers, capacity)?,
            residual: Shape::new(&residual, residual.num_hidden_layers, capacity)?,
            encoder: config.encoder_config.hidden_dim,
            dit: config.dit_config.hidden_dim,
            latent: config.scalar_quantization_latent_dim,
            levels: config.scalar_quantization_scale,
        })
    }
}

/// A linear layer's tensors.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub weight: TensorId,
    pub bias: TensorId,
}

/// The layers between the stacks, under their names in the checkpoint.
#[derive(Clone, Debug)]
pub struct Heads {
    pub enc_to_lm_proj: Linear,
    pub fsq_in_proj: Linear,
    pub fsq_out_proj: Linear,
    pub fusion_concat_proj: Linear,
    pub lm_to_dit_proj: Linear,
    pub res_to_dit_proj: Linear,
    pub stop_proj: Linear,
    /// Its bias is zeros: upstream's stop_head has none.
    pub stop_head: Linear,
}

impl Heads {
    /// Each layer with the checkpoint's name for it, and whether the checkpoint has its bias.
    fn named(&self) -> [(&'static str, Linear, bool); 8] {
        [
            ("enc_to_lm_proj", self.enc_to_lm_proj, true),
            ("fsq_layer.in_proj", self.fsq_in_proj, true),
            ("fsq_layer.out_proj", self.fsq_out_proj, true),
            ("fusion_concat_proj", self.fusion_concat_proj, true),
            ("lm_to_dit_proj", self.lm_to_dit_proj, true),
            ("res_to_dit_proj", self.res_to_dit_proj, true),
            ("stop_proj", self.stop_proj, true),
            ("stop_head", self.stop_head, false),
        ]
    }
}

/// A stack's weights among the LM side's tensors.
#[derive(Clone, Debug)]
pub struct Stack {
    pub layers: Vec<LayerTensors>,
    pub norm: TensorId,
}

/// The LM side's graph, and the tensors the host writes and reads, a row of each for every row of
/// the batch.
#[derive(Clone, Debug)]
pub struct LmSideGraph {
    pub graph: Graph,
    pub shape: LmSideShape,
    pub batch: Batch,
    /// The local encoder's output for the last patch, `[encoder]` f16: the step's input.
    pub enc: TensorId,
    /// Where the patch goes in both LMs' caches, [`Batch::words`] u32: at batch 1, the one
    /// position.
    pub position: TensorId,
    /// The DiT's condition for the next patch, `[2 * dit]` f16: lm_to_dit_proj's output, then
    /// res_to_dit_proj's.
    pub mu: TensorId,
    /// The stop head's logits `[2]` f16, of going on and of stopping after the next patch.
    pub stop_logits: TensorId,
    /// The base LM's output `[hidden]` f16, which the scalar quantizer takes.
    pub base_out: TensorId,
    /// The scalar quantizer's levels `[latent]` f16, each a multiple of `1 / levels`.
    pub levels: TensorId,
    pub cos: TensorId,
    pub sin: TensorId,
    pub base: Stack,
    pub residual: Stack,
    pub heads: Heads,
}

/// Appends `from`'s tensors and ops to `g`, names and labels after `prefix`: each tensor `shared`
/// pairs with one of `g` taken as that one, the rest added. Gives each of `from`'s tensors' place
/// in `g`.
///
/// # Panics
///
/// If a shared pair differs in size.
pub fn splice(
    g: &mut Graph,
    from: &Graph,
    prefix: &str,
    shared: &[(TensorId, TensorId)],
) -> Vec<TensorId> {
    let ids: Vec<TensorId> = from
        .tensors
        .iter()
        .enumerate()
        .map(
            |(i, t)| match shared.iter().find(|(theirs, _)| theirs.0 == i) {
                Some(&(_, ours)) => {
                    let bytes = g.tensors[ours.0].bytes;
                    assert_eq!(bytes, t.bytes, "{prefix}{} as {bytes} bytes", t.name);
                    ours
                }
                None => g.tensor(format!("{prefix}{}", t.name), t.bytes, t.kind),
            },
        )
        .collect();
    for op in &from.ops {
        let bindings: Vec<TensorId> = op.bindings.iter().map(|t| ids[t.0]).collect();
        let label = format!("{prefix}{}", op.label);
        g.op(label, op.pipeline.clone(), &bindings, op.groups);
    }
    ids
}

/// Appends the decode step `stack` to `g` under `prefix`, reading `x` and writing `out`, at `g`'s
/// position and with its rotation tables.
fn append_stack(
    g: &mut Graph,
    stack: &DecodeGraph,
    prefix: &str,
    [x, position, cos, sin, out]: [TensorId; 5],
) -> Stack {
    let shared = [
        (stack.x, x),
        (stack.position, position),
        (stack.cos, cos),
        (stack.sin, sin),
        (stack.out, out),
    ];
    let ids = splice(g, &stack.graph, prefix, &shared);
    let at = |t: TensorId| ids[t.0];
    Stack {
        layers: stack.layers.iter().map(|l| l.map(at)).collect(),
        norm: at(stack.norm),
    }
}

/// What a linear layer does to its sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Act {
    None = 0,
    Silu = 1,
    /// The scalar quantizer's rounding.
    Levels = 2,
}

/// Appends `layer` over the inputs `x` joined, written into `out` from `offset`, to `g`, as
/// gemv_linear runs it cut as `tuning` says, the scalar quantizer rounding to multiples of
/// `1 / levels`. A layer of one input binds it as x1 too, which the kernel reads only past x0's
/// columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_linear(
    g: &mut Graph,
    label: &str,
    layer: Linear,
    x: &[TensorId],
    act: Act,
    out: (TensorId, usize),
    tuning: &Tuning,
    levels: u32,
) -> Result<(), Error> {
    append_batch_linear(g, label, layer, x, act, out, tuning, levels, 1)
}

/// [`append_linear`] over `rows` rows of each input, written into as many rows of `out`.
#[allow(clippy::too_many_arguments)]
fn append_batch_linear(
    g: &mut Graph,
    label: &str,
    layer: Linear,
    x: &[TensorId],
    act: Act,
    (out, offset): (TensorId, usize),
    tuning: &Tuning,
    levels: u32,
    rows: usize,
) -> Result<(), Error> {
    let Tuning {
        subgroup,
        workgroup,
        ..
    } = *tuning;
    let subgroups = (workgroup / subgroup) as usize;
    let len = |t: TensorId| g.tensors[t.0].bytes as usize / 2 / rows;
    let (x0, x1) = (x[0], x[x.len() - 1]);
    let (k0, k1) = (len(x0), if x.len() > 1 { len(x1) } else { 0 });
    for k in [k0, k1] {
        if !k.is_multiple_of(8 * subgroup as usize) {
            return Err(Error::Shape(format!(
                "{label}: {k} columns in steps of {}",
                8 * subgroup
            )));
        }
    }
    let n = g.tensors[layer.bias.0].bytes as usize / 2;
    let (taken, split) = tuning.split(rows);
    let pipeline = Pipeline {
        kernel: &GEMV_LINEAR,
        constants: vec![
            (0, SpecValue::U32(subgroup)),
            (1, SpecValue::U32(workgroup)),
            (2, SpecValue::U32(k0 as u32)),
            (3, SpecValue::U32(k1 as u32)),
            (4, SpecValue::U32(n as u32)),
            (5, SpecValue::U32(offset as u32)),
            (6, SpecValue::U32(act as u32)),
            (7, SpecValue::F32(levels as f32)),
            (8, SpecValue::U32(taken as u32)),
            (9, SpecValue::U32(tuning.span(k0 + k1, taken, 0)?)),
            (10, SpecValue::U32(len(out) as u32)),
            (11, SpecValue::U32(split as u32)),
        ],
        subgroup_size: Some(subgroup),
    };
    let groups = (n.div_ceil(subgroups) * split) as u32;
    let bindings = [x0, x1, layer.weight, layer.bias, out];
    g.op(label, pipeline, &bindings, [groups, 1, 1]);
    Ok(())
}

/// The LM side of `shape` at batch 1 cut as `tuning` says.
pub fn lm_side_graph(shape: &LmSideShape, tuning: &Tuning) -> Result<LmSideGraph, Error> {
    lm_side_batch_graph(shape, tuning, Batch::ONE)
}

/// The LM side of `shape` for `batch` cut as `tuning` says.
pub fn lm_side_batch_graph(
    shape: &LmSideShape,
    tuning: &Tuning,
    batch: Batch,
) -> Result<LmSideGraph, Error> {
    let s = shape;
    let (b, r) = (&s.base, &s.residual);
    if (r.hidden, r.capacity, r.head_dim) != (b.hidden, b.capacity, b.head_dim) {
        return Err(Error::Shape(format!(
            "a residual LM of {} wide, {} positions, heads of {}, after a base LM of {}, {}, {}",
            r.hidden, r.capacity, r.head_dim, b.hidden, b.capacity, b.head_dim
        )));
    }
    let base = decode_batch_graph(b, tuning, batch)?;
    let residual = decode_batch_graph(r, tuning, batch)?;
    let hidden = b.hidden;

    let mut g = Graph::default();
    let rows = batch.rows;
    let vector = |n: usize| (rows * n) as u64 * 2;
    let weight = |n: usize| n as u64 * 2;
    let enc = g.tensor("enc", vector(s.encoder), Kind::Input);
    let position = g.tensor("position", 4 * batch.words() as u64, Kind::Param);
    let table = base.graph.tensors[base.cos.0].bytes;
    let cos = g.tensor("rope.cos", table, Kind::Constant);
    let sin = g.tensor("rope.sin", table, Kind::Constant);
    let mu = g.tensor("mu", vector(2 * s.dit), Kind::Output);
    let stop_logits = g.tensor("stop_logits", vector(2), Kind::Output);
    let mut linear = |name: &str, rows: usize, columns: usize| Linear {
        weight: g.tensor(
            format!("{name}.weight"),
            weight(rows * columns),
            Kind::Constant,
        ),
        bias: g.tensor(format!("{name}.bias"), weight(rows), Kind::Constant),
    };
    let heads = Heads {
        enc_to_lm_proj: linear("enc_to_lm_proj", hidden, s.encoder),
        fsq_in_proj: linear("fsq_layer.in_proj", s.latent, hidden),
        fsq_out_proj: linear("fsq_layer.out_proj", hidden, s.latent),
        fusion_concat_proj: linear("fusion_concat_proj", hidden, 2 * hidden),
        lm_to_dit_proj: linear("lm_to_dit_proj", s.dit, hidden),
        res_to_dit_proj: linear("res_to_dit_proj", s.dit, hidden),
        stop_proj: linear("stop_proj", hidden, hidden),
        stop_head: linear("stop_head", 2, hidden),
    };
    let mut between = |name: &str, n: usize| g.tensor(name, vector(n), Kind::Intermediate);
    let embed = between("embed", hidden);
    let base_out = between("base_lm.out", hidden);
    let levels = between("fsq_layer.levels", s.latent);
    let lm_hidden = between("lm_hidden", hidden);
    let residual_in = between("residual_lm.in", hidden);
    let residual_out = between("residual_lm.out", hidden);
    let stop_hidden = between("stop_hidden", hidden);

    let apply = |g: &mut Graph,
                 label: &str,
                 layer: Linear,
                 x: &[TensorId],
                 act: Act,
                 out: (TensorId, usize)| {
        append_batch_linear(g, label, layer, x, act, out, tuning, s.levels, rows)
    };
    let h = &heads;
    apply(
        &mut g,
        "enc_to_lm_proj",
        h.enc_to_lm_proj,
        &[enc],
        Act::None,
        (embed, 0),
    )?;
    let at = [embed, position, cos, sin, base_out];
    let base_stack = append_stack(&mut g, &base, "base_lm.", at);
    apply(
        &mut g,
        "fsq_layer.in_proj",
        h.fsq_in_proj,
        &[base_out],
        Act::Levels,
        (levels, 0),
    )?;
    apply(
        &mut g,
        "fsq_layer.out_proj",
        h.fsq_out_proj,
        &[levels],
        Act::None,
        (lm_hidden, 0),
    )?;
    // The three readers of the quantized output run together, and stop_head beside the residual
    // LM's first dispatch.
    apply(
        &mut g,
        "fusion_concat_proj",
        h.fusion_concat_proj,
        &[lm_hidden, embed],
        Act::None,
        (residual_in, 0),
    )?;
    apply(
        &mut g,
        "lm_to_dit_proj",
        h.lm_to_dit_proj,
        &[lm_hidden],
        Act::None,
        (mu, 0),
    )?;
    apply(
        &mut g,
        "stop_proj",
        h.stop_proj,
        &[lm_hidden],
        Act::Silu,
        (stop_hidden, 0),
    )?;
    apply(
        &mut g,
        "stop_head",
        h.stop_head,
        &[stop_hidden],
        Act::None,
        (stop_logits, 0),
    )?;
    let at = [residual_in, position, cos, sin, residual_out];
    let residual_stack = append_stack(&mut g, &residual, "residual_lm.", at);
    apply(
        &mut g,
        "res_to_dit_proj",
        h.res_to_dit_proj,
        &[residual_out],
        Act::None,
        (mu, s.dit),
    )?;
    Ok(LmSideGraph {
        graph: g,
        shape: shape.clone(),
        batch,
        enc,
        position,
        mu,
        stop_logits,
        base_out,
        levels,
        cos,
        sin,
        base: base_stack,
        residual: residual_stack,
        heads,
    })
}

/// Writes the weights of VoxCPM2's LM side from `checkpoint`, converted to f16, into `step`, a
/// recording of `graph`, where `ids` puts each of `side`'s tensors: as [`splice`] gives them when
/// the LM side's graph is part of `graph`.
pub fn load_lm_side(
    step: &mut Step,
    graph: &Graph,
    side: &LmSideGraph,
    ids: &[TensorId],
    checkpoint: &Checkpoint,
) -> Result<(), Error> {
    let at = |t: TensorId| ids[t.0];
    for (stack, prefix) in [(&side.base, "base_lm."), (&side.residual, "residual_lm.")] {
        let layers: Vec<LayerTensors> = stack.layers.iter().map(|l| l.map(at)).collect();
        load_stack(step, graph, &layers, at(stack.norm), checkpoint, prefix)?;
    }
    for (name, layer, has_bias) in side.heads.named() {
        let parts = [("weight", layer.weight)]
            .into_iter()
            .chain(has_bias.then_some(("bias", layer.bias)));
        for (part, tensor) in parts {
            let name = format!("{name}.{part}");
            let (_, values) = checkpoint.f16(&name)?;
            let want = graph.tensors[at(tensor).0].bytes as usize / 2;
            if values.len() != want {
                return Err(Error::Checkpoint(format!(
                    "{name}: {} values, not {want}",
                    values.len()
                )));
            }
            step.write(at(tensor), &values)?;
        }
    }
    Ok(())
}

/// What the LM side gives for a patch: what the next is sampled with.
#[derive(Clone, Debug, PartialEq)]
pub struct Conditioning {
    /// The DiT's condition `[2 * dit]` as f16 bits.
    pub mu: Vec<u16>,
    /// The stop head's logits, of going on and of stopping after the next patch.
    pub stop_logits: [f32; 2],
}

/// VoxCPM2's LM side on the GPU, with its weights and both LMs' KV caches.
#[derive(Debug)]
pub struct LmSide {
    step: Step,
    tensors: LmSideGraph,
}

impl LmSide {
    /// Records the LM side of `shape` at batch 1, with its rotation tables written and every
    /// weight zero until [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &LmSideShape,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        Self::with_batch(ctx, kernels, shape, tuning, Batch::ONE, timed)
    }

    /// Records the LM side of `shape` for `batch`, as [`new`](Self::new) records it for one row.
    pub fn with_batch(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &LmSideShape,
        tuning: &Tuning,
        batch: Batch,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = lm_side_batch_graph(shape, tuning, batch)?;
        let mut step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        let (cos, sin) = shape.base.rope_tables();
        step.write(tensors.cos, &cos)?;
        step.write(tensors.sin, &sin)?;
        Ok(LmSide { step, tensors })
    }

    /// Writes the weights from VoxCPM2's checkpoint, converted to f16, as [`load_lm_side`] does.
    pub fn load(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let t = &self.tensors;
        load_lm_side(&mut self.step, &t.graph, t, &identity(&t.graph), checkpoint)
    }

    /// Runs the LM side at batch 1 over `enc`, the local encoder's output for the patch at
    /// `position` as f16 bits, writing both LMs' keys and values into their caches there;
    /// returns what the next patch is sampled with.
    pub fn step(&mut self, position: usize, enc: &[u16]) -> Result<Conditioning, Error> {
        let mut rows = self.step_rows(&[Some(Row { position, slot: 0 })], enc)?;
        Ok(rows.remove(0).expect("a row that runs"))
    }

    /// Runs the LM side over `enc`, the local encoder's outputs `[rows][encoder]` for the
    /// batch's rows as f16 bits, each where `rows` says or spare: writes each row's keys and
    /// values into its slot of both LMs' caches at its position. Returns what each row's next
    /// patch is sampled with, none for a spare row.
    pub fn step_rows(
        &mut self,
        rows: &[Option<Row>],
        enc: &[u16],
    ) -> Result<Vec<Option<Conditioning>>, Error> {
        let t = &self.tensors;
        let words = t.batch.placement(rows, t.shape.base.capacity)?;
        self.step.write(t.enc, enc)?;
        self.step.write(t.position, &words)?;
        self.step.submit()?;
        let logits: Vec<u16> = self.step.read(t.stop_logits)?;
        let mu: Vec<u16> = self.step.read(t.mu)?;
        let width = mu.len() / rows.len();
        let logit = |i: usize| half::f16::from_bits(logits[i]).to_f32();
        Ok(rows
            .iter()
            .enumerate()
            .map(|(b, row)| {
                row.map(|_| Conditioning {
                    mu: mu[b * width..(b + 1) * width].to_vec(),
                    stop_logits: [logit(2 * b), logit(2 * b + 1)],
                })
            })
            .collect())
    }

    /// The LM side recorded again cut as `tuning` says, and timed per dispatch when `timed`, over
    /// the weights and the caches it holds. On an error they go with it.
    pub fn retuned(
        self,
        ctx: &Context,
        kernels: &mut Kernels,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = lm_side_batch_graph(&self.tensors.shape, tuning, self.tensors.batch)?;
        let step = self
            .step
            .recorded_again(ctx, kernels, &tensors.graph, timed)?;
        Ok(LmSide { step, tensors })
    }

    /// Each dispatch's label and GPU time in the last step, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn shape(&self) -> &LmSideShape {
        &self.tensors.shape
    }

    pub fn batch(&self) -> Batch {
        self.tensors.batch
    }

    /// The recorded step's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{SUBGROUPS, decode_graph};

    fn stack(layers: usize, rope: bool) -> Shape {
        Shape {
            hidden: 2048,
            inner: 6144,
            heads: 16,
            kv_heads: 2,
            head_dim: 128,
            layers,
            capacity: 1024,
            eps: 1e-5,
            rope_theta: 10_000.0,
            rope_factors: vec![1.0; 64],
            rope,
        }
    }

    fn shape() -> LmSideShape {
        LmSideShape {
            base: stack(28, true),
            residual: stack(8, false),
            encoder: 1024,
            dit: 1024,
            latent: 512,
            levels: 9,
        }
    }

    fn rope(op: &crate::graph::Op) -> Option<SpecValue> {
        let constant = op.pipeline.constants.iter().find(|(id, _)| *id == 10);
        constant.map(|&(_, v)| v)
    }

    #[test]
    fn the_residual_lm_alone_goes_without_rope() {
        let g = lm_side_graph(&shape(), &Tuning::for_subgroup(64)).unwrap();
        let qkv = |prefix: &str| -> Vec<Option<SpecValue>> {
            let label = format!("{prefix}qkv_rope");
            let ops = g.graph.ops.iter().filter(|op| op.label == label);
            ops.map(rope).collect()
        };
        assert_eq!(qkv("base_lm."), vec![Some(SpecValue::Bool(true)); 28]);
        assert_eq!(qkv("residual_lm."), vec![Some(SpecValue::Bool(false)); 8]);
    }

    #[test]
    fn the_heads_run_beside_each_other_where_they_can() {
        let s = shape();
        let g = lm_side_graph(&s, &Tuning::for_subgroup(64)).unwrap();
        let ops = &g.graph.ops;
        // Both stacks, the final norms included, and eight linear layers.
        assert_eq!(ops.len(), (6 * 28 + 1) + (6 * 8 + 1) + 8);
        let plan = crate::graph::plan(&g.graph);
        let barriers = crate::graph::barriers(&g.graph, &plan);
        let after = |label: &str| {
            let i = ops.iter().position(|op| op.label == label).unwrap();
            barriers[i]
        };
        for label in ["lm_to_dit_proj", "stop_proj", "residual_lm.qkv_rope"] {
            assert!(!after(label), "a barrier before {label}");
        }
        for label in ["fusion_concat_proj", "stop_head", "res_to_dit_proj"] {
            assert!(after(label), "no barrier before {label}");
        }
        // The arena holds a layer's intermediates and the vectors between the stacks.
        assert!(
            plan.arena_bytes < 2 * 16 * 32 * 128 * 4,
            "{}",
            plan.arena_bytes
        );
    }

    #[test]
    fn every_linear_layer_covers_its_rows_at_either_subgroup_size() {
        let s = shape();
        for subgroup in SUBGROUPS {
            let tuning = Tuning::for_subgroup(subgroup);
            let g = lm_side_graph(&s, &tuning).unwrap();
            let per_group = (tuning.workgroup / subgroup) as usize;
            let mut covered = 0;
            for op in &g.graph.ops {
                if op.pipeline.kernel.name != GEMV_LINEAR.name {
                    continue;
                }
                let n = match op.pipeline.constants.iter().find(|(id, _)| *id == 4) {
                    Some(&(_, SpecValue::U32(n))) => n as usize,
                    other => panic!("{}: N {other:?}", op.label),
                };
                let rows = op.groups[0] as usize * per_group;
                assert!(
                    rows >= n && rows < n + per_group,
                    "{} at {subgroup}",
                    op.label
                );
                covered += n;
            }
            let h = s.base.hidden;
            assert_eq!(covered, h + s.latent + h + h + 2 * s.dit + h + 2);
        }
    }

    #[test]
    fn a_batch_writes_each_linear_layer_into_its_rows_of_the_output() {
        let s = shape();
        let batch = Batch { rows: 4, slots: 4 };
        let g = lm_side_batch_graph(&s, &Tuning::for_subgroup(64), batch).unwrap();
        let constant =
            |op: &crate::graph::Op, id| match op.pipeline.constants.iter().find(|c| c.0 == id) {
                Some(&(_, SpecValue::U32(v))) => v as u64,
                other => panic!("{}: constant {id} is {other:?}", op.label),
            };
        let mut layers = 0;
        for op in g
            .graph
            .ops
            .iter()
            .filter(|op| op.pipeline.kernel.name == GEMV_LINEAR.name)
        {
            let out = g.graph.tensors[op.bindings[4].0].bytes;
            let (n, offset, stride) = (constant(op, 4), constant(op, 5), constant(op, 10));
            assert_eq!(constant(op, 8) * constant(op, 11), 4, "{}", op.label);
            assert_eq!(4 * stride * 2, out, "{}", op.label);
            assert!(offset + n <= stride, "{}", op.label);
            layers += 1;
        }
        assert_eq!(layers, 8);
        assert_eq!(g.graph.tensors[g.mu.0].bytes, 4 * 2 * s.dit as u64 * 2);
        assert_eq!(g.graph.tensors[g.stop_logits.0].bytes, 4 * 2 * 2);
    }

    #[test]
    fn stacks_that_do_not_meet_are_refused() {
        let tuning = Tuning::for_subgroup(64);
        let mut s = shape();
        s.residual.capacity = 2048;
        assert!(matches!(lm_side_graph(&s, &tuning), Err(Error::Shape(_))));
        let mut s = shape();
        s.encoder = 1000;
        assert!(matches!(lm_side_graph(&s, &tuning), Err(Error::Shape(_))));
    }

    #[test]
    fn a_spliced_graph_keeps_its_ops_on_the_tensors_it_shares() {
        let s = shape();
        let tuning = Tuning::for_subgroup(64);
        let stack = decode_graph(&stack(1, true), &tuning).unwrap();
        let mut g = Graph::default();
        let x = g.tensor("x", 2 * s.base.hidden as u64, Kind::Intermediate);
        let ids = splice(&mut g, &stack.graph, "a.", &[(stack.x, x)]);
        assert_eq!(ids[stack.x.0], x);
        // The stack's own x is not added: x stands for it.
        assert_eq!(g.tensors.len(), stack.graph.tensors.len());
        assert_eq!(g.ops[0].bindings[0], x);
        assert_eq!(g.ops[0].label, "a.qkv_rope");
        assert_eq!(g.tensors[ids[stack.norm.0].0].name, "a.norm");
    }
}
