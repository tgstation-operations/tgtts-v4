//! VoxCPM2's local encoder, as voxcpm's `LocalEncoder` computes it for one patch: the patch's
//! frames projected, after a learned token, through a bidirectional stack whose output at that
//! token, after the final norm, is the patch's embedding. It embeds a patch for each of several
//! rows at once, a sequence to each.
//!
//! It takes each row's patch as the sampler ([`super::dit`]) gives it, `[feat_dim][patch_size]`
//! in f32, and gives each row's embedding, `[hidden]` in f16, which the LM side projects with
//! `enc_to_lm_proj`.

use vulkan::{Context, SpecValue, Timing};

use super::{LOCAL_EMBED, LOCAL_ROWS, PatchConfig, Stack, Tuning, groups, identity, write_weight};
use crate::{
    Error,
    checkpoint::Checkpoint,
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::Kernels,
    step::Step,
};

/// The encoder's graph and the tensors the host writes and reads, each a row after a row.
#[derive(Clone, Debug)]
pub struct EncoderGraph {
    pub graph: Graph,
    pub stack: Stack,
    pub rows: usize,
    pub patch: TensorId,
    /// The learned token that leads the frames.
    pub special: TensorId,
    pub in_proj: (TensorId, TensorId),
    pub out: TensorId,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The local encoder of `config` over `rows` rows, its kernels cut as `tuning` says.
pub fn encoder_graph(
    config: &PatchConfig,
    rows: usize,
    tuning: &Tuning,
) -> Result<EncoderGraph, Error> {
    let (feat_dim, patch_size) = (config.feat_dim, config.patch_size);
    let tokens = 1 + patch_size;
    let shape = config.stack(&config.encoder_config, tokens)?;
    let hidden = shape.hidden;
    let mut g = Graph::default();
    if rows == 0 {
        return Err(Error::Shape("no rows".into()));
    }
    let stack = Stack::new(&mut g, "encoder", &shape, rows, tuning)?;
    let Tuning {
        subgroup,
        workgroup,
        ..
    } = *tuning;
    let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
        kernel,
        constants: [
            (0, SpecValue::U32(subgroup)),
            (1, SpecValue::U32(workgroup)),
        ]
        .iter()
        .chain(constants)
        .copied()
        .collect(),
        subgroup_size: Some(subgroup),
    };
    if !feat_dim.is_multiple_of(8) {
        return Err(Error::Shape(format!("{feat_dim} channels")));
    }
    let embed = pipeline(
        &LOCAL_EMBED,
        &[
            (2, u(feat_dim)),
            (3, u(patch_size)),
            (4, u(hidden)),
            (5, u(1)),
            (6, u(tokens)),
            (7, u(1)),
            (8, u(0)),
            (9, u(1)),
            (10, SpecValue::Bool(false)),
            (11, SpecValue::Bool(true)),
        ],
    );
    let first = pipeline(&LOCAL_ROWS, &[(2, u(hidden)), (3, u(tokens))]);

    let half = |n: usize| 2 * n as u64;
    let patch = g.tensor(
        "patch",
        4 * (rows * feat_dim * patch_size) as u64,
        Kind::Input,
    );
    let special = g.tensor("special_token", half(hidden), Kind::Constant);
    let in_proj = (
        g.tensor("in_proj.weight", half(hidden * feat_dim), Kind::Constant),
        g.tensor("in_proj.bias", half(hidden), Kind::Constant),
    );
    let x = g.tensor(
        "encoder.x",
        half(rows * tokens * hidden),
        Kind::Intermediate,
    );
    let out = g.tensor("out", half(rows * hidden), Kind::Output);
    let each_row = u32::try_from(rows).map_err(|_| Error::Shape(format!("{rows} rows")))?;
    g.op(
        "in_proj",
        embed,
        &[patch, in_proj.0, in_proj.1, special, x],
        [
            groups("embedded columns", hidden, workgroup as usize)?,
            each_row,
            1,
        ],
    );
    let last = stack.layers(&mut g, "encoder", x);
    // The special token's output is the first of each sequence.
    g.op("first", first, &[last, out], [each_row, 1, 1]);
    Ok(EncoderGraph {
        graph: g,
        stack,
        rows,
        patch,
        special,
        in_proj,
        out,
    })
}

/// Writes the weights of VoxCPM2's local encoder from `checkpoint`, converted to f16, into
/// `step`, a recording of `graph`, where `ids` puts each of `encoder`'s tensors: as
/// [`splice`](crate::lm_side::splice) gives them when the encoder's graph is part of `graph`.
pub fn load_encoder(
    step: &mut Step,
    graph: &Graph,
    encoder: &EncoderGraph,
    ids: &[TensorId],
    checkpoint: &Checkpoint,
) -> Result<(), Error> {
    let t = encoder;
    let at = |t: TensorId| ids[t.0];
    let prefix = "feat_encoder.";
    t.stack
        .load(step, graph, ids, checkpoint, &format!("{prefix}encoder."))?;
    let name = |what: &str| vec![format!("{prefix}{what}")];
    write_weight(
        step,
        graph,
        checkpoint,
        at(t.special),
        &name("special_token"),
    )?;
    write_weight(
        step,
        graph,
        checkpoint,
        at(t.in_proj.0),
        &name("in_proj.weight"),
    )?;
    write_weight(
        step,
        graph,
        checkpoint,
        at(t.in_proj.1),
        &name("in_proj.bias"),
    )
}

/// VoxCPM2's local encoder on the GPU, with its weights.
#[derive(Debug)]
pub struct Encoder {
    step: Step,
    tensors: EncoderGraph,
}

impl Encoder {
    /// Records the encoder of `config` over `rows` rows, every weight zero until
    /// [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        config: &PatchConfig,
        rows: usize,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = encoder_graph(config, rows, tuning)?;
        let step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        Ok(Encoder { step, tensors })
    }

    /// Writes the weights of VoxCPM2's local encoder from `checkpoint`, converted to f16, as
    /// [`load_encoder`] does.
    pub fn load(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let t = &self.tensors;
        load_encoder(&mut self.step, &t.graph, t, &identity(&t.graph), checkpoint)
    }

    /// The embedding of each row's patch of `patches`, laid out as the module's documentation
    /// says, as f16 bits.
    pub fn encode(&mut self, patches: &[f32]) -> Result<Vec<u16>, Error> {
        self.step.write(self.tensors.patch, patches)?;
        self.step.submit()?;
        self.step.read(self.tensors.out)
    }

    /// Each dispatch's label and GPU time in the last run, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn tensors(&self) -> &EncoderGraph {
        &self.tensors
    }

    /// The recorded encoder's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}
