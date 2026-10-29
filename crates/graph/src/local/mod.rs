//! VoxCPM2's local stacks, the work around the LMs for each patch: the sampler with its local DiT
//! ([`dit`]) and the local encoder ([`encoder`]). Each is a bidirectional MiniCPM-4 stack over the
//! few tokens of one patch, run here in sequences of [`Shape::capacity`] tokens, the positions
//! RoPE rotates them by. Per layer: the q/k/v projection and RoPE; attention within each
//! sequence; the output projection; the residual add with the post-attention norm; the gate/up
//! projection and SiLU; the down projection; and the residual add with the next layer's input
//! norm, or after the last the final norm.
//!
//! The projections are products of cooperative matrices, 16 x 16 tiles of f16 summed in f32
//! (`mma.glsl`): each weight is read once for a block of tokens, and a token's products are the
//! same whatever else is in its block, so that each sequence comes out as it would alone.

pub mod dit;
pub mod encoder;

use Access::{Read, Write};
use serde::Deserialize;
use vulkan::{DeviceReport, F16, SpecValue};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{LmConfig, SUBGROUPS, Shape},
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::{Access, KernelInfo},
    step::Step,
};

/// `local_norm.comp`: x, gain -> out.
pub static LOCAL_NORM: KernelInfo = KernelInfo {
    name: "local_norm",
    access: &[Read, Read, Write],
};

/// `local_qkv_rope.comp`: x, W, cos, sin -> q, k, v.
pub static LOCAL_QKV_ROPE: KernelInfo = KernelInfo {
    name: "local_qkv_rope",
    access: &[Read, Read, Read, Read, Write, Write, Write],
};

/// `local_attention.comp`: q, k, v -> out.
pub static LOCAL_ATTENTION: KernelInfo = KernelInfo {
    name: "local_attention",
    access: &[Read, Read, Read, Write],
};

/// `local_partial.comp`: x, W -> partial products over parts of the columns.
pub static LOCAL_PARTIAL: KernelInfo = KernelInfo {
    name: "local_partial",
    access: &[Read, Read, Write],
};

/// `local_add_norm.comp`: partial products, residual, gain -> sum, normed.
pub static LOCAL_ADD_NORM: KernelInfo = KernelInfo {
    name: "local_add_norm",
    access: &[Read, Read, Read, Write, Write],
};

/// `local_gate_up.comp`: x, W -> out.
pub static LOCAL_GATE_UP: KernelInfo = KernelInfo {
    name: "local_gate_up",
    access: &[Read, Read, Write],
};

/// `local_rows.comp`: x -> every few of its rows.
pub static LOCAL_ROWS: KernelInfo = KernelInfo {
    name: "local_rows",
    access: &[Read, Write],
};

/// `local_embed.comp`: frames, W, bias, tokens -> some of a stack's input tokens.
pub static LOCAL_EMBED: KernelInfo = KernelInfo {
    name: "local_embed",
    access: &[Read, Read, Read, Read, Write],
};

/// `cfm_noise.comp`: noise, sampling -> x.
pub static CFM_NOISE: KernelInfo = KernelInfo {
    name: "cfm_noise",
    access: &[Read, Read, Write],
};

/// `dit_euler.comp`: hidden states after the final norm, out_proj, bias, sampling -> x, which it
/// also reads.
pub static DIT_EULER: KernelInfo = KernelInfo {
    name: "dit_euler",
    access: &[Read, Read, Read, Read, Write],
};

/// The tokens gemm.glsl's clusters take a row at a time, fixed in the kernels.
pub const CLUSTER: usize = 16;

/// The rows and columns of a cooperative matrix's tile, fixed in `mma.glsl`.
pub const TILE: usize = 16;

/// What a VoxCPM2 `config.json` says of the local stacks; the rest is ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct PatchConfig {
    pub lm_config: LmConfig,
    /// Latent frames to a patch.
    pub patch_size: usize,
    /// Channels of a latent frame.
    pub feat_dim: usize,
    pub encoder_config: StackConfig,
    pub dit_config: DitConfig,
}

/// `encoder_config`, or the stack of `dit_config`.
#[derive(Clone, Debug, Deserialize)]
pub struct StackConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub kv_channels: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DitConfig {
    #[serde(flatten)]
    pub stack: StackConfig,
    /// Refused when set, as voxcpm refuses it: in mean mode the DiT predicts the mean velocity
    /// over a step, from the step's length.
    #[serde(default)]
    pub mean_mode: bool,
    #[serde(default)]
    pub dit_mean_mode: bool,
}

impl PatchConfig {
    /// The local stack `stack` describes, with sequences of `tokens` tokens, derived from the
    /// base LM's as upstream derives it: head count and width are the stack's, the rest the
    /// base LM's.
    pub fn stack(&self, stack: &StackConfig, tokens: usize) -> Result<Shape, Error> {
        let config = LmConfig {
            hidden_size: stack.hidden_dim,
            intermediate_size: stack.ffn_dim,
            num_attention_heads: stack.num_heads,
            num_hidden_layers: stack.num_layers,
            kv_channels: stack.kv_channels,
            ..self.lm_config.clone()
        };
        Shape::new(&config, stack.num_layers, tokens)
    }
}

/// How the local stacks' kernels cut their work: the subgroup size every kernel runs at, the
/// workgroup width, and how the projections share out their tiles of [`TILE`] rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub subgroup: u32,
    pub workgroup: u32,
    /// Pairs of tiles a subgroup takes at once in the q/k/v and the gate/up projections: of the
    /// rows RoPE rotates together, and of the gate's rows with up's.
    pub qkv_pairs: u32,
    pub gate_up_pairs: u32,
    /// Tiles a subgroup takes at once in the output and down projections.
    pub o_tiles: u32,
    pub down_tiles: u32,
    /// The parts the output and down projections' columns are cut into, each summed by
    /// workgroups of its own and joined after in their order: the order of the sums, and so what
    /// a token's products come to, depends on these alone.
    pub o_parts: u32,
    pub down_parts: u32,
    /// The columns of the tokens each projection stages in shared memory at once.
    pub chunk: u32,
    /// The most tokens a workgroup takes, a multiple of [`TILE`].
    pub block: u32,
    /// The steps of [`TILE`] columns ahead of its products that a subgroup loads its weights.
    pub depth: u32,
    /// The workgroups each projection dispatches for each part and block of tokens, which share
    /// its tiles out: one to each workgroup's worth of them when 0.
    pub groups: u32,
}

impl Tuning {
    /// One pair of tiles or one tile to a subgroup, the output and down projections cut into 2
    /// and 4 parts, in workgroups of 256 that each take a workgroup's worth of them, the tokens in
    /// blocks of 32, at `subgroup`; at 64 the tokens staged 512 columns at a time and the weights
    /// loaded 8 steps ahead, at 32 256 columns and 4 steps.
    ///
    /// On the Radeon 8060S at 64 the sampler's ten steps for one request took 24.4 ms on the GPU
    /// so, and 26.2 ms staging 256 columns 4 steps ahead. On the RTX 4070 Ti SUPER at 32 they took
    /// 23.5 ms from submission to fence for 16 requests so, against 30.3 loaded 8 steps ahead,
    /// and no less staging other columns or blocks of tokens.
    pub fn for_subgroup(subgroup: u32) -> Self {
        let wide = subgroup >= 64;
        Tuning {
            subgroup,
            workgroup: 256,
            qkv_pairs: 1,
            gate_up_pairs: 1,
            o_tiles: 1,
            down_tiles: 1,
            o_parts: 2,
            down_parts: 4,
            chunk: if wide { 512 } else { 256 },
            block: 32,
            depth: if wide { 8 } else { 4 },
            groups: 0,
        }
    }

    /// [`for_subgroup`](Self::for_subgroup) at the narrowest subgroup the kernels are written
    /// for that the device can require, [`fit`](Self::fit) to it. On the Radeon 8060S the
    /// sampler's ten steps took 74 to 77 ms on the GPU for 16 requests at 32 and 97 to 99 at
    /// 64, 32 and 36 ms for 4, and 21 to 23 for one at either.
    pub fn for_device(report: &DeviceReport) -> Result<Self, Error> {
        let subgroup = SUBGROUPS
            .into_iter()
            .rev()
            .find(|&s| report.can_require_subgroup_size(s))
            .ok_or_else(|| {
                Error::Shape(format!(
                    "{} requires subgroups of {} to {}, none of {SUBGROUPS:?}",
                    report.name, report.min_subgroup_size, report.max_subgroup_size
                ))
            })?;
        Tuning::for_subgroup(subgroup).fit(report)
    }

    /// This tuning staging fewer columns, down to 256, and then fewer tokens at once, until it
    /// fits the shared memory of the device `report` describes: 48 KiB on NVIDIA's.
    pub fn fit(self, report: &DeviceReport) -> Result<Self, Error> {
        self.fit_within(report.max_shared_memory as usize)
    }

    fn fit_within(mut self, bytes: usize) -> Result<Self, Error> {
        while self.shared_bytes() > bytes {
            if self.chunk > 256 {
                self.chunk /= 2;
            } else if self.block > TILE as u32 {
                self.block -= TILE as u32;
            } else {
                return Err(Error::Shape(format!(
                    "{} bytes of shared memory for a block of {TILE} tokens, of {bytes}",
                    self.shared_bytes()
                )));
            }
        }
        Ok(self)
    }

    /// The most shared memory a projection's workgroup takes: a block of tokens staged, and
    /// each subgroup's pair of tiles on their way out.
    pub fn shared_bytes(&self) -> usize {
        let (block, chunk) = (self.block as usize, self.chunk as usize);
        let subgroups = (self.workgroup / self.subgroup) as usize;
        2 * block * (chunk + 8) + subgroups * 2 * TILE * TILE * 4
    }
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The number of workgroups when each takes `per_group` of `total`, refusing a remainder: the
/// kernels do not check their bounds against it.
pub(crate) fn groups(what: &str, total: usize, per_group: usize) -> Result<u32, Error> {
    if per_group == 0 || !total.is_multiple_of(per_group) {
        return Err(Error::Shape(format!(
            "{what}: {total} is not a multiple of {per_group}"
        )));
    }
    u32::try_from(total / per_group).map_err(|_| Error::Shape(format!("{what}: {total}")))
}

/// Each of `graph`'s tensors where it is: the map for a graph that is not part of another.
pub(crate) fn identity(graph: &Graph) -> Vec<TensorId> {
    (0..graph.tensors.len()).map(TensorId).collect()
}

fn check(ok: bool, why: impl FnOnce() -> String) -> Result<(), Error> {
    if ok { Ok(()) } else { Err(Error::Shape(why())) }
}

/// One layer's weights.
#[derive(Clone, Debug)]
pub struct LayerTensors {
    pub input_norm: TensorId,
    pub qkv: TensorId,
    pub o: TensorId,
    pub post_norm: TensorId,
    pub gate_up: TensorId,
    pub down: TensorId,
}

/// A local stack's weights and tables in a graph, and the pipelines its layers dispatch.
#[derive(Clone, Debug)]
pub struct Stack {
    pub shape: Shape,
    /// Sequences of `shape.capacity` tokens it runs over at once.
    pub seqs: usize,
    pub layers: Vec<LayerTensors>,
    /// The final norm's gain.
    pub norm: TensorId,
    pub cos: TensorId,
    pub sin: TensorId,
    /// The tokens of the projections' blocks, the last padded past the stack's.
    padded: usize,
    pipelines: Pipelines,
}

#[derive(Clone, Debug)]
struct Pipelines {
    norm: (Pipeline, u32),
    qkv: Projection,
    attention: (Pipeline, u32),
    o: Projection,
    o_add: (Pipeline, u32),
    gate_up: Projection,
    down: Projection,
    down_add: (Pipeline, u32),
}

/// A projection's pipeline and its grid: workgroups, parts of the columns, blocks of tokens.
#[derive(Clone, Debug)]
struct Projection {
    pipeline: Pipeline,
    groups: [u32; 3],
    parts: usize,
}

impl Stack {
    /// Adds the weights and rotation tables of `shape`, run over `seqs` sequences, to `g`,
    /// under `name`, with its kernels cut as `tuning` says.
    pub fn new(
        g: &mut Graph,
        name: &str,
        shape: &Shape,
        seqs: usize,
        tuning: &Tuning,
    ) -> Result<Self, Error> {
        let s = shape;
        let m = seqs * s.capacity;
        let Tuning {
            subgroup,
            workgroup,
            ..
        } = *tuning;
        check(
            workgroup.is_multiple_of(subgroup) && subgroup.is_multiple_of(CLUSTER as u32),
            || format!("workgroups of {workgroup} in subgroups of {subgroup}"),
        )?;
        let block = tuning.block as usize;
        check(block > 0 && block.is_multiple_of(TILE), || {
            format!("blocks of {block} tokens")
        })?;
        check(
            s.heads.is_multiple_of(s.kv_heads)
                && s.head_dim.is_multiple_of(2 * TILE)
                && s.hidden.is_multiple_of(TILE * CLUSTER),
            || {
                format!(
                    "{} heads over {} of {}, {} wide",
                    s.heads, s.kv_heads, s.head_dim, s.hidden
                )
            },
        )?;
        // The fewest blocks that hold the tokens, each the fewest tiles that hold its share.
        let blocks = m.div_ceil(block);
        let block = m.div_ceil(blocks).next_multiple_of(TILE);
        let subgroups = (workgroup / subgroup) as usize;
        let sub = Some(subgroup);
        let common = [
            (0, SpecValue::U32(subgroup)),
            (1, SpecValue::U32(workgroup)),
        ];
        let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
            kernel,
            constants: common.iter().chain(constants).copied().collect(),
            subgroup_size: sub,
        };
        // A projection over k columns cut into `parts`, its tiles in `units` of a subgroup each;
        // the staged columns, the token block and the steps loaded ahead are its last three
        // constants.
        let projection = |kernel,
                          k: usize,
                          units: usize,
                          parts: usize,
                          constants: &[(u32, SpecValue)]|
         -> Result<Projection, Error> {
            let part = k / parts.max(1);
            let chunk = (tuning.chunk as usize).min(part);
            let depth = tuning.depth as usize;
            check(
                parts > 0
                    && k.is_multiple_of(parts)
                    && chunk.is_multiple_of(TILE)
                    && part.is_multiple_of(chunk)
                    && depth > 0
                    && (chunk / TILE).is_multiple_of(depth),
                || format!("{k} columns in {parts} parts, chunks of {chunk}, {depth} steps ahead"),
            )?;
            check(part == chunk || units.is_multiple_of(subgroups), || {
                format!("{units} units restaged by {subgroups} subgroups")
            })?;
            let wide = units.div_ceil(subgroups) as u32;
            let next = constants.len() as u32 + 2;
            let shared = [
                (next, u(chunk)),
                (next + 1, u(block)),
                (next + 2, SpecValue::U32(tuning.depth)),
            ];
            let constants: Vec<_> = constants.iter().chain(&shared).copied().collect();
            Ok(Projection {
                pipeline: pipeline(kernel, &constants),
                groups: [
                    match tuning.groups {
                        0 => wide,
                        groups => groups.min(wide),
                    },
                    parts as u32,
                    blocks as u32,
                ],
                parts,
            })
        };
        let pairs = |what: &str, total: usize, per: u32| -> Result<usize, Error> {
            check(per > 0 && total.is_multiple_of(per as usize), || {
                format!("{what}: {total} in units of {per}")
            })?;
            Ok(total / per as usize)
        };
        let eps = SpecValue::F32(s.eps);
        let heads = s.heads + 2 * s.kv_heads;
        let qkv = projection(
            &LOCAL_QKV_ROPE,
            s.hidden,
            pairs(
                "q/k/v tile pairs",
                heads * s.head_dim / 2 / TILE,
                tuning.qkv_pairs,
            )?,
            1,
            &[
                (2, u(s.hidden)),
                (3, u(m)),
                (4, u(s.capacity)),
                (5, u(s.heads)),
                (6, u(s.kv_heads)),
                (7, u(s.head_dim)),
                (8, SpecValue::U32(tuning.qkv_pairs)),
            ],
        )?;
        let attention = pipeline(
            &LOCAL_ATTENTION,
            &[
                (2, u(m)),
                (3, u(s.capacity)),
                (4, u(s.heads)),
                (5, u(s.kv_heads)),
                (6, u(s.head_dim)),
                (7, SpecValue::F32(1.0 / (s.head_dim as f32).sqrt())),
            ],
        );
        let clusters = (workgroup as usize) / CLUSTER;
        let attention_groups = (m * s.heads).div_ceil(clusters) as u32;
        let residual = |k: usize, tiles: u32, parts: u32| -> Result<Projection, Error> {
            projection(
                &LOCAL_PARTIAL,
                k,
                pairs("residual tiles", s.hidden / TILE, tiles)?,
                parts as usize,
                &[
                    (2, u(k)),
                    (3, u(m)),
                    (4, u(s.hidden)),
                    (5, SpecValue::U32(tiles)),
                    (6, SpecValue::U32(parts)),
                ],
            )
        };
        let padded = blocks * block;
        let add = |parts: usize| {
            pipeline(
                &LOCAL_ADD_NORM,
                &[(2, u(s.hidden)), (3, u(parts)), (4, u(padded)), (5, eps)],
            )
        };
        let o = residual(s.heads * s.head_dim, tuning.o_tiles, tuning.o_parts)?;
        let down = residual(s.inner, tuning.down_tiles, tuning.down_parts)?;
        let gate_up = projection(
            &LOCAL_GATE_UP,
            s.hidden,
            pairs("gate/up tiles", s.inner / TILE, tuning.gate_up_pairs)?,
            1,
            &[
                (2, u(s.hidden)),
                (3, u(m)),
                (4, u(s.inner)),
                (5, SpecValue::U32(tuning.gate_up_pairs)),
            ],
        )?;
        let tokens = m as u32;
        let pipelines = Pipelines {
            norm: (pipeline(&LOCAL_NORM, &[(2, u(s.hidden)), (3, eps)]), tokens),
            qkv,
            attention: (attention, attention_groups),
            o_add: (add(o.parts), tokens),
            o,
            gate_up,
            down_add: (add(down.parts), tokens),
            down,
        };

        let vector = |n: usize| 2 * n as u64;
        let table = (s.capacity * s.head_dim / 2 * 4) as u64;
        let cos = g.tensor(format!("{name}.rope.cos"), table, Kind::Constant);
        let sin = g.tensor(format!("{name}.rope.sin"), table, Kind::Constant);
        let qkv_rows = heads * s.head_dim;
        let layers = (0..s.layers)
            .map(|l| {
                let mut weight = |what: &str, n: usize| {
                    g.tensor(
                        format!("{name}.layers.{l}.{what}"),
                        vector(n),
                        Kind::Constant,
                    )
                };
                LayerTensors {
                    input_norm: weight("input_layernorm", s.hidden),
                    qkv: weight("qkv_proj", qkv_rows * s.hidden),
                    o: weight("o_proj", s.hidden * s.heads * s.head_dim),
                    post_norm: weight("post_attention_layernorm", s.hidden),
                    gate_up: weight("gate_up_proj", 2 * s.inner * s.hidden),
                    down: weight("down_proj", s.hidden * s.inner),
                }
            })
            .collect();
        let norm = g.tensor(format!("{name}.norm"), vector(s.hidden), Kind::Constant);
        Ok(Stack {
            shape: shape.clone(),
            seqs,
            layers,
            norm,
            cos,
            sin,
            padded,
            pipelines,
        })
    }

    /// The tokens it runs over at once.
    pub fn tokens(&self) -> usize {
        self.seqs * self.shape.capacity
    }

    /// Appends the layers over the tokens `x`, `[tokens][hidden]` in f16, to `g`, labelled after
    /// `name`; returns their output after the final norm, an intermediate of `g`.
    pub fn layers(&self, g: &mut Graph, name: &str, x: TensorId) -> TensorId {
        let s = &self.shape;
        let m = self.tokens();
        let p = &self.pipelines;
        let label = |l: usize, what: &str| format!("{name}.layers.{l}.{what}");
        let intermediate = |g: &mut Graph, l: usize, what: &str, bytes: usize| {
            g.tensor(label(l, what), bytes as u64, Kind::Intermediate)
        };
        let half = |n: usize| 2 * m * n;
        let partials = |parts: usize| 4 * parts * self.padded * s.hidden;
        let op = |g: &mut Graph,
                  label: String,
                  (pipeline, groups): &(Pipeline, u32),
                  bindings: &[TensorId]| {
            g.op(label, pipeline.clone(), bindings, [*groups, 1, 1]);
        };
        let project = |g: &mut Graph, label: String, how: &Projection, bindings: &[TensorId]| {
            g.op(label, how.pipeline.clone(), bindings, how.groups);
        };
        let mut normed = intermediate(g, 0, "normed", half(s.hidden));
        op(
            g,
            label(0, "input_norm"),
            &p.norm,
            &[x, self.layers[0].input_norm, normed],
        );
        let mut input = x;
        for (l, t) in self.layers.iter().enumerate() {
            let q = intermediate(g, l, "q", half(s.heads * s.head_dim));
            let k = intermediate(g, l, "k", half(s.kv_heads * s.head_dim));
            let v = intermediate(g, l, "v", half(s.kv_heads * s.head_dim));
            let attended = intermediate(g, l, "attended", half(s.heads * s.head_dim));
            let o = intermediate(g, l, "o_proj.partial", partials(p.o.parts));
            let mid = intermediate(g, l, "mid", half(s.hidden));
            let mid_normed = intermediate(g, l, "mid_normed", half(s.hidden));
            let mlp = intermediate(g, l, "mlp", half(s.inner));
            let down = intermediate(g, l, "down_proj.partial", partials(p.down.parts));
            let output = intermediate(g, l, "out", half(s.hidden));
            let output_normed = intermediate(g, l, "out_normed", half(s.hidden));
            project(
                g,
                label(l, "qkv_rope"),
                &p.qkv,
                &[normed, t.qkv, self.cos, self.sin, q, k, v],
            );
            op(g, label(l, "attention"), &p.attention, &[q, k, v, attended]);
            project(g, label(l, "o_proj"), &p.o, &[attended, t.o, o]);
            op(
                g,
                label(l, "o_proj_add"),
                &p.o_add,
                &[o, input, t.post_norm, mid, mid_normed],
            );
            project(
                g,
                label(l, "gate_up"),
                &p.gate_up,
                &[mid_normed, t.gate_up, mlp],
            );
            project(g, label(l, "down_proj"), &p.down, &[mlp, t.down, down]);
            let gain = self
                .layers
                .get(l + 1)
                .map_or(self.norm, |next| next.input_norm);
            op(
                g,
                label(l, "down_proj_add"),
                &p.down_add,
                &[down, mid, gain, output, output_normed],
            );
            input = output;
            normed = output_normed;
        }
        normed
    }

    /// Writes the rotation tables, and the weights of the stack whose tensors are named after
    /// `prefix` in `checkpoint`, as upstream's state_dict names them, converted to f16, into
    /// `step`, a recording of `graph`, where `ids` puts each tensor of the graph the stack was
    /// added to. The projections' weights are [`tiled`].
    pub fn load(
        &self,
        step: &mut Step,
        graph: &Graph,
        ids: &[TensorId],
        checkpoint: &Checkpoint,
        prefix: &str,
    ) -> Result<(), Error> {
        let at = |t: TensorId| ids[t.0];
        let (cos, sin) = self.shape.rope_tables();
        step.write(at(self.cos), &cos)?;
        step.write(at(self.sin), &sin)?;
        for (l, t) in self.layers.iter().enumerate() {
            let n = |what: &str| format!("{prefix}layers.{l}.{what}.weight");
            let [q, k, v] = ["q_proj", "k_proj", "v_proj"].map(|p| n(&format!("self_attn.{p}")));
            let mut tiles = |t: TensorId, names: &[String], k: usize| {
                let values = read_weight(graph, checkpoint, at(t), names)?;
                step.write(at(t), &tiled(&values, k))
            };
            let s = &self.shape;
            tiles(t.qkv, &[q, k, v], s.hidden)?;
            tiles(t.o, &[n("self_attn.o_proj")], s.heads * s.head_dim)?;
            let [gate, up] = ["gate_proj", "up_proj"].map(|p| n(&format!("mlp.{p}")));
            tiles(t.gate_up, &[gate, up], s.hidden)?;
            tiles(t.down, &[n("mlp.down_proj")], s.inner)?;
            write_weight(
                step,
                graph,
                checkpoint,
                at(t.input_norm),
                &[n("input_layernorm")],
            )?;
            write_weight(
                step,
                graph,
                checkpoint,
                at(t.post_norm),
                &[n("post_attention_layernorm")],
            )?;
        }
        write_weight(
            step,
            graph,
            checkpoint,
            at(self.norm),
            &[format!("{prefix}norm.weight")],
        )
    }
}

/// Writes the tensors `names` of `checkpoint` into tensor `t` of `step`, recorded from `graph`,
/// as f16, joined along their rows when there are several, as upstream's projections are
/// joined to run as one.
pub(crate) fn write_weight(
    step: &mut Step,
    graph: &Graph,
    checkpoint: &Checkpoint,
    t: TensorId,
    names: &[String],
) -> Result<(), Error> {
    let values = read_weight(graph, checkpoint, t, names)?;
    step.write(t, &values)
}

/// What [`write_weight`] writes.
fn read_weight(
    graph: &Graph,
    checkpoint: &Checkpoint,
    t: TensorId,
    names: &[String],
) -> Result<Vec<F16>, Error> {
    let values = match names {
        [name] => checkpoint.f16(name)?.1,
        _ => checkpoint.rows_f16(names)?.1,
    };
    let want = graph.tensors[t.0].bytes as usize / 2;
    if values.len() != want {
        return Err(Error::Checkpoint(format!(
            "{names:?}: {} values, not {want}",
            values.len()
        )));
    }
    Ok(values)
}

/// A row-major matrix of `k` columns in tiles of [`TILE`] rows by [`TILE`] columns, each
/// row-major, a row of tiles' one after another: as `mma.glsl` reads a weight, so that each
/// subgroup reads its rows' columns in one stream. The rows and columns are multiples of
/// [`TILE`], as [`Stack::new`] has them.
pub fn tiled(values: &[F16], k: usize) -> Vec<F16> {
    let n = values.len() / k;
    let mut out = Vec::with_capacity(values.len());
    for rows in 0..n / TILE {
        for columns in 0..k / TILE {
            for r in 0..TILE {
                let at = (rows * TILE + r) * k + columns * TILE;
                out.extend_from_slice(&values[at..at + TILE]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_run_as_they_are_on_the_devices_they_were_measured_on() {
        // NVIDIA's 48 KiB of shared memory at 32, the Radeon's 64 at either.
        for (subgroup, bytes) in [(32, 48 << 10), (32, 64 << 10), (64, 64 << 10)] {
            let tuning = Tuning::for_subgroup(subgroup);
            assert_eq!(
                tuning.fit_within(bytes).unwrap(),
                tuning,
                "{subgroup} in {bytes}"
            );
        }
        let at = |subgroup| {
            let t = Tuning::for_subgroup(subgroup);
            (t.chunk, t.depth)
        };
        assert_eq!(at(32), (256, 4));
        assert_eq!(at(64), (512, 8));
    }

    #[test]
    fn a_tuning_too_large_stages_fewer_columns_then_fewer_tokens() {
        let wide = Tuning {
            chunk: 1024,
            block: 64,
            ..Tuning::for_subgroup(32)
        };
        let fitted = wide.fit_within(48 << 10).unwrap();
        assert_eq!((fitted.chunk, fitted.block), (256, 48));
        assert!(fitted.shared_bytes() <= 48 << 10);
        let fitted = wide.fit_within(32 << 10).unwrap();
        assert_eq!((fitted.chunk, fitted.block), (256, 16));
        assert!(wide.fit_within(16 << 10).is_err());
    }

    #[test]
    fn tiles_hold_16_rows_of_16_columns_and_follow_each_other_along_the_rows() {
        let (n, k) = (2 * TILE, 3 * TILE);
        let values: Vec<F16> = (0..n * k).map(|v| F16::from_bits(v as u16)).collect();
        let tiles = tiled(&values, k);
        assert_eq!(tiles.len(), values.len());
        for row in 0..n {
            for column in 0..k {
                let tile = row / TILE * (k / TILE) + column / TILE;
                let at = tile * TILE * TILE + row % TILE * TILE + column % TILE;
                assert_eq!(tiles[at], values[row * k + column], "({row}, {column})");
            }
        }
    }
}
