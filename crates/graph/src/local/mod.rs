//! VoxCPM2's local stacks at batch 1, the work around the LMs for each patch: the sampler with
//! its local DiT ([`dit`]) and the local encoder ([`encoder`]). Each is a bidirectional
//! MiniCPM-4 stack over the few tokens of one patch, run here in sequences of
//! [`Shape::capacity`] tokens, the positions RoPE rotates them by. Its kernels read each weight
//! once for every token, which is what a stack's time goes to: per layer the input norm, q/k/v
//! projection and RoPE in one dispatch; attention within each sequence; the output projection
//! with the residual; the post-attention norm with the gate/up projection and SiLU; and the down
//! projection with the residual.

pub mod dit;
pub mod encoder;

use Access::{Read, Write};
use serde::Deserialize;
use vulkan::{DeviceReport, SpecValue};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{self, LmConfig, Shape},
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::{Access, KernelInfo},
    step::Step,
};

/// `local_qkv_rope.comp`: x, gain, W, cos, sin -> q, k, v.
pub static LOCAL_QKV_ROPE: KernelInfo = KernelInfo {
    name: "local_qkv_rope",
    access: &[Read, Read, Read, Read, Read, Write, Write, Write],
};

/// `local_attention.comp`: q, k, v -> out.
pub static LOCAL_ATTENTION: KernelInfo = KernelInfo {
    name: "local_attention",
    access: &[Read, Read, Read, Write],
};

/// `local_residual.comp`: x, W, residual -> out.
pub static LOCAL_RESIDUAL: KernelInfo = KernelInfo {
    name: "local_residual",
    access: &[Read, Read, Read, Write],
};

/// `local_gate_up.comp`: x, gain, W -> out.
pub static LOCAL_GATE_UP: KernelInfo = KernelInfo {
    name: "local_gate_up",
    access: &[Read, Read, Read, Write],
};

/// `local_partial.comp`: x, W -> partial products over chunks of the columns.
pub static LOCAL_PARTIAL: KernelInfo = KernelInfo {
    name: "local_partial",
    access: &[Read, Read, Write],
};

/// `local_combine.comp`: partial products, residual -> out.
pub static LOCAL_COMBINE: KernelInfo = KernelInfo {
    name: "local_combine",
    access: &[Read, Read, Write],
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

/// `dit_euler.comp`: hidden states, gain, out_proj, bias, sampling -> x, which it also reads.
pub static DIT_EULER: KernelInfo = KernelInfo {
    name: "dit_euler",
    access: &[Read, Read, Read, Read, Read, Write],
};

/// The tokens gemm.glsl's clusters take a row at a time, fixed in the kernels.
pub const CLUSTER: usize = 16;

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
/// workgroup width, and the rows each cluster of [`CLUSTER`] lanes takes in each projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub subgroup: u32,
    pub workgroup: u32,
    /// Row pairs per cluster of the q/k/v and the gate/up projections.
    pub qkv_pairs: u32,
    pub gate_up_pairs: u32,
    /// Rows per cluster of the output and down projections.
    pub o_rows: u32,
    pub down_rows: u32,
    /// The columns of the tokens each projection stages at once.
    pub chunk: u32,
    /// Whether the output and down projections are cut along their columns too, a chunk to
    /// each row of workgroups, and joined with the residual by a dispatch of their own.
    pub split: bool,
    /// The workgroups each projection dispatches, which share its rows out, over its parts when
    /// it is cut along its columns: one to each workgroup's worth of rows when 0.
    pub groups: u32,
}

impl Tuning {
    /// One row or one pair of rows to a cluster, in workgroups of 256 that each take a
    /// workgroup's worth of them, at `subgroup`.
    pub fn for_subgroup(subgroup: u32) -> Self {
        Tuning {
            subgroup,
            workgroup: 256,
            qkv_pairs: 1,
            gate_up_pairs: 1,
            o_rows: 1,
            down_rows: 1,
            chunk: 1024,
            split: false,
            groups: 0,
        }
    }

    /// [`for_subgroup`](Self::for_subgroup) at the widest subgroup the kernels are written for
    /// that the device can require; on a device that says how many compute units it has, one
    /// workgroup to each, the output and down projections cut along their columns. On the
    /// Radeon 8060S, whose 40 compute units each hold one workgroup's 22 tokens at once, the
    /// sampler's ten steps took 28.6 ms on the GPU so, and about 50 ms at one workgroup to each
    /// workgroup's worth of rows.
    pub fn for_device(report: &DeviceReport) -> Result<Self, Error> {
        let base = decode::Tuning::for_device(report)?;
        let units = report
            .shader_core
            .and_then(|core| core.active_compute_units);
        Ok(match units {
            Some(units) => Tuning {
                split: true,
                groups: units,
                ..Tuning::for_subgroup(base.subgroup)
            },
            None => Tuning::for_subgroup(base.subgroup),
        })
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

/// The sets of `per_slot` of `total` rows the clusters take, and the workgroups to dispatch over
/// them: `groups`, or one to each `clusters` of them when 0, and never more.
fn slots(
    what: &str,
    total: usize,
    per_slot: usize,
    clusters: usize,
    groups: u32,
) -> Result<(u32, u32), Error> {
    let slots = self::groups(what, total, per_slot)?;
    let blocks = self::groups(what, slots as usize, clusters)?;
    Ok((
        slots,
        if groups == 0 {
            blocks
        } else {
            groups.min(blocks)
        },
    ))
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
    /// The final norm's gain, which the stack's user applies.
    pub norm: TensorId,
    pub cos: TensorId,
    pub sin: TensorId,
    pipelines: Pipelines,
}

#[derive(Clone, Debug)]
struct Pipelines {
    qkv: (Pipeline, u32),
    attention: (Pipeline, u32),
    o: Residual,
    gate_up: (Pipeline, u32),
    down: Residual,
}

/// How a projection with the residual add after it is dispatched.
#[derive(Clone, Debug)]
enum Residual {
    /// At once.
    Whole(Pipeline, u32),
    /// Over `parts` chunks of its columns, and joined with the residual after.
    Split {
        partial: (Pipeline, [u32; 3]),
        combine: (Pipeline, u32),
        parts: usize,
    },
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
        let chunk = tuning.chunk as usize;
        let step = 8 * CLUSTER;
        check(
            s.hidden.is_multiple_of(chunk.min(s.hidden))
                && chunk.is_multiple_of(step)
                && (s.heads * s.head_dim).is_multiple_of(chunk)
                && s.inner.is_multiple_of(chunk),
            || {
                format!(
                    "{} and {} columns in chunks of {chunk}, steps of {step}",
                    s.heads * s.head_dim,
                    s.inner
                )
            },
        )?;
        check(
            s.heads.is_multiple_of(s.kv_heads) && s.head_dim.is_multiple_of(step),
            || format!("{} heads over {} of {}", s.heads, s.kv_heads, s.head_dim),
        )?;
        let clusters = (workgroup as usize) / CLUSTER;
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
        let eps = SpecValue::F32(s.eps);
        let (qkv_slots, qkv_groups) = slots(
            "q/k/v row pairs",
            (s.heads + 2 * s.kv_heads) * s.head_dim / 2,
            tuning.qkv_pairs as usize,
            clusters,
            tuning.groups,
        )?;
        let qkv = pipeline(
            &LOCAL_QKV_ROPE,
            &[
                (2, u(s.hidden)),
                (3, u(m)),
                (4, u(s.capacity)),
                (5, u(s.heads)),
                (6, u(s.kv_heads)),
                (7, u(s.head_dim)),
                (8, eps),
                (9, SpecValue::U32(tuning.qkv_pairs)),
                (10, u(chunk.min(s.hidden))),
                (11, SpecValue::U32(qkv_slots)),
            ],
        );
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
        let attention_groups = (m * s.heads).div_ceil(clusters) as u32;
        let residual = |k: usize, rows: u32| -> Result<Residual, Error> {
            let parts = k / chunk.min(k);
            let groups = match tuning.groups {
                0 => 0,
                all if tuning.split => (all / parts as u32).max(1),
                all => all,
            };
            let (slots, groups) =
                slots("residual rows", s.hidden, rows as usize, clusters, groups)?;
            let constants = |kernel| {
                pipeline(
                    kernel,
                    &[
                        (2, u(k)),
                        (3, u(chunk.min(k))),
                        (4, u(m)),
                        (5, u(s.hidden)),
                        (6, SpecValue::U32(rows)),
                        (7, SpecValue::U32(slots)),
                    ],
                )
            };
            if !tuning.split {
                return Ok(Residual::Whole(constants(&LOCAL_RESIDUAL), groups));
            }
            let len = m * s.hidden;
            let combine = pipeline(&LOCAL_COMBINE, &[(2, u(len)), (3, u(parts))]);
            Ok(Residual::Split {
                partial: (constants(&LOCAL_PARTIAL), [groups, parts as u32, 1]),
                combine: (combine, len.div_ceil(workgroup as usize) as u32),
                parts,
            })
        };
        let (gate_up_slots, gate_up_groups) = slots(
            "gate/up row pairs",
            s.inner,
            tuning.gate_up_pairs as usize,
            clusters,
            tuning.groups,
        )?;
        let gate_up = pipeline(
            &LOCAL_GATE_UP,
            &[
                (2, u(s.hidden)),
                (3, u(m)),
                (4, u(s.inner)),
                (5, eps),
                (6, SpecValue::U32(tuning.gate_up_pairs)),
                (7, u(chunk.min(s.hidden))),
                (8, SpecValue::U32(gate_up_slots)),
            ],
        );
        let pipelines = Pipelines {
            qkv: (qkv, qkv_groups),
            attention: (attention, attention_groups),
            o: residual(s.heads * s.head_dim, tuning.o_rows)?,
            gate_up: (gate_up, gate_up_groups),
            down: residual(s.inner, tuning.down_rows)?,
        };

        let vector = |n: usize| 2 * n as u64;
        let table = (s.capacity * s.head_dim / 2 * 4) as u64;
        let cos = g.tensor(format!("{name}.rope.cos"), table, Kind::Constant);
        let sin = g.tensor(format!("{name}.rope.sin"), table, Kind::Constant);
        let qkv_rows = (s.heads + 2 * s.kv_heads) * s.head_dim;
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
            pipelines,
        })
    }

    /// The tokens it runs over at once.
    pub fn tokens(&self) -> usize {
        self.seqs * self.shape.capacity
    }

    /// Appends the layers over the tokens `x`, `[tokens][hidden]` in f16, to `g`, labelled after
    /// `name`; returns their output before the final norm, an intermediate of `g`.
    pub fn layers(&self, g: &mut Graph, name: &str, x: TensorId) -> TensorId {
        let s = &self.shape;
        let m = self.tokens();
        let vector = |n: usize| 2 * (m * n) as u64;
        let p = &self.pipelines;
        let mut input = x;
        for (l, t) in self.layers.iter().enumerate() {
            let label = |what: &str| format!("{name}.layers.{l}.{what}");
            let mut intermediate =
                |what: &str, n: usize| g.tensor(label(what), vector(n), Kind::Intermediate);
            let q = intermediate("q", s.heads * s.head_dim);
            let k = intermediate("k", s.kv_heads * s.head_dim);
            let v = intermediate("v", s.kv_heads * s.head_dim);
            let attended = intermediate("attended", s.heads * s.head_dim);
            let mid = intermediate("mid", s.hidden);
            let mlp = intermediate("mlp", s.inner);
            let output = intermediate("out", s.hidden);
            let op = |g: &mut Graph,
                      what: &str,
                      (pipeline, groups): &(Pipeline, u32),
                      bindings: &[TensorId]| {
                g.op(label(what), pipeline.clone(), bindings, [*groups, 1, 1]);
            };
            op(
                g,
                "qkv_rope",
                &p.qkv,
                &[input, t.input_norm, t.qkv, self.cos, self.sin, q, k, v],
            );
            op(g, "attention", &p.attention, &[q, k, v, attended]);
            self.residual(g, &label("o_proj"), &p.o, [attended, t.o, input, mid]);
            op(
                g,
                "gate_up",
                &p.gate_up,
                &[mid, t.post_norm, t.gate_up, mlp],
            );
            self.residual(g, &label("down_proj"), &p.down, [mlp, t.down, mid, output]);
            input = output;
        }
        input
    }

    /// Appends the projection `x` -> `out` by `w` with the residual `residual` added after, as
    /// `how` dispatches it, labelled `label`.
    fn residual(
        &self,
        g: &mut Graph,
        label: &str,
        how: &Residual,
        [x, w, residual, out]: [TensorId; 4],
    ) {
        match how {
            Residual::Whole(pipeline, groups) => {
                g.op(
                    label,
                    pipeline.clone(),
                    &[x, w, residual, out],
                    [*groups, 1, 1],
                );
            }
            Residual::Split {
                partial,
                combine,
                parts,
            } => {
                let bytes = 4 * (parts * self.tokens() * self.shape.hidden) as u64;
                let partials = g.tensor(format!("{label}.partial"), bytes, Kind::Intermediate);
                g.op(label, partial.0.clone(), &[x, w, partials], partial.1);
                g.op(
                    format!("{label}_combine"),
                    combine.0.clone(),
                    &[partials, residual, out],
                    [combine.1, 1, 1],
                );
            }
        }
    }

    /// Writes the rotation tables, and the weights of the stack whose tensors are named after
    /// `prefix` in `checkpoint`, as upstream's state_dict names them, converted to f16, into
    /// `step`, a recording of `graph`, where `ids` puts each tensor of the graph the stack was
    /// added to.
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
            write_weight(step, graph, checkpoint, at(t.qkv), &[q, k, v])?;
            write_weight(step, graph, checkpoint, at(t.o), &[n("self_attn.o_proj")])?;
            let [gate, up] = ["gate_proj", "up_proj"].map(|p| n(&format!("mlp.{p}")));
            write_weight(step, graph, checkpoint, at(t.gate_up), &[gate, up])?;
            write_weight(step, graph, checkpoint, at(t.down), &[n("mlp.down_proj")])?;
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
    step.write(t, &values)
}
