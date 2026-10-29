//! One decode step of a MiniCPM-4 stack, as VoxCPM2's base LM runs it: per layer, the input
//! norm, q/k/v projection, RoPE and the cache write in one dispatch, attention over the cache
//! split along it and joined, the output projection with the residual, the post-attention norm
//! with the gate/up projection and SiLU, and the down projection with the residual; then the
//! final norm. Six dispatches a layer and one more, recorded once for a cache capacity and
//! submitted at every position, which is read from a buffer the host writes before each.
//!
//! A step runs a [`Batch`] of rows, each a request at its own position in its own slot of the
//! cache: the projections read each weight once for all of them, so that a step of several rows
//! takes about as long as a step of one. A row's sums are taken in the same order whatever the
//! batch and however the kernels are cut, so that it comes out bit for bit as it does alone.

use serde::Deserialize;
use vulkan::{Context, DeviceReport, F16, SpecValue, Submission, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    graph::{Graph, Kind, Pipeline, Tensor, TensorId},
    kernels::{
        ATTENTION_COMBINE, ATTENTION_SPLIT, GEMV_GATE_UP, GEMV_QKV_ROPE, GEMV_RESIDUAL, Kernels,
        RMSNORM,
    },
    step::Step,
};

/// The positions attention_split takes per workgroup, fixed in the kernel.
pub const CHUNK: usize = 32;

/// What a step reads of VoxCPM2's `lm_config`; the rest is ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct LmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub kv_channels: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_scaling: RopeScaling,
    /// As upstream assumes when it is missing.
    #[serde(default = "upstream_use_mup")]
    pub use_mup: bool,
    /// Whether attention goes without RoPE, as VoxCPM2's residual LM does: its stack is the base
    /// LM's with this taken from `residual_lm_no_rope`.
    #[serde(default)]
    pub no_rope: bool,
}

fn upstream_use_mup() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub kind: String,
    pub short_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
}

/// The shapes a step is recorded at.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub hidden: usize,
    pub inner: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    /// The positions the cache holds; a multiple of [`CHUNK`].
    pub capacity: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// LongRoPE's divisors of each frequency, at the native length (the only one VoxCPM2 uses).
    pub rope_factors: Vec<f64>,
    /// Whether q and k are rotated.
    pub rope: bool,
}

impl Shape {
    /// The shape of the stack `config` describes, with its first `layers` layers and a cache of
    /// `capacity` positions. Refuses what VoxCPM2 does not use.
    pub fn new(config: &LmConfig, layers: usize, capacity: usize) -> Result<Self, Error> {
        let unsupported = |why: String| Err(Error::Shape(why));
        if config.use_mup {
            return unsupported("use_mup".into());
        }
        let rope = &config.rope_scaling;
        if !config.no_rope
            && (rope.kind != "longrope"
                || config.max_position_embeddings != rope.original_max_position_embeddings)
        {
            return unsupported(format!(
                "rope_scaling {:?} from {} positions to {}",
                rope.kind, rope.original_max_position_embeddings, config.max_position_embeddings
            ));
        }
        if layers > config.num_hidden_layers {
            return unsupported(format!("{layers} of {} layers", config.num_hidden_layers));
        }
        if capacity > config.max_position_embeddings {
            return unsupported(format!(
                "a cache of {capacity} past {} positions",
                config.max_position_embeddings
            ));
        }
        Ok(Shape {
            hidden: config.hidden_size,
            inner: config.intermediate_size,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config
                .kv_channels
                .unwrap_or(config.hidden_size / config.num_attention_heads),
            layers,
            capacity,
            eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta as f32,
            rope_factors: rope.short_factor.clone(),
            rope: !config.no_rope,
        })
    }

    /// The bytes of the weights one step reads: every layer's projections and norms, and the
    /// final norm, in f16.
    pub fn weight_bytes(&self) -> u64 {
        let (h, i, d) = (self.hidden, self.inner, self.head_dim);
        let qkv = (self.heads + 2 * self.kv_heads) * d * h;
        let layer = qkv + self.heads * d * h + 2 * i * h + i * h + 2 * h;
        2 * (self.layers * layer + h) as u64
    }

    /// The bytes of keys and values a step at `position` reads, over every layer.
    pub fn cache_bytes(&self, position: usize) -> u64 {
        let read = (position + 1).next_multiple_of(CHUNK).min(self.capacity);
        (self.layers * 2 * self.kv_heads * read * self.head_dim * 2) as u64
    }

    /// LongRoPE's cosines and sines for positions below the capacity, `[capacity][head_dim / 2]`
    /// each: computed in f32 in the order upstream's `MiniCPMLongRoPE` takes, so they round as its
    /// do.
    pub fn rope_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let dim = self.head_dim;
        let theta = self.rope_theta;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32));
        let rates: Vec<(f32, f32)> = self
            .rope_factors
            .iter()
            .zip(inv_freq)
            .map(|(&f, inv)| (1.0 / f as f32, inv))
            .collect();
        let mut cos = Vec::with_capacity(self.capacity * dim / 2);
        let mut sin = Vec::with_capacity(self.capacity * dim / 2);
        for t in 0..self.capacity {
            for &(r, inv) in &rates {
                let f = t as f32 * r * inv;
                cos.push(f.cos());
                sin.push(f.sin());
            }
        }
        (cos, sin)
    }
}

/// How many requests a step runs at once, each a row of its tensors, and the slots of the KV
/// cache they are kept in. Where each row runs is written before each run ([`Batch::placement`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    pub rows: usize,
    /// At least `rows`.
    pub slots: usize,
}

impl Batch {
    /// One request in a cache of one slot: the step at batch 1.
    pub const ONE: Batch = Batch { rows: 1, slots: 1 };

    /// The u32 that say where the rows run: each row's position, then each row's slot when there
    /// is more than one.
    pub fn words(&self) -> usize {
        if self.slots > 1 {
            2 * self.rows
        } else {
            self.rows
        }
    }

    /// The words for `rows`, one to a row of the batch, in caches of `capacity` positions: where
    /// each runs, or none for a spare row, which is put past every position so that it writes
    /// into no cache and attends to nothing. Refuses two rows in one slot, whose writes into it
    /// would race.
    pub fn placement(&self, rows: &[Option<Row>], capacity: usize) -> Result<Vec<u32>, Error> {
        if rows.len() != self.rows {
            return Err(Error::Shape(format!(
                "{} rows for a batch of {}",
                rows.len(),
                self.rows
            )));
        }
        let mut words = vec![u32::MAX; self.words()];
        let mut taken = vec![false; self.slots];
        for (b, row) in rows.iter().enumerate() {
            let Some(Row { position, slot }) = *row else {
                continue;
            };
            if position >= capacity || slot >= self.slots || taken[slot] {
                return Err(Error::Shape(format!(
                    "row {b} at position {position} of {capacity} in slot {slot} of {}, \
                     taken: {}",
                    self.slots,
                    taken.get(slot).copied().unwrap_or(false)
                )));
            }
            taken[slot] = true;
            words[b] = position as u32;
            if self.slots > 1 {
                words[self.rows + b] = slot as u32;
            }
        }
        Ok(words)
    }
}

/// Where a row of a batch runs: at a position of the cache slot its request is kept in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub position: usize,
    pub slot: usize,
}

/// How the kernels' work is cut: the subgroup size every kernel runs at (32 or 64 on RADV,
/// 32 on NVIDIA), the GEMVs' workgroup width, the rows each of their subgroups takes, how
/// many of a lane's loads they issue before their prologue, and how much of the vectors they
/// stage at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub subgroup: u32,
    pub workgroup: u32,
    /// Row pairs per subgroup of the q/k/v and the gate/up projections.
    pub qkv_pairs: u32,
    pub gate_up_pairs: u32,
    /// Rows per subgroup of the output and down projections.
    pub residual_rows: u32,
    /// The steps of its rows each GEMV lane loads before the kernel's prologue, the norm or the
    /// copy of x into shared memory, so that they are in flight while it runs; all of them when
    /// a lane has no more steps than this.
    pub ahead: u32,
    /// The bytes of shared memory a GEMV takes at most, most of them for its vectors, the rows
    /// of a batch it takes: when they are longer, a part of each at a time.
    pub stage: u32,
    /// The rows of a batch a GEMV workgroup takes at most. A batch of more is shared out over
    /// workgroups side by side, which take the same rows of the weight and so read them from
    /// cache after the first.
    pub vectors: u32,
}

impl Tuning {
    /// One row, or one pair of rows, to a subgroup, at `subgroup`, and no loads ahead.
    ///
    /// At 64, one subgroup to a workgroup: the fastest measured on the Radeon 8060S, which
    /// reached 220 GB/s over four layers' weights where a streaming read reaches 240; larger
    /// workgroups and more rows to a subgroup were slower. At 32, four subgroups to a workgroup:
    /// on the RTX 4070 Ti SUPER the base LM's step took 4.69 ms on the GPU so in all but the first
    /// of nine runs (5.59 ms), 4.68 to 4.73 at two or eight subgroups to a workgroup, 4.84 at one,
    /// and no less at two rows or pairs to a subgroup.
    pub fn for_subgroup(subgroup: u32) -> Self {
        Tuning {
            subgroup,
            workgroup: if subgroup >= 64 {
                subgroup
            } else {
                4 * subgroup
            },
            qkv_pairs: 1,
            gate_up_pairs: 1,
            residual_rows: 1,
            ahead: 0,
            stage: 32 << 10,
            vectors: 1,
        }
    }

    /// The cut for a batch of `rows` at `subgroup`: [`for_subgroup`](Self::for_subgroup)'s, a
    /// row of the batch to each workgroup, up to 4 rows. From there, at 64, workgroups of four
    /// subgroups taking 4 rows each; at 32, workgroups of eight taking 8 rows each, staged 16 KiB
    /// at a time. As with `for_subgroup`, each is what was fastest on the device that runs at
    /// that size by default.
    ///
    /// On the Radeon 8060S at 64, the base LM's step at position 481 took 12.7 and 12.9 ms at 2
    /// and 4 rows cut so, about what it takes at batch 1, against 13.3 and 14.3 with every row
    /// in every workgroup. At 8 and 16 rows, four-subgroup workgroups took 14.5 and 20.7 ms with
    /// a row each, 14.7 and 16.8 with four each, and 18.9 and 27.6 with every row in each.
    ///
    /// On the RTX 4070 Ti SUPER at 32, it took 4.63 and 4.83 ms at 2 and 4 rows cut so, against
    /// 4.56 at batch 1. At 8 and 16 rows, eight-subgroup workgroups staging 16 KiB took 5.35 and
    /// 6.46 ms with 8 rows each, 5.57 and 7.49 with four, 7.78 and 13.79 with one, and 6.72 at
    /// 16 with every row in each; staging 32 KiB, 5.60 and 6.83 with 8 rows each. Cut as the
    /// Radeon's, in four-subgroup workgroups taking 4 rows, it took 5.53 and 8.86.
    pub fn for_batch(subgroup: u32, rows: usize) -> Self {
        let base = Self::for_subgroup(subgroup);
        if rows <= 4 {
            return base;
        }
        if subgroup >= 64 {
            Tuning {
                workgroup: 4 * subgroup,
                vectors: 4,
                ..base
            }
        } else {
            Tuning {
                workgroup: 8 * subgroup,
                vectors: 8,
                stage: 16 << 10,
                ..base
            }
        }
    }

    /// The width of the workgroup whose order every norm is summed in, however wide the GEMVs'
    /// are: [`for_subgroup`](Self::for_subgroup)'s, as at batch 1.
    pub fn norm_width(&self) -> u32 {
        Self::for_subgroup(self.subgroup).workgroup
    }

    /// The rows of a batch of `rows` a GEMV workgroup takes, the most that divide them up to
    /// [`vectors`](Self::vectors), and the workgroups side by side that take them all.
    pub(crate) fn split(&self, rows: usize) -> (usize, usize) {
        let most = (self.vectors as usize).clamp(1, rows);
        let taken = (1..=most)
            .rev()
            .find(|v| rows.is_multiple_of(*v))
            .unwrap_or(1);
        (taken, rows / taken)
    }

    /// The columns of each of `rows` vectors of `k` a GEMV stages at once, as its SPAN constant:
    /// 0 when all of them fit in [`stage`](Self::stage) beside the `others` bytes of its shared
    /// memory, and otherwise the most that do in whole steps of its subgroups that divide `k`.
    pub(crate) fn span(&self, k: usize, rows: usize, others: usize) -> Result<u32, Error> {
        let stage = (self.stage as usize).saturating_sub(others);
        if rows * k * 2 <= stage {
            return Ok(0);
        }
        let step = 8 * self.subgroup as usize;
        let steps = k / step;
        (1..=steps)
            .rev()
            .filter(|m| steps.is_multiple_of(*m))
            .map(|m| m * step)
            .find(|&columns| rows * columns * 2 <= stage)
            .map(|columns| columns as u32)
            .ok_or_else(|| {
                Error::Shape(format!(
                    "{rows} rows of {step} columns in {stage} bytes of shared memory"
                ))
            })
    }

    /// [`for_subgroup`](Self::for_subgroup) at the widest subgroup the kernels are written for,
    /// 64 or 32, that the device can require.
    pub fn for_device(report: &DeviceReport) -> Result<Self, Error> {
        SUBGROUPS
            .into_iter()
            .find(|&s| report.can_require_subgroup_size(s))
            .map(Self::for_subgroup)
            .ok_or_else(|| {
                Error::Shape(format!(
                    "{} requires subgroups of {} to {}, none of {SUBGROUPS:?}",
                    report.name, report.min_subgroup_size, report.max_subgroup_size
                ))
            })
    }
}

/// The subgroup sizes the kernels are written for, widest first: a multiple of [`CHUNK`].
pub const SUBGROUPS: [u32; 2] = [64, 32];

/// One layer's tensors.
#[derive(Clone, Debug)]
pub struct LayerTensors {
    pub input_norm: Tensor<F16>,
    pub qkv: Tensor<F16>,
    pub o: Tensor<F16>,
    pub post_norm: Tensor<F16>,
    pub gate_up: Tensor<F16>,
    pub down: Tensor<F16>,
    pub k_cache: Tensor<F16>,
    pub v_cache: Tensor<F16>,
}

impl LayerTensors {
    /// The layer's tensors where `ids` puts each, such as in a graph its graph was spliced into
    /// ([`Tensor::moved`]).
    pub fn moved(&self, ids: &[TensorId]) -> Self {
        LayerTensors {
            input_norm: self.input_norm.moved(ids),
            qkv: self.qkv.moved(ids),
            o: self.o.moved(ids),
            post_norm: self.post_norm.moved(ids),
            gate_up: self.gate_up.moved(ids),
            down: self.down.moved(ids),
            k_cache: self.k_cache.moved(ids),
            v_cache: self.v_cache.moved(ids),
        }
    }
}

/// The step's graph and the tensors the host writes and reads, a row of each for every row of
/// the batch.
#[derive(Clone, Debug)]
pub struct DecodeGraph {
    pub graph: Graph,
    pub shape: Shape,
    pub batch: Batch,
    /// The rows' hidden states, `[rows][hidden]`.
    pub x: Tensor<F16>,
    /// Where the rows run, [`Batch::words`]: at batch 1, the one position.
    pub position: Tensor<u32>,
    /// `[capacity][head_dim / 2]` each.
    pub cos: Tensor<f32>,
    pub sin: Tensor<f32>,
    pub norm: Tensor<F16>,
    /// The rows' outputs after the final norm, `[rows][hidden]`.
    pub out: Tensor<F16>,
    /// Their caches hold `[slots][kv_heads][capacity][head_dim]`.
    pub layers: Vec<LayerTensors>,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The number of workgroups when each takes `per_group` of `total`, refusing a remainder: the
/// kernels do not check their bounds against it.
fn groups(what: &str, total: usize, per_group: usize) -> Result<u32, Error> {
    if per_group == 0 || !total.is_multiple_of(per_group) {
        return Err(Error::Shape(format!(
            "{what}: {total} is not a multiple of {per_group}"
        )));
    }
    u32::try_from(total / per_group).map_err(|_| Error::Shape(format!("{what}: {total}")))
}

/// The decode step of `shape` at batch 1 cut as `tuning` says.
pub fn decode_graph(shape: &Shape, tuning: &Tuning) -> Result<DecodeGraph, Error> {
    decode_batch_graph(shape, tuning, Batch::ONE)
}

/// The decode step of `shape` for `batch` cut as `tuning` says.
pub fn decode_batch_graph(
    shape: &Shape,
    tuning: &Tuning,
    batch: Batch,
) -> Result<DecodeGraph, Error> {
    let s = shape;
    let Tuning {
        subgroup,
        workgroup,
        ..
    } = *tuning;
    let check = |ok: bool, why: String| if ok { Ok(()) } else { Err(Error::Shape(why)) };
    check(
        workgroup.is_multiple_of(subgroup) && subgroup.is_multiple_of(CHUNK as u32),
        format!("workgroups of {workgroup} in subgroups of {subgroup}"),
    )?;
    for k in [s.hidden, s.inner] {
        check(
            k.is_multiple_of(8 * subgroup as usize),
            format!("{k} columns in steps of {}", 8 * subgroup),
        )?;
    }
    check(
        s.capacity.is_multiple_of(CHUNK) && s.capacity > 0,
        format!("a cache of {} positions", s.capacity),
    )?;
    check(
        s.heads.is_multiple_of(s.kv_heads) && s.head_dim.is_multiple_of(8),
        format!("{} heads over {} of {}", s.heads, s.kv_heads, s.head_dim),
    )?;
    let Batch { rows, slots } = batch;
    check(
        rows > 0 && rows <= slots,
        format!("{rows} rows in {slots} slots"),
    )?;
    let group = s.heads / s.kv_heads;
    let subgroups = (workgroup / subgroup) as usize;
    let sub = Some(subgroup);
    let common = [
        (0, SpecValue::U32(subgroup)),
        (1, SpecValue::U32(workgroup)),
    ];
    let pipeline = |kernel, constants: &[(u32, SpecValue)], subgroup_size| Pipeline {
        kernel,
        constants: common.iter().chain(constants).copied().collect(),
        subgroup_size,
    };
    let eps = SpecValue::F32(s.eps);
    let norm_width = SpecValue::U32(tuning.norm_width());
    let (taken, split) = tuning.split(rows);
    let span = |k: usize, others: usize| tuning.span(k, taken, others).map(SpecValue::U32);
    // norm.glsl's partial sums, a float to a subgroup.
    let partials = 4 * subgroups;
    let qkv_pipeline = pipeline(
        &GEMV_QKV_ROPE,
        &[
            (2, u(s.hidden)),
            (3, u(s.heads)),
            (4, u(s.kv_heads)),
            (5, u(s.head_dim)),
            (6, u(s.capacity)),
            (7, eps),
            (8, SpecValue::U32(tuning.qkv_pairs)),
            (9, SpecValue::U32(tuning.ahead)),
            (10, SpecValue::Bool(s.rope)),
            (11, u(taken)),
            (12, u(slots)),
            (13, span(s.hidden, partials)?),
            (14, norm_width),
            (15, u(split)),
        ],
        sub,
    );
    let qkv_groups = groups(
        "q/k/v row pairs",
        (s.heads + 2 * s.kv_heads) * s.head_dim / 2,
        subgroups * tuning.qkv_pairs as usize,
    )? * split as u32;
    let split_pipeline = Pipeline {
        kernel: &ATTENTION_SPLIT,
        constants: vec![
            (0, SpecValue::U32(subgroup)),
            (1, u(group * CHUNK)),
            (2, u(s.heads)),
            (3, u(s.kv_heads)),
            (4, u(s.head_dim)),
            (5, u(s.capacity)),
            (6, SpecValue::F32(1.0 / (s.head_dim as f32).sqrt())),
            (7, u(rows)),
            (8, u(slots)),
        ],
        subgroup_size: sub,
    };
    // One invocation per four of a head's dimensions.
    let combine_width = (s.head_dim / 4).next_multiple_of(32);
    let combine_pipeline = Pipeline {
        kernel: &ATTENTION_COMBINE,
        constants: vec![
            (0, SpecValue::U32(32)),
            (1, u(combine_width)),
            (2, u(s.heads)),
            (3, u(s.head_dim)),
            (4, u(s.capacity)),
            (5, u(rows)),
            (6, u(slots)),
        ],
        subgroup_size: None,
    };
    let residual = |k: usize| -> Result<Pipeline, Error> {
        Ok(pipeline(
            &GEMV_RESIDUAL,
            &[
                (2, u(k)),
                (3, SpecValue::U32(tuning.residual_rows)),
                (4, SpecValue::U32(tuning.ahead)),
                (5, u(s.hidden)),
                (6, u(taken)),
                (7, span(k, 0)?),
                (8, u(split)),
            ],
            sub,
        ))
    };
    let o_pipeline = residual(s.heads * s.head_dim)?;
    let down_pipeline = residual(s.inner)?;
    let residual_groups = groups(
        "residual rows",
        s.hidden,
        subgroups * tuning.residual_rows as usize,
    )? * split as u32;
    let gate_up_pipeline = pipeline(
        &GEMV_GATE_UP,
        &[
            (2, u(s.hidden)),
            (3, u(s.inner)),
            (4, eps),
            (5, SpecValue::U32(tuning.gate_up_pairs)),
            (6, SpecValue::U32(tuning.ahead)),
            (7, u(taken)),
            (8, span(s.hidden, partials)?),
            (9, norm_width),
            (10, u(split)),
        ],
        sub,
    );
    let gate_up_groups = groups(
        "gate/up row pairs",
        s.inner,
        subgroups * tuning.gate_up_pairs as usize,
    )? * split as u32;
    // A workgroup to a row, summing in the order batch 1's does.
    let norm_pipeline = Pipeline {
        kernel: &RMSNORM,
        constants: vec![
            (0, SpecValue::U32(subgroup)),
            (1, norm_width),
            (2, u(s.hidden)),
            (3, eps),
        ],
        subgroup_size: sub,
    };

    let mut g = Graph::default();
    let x = g.tensor::<F16>("x", &[rows, s.hidden], Kind::Input);
    let position = g.tensor("position", &[batch.words()], Kind::Param);
    let table = [s.capacity, s.head_dim / 2];
    let cos = g.tensor("rope.cos", &table, Kind::Constant);
    let sin = g.tensor("rope.sin", &table, Kind::Constant);
    let chunks = s.capacity / CHUNK;
    let cache = [slots, s.kv_heads, s.capacity, s.head_dim];
    let qkv_rows = (s.heads + 2 * s.kv_heads) * s.head_dim;
    let attention = s.heads * s.head_dim;
    let mut layers = Vec::with_capacity(s.layers);
    let mut input = x;
    for l in 0..s.layers {
        let name = |what: &str| format!("layers.{l}.{what}");
        let t = LayerTensors {
            input_norm: g.tensor(name("input_layernorm"), &[s.hidden], Kind::Constant),
            qkv: g.tensor(name("qkv_proj"), &[qkv_rows, s.hidden], Kind::Constant),
            o: g.tensor(name("o_proj"), &[s.hidden, attention], Kind::Constant),
            post_norm: g.tensor(
                name("post_attention_layernorm"),
                &[s.hidden],
                Kind::Constant,
            ),
            gate_up: g.tensor(
                name("gate_up_proj"),
                &[2 * s.inner, s.hidden],
                Kind::Constant,
            ),
            down: g.tensor(name("down_proj"), &[s.hidden, s.inner], Kind::Constant),
            k_cache: g.tensor(name("k_cache"), &cache, Kind::State),
            v_cache: g.tensor(name("v_cache"), &cache, Kind::State),
        };
        let q = g.tensor::<F16>(name("q"), &[rows, attention], Kind::Intermediate);
        let partial_o = g.tensor::<f32>(
            name("partial_o"),
            &[rows, s.heads, chunks, s.head_dim],
            Kind::Intermediate,
        );
        // Each chunk's maximum score and its sum of exponentials.
        let partial_ml = g.tensor::<f32>(
            name("partial_ml"),
            &[rows, s.heads, chunks, 2],
            Kind::Intermediate,
        );
        let attended = g.tensor::<F16>(name("attended"), &[rows, attention], Kind::Intermediate);
        let mid = g.tensor::<F16>(name("mid"), &[rows, s.hidden], Kind::Intermediate);
        let mlp = g.tensor::<F16>(name("mlp"), &[rows, s.inner], Kind::Intermediate);
        let output = g.tensor::<F16>(name("out"), &[rows, s.hidden], Kind::Intermediate);
        g.op(
            "qkv_rope",
            qkv_pipeline.clone(),
            (
                input,
                t.input_norm,
                t.qkv,
                cos,
                sin,
                position,
                q,
                t.k_cache,
                t.v_cache,
            ),
            [qkv_groups, 1, 1],
        );
        g.op(
            "attention_split",
            split_pipeline.clone(),
            (q, t.k_cache, t.v_cache, position, partial_o, partial_ml),
            [chunks as u32, s.kv_heads as u32, rows as u32],
        );
        g.op(
            "attention_combine",
            combine_pipeline.clone(),
            (position, partial_o, partial_ml, attended),
            [s.heads as u32, rows as u32, 1],
        );
        g.op(
            "o_proj",
            o_pipeline.clone(),
            &[attended, t.o, input, mid],
            [residual_groups, 1, 1],
        );
        g.op(
            "gate_up",
            gate_up_pipeline.clone(),
            &[mid, t.post_norm, t.gate_up, mlp],
            [gate_up_groups, 1, 1],
        );
        g.op(
            "down_proj",
            down_pipeline.clone(),
            &[mlp, t.down, mid, output],
            [residual_groups, 1, 1],
        );
        layers.push(t);
        input = output;
    }
    let norm = g.tensor("norm", &[s.hidden], Kind::Constant);
    let out = g.tensor("out", &[rows, s.hidden], Kind::Output);
    g.op(
        "final_norm",
        norm_pipeline,
        &[input, norm, out],
        [rows as u32, 1, 1],
    );
    Ok(DecodeGraph {
        graph: g,
        shape: shape.clone(),
        batch,
        x,
        position,
        cos,
        sin,
        norm,
        out,
        layers,
    })
}

/// Writes the weights of a stack into `step`, a recording of `graph`, whose tensors `layers` and
/// `norm` hold them, as [`Decoder::load`] does.
pub fn load_stack(
    step: &mut Step,
    graph: &Graph,
    layers: &[LayerTensors],
    norm: Tensor<F16>,
    checkpoint: &Checkpoint,
    prefix: &str,
) -> Result<(), Error> {
    let mut write = |t: Tensor<F16>, names: &[&str]| -> Result<(), Error> {
        let names: Vec<String> = names.iter().map(|n| format!("{prefix}{n}")).collect();
        let values = if names.len() == 1 && checkpoint.shape(&names[0]).map(<[_]>::len) == Some(1) {
            checkpoint.f16(&names[0])?.1
        } else {
            checkpoint.rows_f16(&names)?.1
        };
        let want = graph.info(t).len();
        if values.len() != want {
            return Err(Error::Checkpoint(format!(
                "{names:?}: {} values, not {want}",
                values.len()
            )));
        }
        step.write(t, &values)
    };
    for (l, t) in layers.iter().enumerate() {
        let n = |what: &str| format!("layers.{l}.{what}.weight");
        let [q, k, v] = ["q_proj", "k_proj", "v_proj"].map(|p| n(&format!("self_attn.{p}")));
        write(t.qkv, &[&q, &k, &v])?;
        write(t.o, &[&n("self_attn.o_proj")])?;
        let [gate, up] = ["gate_proj", "up_proj"].map(|p| n(&format!("mlp.{p}")));
        write(t.gate_up, &[&gate, &up])?;
        write(t.down, &[&n("mlp.down_proj")])?;
        write(t.input_norm, &[&n("input_layernorm")])?;
        write(t.post_norm, &[&n("post_attention_layernorm")])?;
    }
    write(norm, &["norm.weight"])
}

/// A MiniCPM-4 stack's decode step on the GPU, with its weights and KV cache.
#[derive(Debug)]
pub struct Decoder {
    step: Step,
    tensors: DecodeGraph,
}

impl Decoder {
    /// Records the step of `shape` at batch 1, with its rotation tables written and every weight
    /// zero until [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &Shape,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        Self::with_batch(ctx, kernels, shape, tuning, Batch::ONE, timed)
    }

    /// Records the step of `shape` for `batch`, as [`new`](Self::new) records it for one row.
    pub fn with_batch(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &Shape,
        tuning: &Tuning,
        batch: Batch,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = decode_batch_graph(shape, tuning, batch)?;
        let mut step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        let (cos, sin) = shape.rope_tables();
        step.write(tensors.cos, &cos)?;
        step.write(tensors.sin, &sin)?;
        Ok(Decoder { step, tensors })
    }

    /// Writes the weights of the stack whose tensors are named after `prefix` in
    /// `checkpoint`, as upstream's state_dict names them (VoxCPM2's base LM is `base_lm.`),
    /// converted to f16.
    pub fn load(&mut self, checkpoint: &Checkpoint, prefix: &str) -> Result<(), Error> {
        let t = &self.tensors;
        load_stack(
            &mut self.step,
            &t.graph,
            &t.layers,
            t.norm,
            checkpoint,
            prefix,
        )
    }

    /// Runs the stack over `x`, the hidden state of `position`, writing its keys and values into
    /// the cache there and attending over the cache up to it; returns the output after the final
    /// norm.
    pub fn step(&mut self, position: usize, x: &[F16]) -> Result<Vec<F16>, Error> {
        self.set_input(position, x)?;
        self.start()?.wait()?;
        self.output()
    }

    /// Writes the input of the next run at batch 1: `x`, the hidden state of `position`, and the
    /// position.
    pub fn set_input(&mut self, position: usize, x: &[F16]) -> Result<(), Error> {
        self.set_rows(&[Some(Row { position, slot: 0 })], x)
    }

    /// Runs the stack over `x`, the hidden states `[rows][hidden]` of the batch's rows, each where
    /// `rows` says or spare: writes each row's keys and values into its slot of the cache at its
    /// position and attends over the slot up to it. Returns the outputs after the final norm,
    /// `[rows][hidden]`, a spare row's meaningless.
    pub fn step_rows(&mut self, rows: &[Option<Row>], x: &[F16]) -> Result<Vec<F16>, Error> {
        self.set_rows(rows, x)?;
        self.start()?.wait()?;
        self.output()
    }

    /// Writes the input of the next run: `x`, the rows' hidden states, and where each runs, as
    /// [`step_rows`](Self::step_rows) takes them.
    pub fn set_rows(&mut self, rows: &[Option<Row>], x: &[F16]) -> Result<(), Error> {
        let t = &self.tensors;
        let words = t.batch.placement(rows, t.shape.capacity)?;
        self.step.write(t.x, x)?;
        self.step.write(t.position, &words)
    }

    /// Starts a run over the input last written, and returns without waiting for it.
    pub fn start(&mut self) -> Result<Submission<'_>, Error> {
        self.step.start()
    }

    /// The output of the last run, after the final norm: `[rows][hidden]`.
    pub fn output(&self) -> Result<Vec<F16>, Error> {
        self.step.read(self.tensors.out)
    }

    /// The decoder recorded again cut as `tuning` says, and timed per dispatch when `timed`, over
    /// the weights and the cache it holds. On an error they go with it.
    pub fn retuned(
        self,
        ctx: &Context,
        kernels: &mut Kernels,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = decode_batch_graph(&self.tensors.shape, tuning, self.tensors.batch)?;
        let step = self
            .step
            .recorded_again(ctx, kernels, &tensors.graph, timed)?;
        Ok(Decoder { step, tensors })
    }

    /// Each dispatch's label and GPU time in the last step, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn shape(&self) -> &Shape {
        &self.tensors.shape
    }

    pub fn batch(&self) -> Batch {
        self.tensors.batch
    }

    /// The recorded step's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }

    /// The step's tensors, and the step, for programs over its weights and cache to be added to
    /// it ([`Step::add`]) and run.
    pub fn parts_mut(&mut self) -> (&DecodeGraph, &mut Step) {
        (&self.tensors, &mut self.step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(layers: usize) -> Shape {
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
            rope: true,
        }
    }

    #[test]
    fn a_layer_is_six_dispatches_each_after_a_barrier() {
        let s = shape(28);
        let g = decode_graph(&s, &Tuning::for_subgroup(64)).unwrap();
        assert_eq!(g.graph.ops.len(), 6 * 28 + 1);
        let plan = crate::graph::plan(&g.graph);
        let barriers = crate::graph::barriers(&g.graph, &plan);
        assert!(
            barriers[1..].iter().all(|&b| b),
            "every op reads the one before"
        );
        // The base LM's weights: about 1.32 billion f16 parameters.
        assert_eq!(s.weight_bytes(), 2_642_644_992);
        // The arena holds one layer's intermediates at a time, not 28 layers' worth.
        let partials = 16 * 32 * 128 * 4;
        assert!(plan.arena_bytes < 2 * partials, "{}", plan.arena_bytes);
    }

    #[test]
    fn every_gemv_covers_its_rows_once_at_either_subgroup_size() {
        let s = shape(28);
        for subgroup in SUBGROUPS {
            let tuning = Tuning {
                qkv_pairs: 2,
                residual_rows: 4,
                ..Tuning::for_subgroup(subgroup)
            };
            let g = decode_graph(&s, &tuning).unwrap();
            let per_group = (tuning.workgroup / subgroup) as usize;
            for op in &g.graph.ops {
                if op.pipeline.subgroup_size.is_some() {
                    assert_eq!(op.pipeline.subgroup_size, Some(subgroup), "{}", op.label);
                }
                let groups = op.groups[0] as usize * per_group;
                let rows = match op.label.as_str() {
                    "qkv_rope" => groups * 2 * tuning.qkv_pairs as usize,
                    "gate_up" => groups * tuning.gate_up_pairs as usize,
                    "o_proj" | "down_proj" => groups * tuning.residual_rows as usize,
                    _ => continue,
                };
                let want = match op.label.as_str() {
                    "qkv_rope" => (s.heads + 2 * s.kv_heads) * s.head_dim,
                    "gate_up" => s.inner,
                    _ => s.hidden,
                };
                assert_eq!(rows, want, "{} at {subgroup}", op.label);
            }
        }
    }

    #[test]
    fn every_gemv_reads_its_loads_ahead_at_the_constant_its_kernel_declares() {
        let tuning = Tuning {
            ahead: 3,
            ..Tuning::for_subgroup(32)
        };
        let g = decode_graph(&shape(1), &tuning).unwrap();
        let mut gemvs = 0;
        for op in &g.graph.ops {
            let id = match op.pipeline.kernel.name {
                "gemv_residual" => 4,
                "gemv_gate_up" => 6,
                "gemv_qkv_rope" => 9,
                _ => continue,
            };
            assert!(
                op.pipeline.constants.contains(&(id, SpecValue::U32(3))),
                "{}: {:?}",
                op.label,
                op.pipeline.constants
            );
            gemvs += 1;
        }
        assert_eq!(gemvs, 4);
    }

    #[test]
    fn shapes_the_kernels_cannot_cover_exactly_are_refused() {
        let tuning = Tuning::for_subgroup(64);
        let mut s = shape(1);
        s.capacity = 1000;
        assert!(matches!(decode_graph(&s, &tuning), Err(Error::Shape(_))));
        let s = shape(1);
        let odd = Tuning {
            residual_rows: 3,
            ..tuning
        };
        assert!(matches!(decode_graph(&s, &odd), Err(Error::Shape(_))));
        let narrow = Tuning {
            subgroup: 16,
            ..tuning
        };
        assert!(matches!(decode_graph(&s, &narrow), Err(Error::Shape(_))));
    }

    #[test]
    fn a_stack_without_rope_is_taken_from_its_config_and_rotates_nothing() {
        let config = |no_rope: bool| -> LmConfig {
            serde_json::from_value(serde_json::json!({
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "num_hidden_layers": 28,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "kv_channels": 128,
                "max_position_embeddings": 32768,
                "rope_scaling": {
                    "type": "dynamic",
                    "short_factor": vec![1.0; 64],
                    "original_max_position_embeddings": 32768,
                },
                "use_mup": false,
                "no_rope": no_rope,
            }))
            .unwrap()
        };
        // RoPE other than LongRoPE is refused only where it would be used.
        assert!(matches!(
            Shape::new(&config(false), 8, 64),
            Err(Error::Shape(_))
        ));
        let s = Shape::new(&config(true), 8, 64).unwrap();
        assert!(!s.rope);
        let rope = |s: &Shape| -> Vec<SpecValue> {
            let g = decode_graph(s, &Tuning::for_subgroup(64)).unwrap();
            let qkv = g.graph.ops.iter().filter(|op| op.label == "qkv_rope");
            let constant = |op: &crate::graph::Op| {
                let found = op.pipeline.constants.iter().find(|(id, _)| *id == 10);
                found.expect("a ROPE constant").1
            };
            qkv.map(constant).collect()
        };
        assert_eq!(rope(&s), vec![SpecValue::Bool(false); 8]);
        assert_eq!(rope(&shape(2)), vec![SpecValue::Bool(true); 2]);
    }

    #[test]
    fn spare_rows_are_placed_past_every_position_and_a_slot_is_taken_once() {
        let batch = Batch { rows: 3, slots: 4 };
        let at = |position, slot| Some(Row { position, slot });
        let words = batch.placement(&[at(5, 2), None, at(0, 0)], 64).unwrap();
        assert_eq!(words, [5, u32::MAX, 0, 2, u32::MAX, 0]);
        // One slot: the position alone, as at batch 1.
        assert_eq!(Batch::ONE.placement(&[at(7, 0)], 64).unwrap(), [7]);
        for rows in [
            [at(5, 2), at(9, 2), None],
            [at(64, 0), None, None],
            [at(1, 4), None, None],
        ] {
            assert!(
                matches!(batch.placement(&rows, 64), Err(Error::Shape(_))),
                "{rows:?}"
            );
        }
        assert!(batch.placement(&[None, None], 64).is_err());
    }

    #[test]
    fn a_batch_scales_its_rows_tensors_and_grids_and_not_the_weights() {
        let s = shape(2);
        let tuning = Tuning {
            vectors: 4,
            ..Tuning::for_subgroup(32)
        };
        let one = decode_graph(&s, &tuning).unwrap();
        let batch = Batch { rows: 4, slots: 6 };
        let four = decode_batch_graph(&s, &tuning, batch).unwrap();
        let bytes = |g: &DecodeGraph, t: TensorId| g.graph.info(t).bytes();
        assert_eq!(bytes(&four, four.x.id()), 4 * bytes(&one, one.x.id()));
        assert_eq!(bytes(&four, four.out.id()), 4 * bytes(&one, one.out.id()));
        assert_eq!(bytes(&four, four.position.id()), 4 * 2 * 4);
        for (l, w) in one.layers.iter().zip(&four.layers) {
            assert_eq!(
                bytes(&four, w.k_cache.id()),
                6 * bytes(&one, l.k_cache.id())
            );
            assert_eq!(bytes(&four, w.qkv.id()), bytes(&one, l.qkv.id()));
            assert_eq!(bytes(&four, w.down.id()), bytes(&one, l.down.id()));
        }
        let norm_width = (0, SpecValue::U32(tuning.norm_width()));
        for (a, b) in one.graph.ops.iter().zip(&four.graph.ops) {
            let want = match b.label.as_str() {
                "attention_split" => [a.groups[0], a.groups[1], 4],
                "attention_combine" => [a.groups[0], 4, 1],
                "final_norm" => [4, 1, 1],
                _ => a.groups,
            };
            assert_eq!(b.groups, want, "{}", b.label);
            let constant = |id| b.pipeline.constants.iter().find(|c| c.0 == id).copied();
            if b.pipeline.kernel.name == "rmsnorm" {
                assert_eq!(constant(1), Some((1, norm_width.1)));
            }
        }
        assert!(matches!(
            decode_batch_graph(&s, &tuning, Batch { rows: 3, slots: 2 }),
            Err(Error::Shape(_))
        ));
    }

    #[test]
    fn a_batch_is_cut_as_was_fastest_at_its_subgroup_size() {
        for subgroup in SUBGROUPS {
            for rows in 1..=4 {
                let tuning = Tuning::for_batch(subgroup, rows);
                assert_eq!(tuning, Tuning::for_subgroup(subgroup), "{rows} rows");
            }
        }
        // The Radeon's: four subgroups of 64 taking 4 rows.
        let radeon = Tuning::for_batch(64, 16);
        assert_eq!((radeon.workgroup, radeon.stage), (256, 32 << 10));
        assert_eq!((radeon.split(16), radeon.split(8)), ((4, 4), (4, 2)));
        // NVIDIA's: eight subgroups of 32 taking 8 rows, staged 16 KiB at a time.
        let nvidia = Tuning::for_batch(32, 16);
        assert_eq!((nvidia.workgroup, nvidia.stage), (256, 16 << 10));
        assert_eq!((nvidia.split(16), nvidia.split(8)), ((8, 2), (8, 1)));
        assert_eq!(Tuning::for_batch(32, 5).split(5), (5, 1));
    }

    #[test]
    fn vectors_are_staged_whole_unless_a_batch_of_them_does_not_fit() {
        let tuning = Tuning::for_subgroup(64);
        assert_eq!(tuning.span(6144, 1, 4).unwrap(), 0);
        assert_eq!(tuning.span(2048, 8, 0).unwrap(), 0);
        assert_eq!(tuning.span(6144, 4, 4).unwrap(), 3072);
        // Rows of 32 KiB in all leave no room in 32 KiB for the norm's partial sums beside them.
        assert_eq!(tuning.span(2048, 8, 4).unwrap(), 1024);
        assert_eq!(tuning.span(2048, 16, 0).unwrap(), 1024);
        assert_eq!(tuning.span(2048, 16, 4).unwrap(), 512);
        let wider = Tuning {
            stage: 48 << 10,
            ..tuning
        };
        assert_eq!(wider.span(6144, 16, 16).unwrap(), 1024);
        let small = Tuning {
            stage: 1024,
            ..tuning
        };
        assert!(matches!(small.span(2048, 16, 0), Err(Error::Shape(_))));
    }

    #[test]
    fn a_batch_is_shared_out_over_workgroups_side_by_side() {
        let tuning = Tuning::for_subgroup(64);
        let cut = |vectors| Tuning { vectors, ..tuning };
        assert_eq!(cut(4).split(16), (4, 4));
        assert_eq!(cut(4).split(2), (2, 1));
        assert_eq!(cut(4).split(6), (3, 2));
        assert_eq!(cut(16).split(16), (16, 1));
        assert_eq!(cut(0).split(5), (1, 5));
        let s = shape(1);
        let batch = Batch {
            rows: 16,
            slots: 16,
        };
        let whole = decode_batch_graph(&s, &cut(16), batch).unwrap();
        let split = decode_batch_graph(&s, &cut(4), batch).unwrap();
        for (a, b) in whole.graph.ops.iter().zip(&split.graph.ops) {
            let constant = |op: &crate::graph::Op, id| {
                op.pipeline
                    .constants
                    .iter()
                    .find(|c| c.0 == id)
                    .map(|c| c.1)
            };
            let (groups_id, batch_id) = match b.pipeline.kernel.name {
                "gemv_qkv_rope" => (15, 11),
                "gemv_gate_up" => (10, 7),
                "gemv_residual" => (8, 6),
                _ => {
                    assert_eq!(a.groups, b.groups, "{}", b.label);
                    continue;
                }
            };
            assert_eq!(b.groups[0], 4 * a.groups[0], "{}", b.label);
            assert_eq!(
                constant(b, batch_id),
                Some(SpecValue::U32(4)),
                "{}",
                b.label
            );
            assert_eq!(
                constant(b, groups_id),
                Some(SpecValue::U32(4)),
                "{}",
                b.label
            );
            assert_eq!(
                constant(a, groups_id),
                Some(SpecValue::U32(1)),
                "{}",
                a.label
            );
        }
    }

    #[test]
    fn rope_tables_start_at_the_identity_and_turn_at_each_rate() {
        let mut s = shape(1);
        s.capacity = 64;
        let (cos, sin) = s.rope_tables();
        assert_eq!(cos.len(), 64 * 64);
        assert!(cos[..64].iter().all(|&c| c == 1.0) && sin[..64].iter().all(|&v| v == 0.0));
        // Position 1, frequency 0: one radian.
        assert_eq!(cos[64], 1f32.cos());
        assert_eq!(sin[64], 1f32.sin());
    }
}
