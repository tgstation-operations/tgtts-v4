//! One decode step of a MiniCPM-4 stack at batch 1, as VoxCPM2's base LM runs it: per layer,
//! the input norm, q/k/v projection, RoPE and the cache write in one dispatch, attention over the
//! cache split along it and joined, the output projection with the residual, the post-attention
//! norm with the gate/up projection and SiLU, and the down projection with the residual; then the
//! final norm. Six dispatches a layer and one more, recorded once for a cache capacity and
//! submitted at every position, which is read from a buffer the host writes before each.

use serde::Deserialize;
use vulkan::{Context, SpecValue, Timing};

use crate::{
    Error,
    checkpoint::Checkpoint,
    graph::{Graph, Kind, Pipeline, TensorId},
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
}

impl Shape {
    /// The shape of the stack `config` describes, with its first `layers` layers and a cache of
    /// `capacity` positions. Refuses what VoxCPM2 does not use, as minicpm4 does.
    pub fn new(config: &LmConfig, layers: usize, capacity: usize) -> Result<Self, Error> {
        let unsupported = |why: String| Err(Error::Shape(why));
        if config.use_mup {
            return unsupported("use_mup".into());
        }
        let rope = &config.rope_scaling;
        if rope.kind != "longrope"
            || config.max_position_embeddings != rope.original_max_position_embeddings
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
    /// each: computed in f32 in the order minicpm4's `Rope::new` takes, so they round as its do.
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

/// How the kernels' work is cut: the subgroup size every kernel runs at (32 or 64 on RADV,
/// 32 on NVIDIA), the GEMVs' workgroup width, and the rows each of their subgroups takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub subgroup: u32,
    pub workgroup: u32,
    /// Row pairs per subgroup of the q/k/v and the gate/up projections.
    pub qkv_pairs: u32,
    pub gate_up_pairs: u32,
    /// Rows per subgroup of the output and down projections.
    pub residual_rows: u32,
}

/// The fastest measured on the Radeon 8060S: one row, or one pair of rows, to a subgroup of 64
/// and one subgroup to a workgroup, which reached 220 GB/s over four layers' weights where a
/// streaming read reaches 240. Larger workgroups and more rows to a subgroup were slower.
impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            subgroup: 64,
            workgroup: 64,
            qkv_pairs: 1,
            gate_up_pairs: 1,
            residual_rows: 1,
        }
    }
}

/// One layer's tensors.
#[derive(Clone, Debug)]
pub struct LayerTensors {
    pub input_norm: TensorId,
    pub qkv: TensorId,
    pub o: TensorId,
    pub post_norm: TensorId,
    pub gate_up: TensorId,
    pub down: TensorId,
    pub k_cache: TensorId,
    pub v_cache: TensorId,
}

/// The step's graph and the tensors the host writes and reads.
#[derive(Clone, Debug)]
pub struct DecodeGraph {
    pub graph: Graph,
    pub shape: Shape,
    pub x: TensorId,
    pub position: TensorId,
    pub cos: TensorId,
    pub sin: TensorId,
    pub norm: TensorId,
    pub out: TensorId,
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

/// The decode step of `shape` cut as `tuning` says.
pub fn decode_graph(shape: &Shape, tuning: &Tuning) -> Result<DecodeGraph, Error> {
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
        ],
        sub,
    );
    let qkv_groups = groups(
        "q/k/v row pairs",
        (s.heads + 2 * s.kv_heads) * s.head_dim / 2,
        subgroups * tuning.qkv_pairs as usize,
    )?;
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
        ],
        subgroup_size: None,
    };
    let residual = |k: usize| {
        pipeline(
            &GEMV_RESIDUAL,
            &[(2, u(k)), (3, SpecValue::U32(tuning.residual_rows))],
            sub,
        )
    };
    let residual_groups = groups(
        "residual rows",
        s.hidden,
        subgroups * tuning.residual_rows as usize,
    )?;
    let gate_up_pipeline = pipeline(
        &GEMV_GATE_UP,
        &[
            (2, u(s.hidden)),
            (3, u(s.inner)),
            (4, eps),
            (5, SpecValue::U32(tuning.gate_up_pairs)),
        ],
        sub,
    );
    let gate_up_groups = groups(
        "gate/up row pairs",
        s.inner,
        subgroups * tuning.gate_up_pairs as usize,
    )?;
    let norm_pipeline = pipeline(&RMSNORM, &[(2, u(s.hidden)), (3, eps)], sub);

    let mut g = Graph::default();
    let half = 2u64;
    let vector = |n: usize| n as u64 * half;
    let x = g.tensor("x", vector(s.hidden), Kind::Input);
    let position = g.tensor("position", 4, Kind::Param);
    let table = (s.capacity * s.head_dim / 2 * 4) as u64;
    let cos = g.tensor("rope.cos", table, Kind::Constant);
    let sin = g.tensor("rope.sin", table, Kind::Constant);
    let chunks = s.capacity / CHUNK;
    let cache = vector(s.kv_heads * s.capacity * s.head_dim);
    let qkv_rows = (s.heads + 2 * s.kv_heads) * s.head_dim;
    let mut layers = Vec::with_capacity(s.layers);
    let mut input = x;
    for l in 0..s.layers {
        let name = |what: &str| format!("layers.{l}.{what}");
        let t = LayerTensors {
            input_norm: g.tensor(name("input_layernorm"), vector(s.hidden), Kind::Constant),
            qkv: g.tensor(
                name("qkv_proj"),
                vector(qkv_rows * s.hidden),
                Kind::Constant,
            ),
            o: g.tensor(
                name("o_proj"),
                vector(s.hidden * s.heads * s.head_dim),
                Kind::Constant,
            ),
            post_norm: g.tensor(
                name("post_attention_layernorm"),
                vector(s.hidden),
                Kind::Constant,
            ),
            gate_up: g.tensor(
                name("gate_up_proj"),
                vector(2 * s.inner * s.hidden),
                Kind::Constant,
            ),
            down: g.tensor(
                name("down_proj"),
                vector(s.hidden * s.inner),
                Kind::Constant,
            ),
            k_cache: g.tensor(name("k_cache"), cache, Kind::State),
            v_cache: g.tensor(name("v_cache"), cache, Kind::State),
        };
        let q = g.tensor(name("q"), vector(s.heads * s.head_dim), Kind::Intermediate);
        let partial_o = g.tensor(
            name("partial_o"),
            (s.heads * chunks * s.head_dim * 4) as u64,
            Kind::Intermediate,
        );
        let partial_ml = g.tensor(
            name("partial_ml"),
            (s.heads * chunks * 8) as u64,
            Kind::Intermediate,
        );
        let attended = g.tensor(
            name("attended"),
            vector(s.heads * s.head_dim),
            Kind::Intermediate,
        );
        let mid = g.tensor(name("mid"), vector(s.hidden), Kind::Intermediate);
        let mlp = g.tensor(name("mlp"), vector(s.inner), Kind::Intermediate);
        let output = g.tensor(name("out"), vector(s.hidden), Kind::Intermediate);
        g.op(
            "qkv_rope",
            qkv_pipeline.clone(),
            &[
                input,
                t.input_norm,
                t.qkv,
                cos,
                sin,
                position,
                q,
                t.k_cache,
                t.v_cache,
            ],
            [qkv_groups, 1, 1],
        );
        g.op(
            "attention_split",
            split_pipeline.clone(),
            &[q, t.k_cache, t.v_cache, position, partial_o, partial_ml],
            [chunks as u32, s.kv_heads as u32, 1],
        );
        g.op(
            "attention_combine",
            combine_pipeline.clone(),
            &[position, partial_o, partial_ml, attended],
            [s.heads as u32, 1, 1],
        );
        g.op(
            "o_proj",
            residual(s.heads * s.head_dim),
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
            residual(s.inner),
            &[mlp, t.down, mid, output],
            [residual_groups, 1, 1],
        );
        layers.push(t);
        input = output;
    }
    let norm = g.tensor("norm", vector(s.hidden), Kind::Constant);
    let out = g.tensor("out", vector(s.hidden), Kind::Output);
    g.op("final_norm", norm_pipeline, &[input, norm, out], [1, 1, 1]);
    Ok(DecodeGraph {
        graph: g,
        shape: shape.clone(),
        x,
        position,
        cos,
        sin,
        norm,
        out,
        layers,
    })
}

/// A MiniCPM-4 stack's decode step on the GPU, with its weights and KV cache.
#[derive(Debug)]
pub struct Decoder {
    step: Step,
    tensors: DecodeGraph,
}

impl Decoder {
    /// Records the step of `shape`, with its rotation tables written and every weight zero
    /// until [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &Shape,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = decode_graph(shape, tuning)?;
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
        let mut write = |t: TensorId, names: &[&str]| -> Result<(), Error> {
            let names: Vec<String> = names.iter().map(|n| format!("{prefix}{n}")).collect();
            let values =
                if names.len() == 1 && checkpoint.shape(&names[0]).map(<[_]>::len) == Some(1) {
                    checkpoint.f16(&names[0])?.1
                } else {
                    checkpoint.rows_f16(&names)?.1
                };
            let want = self.tensors.graph.tensors[t.0].bytes as usize / 2;
            if values.len() != want {
                return Err(Error::Checkpoint(format!(
                    "{names:?}: {} values, not {want}",
                    values.len()
                )));
            }
            self.step.write(t, &values)
        };
        for (l, t) in self.tensors.layers.clone().iter().enumerate() {
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
        write(self.tensors.norm, &["norm.weight"])
    }

    /// Runs the stack over `x`, the hidden state of `position` as f16 bits, writing its keys and
    /// values into the cache there and attending over the cache up to it; returns the output
    /// after the final norm, as f16 bits.
    pub fn step(&mut self, position: usize, x: &[u16]) -> Result<Vec<u16>, Error> {
        let capacity = self.tensors.shape.capacity;
        if position >= capacity {
            return Err(Error::Shape(format!(
                "position {position} in a cache of {capacity}"
            )));
        }
        self.step.write(self.tensors.x, x)?;
        self.step.write(self.tensors.position, &[position as u32])?;
        self.step.submit()?;
        self.step.read(self.tensors.out)
    }

    /// Each dispatch's label and GPU time in the last step, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn shape(&self) -> &Shape {
        &self.tensors.shape
    }

    /// The recorded step's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
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
        }
    }

    #[test]
    fn a_layer_is_six_dispatches_each_after_a_barrier() {
        let s = shape(28);
        let g = decode_graph(&s, &Tuning::default()).unwrap();
        assert_eq!(g.graph.ops.len(), 6 * 28 + 1);
        let plan = crate::graph::plan(&g.graph);
        let barriers = crate::graph::barriers(&g.graph, &plan);
        assert!(
            barriers[1..].iter().all(|&b| b),
            "every op reads the one before"
        );
        // The base LM's weights as the brief counts them: about 1.32 billion f16 parameters.
        assert_eq!(s.weight_bytes(), 2_642_644_992);
        // The arena holds one layer's intermediates at a time, not 28 layers' worth.
        let partials = 16 * 32 * 128 * 4;
        assert!(plan.arena_bytes < 2 * partials, "{}", plan.arena_bytes);
    }

    #[test]
    fn shapes_the_kernels_cannot_cover_exactly_are_refused() {
        let tuning = Tuning::default();
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
