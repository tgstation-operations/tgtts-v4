//! A prompt run through a MiniCPM-4 stack at batch 1, as VoxCPM2 runs its base and residual LMs
//! over one: a run of positions at once, its keys and values written into a [`Decoder`]'s cache
//! for its steps to go on from. Per layer: the input norm, the q/k/v projection, RoPE and the
//! cache write, causal attention over the cache up to each position, the output projection, the
//! residual add with the post-attention norm, the gate/up projection, SiLU, the down projection,
//! and the residual add with the next layer's input norm, or after the last layer the final norm.
//!
//! The projections are the cooperative-matrix GEMM of `kernels/gpu`, compiled from MLIR at static
//! shapes for every number of rows in steps of [`TILE`] up to [`MAX_ROWS`], and at each subgroup
//! size of [`SUBGROUPS`]; the rest are GLSL. A run of positions takes the program of the fewest
//! rows that hold it, the rows past it padding that nothing reads.

use vulkan::{Context, F16, SpecValue};

use crate::{
    Error,
    decode::{DecodeGraph, Decoder, LayerTensors, SUBGROUPS, Shape, Tuning},
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::{Access, KernelInfo, Kernels},
    step::Program,
};

/// The rows of the GEMM's workgroup tile, which a run's rows are rounded up to.
pub const TILE: usize = 128;

/// The most rows a program takes, and so the most positions one run does.
pub const MAX_ROWS: usize = 1024;

/// The width of the GLSL kernels' workgroups.
const WORKGROUP: u32 = 256;

use Access::{Read, Write};

/// `norm_rows.comp`: run, x, gain -> out.
pub static NORM_ROWS: KernelInfo = KernelInfo {
    name: "norm_rows",
    access: &[Read, Read, Read, Write],
};

/// `add_norm_rows.comp`: run, c, residual, gain -> sum, normed.
pub static ADD_NORM_ROWS: KernelInfo = KernelInfo {
    name: "add_norm_rows",
    access: &[Read, Read, Read, Read, Write, Write],
};

/// `rope_rows.comp`: run, c, cos, sin -> q, k cache, v cache.
pub static ROPE_ROWS: KernelInfo = KernelInfo {
    name: "rope_rows",
    access: &[Read, Read, Read, Read, Write, Write, Write],
};

/// `attention_rows.comp`: run, q, k cache, v cache -> out.
pub static ATTENTION_ROWS: KernelInfo = KernelInfo {
    name: "attention_rows",
    access: &[Read, Read, Read, Read, Write],
};

/// `silu_mul_rows.comp`: run, c -> out.
pub static SILU_MUL_ROWS: KernelInfo = KernelInfo {
    name: "silu_mul_rows",
    access: &[Read, Read, Write],
};

/// The query positions attention_rows takes per workgroup: four invocations to each of the
/// queries a key-value head serves.
fn attention_rows(group: usize) -> usize {
    WORKGROUP as usize / 4 / group
}

macro_rules! gemm {
    ($m:literal, $n:literal, $k:literal, $subgroup:literal) => {
        KernelInfo {
            name: concat!("gemm_f32acc_", $m, "x", $n, "x", $k, "_sg", $subgroup),
            access: &[Read, Read, Write],
        }
    };
}

macro_rules! gemms {
    ($subgroup:literal: $($m:literal)*) => {
        [$(
            gemm!($m, 2560, 2048, $subgroup),
            gemm!($m, 2048, 2048, $subgroup),
            gemm!($m, 12288, 2048, $subgroup),
            gemm!($m, 2048, 6144, $subgroup),
        )*]
    };
}

/// The GEMMs `nix build .#gpu-kernels` compiles for the prefill, C = A B^T with f32
/// accumulators: a, b -> c. The projections of VoxCPM2's LMs, q/k/v, output, gate/up and down,
/// at every number of rows a program takes, for subgroups of each size of [`SUBGROUPS`] in turn.
pub static GEMMS: [[KernelInfo; 32]; SUBGROUPS.len()] = [
    gemms!(64: 128 256 384 512 640 768 896 1024),
    gemms!(32: 128 256 384 512 640 768 896 1024),
];

/// The GEMM of `m` rows of A, `n` of B and `k` columns for subgroups of `subgroup`, if it was
/// compiled.
pub fn gemm(m: usize, n: usize, k: usize, subgroup: u32) -> Result<&'static KernelInfo, Error> {
    let name = format!("gemm_f32acc_{m}x{n}x{k}_sg{subgroup}");
    GEMMS
        .iter()
        .flatten()
        .find(|g| g.name == name)
        .ok_or_else(|| {
            Error::Shape(format!(
                "no GEMM of {m}x{n}x{k} for subgroups of {subgroup} is compiled: \
             nix/gpu-kernels.nix and GEMMS list them"
            ))
        })
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// A prefill program: the decode step's graph with its ops replaced by the run's, over the same
/// weights, rotation tables and cache, and tensors of its own for `max_rows` rows, which every
/// program of the same `max_rows` lays out alike whatever rows it runs.
#[derive(Clone, Debug)]
pub struct PrefillGraph {
    pub graph: Graph,
    /// The rows it runs, a multiple of [`TILE`].
    pub rows: usize,
    pub max_rows: usize,
    /// The first position and the length of the run, and the slot of the cache it goes to, three
    /// u32.
    pub run: TensorId,
    /// The run's hidden states `[max_rows][hidden]` in f16, and the outputs after the final
    /// norm.
    pub x: TensorId,
    pub out: TensorId,
}

/// A stack's tensors in a graph, which a prefill runs over: its weights, its cache and the
/// rotation tables.
#[derive(Clone, Copy, Debug)]
pub struct StackTensors<'a> {
    pub shape: &'a Shape,
    /// The slots of the cache.
    pub slots: usize,
    pub layers: &'a [LayerTensors],
    pub norm: TensorId,
    pub cos: TensorId,
    pub sin: TensorId,
}

impl<'a> From<&'a DecodeGraph> for StackTensors<'a> {
    fn from(decode: &'a DecodeGraph) -> Self {
        StackTensors {
            shape: &decode.shape,
            slots: decode.batch.slots,
            layers: &decode.layers,
            norm: decode.norm,
            cos: decode.cos,
            sin: decode.sin,
        }
    }
}

/// The prefill program of `rows` rows over the stack `decode` runs, with tensors for `max_rows`,
/// its kernels at `tuning`'s subgroup size.
pub fn prefill_graph(
    decode: &DecodeGraph,
    rows: usize,
    max_rows: usize,
    tuning: &Tuning,
) -> Result<PrefillGraph, Error> {
    let mut g = decode.graph.clone();
    g.ops.clear();
    let half = (max_rows * decode.shape.hidden * 2) as u64;
    let run = g.tensor("prefill.run", 12, Kind::Param);
    let x = g.tensor("prefill.x", half, Kind::Input);
    let out = g.tensor("prefill.out", half, Kind::Output);
    let stack = StackTensors::from(decode);
    append_prefill(
        &mut g,
        "prefill",
        stack,
        rows,
        max_rows,
        tuning,
        [run, x, out],
    )?;
    Ok(PrefillGraph {
        graph: g,
        rows,
        max_rows,
        run,
        x,
        out,
    })
}

/// Appends to `g` the run of `rows` rows through `stack`, one of `g`'s, from the hidden states
/// `x` of the positions `run` gives, in the slot of the cache it gives, to their outputs after the
/// final norm, `out`, both `[max_rows][hidden]` in f16; its tensors of its own named after `name`
/// and sized for `max_rows` rows, so that the runs of any rows lay them out alike. Its kernels run
/// at `tuning`'s subgroup size.
pub fn append_prefill(
    g: &mut Graph,
    name: &str,
    stack: StackTensors,
    rows: usize,
    max_rows: usize,
    tuning: &Tuning,
    [run, x, out]: [TensorId; 3],
) -> Result<(), Error> {
    let s = stack.shape;
    let check = |ok: bool, why: String| if ok { Ok(()) } else { Err(Error::Shape(why)) };
    check(
        rows > 0 && rows.is_multiple_of(TILE) && rows <= max_rows,
        format!("a program of {rows} rows, of at most {max_rows}"),
    )?;
    check(
        max_rows.is_multiple_of(TILE) && max_rows <= MAX_ROWS,
        format!("programs of up to {max_rows} rows"),
    )?;
    let subgroup = tuning.subgroup;
    check(
        WORKGROUP.is_multiple_of(subgroup),
        format!("subgroups of {subgroup}"),
    )?;
    check(
        s.hidden.is_multiple_of(8 * WORKGROUP as usize) && s.inner.is_multiple_of(4),
        format!("{} hidden and {} inner", s.hidden, s.inner),
    )?;
    check(
        s.kv_heads > 0
            && s.heads.is_multiple_of(s.kv_heads)
            && s.head_dim.is_multiple_of(16)
            && (WORKGROUP as usize).is_multiple_of(4 * s.heads / s.kv_heads),
        format!("{} heads over {} of {}", s.heads, s.kv_heads, s.head_dim),
    )?;
    let group = s.heads / s.kv_heads;
    let per_group = attention_rows(group);
    check(
        rows.is_multiple_of(per_group),
        format!("{rows} rows in workgroups of {per_group}"),
    )?;

    let common = [
        (0, SpecValue::U32(subgroup)),
        (1, SpecValue::U32(WORKGROUP)),
    ];
    let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
        kernel,
        constants: common.iter().chain(constants).copied().collect(),
        subgroup_size: Some(subgroup),
    };
    let eps = SpecValue::F32(s.eps);
    let norm = pipeline(&NORM_ROWS, &[(2, u(s.hidden)), (3, eps)]);
    let add_norm = pipeline(&ADD_NORM_ROWS, &[(2, u(s.hidden)), (3, eps)]);
    let shape = [
        (2, u(s.heads)),
        (3, u(s.kv_heads)),
        (4, u(s.head_dim)),
        (5, u(s.capacity)),
    ];
    let slots = (7, u(stack.slots));
    let rope_pipeline = pipeline(
        &ROPE_ROWS,
        &[shape.as_slice(), &[(6, SpecValue::Bool(s.rope)), slots]].concat(),
    );
    let scale = SpecValue::F32(1.0 / (s.head_dim as f32).sqrt());
    let attention = pipeline(
        &ATTENTION_ROWS,
        &[shape.as_slice(), &[(6, scale), slots]].concat(),
    );
    let silu = pipeline(&SILU_MUL_ROWS, &[(2, u(s.inner))]);
    let qkv_width = (s.heads + 2 * s.kv_heads) * s.head_dim;
    let gemm_op = |n: usize, k: usize| -> Result<(Pipeline, [u32; 3]), Error> {
        let pipeline = Pipeline {
            kernel: gemm(rows, n, k, subgroup)?,
            constants: Vec::new(),
            subgroup_size: Some(subgroup),
        };
        check(
            n.is_multiple_of(TILE),
            format!("a GEMM of {n} columns of C"),
        )?;
        Ok((pipeline, [(rows / TILE) as u32, (n / TILE) as u32, 1]))
    };
    let qkv_gemm = gemm_op(qkv_width, s.hidden)?;
    let o_gemm = gemm_op(s.hidden, s.heads * s.head_dim)?;
    let gate_up_gemm = gemm_op(2 * s.inner, s.hidden)?;
    let down_gemm = gemm_op(s.hidden, s.inner)?;

    let half = |n: usize| (max_rows * n * 2) as u64;
    let single = |n: usize| (max_rows * n * 4) as u64;
    for (t, want) in [(run, 12), (x, half(s.hidden)), (out, half(s.hidden))] {
        let bytes = g.tensors[t.0].bytes;
        check(
            bytes == want,
            format!("{name}: tensor {} of {bytes} bytes, not {want}", t.0),
        )?;
    }
    let row_groups = [rows as u32, 1, 1];
    let mut input = x;
    let mut normed = g.tensor(
        format!("{name}.layers.0.normed"),
        half(s.hidden),
        Kind::Intermediate,
    );
    g.op(
        format!("{name}.input_norm"),
        norm,
        &[run, x, stack.layers[0].input_norm, normed],
        row_groups,
    );
    for (l, w) in stack.layers.iter().enumerate() {
        let mut t = |what: &str, bytes| {
            g.tensor(
                format!("{name}.layers.{l}.{what}"),
                bytes,
                Kind::Intermediate,
            )
        };
        let qkv = t("qkv", single(qkv_width));
        let q = t("q", half(s.heads * s.head_dim));
        let attended = t("attended", half(s.heads * s.head_dim));
        let o = t("o", single(s.hidden));
        let mid = t("mid", half(s.hidden));
        let mid_normed = t("mid_normed", half(s.hidden));
        let gate_up = t("gate_up", single(2 * s.inner));
        let mlp = t("mlp", half(s.inner));
        let down = t("down", single(s.hidden));
        let output = t("out", half(s.hidden));
        let (next_normed, gain) = match stack.layers.get(l + 1) {
            Some(next) => (
                g.tensor(
                    format!("{name}.layers.{}.normed", l + 1),
                    half(s.hidden),
                    Kind::Intermediate,
                ),
                next.input_norm,
            ),
            None => (out, stack.norm),
        };
        g.op(
            "qkv_proj",
            qkv_gemm.0.clone(),
            &[normed, w.qkv, qkv],
            qkv_gemm.1,
        );
        g.op(
            "rope_rows",
            rope_pipeline.clone(),
            &[run, qkv, stack.cos, stack.sin, q, w.k_cache, w.v_cache],
            row_groups,
        );
        g.op(
            "attention_rows",
            attention.clone(),
            &[run, q, w.k_cache, w.v_cache, attended],
            [(rows / per_group) as u32, s.kv_heads as u32, 1],
        );
        g.op("o_proj", o_gemm.0.clone(), &[attended, w.o, o], o_gemm.1);
        g.op(
            "add_norm_rows",
            add_norm.clone(),
            &[run, o, input, w.post_norm, mid, mid_normed],
            row_groups,
        );
        g.op(
            "gate_up_proj",
            gate_up_gemm.0.clone(),
            &[mid_normed, w.gate_up, gate_up],
            gate_up_gemm.1,
        );
        g.op(
            "silu_mul_rows",
            silu.clone(),
            &[run, gate_up, mlp],
            row_groups,
        );
        g.op(
            "down_proj",
            down_gemm.0.clone(),
            &[mlp, w.down, down],
            down_gemm.1,
        );
        g.op(
            "add_norm_rows",
            add_norm.clone(),
            &[run, down, mid, gain, output, next_normed],
            row_groups,
        );
        input = output;
        normed = next_normed;
    }
    Ok(())
}

/// A stack's prefill on the GPU: programs over a [`Decoder`]'s memory, one for each number of
/// rows up to its most in steps of [`TILE`], which write a run's keys and values into the first
/// slot of the decoder's cache for its steps to go on from.
#[derive(Debug)]
pub struct Prefill {
    /// `programs[i]` runs `(i + 1) TILE` rows.
    programs: Vec<Program>,
    run: TensorId,
    x: TensorId,
    out: TensorId,
    max_rows: usize,
    hidden: usize,
    capacity: usize,
}

impl Prefill {
    /// Records the prefill programs of the stack `decoder` runs over its weights, rotation tables
    /// and cache, for runs of up to `max_rows` positions at once, the kernels at `tuning`'s
    /// subgroup size; timed per dispatch when `timed`.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        decoder: &mut Decoder,
        max_rows: usize,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let (decode, step) = decoder.parts_mut();
        let graphs = (1..=max_rows / TILE)
            .map(|tiles| prefill_graph(decode, tiles * TILE, max_rows, tuning))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(largest) = graphs.last() else {
            return Err(Error::Shape(format!("programs of up to {max_rows} rows")));
        };
        let (run, x, out) = (largest.run, largest.x, largest.out);
        let refs: Vec<&Graph> = graphs.iter().map(|p| &p.graph).collect();
        let programs = step.add(ctx, kernels, &refs, timed)?;
        Ok(Prefill {
            programs,
            run,
            x,
            out,
            max_rows,
            hidden: decode.shape.hidden,
            capacity: decode.shape.capacity,
        })
    }

    /// The most positions one program runs.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Runs the stack over `x`, the hidden states `[len][hidden]` of positions `start` to
    /// `start + len - 1`, in `decoder`, the one the programs were recorded over: writes their
    /// keys and values into its cache's first slot there, each attending over the slot up to it.
    /// Returns their outputs after the final norm, `[len][hidden]`. A run of more than
    /// [`max_rows`](Self::max_rows) positions runs in pieces.
    pub fn run(&self, decoder: &mut Decoder, start: usize, x: &[F16]) -> Result<Vec<F16>, Error> {
        let hidden = self.hidden;
        let len = x.len() / hidden;
        if len == 0 || !x.len().is_multiple_of(hidden) || start + len > self.capacity {
            return Err(Error::Shape(format!(
                "{} values from position {start} in a cache of {}",
                x.len(),
                self.capacity
            )));
        }
        let (_, step) = decoder.parts_mut();
        if !step.holds(self.programs[0]) {
            return Err(Error::Shape(
                "a decoder the prefill was not recorded for".into(),
            ));
        }
        let max_rows = self.max_rows;
        let mut out = Vec::with_capacity(x.len());
        let mut padded = vec![F16::ZERO; max_rows * hidden];
        for (i, piece) in x.chunks(max_rows * hidden).enumerate() {
            let rows = piece.len() / hidden;
            let at = start + i * max_rows;
            padded[..piece.len()].copy_from_slice(piece);
            step.write(self.run, &[at as u32, rows as u32, 0])?;
            step.write(self.x, &padded)?;
            step.run(self.programs[rows.div_ceil(TILE) - 1])?;
            let all: Vec<F16> = step.read(self.out)?;
            out.extend_from_slice(&all[..piece.len()]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decode::{SUBGROUPS, Shape, decode_graph},
        graph::{barriers, plan},
    };

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

    fn program(layers: usize, rows: usize, max_rows: usize) -> Result<PrefillGraph, Error> {
        let tuning = Tuning::for_subgroup(64);
        let decode = decode_graph(&shape(layers), &tuning)?;
        prefill_graph(&decode, rows, max_rows, &tuning)
    }

    #[test]
    fn a_layer_is_nine_dispatches_after_the_first_norm_each_after_a_barrier() {
        let p = program(28, 512, 1024).unwrap();
        assert_eq!(p.graph.ops.len(), 1 + 9 * 28);
        let plan = plan(&p.graph);
        let barriers = barriers(&p.graph, &plan);
        assert!(
            barriers[1..].iter().all(|&b| b),
            "every op reads the one before"
        );
        // The arena holds one layer's intermediates at a time, not 28 layers' worth: at most
        // what is live at SiLU, the gate/up projection in f32, its output and the residual.
        let silu = 1024 * (2 * 6144 * 4 + 6144 * 2 + 2048 * 2);
        assert!(plan.arena_bytes <= silu, "{}", plan.arena_bytes);
    }

    #[test]
    fn every_program_of_a_size_lays_its_tensors_out_alike() {
        let first = program(2, TILE, 512).unwrap();
        let plan_of = |p: &PrefillGraph| plan(&p.graph);
        for rows in (2 * TILE..=512).step_by(TILE) {
            let p = program(2, rows, 512).unwrap();
            let tensors = |p: &PrefillGraph| {
                let t = p.graph.tensors.iter();
                t.map(|t| (t.name.clone(), t.bytes, t.kind))
                    .collect::<Vec<_>>()
            };
            assert_eq!(tensors(&p), tensors(&first), "{rows} rows");
            assert_eq!(p.graph.lifetimes(), first.graph.lifetimes(), "{rows} rows");
            assert_eq!(plan_of(&p), plan_of(&first), "{rows} rows");
        }
    }

    #[test]
    fn the_gemms_compiled_are_every_projection_at_every_size() {
        for subgroup in SUBGROUPS {
            for rows in (TILE..=MAX_ROWS).step_by(TILE) {
                let tuning = Tuning::for_subgroup(subgroup);
                let decode = decode_graph(&shape(1), &tuning).unwrap();
                let p = prefill_graph(&decode, rows, MAX_ROWS, &tuning).unwrap();
                let gemms = p.graph.ops.iter().filter(|op| op.label.ends_with("_proj"));
                for op in gemms {
                    let [m, n, _] = op.groups.map(|g| g as usize * TILE);
                    assert_eq!(m, rows, "{}", op.label);
                    let name = op.pipeline.kernel.name;
                    assert!(name.contains(&format!("_{rows}x{n}x")), "{name}");
                    assert!(name.ends_with(&format!("_sg{subgroup}")), "{name}");
                    assert_eq!(op.pipeline.subgroup_size, Some(subgroup));
                }
            }
        }
        let names: std::collections::BTreeSet<_> = GEMMS.iter().flatten().map(|g| g.name).collect();
        assert_eq!(names.len(), GEMMS.len() * GEMMS[0].len());
    }

    #[test]
    fn runs_the_kernels_cannot_cover_exactly_are_refused() {
        for (rows, max_rows) in [(100, 1024), (256, 128), (0, 128), (1152, 1152), (128, 1000)] {
            assert!(
                matches!(program(1, rows, max_rows), Err(Error::Shape(_))),
                "{rows} of {max_rows}"
            );
        }
        let tuning = Tuning::for_subgroup(64);
        let mut s = shape(1);
        s.inner = 5120;
        let decode = decode_graph(&s, &tuning).unwrap();
        assert!(matches!(
            prefill_graph(&decode, 128, 128, &tuning),
            Err(Error::Shape(why)) if why.contains("no GEMM")
        ));
    }
}
