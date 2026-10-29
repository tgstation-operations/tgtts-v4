//! VoxCPM2's patch sampler, as voxcpm's `Cfm::sample` computes it: Euler steps along the flow
//! the local DiT predicts, from noise at time 1 to a patch at time 0, with classifier-free
//! guidance as CFG-Zero* has it. It samples a patch for each of several rows, requests apart
//! from each other, at once: every step runs the DiT once over each row's two sequences, the
//! conditional and the unconditional, so that its weights are read once for them all; the whole
//! sample is one submission.
//!
//! The DiT's sequence is [`MU_TOKENS`] condition tokens, the time, the patch before and the
//! noisy patch. What the LM side hands the sampler for each row, as [`Sampler::sample`] takes
//! it, each row's after the one before:
//!
//! - `mu`, `[MU_TOKENS][hidden]` in f16: `lm_to_dit_proj` of the base LM's last hidden state,
//!   then `res_to_dit_proj` of the residual LM's, as voxcpm's model joins them;
//! - `cond`, `[feat_dim][patch_size]` in f32: the patch before, as the sampler gave it;
//! - `noise`, `[feat_dim][patch_size]` in f32: standard normal, as voxcpm's `Noise` draws it;
//! - the temperature and the guidance scale.
//!
//! It hands back each row's patch, `[feat_dim][patch_size]` in f32, the layout voxcpm's sampler
//! gives and the local encoder ([`super::encoder`]) takes; voxcpm's model hands its LMs the patch
//! transposed.

use vulkan::{Context, SpecValue, Timing};

use super::{
    CFM_NOISE, CLUSTER, DIT_EULER, LOCAL_EMBED, PatchConfig, Stack, Tuning, groups, identity,
    write_weight,
};
use crate::{
    Error,
    checkpoint::Checkpoint,
    graph::{Graph, Kind, Pipeline, TensorId},
    kernels::Kernels,
    step::Step,
};

/// The condition tokens leading the DiT's sequence, the two LMs' hidden states projected.
pub const MU_TOKENS: usize = 2;

/// Where the parts of the DiT's sequence start: the condition tokens, the time, the patch
/// before, the noisy patch.
const TIME_AT: usize = MU_TOKENS;
const COND_AT: usize = TIME_AT + 1;

/// The `timesteps + 1` times upstream's sampler steps through, as voxcpm's `schedule`: evenly
/// spaced from 1 to 0, as torch's `linspace` computes them in f32, then swayed towards 1 by
/// `t + cos(pi t / 2) - 1`.
pub fn schedule(timesteps: usize) -> Vec<f32> {
    let steps = timesteps + 1;
    let step = (0.0 - 1.0) / timesteps as f32;
    (0..steps)
        .map(|i| {
            let t = if i < steps / 2 {
                1.0 + step * i as f32
            } else {
                0.0 - step * (steps - i - 1) as f32
            };
            t + ((std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
        })
        .collect()
}

/// The time and the step length of each step the DiT runs at, in `timesteps` steps, as
/// voxcpm's `solve_euler` steps them in f32: the first 4% of the steps, but at least one, stay
/// where they are, and run nothing.
pub fn euler_steps(timesteps: usize) -> Vec<(f32, f32)> {
    let t_span = schedule(timesteps.max(1));
    let zero_init_steps = 1.max((t_span.len() as f64 * 0.04) as usize);
    let (mut t, mut dt) = (t_span[0], t_span[0] - t_span[1]);
    let mut steps = Vec::new();
    for step in 1..t_span.len() {
        if step > zero_init_steps {
            steps.push((t, dt));
        }
        t -= dt;
        if step < t_span.len() - 1 {
            dt = t - t_span[step + 1];
        }
    }
    steps
}

/// The sampler's graph and the tensors the host writes and reads, each a row after a row.
#[derive(Clone, Debug)]
pub struct SamplerGraph {
    pub graph: Graph,
    pub stack: Stack,
    pub rows: usize,
    pub feat_dim: usize,
    pub patch_size: usize,
    /// The time and step length of each step the DiT runs at.
    pub steps: Vec<(f32, f32)>,
    pub mu: TensorId,
    pub cond: TensorId,
    pub noise: TensorId,
    /// Each row's temperature, then its guidance scale, in f32.
    pub sampling: TensorId,
    /// Each row's patch, written in place by each step.
    pub patch: TensorId,
    pub in_proj: (TensorId, TensorId),
    pub cond_proj: (TensorId, TensorId),
    pub out_proj: (TensorId, TensorId),
    /// The time token of each step.
    pub times: Vec<TensorId>,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The sampler of `config`'s local DiT in `timesteps` steps over `rows` rows, its kernels cut as
/// `tuning` says.
pub fn sampler_graph(
    config: &PatchConfig,
    timesteps: usize,
    rows: usize,
    tuning: &Tuning,
) -> Result<SamplerGraph, Error> {
    let dit = &config.dit_config;
    if dit.mean_mode || dit.dit_mean_mode {
        return Err(Error::Shape("the DiT's mean mode".into()));
    }
    let (feat_dim, patch_size) = (config.feat_dim, config.patch_size);
    let tokens = COND_AT + 2 * patch_size;
    let shape = config.stack(&dit.stack, tokens)?;
    let hidden = shape.hidden;
    let mut g = Graph::default();
    if rows == 0 {
        return Err(Error::Shape("no rows".into()));
    }
    let stack = Stack::new(&mut g, "dit", &shape, 2 * rows, tuning)?;
    let x_at = COND_AT + patch_size;
    let Tuning {
        subgroup,
        workgroup,
        ..
    } = *tuning;
    let common = [
        (0, SpecValue::U32(subgroup)),
        (1, SpecValue::U32(workgroup)),
    ];
    let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
        kernel,
        constants: common.iter().chain(constants).copied().collect(),
        subgroup_size: Some(subgroup),
    };
    // The tokens copied are mu, a row's own, or the time, the same for every row.
    let embed = |first: usize, at: usize, copied: usize, guided: bool, shared: bool| {
        pipeline(
            &LOCAL_EMBED,
            &[
                (2, u(feat_dim)),
                (3, u(patch_size)),
                (4, u(hidden)),
                (5, u(2)),
                (6, u(tokens)),
                (7, u(first)),
                (8, u(at)),
                (9, u(copied)),
                (10, SpecValue::Bool(guided)),
                (11, SpecValue::Bool(shared)),
            ],
        )
    };
    let embed_groups = groups("embedded columns", hidden, workgroup as usize)?;
    if !feat_dim.is_multiple_of(8) {
        return Err(Error::Shape(format!("{feat_dim} channels")));
    }
    let len = feat_dim * patch_size;
    let clusters = workgroup as usize / CLUSTER;
    if !feat_dim.is_multiple_of(clusters) {
        return Err(Error::Shape(format!(
            "{feat_dim} channels over {clusters} clusters"
        )));
    }
    let euler = |dt: f32| {
        pipeline(
            &DIT_EULER,
            &[
                (2, u(hidden)),
                (3, u(tokens)),
                (4, u(x_at)),
                (5, u(patch_size)),
                (6, u(feat_dim)),
                (7, SpecValue::F32(dt)),
                (8, u(feat_dim / clusters)),
            ],
        )
    };

    let half = |n: usize| 2 * n as u64;
    let float = |n: usize| 4 * n as u64;
    let mu = g.tensor("mu", half(rows * MU_TOKENS * hidden), Kind::Input);
    let cond = g.tensor("cond", float(rows * len), Kind::Input);
    let noise = g.tensor("noise", float(rows * len), Kind::Input);
    let sampling = g.tensor("sampling", float(rows * 2), Kind::Param);
    let patch = g.tensor("patch", float(rows * len), Kind::Output);
    let mut linear = |name: &str, n: usize, k: usize| {
        (
            g.tensor(format!("{name}.weight"), half(n * k), Kind::Constant),
            g.tensor(format!("{name}.bias"), half(n), Kind::Constant),
        )
    };
    let in_proj = linear("in_proj", hidden, feat_dim);
    let cond_proj = linear("cond_proj", hidden, feat_dim);
    let out_proj = linear("out_proj", feat_dim, hidden);
    let steps = euler_steps(timesteps);
    let times: Vec<TensorId> = (0..steps.len())
        .map(|s| g.tensor(format!("time.{s}"), half(hidden), Kind::Constant))
        .collect();
    let x = g.tensor("dit.x", half(stack.tokens() * hidden), Kind::Intermediate);

    let each_row = u32::try_from(rows).map_err(|_| Error::Shape(format!("{rows} rows")))?;
    g.op(
        "noise",
        pipeline(&CFM_NOISE, &[(2, u(len)), (3, u(rows))]),
        &[noise, sampling, patch],
        [1, 1, 1],
    );
    g.op(
        "cond_proj",
        embed(COND_AT, 0, MU_TOKENS, true, false),
        &[cond, cond_proj.0, cond_proj.1, mu, x],
        [embed_groups, each_row, 1],
    );
    for (s, &(_, dt)) in steps.iter().enumerate() {
        g.op(
            format!("step.{s}.in_proj"),
            embed(x_at, TIME_AT, 1, false, true),
            &[patch, in_proj.0, in_proj.1, times[s], x],
            [embed_groups, each_row, 1],
        );
        let out = stack.layers(&mut g, &format!("step.{s}.dit"), x);
        g.op(
            format!("step.{s}.euler"),
            euler(dt),
            &[out, out_proj.0, out_proj.1, sampling, patch],
            [1, each_row, 1],
        );
    }
    Ok(SamplerGraph {
        graph: g,
        stack,
        rows,
        feat_dim,
        patch_size,
        steps,
        mu,
        cond,
        noise,
        sampling,
        patch,
        in_proj,
        cond_proj,
        out_proj,
        times,
    })
}

/// VoxCPM2's patch sampler on the GPU, with its weights.
#[derive(Debug)]
pub struct Sampler {
    step: Step,
    tensors: SamplerGraph,
}

impl Sampler {
    /// Records the sampler of `config` in `timesteps` steps over `rows` rows, every weight zero
    /// until [`load`](Self::load).
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        config: &PatchConfig,
        timesteps: usize,
        rows: usize,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = sampler_graph(config, timesteps, rows, tuning)?;
        let step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        Ok(Sampler { step, tensors })
    }

    /// Writes the weights of VoxCPM2's local DiT from `checkpoint`, converted to f16, and each
    /// step's time token, computed from them, as [`load_sampler`] does.
    pub fn load(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let t = &self.tensors;
        load_sampler(&mut self.step, &t.graph, t, &identity(&t.graph), checkpoint)
    }

    /// Samples each row's patch from its `noise`, given its conditions `mu` and its patch
    /// before, `cond`, at its temperature and guidance scale in `sampling`, each as the module's
    /// documentation lays it out.
    pub fn sample(
        &mut self,
        mu: &[u16],
        cond: &[f32],
        noise: &[f32],
        sampling: &[[f32; 2]],
    ) -> Result<Vec<f32>, Error> {
        let t = &self.tensors;
        self.step.write(t.mu, mu)?;
        self.step.write(t.cond, cond)?;
        self.step.write(t.noise, noise)?;
        self.step.write(t.sampling, sampling.as_flattened())?;
        self.step.submit()?;
        self.step.read(t.patch)
    }

    /// Each dispatch's label and GPU time in the last sample, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn tensors(&self) -> &SamplerGraph {
        &self.tensors
    }

    /// The recorded sample's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}

/// Writes the weights of VoxCPM2's local DiT from `checkpoint`, converted to f16, and each
/// step's time token, computed from them, into `step`, a recording of `graph`, where `ids` puts
/// each of `sampler`'s tensors: as [`splice`](crate::lm_side::splice) gives them when the
/// sampler's graph is part of `graph`.
pub fn load_sampler(
    step: &mut Step,
    graph: &Graph,
    sampler: &SamplerGraph,
    ids: &[TensorId],
    checkpoint: &Checkpoint,
) -> Result<(), Error> {
    let t = sampler;
    let at = |t: TensorId| ids[t.0];
    let prefix = "feat_decoder.estimator.";
    t.stack
        .load(step, graph, ids, checkpoint, &format!("{prefix}decoder."))?;
    for (name, (weight, bias)) in [
        ("in_proj", t.in_proj),
        ("cond_proj", t.cond_proj),
        ("out_proj", t.out_proj),
    ] {
        let name = |part: &str| vec![format!("{prefix}{name}.{part}")];
        write_weight(step, graph, checkpoint, at(weight), &name("weight"))?;
        write_weight(step, graph, checkpoint, at(bias), &name("bias"))?;
    }
    let hidden = t.stack.shape.hidden;
    let mlp = |name: &str| TimeMlp::read(checkpoint, &format!("{prefix}{name}"), hidden);
    let (time, delta) = (mlp("time_mlp")?, mlp("delta_time_mlp")?);
    // The step length is the DiT's input only in mean mode, which the graph refuses: the delta
    // time MLP sees zero at every step.
    let constant = delta.forward(&time_embedding(0.0, hidden));
    for (&token, &(time_at, _)) in t.times.iter().zip(&t.steps) {
        let values: Vec<u16> = time
            .forward(&time_embedding(time_at, hidden))
            .iter()
            .zip(&constant)
            .map(|(a, b)| time_constant(a + b))
            .collect();
        step.write(at(token), &values)?;
    }
    Ok(())
}

/// The bits of `value` as a time token holds it, an f16 rounded through f32 in every build:
/// half's f16::from_f64 rounds an f64 to f32 first where half is built with std and the CPU has
/// F16C, and once where not, and half has std only in builds where something else asks for it.
pub fn time_constant(value: f64) -> u16 {
    half::f16::from_f32(value as f32).to_bits()
}

/// Upstream's `SinusoidalPosEmb` at scale 1000, in f32 and in the order voxcpm's operations
/// round in f32: the sines, then the cosines, of `1000 t` at each frequency.
fn time_embedding(t: f32, hidden: usize) -> Vec<f64> {
    let half = hidden / 2;
    let rate = -(10000f64.ln() / (half - 1) as f64) as f32;
    let t = t * 1000.0;
    let angles: Vec<f32> = (0..half).map(|i| t * (i as f32 * rate).exp()).collect();
    let sines = angles.iter().map(|a| f64::from(a.sin()));
    sines
        .chain(angles.iter().map(|a| f64::from(a.cos())))
        .collect()
}

/// A timestep embedding's MLP, `linear_2(SiLU(linear_1(x)))`, on the host in f64 over its
/// weights rounded to f16, as the GPU holds them.
struct TimeMlp {
    layers: [(Vec<f64>, Vec<f64>); 2],
}

impl TimeMlp {
    fn read(checkpoint: &Checkpoint, prefix: &str, hidden: usize) -> Result<Self, Error> {
        let read = |name: String, len: usize| -> Result<Vec<f64>, Error> {
            let (_, values) = checkpoint.f16(&name)?;
            if values.len() != len {
                return Err(Error::Checkpoint(format!(
                    "{name}: {} values, not {len}",
                    values.len()
                )));
            }
            Ok(values
                .into_iter()
                .map(|h| half::f16::from_bits(h).to_f64())
                .collect())
        };
        let layer = |i: usize| -> Result<(Vec<f64>, Vec<f64>), Error> {
            Ok((
                read(format!("{prefix}.linear_{i}.weight"), hidden * hidden)?,
                read(format!("{prefix}.linear_{i}.bias"), hidden)?,
            ))
        };
        Ok(TimeMlp {
            layers: [layer(1)?, layer(2)?],
        })
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let linear = |(w, b): &(Vec<f64>, Vec<f64>), x: &[f64]| -> Vec<f64> {
            w.chunks(x.len())
                .zip(b)
                .map(|(row, b)| b + row.iter().zip(x).map(|(w, x)| w * x).sum::<f64>())
                .collect()
        };
        let inner: Vec<f64> = linear(&self.layers[0], x)
            .into_iter()
            .map(|v| v / (1.0 + (-v).exp()))
            .collect();
        linear(&self.layers[1], &inner)
    }
}
