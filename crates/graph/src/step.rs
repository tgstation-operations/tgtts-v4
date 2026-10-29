//! A [`Graph`] made runnable: its arena and buffers allocated as [`plan`] lays them out, each in
//! the memory its kind wants ([`Kind::residence`](crate::graph::Kind::residence)), and its
//! dispatches recorded once, with the barriers [`barriers`] finds, into a [`Recording`]
//! submitted every step. More graphs over the same memory may be recorded beside it as programs
//! of their own ([`Step::add`]), such as a prefill beside a decode step.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use vulkan::{
    Binding, Context, Dispatch, Element, Memory, Recording, Residence, Submission, Timing,
};

use crate::{
    Error,
    graph::{ALIGN, Graph, Holds, Kind, Plan, TensorId, barriers, plan, plan_programs},
    kernels::Kernels,
};

/// Where a tensor's bytes live: from an offset of an arena, or from the start of a buffer of
/// its own.
#[derive(Clone, Copy, Debug)]
struct Place {
    buffer: usize,
    offset: u64,
}

/// What decides where a graph's tensors live: their sizes and kinds, and the arena's plan.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    tensors: Vec<(u64, Kind)>,
    plan: Plan,
}

impl Layout {
    fn of(graph: &Graph) -> Self {
        Layout {
            tensors: graph.tensors.iter().map(|t| (t.bytes, t.kind)).collect(),
            plan: plan(graph),
        }
    }
}

/// Bytes copied between a tensor and a buffer outside the step ([`Step::copy_out`],
/// [`Step::copy_in`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub tensor: TensorId,
    /// The offset in the tensor, and in the buffer outside.
    pub offset: u64,
    pub other: u64,
    pub bytes: u64,
}

/// One of the graphs a step runs, recorded over its memory ([`Step::add`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Program {
    step: u64,
    index: usize,
}

/// What a program's dispatches are called, and how many barriers it waits at.
#[derive(Clone, Debug)]
struct Recorded {
    labels: Vec<String>,
    barriers: usize,
}

/// Tells steps apart, so that a program is run only by the step it was recorded for.
static STEPS: AtomicU64 = AtomicU64::new(0);

/// A graph recorded for the GPU, with the buffers it binds, and any more graphs recorded over
/// them.
pub struct Step {
    recording: Recording,
    /// The first graph's.
    layout: Layout,
    /// Every program's tensors, the first graph's first.
    tensors: Vec<(u64, Kind)>,
    places: Vec<Place>,
    programs: Vec<Recorded>,
    /// The program run last.
    last: usize,
    id: u64,
    recorded_in: Duration,
}

impl std::fmt::Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Step")
            .field("dispatches", &self.programs[0].labels.len())
            .field("barriers", &self.programs[0].barriers)
            .field("arena_bytes", &self.layout.plan.arena_bytes)
            .field("programs", &self.programs.len())
            .finish_non_exhaustive()
    }
}

impl Step {
    /// Allocates `graph`'s memory, zero-filled, and records its dispatches.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        graph: &Graph,
        timed: bool,
    ) -> Result<Self, Error> {
        let layout = Layout::of(graph);
        let mut buffers =
            vec![ctx.memory_in(layout.plan.arena_bytes.max(ALIGN), Residence::Device)?];
        let mut places = Vec::with_capacity(graph.tensors.len());
        for (tensor, offset) in graph.tensors.iter().zip(&layout.plan.offsets) {
            places.push(match offset {
                &Some(offset) => Place { buffer: 0, offset },
                None => {
                    buffers.push(ctx.memory_in(tensor.bytes, tensor.kind.residence())?);
                    Place {
                        buffer: buffers.len() - 1,
                        offset: 0,
                    }
                }
            });
        }
        let tensors = layout.tensors.clone();
        let (dispatches, recorded) = dispatches(kernels, graph, &layout.plan, &tensors, &places)?;
        let start = Instant::now();
        let recording = ctx.record(buffers, &dispatches, timed)?;
        Ok(Step {
            recording,
            recorded_in: start.elapsed(),
            layout,
            tensors,
            places,
            programs: vec![recorded],
            last: 0,
            id: STEPS.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The step with its first graph recorded again from `graph`, which must lay its tensors out
    /// as that graph does, such as the same graph cut another way, over the memory this step
    /// holds, weights and state included; timed per dispatch when `timed`. The programs added
    /// since stay as they were. On an error the memory goes with the step.
    pub fn recorded_again(
        mut self,
        _ctx: &Context,
        kernels: &mut Kernels,
        graph: &Graph,
        timed: bool,
    ) -> Result<Self, Error> {
        let layout = Layout::of(graph);
        if layout != self.layout {
            return Err(Error::Shape(
                "a graph whose tensors are laid out unlike the step's".into(),
            ));
        }
        let (dispatches, recorded) =
            dispatches(kernels, graph, &layout.plan, &self.tensors, &self.places)?;
        let start = Instant::now();
        self.recording.replace_program(0, &dispatches, timed)?;
        self.recorded_in = start.elapsed();
        self.programs[0] = recorded;
        Ok(self)
    }

    /// Records `graphs` as more programs over this step's memory, timed per dispatch when
    /// `timed`, and returns them in order. Each extends the graphs recorded so far: its first
    /// tensors are theirs, at the same sizes and kinds, and the rest its own, the same in each
    /// of `graphs`. Their own tensors are allocated, zero-filled, those of arena kinds in an
    /// arena of their own laid out over all of `graphs` ([`plan_programs`]); none may use the
    /// tensors of an earlier arena.
    pub fn add(
        &mut self,
        ctx: &Context,
        kernels: &mut Kernels,
        graphs: &[&Graph],
        timed: bool,
    ) -> Result<Vec<Program>, Error> {
        let Some(&first) = graphs.first() else {
            return Ok(Vec::new());
        };
        let held = self.tensors.len();
        let shape = |g: &Graph| -> Vec<(u64, Kind)> {
            g.tensors.iter().map(|t| (t.bytes, t.kind)).collect()
        };
        let tensors = shape(first);
        if tensors.get(..held) != Some(&self.tensors[..])
            || graphs.iter().any(|&g| shape(g) != tensors)
        {
            return Err(Error::Shape(
                "graphs that do not extend the step's tensors alike".into(),
            ));
        }
        for op in graphs.iter().flat_map(|g| &g.ops) {
            if let Some(t) = op
                .bindings
                .iter()
                .find(|t| t.0 < held && tensors[t.0].1.in_arena())
            {
                return Err(Error::Shape(format!(
                    "{}: tensor {} of an earlier arena",
                    op.label, t.0
                )));
            }
        }
        let plan = plan_programs(graphs, held);
        let arena = match plan.arena_bytes {
            0 => None,
            bytes => Some(
                self.recording
                    .add_buffer(ctx.memory_in(bytes, Residence::Device)?)?,
            ),
        };
        for (t, &(bytes, kind)) in tensors.iter().enumerate().skip(held) {
            let place = match (plan.offsets[t], arena) {
                (Some(offset), Some(buffer)) => Place { buffer, offset },
                _ => Place {
                    buffer: self
                        .recording
                        .add_buffer(ctx.memory_in(bytes, kind.residence())?)?,
                    offset: 0,
                },
            };
            self.places.push(place);
            self.tensors.push((bytes, kind));
        }
        let mut programs = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let (dispatches, recorded) =
                dispatches(kernels, graph, &plan, &self.tensors, &self.places)?;
            let index = self.recording.add_program(&dispatches, timed)?;
            self.programs.push(recorded);
            programs.push(Program {
                step: self.id,
                index,
            });
        }
        Ok(programs)
    }

    /// The buffer tensor `t` lives in, and where in it.
    fn place(&self, t: TensorId) -> (usize, u64) {
        let Place { buffer, offset } = self.places[t.0];
        (buffer, offset)
    }

    /// The buffer tensor `t` of elements of `E` lives in, and the element of it it starts at: an
    /// arena's offsets are multiples of [`ALIGN`], and so of any element's size.
    fn element_place<E: Element>(&self, t: TensorId) -> (usize, usize) {
        let (buffer, offset) = self.place(t);
        (buffer, (offset / E::TYPE.bytes()) as usize)
    }

    /// Writes `values` into tensor `t`, which they must fill exactly: in place if the host maps
    /// it, through a staging copy if it lives in the GPU's own memory.
    ///
    /// # Panics
    ///
    /// If they do not fill it.
    pub fn write<E: Element>(&mut self, t: impl Holds<E>, values: &[E]) -> Result<(), Error> {
        let t = t.id();
        let len = self.tensors[t.0].0;
        assert_eq!(
            size_of_val(values) as u64,
            len,
            "tensor {} is {len} bytes",
            t.0
        );
        let (buffer, at) = self.element_place::<E>(t);
        Ok(self.recording.buffer_mut(buffer).write_at(at, values)?)
    }

    /// Tensor `t`'s contents.
    pub fn read<E: Element>(&self, t: impl Holds<E>) -> Result<Vec<E>, Error> {
        let t = t.id();
        let (buffer, at) = self.element_place::<E>(t);
        let count = (self.tensors[t.0].0 / E::TYPE.bytes()) as usize;
        Ok(self.recording.buffer::<E>(buffer).read_at(at, count)?)
    }

    /// Copies `regions` of tensors into `to`, on the GPU after the last run.
    pub fn copy_out(&mut self, regions: &[Region], to: &mut Memory) -> Result<(), Error> {
        let regions = self.regions(regions)?;
        Ok(self.recording.copy_out(&regions, to)?)
    }

    /// Copies `regions` of `from` into tensors, on the GPU after the last run.
    pub fn copy_in(&mut self, from: &Memory, regions: &[Region]) -> Result<(), Error> {
        let regions = self.regions(regions)?;
        Ok(self.recording.copy_in(from, &regions)?)
    }

    /// `regions` in the buffers the tensors live in.
    fn regions(&self, regions: &[Region]) -> Result<Vec<vulkan::Region>, Error> {
        regions
            .iter()
            .map(|r| {
                let len = self.tensors[r.tensor.0].0;
                if r.offset.checked_add(r.bytes).is_none_or(|end| end > len) {
                    return Err(Error::Shape(format!(
                        "{} bytes from {} of tensor {} of {len}",
                        r.bytes, r.offset, r.tensor.0
                    )));
                }
                let (buffer, at) = self.place(r.tensor);
                Ok(vulkan::Region {
                    buffer,
                    offset: at + r.offset,
                    other: r.other,
                    bytes: r.bytes,
                })
            })
            .collect()
    }

    /// Runs the step's first graph once and waits for it.
    pub fn submit(&mut self) -> Result<(), Error> {
        self.last = 0;
        Ok(self.recording.submit()?)
    }

    /// Starts the step's first graph and returns without waiting for it.
    pub fn start(&mut self) -> Result<Submission<'_>, Error> {
        self.last = 0;
        Ok(self.recording.start()?)
    }

    /// Runs `program` once and waits for it.
    pub fn run(&mut self, program: Program) -> Result<(), Error> {
        Ok(self.start_program(program)?.wait()?)
    }

    /// The step's first graph, as a program to [`run`](Self::run) beside those added.
    pub fn first(&self) -> Program {
        Program {
            step: self.id,
            index: 0,
        }
    }

    /// Whether `program` was recorded for this step, and so may run on it.
    pub fn holds(&self, program: Program) -> bool {
        program.step == self.id
    }

    /// Starts `program` and returns without waiting for it.
    pub fn start_program(&mut self, program: Program) -> Result<Submission<'_>, Error> {
        if !self.holds(program) {
            return Err(Error::Shape("a program of another step".into()));
        }
        self.last = program.index;
        Ok(self.recording.start_program(program.index)?)
    }

    /// The GPU time of the last run, from before its first dispatch to after its last.
    pub fn elapsed(&self) -> Result<Timing, Error> {
        Ok(self.recording.elapsed()?)
    }

    /// The last run's timestamps before its first dispatch and after its last, in ticks.
    pub fn ends(&self) -> Result<[u64; 2], Error> {
        Ok(self.recording.ends()?)
    }

    /// Each dispatch's label and GPU time in the last run, when its program was recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        let timings = self.recording.timings()?;
        let labels = &self.programs[self.last].labels;
        Ok(timings.map(|t| labels.iter().map(String::as_str).zip(t).collect()))
    }

    /// The first graph's dispatches.
    pub fn dispatches(&self) -> usize {
        self.programs[0].labels.len()
    }

    /// The barriers the first graph's dispatches wait at.
    pub fn barriers(&self) -> usize {
        self.programs[0].barriers
    }

    pub fn arena_bytes(&self) -> u64 {
        self.layout.plan.arena_bytes
    }

    /// How long writing the descriptor sets and recording the commands of the first graph took,
    /// when it was last recorded.
    pub fn recorded_in(&self) -> Duration {
        self.recorded_in
    }
}

/// `graph`'s dispatches with the barriers they need, its tensors of `tensors` where `places`
/// puts them and its arena's as `plan` lays them out, each checked against its kernel's
/// manifest; and what they are called.
fn dispatches(
    kernels: &mut Kernels,
    graph: &Graph,
    plan: &Plan,
    tensors: &[(u64, Kind)],
    places: &[Place],
) -> Result<(Vec<Dispatch>, Recorded), Error> {
    let needs_barrier = barriers(graph, plan);
    let binding = |t: TensorId| {
        let Place { buffer, offset } = places[t.0];
        Binding::range(buffer, offset, tensors[t.0].0)
    };
    let mut dispatches = Vec::with_capacity(graph.ops.len());
    for (op, &barrier) in graph.ops.iter().zip(&needs_barrier) {
        kernels.check(op, |t| tensors[t].0)?;
        dispatches.push(Dispatch {
            kernel: kernels.get(&op.pipeline)?,
            bindings: op.bindings.iter().map(|&t| binding(t)).collect(),
            groups: op.groups,
            barrier,
        });
    }
    let recorded = Recorded {
        labels: graph.ops.iter().map(|op| op.label.clone()).collect(),
        barriers: needs_barrier.iter().filter(|&&b| b).count(),
    };
    Ok((dispatches, recorded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Shape, Tuning, decode_graph};

    #[test]
    fn the_same_graph_cut_another_way_is_laid_out_alike_and_another_shape_is_not() {
        let shape = |capacity| Shape {
            hidden: 2048,
            inner: 6144,
            heads: 16,
            kv_heads: 2,
            head_dim: 128,
            layers: 2,
            capacity,
            eps: 1e-5,
            rope_theta: 10_000.0,
            rope_factors: vec![1.0; 64],
            rope: true,
        };
        let layout = |s: &Shape, t: &Tuning| Layout::of(&decode_graph(s, t).unwrap().graph);
        let base = Tuning::for_subgroup(32);
        let other = Tuning {
            workgroup: 256,
            qkv_pairs: 2,
            residual_rows: 2,
            ..base
        };
        let wave64 = Tuning::for_subgroup(64);
        assert_eq!(layout(&shape(256), &base), layout(&shape(256), &other));
        assert_eq!(layout(&shape(256), &base), layout(&shape(256), &wave64));
        assert_ne!(layout(&shape(256), &base), layout(&shape(512), &base));
    }
}
