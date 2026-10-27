//! A [`Graph`] made runnable: its arena and buffers allocated as [`plan`] lays them out, each in
//! the memory its kind wants ([`Kind::residence`](crate::graph::Kind::residence)), and its
//! dispatches recorded once, with the barriers [`barriers`] finds, into a [`Recording`]
//! submitted every step.

use std::time::{Duration, Instant};

use vulkan::{
    Binding, Buffer, Context, Dispatch, Recording, Residence, Submission, Timing, memory::Plain,
};

use crate::{
    Error,
    graph::{ALIGN, Graph, Kind, Plan, TensorId, barriers, plan},
    kernels::Kernels,
};

/// Where a tensor's bytes live.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// At this offset of the arena, buffer 0.
    Arena(u64),
    /// In a buffer of its own.
    Own(usize),
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

/// A graph recorded for the GPU, with the buffers it binds.
pub struct Step {
    recording: Recording,
    layout: Layout,
    places: Vec<Place>,
    bytes: Vec<u64>,
    labels: Vec<String>,
    barriers: usize,
    recorded_in: Duration,
}

impl std::fmt::Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Step")
            .field("dispatches", &self.labels.len())
            .field("barriers", &self.barriers)
            .field("arena_bytes", &self.layout.plan.arena_bytes)
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
            vec![ctx.buffer_in(layout.plan.arena_bytes.max(ALIGN), Residence::Device)?];
        let mut places = Vec::with_capacity(graph.tensors.len());
        for (tensor, offset) in graph.tensors.iter().zip(&layout.plan.offsets) {
            places.push(match offset {
                Some(offset) => Place::Arena(*offset),
                None => {
                    buffers.push(ctx.buffer_in(tensor.bytes, tensor.kind.residence())?);
                    Place::Own(buffers.len() - 1)
                }
            });
        }
        Self::record(ctx, kernels, graph, layout, places, buffers, timed)
    }

    /// The step recorded again from `graph`, which must lay its tensors out as this step's graph
    /// does, such as the same graph cut another way, over the memory this step holds, weights and
    /// state included; timed per dispatch when `timed`. On an error the memory goes with the step.
    pub fn recorded_again(
        self,
        ctx: &Context,
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
        let Step {
            recording, places, ..
        } = self;
        let buffers = recording.into_buffers()?;
        Self::record(ctx, kernels, graph, layout, places, buffers, timed)
    }

    /// Records `graph`'s dispatches over `buffers`, where `places` puts its tensors.
    fn record(
        ctx: &Context,
        kernels: &mut Kernels,
        graph: &Graph,
        layout: Layout,
        places: Vec<Place>,
        buffers: Vec<Buffer>,
        timed: bool,
    ) -> Result<Self, Error> {
        let needs_barrier = barriers(graph, &layout.plan);
        let bytes: Vec<u64> = graph.tensors.iter().map(|t| t.bytes).collect();
        let binding = |t: TensorId| match places[t.0] {
            Place::Arena(offset) => Binding::range(0, offset, bytes[t.0]),
            Place::Own(buffer) => Binding::whole(buffer),
        };
        let mut dispatches = Vec::with_capacity(graph.ops.len());
        for (op, &barrier) in graph.ops.iter().zip(&needs_barrier) {
            dispatches.push(Dispatch {
                kernel: kernels.get(&op.pipeline)?,
                bindings: op.bindings.iter().map(|&t| binding(t)).collect(),
                groups: op.groups,
                barrier,
            });
        }
        let start = Instant::now();
        let recording = ctx.record(buffers, &dispatches, timed)?;
        Ok(Step {
            recording,
            recorded_in: start.elapsed(),
            layout,
            places,
            bytes,
            labels: graph.ops.iter().map(|op| op.label.clone()).collect(),
            barriers: needs_barrier.iter().filter(|&&b| b).count(),
        })
    }

    /// The buffer tensor `t` lives in, and where in it.
    fn place(&self, t: TensorId) -> (usize, u64) {
        match self.places[t.0] {
            Place::Arena(offset) => (0, offset),
            Place::Own(buffer) => (buffer, 0),
        }
    }

    /// Writes `values` into tensor `t`, which they must fill exactly: in place if the host maps
    /// it, through a staging copy if it lives in the GPU's own memory.
    ///
    /// # Panics
    ///
    /// If they do not fill it.
    pub fn write<T: Plain>(&mut self, t: TensorId, values: &[T]) -> Result<(), Error> {
        let len = self.bytes[t.0];
        assert_eq!(
            size_of_val(values) as u64,
            len,
            "tensor {} is {len} bytes",
            t.0
        );
        let (buffer, offset) = self.place(t);
        Ok(self.recording.buffer_mut(buffer).write_at(offset, values)?)
    }

    /// Tensor `t`'s contents.
    pub fn read<T: Plain>(&self, t: TensorId) -> Result<Vec<T>, Error> {
        let (buffer, offset) = self.place(t);
        let count = self.bytes[t.0] as usize / size_of::<T>();
        Ok(self.recording.buffer(buffer).read_at(offset, count)?)
    }

    /// Runs the step once and waits for it.
    pub fn submit(&mut self) -> Result<(), Error> {
        Ok(self.recording.submit()?)
    }

    /// Starts the step and returns without waiting for it.
    pub fn start(&mut self) -> Result<Submission<'_>, Error> {
        Ok(self.recording.start()?)
    }

    /// The GPU time of the last run, from before its first dispatch to after its last.
    pub fn elapsed(&self) -> Result<Timing, Error> {
        Ok(self.recording.elapsed()?)
    }

    /// The last run's timestamps before its first dispatch and after its last, in ticks.
    pub fn ends(&self) -> Result<[u64; 2], Error> {
        Ok(self.recording.ends()?)
    }

    /// Each dispatch's label and GPU time in the last run, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        let timings = self.recording.timings()?;
        Ok(timings.map(|t| self.labels.iter().map(String::as_str).zip(t).collect()))
    }

    pub fn dispatches(&self) -> usize {
        self.labels.len()
    }

    pub fn barriers(&self) -> usize {
        self.barriers
    }

    pub fn arena_bytes(&self) -> u64 {
        self.layout.plan.arena_bytes
    }

    /// How long writing the descriptor sets and recording the commands took.
    pub fn recorded_in(&self) -> Duration {
        self.recorded_in
    }
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
