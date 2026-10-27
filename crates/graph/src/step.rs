//! A [`Graph`] made runnable: its arena and buffers allocated as [`plan`] lays them out, each in
//! the memory its kind wants ([`Kind::residence`](crate::graph::Kind::residence)), and its
//! dispatches recorded once, with the barriers [`barriers`] finds, into a [`Recording`]
//! submitted every step.

use std::time::{Duration, Instant};

use vulkan::{Binding, Context, Dispatch, Recording, Residence, Timing, memory::Plain};

use crate::{
    Error,
    graph::{ALIGN, Graph, TensorId, barriers, plan},
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

/// A graph recorded for the GPU, with the buffers it binds.
pub struct Step {
    recording: Recording,
    places: Vec<Place>,
    bytes: Vec<u64>,
    labels: Vec<String>,
    barriers: usize,
    arena_bytes: u64,
    recorded_in: Duration,
}

impl std::fmt::Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Step")
            .field("dispatches", &self.labels.len())
            .field("barriers", &self.barriers)
            .field("arena_bytes", &self.arena_bytes)
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
        let plan = plan(graph);
        let needs_barrier = barriers(graph, &plan);
        let mut buffers = vec![ctx.buffer_in(plan.arena_bytes.max(ALIGN), Residence::Device)?];
        let mut places = Vec::with_capacity(graph.tensors.len());
        for (tensor, offset) in graph.tensors.iter().zip(&plan.offsets) {
            places.push(match offset {
                Some(offset) => Place::Arena(*offset),
                None => {
                    buffers.push(ctx.buffer_in(tensor.bytes, tensor.kind.residence())?);
                    Place::Own(buffers.len() - 1)
                }
            });
        }
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
            places,
            bytes,
            labels: graph.ops.iter().map(|op| op.label.clone()).collect(),
            barriers: needs_barrier.iter().filter(|&&b| b).count(),
            arena_bytes: plan.arena_bytes,
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
        self.arena_bytes
    }

    /// How long writing the descriptor sets and recording the commands took.
    pub fn recorded_in(&self) -> Duration {
        self.recorded_in
    }
}
