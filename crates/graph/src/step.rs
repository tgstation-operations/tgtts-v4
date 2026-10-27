//! A [`Graph`] made runnable: its arena and buffers allocated as [`plan`] lays them out, and its
//! dispatches recorded once, with the barriers [`barriers`] finds, into a [`Recording`]
//! submitted every step.

use std::time::{Duration, Instant};

use vulkan::{Binding, Context, Dispatch, Recording, Timing};

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

/// A value the host writes into a tensor or reads out of it.
pub trait Scalar: Copy {
    const SIZE: usize;
    fn put(self, out: &mut [u8]);
    fn get(bytes: &[u8]) -> Self;
}

macro_rules! scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            const SIZE: usize = size_of::<$t>();
            fn put(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_ne_bytes());
            }
            fn get(bytes: &[u8]) -> Self {
                <$t>::from_ne_bytes(bytes.try_into().expect("a whole value"))
            }
        }
    )*};
}
scalar!(u16, u32, f32);

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
        let mut buffers = vec![ctx.buffer(plan.arena_bytes.max(ALIGN))?];
        let mut places = Vec::with_capacity(graph.tensors.len());
        for (tensor, offset) in graph.tensors.iter().zip(&plan.offsets) {
            places.push(match offset {
                Some(offset) => Place::Arena(*offset),
                None => {
                    buffers.push(ctx.buffer(tensor.bytes)?);
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

    fn range(&self, t: TensorId) -> (usize, std::ops::Range<usize>) {
        let len = self.bytes[t.0] as usize;
        match self.places[t.0] {
            Place::Arena(offset) => (0, offset as usize..offset as usize + len),
            Place::Own(buffer) => (buffer, 0..len),
        }
    }

    /// Writes `values` into tensor `t`, which they must fill exactly.
    ///
    /// # Panics
    ///
    /// If they do not.
    pub fn write<T: Scalar>(&mut self, t: TensorId, values: &[T]) {
        let (buffer, range) = self.range(t);
        assert_eq!(
            values.len() * T::SIZE,
            range.len(),
            "tensor {} is {} bytes",
            t.0,
            range.len()
        );
        let bytes = &mut self.recording.buffer_mut(buffer).bytes_mut()[range];
        for (v, out) in values.iter().zip(bytes.chunks_exact_mut(T::SIZE)) {
            v.put(out);
        }
    }

    /// Tensor `t`'s contents.
    pub fn read<T: Scalar>(&self, t: TensorId) -> Vec<T> {
        let (buffer, range) = self.range(t);
        let bytes = &self.recording.buffer(buffer).bytes()[range];
        bytes.chunks_exact(T::SIZE).map(T::get).collect()
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
