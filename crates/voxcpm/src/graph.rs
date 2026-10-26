use std::{
    collections::HashMap,
    fmt,
    ops::Range,
    sync::{Arc, Mutex, PoisonError},
    thread::{self, ThreadId},
};

use burn::tensor::{Device, Graph, Int, Tensor, TensorData, capture, kind::Basic};
use minicpm4::KvCache;

use crate::{Error, Generation, Model, Noise, RowId, Sampling, Step};

/// How a generation computes each patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stepping {
    /// Kernel by kernel, the LMs attending over the positions their caches hold, a patch
    /// sampled and then fed to the LMs before the next.
    Eager,
    /// Kernel by kernel, by a step whose every shape is the same at every position: a patch
    /// sampled and fed to the LMs at once, the LMs attending over the whole capacity of their
    /// caches, at positions on the device. It computes what `Eager` does.
    Stable,
    /// By replaying a graph of the stable step, captured once for each capacity of cache and
    /// number of sampling steps on each thread: one launch where the device records graphs, the
    /// step run again where it does not.
    Captured,
}

impl Stepping {
    /// `Captured` where `device` records graphs, `Eager` where a replay would only run the step
    /// again.
    pub fn for_device(device: &Device) -> Self {
        let probe = capture(device, || Tensor::<1>::zeros([1], device).add_scalar(1));
        if probe.is_hardware() {
            Self::Captured
        } else {
            Self::Eager
        }
    }
}

/// The graphs a model has captured of its step, with the buffers each reads and writes, each
/// kept for the next generation of its key while none is using it.
#[derive(Default)]
pub(crate) struct Graphs(Mutex<HashMap<Key, StepGraph>>);

/// The rows and the capacity of the caches a graph steps, the sampling steps of each patch, and
/// the thread that captured it: a replay must be ordered after the writes of its inputs, which
/// cubecl orders only on the same thread.
type Key = (usize, usize, usize, ThreadId);

impl Graphs {
    fn take(&self, key: &Key) -> Option<StepGraph> {
        let mut graphs = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let graph = graphs.remove(key)?;
        {
            let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
            for row in 0..key.0 {
                state.caches.0.clear(row);
                state.caches.1.clear(row);
            }
        }
        Some(graph)
    }

    /// Keeps `graph` for `key` unless another generation kept one there first.
    fn give(&self, key: Key, graph: StepGraph) {
        let mut graphs = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        graphs.entry(key).or_insert(graph);
    }
}

/// A clone of a model captures graphs of its own.
impl Clone for Graphs {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl fmt::Debug for Graphs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let graphs = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        write!(f, "Graphs({})", graphs.len())
    }
}

/// What the stable step reads and writes, each in a buffer that stays where it is from one step
/// to the next: a graph replays its kernels on the buffers it was captured on. A row for each
/// generation, `rows` of them, in caches of `capacity` positions.
///
/// What a generation brings is written when it joins; between two steps the state moves on only
/// by copies of what the step gave, nothing written from the host.
struct StepState {
    caches: (KvCache, KvCache),
    /// The position each row's patch goes to, `[rows]`.
    positions: Tensor<1, Int>,
    /// The LMs' last hidden states `[rows, hidden]` and the last patches `[rows, patch_size,
    /// feat_dim]`, which the next patches are sampled after.
    lm_hidden: Tensor<2>,
    residual_hidden: Tensor<2>,
    cond: Tensor<3>,
    /// The noise of each row's patches, `capacity` of them a row, `[rows * capacity, feat_dim,
    /// patch_size]`; where each row's begins, `[rows]`; and which of its patches is next.
    noise: Tensor<3>,
    starts: Tensor<1, Int>,
    draws: Tensor<1, Int>,
    /// The temperature and the classifier-free guidance scale of each row, `[rows, 1, 1]`.
    temperature: Tensor<3>,
    cfg: Tensor<3>,
}

/// What the stable step gives: each row's patch, sampled after the state's, and the state after
/// it.
#[derive(Clone)]
struct StepOutputs {
    step: Step,
    lm_hidden: Tensor<2>,
    residual_hidden: Tensor<2>,
    positions: Tensor<1, Int>,
    draws: Tensor<1, Int>,
}

type StepFn = Box<dyn FnMut() -> StepOutputs + Send>;

/// The stable step's state, and the graph captured of it once it is.
struct StepGraph {
    state: Arc<Mutex<StepState>>,
    graph: Option<Graph<StepOutputs, StepFn>>,
    /// Whether the device says if a tensor shares its buffer, which a write to the state must
    /// not, or it would move where a replay does not look.
    shares_known: bool,
}

impl Model {
    /// The patch of each row sampled after `state`, then fed to the LMs at the row's position,
    /// in `timesteps` sampling steps; nothing in `state` written but the caches at the positions,
    /// so that running it again does what it did. A row past its last position stays at it.
    fn stable_step(&self, state: &mut StepState, timesteps: usize) -> StepOutputs {
        let last = state.caches.0.capacity() as i64 - 1;
        let StepState {
            caches,
            positions,
            draws,
            temperature,
            cfg,
            ..
        } = state;
        let at = state.starts.clone() + draws.clone();
        let noise = state.noise.clone().select(0, at);
        let step = self.sample_after(
            state.lm_hidden.clone(),
            state.residual_hidden.clone(),
            state.cond.clone(),
            |cfm, mu, cond| {
                let scales = (temperature.clone(), cfg.clone());
                cfm.sample_scaled(mu, cond, noise, scales, timesteps)
            },
        );
        let embeds = self.embed_patches(step.patch.clone().unsqueeze_dim(1));
        let (lm_hidden, residual_hidden) =
            self.advance_lms(embeds, (&mut caches.0, &mut caches.1), |lm, x, cache| {
                lm.forward_at(x, cache, positions.clone())
            });
        let next = |at: &Tensor<1, Int>| at.clone().add_scalar(1).clamp_max(last);
        StepOutputs {
            step,
            lm_hidden,
            residual_hidden,
            positions: next(positions),
            draws: next(draws),
        }
    }

    /// The state of the stable step for `rows` rows in caches of `capacity` positions, holding
    /// nothing, not yet captured.
    fn step_graph(&self, rows: usize, capacity: usize) -> StepGraph {
        let (device, dtype) = (&self.device, self.lm_dtype);
        let hidden = || Tensor::zeros([rows, self.hidden], (device, dtype));
        let scale = || Tensor::zeros([rows, 1, 1], device);
        let index = || Tensor::<1, Int>::zeros([rows], device);
        let starts = (0..rows).map(|row| (row * capacity) as i64).collect();
        let mut caches = (
            self.base_lm.cache(rows, capacity),
            self.residual_lm.cache(rows, capacity),
        );
        for row in 0..rows {
            caches.0.clear(row);
            caches.1.clear(row);
        }
        let state = StepState {
            caches,
            positions: index(),
            lm_hidden: hidden(),
            residual_hidden: hidden(),
            cond: Tensor::zeros([rows, self.patch_size, self.feat_dim], device),
            noise: Tensor::zeros([rows * capacity, self.feat_dim, self.patch_size], device),
            starts: Tensor::from_data(TensorData::new(starts, [rows]), device),
            draws: index(),
            temperature: scale(),
            cfg: scale(),
        };
        StepGraph {
            state: Arc::new(Mutex::new(state)),
            graph: None,
            shares_known: Tensor::<1>::zeros([1], device).can_mut(),
        }
    }
}

impl StepGraph {
    /// Writes `value` over the `rows` of `target`, in its buffer.
    ///
    /// # Panics
    ///
    /// If the device says `target` shares its buffer, which the write would then move.
    fn write<const D: usize, K: Basic>(
        &self,
        target: &mut Tensor<D, K>,
        rows: Range<usize>,
        value: Tensor<D, K>,
    ) {
        assert!(
            !self.shares_known || target.can_mut(),
            "a buffer the step reads is shared, so a write would move it from under the graph"
        );
        let mut at = target.dims().map(|n| 0..n);
        at[0] = rows;
        target.inplace(|target| target.slice_assign(at, value));
    }

    /// Writes `value` over the whole of `target`, as [`write`](Self::write) does.
    fn write_all<const D: usize, K: Basic>(&self, target: &mut Tensor<D, K>, value: Tensor<D, K>) {
        let rows = target.dims()[0];
        self.write(target, 0..rows, value);
    }

    /// The graph of the stable step on this state, captured the first time.
    fn captured(&mut self, model: &Model, timesteps: usize) -> &mut Graph<StepOutputs, StepFn> {
        let state = &self.state;
        self.graph.get_or_insert_with(|| {
            let (device, model, state) = (model.device.clone(), model.clone(), state.clone());
            let step: StepFn = Box::new(move || {
                let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
                model.stable_step(&mut state, timesteps)
            });
            capture(&device, step)
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, StepState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Generations going on together by the stable step, [`Stepping::Stable`] or
/// [`Stepping::Captured`], a row each: each [`step`](Self::step) samples a patch of every one and
/// feeds it to the LMs at once. As a [`Batch`](crate::Batch), each is computed as it would be
/// alone, rows that hold none included; the rows double to make room for one more, and halve as
/// most are left empty, the graph of each number of rows captured of its own.
pub struct Stepper<'a> {
    model: &'a Model,
    stepping: Stepping,
    timesteps: usize,
    /// The graph the rows are in, with its key, once a generation has joined.
    graph: Option<(Key, StepGraph)>,
    rows: Vec<Option<Row>>,
    /// The rows that hold generations, `[generations]`, which the patches of a step are taken
    /// from.
    held: Option<Tensor<1, Int>>,
    next: u64,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    id: RowId,
    /// The positions the caches hold of it, and those they may.
    len: usize,
    end: usize,
}

impl fmt::Debug for Stepper<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stepper")
            .field("stepping", &self.stepping)
            .field("key", &self.graph.as_ref().map(|(key, _)| key))
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl<'a> Generation<'a> {
    /// The generation going on alone as `stepping` says, sampled as `sampling` says from
    /// `noise`, as [`Stepper::join`] takes it.
    ///
    /// # Errors
    ///
    /// As [`Stepper::join`].
    ///
    /// # Panics
    ///
    /// If `stepping` is [`Stepping::Eager`], which needs no stepper.
    pub fn stepper(
        self,
        sampling: &Sampling,
        stepping: Stepping,
        noise: &mut impl Noise,
    ) -> Result<Stepper<'a>, Error> {
        let mut stepper = Stepper::new(self.model, stepping, sampling.timesteps);
        stepper.join(self, sampling, noise)?;
        Ok(stepper)
    }
}

impl<'a> Stepper<'a> {
    /// A stepper of no generations of `model`, each patch sampled in `timesteps` steps.
    ///
    /// # Panics
    ///
    /// If `stepping` is [`Stepping::Eager`], which needs no stepper.
    pub fn new(model: &'a Model, stepping: Stepping, timesteps: usize) -> Self {
        assert_ne!(stepping, Stepping::Eager, "a stepper of the eager step");
        Self {
            model,
            stepping,
            timesteps,
            graph: None,
            rows: vec![],
            held: None,
            next: 0,
        }
    }

    /// The generations, in the order [`step`](Self::step) gives their patches.
    pub fn rows(&self) -> Vec<RowId> {
        self.rows.iter().flatten().map(|row| row.id).collect()
    }

    /// Takes `generation` in, sampled as `sampling` says from `noise`, of which it draws all its
    /// patches' at once: into caches of a capacity of a power of two, those of the graph
    /// captured for its rows and capacity on this thread where there is one, into which its keys
    /// and values are copied.
    ///
    /// # Errors
    ///
    /// If `sampling` takes another number of steps, or the stepper captures graphs and this is
    /// not the thread the model loaded on: cubecl gives each thread a stream of its own, and a
    /// graph reading buffers of another stream is what cubecl-cuda has crashed on.
    ///
    /// # Panics
    ///
    /// If `generation` is of another model.
    pub fn join(
        &mut self,
        generation: Generation<'a>,
        sampling: &Sampling,
        noise: &mut impl Noise,
    ) -> Result<RowId, Error> {
        let model = self.model;
        assert!(
            std::ptr::eq(model, generation.model),
            "a generation of another model"
        );
        if self.stepping == Stepping::Captured && thread::current().id() != model.loaded_on {
            return Err(Error::OtherThread);
        }
        if sampling.timesteps != self.timesteps {
            return Err(Error::Request(format!(
                "patches sampled in {} steps among patches sampled in {}",
                sampling.timesteps, self.timesteps
            )));
        }
        let end = generation.base_cache.capacity();
        let len = generation
            .base_cache
            .len(0)
            .expect("a generation's row holds it");
        let needed = end.next_power_of_two().min(model.max_positions);
        let slot = self.rows.iter().position(Option::is_none);
        let rows = match slot {
            Some(_) => self.rows.len(),
            None => (2 * self.rows.len()).max(1),
        };
        let capacity = self.graph.as_ref().map_or(0, |(key, _)| key.1).max(needed);
        let kept: Vec<usize> = (0..self.rows.len()).collect();
        self.move_to(&kept, rows, capacity);
        let slot = slot.unwrap_or(kept.len());

        let (patches, draw) = (end - len, model.feat_dim * model.patch_size);
        let draws = (0..patches).flat_map(|_| noise.next(draw)).collect();
        let draws = TensorData::new(draws, [patches, model.feat_dim, model.patch_size]);
        let (_, graph) = self.graph.as_ref().expect("moved to one");
        let mut guard = graph.state();
        let state = &mut *guard;
        state.caches.0.insert(slot, &generation.base_cache, 0);
        state.caches.1.insert(slot, &generation.residual_cache, 0);
        let device = &model.device;
        let row = slot..slot + 1;
        graph.write(&mut state.lm_hidden, row.clone(), generation.lm_hidden);
        graph.write(
            &mut state.residual_hidden,
            row.clone(),
            generation.residual_hidden,
        );
        graph.write(&mut state.cond, row.clone(), generation.cond);
        let start = slot * capacity;
        let noise = Tensor::from_data(draws, device);
        graph.write(&mut state.noise, start..start + patches, noise);
        let at = |n: usize| Tensor::from_data(TensorData::from([n as i64]), device);
        graph.write(&mut state.positions, row.clone(), at(len));
        graph.write(&mut state.draws, row.clone(), at(0));
        let scale = |value| Tensor::full([1, 1, 1], value, device);
        graph.write(
            &mut state.temperature,
            row.clone(),
            scale(sampling.temperature),
        );
        graph.write(&mut state.cfg, row, scale(sampling.cfg_value));
        drop(guard);

        let id = RowId(self.next);
        self.next += 1;
        self.rows[slot] = Some(Row { id, len, end });
        self.hold();
        Ok(id)
    }

    /// Drops the generation `row`.
    ///
    /// # Panics
    ///
    /// If it is not in the stepper.
    pub fn leave(&mut self, row: RowId) {
        let slot = self
            .rows
            .iter()
            .position(|held| held.is_some_and(|held| held.id == row))
            .expect("a generation in the stepper");
        self.rows[slot] = None;
        if let Some((_, graph)) = &self.graph {
            let mut state = graph.state();
            state.caches.0.clear(slot);
            state.caches.1.clear(slot);
        }
        let held: Vec<usize> = (0..self.rows.len())
            .filter(|&slot| self.rows[slot].is_some())
            .collect();
        if held.is_empty() {
            if let Some((key, graph)) = self.graph.take() {
                self.model.graphs.give(key, graph);
            }
            self.rows.clear();
        } else if 4 * held.len() <= self.rows.len() {
            let capacity = self.graph.as_ref().expect("rows are held").0.1;
            self.move_to(&held, self.rows.len() / 2, capacity);
        }
        self.hold();
    }

    /// Moves the rows `kept` into the first rows of the graph for `rows` rows of `capacity`
    /// positions, unless they are in it.
    fn move_to(&mut self, kept: &[usize], rows: usize, capacity: usize) {
        let model = self.model;
        let key = (rows, capacity, self.timesteps, thread::current().id());
        let identity = kept.iter().enumerate().all(|(i, &slot)| i == slot);
        if identity && self.graph.as_ref().is_some_and(|(held, _)| *held == key) {
            return;
        }
        let graph = model
            .graphs
            .take(&key)
            .unwrap_or_else(|| model.step_graph(rows, capacity));
        if let Some((old_key, old)) = self.graph.take() {
            {
                let (from, mut to) = (old.state(), graph.state());
                let to = &mut *to;
                for (i, &slot) in kept.iter().enumerate() {
                    if self.rows[slot].is_none() {
                        continue;
                    }
                    to.caches.0.insert(i, &from.caches.0, slot);
                    to.caches.1.insert(i, &from.caches.1, slot);
                    let at = i..i + 1;
                    graph.write(&mut to.lm_hidden, at.clone(), row(&from.lm_hidden, slot));
                    graph.write(
                        &mut to.residual_hidden,
                        at.clone(),
                        row(&from.residual_hidden, slot),
                    );
                    graph.write(&mut to.cond, at.clone(), row(&from.cond, slot));
                    graph.write(
                        &mut to.temperature,
                        at.clone(),
                        row(&from.temperature, slot),
                    );
                    graph.write(&mut to.cfg, at.clone(), row(&from.cfg, slot));
                    graph.write(&mut to.positions, at.clone(), row(&from.positions, slot));
                    graph.write(&mut to.draws, at, row(&from.draws, slot));
                    let noise = from.noise.clone().narrow(0, slot * old_key.1, old_key.1);
                    graph.write(&mut to.noise, i * capacity..i * capacity + old_key.1, noise);
                }
            }
            model.graphs.give(old_key, old);
        }
        let mut moved: Vec<Option<Row>> = kept.iter().map(|&slot| self.rows[slot]).collect();
        moved.resize(rows, None);
        self.rows = moved;
        self.graph = Some((key, graph));
    }

    /// Keeps the rows that hold generations on the device, for the next steps.
    fn hold(&mut self) {
        let held: Vec<i64> = (0..self.rows.len())
            .filter(|&slot| self.rows[slot].is_some())
            .map(|slot| slot as i64)
            .collect();
        self.held = (!held.is_empty()).then(|| {
            let held = TensorData::new(held.clone(), [held.len()]);
            Tensor::from_data(held, &self.model.device)
        });
    }

    /// Whether its steps replay a graph the device recorded of the step, as one launch: not
    /// before the first step has captured it, nor where the device records none.
    pub fn is_recorded(&self) -> bool {
        let graph = self
            .graph
            .as_ref()
            .and_then(|(_, graph)| graph.graph.as_ref());
        graph.is_some_and(Graph::is_hardware)
    }

    /// Samples the next patch of each generation from its noise and feeds it to the LMs.
    ///
    /// # Panics
    ///
    /// If there is none, or one has sampled as many patches as it was started for.
    pub fn step(&mut self) -> Step {
        let (_, graph) = self.graph.as_mut().expect("a generation in the stepper");
        for row in self.rows.iter().flatten() {
            assert!(row.len < row.end, "a patch past the {} positions", row.end);
        }
        let held = self.held.clone().expect("a generation in the stepper");
        let outputs = match self.stepping {
            Stepping::Captured => {
                let graph = graph.captured(self.model, self.timesteps);
                // Safety: the closure owns the state, and with it every buffer it reads and
                // writes; the inputs were written, and the outputs are read, on this thread, the
                // one the graph was captured on, as its key says.
                unsafe { graph.replay() }.clone()
            }
            _ => self.model.stable_step(&mut graph.state(), self.timesteps),
        };
        // Taken out of the buffers the next replay writes.
        let step = Step {
            patch: outputs.step.patch.clone().select(0, held.clone()),
            stop_logits: outputs.step.stop_logits.clone().select(0, held),
        };
        let mut state = graph.state();
        graph.write_all(&mut state.lm_hidden, outputs.lm_hidden);
        graph.write_all(&mut state.residual_hidden, outputs.residual_hidden);
        graph.write_all(&mut state.cond, outputs.step.patch);
        graph.write_all(&mut state.positions, outputs.positions);
        graph.write_all(&mut state.draws, outputs.draws);
        state.caches.0.extend(1);
        state.caches.1.extend(1);
        for row in self.rows.iter_mut().flatten() {
            row.len += 1;
        }
        step
    }
}

/// Row `slot` of `tensor`.
fn row<const D: usize, K: Basic>(tensor: &Tensor<D, K>, slot: usize) -> Tensor<D, K> {
    tensor.clone().narrow(0, slot, 1)
}

impl Drop for Stepper<'_> {
    fn drop(&mut self) {
        if let Some((key, graph)) = self.graph.take() {
            self.model.graphs.give(key, graph);
        }
    }
}
