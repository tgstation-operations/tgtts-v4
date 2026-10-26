use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, PoisonError},
    thread::{self, ThreadId},
};

use burn::tensor::{Device, Graph, Int, Tensor, TensorData, capture, kind::Basic};
use minicpm4::KvCache;

use crate::{Generation, Model, Sampling, Step};

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

/// The capacity of the caches a graph steps, the sampling steps of each patch, and the thread
/// that captured it: a replay must be ordered after the writes of its inputs, which cubecl orders
/// only on the same thread.
type Key = (usize, usize, ThreadId);

impl Graphs {
    fn take(&self, key: &Key) -> Option<StepGraph> {
        let mut graphs = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        graphs.remove(key)
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
/// generation, `rows` of them.
struct StepState {
    caches: (KvCache, KvCache),
    /// The position each row's patch goes to, `[rows]`.
    positions: Tensor<1, Int>,
    /// The LMs' last hidden states `[rows, hidden]` and the last patches `[rows, patch_size,
    /// feat_dim]`, which the next patches are sampled after.
    lm_hidden: Tensor<2>,
    residual_hidden: Tensor<2>,
    cond: Tensor<3>,
    /// The noise of the next patches, `[rows, feat_dim, patch_size]`.
    noise: Tensor<3>,
    /// The temperature and the classifier-free guidance scale of each row, `[rows, 1, 1]`.
    temperature: Tensor<3>,
    cfg: Tensor<3>,
}

/// What the stable step gives: the patch, sampled after the state's, and the LMs' hidden states
/// after it.
type StepOutputs = (Step, Tensor<2>, Tensor<2>);

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
    /// so that running it again does what it did.
    fn stable_step(&self, state: &mut StepState, timesteps: usize) -> StepOutputs {
        let StepState {
            caches,
            positions,
            noise,
            temperature,
            cfg,
            ..
        } = state;
        let step = self.sample_after(
            state.lm_hidden.clone(),
            state.residual_hidden.clone(),
            state.cond.clone(),
            |cfm, mu, cond| {
                let scales = (temperature.clone(), cfg.clone());
                cfm.sample_scaled(mu, cond, noise.clone(), scales, timesteps)
            },
        );
        let embeds = self.embed_patches(step.patch.clone().unsqueeze_dim(1));
        let (lm_hidden, residual_hidden) =
            self.advance_lms(embeds, (&mut caches.0, &mut caches.1), |lm, x, cache| {
                lm.forward_at(x, cache, positions.clone())
            });
        (step, lm_hidden, residual_hidden)
    }

    /// The state of the stable step for one row in caches of `capacity` positions, not yet
    /// captured.
    fn step_graph(&self, capacity: usize) -> StepGraph {
        let (device, dtype) = (&self.device, self.lm_dtype);
        let hidden = || Tensor::zeros([1, self.hidden], (device, dtype));
        let scale = || Tensor::zeros([1, 1, 1], device);
        let state = StepState {
            caches: (
                self.base_lm.cache(1, capacity),
                self.residual_lm.cache(1, capacity),
            ),
            positions: Tensor::from_data(TensorData::from([0i64]), device),
            lm_hidden: hidden(),
            residual_hidden: hidden(),
            cond: Tensor::zeros([1, self.patch_size, self.feat_dim], device),
            noise: Tensor::zeros([1, self.feat_dim, self.patch_size], device),
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

/// A generation going on by the stable step, [`Stepping::Stable`] or [`Stepping::Captured`]:
/// each [`step`](Self::step) samples a patch and feeds it to the LMs at once.
pub struct Stepper<'a> {
    model: &'a Model,
    key: Key,
    graph: Option<StepGraph>,
    captured: bool,
    /// The positions the caches hold.
    len: usize,
    timesteps: usize,
}

impl fmt::Debug for Stepper<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stepper")
            .field("capacity", &self.key.0)
            .field("captured", &self.captured)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<'a> Generation<'a> {
    /// The generation going on as `stepping` says, sampled as `sampling` says: in caches of a
    /// capacity of a power of two, those of the graph captured for it on this thread where there
    /// is one, into which its keys and values are copied.
    ///
    /// # Panics
    ///
    /// If `stepping` is [`Stepping::Eager`], which needs no stepper.
    pub fn stepper(self, sampling: &Sampling, stepping: Stepping) -> Stepper<'a> {
        assert_ne!(stepping, Stepping::Eager, "a stepper of the eager step");
        let model = self.model;
        let needed = self.base_cache.capacity();
        let capacity = needed.next_power_of_two().min(model.max_positions);
        let key = (capacity, sampling.timesteps, thread::current().id());
        let graph = model
            .graphs
            .take(&key)
            .unwrap_or_else(|| model.step_graph(capacity));
        let len = self.base_cache.len(0).expect("a generation's row holds it");
        {
            let mut guard = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
            let state = &mut *guard;
            state.caches.0.insert(0, &self.base_cache, 0);
            state.caches.1.insert(0, &self.residual_cache, 0);
            graph.write(&mut state.lm_hidden, self.lm_hidden);
            graph.write(&mut state.residual_hidden, self.residual_hidden);
            graph.write(&mut state.cond, self.cond);
            let scale = |value| Tensor::full([1, 1, 1], value, &model.device);
            graph.write(&mut state.temperature, scale(sampling.temperature));
            graph.write(&mut state.cfg, scale(sampling.cfg_value));
        }
        Stepper {
            model,
            key,
            graph: Some(graph),
            captured: stepping == Stepping::Captured,
            len,
            timesteps: sampling.timesteps,
        }
    }
}

impl StepGraph {
    /// Writes `value` over `target`, in its buffer.
    ///
    /// # Panics
    ///
    /// If the device says `target` shares its buffer, which the write would then move.
    fn write<const D: usize, K: Basic>(&self, target: &mut Tensor<D, K>, value: Tensor<D, K>) {
        assert!(
            !self.shares_known || target.can_mut(),
            "a buffer the step reads is shared, so a write would move it from under the graph"
        );
        let at = target.dims().map(|n| 0..n);
        target.inplace(|target| target.slice_assign(at, value));
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
}

impl Stepper<'_> {
    /// Whether its steps replay a graph the device recorded of the step, as one launch: not
    /// before the first step has captured it, nor where the device records none.
    pub fn is_recorded(&self) -> bool {
        let graph = self.graph.as_ref().and_then(|graph| graph.graph.as_ref());
        graph.is_some_and(Graph::is_hardware)
    }

    /// Samples the next patch from `noise` `[1, feat_dim, patch_size]` and feeds it to the LMs.
    ///
    /// # Panics
    ///
    /// If the caches are full.
    pub fn step(&mut self, noise: Tensor<3>) -> Step {
        let graph = self.graph.as_mut().expect("held until dropped");
        let capacity = self.key.0;
        assert!(
            self.len < capacity,
            "KV cache is full at {capacity} positions"
        );
        let position = Tensor::from_data(TensorData::from([self.len as i64]), &self.model.device);
        {
            let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
            graph.write(&mut state.noise, noise);
            graph.write(&mut state.positions, position);
        }
        let (step, lm_hidden, residual_hidden) = match self.captured {
            true => {
                let graph = graph.captured(self.model, self.timesteps);
                // Safety: the closure owns the state, and with it every buffer it reads and
                // writes; the inputs were written, and the outputs are read, on this thread, the
                // one the graph was captured on, as its key says.
                let (step, lm_hidden, residual_hidden) = unsafe { graph.replay() };
                // The next replay writes its outputs where this one did.
                let step = Step {
                    patch: copy(&step.patch),
                    stop_logits: copy(&step.stop_logits),
                };
                (step, lm_hidden.clone(), residual_hidden.clone())
            }
            false => {
                let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
                self.model.stable_step(&mut state, self.timesteps)
            }
        };
        let graph = self.graph.as_ref().expect("held until dropped");
        let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
        graph.write(&mut state.lm_hidden, lm_hidden);
        graph.write(&mut state.residual_hidden, residual_hidden);
        graph.write(&mut state.cond, step.patch.clone());
        state.caches.0.extend(1);
        state.caches.1.extend(1);
        self.len += 1;
        step
    }
}

/// A copy of `tensor` in a buffer of its own.
fn copy<const D: usize>(tensor: &Tensor<D>) -> Tensor<D> {
    let at = tensor.dims().map(|n| 0..n);
    let options = (&tensor.device(), tensor.dtype());
    Tensor::empty(tensor.dims(), options).slice_assign(at, tensor.clone())
}

impl Drop for Stepper<'_> {
    fn drop(&mut self) {
        if let Some(graph) = self.graph.take() {
            self.model.graphs.give(self.key, graph);
        }
    }
}
