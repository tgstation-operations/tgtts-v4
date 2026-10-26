use std::{
    collections::{HashMap, VecDeque},
    ptr,
};

use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};
use minicpm4::KvCache;

use crate::{Error, Generation, Model, Noise, Prefill, Prompt, PromptCache, Sampling, Step};

/// Generations sampled together, a patch of each at a time: a generation joins once its prompt
/// has run through the LMs and leaves when it is done, while the others go on.
///
/// Each is computed as it would be alone. The LMs run each at its own position, every row of
/// their caches at once, rows that hold no generation included: on a GPU a step costs about as
/// much for 16 rows as for one. The local encoder and the DiT run over the generations alone.
#[derive(Debug)]
pub struct Batch<'a> {
    model: &'a Model,
    caches: (KvCache, KvCache),
    /// The generation each row of the caches holds and how it is sampled, if any.
    rows: Vec<Option<(RowId, Sampling)>>,
    /// The LMs' last hidden states and the last patch of each row.
    lm_hidden: Tensor<2>,
    residual_hidden: Tensor<2>,
    cond: Tensor<3>,
    next: u64,
}

/// A generation in a [`Batch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowId(u64);

impl<'a> Batch<'a> {
    /// A batch of no generations of `model`.
    pub fn new(model: &'a Model) -> Self {
        let mut caches = (model.base_lm.cache(1, 1), model.residual_lm.cache(1, 1));
        caches.0.clear(0);
        caches.1.clear(0);
        let (hidden, dtype) = (model.hidden, model.lm_dtype);
        Self {
            model,
            caches,
            rows: vec![None],
            lm_hidden: Tensor::zeros([1, hidden], (&model.device, dtype)),
            residual_hidden: Tensor::zeros([1, hidden], (&model.device, dtype)),
            cond: Tensor::zeros([1, model.patch_size, model.feat_dim], &model.device),
            next: 0,
        }
    }

    /// The generations, in the order [`sample`](Self::sample) and [`advance`](Self::advance)
    /// take them.
    pub fn rows(&self) -> Vec<RowId> {
        self.rows.iter().flatten().map(|&(id, _)| id).collect()
    }

    /// Takes `generation` in, to be sampled as `sampling` says.
    ///
    /// # Errors
    ///
    /// If the generations in the batch are sampled in another number of steps.
    ///
    /// # Panics
    ///
    /// If `generation` is of another model.
    pub fn join(
        &mut self,
        generation: Generation<'a>,
        sampling: &Sampling,
    ) -> Result<RowId, Error> {
        assert!(
            ptr::eq(self.model, generation.model),
            "a generation of another model"
        );
        let mut timesteps = self.rows.iter().flatten().map(|(_, row)| row.timesteps);
        if let Some(timesteps) = timesteps.find(|&t| t != sampling.timesteps) {
            return Err(Error::Request(format!(
                "patches sampled in {} steps among patches sampled in {timesteps}",
                sampling.timesteps
            )));
        }
        let slot = match self.rows.iter().position(Option::is_none) {
            Some(slot) => slot,
            None => {
                let slot = self.rows.len();
                let capacity = self.caches.0.capacity();
                self.reshape(&(0..slot).collect::<Vec<_>>(), 2 * slot, capacity);
                slot
            }
        };
        let capacity = generation.base_cache.capacity();
        if capacity > self.caches.0.capacity() {
            let rows: Vec<usize> = (0..self.rows.len()).collect();
            self.caches.0.reshape(&rows, rows.len(), capacity);
            self.caches.1.reshape(&rows, rows.len(), capacity);
        }
        self.caches.0.insert(slot, &generation.base_cache, 0);
        self.caches.1.insert(slot, &generation.residual_cache, 0);
        let hidden = self.model.hidden;
        let row = |all: &mut Tensor<2>, one: Tensor<2>| {
            all.inplace(|all| all.slice_assign([slot..slot + 1, 0..hidden], one));
        };
        row(&mut self.lm_hidden, generation.lm_hidden);
        row(&mut self.residual_hidden, generation.residual_hidden);
        let [_, patch_size, feat_dim] = generation.cond.dims();
        let at = [slot..slot + 1, 0..patch_size, 0..feat_dim];
        self.cond
            .inplace(|cond| cond.slice_assign(at, generation.cond));
        let id = RowId(self.next);
        self.next += 1;
        self.rows[slot] = Some((id, sampling.clone()));
        Ok(id)
    }

    /// Drops the generation `row`.
    ///
    /// # Panics
    ///
    /// If it is not in the batch.
    pub fn leave(&mut self, row: RowId) {
        let slot = self
            .rows
            .iter()
            .position(|held| held.as_ref().is_some_and(|(id, _)| *id == row))
            .expect("a generation in the batch");
        self.rows[slot] = None;
        self.caches.0.clear(slot);
        self.caches.1.clear(slot);
        // Fewer rows to run once most are empty, halving as they doubled to make room, and none
        // of their keys and values kept once all are.
        let held: Vec<usize> = (0..self.rows.len())
            .filter(|&slot| self.rows[slot].is_some())
            .collect();
        if held.is_empty() {
            self.reshape(&[], 1, 1);
        } else if 4 * held.len() <= self.rows.len() {
            let capacity = self.caches.0.capacity();
            self.reshape(&held, self.rows.len() / 2, capacity);
        }
    }

    /// Samples the next patch of each generation from `noise` `[generations, feat_dim,
    /// patch_size]`.
    pub fn sample(&self, noise: Tensor<3>) -> Step {
        let held = self.held();
        let sampling: Vec<Sampling> = self
            .rows
            .iter()
            .flatten()
            .map(|(_, sampling)| sampling.clone())
            .collect();
        self.model.sample_after(
            held_rows(&self.lm_hidden, held.as_ref()),
            held_rows(&self.residual_hidden, held.as_ref()),
            held_rows(&self.cond, held.as_ref()),
            |cfm, mu, cond| cfm.sample_rows(mu, cond, noise, &sampling),
        )
    }

    /// Feeds each generation its patch of `patches` `[generations, patch_size, feat_dim]` as
    /// its next position.
    pub fn advance(&mut self, patches: Tensor<3>) {
        let model = self.model;
        let embeds = model.embed_patches(patches.clone().unsqueeze_dim(1));
        let (embeds, cond) = match self.held() {
            Some(held) => {
                let [_, _, hidden] = embeds.dims();
                let options = (&embeds.device(), embeds.dtype());
                let all = Tensor::zeros([self.rows.len(), 1, hidden], options);
                let assign = IndexingUpdateOp::Assign;
                (
                    all.select_assign(0, held.clone(), embeds, assign),
                    self.cond.clone().select_assign(0, held, patches, assign),
                )
            }
            None => (embeds, patches),
        };
        let caches = (&mut self.caches.0, &mut self.caches.1);
        (self.lm_hidden, self.residual_hidden) = model.advance_lms(embeds, caches);
        self.cond = cond;
    }

    /// The rows that hold generations, unless all do.
    fn held(&self) -> Option<Tensor<1, Int>> {
        let held: Vec<i64> = (0..self.rows.len())
            .filter(|&slot| self.rows[slot].is_some())
            .map(|slot| slot as i64)
            .collect();
        (held.len() < self.rows.len())
            .then(|| Tensor::from_data(held.as_slice(), &self.model.device))
    }

    /// Gives the batch `rows` rows of `capacity` positions, the first of them holding what the
    /// rows `keep` held.
    fn reshape(&mut self, keep: &[usize], rows: usize, capacity: usize) {
        self.caches.0.reshape(keep, rows, capacity);
        self.caches.1.reshape(keep, rows, capacity);
        let indices: Vec<i64> = keep.iter().map(|&slot| slot as i64).collect();
        let indices = (!keep.is_empty())
            .then(|| Tensor::<1, Int>::from_data(indices.as_slice(), &self.model.device));
        fn kept<const D: usize>(
            all: &mut Tensor<D>,
            indices: Option<&Tensor<1, Int>>,
            rows: usize,
        ) {
            let mut shape = all.dims();
            shape[0] = rows;
            let fresh = Tensor::zeros(shape, (&all.device(), all.dtype()));
            *all = match indices {
                Some(indices) => {
                    let mut at = shape.map(|n| 0..n);
                    at[0] = 0..indices.dims()[0];
                    fresh.slice_assign(at, all.clone().select(0, indices.clone()))
                }
                None => fresh,
            };
        }
        kept(&mut self.lm_hidden, indices.as_ref(), rows);
        kept(&mut self.residual_hidden, indices.as_ref(), rows);
        kept(&mut self.cond, indices.as_ref(), rows);
        let mut held: Vec<_> = keep.iter().map(|&slot| self.rows[slot].take()).collect();
        held.resize(rows, None);
        self.rows = held;
    }
}

/// The rows of `all` that `held` names, or all of them.
fn held_rows<const D: usize>(all: &Tensor<D>, held: Option<&Tensor<1, Int>>) -> Tensor<D> {
    match held {
        Some(held) => all.clone().select(0, held.clone()),
        None => all.clone(),
    }
}

/// Requests generated together in a [`Batch`], each until its stop head says to stop or it has
/// its most patches.
///
/// A request's prompt runs through the LMs from where what a [`PromptCache`] keeps of its voice
/// leaves off, then the request joins the batch. While other requests run, at most a set
/// number of positions of prompts run between two of their patches, which bounds how much
/// longer a new request makes their next patch take.
#[derive(Debug)]
pub struct Scheduler<'a, K, N> {
    model: &'a Model,
    batch: Batch<'a>,
    cache: PromptCache,
    waiting: VecDeque<(Request<K, N>, Prefill<'a>)>,
    running: HashMap<RowId, Request<K, N>>,
    prefill_positions: usize,
}

#[derive(Debug)]
struct Request<K, N> {
    key: K,
    sampling: Sampling,
    /// The patches it may still have.
    left: usize,
    noise: N,
}

/// A patch of a request's speech, `[1, patch_size, feat_dim]` in f32, and whether it is its
/// last.
#[derive(Debug)]
pub struct Patch<K> {
    pub key: K,
    pub patch: Tensor<3>,
    pub last: bool,
}

impl<'a, K: Clone, N: Noise> Scheduler<'a, K, N> {
    /// Generates from `model`, keeping what prompts compute in `cache`, and running at most
    /// `prefill_positions` positions of prompts between two patches of the requests that run.
    pub fn new(model: &'a Model, cache: PromptCache, prefill_positions: usize) -> Self {
        Self {
            model,
            batch: Batch::new(model),
            cache,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            prefill_positions,
        }
    }

    /// Adds a request, `key`, for up to `max_patches` patches after `prompt`, sampled as
    /// `sampling` says from `noise`.
    ///
    /// # Errors
    ///
    /// As [`Model::prefill`], and if the other requests' patches are sampled in another number
    /// of steps.
    pub fn add(
        &mut self,
        key: K,
        prompt: &Prompt,
        sampling: &Sampling,
        max_patches: usize,
        noise: N,
    ) -> Result<(), Error> {
        let others = self.waiting.iter().map(|(request, _)| request);
        let other = others.chain(self.running.values()).next();
        if let Some(other) = other.filter(|other| other.sampling.timesteps != sampling.timesteps) {
            return Err(Error::Request(format!(
                "patches sampled in {} steps among patches sampled in {}",
                sampling.timesteps, other.sampling.timesteps
            )));
        }
        let prefill = self.model.prefill(prompt, max_patches, &mut self.cache)?;
        let request = Request {
            key,
            sampling: sampling.clone(),
            left: max_patches,
            noise,
        };
        self.waiting.push_back((request, prefill));
        Ok(())
    }

    /// The requests waiting for their prompts to run and those generating.
    pub fn requests(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// What it keeps of the voices of the prompts.
    pub fn cache(&self) -> &PromptCache {
        &self.cache
    }

    /// Runs the prompts of waiting requests, all of the first if no request is generating to
    /// wait for it, then samples the next patch of every generating request.
    pub fn step(&mut self) -> Vec<Patch<K>> {
        let mut budget = self.prefill_positions;
        while let Some((_, prefill)) = self.waiting.front_mut() {
            let unbounded = self.running.is_empty();
            if budget == 0 && !unbounded {
                break;
            }
            let positions = if unbounded { usize::MAX } else { budget };
            budget -= prefill.left().min(budget);
            if !prefill.run(positions, &mut self.cache) {
                break;
            }
            let (request, prefill) = self.waiting.pop_front().expect("the front");
            let row = self
                .batch
                .join(prefill.finish(), &request.sampling)
                .expect("checked when added");
            self.running.insert(row, request);
        }

        let rows = self.batch.rows();
        if rows.is_empty() {
            return vec![];
        }
        let (model, len) = (self.model, self.model.feat_dim * self.model.patch_size);
        let noise = rows.iter().flat_map(|row| {
            let request = self.running.get_mut(row).expect("running");
            request.noise.next(len)
        });
        let shape = [rows.len(), model.feat_dim, model.patch_size];
        let noise = Tensor::from_data(TensorData::new(noise.collect(), shape), &model.device);
        let step = self.batch.sample(noise);
        let stops = step.stop_logits.argmax(1).into_data();
        let mut patches = vec![];
        let mut going = vec![];
        for (i, (row, stop)) in rows.iter().zip(stops.iter::<i64>()).enumerate() {
            let request = self.running.get_mut(row).expect("running");
            request.left -= 1;
            let last = stop == 1 || request.left == 0;
            let patch = step.patch.clone().narrow(0, i, 1);
            patches.push(Patch {
                key: request.key.clone(),
                patch,
                last,
            });
            if last {
                self.running.remove(row);
                self.batch.leave(*row);
            } else {
                going.push(i as i64);
            }
        }
        if !going.is_empty() {
            let next = if going.len() == rows.len() {
                step.patch
            } else {
                let going = Tensor::<1, Int>::from_data(going.as_slice(), &model.device);
                step.patch.select(0, going)
            };
            self.batch.advance(next);
        }
        patches
    }
}
