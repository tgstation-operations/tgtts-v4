use std::{
    collections::HashMap,
    fmt,
    ops::Range,
    ptr,
    sync::{Arc, Mutex, PoisonError},
    thread,
};

use burn::tensor::{Bool, Graph, Int, Tensor, TensorData, capture, kind::Basic};
use minicpm4::KvCache;

use crate::{Error, Generation, Model, Prompt, Stepping, speech::write};

/// The positions of keys the graphs of runs of prompts that join together hold: past them, a
/// prompt runs as it comes.
pub const PREFILL_KEYS: usize = 1024;

/// The graphs a model has captured of runs of prompts that join together, by the rows and
/// positions run they were captured for, with what they read and write: the runs of the same
/// rows share caches of keys and values. Each is kept as long as the model lives.
pub(crate) struct PrefillGraphs {
    /// The rows and positions run of the runs to capture, each a power of two.
    pub(crate) buckets: Vec<(usize, usize)>,
    kept: Mutex<HashMap<(usize, usize), PrefillGraph>>,
    states: Mutex<HashMap<usize, Arc<Mutex<PrefillState>>>>,
    /// Whether the device says if a tensor shares its buffer, which a write to what a graph
    /// reads must not, or it would move where a replay does not look.
    shares_known: bool,
}

impl PrefillGraphs {
    pub(crate) fn new(shares_known: bool) -> Self {
        Self {
            buckets: vec![],
            kept: Mutex::default(),
            states: Mutex::default(),
            shares_known,
        }
    }

    /// How many it has captured.
    pub(crate) fn captured(&self) -> usize {
        self.kept
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// A clone of a model captures graphs of its own.
impl Clone for PrefillGraphs {
    fn clone(&self) -> Self {
        Self {
            buckets: self.buckets.clone(),
            ..Self::new(self.shares_known)
        }
    }
}

impl fmt::Debug for PrefillGraphs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrefillGraphs({:?}, {})", self.buckets, self.captured())
    }
}

struct PrefillGraph {
    state: Arc<Mutex<PrefillState>>,
    graph: Graph<(Tensor<3>, Tensor<3>), PrefillFn>,
}

type PrefillFn = Box<dyn FnMut() -> (Tensor<3>, Tensor<3>) + Send>;

/// The caches the runs of some number of rows go on in, and what each run of them reads, by
/// its positions.
struct PrefillState {
    caches: (KvCache, KvCache),
    inputs: HashMap<usize, PrefillInputs>,
}

#[derive(Clone)]
struct PrefillInputs {
    /// Each row's tokens, whether each is audio and the embeddings of their patches.
    tokens: Tensor<2, Int>,
    audio: Tensor<3, Bool>,
    embeds: Tensor<3>,
    /// The positions each row holds before the run, `[rows]`, and `0..run`.
    starts: Tensor<1, Int>,
    run: Tensor<1, Int>,
}

/// What the model computes over the part of a prompt its voice alone decides, kept for the next
/// prompts in that voice, up to a number of bytes on the device: the voice's patches as the LMs
/// embed them, which the prompt holds twice, and the LMs' keys and values over the voice as
/// reference audio and the text after it.
///
/// The LMs are causal, so their keys and values over a prompt's first positions are those of
/// any prompt that begins with the same positions. The text is the voice's transcript and the
/// text to speak tokenized together, and its last token of transcript can merge with the first
/// of the text: a prompt goes on from as many tokens of the kept text as it shares.
#[derive(Debug)]
pub struct PromptCache {
    budget: usize,
    bytes: usize,
    clock: u64,
    /// Each voice by the bits of its patches.
    voices: HashMap<Vec<u32>, Voice>,
}

#[derive(Debug)]
struct Voice {
    /// The voice's patches as the LMs embed them, `[patches, hidden]`.
    embeds: Tensor<2>,
    prefix: Option<Prefix>,
    /// When it was last used, on the cache's clock.
    used: u64,
}

/// The LMs' keys and values over a prompt up to the end of its text.
#[derive(Debug)]
struct Prefix {
    text: Vec<u32>,
    base: KvCache,
    residual: KvCache,
}

impl Voice {
    fn bytes(&self) -> usize {
        let embeds = self.embeds.shape().num_elements() * self.embeds.dtype().size();
        let prefix = self.prefix.as_ref();
        embeds + prefix.map_or(0, |prefix| prefix.base.bytes() + prefix.residual.bytes())
    }
}

impl PromptCache {
    /// An empty cache that keeps at most `budget` bytes, dropping the voices used longest ago to
    /// make room for others.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            clock: 0,
            voices: HashMap::new(),
        }
    }

    /// The bytes it keeps on the device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The number of voices it keeps.
    pub fn voices(&self) -> usize {
        self.voices.len()
    }

    fn get(&mut self, key: &[u32]) -> Option<&Voice> {
        self.clock += 1;
        let voice = self.voices.get_mut(key)?;
        voice.used = self.clock;
        Some(voice)
    }

    /// Keeps `embeds` for the voice `key`, if they fit.
    fn insert(&mut self, key: Vec<u32>, embeds: Tensor<2>) {
        let voice = Voice {
            embeds,
            prefix: None,
            used: self.clock,
        };
        let bytes = voice.bytes();
        if self.make_room(&key, bytes) {
            self.bytes += bytes;
            self.voices.insert(key, voice);
        }
    }

    /// Keeps `prefix` for the voice `key`, if it is still kept without one and it fits.
    fn keep_prefix(&mut self, key: &[u32], prefix: Prefix) {
        let Some(voice) = self.voices.get(key).filter(|voice| voice.prefix.is_none()) else {
            return;
        };
        let bytes = prefix.base.bytes() + prefix.residual.bytes();
        if bytes + voice.bytes() <= self.budget && self.make_room(key, bytes) {
            let voice = self.voices.get_mut(key).expect("kept");
            voice.prefix = Some(prefix);
            self.bytes += bytes;
        }
    }

    /// Drops the voices other than `key` used longest ago until `bytes` more fit, if they can.
    fn make_room(&mut self, key: &[u32], bytes: usize) -> bool {
        while self.bytes + bytes > self.budget {
            let oldest = self
                .voices
                .iter()
                .filter(|(other, _)| other.as_slice() != key)
                .min_by_key(|(_, voice)| voice.used)
                .map(|(other, _)| other.clone());
            let Some(oldest) = oldest else {
                return false;
            };
            let voice = self.voices.remove(&oldest).expect("found");
            self.bytes -= voice.bytes();
        }
        true
    }
}

/// A prompt run through the LMs a run of positions at a time, as [`Model::start`] runs it
/// whole, so that a batch need not wait for all of it between two patches; with what a
/// [`PromptCache`] keeps of its voice, from where that leaves off.
#[derive(Debug)]
pub struct Prefill<'a> {
    model: &'a Model,
    prompt: Prompt,
    tokens: Tensor<2, Int>,
    audio: Tensor<3, Bool>,
    /// The embeddings of the voice's patches where the prompt has them, `[1, len, hidden]`.
    embeds: Tensor<3>,
    /// The positions its generation may hold.
    capacity: usize,
    /// The LMs' keys and values, once a run has made them, and before, those the cache kept of
    /// the prompt's first positions, which it shares.
    caches: Option<(KvCache, KvCache)>,
    kept: Option<(KvCache, KvCache)>,
    /// The positions run so far, or kept.
    done: usize,
    /// The voice's key in the cache, while the LMs' keys and values over the text are still to
    /// be kept there.
    keep: Option<Vec<u32>>,
    text: Range<usize>,
    /// The outputs of the LMs over the last run.
    outputs: Option<(Tensor<3>, Tensor<3>)>,
}

impl Model {
    /// Starts running `prompt` through the LMs as [`start`](Self::start) does, a run of positions
    /// at a time from where what `cache` keeps of its voice leaves off, keeping there what it
    /// computes that the next prompts in the voice can use.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start), and if the prompt is not as
    /// [`Prompt::cloning`](crate::Prompt::cloning) lays it out.
    pub fn prefill(
        &self,
        prompt: &Prompt,
        max_patches: usize,
        cache: &mut PromptCache,
    ) -> Result<Prefill<'_>, Error> {
        let capacity = self.capacity(prompt, max_patches)?;
        let Some((reference, voice)) = prompt.voice() else {
            return Err(Error::Request(
                "a prompt not laid out as for a voice".into(),
            ));
        };
        let key: Vec<u32> = voice.iter().map(|value| value.to_bits()).collect();
        let embeds = match cache.get(&key) {
            Some(voice) => voice.embeds.clone(),
            None => {
                let embeds = self.embed_voice(voice, reference.len());
                cache.insert(key.clone(), embeds.clone());
                embeds
            }
        };

        let (device, len, hidden) = (&self.device, prompt.len(), self.hidden);
        let continuation = len - reference.len()..len;
        let dtype = embeds.dtype();
        let embeds = embeds.unsqueeze_dim::<3>(0);
        let at = |range: Range<usize>| [0..1, range, 0..hidden];
        let embeds = Tensor::zeros([1, len, hidden], (device, dtype))
            .slice_assign(at(reference), embeds.clone())
            .slice_assign(at(continuation), embeds);
        let tokens = prompt
            .tokens
            .iter()
            .map(|&token| i64::from(token))
            .collect();
        let tokens = Tensor::from_data(TensorData::new(tokens, [1, len]), device);
        let audio = TensorData::new(prompt.audio.clone(), [1, len, 1]);

        let text = prompt.text();
        let (mut done, mut keep, mut kept) = (0, None, None);
        match cache.get(&key).map(|voice| &voice.prefix) {
            Some(Some(prefix)) => {
                let tokens = &prompt.tokens[text.clone()];
                let shared = prefix.text.iter().zip(tokens);
                done = text.start + shared.take_while(|(a, b)| a == b).count();
                kept = Some((prefix.base.clone(), prefix.residual.clone()));
            }
            Some(None) => keep = Some(key),
            None => {}
        }
        Ok(Prefill {
            model: self,
            prompt: prompt.clone(),
            tokens,
            audio: Tensor::from_data(audio, device),
            embeds,
            capacity,
            caches: None,
            kept,
            done,
            keep,
            text,
            outputs: None,
        })
    }

    /// The generations that go on from the prompts of `prefills`, each run whole through the LMs
    /// from where it was left, all of them in one run as rows of a batch: each prompt's positions
    /// are followed by as many as the longest has more, which no position before them attends
    /// to. What the LMs computed over a text is kept in `cache` as [`Prefill::run`] keeps it.
    ///
    /// The batch's keys and values are held for as many positions as the first power of two
    /// that holds them, and each generation shares its row of them, to join a batch from.
    ///
    /// # Panics
    ///
    /// If `prefills` is empty, or one is of another model.
    pub(crate) fn prefill_together<'a>(
        &'a self,
        prefills: Vec<Prefill<'a>>,
        cache: &mut PromptCache,
    ) -> Vec<Generation<'a>> {
        assert!(!prefills.is_empty(), "no prompts to run");
        for prefill in &prefills {
            assert!(ptr::eq(self, prefill.model), "a prompt of another model");
        }
        let prefills = match self.prefill_replayed(prefills, cache) {
            Ok(generations) => return generations,
            Err(prefills) => prefills,
        };
        let rows = prefills.len();
        let longest = prefills.iter().map(Prefill::left).max().expect("prompts");
        let held = prefills
            .iter()
            .map(|prefill| prefill.done)
            .max()
            .expect("prompts");
        let positions = (held + longest).next_power_of_two();
        let mut caches = (
            self.base_lm.cache(rows, positions),
            self.residual_lm.cache(rows, positions),
        );
        for (row, prefill) in prefills.iter().enumerate() {
            if let Some((base, residual)) = prefill.caches.as_ref().or(prefill.kept.as_ref())
                && prefill.done > 0
            {
                caches.0.insert(row, base, 0);
                caches.1.insert(row, residual, 0);
                caches.0.truncate(row, prefill.done);
                caches.1.truncate(row, prefill.done);
            }
        }
        let (tokens, audio, embeds) = self.run_inputs(&prefills, longest);
        let outputs = self.run_prompt(tokens, audio, embeds, (&mut caches.0, &mut caches.1));
        self.finish_together(prefills, &mut caches, &outputs, cache)
    }

    /// What [`prefill_together`](Self::prefill_together) runs for `prefills`: their tokens,
    /// whether each is audio, and the embeddings of their patches, each followed by as many
    /// positions as the `longest` has more.
    fn run_inputs(
        &self,
        prefills: &[Prefill<'_>],
        longest: usize,
    ) -> (Tensor<2, Int>, Tensor<3, Bool>, Tensor<3>) {
        let (device, hidden) = (&self.device, self.hidden);
        let (mut tokens, mut audio, mut embeds) = (vec![], vec![], vec![]);
        for prefill in prefills {
            let (start, left) = (prefill.done, prefill.left());
            let pad = longest - left;
            let options = (device, prefill.embeds.dtype());
            tokens.push(padded(
                pad,
                prefill.tokens.clone().narrow(1, start, left),
                Tensor::zeros([1, pad], device),
            ));
            audio.push(padded(
                pad,
                prefill.audio.clone().narrow(1, start, left),
                Tensor::full([1, pad, 1], false, device),
            ));
            embeds.push(padded(
                pad,
                prefill.embeds.clone().narrow(1, start, left),
                Tensor::zeros([1, pad, hidden], options),
            ));
        }
        (
            Tensor::cat(tokens, 0),
            Tensor::cat(audio, 0),
            Tensor::cat(embeds, 0),
        )
    }

    /// The generations of `prefills`, run in the rows of `caches` and giving `outputs`, each
    /// sharing its row of the caches, what the LMs computed over a text kept in `cache`.
    fn finish_together<'a>(
        &'a self,
        prefills: Vec<Prefill<'a>>,
        caches: &mut (KvCache, KvCache),
        outputs: &(Tensor<3>, Tensor<3>),
        cache: &mut PromptCache,
    ) -> Vec<Generation<'a>> {
        prefills
            .into_iter()
            .enumerate()
            .map(|(row, mut prefill)| {
                let len = prefill.prompt.len();
                caches.0.hold(row, len);
                caches.1.hold(row, len);
                if let Some(key) = prefill.keep.take() {
                    let text = prefill.text.clone();
                    let prefix = Prefix {
                        text: prefill.prompt.tokens[text.clone()].to_vec(),
                        base: caches.0.prefix(row, text.end),
                        residual: caches.1.prefix(row, text.end),
                    };
                    cache.keep_prefix(&key, prefix);
                }
                let own = (caches.0.row(row), caches.1.row(row));
                let left = prefill.left();
                let row_of = |t: &Tensor<3>| t.clone().narrow(0, row, 1).narrow(1, 0, left);
                let outputs = (row_of(&outputs.0), row_of(&outputs.1));
                let mut generation = self.generation(own, outputs, &prefill.prompt);
                generation.capacity = prefill.capacity;
                generation
            })
            .collect()
    }

    /// The generations of `prefills` run by replaying the graph captured of their rows and
    /// positions run, if it was, over caches whose rows they first write what was kept of their
    /// prompts to; or `prefills` again.
    fn prefill_replayed<'a>(
        &'a self,
        prefills: Vec<Prefill<'a>>,
        cache: &mut PromptCache,
    ) -> Result<Vec<Generation<'a>>, Vec<Prefill<'a>>> {
        let graphs = &self.prefill_graphs;
        let keys = self.graph_capacity(PREFILL_KEYS, 0);
        let longest = prefills.iter().map(Prefill::left).max().expect("prompts");
        let (rows, run) = (
            prefills.len().next_power_of_two(),
            longest.next_power_of_two(),
        );
        let mut kept = graphs.kept.lock().unwrap_or_else(PoisonError::into_inner);
        let fits = prefills.iter().all(|prefill| prefill.prompt.len() <= keys);
        let Some(graph) = kept.get_mut(&(rows, run)).filter(|_| fits) else {
            return Err(prefills);
        };
        {
            let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
            let PrefillState { caches, inputs } = &mut *state;
            assert!(
                !graphs.shares_known || caches.0.can_mut() && caches.1.can_mut(),
                "the keys and values a graph runs over are shared, so a write would move them"
            );
            for (row, prefill) in prefills.iter().enumerate() {
                match prefill.caches.as_ref().or(prefill.kept.as_ref()) {
                    Some((base, residual)) if prefill.done > 0 => {
                        caches.0.write_prefix(row, base, 0, prefill.done);
                        caches.1.write_prefix(row, residual, 0, prefill.done);
                    }
                    _ => {
                        caches.0.hold(row, 0);
                        caches.1.hold(row, 0);
                    }
                }
            }
            let (tokens, audio, embeds) = self.run_inputs(&prefills, run);
            let inputs = inputs.get_mut(&run).expect("a graph's inputs");
            let (known, n) = (graphs.shares_known, prefills.len());
            write(known, &mut inputs.tokens, 0..n, tokens);
            write(known, &mut inputs.audio, 0..n, audio);
            write(known, &mut inputs.embeds, 0..n, embeds);
            let mut starts: Vec<i64> = prefills.iter().map(|p| p.done as i64).collect();
            starts.resize(rows, 0);
            let starts = Tensor::from_data(TensorData::new(starts, [rows]), &self.device);
            write(known, &mut inputs.starts, 0..rows, starts);
        }
        // Safety: the closure owns the state, and with it every buffer it reads and writes; the
        // inputs were written, and the outputs are read, on this thread, the one the model
        // loaded on and the graph was captured on.
        let outputs = unsafe { graph.graph.replay() }.clone();
        let mut state = graph.state.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(self.finish_together(prefills, &mut state.caches, &outputs, cache))
    }

    /// Captures a graph of the runs of prompts that join together of each rows and positions run
    /// [`set_prefill_graphs`](Self::set_prefill_graphs) gave, where the model replays graphs of
    /// its step, on this thread, the one it loaded on: over caches of [`PREFILL_KEYS`] positions,
    /// or as many as the LMs have, that the runs of the same rows share. Each is kept as long as
    /// the model lives, and a run of the shape of none is run as it comes.
    pub fn warm_prefill_graphs(&self) {
        if self.stepping() != Stepping::Captured || thread::current().id() != self.loaded_on {
            return;
        }
        let graphs = &self.prefill_graphs;
        let keys = self.graph_capacity(PREFILL_KEYS, 0);
        let (device, hidden) = (&self.device, self.hidden);
        for &(rows, run) in graphs.buckets.iter().filter(|&&(_, run)| run <= keys) {
            let mut kept = graphs.kept.lock().unwrap_or_else(PoisonError::into_inner);
            if kept.contains_key(&(rows, run)) {
                continue;
            }
            let mut states = graphs.states.lock().unwrap_or_else(PoisonError::into_inner);
            let state = states.entry(rows).or_insert_with(|| {
                let caches = (
                    self.base_lm.cache(rows, keys),
                    self.residual_lm.cache(rows, keys),
                );
                Arc::new(Mutex::new(PrefillState {
                    caches,
                    inputs: HashMap::new(),
                }))
            });
            let inputs = PrefillInputs {
                tokens: Tensor::zeros([rows, run], device),
                audio: Tensor::full([rows, run, 1], true, device),
                embeds: Tensor::zeros([rows, run, hidden], (device, self.lm_dtype)),
                starts: Tensor::zeros([rows], device),
                run: Tensor::arange(0..run as i64, device),
            };
            let mut held = state.lock().unwrap_or_else(PoisonError::into_inner);
            held.inputs.insert(run, inputs);
            drop(held);
            let (model, state) = (self.clone(), state.clone());
            let shared = state.clone();
            let step: PrefillFn = Box::new(move || {
                let mut state = shared.lock().unwrap_or_else(PoisonError::into_inner);
                let PrefillState { caches, inputs } = &mut *state;
                let inputs = inputs[&run].clone();
                let starts = inputs.starts.reshape([rows, 1]).expand([rows, run]);
                let positions = starts + inputs.run.reshape([1, run]).expand([rows, run]);
                let run = (inputs.tokens, inputs.audio, inputs.embeds);
                model.run_prompt_at(run, (&mut caches.0, &mut caches.1), positions)
            });
            let graph = capture(device, step);
            kept.insert((rows, run), PrefillGraph { state, graph });
        }
    }

    /// Runs the LMs as [`Scheduler`](crate::Scheduler) runs the prompts of requests that join
    /// together, over `rows` rows of `len` positions after `held` each: on prompts of nothing in
    /// particular, for autotune to choose the kernels of that shape and the pools to hold what it
    /// takes before a request waits for either. The held positions are counted rather than run,
    /// as a request's come from the prompt cache.
    pub fn warm_prefill(&self, rows: usize, held: usize, len: usize) {
        let (device, hidden) = (&self.device, self.hidden);
        let mut caches = (
            self.base_lm.cache(rows, (held + len).next_power_of_two()),
            self.residual_lm
                .cache(rows, (held + len).next_power_of_two()),
        );
        caches.0.extend(held);
        caches.1.extend(held);
        self.run_prompt(
            Tensor::zeros([rows, len], device),
            Tensor::full([rows, len, 1], true, device),
            Tensor::zeros([rows, len, hidden], (device, self.lm_dtype)),
            (&mut caches.0, &mut caches.1),
        );
    }

    /// Embeds a voice of `patches` patches of nothing in particular, as a prompt in a voice the
    /// prompt cache does not hold is embedded, to warm it up as
    /// [`warm_prefill`](Self::warm_prefill) does.
    pub fn warm_voice(&self, patches: usize) {
        let voice = vec![0.0; patches * self.patch_size * self.feat_dim];
        self.embed_voice(&voice, patches);
    }

    /// The LMs' embeddings `[patches, hidden]` of the voice's `patches` patches.
    fn embed_voice(&self, voice: &[f32], patches: usize) -> Tensor<2> {
        let shape = [1, patches, self.patch_size, self.feat_dim];
        let voice = Tensor::from_data(TensorData::new(voice.to_vec(), shape), &self.device);
        self.embed_patches(voice).reshape([patches, self.hidden])
    }
}

impl<'a> Prefill<'a> {
    /// The positions still to run.
    pub fn left(&self) -> usize {
        self.prompt.len() - self.done
    }

    /// Runs up to `positions` more positions, at least one, and gives whether every position
    /// has run. What the LMs computed over the text is kept in `cache` once they are past it.
    pub fn run(&mut self, positions: usize, cache: &mut PromptCache) -> bool {
        let len = self.prompt.len();
        let end = self.done.saturating_add(positions.max(1)).min(len);
        let caches = self.caches.get_or_insert_with(|| {
            let model = self.model;
            let mut caches = (
                model.base_lm.cache(1, self.capacity),
                model.residual_lm.cache(1, self.capacity),
            );
            if let Some((base, residual)) = self.kept.take() {
                for (cache, kept) in [(&mut caches.0, &base), (&mut caches.1, &residual)] {
                    cache.insert(0, kept, 0);
                    cache.truncate(0, self.done);
                }
            }
            caches
        });
        while self.done < end {
            let stop = match &self.keep {
                Some(_) if self.done < self.text.end => end.min(self.text.end),
                _ => end,
            };
            let (start, run) = (self.done, stop - self.done);
            let outputs = self.model.run_prompt(
                self.tokens.clone().narrow(1, start, run),
                self.audio.clone().narrow(1, start, run),
                self.embeds.clone().narrow(1, start, run),
                (&mut caches.0, &mut caches.1),
            );
            self.outputs = Some(outputs);
            self.done = stop;
            if self.done == self.text.end
                && let Some(key) = self.keep.take()
            {
                let prefix = Prefix {
                    text: self.prompt.tokens[self.text.clone()].to_vec(),
                    base: caches.0.prefix(0, self.done),
                    residual: caches.1.prefix(0, self.done),
                };
                cache.keep_prefix(&key, prefix);
            }
        }
        self.done == len
    }

    /// The generation that goes on from the prompt.
    ///
    /// # Panics
    ///
    /// If positions are still to run.
    pub fn finish(self) -> Generation<'a> {
        assert_eq!(self.left(), 0, "positions of the prompt still to run");
        let outputs = self.outputs.expect("a prompt of at least one position");
        let caches = self.caches.expect("run as it was finished");
        self.model.generation(caches, outputs, &self.prompt)
    }
}

/// `run` followed along its positions by `padding`, which is `pad` positions long.
fn padded<const D: usize, K: Basic>(
    pad: usize,
    run: Tensor<D, K>,
    padding: Tensor<D, K>,
) -> Tensor<D, K> {
    match pad {
        0 => run,
        _ => Tensor::cat(vec![run, padding], 1),
    }
}
