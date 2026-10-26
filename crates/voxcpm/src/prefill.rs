use std::{collections::HashMap, ops::Range};

use burn::tensor::{Bool, Int, Tensor, TensorData};
use minicpm4::KvCache;

use crate::{Error, Generation, Model, Prompt};

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
    caches: (KvCache, KvCache),
    /// The positions run so far.
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

        let mut caches = (
            self.base_lm.cache(1, capacity),
            self.residual_lm.cache(1, capacity),
        );
        let text = prompt.text();
        let (mut done, mut keep) = (0, None);
        match cache.get(&key).map(|voice| &voice.prefix) {
            Some(Some(prefix)) => {
                let tokens = &prompt.tokens[text.clone()];
                let shared = prefix.text.iter().zip(tokens);
                done = text.start + shared.take_while(|(a, b)| a == b).count();
                for (cache, kept) in [
                    (&mut caches.0, &prefix.base),
                    (&mut caches.1, &prefix.residual),
                ] {
                    cache.insert(0, kept, 0);
                    cache.truncate(0, done);
                }
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
            caches,
            done,
            keep,
            text,
            outputs: None,
        })
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
                (&mut self.caches.0, &mut self.caches.1),
            );
            self.outputs = Some(outputs);
            self.done = stop;
            if self.done == self.text.end
                && let Some(key) = self.keep.take()
            {
                let prefix = Prefix {
                    text: self.prompt.tokens[self.text.clone()].to_vec(),
                    base: self.caches.0.prefix(0, self.done),
                    residual: self.caches.1.prefix(0, self.done),
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
        self.model.generation(self.caches, outputs, &self.prompt)
    }
}
