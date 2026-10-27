use std::{
    collections::HashMap,
    fmt, fs, mem,
    num::NonZeroUsize,
    ops::Range,
    path::Path,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use audiovae::{DecodeSlots, DecodeStream, Decoder, Weights};
use burn::tensor::{
    Device, Distribution, FloatDType, Graph, Int, Tensor, TensorData, capture, kind::Basic,
};
use voices::{LATENT_DIM, Latent};

use crate::{Config, Dtypes, Error, Model, Noise, Prompt, Sampling, Stepping, Tokenizer};

/// The voice's last frames that nanovllm-voxcpm's streaming decoder is primed with, its
/// `n_decode_pad_frames`, so that speech starts as if it went on from the voice.
const DECODE_CONTEXT_FRAMES: usize = 12;

/// What a request asks for beyond its text and voice. The default is what the Python service
/// asked for, decoded a second at a time.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub sampling: Sampling,
    /// The most patches to generate, nanovllm-voxcpm's `max_generate_length`.
    pub max_patches: usize,
    /// The most latent frames the audio VAE decodes in one call, so that no call holds a GPU
    /// long enough for its driver to reset it. The audio is the same, to rounding, however the
    /// frames are split.
    pub decode_chunk_frames: NonZeroUsize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sampling: Sampling::default(),
            max_patches: 256,
            decode_chunk_frames: NonZeroUsize::new(25).expect("not zero"),
        }
    }
}

/// VoxCPM2 as the Python service ran it: text spoken in a voice, as audio.
#[derive(Debug)]
pub struct TextToSpeech {
    config: Config,
    tokenizer: Tokenizer,
    model: Model,
    decoder: Decoder,
}

impl TextToSpeech {
    /// Loads the VoxCPM2 checkpoint in `dir`, its `config.json`, `tokenizer.json`,
    /// `model.safetensors` and `audiovae.pth`, each part in its dtype from `dtypes`, on
    /// `device`.
    ///
    /// # Errors
    ///
    /// If a file cannot be read, asks for what is not implemented, or does not fit the config.
    pub fn load(dir: &Path, dtypes: &Dtypes, device: &Device) -> Result<Self, Error> {
        let read = |name| {
            let path = dir.join(name);
            fs::read_to_string(&path).map_err(|e| Error::Io(path, e))
        };
        let config = Config::from_json(&read("config.json")?)?;
        if config.feat_dim != LATENT_DIM {
            return Err(Error::Unsupported(format!(
                "feat_dim {} where voices have {LATENT_DIM} channels",
                config.feat_dim
            )));
        }
        let tokenizer = Tokenizer::from_json(&read("tokenizer.json")?)?;
        let model = Model::load(&config, &dir.join("model.safetensors"), dtypes, device)?;
        let weights = Weights::open(dir.join("audiovae.pth")).map_err(Error::AudioVae)?;
        let decoder = Decoder::load(&config.audio_vae(), &weights, dtypes.audio_decoder, device)
            .map_err(Error::AudioVae)?;
        Ok(Self {
            config,
            tokenizer,
            model,
            decoder,
        })
    }

    /// The rate of the audio [`synthesize`](Self::synthesize) returns, in Hz: the audio VAE's
    /// `out_sample_rate`.
    pub fn out_sample_rate(&self) -> u32 {
        self.config.audio_vae().out_sample_rate
    }

    /// `text` spoken in `voice`, as mono samples at [`out_sample_rate`](Self::out_sample_rate),
    /// sampled from `noise`: generated from the prompt the Python service gave with a voice, its
    /// transcript before the text and its latents both as reference audio and as audio to
    /// continue, and decoded after the voice's last frames.
    ///
    /// # Errors
    ///
    /// If the voice is not a whole number of patches, or it, the text and `params.max_patches`
    /// would run past the positions the LM has, or a patch comes out NaN or infinite.
    pub fn synthesize(
        &self,
        text: &str,
        voice: &Latent,
        params: &Params,
        noise: &mut impl Noise,
    ) -> Result<Vec<f32>, Error> {
        let prompt = self.prompt(text, voice)?;
        let patches = self
            .model
            .generate(&prompt, &params.sampling, params.max_patches, noise)?;
        Ok(self.speech_decoder(voice, params).push(patches))
    }

    /// `text` spoken in `voice` as [`synthesize`](Self::synthesize) speaks it, to rounding, but
    /// a patch's audio at a time, each decoded as soon as it is generated, for a caller to play
    /// or send on while the rest is generated.
    ///
    /// # Errors
    ///
    /// As [`synthesize`](Self::synthesize), the last as the last item.
    pub fn stream<'a, N: Noise>(
        &'a self,
        text: &str,
        voice: &Latent,
        params: &'a Params,
        noise: &'a mut N,
    ) -> Result<impl Iterator<Item = Result<Vec<f32>, Error>> + use<'a, N>, Error> {
        let prompt = self.prompt(text, voice)?;
        let (sampling, max_patches) = (&params.sampling, params.max_patches);
        let patches = self.model.patches(&prompt, sampling, max_patches, noise)?;
        let mut decoder = self.speech_decoder(voice, params);
        Ok(patches.map(move |patch| patch.map(|patch| decoder.push(patch))))
    }

    /// A decoder of the patches generated from the [`prompt`](Self::prompt) of `voice`, in
    /// calls of at most `params.decode_chunk_frames` frames.
    pub fn speech_decoder(&self, voice: &Latent, params: &Params) -> SpeechDecoder<'_> {
        let (frames, chunk_frames) = (voice.frames.as_flattened(), params.decode_chunk_frames);
        SpeechDecoder::new(&self.decoder, frames, LATENT_DIM, chunk_frames)
    }

    /// What decodes up to `speeches` speeches generated from the [`prompt`](Self::prompt) of a
    /// voice a chunk at a time, several speeches' chunks together, replaying graphs where
    /// `captured` says.
    pub fn speech_chunks(&self, speeches: usize, captured: bool) -> SpeechChunks {
        let device = self.model.device.clone();
        SpeechChunks::new(&self.decoder, LATENT_DIM, speeches, captured, &device)
    }

    /// The prompt [`synthesize`](Self::synthesize) generates from.
    ///
    /// # Errors
    ///
    /// If the voice is not a whole number of patches.
    pub fn prompt(&self, text: &str, voice: &Latent) -> Result<Prompt, Error> {
        let tokens = self.tokenizer.encode(&format!("{}{text}", voice.text));
        Prompt::cloning(&tokens, voice.frames.as_flattened(), &self.config)
    }

    /// The model, to generate from a prompt a patch at a time.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Has synthesis compute each patch as `stepping` says.
    pub fn set_stepping(&mut self, stepping: Stepping) {
        self.model.set_stepping(stepping);
    }

    /// Has the model capture graphs of the runs of prompts of `buckets`, as
    /// [`Model::set_prefill_graphs`] does.
    pub fn set_prefill_graphs(&mut self, buckets: &[(usize, usize)]) {
        self.model.set_prefill_graphs(buckets);
    }
}

/// The latent frames a [`ChunkDecoder`] decodes at a time: each chunk the same shape, so that a
/// GPU runs the same kernels for every one, the first the voice's last frames and the patches
/// after them.
pub const CHUNK_FRAMES: usize = 24;

/// Speeches decoded [`CHUNK_FRAMES`] latent frames at a time as their patches come, each after
/// the last frames of the voice it goes on from, several speeches' chunks at once: the audio
/// [`SpeechDecoder`] decodes, to rounding. A speech's last chunk is filled out with silence,
/// which the decoder, being causal, does not reach back from.
///
/// Each speech carries its state in a slot of [`DecodeSlots`], up to as many at once as it was
/// made for. Made to replay graphs, it decodes each power of two of speeches at once by replaying
/// a graph of the decoder that [`warm_up`](Self::warm_up) captured, on a slot no speech has, and
/// keeps them as long as it lives; it never captures one as it decodes speech, which a capture
/// would decode again.
pub struct SpeechChunks {
    state: Arc<Mutex<ChunkState>>,
    free: Vec<usize>,
    spare: usize,
    graphs: HashMap<usize, Graph<Tensor<3>, ChunkFn>>,
    captured: bool,
    channels: usize,
    device: Device,
    /// Whether the device says if a tensor shares its buffer, which a write to a graph's inputs
    /// must not, or it would move where a replay does not look.
    shares_known: bool,
}

type ChunkFn = Box<dyn FnMut() -> Tensor<3> + Send>;

struct ChunkState {
    slots: DecodeSlots,
    /// What each graph decodes, by its rows: the slot of each row `[rows]` and the latents
    /// `[rows, channels, CHUNK_FRAMES]`.
    inputs: HashMap<usize, (Tensor<1, Int>, Tensor<3>)>,
}

impl fmt::Debug for SpeechChunks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeechChunks")
            .field("free", &self.free.len())
            .field("graphs", &self.graphs.len())
            .finish_non_exhaustive()
    }
}

/// A speech [`SpeechChunks`] decodes: its slot and the frames that wait to be decoded.
#[derive(Debug)]
pub struct ChunkDecoder {
    slot: usize,
    /// The frames to decode next, each run `[frames, channels]`, the voice's first.
    pending: Vec<Tensor<2>>,
    frames: usize,
    /// The voice's frames at the start of the next chunk, whose audio is not the speech's.
    context: usize,
}

impl ChunkDecoder {
    /// Takes the next patches of the speech, `[n, patch_size, channels]`.
    pub fn push(&mut self, patches: Tensor<3>) {
        let [n, patch_size, channels] = patches.dims();
        self.pending
            .push(patches.reshape([n * patch_size, channels]));
        self.frames += n * patch_size;
    }

    /// Whether a whole chunk waits to be decoded.
    pub fn ready(&self) -> bool {
        self.frames >= CHUNK_FRAMES
    }
}

impl SpeechChunks {
    /// Decodes with `decoder`, of latents of `channels` channels, up to `speeches` speeches at
    /// once, on `device`; by replaying graphs where `captured` says, once warmed up.
    pub fn new(
        decoder: &Decoder,
        channels: usize,
        speeches: usize,
        captured: bool,
        device: &Device,
    ) -> Self {
        let state = ChunkState {
            slots: DecodeSlots::new(decoder, speeches + 1, device),
            inputs: HashMap::new(),
        };
        Self {
            state: Arc::new(Mutex::new(state)),
            free: (0..speeches).rev().collect(),
            spare: speeches,
            graphs: HashMap::new(),
            captured,
            channels,
            device: device.clone(),
            shares_known: Tensor::<1>::zeros([1], device).can_mut(),
        }
    }

    /// The most speeches it decodes at once.
    pub fn slots(&self) -> usize {
        self.spare
    }

    /// How many graphs of the decoder it has captured, each kept as long as it lives.
    pub fn graphs(&self) -> usize {
        self.graphs.len()
    }

    /// A speech that goes on from `voice`, its latent frames one after another.
    ///
    /// # Panics
    ///
    /// If as many speeches are being decoded as it was made for.
    pub fn start(&mut self, voice: &[f32]) -> ChunkDecoder {
        let slot = self.free.pop().expect("a slot for one more speech");
        self.state().slots.clear(slot);
        let channels = self.channels;
        let context = (voice.len() / channels).min(DECODE_CONTEXT_FRAMES);
        let frames = voice[voice.len() - context * channels..].to_vec();
        let frames = TensorData::new(frames, [context, channels]);
        ChunkDecoder {
            slot,
            pending: match context {
                0 => vec![],
                _ => vec![Tensor::from_data(frames, &self.device)],
            },
            frames: context,
            context,
        }
    }

    /// Gives back the slot of a speech that is not decoded to its end.
    pub fn release(&mut self, decoder: ChunkDecoder) {
        self.free.push(decoder.slot);
    }

    /// Decodes a chunk of each of `decoders` at once and gives each one's audio, mono samples:
    /// the next whole chunk, or for those `last` says end, what is left, their slots given back.
    ///
    /// # Panics
    ///
    /// If there is no decoder, `last` says of another number, a decoder that goes on has less
    /// than a chunk waiting, or one that ends has more.
    pub fn decode(&mut self, decoders: &mut [&mut ChunkDecoder], last: &[bool]) -> Vec<Vec<f32>> {
        assert!(!decoders.is_empty(), "no speech to decode");
        assert_eq!(decoders.len(), last.len(), "whether each speech ends");
        let mut chunks = vec![];
        let mut kept = vec![];
        for (decoder, &last) in decoders.iter_mut().zip(last) {
            let frames = Tensor::cat(mem::take(&mut decoder.pending), 0);
            let [held, channels] = frames.dims();
            let taken = if last { held } else { CHUNK_FRAMES };
            match last {
                true => assert!(held <= CHUNK_FRAMES, "{held} frames left, past a chunk"),
                false => assert!(held >= CHUNK_FRAMES, "{held} frames, short of a chunk"),
            }
            if held > taken {
                decoder.pending = vec![frames.clone().narrow(0, taken, held - taken)];
            }
            decoder.frames = held - taken;
            let mut chunk = frames.narrow(0, 0, taken);
            if taken < CHUNK_FRAMES {
                let options = (&chunk.device(), chunk.dtype());
                let silence = Tensor::zeros([CHUNK_FRAMES - taken, channels], options);
                chunk = Tensor::cat(vec![chunk, silence], 0);
            }
            chunks.push(chunk.swap_dims(0, 1).unsqueeze_dim::<3>(0));
            kept.push((mem::take(&mut decoder.context), taken));
        }
        let slots: Vec<usize> = decoders.iter().map(|decoder| decoder.slot).collect();
        let audio = self.push(&slots, Tensor::cat(chunks, 0));
        for (decoder, &last) in decoders.iter().zip(last) {
            if last {
                self.free.push(decoder.slot);
            }
        }
        let [_, _, samples] = audio.dims();
        let per_frame = samples / CHUNK_FRAMES;
        let audio: Vec<f32> = audio.cast(FloatDType::F32).into_data().iter().collect();
        let rows = audio.chunks(samples);
        let speech = rows
            .zip(kept)
            .map(|(row, (context, taken))| row[context * per_frame..taken * per_frame].to_vec());
        speech.collect()
    }

    /// The audio of `latents` `[slots.len(), channels, CHUNK_FRAMES]` through the streams in
    /// `slots`: by replaying the graph of their power of two of rows where there is one, the
    /// rows past theirs in the spare slot, or as they are.
    fn push(&mut self, slots: &[usize], latents: Tensor<3>) -> Tensor<3> {
        let rows = slots.len().next_power_of_two();
        let Some(graph) = self.graphs.get_mut(&rows) else {
            let at: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
            let at = Tensor::from_data(TensorData::new(at, [slots.len()]), &self.device);
            return self
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .slots
                .push(at, latents);
        };
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let (at, input) = state.inputs.get_mut(&rows).expect("a graph's inputs");
            let mut padded: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
            padded.resize(rows, self.spare as i64);
            let padded = Tensor::from_data(TensorData::new(padded, [rows]), &self.device);
            write(self.shares_known, at, 0..rows, padded);
            write(self.shares_known, input, 0..slots.len(), latents);
        }
        // Safety: the closure owns the state, and with it every buffer it reads and writes; the
        // inputs were written, and the audio is read, on this thread, the one it was captured
        // on, which holds `self`.
        let audio = unsafe { graph.replay() }.clone();
        audio.narrow(0, 0, slots.len())
    }

    /// Runs the decoder over every power of two of speeches at once up to as many as it was
    /// made for, in its spare slot, on latents of nothing in particular: for autotune to choose
    /// its kernels, and where it replays graphs, to capture one of each.
    pub fn warm_up(&mut self) {
        let normal = Distribution::Normal(0.0, 1.0);
        let powers = (0..).map(|power| 1usize << power);
        let most = self.spare.next_power_of_two();
        for rows in powers.take_while(|&rows| rows <= most) {
            if !self.captured {
                // Through decode, whose chunks reach the decoder in a layout of their own, which
                // keys the kernels autotune chooses.
                let frames = [CHUNK_FRAMES, self.channels];
                let decoder = |_| ChunkDecoder {
                    slot: self.spare,
                    pending: vec![Tensor::random(frames, normal, &self.device)],
                    frames: CHUNK_FRAMES,
                    context: 0,
                };
                let mut decoders: Vec<_> = (0..rows).map(decoder).collect();
                let mut decoding: Vec<_> = decoders.iter_mut().collect();
                self.decode(&mut decoding, &vec![false; rows]);
                continue;
            }
            if self.graphs.contains_key(&rows) {
                continue;
            }
            let shape = [rows, self.channels, CHUNK_FRAMES];
            let latents = Tensor::random(shape, normal, &self.device);
            let at = Tensor::<1, Int>::full([rows], self.spare as i64, &self.device);
            let state = self.state.clone();
            state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .inputs
                .insert(rows, (at, latents));
            let step: ChunkFn = Box::new(move || {
                let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
                let (at, latents) = state.inputs[&rows].clone();
                state.slots.push(at, latents)
            });
            self.graphs.insert(rows, capture(&self.device, step));
        }
    }

    fn state(&self) -> MutexGuard<'_, ChunkState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Writes `value` over the `rows` of `target`, in its buffer.
///
/// # Panics
///
/// If the device says `target` shares its buffer, which the write would then move.
pub(crate) fn write<const D: usize, K: Basic>(
    shares_known: bool,
    target: &mut Tensor<D, K>,
    rows: Range<usize>,
    value: Tensor<D, K>,
) {
    assert!(
        !shares_known || target.can_mut(),
        "a buffer a graph reads is shared, so a write would move it from under the graph"
    );
    let mut at = target.dims().map(|n| 0..n);
    at[0] = rows;
    target.inplace(|target| target.slice_assign(at, value));
}

/// Speech decoded from its patches as they come, after the last frames of the voice it goes on
/// from, as nanovllm-voxcpm primes its streaming decoder with them, in calls of at most a given
/// number of frames.
///
/// The decoder being causal, the audio is what decoding the voice's frames and every patch at
/// once and cutting off the voice's samples gives, to rounding, however the patches come.
#[derive(Debug)]
pub struct SpeechDecoder<'a> {
    stream: DecodeStream<'a>,
    /// The voice's last frames, until they are decoded before the first patches.
    context: Vec<f32>,
    chunk_frames: NonZeroUsize,
}

impl<'a> SpeechDecoder<'a> {
    /// Decodes with `decoder` the speech that goes on from `voice`, its latent frames of
    /// `channels` channels one after another, at most `chunk_frames` frames a call.
    pub fn new(
        decoder: &'a Decoder,
        voice: &[f32],
        channels: usize,
        chunk_frames: NonZeroUsize,
    ) -> Self {
        let context = (voice.len() / channels).min(DECODE_CONTEXT_FRAMES);
        Self {
            stream: decoder.stream(),
            context: voice[voice.len() - context * channels..].to_vec(),
            chunk_frames,
        }
    }

    /// The audio of the next `patches` `[n, patch_size, channels]`, as mono samples.
    pub fn push(&mut self, patches: Tensor<3>) -> Vec<f32> {
        let [n, patch_size, channels] = patches.dims();
        let device = patches.device();
        let mut frames = patches.reshape([n * patch_size, channels]);
        let context = mem::take(&mut self.context);
        let context_frames = context.len() / channels;
        if context_frames > 0 {
            let context = TensorData::new(context, [context_frames, channels]);
            frames = Tensor::cat(vec![Tensor::from_data(context, &device), frames], 0);
        }
        let [total, _] = frames.dims();
        let chunk = self.chunk_frames.get();
        let audio = (0..total).step_by(chunk).map(|start| {
            let chunk = frames.clone().narrow(0, start, chunk.min(total - start));
            self.stream.push(chunk.swap_dims(0, 1).unsqueeze_dim(0))
        });
        let audio = Tensor::cat(audio.collect(), 2);
        let [_, _, samples] = audio.dims();
        let cut = samples / total * context_frames;
        audio
            .narrow(2, cut, samples - cut)
            .cast(FloatDType::F32)
            .into_data()
            .iter::<f32>()
            .collect()
    }
}
