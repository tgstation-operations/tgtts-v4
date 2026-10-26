use std::{fs, mem, num::NonZeroUsize, path::Path};

use audiovae::{DecodeStream, Decoder, Weights};
use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use voices::{LATENT_DIM, Latent};

use crate::{Config, Dtypes, Error, Model, Noise, Prompt, Sampling, Tokenizer};

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
