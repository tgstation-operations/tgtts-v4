use std::{fs, path::Path};

use audiovae::{Decoder, Weights};
use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use voices::{LATENT_DIM, Latent};

use crate::{Config, Dtypes, Error, Model, Noise, Prompt, Sampling, Tokenizer};

/// The voice's last frames that nanovllm-voxcpm's streaming decoder is primed with, its
/// `n_decode_pad_frames`, so that speech starts as if it went on from the voice.
const DECODE_CONTEXT_FRAMES: usize = 12;

/// What a request asks for beyond its text and voice. The default is what the Python service
/// asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub sampling: Sampling,
    /// The most patches to generate, nanovllm-voxcpm's `max_generate_length`.
    pub max_patches: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sampling: Sampling::default(),
            max_patches: 256,
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
    /// would run past the positions the LM has.
    pub fn synthesize(
        &self,
        text: &str,
        voice: &Latent,
        params: &Params,
        noise: &mut impl Noise,
    ) -> Result<Vec<f32>, Error> {
        let tokens = self.tokenizer.encode(&format!("{}{text}", voice.text));
        let frames = voice.frames.as_flattened();
        let prompt = Prompt::cloning(&tokens, frames, &self.config)?;
        let patches = self
            .model
            .generate(&prompt, &params.sampling, params.max_patches, noise)?;
        Ok(decode(&self.decoder, frames, patches))
    }
}

/// The audio of `patches` `[n, patch_size, channels]` as mono samples, decoded after the last
/// frames of `voice`, frames of `channels` one after another, as nanovllm-voxcpm primes its
/// streaming decoder with them.
///
/// The decoder being causal, this is what that streaming decoder gives a patch at a time,
/// computed at once.
pub fn decode(decoder: &Decoder, voice: &[f32], patches: Tensor<3>) -> Vec<f32> {
    let [n, patch_size, channels] = patches.dims();
    let device = patches.device();
    let context = (voice.len() / channels).min(DECODE_CONTEXT_FRAMES);
    let voice = &voice[voice.len() - context * channels..];
    let voice = Tensor::<2>::from_data(
        TensorData::new(voice.to_vec(), [context, channels]),
        &device,
    );
    let frames = Tensor::cat(vec![voice, patches.reshape([n * patch_size, channels])], 0);
    let audio = decoder.decode(frames.swap_dims(0, 1).unsqueeze_dim(0));
    let [_, _, samples] = audio.dims();
    let per_frame = samples / (context + n * patch_size);
    let audio = audio.narrow(2, context * per_frame, samples - context * per_frame);
    audio
        .cast(FloatDType::F32)
        .into_data()
        .iter::<f32>()
        .collect()
}
