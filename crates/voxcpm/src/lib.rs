//! VoxCPM2, the text-to-speech model the service runs: its MiniCPM-4 language models, local
//! encoder and local DiT around the audio VAE, following upstream VoxCPM's
//! `voxcpm.model.voxcpm2` and computing what nanovllm-voxcpm, which the Python service ran,
//! computes.
//!
//! Each part loads from the checkpoint's `model.safetensors` under upstream's state_dict names.

mod cfm;
mod config;
mod local_dit;
mod local_encoder;
mod model;
mod noise;
mod prefill;
mod prompt;
mod speech;
mod tokenizer;
mod weights;

use std::{fmt, io, path::PathBuf};

use burn::store::SafetensorsStoreError;

pub use crate::{
    cfm::{Cfm, Sampling, schedule},
    config::{CfmConfig, Config, DitConfig, EncoderConfig},
    local_dit::LocalDit,
    local_encoder::LocalEncoder,
    model::{Dtypes, Generation, Model, Patches, Step},
    noise::{Noise, SeededNoise},
    prefill::{Prefill, PromptCache},
    prompt::Prompt,
    speech::{Params, SpeechDecoder, TextToSpeech},
    tokenizer::Tokenizer,
};

/// Why a model could not be loaded, or a request served.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io(PathBuf, io::Error),
    /// `config.json` is not a config upstream would read.
    Config(serde_json::Error),
    /// The config asks for something this crate does not implement.
    Unsupported(String),
    /// A MiniCPM-4 stack could not be loaded.
    Stack(minicpm4::Error),
    /// The weights around the stacks could not be read, or do not fit the config.
    Load(SafetensorsStoreError),
    /// The audio VAE could not be loaded.
    AudioVae(audiovae::Error),
    /// `tokenizer.json` is not one this crate implements.
    Tokenizer(String),
    /// A request the model cannot serve.
    Request(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Config(e) => write!(f, "VoxCPM2 config: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported VoxCPM2 config: {what}"),
            Self::Stack(e) => write!(f, "{e}"),
            Self::Load(e) => write!(f, "loading VoxCPM2 weights: {e}"),
            Self::AudioVae(e) => write!(f, "{e}"),
            Self::Tokenizer(what) => write!(f, "tokenizer.json: {what}"),
            Self::Request(what) => write!(f, "cannot generate from {what}"),
        }
    }
}

impl std::error::Error for Error {}
