//! The parts of VoxCPM2 that compute no tensors: the checkpoint's `config.json`, with its
//! stacks' and audio VAE's sections, the tokenizer, the prompt a request is laid out as, how its
//! patches are sampled and the noise they are sampled from. The `voxcpm` crate, which runs the
//! model on burn, re-exports them, and the service's own runtime takes them without burn. The
//! dtypes each part may be held in are named here too, for a configuration read without burn;
//! `voxcpm` holds burn's own.

mod audio_vae;
mod config;
mod dtypes;
pub mod minicpm4;
mod noise;
mod params;
mod prompt;
mod tokenizer;

use std::fmt;

pub use crate::{
    audio_vae::AudioVaeConfig,
    config::{CfmConfig, Config, DitConfig, EncoderConfig},
    dtypes::{Dtype, Dtypes},
    noise::{Noise, SeededNoise},
    params::{Params, Sampling},
    prompt::Prompt,
    tokenizer::Tokenizer,
};

/// Why a config, a tokenizer or a prompt was refused.
#[derive(Debug)]
pub enum Error {
    /// `config.json` is not a config upstream would read.
    Config(serde_json::Error),
    /// The config asks for something the `voxcpm` crate does not implement.
    Unsupported(String),
    /// `tokenizer.json` is not one this crate implements.
    Tokenizer(String),
    /// A request the model cannot serve.
    Request(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "VoxCPM2 config: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported VoxCPM2 config: {what}"),
            Self::Tokenizer(what) => write!(f, "tokenizer.json: {what}"),
            Self::Request(what) => write!(f, "cannot generate from {what}"),
        }
    }
}

impl std::error::Error for Error {}
