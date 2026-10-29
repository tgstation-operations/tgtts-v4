//! The parts of VoxCPM2 that compute no tensors: the checkpoint's `config.json`, with its
//! stacks' and audio VAE's sections, the tokenizer, the prompt a request is laid out as, how its
//! patches are sampled and the noise they are sampled from. The service lays out its requests
//! and draws their noise with them, and the `graph` crate runs the model on the GPU.

mod audio_vae;
mod config;
pub mod minicpm4;
mod noise;
mod params;
mod prompt;
mod tokenizer;

pub use crate::{
    audio_vae::AudioVaeConfig,
    config::{CfmConfig, Config, DitConfig, EncoderConfig},
    noise::{Noise, SeededNoise},
    params::{Params, Sampling},
    prompt::Prompt,
    tokenizer::Tokenizer,
};

/// Why a config, a tokenizer or a prompt was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `config.json` is not a config upstream would read.
    #[error("VoxCPM2 config: {0}")]
    Config(serde_json::Error),
    /// The config asks for something tts-rs does not implement.
    #[error("unsupported VoxCPM2 config: {0}")]
    Unsupported(String),
    /// `tokenizer.json` is not one this crate implements.
    #[error("tokenizer.json: {0}")]
    Tokenizer(String),
    /// A request the model cannot serve.
    #[error("cannot generate from {0}")]
    Request(String),
}
