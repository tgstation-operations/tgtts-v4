//! VoxCPM2, the text-to-speech model the service runs: its MiniCPM-4 language models, local
//! encoder and local DiT around the audio VAE, following upstream VoxCPM's
//! `voxcpm.model.voxcpm2` and computing what nanovllm-voxcpm, which the Python service ran,
//! computes.

mod config;

use std::fmt;

pub use crate::config::{CfmConfig, Config, DitConfig, EncoderConfig};

/// Why a model could not be loaded.
#[derive(Debug)]
pub enum Error {
    /// `config.json` is not a config upstream would read.
    Config(serde_json::Error),
    /// The config asks for something this crate does not implement.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "VoxCPM2 config: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported VoxCPM2 config: {what}"),
        }
    }
}

impl std::error::Error for Error {}
