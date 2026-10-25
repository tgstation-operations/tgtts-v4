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
mod weights;

use std::fmt;

use burn::store::SafetensorsStoreError;

pub use crate::{
    cfm::{Cfm, Sampling, schedule},
    config::{CfmConfig, Config, DitConfig, EncoderConfig},
    local_dit::LocalDit,
    local_encoder::LocalEncoder,
};

/// Why a model could not be loaded.
#[derive(Debug)]
pub enum Error {
    /// `config.json` is not a config upstream would read.
    Config(serde_json::Error),
    /// The config asks for something this crate does not implement.
    Unsupported(String),
    /// A MiniCPM-4 stack could not be loaded.
    Stack(minicpm4::Error),
    /// The weights around the stacks could not be read, or do not fit the config.
    Load(SafetensorsStoreError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "VoxCPM2 config: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported VoxCPM2 config: {what}"),
            Self::Stack(e) => write!(f, "{e}"),
            Self::Load(e) => write!(f, "loading VoxCPM2 weights: {e}"),
        }
    }
}

impl std::error::Error for Error {}
