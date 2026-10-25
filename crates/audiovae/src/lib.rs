//! The audio VAE of VoxCPM2, ported from upstream VoxCPM's `audio_vae_v2.py`: causal
//! convolutional networks between audio and the latent frames the language model speaks in.
//! VoxCPM2's decoder writes 48 kHz audio, 1920 samples to a 64-channel frame at 25 Hz.
//!
//! The modules mirror upstream's, its `nn.Sequential`s included, so that `audiovae.pth`
//! loads under its own state_dict names.

mod config;
mod decoder;
mod layers;
mod weights;

use std::fmt;

pub use crate::{config::AudioVaeConfig, decoder::Decoder, weights::Weights};

/// Why a decoder could not be built or loaded.
#[derive(Debug)]
pub enum Error {
    /// The config asks for something this port does not implement.
    Unsupported(String),
    /// The weights could not be read, or do not fit the module the config describes.
    Weights(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "unsupported audio VAE config: {what}"),
            Self::Weights(what) => write!(f, "audio VAE weights: {what}"),
        }
    }
}

impl std::error::Error for Error {}
