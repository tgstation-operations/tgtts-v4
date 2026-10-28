//! VoxCPM2, the text-to-speech model the service runs: its MiniCPM-4 language models, local
//! encoder and local DiT around the audio VAE, following upstream VoxCPM's
//! `voxcpm.model.voxcpm2` and computing what nanovllm-voxcpm, which the Python service ran,
//! computes.
//!
//! Each part loads from the checkpoint's `model.safetensors` under upstream's state_dict names.

mod batch;
mod cfm;
mod graph;
mod local_dit;
mod local_encoder;
mod model;
mod prefill;
mod speech;
mod weights;

use std::{fmt, io, path::PathBuf};

use burn::store::SafetensorsStoreError;
pub use voxcpm_core::{
    CfmConfig, Config, DitConfig, EncoderConfig, Noise, Params, Prompt, Sampling, SeededNoise,
    Tokenizer,
};

pub use crate::{
    batch::{Batch, PREFILL_BATCH, Patch, RowId, Scheduler, Started, StepReport},
    cfm::{Cfm, schedule},
    graph::{GraphCounts, Stepper, Stepping},
    local_dit::LocalDit,
    local_encoder::LocalEncoder,
    model::{Dtypes, Generation, Model, Patches, Step},
    prefill::{PREFILL_KEYS, Prefill, PromptCache},
    speech::{CHUNK_FRAMES, ChunkDecoder, SpeechChunks, SpeechDecoder, TextToSpeech},
};

/// Why a model could not be loaded, or a request served.
#[derive(Debug)]
pub enum Error {
    /// A file of the checkpoint could not be read.
    Io(PathBuf, io::Error),
    /// The config, the tokenizer or a prompt was refused.
    Core(voxcpm_core::Error),
    /// The config asks for something this crate does not implement.
    Unsupported(String),
    /// A MiniCPM-4 stack could not be loaded.
    Stack(minicpm4::Error),
    /// The weights around the stacks could not be read, or do not fit the config.
    Load(SafetensorsStoreError),
    /// The audio VAE could not be loaded.
    AudioVae(audiovae::Error),
    /// A request the model cannot serve.
    Request(String),
    /// The patch, by its index, came out NaN or infinite, as an overflow or a kernel that wrote
    /// nothing leaves it, after which the stop head might never say to stop.
    NotFinite(usize),
    /// A graph of the step would be captured on another thread than the model loaded on, whose
    /// weights it would then read from another cubecl stream, which cubecl-cuda has crashed on.
    OtherThread,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Core(e) => write!(f, "{e}"),
            Self::Unsupported(what) => write!(f, "unsupported VoxCPM2 config: {what}"),
            Self::Stack(e) => write!(f, "{e}"),
            Self::Load(e) => write!(f, "loading VoxCPM2 weights: {e}"),
            Self::AudioVae(e) => write!(f, "{e}"),
            Self::Request(what) => write!(f, "cannot generate from {what}"),
            Self::NotFinite(patch) => write!(f, "patch {patch} came out NaN or infinite"),
            Self::OtherThread => f.write_str(
                "a graph of the step would read the model's weights from another thread's stream",
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<voxcpm_core::Error> for Error {
    fn from(e: voxcpm_core::Error) -> Self {
        Self::Core(e)
    }
}
