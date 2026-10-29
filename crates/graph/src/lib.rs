//! A step of a model as a graph of GPU dispatches at fixed shapes, recorded once and submitted
//! every step: the layer above the `vulkan` crate's driver, toward a runtime of its own. Its tests
//! on the real weights hold it to goldens of its own output, testkit's.
//!
//! - [`graph`]: the step as data, the ops over its tensors in order; one arena laid out from
//!   their lifetimes ([`graph::plan`]), and the barriers the order needs ([`graph::barriers`]).
//! - [`kernels`]: the hand-written GLSL kernels it dispatches, compiled ahead of time by
//!   `nix build .#glsl-kernels`, and their pipelines.
//! - [`step`]: a graph allocated and recorded into one command buffer.
//! - [`decode`]: MiniCPM-4's decode step at batch 1, as VoxCPM2's base LM runs it, with its
//!   weights read from the checkpoint ([`checkpoint`]).
//! - [`lm_side`]: VoxCPM2's LM side of a patch, both LMs and the layers around them, from the
//!   local encoder's output to the DiT's condition and the stop logits.
//! - [`local`]: VoxCPM2's patch sampler with its local DiT, and its local encoder, at batch 1.
//! - [`patch`]: a patch of VoxCPM2's generation as one step: the local encoder, the LM side and
//!   the sampler.
//! - [`prompt`]: a prompt run into the patch step's caches.
//! - [`generator`]: VoxCPM2's text-to-speech from a prompt to audio, a request at a time.
//!
//! The tests that need a GPU are built with the `device` feature: `nix run .#device-tests` runs
//! them under the validation layer.

pub mod checkpoint;
pub mod decode;
pub mod generator;
pub mod graph;
pub mod kernels;
pub mod lm_side;
pub mod local;
pub mod patch;
pub mod prefill;
pub mod prompt;
pub mod pth;
pub mod step;
pub mod vae;

/// Why a step could not be built or run.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Vulkan(#[from] vulkan::Error),
    /// A kernel's SPIR-V could not be found or read.
    #[error("kernels: {0}")]
    Kernels(String),
    /// The checkpoint could not be read, or lacks a weight at the shape the step needs.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// Shapes the kernels cannot run, or a config VoxCPM2 does not use.
    #[error("shape: {0}")]
    Shape(String),
    /// The patch, by its index, came out NaN or infinite, after which the stop head might never
    /// say to stop.
    #[error("patch {0} came out NaN or infinite")]
    NotFinite(usize),
}
