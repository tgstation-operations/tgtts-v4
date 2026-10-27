//! A step of a model as a graph of GPU dispatches at fixed shapes, recorded once and submitted
//! every step: the layer above the `vulkan` crate's driver, toward a runtime of its own. burn's
//! CPU backend stays the oracle it is checked against.
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
//!
//! The tests that need a GPU are built with the `device` feature: `nix run .#device-tests` runs
//! them under the validation layer.

pub mod checkpoint;
pub mod decode;
pub mod graph;
pub mod kernels;
pub mod lm_side;
pub mod prefill;
pub mod step;
pub mod vae;

use std::fmt;

/// Why a step could not be built or run.
#[derive(Debug)]
pub enum Error {
    Vulkan(vulkan::Error),
    /// A kernel's SPIR-V could not be found or read.
    Kernels(String),
    /// The checkpoint could not be read, or lacks a weight at the shape the step needs.
    Checkpoint(String),
    /// Shapes the kernels cannot run, or a config VoxCPM2 does not use.
    Shape(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Vulkan(e) => write!(f, "{e}"),
            Error::Kernels(why) => write!(f, "kernels: {why}"),
            Error::Checkpoint(why) => write!(f, "checkpoint: {why}"),
            Error::Shape(why) => write!(f, "shape: {why}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Vulkan(e) => Some(e),
            _ => None,
        }
    }
}

impl From<vulkan::Error> for Error {
    fn from(e: vulkan::Error) -> Self {
        Error::Vulkan(e)
    }
}
