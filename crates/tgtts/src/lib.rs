//! The text-to-speech service the game's TTS subsystem talks to.

#[cfg(not(any(feature = "burn", feature = "rt")))]
compile_error!("tgtts needs a device to serve from: build it with burn, a default feature, or rt");

pub mod cache;
pub mod config;
pub mod device;
pub mod ffmpeg;
pub mod handoff;
pub mod levels;
pub mod load;
pub mod metrics;
#[cfg(feature = "burn")]
pub mod model;
pub mod ogg;
pub mod opus;
pub mod replay;
pub mod rt;
pub mod service;
pub mod synth;
pub mod vorbis;
pub mod wav;
