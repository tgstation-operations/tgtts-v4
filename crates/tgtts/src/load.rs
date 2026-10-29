//! VoxCPM2 loaded as the configuration asks, on tts-rs's own runtime on the GPU the `vulkan` crate
//! chooses, warmed up: as the service serves it, and as `tgtts-gains` measures it.

use std::{error::Error, sync::Arc, time::Instant};

use tracing::info;
use voices::Voices;
use voxcpm::Params;

use crate::{config::Config, rt::Rt, synth::Synthesizer};

/// The bytes on the device the prompt cache keeps of the voices unless the configuration says
/// otherwise: on a GPU in f16 a median voice takes about 11 MB, so about 45 of them. On real
/// rounds twice as much made no difference to how long requests took.
pub const PROMPT_CACHE_BYTES: usize = 1 << 29;

/// VoxCPM2 as `config` asks for it, warmed up.
///
/// # Errors
///
/// If the checkpoint, the voices or the kernels cannot be read, or the GPU cannot run them.
pub fn synthesizer(config: &Config) -> Result<Arc<dyn Synthesizer>, Box<dyn Error + Send + Sync>> {
    info!(checkpoint = %config.checkpoint_dir.display(), "loading VoxCPM2");
    let started = Instant::now();
    let mut model = Rt::load(
        &config.checkpoint_dir,
        &config.glsl_kernels_dir,
        &config.mlir_kernels_dir,
        &Params::default(),
        config.prompt_cache_bytes,
    )?;
    model.set_babble_cap(config.babble_cap);
    info!(seconds = started.elapsed().as_secs_f64(), "loaded VoxCPM2");
    model.warm_up(&Voices::open(&config.voices_dir)?)?;
    info!("ready");
    Ok(Arc::new(model))
}
