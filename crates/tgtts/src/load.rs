//! VoxCPM2 loaded as the configuration asks, on the device it names, warmed up: as the service
//! serves it, and as `tgtts-gains` measures it.

#[cfg(feature = "rt")]
use std::path::PathBuf;
use std::{error::Error, sync::Arc, time::Instant};

use burn::tensor::Device;
use tracing::info;
use voices::Voices;
use voxcpm::Params;

use crate::{
    config::Config,
    device::{self, Opened},
    model::{Batching, VoxCpm},
    synth::Synthesizer,
};

/// VoxCPM2 as `config` asks for it, on the device it opened, `device`, warmed up.
///
/// # Errors
///
/// If the checkpoint, the voices or the kernels cannot be read, or the device cannot run them.
pub fn synthesizer(
    config: &Config,
    device: &Opened,
) -> Result<Arc<dyn Synthesizer>, Box<dyn Error + Send + Sync>> {
    match device {
        Opened::Burn(device) => burn(config, device),
        #[cfg(feature = "rt")]
        Opened::Rt => rt(config),
    }
}

/// VoxCPM2 as `config` asks for it, on `device`, warmed up where its kernels are chosen as each
/// shape first runs.
fn burn(
    config: &Config,
    device: &Device,
) -> Result<Arc<dyn Synthesizer>, Box<dyn Error + Send + Sync>> {
    info!(
        checkpoint = %config.checkpoint_dir.display(),
        device = %config.device,
        dtypes = ?config.dtypes,
        "loading VoxCPM2"
    );
    let started = Instant::now();
    let batching = Batching {
        window: config.batch_window,
        requests: config.synthesis_limit,
        // Replayed on Vulkan, 16 requests' first patches came out NaN, and replays saved nothing
        // on the Radeon they were measured on.
        graphs: config.step_graphs && config.device == device::Kind::Cuda,
        prefill_positions: config.prefill_positions,
        prefill_graphs: config.prefill_graphs.clone(),
        prefill_batch: config.prefill_batch,
        prompt_cache_bytes: config.prompt_cache_bytes,
    };
    let model = VoxCpm::load(
        &config.checkpoint_dir,
        &config.dtypes,
        device,
        Params::default(),
        batching,
    )?;
    info!(seconds = started.elapsed().as_secs_f64(), "loaded VoxCPM2");
    if config.device.autotunes() {
        let voices = Voices::open(&config.voices_dir)?;
        model.warm_up(&voices)?;
    }
    info!("ready");
    Ok(Arc::new(model))
}

/// VoxCPM2 on the rt device as `config` asks for it, warmed up.
#[cfg(feature = "rt")]
fn rt(config: &Config) -> Result<Arc<dyn Synthesizer>, Box<dyn Error + Send + Sync>> {
    info!(
        checkpoint = %config.checkpoint_dir.display(),
        device = %config.device,
        "loading VoxCPM2"
    );
    let started = Instant::now();
    let kernels = |dir: &Option<PathBuf>| dir.clone().expect("the config requires it for rt");
    let mut model = crate::rt::Rt::load(
        &config.checkpoint_dir,
        &kernels(&config.glsl_kernels_dir),
        &kernels(&config.mlir_kernels_dir),
        &Params::default(),
        config.prompt_cache_bytes,
    )?;
    model.set_babble_cap(config.babble_cap);
    info!(seconds = started.elapsed().as_secs_f64(), "loaded VoxCPM2");
    model.warm_up(&Voices::open(&config.voices_dir)?)?;
    info!("ready");
    Ok(Arc::new(model))
}
