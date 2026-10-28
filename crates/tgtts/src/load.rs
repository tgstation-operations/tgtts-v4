//! VoxCPM2 loaded as the configuration asks, on the device it names, warmed up: as the service
//! serves it, and as `tgtts-gains` measures it.

#[cfg(feature = "rt")]
use std::path::PathBuf;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use burn::tensor::Device;
use tracing::info;
use voices::Voices;
use voxcpm::Params;

use crate::{
    config::Config,
    device::{self, Opened},
    model::VoxCpm,
    synth::Synthesizer,
};

/// The bytes on the device the prompt cache keeps of the voices unless the configuration says
/// otherwise: on a GPU in f16 a median voice takes about 11 MB, so about 45 of them. On real
/// rounds twice as much made no difference to how long requests took.
pub const PROMPT_CACHE_BYTES: usize = 1 << 29;

/// The positions of waiting prompts run between two patches while other requests generate,
/// unless the configuration says otherwise, past which only the first runs.
pub const PREFILL_POSITIONS: usize = 128;

/// The runs of prompts joining together replayed from graphs unless the configuration says
/// otherwise, as rows and positions run: on real rounds nine in ten of those that join between
/// two patches are one prompt, of 100 to 600 positions. Each graph holds pages of the GPU's
/// memory of its own, and graphs of two prompts too took the service past 15 GB of a 16 GB card.
pub const PREFILL_GRAPHS: [(usize, usize); 2] = [(1, 256), (1, 512)];

/// How requests are batched.
#[derive(Clone, Debug, PartialEq)]
pub struct Batching {
    /// How long the first request to an idle model waits for others to start with it.
    pub window: Duration,
    /// The most requests at once, which stop the wait.
    pub requests: usize,
    /// Whether a batch steps by replaying graphs of its step where the device records them.
    pub graphs: bool,
    /// The positions of prompts run between two patches while other requests generate.
    pub prefill_positions: usize,
    /// The most positions of prompts run together, rows and longest rounded up to powers of
    /// two: voxcpm's own `PREFILL_BATCH` where it is `None`, which a build without burn cannot
    /// name.
    pub prefill_batch: Option<usize>,
    /// The bytes on the device the prompt cache keeps of the voices.
    pub prompt_cache_bytes: usize,
    /// The rows and positions run of the runs of prompts that join together replayed from
    /// graphs, where the step is.
    pub prefill_graphs: Vec<(usize, usize)>,
}

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
