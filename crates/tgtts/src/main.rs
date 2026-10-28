//! `tgtts [CONFIG]`: serves the game's TTS routes, configured by the TOML file `CONFIG` and the
//! environment, as the `config` module describes.

use std::{
    env, error::Error, future::IntoFuture, path::PathBuf, process::ExitCode, sync::Arc, thread,
    time::Instant,
};

use burn::tensor::Device;
use tgtts::{
    cache::Cache,
    config::{self, Config},
    device,
    ffmpeg::Ffmpeg,
    levels::Levels,
    metrics::Metrics,
    model::{Batching, VoxCpm},
    service::{self, Limits, Logging, Service, Setup, Sounds},
    synth::Synthesizer,
};
use tokio::{
    signal::unix::{SignalKind, signal},
    sync::oneshot,
};
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, reload, util::SubscriberInitExt};
use voices::Voices;
use voxcpm::Params;

/// How many metrics rows may wait to be written before more are dropped.
const METRICS_QUEUE: usize = 1024;

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let file = match &args[..] {
        [] => None,
        [file] => Some(PathBuf::from(file)),
        _ => {
            eprintln!("usage: tgtts [CONFIG]");
            return ExitCode::from(2);
        }
    };
    let config = match Config::load(file.as_deref(), |key| env::var(key).ok()) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("tgtts: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (level, reload) = reload::Layer::new(device::log_filter(config.log_level));
    tracing_subscriber::registry()
        .with(level)
        .with(tracing_subscriber::fmt::layer())
        .with(device::Tunings)
        .init();
    let set_level = move |level: LevelFilter| {
        if let Err(e) = reload.modify(|filter| *filter = device::log_filter(level)) {
            error!("setting the log level: {e}");
        }
    };
    let logging = Logging::new(config.log_level, set_level);
    match run(config, logging) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(config: Config, logging: Logging) -> Result<(), Box<dyn Error + Send + Sync>> {
    let token = config::read_token(&config.token_file)?;
    let voices = Voices::open(&config.voices_dir)?;
    let sounds = Sounds::open(&config.assets_dir)?;
    let cache = Cache::open(&config.cache_dir)?;
    let levels = match &config.voice_levels_file {
        Some(path) => {
            let levels = Levels::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            info!(voices = levels.len(), "read the voices' levels");
            levels
        }
        None => Levels::default(),
    };
    let metrics = match &config.metrics_db {
        Some(path) => Some(Metrics::open(path, METRICS_QUEUE)?),
        None => None,
    };
    let autotune_dir = config.autotune_cache_dir.as_deref();
    let device = device::open(config.device, &config.dtypes, autotune_dir)?;
    let service = Service::new(Setup {
        token,
        voices,
        synthesizer: None,
        cache,
        ffmpeg: Ffmpeg::new(&config.ffmpeg),
        sounds,
        metrics,
        logging,
        loudness: config.loudness,
        levels,
        limits: Limits {
            synthesis: config.synthesis_limit,
            cpu: config.cpu_limit,
            queue_timeout: config.queue_timeout,
            radio_wait: config.radio_wait,
            radio_wait_ceiling: config.radio_wait_ceiling,
        },
        blip_seed: None,
    });
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(config.listen).await?;
        info!(address = %config.listen, "listening");
        let (loaded, loading) = oneshot::channel();
        let ready = service.clone();
        // A thread of its own, which shutting down does not wait for as it would for the
        // runtime's blocking pool.
        thread::spawn(move || {
            let model = match &device {
                device::Opened::Burn(device) => load(&config, device),
                #[cfg(feature = "rt")]
                device::Opened::Rt => load_rt(&config),
            };
            let _ = loaded.send(model.map(|model| ready.ready(model)));
        });
        let serve = axum::serve(listener, service::router(service))
            .with_graceful_shutdown(shutdown())
            .into_future();
        tokio::pin!(serve);
        tokio::select! {
            served = &mut serve => return Ok(served?),
            loaded = loading => loaded.map_err(|_| "loading VoxCPM2 panicked")??,
        }
        serve.await?;
        Ok(())
    })
}

/// VoxCPM2 as `config` asks for it, on `device`, warmed up where its kernels are chosen as each
/// shape first runs.
fn load(
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
fn load_rt(config: &Config) -> Result<Arc<dyn Synthesizer>, Box<dyn Error + Send + Sync>> {
    info!(
        checkpoint = %config.checkpoint_dir.display(),
        device = %config.device,
        "loading VoxCPM2"
    );
    let started = Instant::now();
    let kernels = |dir: &Option<PathBuf>| dir.clone().expect("the config requires it for rt");
    let model = tgtts::rt::Rt::load(
        &config.checkpoint_dir,
        &kernels(&config.glsl_kernels_dir),
        &kernels(&config.mlir_kernels_dir),
        &Params::default(),
        config.prompt_cache_bytes,
    )?;
    info!(seconds = started.elapsed().as_secs_f64(), "loaded VoxCPM2");
    model.warm_up(&Voices::open(&config.voices_dir)?)?;
    info!("ready");
    Ok(Arc::new(model))
}

async fn shutdown() {
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM can be handled");
    tokio::select! {
        _ = terminate.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
    info!("shutting down");
}
