//! `tgtts [CONFIG]`: serves the game's TTS routes, configured by the TOML file `CONFIG` and the
//! environment, as the `config` module describes.

use std::{env, error::Error, path::PathBuf, process::ExitCode, sync::Arc};

use burn::tensor::{Device, FloatDType};
use tgtts::{
    cache::Cache,
    config::{self, Config},
    ffmpeg::Ffmpeg,
    metrics::Metrics,
    model::VoxCpm,
    service::{self, Limits, Logging, Service, Setup, Sounds},
};
use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, reload, util::SubscriberInitExt};
use voices::Voices;
use voxcpm::{Dtypes, Params};

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
    let (level, reload) = reload::Layer::new(config.log_level);
    tracing_subscriber::registry()
        .with(level)
        .with(tracing_subscriber::fmt::layer())
        .init();
    let set_level = move |level: LevelFilter| {
        if let Err(e) = reload.modify(|filter| *filter = level) {
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

fn run(config: Config, logging: Logging) -> Result<(), Box<dyn Error>> {
    let token = config::read_token(&config.token_file)?;
    let voices = Voices::open(&config.voices_dir)?;
    let sounds = Sounds::open(&config.assets_dir)?;
    let cache = Cache::open(&config.cache_dir)?;
    let metrics = match &config.metrics_db {
        Some(path) => Some(Metrics::open(path, METRICS_QUEUE)?),
        None => None,
    };
    info!(checkpoint = %config.checkpoint_dir.display(), "loading VoxCPM2");
    let f32 = FloatDType::F32;
    let dtypes = Dtypes {
        lm: f32,
        local_encoder: f32,
        local_dit: f32,
        audio_decoder: f32,
    };
    let model = VoxCpm::load(
        &config.checkpoint_dir,
        &dtypes,
        &Device::flex(),
        Params::default(),
    )?;
    let service = Service::new(Setup {
        token,
        voices,
        synthesizer: Arc::new(model),
        cache,
        ffmpeg: Ffmpeg::new(&config.ffmpeg),
        sounds,
        metrics,
        logging,
        loudness: config.loudness,
        limits: Limits {
            synthesis: config.synthesis_limit,
            cpu: config.cpu_limit,
            queue_timeout: config.queue_timeout,
            radio_wait: config.radio_wait,
        },
    });
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(config.listen).await?;
        info!(address = %config.listen, "listening");
        let shutdown = async {
            let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM can be handled");
            tokio::select! {
                _ = terminate.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
            info!("shutting down");
        };
        axum::serve(listener, service::router(service))
            .with_graceful_shutdown(shutdown)
            .await
    })?;
    Ok(())
}
