//! `tgtts [CONFIG]`: serves the game's TTS routes, configured by the TOML file `CONFIG` and the
//! environment, as the `config` module describes.

use std::{env, error::Error, future::IntoFuture, path::PathBuf, process::ExitCode, thread};

use argh::FromArgs;
use tgtts::{
    cache::Cache,
    config::{self, Config},
    ffmpeg::Ffmpeg,
    levels::Levels,
    load,
    metrics::Metrics,
    service::{self, Limits, Logging, Service, Setup, Sounds},
};
use tokio::{
    signal::unix::{SignalKind, signal},
    sync::oneshot,
};
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, reload, util::SubscriberInitExt};
use voices::Voices;

/// How many metrics rows may wait to be written before more are dropped.
const METRICS_QUEUE: usize = 1024;

/// Serves the game's TTS routes, configured by the TOML file CONFIG, any key of which TGTTS_<KEY>
/// in the environment overrides.
#[derive(FromArgs)]
struct Args {
    /// the configuration, a TOML file
    #[argh(positional, arg_name = "CONFIG")]
    config: Option<PathBuf>,
}

fn main() -> ExitCode {
    let Args { config: file } = argh::from_env();
    let config = match Config::load(file.as_deref(), |key| env::var(key).ok()) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("tgtts: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (level, reload) = reload::Layer::new(config::log_filter(config.log_level));
    tracing_subscriber::registry()
        .with(level)
        .with(tracing_subscriber::fmt::layer())
        .init();
    let set_level = move |level: LevelFilter| {
        if let Err(e) = reload.modify(|filter| *filter = config::log_filter(level)) {
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
            let model = load::synthesizer(&config);
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

async fn shutdown() {
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM can be handled");
    tokio::select! {
        _ = terminate.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
    info!("shutting down");
}
