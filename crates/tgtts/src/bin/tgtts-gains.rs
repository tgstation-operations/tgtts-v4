//! `tgtts-gains CONFIG LEVELS [--requests N] [--keep DIR] [--seconds S]`, or `tgtts-gains --report
//! LEVELS`: measures how loud each voice comes out of the model, as [`tgtts::levels`] has it, for
//! the service's `voice_levels_file`, on the model the service loads from the TOML file `CONFIG`
//! and the environment.
//!
//! It reads the radio's clicks and static from the configuration's assets too, for the peaks the
//! radio scales each voice's streamed speech by. Up to `N` lines are asked for at once, the
//! service's `synthesis_limit` by default, so that a model that batches them does. The table is
//! written to `LEVELS` as each voice is done, whole or not at all, and a run that finds it there
//! goes on with the voices it does not have; a voice cut short is spoken again. Each line goes to
//! `LEVELS.lines` as a line of JSON once its voice is done: its loudness, and how far the radio
//! on a stream of it is from today's. With `--keep DIR`, each voice's first line is kept there as
//! 16-bit WAV, as the model said it. With `--seconds S` it asks for no more lines after S
//! seconds, measures those it asked for, and shuts the model down, leaving the voices it did not
//! finish to the next run. With `--report` it loads no model, and prints what `LEVELS.lines` says,
//! as [`tgtts::levels::report`] has it.

use std::{
    env,
    error::Error,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use tgtts::{
    config::Config,
    device,
    levels::{self, Levels},
    load,
    service::Sounds,
};
use tracing::error;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use voices::Voices;

fn main() -> ExitCode {
    let usage = || {
        eprintln!(
            "usage: tgtts-gains CONFIG LEVELS [--requests N] [--keep DIR] [--seconds S]\n       \
             tgtts-gains --report LEVELS"
        );
        ExitCode::from(2)
    };
    let (mut positional, mut requests, mut keep, mut until) = (Vec::new(), None, None, None);
    let mut report = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--requests" => match args.next().and_then(|n| n.parse::<usize>().ok()) {
                Some(n) if n > 0 => requests = Some(n),
                _ => return usage(),
            },
            "--keep" => match args.next() {
                Some(dir) => keep = Some(PathBuf::from(dir)),
                None => return usage(),
            },
            "--report" => report = true,
            "--seconds" => match args.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(s) if s >= 0.0 && s.is_finite() => {
                    until = Some(Instant::now() + Duration::from_secs_f64(s));
                }
                _ => return usage(),
            },
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    if report {
        let [levels] = &positional[..] else {
            return usage();
        };
        let mut lines = levels.as_os_str().to_owned();
        lines.push(".lines");
        let reported = fs::read_to_string(&lines)
            .map_err(|e| e.to_string())
            .and_then(|log| levels::report(&log).map_err(|e| e.to_string()));
        return match reported {
            Ok(report) => {
                print!("{report}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("tgtts-gains: {}: {e}", PathBuf::from(lines).display());
                ExitCode::FAILURE
            }
        };
    }
    let [config, levels] = &positional[..] else {
        return usage();
    };
    let config = match Config::load(Some(config), |key| env::var(key).ok()) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("tgtts-gains: {e}");
            return ExitCode::FAILURE;
        }
    };
    tracing_subscriber::registry()
        .with(device::log_filter(config.log_level))
        .with(tracing_subscriber::fmt::layer())
        .init();
    let requests = requests.unwrap_or(config.synthesis_limit);
    match run(&config, levels, requests, keep.as_deref(), until) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(
    config: &Config,
    path: &Path,
    requests: usize,
    keep: Option<&Path>,
    until: Option<Instant>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let voices = Voices::open(&config.voices_dir)?;
    let mut levels = match fs::exists(path)? {
        true => Levels::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
        false => Levels::default(),
    };
    if let Some(keep) = keep {
        fs::create_dir_all(keep)?;
    }
    let mut lines = path.as_os_str().to_owned();
    lines.push(".lines");
    let mut log = OpenOptions::new().create(true).append(true).open(&lines)?;
    let device = device::open(
        config.device,
        &config.dtypes,
        config.autotune_cache_dir.as_deref(),
    )?;
    let model = load::synthesizer(config, &device)?;
    let sounds = Sounds::open(&config.assets_dir)?;
    let measure = levels::Measure {
        model: &*model,
        voices: &voices,
        requests,
        loudness: config.loudness,
        radio: sounds.radio(),
        keep,
        until,
    };
    measure.run(&mut levels, path, &mut log)?;
    Ok(())
}
