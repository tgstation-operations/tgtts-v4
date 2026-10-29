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

use argh::FromArgs;
use tgtts::{
    config::{self, Config},
    levels::{self, Levels},
    load,
    service::Sounds,
};
use tracing::error;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use voices::Voices;

/// Measures how loud each voice comes out of the model, for the service's voice_levels_file, into
/// the table LEVELS, on the model the service loads from the TOML file CONFIG and the
/// environment; or with --report and LEVELS alone, loads no model and prints what LEVELS.lines
/// says.
#[derive(FromArgs)]
struct Args {
    /// the service's configuration, or with --report the table
    #[argh(positional, arg_name = "CONFIG")]
    config: PathBuf,
    /// the table, which a run that finds it goes on with
    #[argh(positional, arg_name = "LEVELS")]
    levels: Option<PathBuf>,
    /// the lines asked for at once, the service's synthesis_limit by default
    #[argh(option, arg_name = "N", from_str_fn(count))]
    requests: Option<usize>,
    /// where each voice's first line is kept as 16-bit WAV, as the model said it
    #[argh(option, arg_name = "DIR")]
    keep: Option<PathBuf>,
    /// how long lines are asked for, after which those asked for are measured
    #[argh(option, arg_name = "S", from_str_fn(seconds))]
    seconds: Option<Duration>,
    /// print what LEVELS.lines says
    #[argh(switch)]
    report: bool,
}

/// `value` as a count of at least 1.
fn count(value: &str) -> Result<usize, String> {
    match value.parse() {
        Ok(0) => Err("not at least 1".to_owned()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

/// `value` as a time of at least 0 s.
fn seconds(value: &str) -> Result<Duration, String> {
    let seconds = value.parse::<f64>().map_err(|e| e.to_string())?;
    Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let usage = || {
        eprintln!(
            "usage: tgtts-gains CONFIG LEVELS [--requests N] [--keep DIR] [--seconds S]\n       \
             tgtts-gains --report LEVELS"
        );
        ExitCode::from(2)
    };
    let args: Args = argh::from_env();
    let until = args.seconds.map(|seconds| Instant::now() + seconds);
    if args.report {
        let (levels, None) = (&args.config, &args.levels) else {
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
    let (config, Some(levels)) = (&args.config, &args.levels) else {
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
        .with(config::log_filter(config.log_level))
        .with(tracing_subscriber::fmt::layer())
        .init();
    let requests = args.requests.unwrap_or(config.synthesis_limit);
    match run(&config, levels, requests, args.keep.as_deref(), until) {
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
    let model = load::synthesizer(config)?;
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
