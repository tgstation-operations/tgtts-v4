//! Times the hot paths of the text-to-speech models on fixed, seeded inputs, with the real
//! weights where their dtype and shapes set the cost, and compares runs.
//!
//! A run takes `/tmp/tts-rs-bench.lock` for its whole length and refuses to start on a machine
//! that is not quiet; benchmarks that hold whole MiniCPM-4 stacks also take
//! `/tmp/tts-rs-bigmem.lock`. It writes its timings and the conditions they were taken under
//! as JSON and prints a summary.

mod compare;
mod conditions;
mod counters;
mod runner;
mod stats;
mod suite;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use burn::tensor::{Device, FloatDType};
use serde::{Deserialize, Serialize};

use crate::{
    conditions::{BENCH_LOCK, Conditions, CounterSupport, Quietness, lock},
    counters::Counters,
    runner::{Counts, Measurement, Runner, line},
    suite::{Data, NAMES},
};

const USAGE: &str = "\
usage: bench [run] [--filter TEXT]... [--quick] [--warmup N] [--reps N] [--dtype f32|f16|bf16]
             [--device flex] [--voice NAME] [--label TEXT] [--out DIR] [--max-load CPUS]
             [--noisy-ok]
       bench compare BASE.json NEW.json [--spread RUN.json RUN.json...]
       bench show RUN.json
       bench list

run times the benchmarks whose names contain any --filter (all by default), with VOXCPM2_DIR
naming the VoxCPM2 checkpoint and VOICES_DIR a converted voices directory, and writes
<commit>-<time>.json to --out (target/bench by default). --quick is 1 warmup and 3 timed
repetitions, for trying the harness, not for measuring. It refuses to start, with exit status 3,
while the 1-minute load average or the CPUs others keep busy exceed --max-load (default 1);
--noisy-ok times anyway and marks the run noisy.

compare says which benchmarks changed between two runs beyond their noise; --spread names runs
of one commit taken back to back, whose differences measure the run-to-run noise.";

/// One CPU's worth of other work. A threaded benchmark's parallel regions finish with their
/// slowest worker, so even one busy foreign thread can stretch them far beyond its share of
/// the machine; below one CPU on average, other work is intermittent rather than sustained.
const DEFAULT_MAX_LOAD: f64 = 1.0;

/// A run as its JSON file holds it.
#[derive(Debug, Serialize, Deserialize)]
pub struct Run {
    pub label: Option<String>,
    pub started: String,
    pub finished: String,
    pub options: Options,
    /// The check before anything was timed.
    pub quietness: Quietness,
    /// Why the timings may not be trusted; empty for a quiet run.
    pub noisy: Vec<String>,
    pub conditions: Conditions,
    pub benchmarks: Vec<Measurement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Options {
    pub device: String,
    pub dtype: String,
    pub filters: Vec<String>,
    pub warmup: Option<usize>,
    pub reps: Option<usize>,
    pub quick: bool,
    pub voice: Option<String>,
    pub max_load: f64,
    pub noisy_ok: bool,
}

pub enum Failure {
    Usage(String),
    NotQuiet(String),
    Failed(String),
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("compare") => compare::main(&args[1..]),
        Some("show") => show(&args[1..]),
        Some("list") => {
            NAMES.iter().for_each(|name| println!("{name}"));
            Ok(())
        }
        Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => run(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(e)) => {
            eprintln!("bench: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::NotQuiet(e)) => {
            eprintln!("bench: {e}");
            ExitCode::from(3)
        }
        Err(Failure::Failed(e)) => {
            eprintln!("bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), Failure> {
    // Before anything can start a thread, so that the counters follow every thread there is.
    let (counters, refused) = Counters::open();
    let support = CounterSupport {
        recorded: counters.names(),
        refused,
    };
    let mut options = Options {
        device: "flex".into(),
        dtype: "f32".into(),
        filters: Vec::new(),
        warmup: None,
        reps: None,
        quick: false,
        voice: None,
        max_load: DEFAULT_MAX_LOAD,
        noisy_ok: false,
    };
    let (mut label, mut out) = (None, None);
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || {
            let value = args.next().cloned();
            value.ok_or_else(|| Failure::Usage(format!("{arg} needs a value")))
        };
        match arg.as_str() {
            "--filter" => options.filters.push(value()?),
            "--quick" => options.quick = true,
            "--warmup" => options.warmup = Some(number(arg, &value()?)?),
            "--reps" => options.reps = Some(number(arg, &value()?)?),
            "--dtype" => options.dtype = value()?,
            "--device" => options.device = value()?,
            "--voice" => options.voice = Some(value()?),
            "--label" => label = Some(value()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--max-load" => options.max_load = number(arg, &value()?)?,
            "--noisy-ok" => options.noisy_ok = true,
            other => return Err(Failure::Usage(format!("unknown argument {other}"))),
        }
    }
    let dtype = match options.dtype.as_str() {
        "f32" => FloatDType::F32,
        "f16" => FloatDType::F16,
        "bf16" => FloatDType::BF16,
        other => return Err(Failure::Usage(format!("unknown dtype {other}"))),
    };
    let device = match options.device.as_str() {
        "flex" => Device::flex(),
        other => return Err(Failure::Usage(format!("this build has no device {other}"))),
    };
    let counts = if options.quick {
        Counts {
            warmup: Some(options.warmup.unwrap_or(1)),
            reps: Some(options.reps.unwrap_or(3)),
        }
    } else {
        Counts {
            warmup: options.warmup,
            reps: options.reps,
        }
    };
    let filters = options.filters.clone();
    let counters = (!support.recorded.is_empty()).then_some(counters);
    let mut runner = Runner::new(
        device,
        dtype,
        NAMES,
        filters,
        counts,
        options.max_load,
        counters,
    );
    let dir = |var| {
        env::var_os(var)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let (checkpoint, voices) = (dir("VOXCPM2_DIR"), dir("VOICES_DIR"));
    let data = Data {
        checkpoint: checkpoint.as_deref(),
        voices: voices.as_deref(),
        voice: options.voice.as_deref(),
    };
    suite::check(&runner, &data).map_err(Failure::Usage)?;

    let (_lock, _) = lock(BENCH_LOCK).map_err(|e| Failure::Failed(format!("{BENCH_LOCK}: {e}")))?;
    let quietness = Quietness::sample(options.max_load);
    let state = format!(
        "1-minute load {:.2}, {:.2} CPUs busy over 2 s, limit {:.2}",
        quietness.load1, quietness.busy_cores, quietness.limit
    );
    if !quietness.is_quiet() {
        if !options.noisy_ok {
            return Err(Failure::NotQuiet(format!(
                "the machine is not quiet: {state}; wait, or pass --noisy-ok to time anyway \
                 and mark the run noisy"
            )));
        }
        runner.noisy.push(format!("before timing: {state}"));
    }
    eprintln!("bench: {state}");

    let mut conditions = Conditions::before(support);
    let started = now();
    suite::run(&mut runner, &data).map_err(Failure::Failed)?;
    conditions.finish();
    let run = Run {
        label,
        started: started.iso,
        finished: now().iso,
        options,
        quietness,
        noisy: runner.noisy,
        conditions,
        benchmarks: runner.measurements,
    };

    let out =
        out.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/bench"));
    fs::create_dir_all(&out).map_err(|e| Failure::Failed(format!("{}: {e}", out.display())))?;
    let out = fs::canonicalize(&out).unwrap_or(out);
    let commit = run.conditions.commit.as_deref().unwrap_or("unknown");
    let dirty = if run.conditions.dirty == Some(false) {
        ""
    } else {
        "-dirty"
    };
    let path = out.join(format!(
        "{}{dirty}-{}.json",
        &commit[..commit.len().min(12)],
        started.stamp
    ));
    let json = serde_json::to_string_pretty(&run).expect("a run serializes");
    fs::write(&path, json + "\n")
        .map_err(|e| Failure::Failed(format!("{}: {e}", path.display())))?;
    print!("{}", summary(&run));
    eprintln!("bench: wrote {}", path.display());
    Ok(())
}

fn number<T: std::str::FromStr>(arg: &str, value: &str) -> Result<T, Failure> {
    value
        .parse()
        .map_err(|_| Failure::Usage(format!("{arg}: {value} is not a number")))
}

fn show(args: &[String]) -> Result<(), Failure> {
    let [path] = args else {
        return Err(Failure::Usage("show takes one run".into()));
    };
    print!("{}", summary(&read_run(path)?));
    Ok(())
}

pub fn read_run(path: &str) -> Result<Run, Failure> {
    let text = fs::read_to_string(path).map_err(|e| Failure::Failed(format!("{path}: {e}")))?;
    serde_json::from_str(&text).map_err(|e| Failure::Failed(format!("{path}: {e}")))
}

/// The run's conditions in brief, then a line per benchmark.
pub fn summary(run: &Run) -> String {
    let c = &run.conditions;
    let commit = c.commit.as_deref().unwrap_or("unknown commit");
    let dirty = match c.dirty {
        Some(false) => "",
        Some(true) => " (dirty)",
        None => " (dirty unknown)",
    };
    let b = &c.build;
    let mut out = format!(
        "run {}{} at {commit}{dirty}, {} to {}\n",
        run.label.as_deref().unwrap_or("unlabeled"),
        if run.options.quick { " (quick)" } else { "" },
        run.started,
        run.finished
    );
    out += &format!(
        "build: {} opt-level {} on {}, rustflags [{}], cargo args [{}]\n",
        b.profile,
        b.opt_level,
        b.rustc,
        b.rustflags,
        b.cargo_args.as_deref().unwrap_or("unknown")
    );
    if let Some(features) = &b.resolved_features {
        out += &format!("features: {}\n", features.replace('\n', "; "));
    }
    out += &format!(
        "host: {}, {} CPUs, governor {}, EPP {}, platform profile {}\n",
        c.host.cpu_model, c.host.logical_cpus, c.host.governor, c.host.epp, c.host.platform_profile
    );
    out += &format!(
        "device {} in {}; threads: {} available, {} at the end; load {} before, {} after\n",
        run.options.device,
        run.options.dtype,
        c.threads.available_parallelism,
        c.threads
            .process_threads_at_end
            .map_or("unknown".into(), |n| n.to_string()),
        c.loadavg_before,
        c.loadavg_after
    );
    let counters = &c.counters;
    if !counters.recorded.is_empty() || !counters.refused.is_empty() {
        let refused: Vec<String> = counters
            .refused
            .iter()
            .map(|(event, why)| format!("{event} ({why})"))
            .collect();
        out += &format!(
            "counters per repetition: [{}], refused: [{}]\n",
            counters.recorded.join(", "),
            refused.join(", ")
        );
    }
    if run.noisy.is_empty() {
        out += "quiet throughout\n";
    } else {
        out += "NOISY:\n";
        for reason in &run.noisy {
            out += &format!("  {reason}\n");
        }
    }
    for m in &run.benchmarks {
        out += &line(m);
        out += "\n";
    }
    out
}

struct Now {
    /// `2026-10-25T15:30:00Z`
    iso: String,
    /// `20261025T153000Z`, for file names.
    stamp: String,
}

fn now() -> Now {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is past 1970")
        .as_secs();
    let (days, time) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (time / 3600, time / 60 % 60, time % 60);
    let (year, month, day) = civil(days as i64);
    Now {
        iso: format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z"),
        stamp: format!("{year:04}{month:02}{day:02}T{h:02}{m:02}{s:02}Z"),
    }
}

/// The proleptic Gregorian date `days` after 1970-01-01, by Howard Hinnant's `civil_from_days`.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_cross_leap_days_and_centuries() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(11_016), (2000, 2, 29));
        assert_eq!(civil(20_751), (2026, 10, 25));
        assert_eq!(civil(-1), (1969, 12, 31));
    }
}
