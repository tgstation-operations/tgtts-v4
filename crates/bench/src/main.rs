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
mod device;
mod runner;
mod stats;
mod suite;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use burn::tensor::FloatDType;
use serde::{Deserialize, Serialize};
use voxcpm::Dtypes;

use crate::{
    conditions::{BENCH_LOCK, Conditions, CounterSupport, Quietness, lock},
    counters::Counters,
    device::{dtype_name, parse_dtype},
    runner::{Counts, Measurement, Runner, line},
    suite::{CPU_ONLY, Data, NAMES},
};

const USAGE: &str = "\
usage: bench [run] [--filter TEXT]... [--quick] [--warmup N] [--reps N] [--device flex|vulkan|cuda]
             [--dtype D] [--lm-dtype D] [--local-encoder-dtype D] [--local-dit-dtype D]
             [--audio-decoder-dtype D] [--autotune-cache DIR] [--voice NAME] [--label TEXT]
             [--out DIR] [--max-load CPUS] [--noisy-ok]
       bench compare BASE.json NEW.json [--spread RUN.json RUN.json...]
       bench show RUN.json
       bench list

run times the benchmarks whose names contain any --filter (all by default), with VOXCPM2_DIR
naming the VoxCPM2 checkpoint, VOICES_DIR a converted voices directory and ASSETS_DIR the
radio's sounds, and writes
<commit>-<time>.json to --out (target/bench by default). --quick is 1 warmup and 3 timed
repetitions, for trying the harness, not for measuring. It refuses to start, with exit status 3,
while the 1-minute load average or the CPUs others keep busy exceed --max-load (default 1);
--noisy-ok times anyway and marks the run noisy.

Each part of the model runs in --dtype (f32, f16 or bf16), or in the dtype its own option names.
--device vulkan, in a build with the vulkan feature (nix run .#bench -- --features vulkan --
--device vulkan), times the Vulkan GPU wgpu prefers, refusing a CPU that presents itself as one,
by default in the dtypes the service serves in there, f16 but for the audio decoder's f32. Its
autotune keeps the kernels it chose in --autotune-cache (target/bench/autotune by default), and
the benchmarks that grow through KV lengths first run through them untimed; the audio VAE's
encoder is never timed there. --device cuda, in a build with the cuda feature, times the first
CUDA device alike. A GPU's times run to the device's having finished, and its
counters count only the host's share: launching the kernels and waiting for them.

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
    /// The dtype --dtype gave every part, or `f32` where none was given before parts had
    /// dtypes of their own.
    pub dtype: Option<String>,
    /// What each part ran in, absent from runs before parts had dtypes of their own.
    #[serde(default)]
    pub dtypes: Option<PartDtypes>,
    pub filters: Vec<String>,
    pub warmup: Option<usize>,
    pub reps: Option<usize>,
    pub quick: bool,
    pub voice: Option<String>,
    pub max_load: f64,
    pub noisy_ok: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartDtypes {
    pub lm: String,
    pub local_encoder: String,
    pub local_dit: String,
    pub audio_decoder: String,
}

impl From<Dtypes> for PartDtypes {
    fn from(dtypes: Dtypes) -> Self {
        Self {
            lm: dtype_name(dtypes.lm).into(),
            local_encoder: dtype_name(dtypes.local_encoder).into(),
            local_dit: dtype_name(dtypes.local_dit).into(),
            audio_decoder: dtype_name(dtypes.audio_decoder).into(),
        }
    }
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
        dtype: None,
        dtypes: None,
        filters: Vec::new(),
        warmup: None,
        reps: None,
        quick: false,
        voice: None,
        max_load: DEFAULT_MAX_LOAD,
        noisy_ok: false,
    };
    let (mut label, mut out, mut autotune_cache) = (None, None, None);
    let mut parts: [Option<String>; 4] = Default::default();
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
            "--dtype" => options.dtype = Some(value()?),
            "--lm-dtype" => parts[0] = Some(value()?),
            "--local-encoder-dtype" => parts[1] = Some(value()?),
            "--local-dit-dtype" => parts[2] = Some(value()?),
            "--audio-decoder-dtype" => parts[3] = Some(value()?),
            "--autotune-cache" => autotune_cache = Some(PathBuf::from(value()?)),
            "--device" => options.device = value()?,
            "--voice" => options.voice = Some(value()?),
            "--label" => label = Some(value()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--max-load" => options.max_load = number(arg, &value()?)?,
            "--noisy-ok" => options.noisy_ok = true,
            other => return Err(Failure::Usage(format!("unknown argument {other}"))),
        }
    }
    let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
    let autotune_cache = autotune_cache.unwrap_or_else(|| target.join("bench/autotune"));
    let (device, gpu) = device::open(&options.device, &autotune_cache).map_err(Failure::Usage)?;
    let dtypes = dtypes(gpu.is_some(), options.dtype.as_deref(), &parts).map_err(Failure::Usage)?;
    for (dtype, part) in [
        (dtypes.lm, "lm"),
        (dtypes.local_encoder, "local encoder"),
        (dtypes.local_dit, "local DiT"),
        (dtypes.audio_decoder, "audio decoder"),
    ] {
        if !device.supports_dtype(dtype) {
            let name = dtype_name(dtype);
            let reason = format!(
                "the {} device does not compute the {part} in {name}",
                options.device
            );
            return Err(Failure::Usage(reason));
        }
    }
    options.dtypes = Some(dtypes.into());
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
        dtypes,
        NAMES,
        filters,
        counts,
        options.max_load,
        counters,
    );
    if gpu.is_some() {
        runner.on_gpu(CPU_ONLY);
    }
    let dir = |var| {
        env::var_os(var)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let (checkpoint, voices, assets) = (dir("VOXCPM2_DIR"), dir("VOICES_DIR"), dir("ASSETS_DIR"));
    let data = Data {
        checkpoint: checkpoint.as_deref(),
        voices: voices.as_deref(),
        voice: options.voice.as_deref(),
        assets: assets.as_deref(),
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

    let mut conditions = Conditions::before(support, gpu);
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

    let out = out.unwrap_or_else(|| target.join("bench"));
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

/// What each part runs in: its own option's dtype, else --dtype's, else f32 on the CPU, and on a
/// GPU what the service serves in there, f16 but for the audio decoder, whose first transposed
/// convolution sums in f16 there.
fn dtypes(gpu: bool, all: Option<&str>, parts: &[Option<String>; 4]) -> Result<Dtypes, String> {
    let (f32, f16) = (FloatDType::F32, FloatDType::F16);
    let mut dtypes = match all {
        Some(name) => {
            let dtype = parse_dtype(name)?;
            Dtypes {
                lm: dtype,
                local_encoder: dtype,
                local_dit: dtype,
                audio_decoder: dtype,
            }
        }
        None if gpu => Dtypes {
            lm: f16,
            local_encoder: f16,
            local_dit: f16,
            audio_decoder: f32,
        },
        None => Dtypes {
            lm: f32,
            local_encoder: f32,
            local_dit: f32,
            audio_decoder: f32,
        },
    };
    let fields = [
        &mut dtypes.lm,
        &mut dtypes.local_encoder,
        &mut dtypes.local_dit,
        &mut dtypes.audio_decoder,
    ];
    for (field, part) in fields.into_iter().zip(parts) {
        if let Some(name) = part {
            *field = parse_dtype(name)?;
        }
    }
    Ok(dtypes)
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
    let dtypes = match &run.options.dtypes {
        Some(d) => format!(
            "lm {}, local encoder {}, local DiT {}, audio decoder {}",
            d.lm, d.local_encoder, d.local_dit, d.audio_decoder
        ),
        None => run.options.dtype.clone().unwrap_or_else(|| "f32".into()),
    };
    out += &format!(
        "device {} in {dtypes}; threads: {} available, {} at the end; load {} before, {} after\n",
        run.options.device,
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
            "counters per repetition{}: [{}], refused: [{}]\n",
            if c.gpu.is_some() {
                ", the host's share only"
            } else {
                ""
            },
            counters.recorded.join(", "),
            refused.join(", ")
        );
    }
    if let Some(gpu) = &c.gpu {
        out += &format!(
            "GPU: {} at {}, {} {}, {}; autotune cache {} ({} bytes before)\n",
            gpu.name,
            gpu.pci_bus_id,
            gpu.driver,
            gpu.driver_info,
            gpu.fingerprint,
            gpu.autotune_cache,
            gpu.autotune_cache_bytes_before
        );
        if let Some(cuda) = &gpu.cuda {
            out += &format!(
                "CUDA {} by the driver, bound for {}, NVRTC {}, kernels compiled by {}\n",
                cuda.driver,
                cuda.bindings,
                cuda.nvrtc.as_deref().unwrap_or("absent"),
                cuda.compiler
            );
        }
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
