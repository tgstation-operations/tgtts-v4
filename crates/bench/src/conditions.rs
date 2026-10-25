//! What a run was measured under, and whether the machine was quiet enough to measure on.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io,
    path::Path,
    process::Command,
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};

/// Taken for the whole of a run, so that two runs never time each other.
pub const BENCH_LOCK: &str = "/tmp/tts-rs-bench.lock";
/// Taken by whatever loads whole model stacks, tests included, while it holds them.
pub const BIGMEM_LOCK: &str = "/tmp/tts-rs-bigmem.lock";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conditions {
    pub commit: Option<String>,
    pub dirty: Option<bool>,
    /// `git status --porcelain`, for a dirty tree.
    pub dirty_files: Vec<String>,
    pub build: Build,
    pub host: Host,
    pub threads: Threads,
    pub loadavg_before: String,
    pub loadavg_after: String,
    pub mem_available_mib_before: Option<u64>,
    pub mem_available_mib_after: Option<u64>,
    /// The process's peak resident set at the end of the run.
    pub peak_rss_mib: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Build {
    pub rustc: String,
    pub crates: String,
    pub profile: String,
    pub opt_level: String,
    pub debug: String,
    pub rustflags: String,
    pub target: String,
    pub target_features: String,
    /// What `nix run .#bench` passed to cargo, and the features cargo resolved for burn's CPU
    /// backend with them; unknown when the binary was run some other way.
    pub cargo_args: Option<String>,
    pub resolved_features: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Host {
    pub hostname: String,
    pub kernel: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub mem_total_mib: Option<u64>,
    /// Each sysfs setting as its distinct values across the CPUs, with how many have each.
    pub governor: String,
    pub epp: String,
    pub platform_profile: String,
    pub amd_pstate: String,
    pub boost: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Threads {
    pub available_parallelism: usize,
    pub rayon_num_threads: Option<String>,
    /// The process's threads when the run ended: more than one shows a pool spun up.
    pub process_threads_at_end: Option<u64>,
}

impl Conditions {
    /// What can be read before the run; the `after` fields are filled by
    /// [`finish`](Self::finish).
    pub fn before() -> Self {
        let (commit, dirty, dirty_files) = git();
        Self {
            commit,
            dirty,
            dirty_files,
            build: Build {
                rustc: env!("BENCH_RUSTC").into(),
                crates: env!("BENCH_CRATES").into(),
                profile: env!("BENCH_PROFILE").into(),
                opt_level: env!("BENCH_OPT_LEVEL").into(),
                debug: env!("BENCH_DEBUG").into(),
                rustflags: env!("BENCH_RUSTFLAGS").into(),
                target: env!("BENCH_TARGET").into(),
                target_features: env!("BENCH_TARGET_FEATURES").into(),
                cargo_args: std::env::var("BENCH_CARGO_ARGS").ok(),
                resolved_features: std::env::var("BENCH_RESOLVED_FEATURES").ok(),
            },
            host: Host {
                hostname: read("/proc/sys/kernel/hostname"),
                kernel: read("/proc/sys/kernel/osrelease"),
                cpu_model: cpu_model(),
                logical_cpus: logical_cpus(),
                mem_total_mib: meminfo_mib("MemTotal"),
                governor: per_cpu("cpufreq/scaling_governor"),
                epp: per_cpu("cpufreq/energy_performance_preference"),
                platform_profile: read("/sys/firmware/acpi/platform_profile"),
                amd_pstate: read("/sys/devices/system/cpu/amd_pstate/status"),
                boost: read("/sys/devices/system/cpu/cpufreq/boost"),
            },
            threads: Threads {
                available_parallelism: thread::available_parallelism().map_or(0, |n| n.get()),
                rayon_num_threads: std::env::var("RAYON_NUM_THREADS").ok(),
                process_threads_at_end: None,
            },
            loadavg_before: read("/proc/loadavg"),
            loadavg_after: String::new(),
            mem_available_mib_before: meminfo_mib("MemAvailable"),
            mem_available_mib_after: None,
            peak_rss_mib: None,
        }
    }

    pub fn finish(&mut self) {
        self.loadavg_after = read("/proc/loadavg");
        self.mem_available_mib_after = meminfo_mib("MemAvailable");
        let status = read("/proc/self/status");
        let field = |key: &str| {
            status
                .lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        };
        self.threads.process_threads_at_end = field("Threads:");
        self.peak_rss_mib = field("VmHWM:").map(|kib| kib / 1024);
    }
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "unreadable".into())
}

fn git() -> (Option<String>, Option<bool>, Vec<String>) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let run = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let commit = run(&["rev-parse", "HEAD"]).map(|s| s.trim().to_owned());
    let status = run(&["status", "--porcelain", "--untracked-files=normal"]);
    let files: Vec<String> = status
        .iter()
        .flat_map(|s| s.lines().map(str::to_owned))
        .collect();
    (commit, status.map(|_| !files.is_empty()), files)
}

fn cpu_model() -> String {
    let info = read("/proc/cpuinfo");
    let model = info.lines().find_map(|l| l.strip_prefix("model name"));
    model
        .and_then(|rest| rest.split_once(':'))
        .map_or_else(|| "unknown".into(), |(_, name)| name.trim().to_owned())
}

fn logical_cpus() -> usize {
    let stat = read("/proc/stat");
    stat.lines()
        .filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(u8::is_ascii_digit))
        .count()
}

fn meminfo_mib(key: &str) -> Option<u64> {
    let info = read("/proc/meminfo");
    let line = info
        .lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(':'))?;
    let kib: u64 = line.split_whitespace().next()?.parse().ok()?;
    Some(kib / 1024)
}

/// The distinct values of `/sys/devices/system/cpu/cpu<N>/<file>` with their counts, as
/// "performance x32".
fn per_cpu(file: &str) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for cpu in 0..logical_cpus() {
        let value = read(format!("/sys/devices/system/cpu/cpu{cpu}/{file}"));
        *counts.entry(value).or_default() += 1;
    }
    let values: Vec<String> = counts.iter().map(|(v, n)| format!("{v} x{n}")).collect();
    values.join(", ")
}

/// An exclusive flock(2) on `path`, as flock(1) takes it, held until dropped, and whether
/// another holder kept it waiting.
pub fn lock(path: &str) -> io::Result<(File, bool)> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let waited = file.try_lock().is_err();
    if waited {
        eprintln!("bench: waiting for {path}");
        file.lock()?;
    }
    Ok((file, waited))
}

/// Jiffies the whole machine spent busy and in all, and this process spent on CPU.
#[derive(Clone, Copy, Debug)]
pub struct CpuTimes {
    busy: u64,
    total: u64,
    own: u64,
}

impl CpuTimes {
    pub fn now() -> Self {
        let stat = read("/proc/stat");
        let fields: Vec<u64> = stat
            .lines()
            .next()
            .and_then(|l| l.strip_prefix("cpu "))
            .map(|l| {
                l.split_whitespace()
                    .filter_map(|v| v.parse().ok())
                    .collect()
            })
            .unwrap_or_default();
        // user nice system idle iowait irq softirq steal; guest time is already in user.
        let at = |i: usize| fields.get(i).copied().unwrap_or(0);
        let busy = at(0) + at(1) + at(2) + at(5) + at(6) + at(7);
        let total = busy + at(3) + at(4);
        // utime and stime are the 14th and 15th fields; the name, the 2nd, can hold spaces.
        let own_stat = read("/proc/self/stat");
        let after_name: Vec<&str> = own_stat
            .rsplit_once(')')
            .map(|(_, rest)| rest.split_whitespace().collect())
            .unwrap_or_default();
        let field = |n: usize| -> u64 {
            let value = after_name.get(n - 3);
            value.and_then(|v| v.parse().ok()).unwrap_or(0)
        };
        let own = field(14) + field(15);
        Self { busy, total, own }
    }

    /// How many CPUs' worth of work other processes did between `self` and `later`.
    pub fn foreign_cores(&self, later: &Self) -> f64 {
        let total = later.total.saturating_sub(self.total);
        if total == 0 {
            return 0.0;
        }
        let busy = later.busy.saturating_sub(self.busy);
        let foreign = busy.saturating_sub(later.own.saturating_sub(self.own));
        foreign as f64 / total as f64 * logical_cpus() as f64
    }
}

/// The system-wide signals of other work, which a process that sees only its own processes
/// still has.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quietness {
    /// The 1-minute load average.
    pub load1: f64,
    /// CPUs' worth of work others did over a two-second sample.
    pub busy_cores: f64,
    pub limit: f64,
}

impl Quietness {
    pub fn sample(limit: f64) -> Self {
        let before = CpuTimes::now();
        thread::sleep(Duration::from_secs(2));
        let busy_cores = before.foreign_cores(&CpuTimes::now());
        let loadavg = read("/proc/loadavg");
        let load1 = loadavg
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::INFINITY);
        Self {
            load1,
            busy_cores,
            limit,
        }
    }

    pub fn is_quiet(&self) -> bool {
        self.load1 <= self.limit && self.busy_cores <= self.limit
    }
}
