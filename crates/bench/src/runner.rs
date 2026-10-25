use std::{collections::BTreeMap, time::Instant};

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    conditions::{CpuTimes, Quietness, lock},
    counters::Counters,
    stats::{Summary, duration},
};

/// The shortest timed stretch whose share of other processes' work is judged: /proc/stat
/// counts in 10 ms ticks, so over a few milliseconds one tick on one CPU reads as several
/// CPUs busy.
const FOREIGN_WINDOW_SECONDS: f64 = 1.0;

/// One benchmark's timings and what they mean.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    /// The shapes and settings it ran at.
    pub params: BTreeMap<String, Value>,
    pub warmup: usize,
    /// Each timed repetition, in seconds.
    pub samples: Vec<f64>,
    pub summary: Summary,
    /// Work done per second at the median.
    pub throughput: Option<Throughput>,
    /// Seconds of compute per second of audio at the median: below 1 is faster than real time.
    pub rtf: Option<f64>,
    /// CPUs' worth of work other processes did while it was timed, if that took long enough
    /// to tell.
    pub foreign_cores: Option<f64>,
    /// Hardware events per repetition, its untimed setup included, of every thread, by perf's
    /// names.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counters: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Throughput {
    pub value: f64,
    pub unit: String,
}

/// A benchmark to run: its name, how many times, and the work one repetition does.
pub struct Spec {
    name: String,
    warmup: usize,
    reps: usize,
    work: Option<(f64, &'static str)>,
    audio_seconds: Option<f64>,
    params: BTreeMap<String, Value>,
}

impl Spec {
    pub fn new(name: &str, warmup: usize, reps: usize) -> Self {
        Self {
            name: name.into(),
            warmup,
            reps,
            work: None,
            audio_seconds: None,
            params: BTreeMap::new(),
        }
    }

    /// One repetition does `count` of `unit`, reported per second.
    pub fn work(mut self, count: f64, unit: &'static str) -> Self {
        self.work = Some((count, unit));
        self
    }

    /// One repetition yields `seconds` of audio.
    pub fn audio(mut self, seconds: f64) -> Self {
        self.audio_seconds = Some(seconds);
        self
    }

    pub fn param(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

pub struct Runner {
    pub device: Device,
    pub dtype: FloatDType,
    names: &'static [&'static str],
    filters: Vec<String>,
    warmup: Option<usize>,
    reps: Option<usize>,
    limit: f64,
    counters: Option<Counters>,
    pub measurements: Vec<Measurement>,
    /// Why the run's timings may not be trusted, if they may not.
    pub noisy: Vec<String>,
}

pub struct Counts {
    pub warmup: Option<usize>,
    pub reps: Option<usize>,
}

impl Runner {
    pub fn new(
        device: Device,
        dtype: FloatDType,
        names: &'static [&'static str],
        filters: Vec<String>,
        counts: Counts,
        limit: f64,
        counters: Option<Counters>,
    ) -> Self {
        Self {
            device,
            dtype,
            names,
            filters,
            warmup: counts.warmup,
            reps: counts.reps,
            limit,
            counters,
            measurements: Vec::new(),
            noisy: Vec::new(),
        }
    }

    pub fn selected(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| name.contains(f.as_str()))
    }

    pub fn any_selected(&self, prefix: &str) -> bool {
        let mut names = self.names.iter().filter(|n| n.starts_with(prefix));
        names.any(|n| self.selected(n))
    }

    /// The warmup and timed repetitions `spec` will run.
    pub fn counts(&self, spec: &Spec) -> (usize, usize) {
        (
            self.warmup.unwrap_or(spec.warmup),
            self.reps.unwrap_or(spec.reps).max(1),
        )
    }

    /// Takes the exclusive lock `path`, and if another holder kept it waiting, checks again that
    /// the machine is quiet: what held it may still be winding down.
    pub fn lock(&mut self, path: &str) -> Result<std::fs::File, String> {
        let (file, waited) = lock(path).map_err(|e| format!("{path}: {e}"))?;
        if waited {
            let quiet = Quietness::sample(self.limit);
            if !quiet.is_quiet() {
                self.noisy.push(format!(
                    "after waiting for {path}: load {:.2}, {:.2} CPUs busy, over {:.2}",
                    quiet.load1, quiet.busy_cores, quiet.limit
                ));
            }
        }
        Ok(file)
    }

    /// Runs `spec` if the filters select it: `setup` before each repetition, untimed, then
    /// `run` on what it made, timed up to the device's having finished. Gives what the last
    /// timed `run` returned, which is dropped only after the device has synchronized.
    ///
    /// # Panics
    ///
    /// If `spec` is not among the names the runner was given, which `list` and the group
    /// filters read.
    pub fn measure<S, T>(
        &mut self,
        spec: Spec,
        mut setup: impl FnMut() -> S,
        mut run: impl FnMut(S) -> T,
    ) -> Option<T> {
        assert!(
            self.names.contains(&spec.name.as_str()),
            "{} is not in the list of benchmarks",
            spec.name
        );
        if !self.selected(&spec.name) {
            return None;
        }
        let (warmup, reps) = self.counts(&spec);
        eprintln!("bench: {} ({warmup} warmup, {reps} timed)", spec.name);
        for _ in 0..warmup {
            let out = run(setup());
            self.sync();
            drop(out);
        }
        let before = CpuTimes::now();
        let counted = self.counters.as_ref().map(Counters::read);
        let mut samples = Vec::with_capacity(reps);
        let mut last = None;
        for _ in 0..reps {
            let state = setup();
            let start = Instant::now();
            let out = run(state);
            self.sync();
            samples.push(start.elapsed().as_secs_f64());
            last = Some(out);
        }
        let counters = match (&self.counters, counted) {
            (Some(counters), Some(before)) => before.per_rep(&counters.read(), reps),
            _ => BTreeMap::new(),
        };
        let timed: f64 = samples.iter().sum();
        let foreign_cores =
            (timed >= FOREIGN_WINDOW_SECONDS).then(|| before.foreign_cores(&CpuTimes::now()));
        let summary = Summary::of(&samples);
        if let Some(foreign_cores) = foreign_cores.filter(|&cores| cores > self.limit) {
            self.noisy.push(format!(
                "{}: other processes kept {foreign_cores:.2} CPUs busy, over {:.2}",
                spec.name, self.limit
            ));
        }
        let throughput = spec.work.map(|(count, unit)| Throughput {
            value: count / summary.median,
            unit: format!("{unit}/s"),
        });
        let measurement = Measurement {
            name: spec.name,
            params: spec.params,
            warmup,
            rtf: spec.audio_seconds.map(|audio| summary.median / audio),
            samples,
            summary,
            throughput,
            foreign_cores,
            counters,
        };
        eprintln!("bench:   {}", line(&measurement));
        self.measurements.push(measurement);
        last
    }

    /// Gives the measurement `name` a throughput of `count` of `unit` per repetition, for work
    /// that is known only once it has run.
    pub fn set_work(&mut self, name: &str, count: f64, unit: &str) {
        let measurement = self.measurements.iter_mut().rev().find(|m| m.name == name);
        if let Some(m) = measurement {
            m.throughput = Some(Throughput {
                value: count / m.summary.median,
                unit: format!("{unit}/s"),
            });
        }
    }

    fn sync(&self) {
        self.device
            .sync()
            .expect("the device failed to run a benchmark");
    }

    /// A tensor of `shape` in the runner's dtype, uniform in [-1, 1) from `seed`, the same on
    /// every run.
    pub fn seeded<const D: usize>(&self, shape: [usize; D], seed: u64) -> Tensor<D> {
        let values = uniform(seed, shape.iter().product());
        Tensor::<D>::from_data(TensorData::new(values, shape), &self.device).cast(self.dtype)
    }
}

/// splitmix64's stream: the same draws from the same seed on every run.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Floats uniform in [-1, 1) from `seed`.
pub fn uniform(seed: u64, len: usize) -> Vec<f32> {
    let mut draws = SplitMix(seed);
    (0..len)
        .map(|_| (draws.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        .collect()
}

/// A measurement on one line: median, interquartile range, extremes and what it achieved.
pub fn line(m: &Measurement) -> String {
    let s = &m.summary;
    let mut out = format!(
        "{:<36} {:>12}  IQR {:>11}  min {:>11}  max {:>11}  n {:>3}",
        m.name,
        duration(s.median),
        duration(s.iqr),
        duration(s.min),
        duration(s.max),
        m.samples.len()
    );
    if let Some(t) = &m.throughput {
        out += &format!("  {} {}", significant(t.value), t.unit);
    }
    if let Some(rtf) = m.rtf {
        out += &format!("  RTF {rtf:.3}");
    }
    if let Some(instructions) = m.counters.get("instructions") {
        out += &format!("  {} instr", significant(*instructions));
    }
    out
}

fn significant(value: f64) -> String {
    if value >= 1e5 {
        format!("{value:.3e}")
    } else {
        format!("{value:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_stays_in_range_and_repeats() {
        let values = uniform(7, 10_000);
        assert!(values.iter().all(|v| (-1.0..1.0).contains(v)));
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert_eq!(values, uniform(7, 10_000));
        assert_ne!(values, uniform(8, 10_000));
    }
}
