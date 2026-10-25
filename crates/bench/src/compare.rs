//! Which benchmarks two runs differ on beyond their noise.
//!
//! A change counts when it clears both kinds of noise a run has. Within a run, the
//! repetitions scatter: the two runs' interquartile ranges have to be disjoint. Between runs
//! of the same code, the medians themselves move (frequency, memory placement, what else the
//! machine did): the change has to exceed twice the run-to-run spread, the range of the
//! medians of back-to-back runs of one commit over their median. Twice, because the spread
//! comes from a few runs and a single pair understates it about as often as it overstates it.
//!
//! Instruction counts answer another question: whether the code path changed. Other work on
//! the machine barely moves them, so a change beyond their own noise is the code's, however
//! noisy the timings; and an unchanged count with a changed time points at memory, threads or
//! the machine rather than the code.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    Failure, Run, read_run,
    runner::Measurement,
    stats::{Summary, duration},
};

/// The least relative change in instructions per repetition that marks a changed code path.
/// Single-threaded, two runs of one commit differed by at most 0.023 % on the model
/// benchmarks, but by 0.55 % on the audio VAE's loads and 2.4 % on the microsecond voice
/// reads, and a thread pool's spinning moved threaded counts by up to 5.5 %: so a change has to
/// clear twice the instruction spread of the --spread runs as well, as the timings do.
const CODE_CHANGE: f64 = 0.005;

pub fn main(args: &[String]) -> Result<(), Failure> {
    let (runs, spread) = match args.iter().position(|a| a == "--spread") {
        Some(at) => (&args[..at], &args[at + 1..]),
        None => (args, &[][..]),
    };
    let [base, new] = runs else {
        return Err(Failure::Usage(
            "compare takes a base run and a new run".into(),
        ));
    };
    if spread.len() == 1 {
        return Err(Failure::Usage(
            "--spread takes at least two runs of one commit".into(),
        ));
    }
    let spread_runs = spread
        .iter()
        .map(|path| read_run(path))
        .collect::<Result<Vec<_>, _>>()?;
    print!(
        "{}",
        report(&read_run(base)?, &read_run(new)?, &spread_runs)
    );
    Ok(())
}

/// The relative range of `value` of each benchmark over `runs`, where at least two hold it.
fn spreads(runs: &[Run], value: impl Fn(&Measurement) -> Option<f64>) -> BTreeMap<&str, f64> {
    let mut values = BTreeMap::<&str, Vec<f64>>::new();
    for m in runs.iter().flat_map(|run| &run.benchmarks) {
        if let Some(value) = value(m) {
            values.entry(&m.name).or_default().push(value);
        }
    }
    values
        .into_iter()
        .filter(|(_, values)| values.len() >= 2)
        .map(|(name, values)| {
            let summary = Summary::of(&values);
            (name, (summary.max - summary.min) / summary.median)
        })
        .collect()
}

fn median(m: &Measurement) -> Option<f64> {
    Some(m.summary.median)
}

fn instructions(m: &Measurement) -> Option<f64> {
    m.counters.get("instructions").copied()
}

fn code_changed(change: f64, spread: Option<f64>) -> bool {
    change.abs() > CODE_CHANGE.max(2.0 * spread.unwrap_or(0.0))
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Faster,
    Slower,
    WithinNoise,
}

fn verdict(change: f64, disjoint: bool, spread: Option<f64>) -> Verdict {
    let clears_spread = spread.is_none_or(|spread| change.abs() > 2.0 * spread);
    match (disjoint && clears_spread, change < 0.0) {
        (false, _) => Verdict::WithinNoise,
        (true, true) => Verdict::Faster,
        (true, false) => Verdict::Slower,
    }
}

fn report(base: &Run, new: &Run, spread_runs: &[Run]) -> String {
    let instruction_spreads = spreads(spread_runs, instructions);
    let spreads = spreads(spread_runs, median);
    let mut out = format!(
        "{:<36} {:>12} {:>12} {:>8} {:>8} {:>9}  verdict\n",
        "benchmark", "base", "new", "change", "spread", "instr"
    );
    for b in &base.benchmarks {
        let Some(n) = new.benchmarks.iter().find(|n| n.name == b.name) else {
            out += &format!("{:<36} only in the base run\n", b.name);
            continue;
        };
        let (b_s, n_s) = (&b.summary, &n.summary);
        let change = n_s.median / b_s.median - 1.0;
        let disjoint = b_s.q3 < n_s.q1 || n_s.q3 < b_s.q1;
        let spread = spreads.get(b.name.as_str()).copied();
        let verdict = match verdict(change, disjoint, spread) {
            Verdict::Faster => "faster",
            Verdict::Slower => "slower",
            Verdict::WithinNoise if !disjoint => "within noise (IQRs overlap)",
            Verdict::WithinNoise => "within noise (inside 2x spread)",
        };
        let instr = match (instructions(b), instructions(n)) {
            (Some(b_count), Some(n_count)) => {
                let change = n_count / b_count - 1.0;
                let spread = instruction_spreads.get(b.name.as_str()).copied();
                let mark = if code_changed(change, spread) {
                    "!"
                } else {
                    " "
                };
                format!("{:+.2}%{mark}", change * 100.0)
            }
            _ => "-".into(),
        };
        out += &format!(
            "{:<36} {:>12} {:>12} {:>+7.1}% {:>8} {instr:>9}  {verdict}\n",
            b.name,
            duration(b_s.median),
            duration(n_s.median),
            change * 100.0,
            spread.map_or("unknown".into(), |s| format!("{:.1}%", s * 100.0)),
        );
    }
    for n in &new.benchmarks {
        if !base.benchmarks.iter().any(|b| b.name == n.name) {
            out += &format!("{:<36} only in the new run\n", n.name);
        }
    }
    if spreads.is_empty() {
        out += "\nno --spread runs: verdicts rest on the interquartile ranges alone\n";
    }
    out += &format!(
        "instr: instructions per repetition; ! marks a changed code path, a change beyond {:.1}% \
         and beyond twice the --spread runs' instruction spread\n",
        CODE_CHANGE * 100.0
    );
    let threaded = |run: &Run| {
        run.conditions
            .threads
            .process_threads_at_end
            .is_some_and(|n| n > 1)
    };
    if threaded(base) || threaded(new) {
        out += "a thread pool ran: its spinning counts as instructions and varies with timing, \
                so read instr only between runs threaded alike\n";
    }
    for (run, which) in [(base, "base"), (new, "new")] {
        if let Some(first) = run.noisy.first() {
            out += &format!(
                "\nthe {which} run is marked noisy for {} reasons, the first: {first}\n",
                run.noisy.len()
            );
        }
        if run.options.quick {
            out += &format!("\nthe {which} run is a quick one, not a measurement\n");
        }
    }
    let differences = differences(base, new);
    if !differences.is_empty() {
        out += "\nconditions that differ:\n";
        for (key, (b, n)) in differences {
            out += &format!("  {key}: {b} -> {n}\n");
        }
    }
    out
}

/// The conditions and options that differ between the runs, but those that move with every
/// run and are read for noise rather than for comparability.
fn differences(base: &Run, new: &Run) -> BTreeMap<String, (String, String)> {
    const VOLATILE: &[&str] = &[
        "conditions.loadavg_before",
        "conditions.loadavg_after",
        "conditions.mem_available_mib_before",
        "conditions.mem_available_mib_after",
        "conditions.peak_rss_mib",
        "conditions.dirty_files",
    ];
    let flat = |run: &Run| {
        let mut flat = BTreeMap::new();
        let value = serde_json::json!({
            "conditions": run.conditions,
            "options": run.options,
        });
        flatten(String::new(), &value, &mut flat);
        flat
    };
    let (b, n) = (flat(base), flat(new));
    let keys: std::collections::BTreeSet<&String> = b.keys().chain(n.keys()).collect();
    keys.into_iter()
        .filter(|key| !VOLATILE.contains(&key.as_str()))
        .filter_map(|key| {
            let missing = || "(none)".to_owned();
            let (bv, nv) = (b.get(key).cloned(), n.get(key).cloned());
            (bv != nv).then(|| {
                (
                    key.clone(),
                    (bv.unwrap_or_else(missing), nv.unwrap_or_else(missing)),
                )
            })
        })
        .collect()
}

fn flatten(prefix: String, value: &Value, out: &mut BTreeMap<String, String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(key, value, out);
            }
        }
        other => {
            out.insert(prefix, other.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_change_must_clear_the_quartiles_and_twice_the_spread() {
        assert_eq!(verdict(-0.10, true, Some(0.04)), Verdict::Faster);
        assert_eq!(verdict(0.10, true, Some(0.04)), Verdict::Slower);
        assert_eq!(verdict(-0.07, true, Some(0.04)), Verdict::WithinNoise);
        assert_eq!(verdict(-0.50, false, Some(0.01)), Verdict::WithinNoise);
        assert_eq!(verdict(-0.01, true, None), Verdict::Faster);
    }

    #[test]
    fn a_code_change_must_clear_the_floor_and_twice_the_count_spread() {
        assert!(code_changed(-0.006, None));
        assert!(!code_changed(0.004, None));
        assert!(!code_changed(0.08, Some(0.05)));
        assert!(code_changed(0.12, Some(0.05)));
    }
}
