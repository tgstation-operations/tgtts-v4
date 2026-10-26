//! Prints what the Vulkan devices report about themselves, which of them the runtime would choose,
//! and why the others were passed over.
//!
//! ```text
//! cargo run -p vulkan --example properties [-- --all]
//! ```
//!
//! By default only the chosen device is reported in full; `--all` reports every device.

use std::process::ExitCode;

use vulkan::{Instance, select};

fn main() -> ExitCode {
    let all = std::env::args().skip(1).any(|a| a == "--all");
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let devices = match instance.devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let candidates: Vec<_> = devices.iter().map(|(_, r)| r.candidate()).collect();
    let chosen = select::choose(&candidates);

    println!("{} device(s) enumerated", devices.len());
    for (i, c) in candidates.iter().enumerate() {
        let verdict = match (&chosen, c.rejection()) {
            (Ok(n), _) if *n == i => "chosen".to_string(),
            (_, Some(why)) => format!("passed over -- {why}"),
            (_, None) => "qualifies, not preferred".to_string(),
        };
        println!("  {i}  {}  {verdict}", c.name);
    }
    for (i, (_, report)) in devices.iter().enumerate() {
        if all || chosen.as_ref().is_ok_and(|n| *n == i) {
            println!("\n== device {i}\n{report}");
        }
    }
    if chosen.is_err() {
        eprintln!("no device qualifies");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
