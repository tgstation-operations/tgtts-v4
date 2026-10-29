//! Prints what the Vulkan devices report about themselves, which of them the runtime would choose,
//! and why the others were passed over.
//!
//! ```text
//! cargo run -p vulkan --example properties [-- --all]
//! ```

use std::process::ExitCode;

use argh::FromArgs;
use vulkan::{Instance, select};

/// Prints what the Vulkan devices report about themselves, which of them the runtime would
/// choose, and why the others were passed over.
#[derive(FromArgs)]
struct Args {
    /// report every device in full, not only the chosen one
    #[argh(switch)]
    all: bool,
}

fn main() -> ExitCode {
    let Args { all } = argh::from_env();
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
