//! Records how the binary was built, which a run reports with its timings: cargo tells a build
//! script what it tells no binary.

use std::{env, fs, path::Path, process::Command};

fn main() {
    let lock = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../Cargo.lock");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", lock.display());

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_default();
    let line = |key: &str| {
        version
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or("unknown")
            .trim()
            .to_owned()
    };
    let rustc = format!(
        "{} (LLVM {})",
        version.lines().next().unwrap_or("unknown"),
        line("LLVM version:")
    );
    set("BENCH_RUSTC", &rustc);

    let lock = fs::read_to_string(lock).unwrap_or_default();
    let versions: Vec<String> = ["burn", "burn-flex", "gemm", "pulp", "cubecl", "wgpu"]
        .iter()
        .map(|name| format!("{name} {}", locked_versions(&lock, name).join("/")))
        .collect();
    set("BENCH_CRATES", &versions.join(", "));

    for (var, from) in [
        ("BENCH_PROFILE", "PROFILE"),
        ("BENCH_OPT_LEVEL", "OPT_LEVEL"),
        ("BENCH_DEBUG", "DEBUG"),
        ("BENCH_TARGET", "TARGET"),
        ("BENCH_TARGET_FEATURES", "CARGO_CFG_TARGET_FEATURE"),
    ] {
        set(var, &env::var(from).unwrap_or_default());
    }
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    set("BENCH_RUSTFLAGS", &rustflags.replace('\x1f', " "));
}

fn set(var: &str, value: &str) {
    println!("cargo::rustc-env={var}={value}");
}

/// Every version of `name` that Cargo.lock holds.
fn locked_versions(lock: &str, name: &str) -> Vec<String> {
    let mut lines = lock.lines();
    let mut versions = Vec::new();
    while let Some(line) = lines.next() {
        if line == format!("name = \"{name}\"")
            && let Some(version) = lines.next().and_then(|l| l.strip_prefix("version = "))
        {
            versions.push(version.trim_matches('"').to_owned());
        }
    }
    versions
}
