//! The service on the rt device, tts-rs's own runtime on a Vulkan GPU, on the real model, voices
//! and assets: the device started, speaking and shut down, which leaves nothing of it running; a
//! message for everyone, as the game asks for it, the model warmed up as the service warms it; a
//! request in a voice the prompt cache keeps, which comes out as it did the first time; lines
//! asked for at once, which come out as each does alone; and voxcpm's golden line in its voice
//! spoken from a seed as the burn CPU device speaks it from that seed, held to burn's own f16
//! there. Built with the `rt` feature, and run with `VOXCPM2_DIR`,
//! `VOICES_DIR` and `ASSETS_DIR` set and `GLSL_KERNELS` and `VULKAN_KERNELS` naming the compiled
//! kernels, as `nix run .#device-tests` sets them, one test at a time; under the validation layer,
//! as it runs, a request fails on anything the layer reports.
//!
//! The CPU's speech is kept as JSON under the directory `ORACLE_DIR` names, when it is set, and
//! read from there by later runs: `keeps_the_cpu_speech` makes it alone, touching no GPU.
#![cfg(feature = "rt")]

mod common;

use std::{fs, path::PathBuf, thread};

use burn::tensor::Device;
use common::real;
use serde::{Deserialize, Serialize};
use tgtts::{
    device::Kind,
    model::{PROMPT_CACHE_BYTES, VoxCpm},
    rt::Rt,
    synth::{Speech, Synthesizer},
};
use voices::Voices;
use voxcpm::Params;

/// voxcpm's golden line and voice: run free from the seed, the generator stops after its 24th
/// patch, where voxcpm in f32 on the CPU does.
const LINE: &str = "Security, report to the bridge. ";
const VOICE: &str = "Man (Infected)";
const SEED: u64 = 0;

fn rt(params: &Params) -> Rt {
    Rt::load(
        &real::dir("VOXCPM2_DIR"),
        &real::dir("GLSL_KERNELS"),
        &real::dir("VULKAN_KERNELS"),
        params,
        PROMPT_CACHE_BYTES,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn voices() -> Voices {
    Voices::open(real::dir("VOICES_DIR")).unwrap()
}

/// The names of the process's threads.
fn threads() -> Vec<String> {
    let tasks = fs::read_dir("/proc/self/task").unwrap();
    let comm = |task: fs::DirEntry| fs::read_to_string(task.path().join("comm")).ok();
    tasks
        .filter_map(|task| comm(task.unwrap()))
        .map(|name| name.trim_end().to_owned())
        .collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn starts_speaks_and_shuts_down_leaving_nothing_running() {
    let rt = rt(&real::params());
    assert!(threads().iter().any(|name| name == "rt"));
    let voice = voices().latent(VOICE).unwrap();
    rt.synthesize(LINE, &voice, SEED).unwrap();
    // The validation layer, where it is on, has said nothing of the GPU's work or of its
    // objects' destruction.
    rt.shut_down().unwrap_or_else(|e| panic!("{e}"));
    let left = threads();
    assert!(!left.iter().any(|name| name == "rt"), "{left:?}");
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, ASSETS_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn serves_a_message_on_the_rt_device() {
    real::serves_a_message(rt(&real::params()), |rt, voices| {
        rt.warm_up(voices).unwrap();
    });
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn a_voice_kept_speaks_as_it_did_the_first_time() {
    let rt = rt(&real::params());
    let voices = voices();
    let names: Vec<&str> = voices.names().take(2).collect();
    let [voice, other] = [names[0], names[1]].map(|name| voices.latent(name).unwrap());
    let speak = |voice| rt.synthesize("Clown in the armory.", voice, 1).unwrap();
    let first = speak(&voice);
    // Another voice between, so that the caches hold its keys and values, not the first's.
    speak(&other);
    let again = speak(&voice);
    let [start, kept] = [&first, &again].map(|speech| speech.start.unwrap());
    assert_eq!(first.patches, again.patches);
    assert_eq!(start.prompt, kept.prompt);
    assert_eq!(start.prefilled, start.prompt);
    // All but the voice spoken again, after the text, came from what was kept of the voice.
    let patches = voice.frames.len() / 4;
    assert_eq!(kept.prefilled, patches);
    let bits =
        |speech: &Speech| -> Vec<u32> { speech.samples.iter().map(|s| s.to_bits()).collect() };
    assert!(
        bits(&first) == bits(&again),
        "the speech differs once the voice is kept"
    );
    eprintln!(
        "a prompt of {} positions, {} run the second time: {} patches alike, bit for bit",
        start.prompt, kept.prefilled, first.patches
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn lines_asked_for_at_once_come_out_as_each_does_alone() {
    let rt = rt(&real::params());
    let voices = voices();
    let names: Vec<&str> = voices.names().take(3).collect();
    let latents: Vec<_> = names
        .iter()
        .map(|name| voices.latent(name).unwrap())
        .collect();
    // The same line in the same voice from two seeds among them.
    let asked = [
        ("Hello, crew.", 0, 1),
        ("Hello, crew.", 0, 2),
        ("Clown in the armory.", 1, 3),
        ("Security, report to the bridge.", 2, 4),
        ("Someone set the kitchen on fire.", 0, 5),
        ("Clown in the armory.", 2, 6),
        ("Hello, crew.", 1, 7),
        ("Security, report to the bridge.", 0, 8),
    ];
    let speak = |&(line, voice, seed): &(&str, usize, u64)| {
        let speech = rt.synthesize(line, &latents[voice], seed).unwrap();
        let bits: Vec<u32> = speech.samples.iter().map(|s| s.to_bits()).collect();
        bits
    };
    let alone: Vec<Vec<u32>> = asked.iter().map(speak).collect();
    let together: Vec<Vec<u32>> = thread::scope(|s| {
        let speaking: Vec<_> = asked.iter().map(|a| s.spawn(|| speak(a))).collect();
        speaking.into_iter().map(|t| t.join().unwrap()).collect()
    });
    for (i, (got, want)) in together.iter().zip(&alone).enumerate() {
        assert!(
            got == want,
            "line {i} asked for with others is not as it is alone"
        );
    }
    assert!(alone[0] != alone[1], "two seeds speak alike");
    rt.shut_down().unwrap_or_else(|e| panic!("{e}"));
}

/// What a device said, as the test keeps it.
#[derive(Serialize, Deserialize)]
struct Said {
    samples: Vec<f32>,
    patches: usize,
    prompt: usize,
}

impl From<Speech> for Said {
    fn from(speech: Speech) -> Self {
        Said {
            prompt: speech.start.map_or(0, |start| start.prompt),
            samples: speech.samples,
            patches: speech.patches,
        }
    }
}

/// The golden line in its voice from the seed, as the burn CPU device says it in `kind`'s dtypes,
/// kept under `ORACLE_DIR` when it is set.
fn cpu(kind: Kind) -> Said {
    let name = format!("tgtts-cpu-{kind}-{SEED}");
    let compute = || {
        let model = VoxCpm::load(
            &real::dir("VOXCPM2_DIR"),
            &kind.dtypes(),
            &Device::flex(),
            Params::default(),
            real::batching(),
        )
        .unwrap();
        let voice = voices().latent(VOICE).unwrap();
        Said::from(model.synthesize(LINE, &voice, SEED).unwrap())
    };
    let Some(dir) = std::env::var_os("ORACLE_DIR").filter(|dir| !dir.is_empty()) else {
        return compute();
    };
    let path = PathBuf::from(dir).join(format!("{name}.json"));
    if let Ok(json) = fs::read(&path) {
        eprintln!("the CPU's speech from {}", path.display());
        return serde_json::from_slice(&json).unwrap_or_else(|e| panic!("{e}"));
    }
    let said = compute();
    fs::write(&path, serde_json::to_vec(&said).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    said
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR and ORACLE_DIR, a directory to keep the CPU's speech in"]
fn keeps_the_cpu_speech() {
    if std::env::var_os("ORACLE_DIR").is_none_or(|dir| dir.is_empty()) {
        eprintln!("ORACLE_DIR is not set: nothing to keep the CPU's speech in");
        return;
    }
    // f32, as on the CPU, and f16, as on a GPU.
    cpu(Kind::Cpu);
    cpu(Kind::Vulkan);
}

/// The RMS of the difference over the RMS of `want`, or infinite where their lengths differ.
fn relative_rms(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let (mut err2, mut ref2) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        err2 += (g - w) * (g - w);
        ref2 += w * w;
    }
    (err2 / ref2).sqrt()
}

#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn speaks_a_line_as_burn_does_on_the_cpu() {
    let [want, burn] = [Kind::Cpu, Kind::Vulkan].map(cpu);
    let rt = rt(&Params::default());
    let voice = voices().latent(VOICE).unwrap();
    let speak = |seed| Said::from(rt.synthesize(LINE, &voice, seed).unwrap());
    let said = speak(SEED);
    // Defect: the noise of another seed.
    let other = speak(SEED + 1);
    drop(rt);
    assert_eq!((said.prompt, burn.prompt), (want.prompt, want.prompt));
    eprintln!(
        "a prompt of {} positions: patches f32 {}, f16 {}, rt {}; rt from another seed {}",
        want.prompt, want.patches, burn.patches, said.patches, other.patches
    );
    assert_eq!(said.patches, want.patches, "stopped after another patch");
    let margin = |got: &Said| relative_rms(&got.samples, &want.samples);
    let (ours, theirs) = (margin(&said), margin(&burn));
    eprintln!("audio against f32's: rt at {ours:.3e} relative RMS, burn's f16 at {theirs:.3e}");
    assert!(ours <= 1.25 * theirs, "{ours} against {theirs}");
    let stale = margin(&other);
    assert!(
        stale > 1.25 * theirs,
        "the noise of another seed passes: {stale}"
    );
}
