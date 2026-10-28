//! The service on the rt device, tts-rs's own runtime on a Vulkan GPU, on the real model, voices
//! and assets: the device started, speaking and shut down, which leaves nothing of it running; a
//! message for everyone, as the game asks for it, the model warmed up as the service warms it; a
//! request in a voice the prompt cache keeps, which comes out as it did the first time. Built with
//! the `rt` feature, and run with `VOXCPM2_DIR`, `VOICES_DIR` and `ASSETS_DIR` set and
//! `GLSL_KERNELS` and `VULKAN_KERNELS` naming the compiled kernels, as `nix run .#device-tests`
//! sets them, one test at a time; under the validation layer, as it runs, a request fails on
//! anything the layer reports.
#![cfg(feature = "rt")]

mod common;

use std::fs;

use common::real;
use tgtts::{
    model::PROMPT_CACHE_BYTES,
    rt::Rt,
    synth::{Speech, Synthesizer},
};
use voices::Voices;
use voxcpm::Params;

/// voxcpm's golden line and voice.
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
