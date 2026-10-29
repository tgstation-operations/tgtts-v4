//! The service's model, tts-rs's own runtime on a Vulkan GPU, on the real model, voices and
//! assets: started, speaking and shut down, which leaves nothing of it running; a message for
//! everyone, as the game asks for it, the model warmed up as the service warms it; a request in a
//! voice the prompt cache keeps, which comes out as it did the first time; lines asked for at
//! once, which come out as each does alone, streamed or not, a streamed one's first audio its
//! first patch's; and voxcpm's golden line in its voice spoken from a seed as this GPU spoke it
//! when its golden was captured. Run with `VOXCPM2_DIR`, `VOICES_DIR`, `ASSETS_DIR` and
//! `GOLDEN_DIR` set and `GLSL_KERNELS` and `VULKAN_KERNELS` naming the compiled kernels, as
//! `nix run .#device-tests` sets the last two, one test at a time; under the validation layer, as
//! it runs, a request fails on anything the layer reports.

mod common;

use std::{
    fs, thread,
    time::{Duration, Instant},
};

use common::real;
use serde::{Deserialize, Serialize};
use testkit::golden;
use tgtts::{
    load::PROMPT_CACHE_BYTES,
    rt::Rt,
    synth::{Speech, Synthesizer},
};
use voices::Voices;
use voxcpm_core::Params;

/// voxcpm's golden line and voice: run free from the seed, the generator stops after its 24th
/// patch, where voxcpm in f32 on the CPU did.
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

/// When each run of a stream's audio came, and its samples.
type Runs = Vec<(Duration, usize)>;

/// Lines streamed among lines asked for whole, all at once, come out as each does alone, and a
/// streamed line's first audio is its first patch's, decoded as soon as it is sampled.
#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS and VULKAN_KERNELS"]
fn lines_streamed_come_out_as_they_do_whole_from_their_first_patch() {
    let rt = rt(&real::params());
    let voices = voices();
    let names: Vec<&str> = voices.names().take(2).collect();
    let latents: Vec<_> = names
        .iter()
        .map(|name| voices.latent(name).unwrap())
        .collect();
    // (line, voice, seed, streamed)
    let asked = [
        ("Hello, crew.", 0, 1, true),
        ("Clown in the armory.", 1, 2, false),
        ("Security, report to the bridge.", 0, 3, true),
        ("Someone set the kitchen on fire.", 1, 4, true),
        ("Hello, crew.", 1, 5, false),
    ];
    let bits = |samples: &[f32]| -> Vec<u32> { samples.iter().map(|s| s.to_bits()).collect() };
    let alone: Vec<Vec<u32>> = (asked.iter())
        .map(|&(line, voice, seed, _)| {
            bits(&rt.synthesize(line, &latents[voice], seed).unwrap().samples)
        })
        .collect();
    let started = Instant::now();
    let together: Vec<(Vec<u32>, Runs)> = thread::scope(|s| {
        let speaking: Vec<_> = (asked.iter())
            .map(|&(line, voice, seed, streamed)| {
                let (rt, latent) = (&rt, &latents[voice]);
                s.spawn(move || {
                    if !streamed {
                        return (
                            bits(&rt.synthesize(line, latent, seed).unwrap().samples),
                            vec![],
                        );
                    }
                    let (mut runs, mut audio) = (vec![], vec![]);
                    let speech = rt
                        .stream(line, latent, seed, &mut |run| {
                            runs.push((started.elapsed(), run.len()));
                            audio.extend_from_slice(run);
                        })
                        .unwrap();
                    assert!(
                        bits(&audio) == bits(&speech.samples),
                        "{line}: streamed apart from its speech"
                    );
                    (bits(&speech.samples), runs)
                })
            })
            .collect();
        speaking.into_iter().map(|t| t.join().unwrap()).collect()
    });
    // A patch of audio: four latent frames of 1920 samples.
    const PATCH: usize = 7680;
    for (i, ((got, runs), want)) in together.iter().zip(&alone).enumerate() {
        assert!(got == want, "line {i} is not as it is alone");
        if asked[i].3 {
            let sizes: Vec<usize> = runs.iter().map(|&(_, n)| n).collect();
            let ms: Vec<u128> = runs.iter().map(|(at, _)| at.as_millis()).collect();
            eprintln!(
                "line {i}: runs of {sizes:?} samples at {ms:?} ms (smoke, not a measurement)"
            );
            assert_eq!(sizes.first(), Some(&PATCH), "line {i}'s first audio");
            assert_eq!(sizes.iter().sum::<usize>(), got.len());
        } else {
            assert!(runs.is_empty());
        }
    }
    rt.shut_down().unwrap_or_else(|e| panic!("{e}"));
}

/// What the model said, as the test keeps it.
#[derive(Clone, Serialize, Deserialize)]
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

/// The GPU the model runs on, by its PCI vendor and device ids.
fn gpu() -> String {
    let ctx = vulkan::Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report();
    format!("{:04x}-{:04x}", report.vendor_id, report.device_id)
}

/// The golden line in its voice from the seed, held to what this GPU said when its golden was
/// captured, bit for bit: a generation runs free, and the first level of the scalar quantizer
/// that another GPU's rounding puts apart sends it elsewhere, so each GPU has a golden of its
/// own. The graph crate's goldens hold the model's parts on every GPU, a generation forced along
/// one path; a driver that rounds otherwise wants this one captured again once those pass.
#[test]
#[ignore = "needs VOXCPM2_DIR, VOICES_DIR, GLSL_KERNELS, VULKAN_KERNELS and GOLDEN_DIR"]
fn speaks_a_line_as_this_gpu_spoke_it() {
    let rt = rt(&Params::default());
    let voice = voices().latent(VOICE).unwrap();
    let speak = |seed| Said::from(rt.synthesize(LINE, &voice, seed).unwrap());
    let said = speak(SEED);
    // Defect: the noise of another seed.
    let other = speak(SEED + 1);
    drop(rt);
    let name = format!("tgtts-line-{}-{SEED}", gpu());
    let want = golden(&name, || said.clone());
    eprintln!(
        "a prompt of {} positions: patches {}, the golden's {}; from another seed {}",
        said.prompt, said.patches, want.patches, other.patches
    );
    let bits = |said: &Said| -> Vec<u32> { said.samples.iter().map(|s| s.to_bits()).collect() };
    assert_eq!((said.prompt, said.patches), (want.prompt, want.patches));
    assert!(
        bits(&said) == bits(&want),
        "the line is not as this GPU spoke it"
    );
    assert!(
        bits(&other) != bits(&want),
        "the noise of another seed passes"
    );
}
