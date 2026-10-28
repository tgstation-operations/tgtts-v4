//! The service end to end on the real model, voices and assets: a message for everyone, as the
//! game asks for it.

mod common;

use std::{thread, time::Duration};

use burn::tensor::Device;
use common::real;
use tgtts::{
    device::Kind,
    load::{Batching, PREFILL_GRAPHS, PREFILL_POSITIONS, PROMPT_CACHE_BYTES},
    model::VoxCpm,
    synth::Synthesizer,
};
use voices::Voices;

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn serves_a_message_on_the_real_model() {
    let checkpoint = real::dir("VOXCPM2_DIR");
    let dtypes = Kind::Cpu.dtypes();
    let model = VoxCpm::load(
        &checkpoint,
        &dtypes,
        &Device::flex(),
        real::params(),
        real::batching(),
    )
    .unwrap();
    real::serves_a_message(model, |_, _| {});
}

/// Requests sent at once generate together, each to what it is one at a time.
#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn synthesizes_requests_at_once_as_one_at_a_time_on_the_real_model() {
    let checkpoint = real::dir("VOXCPM2_DIR");
    let voices = Voices::open(real::dir("VOICES_DIR")).unwrap();
    let dtypes = Kind::Cpu.dtypes();
    let batching = Batching {
        window: Duration::from_secs(1),
        requests: 3,
        graphs: true,
        prefill_positions: PREFILL_POSITIONS,
        prefill_graphs: PREFILL_GRAPHS.to_vec(),
        prefill_batch: None,
        prompt_cache_bytes: PROMPT_CACHE_BYTES,
    };
    let params = real::params();
    let model = VoxCpm::load(&checkpoint, &dtypes, &Device::flex(), params, batching).unwrap();
    let names: Vec<&str> = voices.names().take(3).collect();
    let lines = [
        "Hello, crew.",
        "Someone set the kitchen on fire.",
        "Clown in the armory.",
    ];
    let requests: Vec<_> = names
        .iter()
        .zip(lines)
        .enumerate()
        .map(|(i, (name, line))| (voices.latent(name).unwrap(), line, i as u64))
        .collect();
    let alone: Vec<_> = requests
        .iter()
        .map(|(voice, line, seed)| model.synthesize(line, voice, *seed).unwrap())
        .collect();
    let together: Vec<_> = thread::scope(|scope| {
        let threads: Vec<_> = requests
            .iter()
            .map(|(voice, line, seed)| scope.spawn(|| model.synthesize(line, voice, *seed)))
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().unwrap().unwrap())
            .collect()
    });
    for (i, (alone, together)) in alone.iter().zip(&together).enumerate() {
        assert_eq!(together.patches, alone.patches, "request {i}");
        let diff = alone
            .samples
            .iter()
            .zip(&together.samples)
            .fold(0f32, |worst, (a, b)| worst.max((a - b).abs()));
        eprintln!(
            "request {i}: {} patches, audio within {diff:.1e}",
            alone.patches
        );
        // f32 on the CPU, a product's rows summed as one or as several: the audio came within
        // 4.8e-4 of alone's.
        assert!(diff < 1e-2, "request {i}: audio {diff} from alone");
    }
}
