//! The service end to end on the real model on the GPU, in the dtypes it serves in there and
//! warmed up as it starts: a message for everyone, as the game asks for it.
#![cfg(any(feature = "vulkan", feature = "cuda"))]

#[path = "../../tgtts/tests/common/mod.rs"]
mod service;

use std::{thread, time::Duration};

use service::real;
use tgtts::{
    device::Kind,
    model::{Batching, VoxCpm},
    synth::Synthesizer,
};
use voices::Voices;
use voxcpm::Params;

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn serves_a_message_on_the_real_model() {
    let (checkpoint, device) = (real::dir("VOXCPM2_DIR"), gpu::device());
    let kind = if cfg!(feature = "cuda") {
        Kind::Cuda
    } else {
        Kind::Vulkan
    };
    let dtypes = kind.dtypes();
    let model = VoxCpm::load(
        &checkpoint,
        &dtypes,
        &device,
        real::params(),
        real::batching(),
    )
    .unwrap();
    real::serves_a_message(model, |model, voices| model.warm_up(voices).unwrap());
}

/// The model warmed up as the service warms it, then rounds of requests at once, 1 to 16 of
/// them, of lines of mixed lengths in voices across the set: the pools reserve no more of the
/// GPU's memory than they did after the warm-up, so a service that started does not run out of
/// it later.
fn holds_the_memory_it_warmed_up_with(graphs: bool) {
    let (checkpoint, device) = (real::dir("VOXCPM2_DIR"), gpu::device());
    let kind = if cfg!(feature = "cuda") {
        Kind::Cuda
    } else {
        Kind::Vulkan
    };
    let batching = Batching {
        window: Duration::from_millis(100),
        requests: 16,
        graphs,
    };
    let params = Params::default();
    let model = VoxCpm::load(&checkpoint, &kind.dtypes(), &device, params, batching).unwrap();
    let voices = Voices::open(real::dir("VOICES_DIR")).unwrap();
    model.warm_up(&voices).unwrap();
    let mib = || device.memory_pool_usage().unwrap().bytes_reserved >> 20;
    let warmed = mib();
    let names: Vec<&str> = voices.names().collect();
    let lines = [
        "Help.",
        "Clown in the armory, send security.",
        "Hello there, could someone please come to medbay? I need help.",
        "The supermatter is delaminating, everyone get to the escape shuttle right now, this is \
         not a drill, I repeat, this is not a drill.",
    ];
    let mut rounds = vec![];
    for (round, size) in [16, 1, 5, 16, 2, 9, 16, 3, 12, 16].into_iter().enumerate() {
        thread::scope(|scope| {
            for i in 0..size {
                let (voices, names, model) = (&voices, &names, &model);
                scope.spawn(move || {
                    let name = names[(7 * round + 13 * i) % names.len()];
                    let latent = voices.latent(name).unwrap();
                    let line = lines[(round + i) % lines.len()];
                    model.synthesize(line, &latent, i as u64).unwrap();
                });
            }
        });
        rounds.push(mib());
    }
    eprintln!("reserved {warmed} MiB after the warm-up, then after each round {rounds:?}");
    assert!(
        rounds.iter().all(|&reserved| reserved <= warmed),
        "reserved {warmed} MiB after the warm-up, then {rounds:?}"
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn holds_the_memory_it_warmed_up_with_on_the_real_model() {
    holds_the_memory_it_warmed_up_with(true);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn holds_the_memory_it_warmed_up_with_on_the_real_model_stepping_eagerly() {
    holds_the_memory_it_warmed_up_with(false);
}
