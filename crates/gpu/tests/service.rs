//! The service end to end on the real model on the GPU, in the dtypes it serves in there and
//! warmed up as it starts: a message for everyone, as the game asks for it.
#![cfg(any(feature = "vulkan", feature = "cuda"))]

#[path = "../../tgtts/tests/common/mod.rs"]
mod service;

use std::{thread, time::Duration};

use service::real;
use tgtts::{
    device::Kind,
    model::{Batching, PREFILL_POSITIONS, PROMPT_CACHE_BYTES, VoxCpm},
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
        // As the service runs it: graphs are replayed on CUDA alone.
        Batching {
            graphs: cfg!(feature = "cuda"),
            ..real::batching()
        },
    )
    .unwrap();
    real::serves_a_message(model, |model, voices| model.warm_up(voices).unwrap());
}

/// The model warmed up as the service warms it, then rounds of requests at once, 1 to 16 of
/// them, of lines of mixed lengths in voices across the set: the pools hold no more pages than
/// they held at most during the warm-up, but for those the prompt cache fills as it keeps the
/// voices, up to its budget, so a service that started does not run out of memory later.
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
        prefill_positions: PREFILL_POSITIONS,
    };
    let params = Params::default();
    let model = VoxCpm::load(&checkpoint, &kind.dtypes(), &device, params, batching).unwrap();
    let voices = Voices::open(real::dir("VOICES_DIR")).unwrap();
    model.warm_up(&voices).unwrap();
    // Each pool's pages, at most and now, in bytes.
    let pools = || {
        let pools = device.memory_pool_report().unwrap();
        let peak = pools.iter().map(|pool| pool.pages_peak * pool.page_size);
        let now = pools.iter().map(|pool| pool.pages * pool.page_size);
        let largest = pools.iter().map(|pool| pool.page_size).max().unwrap_or(1);
        (peak.sum::<u64>(), now.sum::<u64>(), largest)
    };
    let (warmed, _, page) = pools();
    let graphs = model.graph_counts();
    // The cache's budget in whole pages of the largest pool, where it keeps its keys and values.
    let cache = (PROMPT_CACHE_BYTES as u64).div_ceil(page) * page;
    let mib = |bytes: u64| bytes >> 20;
    let names: Vec<&str> = voices.names().collect();
    let lines = [
        "Help.",
        "Clown in the armory, send security.",
        "Hello there, could someone please come to medbay? I need help.",
        "The supermatter is delaminating, everyone get to the escape shuttle right now, this is \
         not a drill, I repeat, this is not a drill.",
    ];
    let mut rounds = vec![];
    let sizes = [16, 1, 5, 16, 2, 9, 16, 3, 12, 16].repeat(2);
    for (round, size) in sizes.into_iter().enumerate() {
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
        rounds.push(mib(pools().1));
    }
    let (warmed, cache) = (mib(warmed), mib(cache));
    eprintln!(
        "pages of {warmed} MiB at most during the warm-up, the cache's {cache} MiB of pages, then \
         after each round {rounds:?}"
    );
    // The warm-up captured each graph the requests replay, and none is destroyed while it lives.
    assert_eq!(model.graph_counts(), graphs, "graphs after the warm-up");
    assert_eq!(graphs.destroyed, 0, "graphs destroyed in the warm-up");
    assert!(
        rounds.iter().all(|&held| held <= warmed + cache),
        "pages of {warmed} MiB at most during the warm-up and {cache} MiB for the cache, then \
         {rounds:?}"
    );
}

// The service replays graphs on CUDA alone: on Vulkan, replayed, 16 requests' first patches came
// out NaN.
#[cfg(feature = "cuda")]
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
