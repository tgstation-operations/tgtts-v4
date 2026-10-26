//! The service end to end on the real model on the GPU, in the dtypes it serves in there and
//! warmed up as it starts: a message for everyone, as the game asks for it.
#![cfg(feature = "vulkan")]

#[path = "../../tgtts/tests/common/mod.rs"]
mod service;

use service::real;
use tgtts::{device::Kind, model::VoxCpm};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn serves_a_message_on_the_real_model() {
    let (checkpoint, device) = (real::dir("VOXCPM2_DIR"), gpu::device());
    let dtypes = Kind::Vulkan.dtypes();
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
