//! The service end to end on the real model, voices and assets: a message for everyone, as the
//! game asks for it.

mod common;

use burn::tensor::Device;
use common::real;
use tgtts::{device::Kind, model::VoxCpm};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint, VOICES_DIR a voices directory and \
            ASSETS_DIR the assets"]
fn serves_a_message_on_the_real_model() {
    let checkpoint = real::dir("VOXCPM2_DIR");
    let dtypes = Kind::Cpu.dtypes();
    let model = VoxCpm::load(&checkpoint, &dtypes, &Device::flex(), real::params()).unwrap();
    real::serves_a_message(model, |_, _| {});
}
