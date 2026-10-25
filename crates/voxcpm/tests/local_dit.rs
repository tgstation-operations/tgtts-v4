mod common;

use burn::tensor::FloatDType;
use common::tiny;
use testkit::{assert_close, device, fixture};
use voxcpm::LocalDit;

#[test]
fn predicts_velocities_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_local_dit"), device());
    let dit = LocalDit::load(&tiny(), fixture.path(), FloatDType::F32, &device).unwrap();
    let input = |name: &str| format!("input.{name}");
    let velocity = dit.forward(
        fixture.tensor(&input("x"), &device),
        fixture.tensor(&input("mu"), &device),
        fixture.tensor(&input("t"), &device),
        fixture.tensor(&input("cond"), &device),
        fixture.tensor(&input("dt"), &device),
    );
    // f32 here and in upstream, summing in other orders: these velocities, of magnitude under
    // 1.4, came within 3.0e-7 (5 ulps) of upstream's.
    let expected = fixture.tensor("output.velocity", &device);
    assert_close(velocity, expected, 2e-6, 1e-6);
}
