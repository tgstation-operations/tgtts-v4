mod common;

use burn::tensor::FloatDType;
use common::tiny;
use testkit::{assert_close, device, fixture};
use voxcpm::{Error, LocalEncoder};

#[test]
fn embeds_patches_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_local_encoder"), device());
    let encoder = LocalEncoder::load(&tiny(), fixture.path(), FloatDType::F32, &device).unwrap();
    let embeddings = encoder.forward(fixture.tensor("input.patches", &device));
    // f32 here and in upstream, summing in other orders: these embeddings, of magnitude under
    // 1.8, came within 3.6e-7 (3 ulps) of upstream's.
    let expected = fixture.tensor("output.embeddings", &device);
    assert_close(embeddings, expected, 2e-6, 1e-6);
}

#[test]
fn weights_of_another_shape_are_refused() {
    let mut config = tiny();
    config.feat_dim = 4;
    let fixture = fixture!("voxcpm_local_encoder");
    let error =
        LocalEncoder::load(&config, fixture.path(), FloatDType::F32, &device()).unwrap_err();
    assert!(matches!(error, Error::Load(_)), "{error}");
}
