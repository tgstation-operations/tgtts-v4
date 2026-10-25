mod common;

use audiovae::{AudioVaeConfig, Decoder, Error};
use burn::tensor::FloatDType;
use common::{tiny, weights};
use testkit::{assert_close, device, fixture};

#[test]
fn decodes_as_upstream_in_the_bucket_of_out_sample_rate() {
    let device = device();
    let decoder = Decoder::load(&tiny(), &weights(), FloatDType::F32, &device).unwrap();
    let fixture = fixture!("audiovae");
    let audio = decoder.decode(fixture.tensor("input.decode.z", &device));
    // The output is within (-1, 1). Upstream's own f32 result is 2e-5 from its f64 one: the
    // random weights amplify rounding through thirteen convolutions.
    assert_close(audio, fixture.tensor("output.decode", &device), 1e-4, 0.0);
}

#[test]
fn boundaries_other_than_the_weights_are_refused() {
    let config = AudioVaeConfig {
        sr_bin_boundaries: Some(vec![16000, 24000, 32000]),
        ..tiny()
    };
    let error = Decoder::load(&config, &weights(), FloatDType::F32, &device()).unwrap_err();
    let Error::Weights(message) = error else {
        panic!("expected a weights error, got {error:?}");
    };
    assert_eq!(
        message,
        "decoder.sr_bin_boundaries: [20000, 30000, 40000], but the config has [16000, 24000, 32000]"
    );
}

#[test]
fn weights_of_another_shape_are_refused() {
    let config = AudioVaeConfig {
        decoder_dim: 64,
        ..tiny()
    };
    let error = Decoder::load(&config, &weights(), FloatDType::F32, &device()).unwrap_err();
    assert!(
        matches!(&error, Error::Weights(message) if message.contains("Shape mismatch")),
        "{error}"
    );
}
