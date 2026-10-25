mod common;

use audiovae::Encoder;
use burn::tensor::FloatDType;
use common::{tiny, weights};
use testkit::{assert_close, device, fixture};

#[test]
fn encodes_as_upstream_padding_to_whole_frames() {
    let device = device();
    let encoder = Encoder::load(&tiny(), &weights(), FloatDType::F32, &device).unwrap();
    let fixture = fixture!("audiovae");
    let latents = encoder.encode(fixture.tensor("input.encode.audio", &device));
    // Upstream's own f32 result is 2.4e-6 from its f64 one.
    assert_close(
        latents,
        fixture.tensor("output.encode", &device),
        1e-5,
        1e-5,
    );
}
