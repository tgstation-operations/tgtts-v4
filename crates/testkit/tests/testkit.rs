use burn::tensor::Tensor;
use testkit::{assert_close, checkpoint_dir, device};

fn floats<const N: usize>(values: [f32; N]) -> Tensor<1> {
    Tensor::from_floats(values, &device())
}

#[test]
fn close_within_either_tolerance() {
    assert_close(floats([1.0005, -0.0]), floats([1.0, 0.0]), 1e-3, 0.0);
    assert_close(
        floats([100.5, f32::INFINITY]),
        floats([100.0, f32::INFINITY]),
        0.0,
        1e-2,
    );
}

#[test]
#[should_panic(expected = "1 of 1 elements not close (atol 1e-3, rtol 0e0)")]
fn atol_alone_is_absolute() {
    assert_close(floats([100.5]), floats([100.0]), 1e-3, 0.0);
}

#[test]
#[should_panic(
    expected = "2 of 6 elements not close (atol 1e-2, rtol 1e-2) in shape [2, 3]\n\
                           max abs diff 5.000e-1, max rel diff 1.667e-1\n\
                           worst at [1, 0]: actual 3.5, expected 3"
)]
fn report_gives_shape_differences_and_worst_index() {
    let device = device();
    assert_close(
        Tensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.5, 4.0, 5.1]], &device),
        Tensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device),
        0.01,
        0.01,
    );
}

#[test]
#[should_panic(expected = "1 of 1 elements not close")]
fn nan_is_never_close() {
    assert_close(floats([f32::NAN]), floats([f32::NAN]), 1.0, 1.0);
}

#[test]
#[should_panic(expected = "shapes differ: actual [1, 2], expected [2, 1]")]
fn shapes_must_match() {
    let device = device();
    assert_close(
        Tensor::<2>::from_floats([[1.0, 2.0]], &device),
        Tensor::<2>::from_floats([[1.0], [2.0]], &device),
        1.0,
        1.0,
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn checkpoint_dir_holds_the_weights() {
    let dir = checkpoint_dir();
    assert!(dir.join("model.safetensors").is_file());
    assert!(dir.join("audiovae.pth").is_file());
}
