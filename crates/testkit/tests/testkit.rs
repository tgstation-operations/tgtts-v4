use burn::tensor::Tensor;
use safetensors::SafeTensors;
use testkit::{assert_close, checkpoint_dir, device, fixture};

fn floats<const N: usize>(values: [f32; N]) -> Tensor<1> {
    Tensor::from_floats(values, &device())
}

fn bits<const D: usize>(tensor: Tensor<D>) -> Vec<u32> {
    tensor.into_data().iter::<f32>().map(f32::to_bits).collect()
}

#[test]
fn fixture_names_its_generator() {
    let fixture = fixture!("testkit");
    let metadata = fixture.metadata();
    assert_eq!(
        metadata.keys().collect::<Vec<_>>(),
        ["generator", "seed", "torch", "upstream"]
    );
    assert_eq!(metadata["generator"], "fixtures/testkit.py");
    assert_eq!(metadata["seed"], "0");
    assert!(metadata["upstream"].starts_with("OpenBMB/VoxCPM@"));
}

#[test]
fn f32_reads_exactly() {
    let f32s = fixture!("testkit").tensor::<2>("f32", &device());
    assert_eq!(f32s.dims(), [2, 3]);
    let expected: [f32; 6] = [1.5, -2.25, 0.1, 3.0, 1e-3, -0.0];
    assert_eq!(bits(f32s), expected.map(f32::to_bits));
}

#[test]
fn bf16_widens_to_the_top_half_of_an_f32() {
    let bf16 = fixture!("testkit").tensor::<1>("bf16", &device());
    assert_eq!(
        bits(bf16),
        [0x3f80_0000, 0xc020_0000, 0x3dcd_0000, 0x4049_0000]
    );
}

#[test]
fn f16_widens_exactly() {
    let f16 = fixture!("testkit").tensor::<1>("f16", &device());
    let expected: [f32; 3] = [0.099_975_586, -65504.0, 5.960_464_5e-8];
    assert_eq!(bits(f16), expected.map(f32::to_bits));
}

#[test]
fn half_floats_are_stored_as_torch_rounds_them() {
    let bytes = std::fs::read(fixture!("testkit").path()).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let data = |name| file.tensor(name).unwrap().data();
    assert_eq!(
        data("bf16"),
        [0x80, 0x3f, 0x20, 0xc0, 0xcd, 0x3d, 0x49, 0x40]
    );
    assert_eq!(data("f16"), [0x66, 0x2e, 0xff, 0xfb, 0x01, 0x00]);
}

#[test]
fn upstream_rms_layernorm_divides_by_the_root_mean_square() {
    let device = device();
    let rms = 12.5f32.sqrt();
    assert_close(
        fixture!("testkit").tensor::<2>("rms_layernorm", &device),
        Tensor::from_floats([[3.0 / rms, 4.0 * 2.0 / rms]], &device),
        1e-6,
        1e-6,
    );
}

#[test]
#[should_panic(expected = "testkit.safetensors: no tensor \"f64\" among \
                           [\"bf16\", \"f16\", \"f32\", \"i64\", \"rms_layernorm\"]")]
fn missing_tensor_is_named_with_its_file() {
    let _ = fixture!("testkit").tensor::<1>("f64", &device());
}

#[test]
#[should_panic(expected = "testkit.safetensors: tensor \"f32\" has shape [2, 3], not rank 1")]
fn wrong_rank_is_named_with_its_file() {
    let _ = fixture!("testkit").tensor::<1>("f32", &device());
}

#[test]
#[should_panic(expected = "testkit.safetensors: tensor \"i64\" is I64, not F32, F16 or BF16")]
fn integer_tensor_is_refused() {
    let _ = fixture!("testkit").tensor::<1>("i64", &device());
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
