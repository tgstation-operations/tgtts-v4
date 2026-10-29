use safetensors::SafeTensors;
use testkit::{checkpoint_dir, fixture};

fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
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
    let (shape, f32s) = fixture!("testkit").floats("f32");
    assert_eq!(shape, [2, 3]);
    let expected: [f32; 6] = [1.5, -2.25, 0.1, 3.0, 1e-3, -0.0];
    assert_eq!(bits(f32s), expected.map(f32::to_bits));
}

#[test]
fn bf16_widens_to_the_top_half_of_an_f32() {
    let (_, bf16) = fixture!("testkit").floats("bf16");
    assert_eq!(
        bits(bf16),
        [0x3f80_0000, 0xc020_0000, 0x3dcd_0000, 0x4049_0000]
    );
}

#[test]
fn f16_widens_exactly() {
    let (_, f16) = fixture!("testkit").floats("f16");
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
    let (shape, normed) = fixture!("testkit").floats("rms_layernorm");
    assert_eq!(shape, [1, 2]);
    let rms = 12.5f32.sqrt();
    for (got, want) in normed.iter().zip([3.0 / rms, 4.0 * 2.0 / rms]) {
        assert!(
            (got - want).abs() <= 1e-6 + 1e-6 * want.abs(),
            "{got} for {want}"
        );
    }
}

#[test]
#[should_panic(expected = "testkit.safetensors: no tensor \"f64\" among \
                           [\"bf16\", \"f16\", \"f32\", \"i64\", \"rms_layernorm\"]")]
fn missing_tensor_is_named_with_its_file() {
    let _ = fixture!("testkit").floats("f64");
}

#[test]
#[should_panic(expected = "testkit.safetensors: tensor \"i64\" is I64, not F32, F16 or BF16")]
fn integer_tensor_is_refused() {
    let _ = fixture!("testkit").floats("i64");
}

#[test]
fn i64_reads_exactly() {
    assert_eq!(fixture!("testkit").ints("i64"), (vec![2], vec![7, -1]));
}

#[test]
#[should_panic(expected = "testkit.safetensors: tensor \"f32\" is F32, not I64")]
fn float_tensor_is_not_an_int_tensor() {
    let _ = fixture!("testkit").ints("f32");
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn checkpoint_dir_holds_the_weights() {
    let dir = checkpoint_dir();
    assert!(dir.join("model.safetensors").is_file());
    assert!(dir.join("audiovae.pth").is_file());
}
