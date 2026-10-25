//! The model crates' fixtures and real-weight goldens, run on a GPU against what upstream
//! computes, in f32 and in the dtypes the GPU serves in.
//!
//! The tests are built only with the feature of a GPU API, and run only where there is such a
//! GPU: `cargo test -p gpu --features vulkan`.

/// The GPU the tests run on: the one wgpu prefers among the Vulkan devices, or the one
/// `CUBECL_WGPU_DEFAULT_DEVICE` names.
///
/// # Panics
///
/// If there is none, or it is a CPU presenting itself as a GPU, as Mesa's llvmpipe does.
#[cfg(feature = "vulkan")]
pub fn device() -> burn::tensor::Device {
    let device = burn::tensor::Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);
    let identity = device.identity().expect("a Vulkan device says what it is");
    assert!(
        identity.physical.is_some(),
        "the Vulkan device is {}, which is not a GPU",
        identity.name
    );
    device
}
