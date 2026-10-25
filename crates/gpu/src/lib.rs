//! The model crates' fixtures and real-weight goldens, run on a GPU against what upstream
//! computes, in f32 and in the dtypes the GPU serves in.
//!
//! The tests are built only with the feature of a GPU API, and run only where there is such a
//! GPU: `cargo test -p gpu --features vulkan`.

#[cfg(feature = "vulkan")]
pub use vulkan::{Gpu, device};

#[cfg(feature = "vulkan")]
mod vulkan {
    use std::{
        ops::Deref,
        sync::{Mutex, MutexGuard, PoisonError},
    };

    use burn::tensor::{Device, DeviceKind};

    /// Held by whichever test of the process has the GPU. Autotune times the kernels it chooses
    /// between, and two tests sharing the GPU slow each other's down at random, so the kernels
    /// chosen, and how they round, would differ from those of the same test run alone.
    static GPU: Mutex<()> = Mutex::new(());

    /// The GPU, for one test at a time: the device, and the right to run on it until dropped.
    pub struct Gpu {
        device: Device,
        _exclusive: MutexGuard<'static, ()>,
    }

    impl Deref for Gpu {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }

    /// The GPU the tests run on, once every other test of the process is done with it: the one
    /// wgpu prefers among the Vulkan devices, or the one `CUBECL_WGPU_DEFAULT_DEVICE` names.
    ///
    /// # Panics
    ///
    /// If there is none, or it is a CPU presenting itself as a GPU, as Mesa's llvmpipe does.
    pub fn device() -> Gpu {
        // A test that failed on the GPU has left it as any other does.
        let exclusive = GPU.lock().unwrap_or_else(PoisonError::into_inner);
        let device = Device::vulkan(DeviceKind::DefaultDevice);
        let identity = device.identity().expect("a Vulkan device says what it is");
        assert!(
            identity.physical.is_some(),
            "the Vulkan device is {}, which is not a GPU",
            identity.name
        );
        Gpu {
            device,
            _exclusive: exclusive,
        }
    }
}
