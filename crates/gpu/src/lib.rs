//! The GPU the models run on, and the model crates' fixtures and real-weight goldens run there
//! against what upstream computes, in f32 and in the dtypes the GPU serves in.
//!
//! The tests are built only with the feature of a GPU API, and run only where there is such a
//! GPU: `cargo test -p gpu --features vulkan`.

#[cfg(feature = "vulkan")]
pub use vulkan::{Error, Gpu, device, keep_autotune_in, vulkan};

#[cfg(feature = "vulkan")]
mod vulkan {
    use std::{
        error, fmt,
        ops::Deref,
        path::Path,
        sync::{Mutex, MutexGuard, PoisonError},
    };

    use burn::{
        cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig},
        tensor::{Device, DeviceKind},
    };

    /// Why the GPU cannot be used as asked.
    #[derive(Debug)]
    pub enum Error {
        /// The Vulkan device, by its name, is not a GPU.
        NotAGpu(String),
        /// cubecl read its configuration before, when a device opened, so where autotune keeps
        /// its results can no longer change.
        Configured,
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::NotAGpu(name) => write!(f, "the Vulkan device is {name}, which is not a GPU"),
                Self::Configured => f.write_str(
                    "a device opened before autotune was told where to keep its results",
                ),
            }
        }
    }

    impl error::Error for Error {}

    /// Has autotune keep its results in `dir`, where cubecl would keep them under the outermost
    /// `target/` above the working directory, for the next process to read in a build with
    /// burn's `persistence` feature. The rest of cubecl's configuration is its own: a
    /// `cubecl.toml` above the working directory and the `CUBECL_*` variables.
    ///
    /// # Errors
    ///
    /// If a device has opened.
    pub fn keep_autotune_in(dir: &Path) -> Result<(), Error> {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.environment.path = CacheConfig::Directory(dir.to_owned());
        if CubeClRuntimeConfig::try_set(config) {
            Ok(())
        } else {
            Err(Error::Configured)
        }
    }

    /// The Vulkan device wgpu prefers, or the one `CUBECL_WGPU_DEFAULT_DEVICE` names, opened.
    ///
    /// # Errors
    ///
    /// If it is a CPU presenting itself as a GPU, as Mesa's llvmpipe does.
    pub fn vulkan() -> Result<Device, Error> {
        let device = Device::vulkan(DeviceKind::DefaultDevice);
        let identity = device.identity().expect("a Vulkan device says what it is");
        match identity.physical {
            Some(_) => Ok(device),
            None => Err(Error::NotAGpu(identity.name)),
        }
    }

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

    /// The GPU the tests run on, [`vulkan`], once every other test of the process is done with
    /// it.
    ///
    /// # Panics
    ///
    /// If there is none.
    pub fn device() -> Gpu {
        // A test that failed on the GPU has left it as any other does.
        let exclusive = GPU.lock().unwrap_or_else(PoisonError::into_inner);
        Gpu {
            device: vulkan().unwrap_or_else(|e| panic!("{e}")),
            _exclusive: exclusive,
        }
    }
}
