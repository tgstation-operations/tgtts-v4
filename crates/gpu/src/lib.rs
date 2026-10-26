//! The GPU the models run on, and the model crates' fixtures and real-weight goldens run there
//! against what upstream computes, in f32 and in the dtypes the GPU serves in.
//!
//! The tests are built only with the feature of a GPU API, and run only where there is such a
//! GPU: `cargo test -p gpu --features vulkan`, or `--features cuda`. A build with both runs
//! them on CUDA.

#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use common::{Error, Gpu, device, keep_autotune_in};
#[cfg(feature = "cuda")]
pub use cuda::{CudaVersions, cuda, cuda_versions};
#[cfg(feature = "vulkan")]
pub use vulkan::vulkan;

#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod common {
    use std::{
        error, fmt,
        ops::Deref,
        path::Path,
        sync::{Mutex, MutexGuard, PoisonError},
    };

    use burn::{
        cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig},
        tensor::Device,
    };

    /// Why the GPU cannot be used as asked.
    #[derive(Debug)]
    pub enum Error {
        /// The Vulkan device, by its name, is not a GPU.
        #[cfg(feature = "vulkan")]
        NotAGpu(String),
        /// No CUDA driver library could be loaded: there is no NVIDIA driver, or it is not on
        /// the library path.
        #[cfg(feature = "cuda")]
        NoCudaDriver,
        /// The CUDA driver failed to start, as it said.
        #[cfg(feature = "cuda")]
        Cuda(String),
        /// The CUDA driver started and lists no device.
        #[cfg(feature = "cuda")]
        NoCudaDevice,
        /// This build compiles its kernels with LLVM, which takes CUDA's math functions from its
        /// libdevice, and there is none where it looks.
        #[cfg(all(feature = "cuda", not(feature = "nvrtc")))]
        NoLibdevice,
        /// This build compiles its kernels with NVRTC, which could not be loaded.
        #[cfg(feature = "nvrtc")]
        NoNvrtc,
        /// This build compiles its kernels with NVRTC, and the CUDA headers they include are not
        /// at the path given.
        #[cfg(feature = "nvrtc")]
        NoCudaHeaders(String),
        /// cubecl read its configuration before, when a device opened, so where autotune keeps
        /// its results can no longer change.
        Configured,
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                #[cfg(feature = "vulkan")]
                Self::NotAGpu(name) => write!(f, "the Vulkan device is {name}, which is not a GPU"),
                #[cfg(feature = "cuda")]
                Self::NoCudaDriver => {
                    f.write_str("no CUDA driver: libcuda could not be loaded from the library path")
                }
                #[cfg(feature = "cuda")]
                Self::Cuda(reason) => write!(f, "the CUDA driver failed to start: {reason}"),
                #[cfg(feature = "cuda")]
                Self::NoCudaDevice => f.write_str("the CUDA driver lists no device"),
                #[cfg(all(feature = "cuda", not(feature = "nvrtc")))]
                Self::NoLibdevice => f.write_str(
                    "this build compiles its kernels with LLVM, which links CUDA's math functions \
                     from nvvm/libdevice/libdevice.10.bc, and there is none under CUDA_PATH: set \
                     it to a CUDA installation",
                ),
                #[cfg(feature = "nvrtc")]
                Self::NoNvrtc => f.write_str(
                    "this build compiles its kernels with NVRTC, and libnvrtc could not be loaded \
                     from the library path",
                ),
                #[cfg(feature = "nvrtc")]
                Self::NoCudaHeaders(path) => write!(
                    f,
                    "this build compiles its kernels with NVRTC, and there are no CUDA headers in \
                     {path}: set CUDA_PATH to a CUDA installation"
                ),
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

    /// Held by whichever test of the process has the GPU. Autotune times the kernels it chooses
    /// between, and two tests sharing the GPU slow each other's down at random, so the kernels
    /// chosen, and how they round, would differ from those of the same test run alone.
    static GPU: Mutex<()> = Mutex::new(());

    /// The GPU, for one test at a time: the device, and the right to run on it until dropped.
    pub struct Gpu {
        device: Device,
        tf32: bool,
        _exclusive: MutexGuard<'static, ()>,
    }

    impl Gpu {
        /// Whether its f32 matmuls and convolutions are computed in TF32, which keeps f32's
        /// range and sums in f32 but rounds the inputs to f16's 10-bit mantissa: cubecl's run on
        /// tensor cores in TF32 wherever its runtime has the type, as CUDA's has from compute
        /// capability 8.0 and its Vulkan runtime never does.
        pub fn tf32(&self) -> bool {
            self.tf32
        }
    }

    impl Drop for Gpu {
        /// cubecl keeps memory per stream, a stream per thread, and holds the pages a stream
        /// emptied until asked to give them back. Each test runs on a thread of its own, so
        /// without that every test of a process would keep what it used from the next.
        fn drop(&mut self) {
            self.device.memory_cleanup();
        }
    }

    impl Deref for Gpu {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }

    /// The GPU the tests run on, [`cuda`](crate::cuda) in a build with CUDA and
    /// [`vulkan`](crate::vulkan) otherwise, once every other test of the process is done with
    /// it.
    ///
    /// # Panics
    ///
    /// If there is none.
    pub fn device() -> Gpu {
        // A test that failed on the GPU has left it as any other does.
        let exclusive = GPU.lock().unwrap_or_else(PoisonError::into_inner);
        #[cfg(feature = "cuda")]
        let device = crate::cuda();
        #[cfg(not(feature = "cuda"))]
        let device = crate::vulkan();
        let device = device.unwrap_or_else(|e| panic!("{e}"));
        #[cfg(feature = "cuda")]
        let tf32 = crate::cuda::supports_tf32();
        #[cfg(not(feature = "cuda"))]
        let tf32 = false;
        Gpu {
            device,
            tf32,
            _exclusive: exclusive,
        }
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use burn::tensor::{Device, DeviceKind};

    use crate::Error;

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
}

#[cfg(feature = "cuda")]
mod cuda {
    use burn::tensor::Device;
    use cudarc::driver::{result, sys};

    use crate::Error;

    /// The first CUDA device the driver lists, the first `CUDA_VISIBLE_DEVICES` names where it is
    /// set, opened.
    ///
    /// # Errors
    ///
    /// If there is no CUDA driver, it fails to start or lists no device, or, in a build that
    /// compiles its kernels with NVRTC, NVRTC or the CUDA headers are missing.
    pub fn cuda() -> Result<Device, Error> {
        // cubecl unwraps every driver call as it opens the device, so what can be missing is
        // asked about first, where it can still be an error.
        // SAFETY: it only tries to load the library under each of its names.
        if !unsafe { sys::is_culib_present() } {
            return Err(Error::NoCudaDriver);
        }
        result::init().map_err(|e| Error::Cuda(e.to_string()))?;
        if result::device::get_count().map_err(|e| Error::Cuda(e.to_string()))? == 0 {
            return Err(Error::NoCudaDevice);
        }
        #[cfg(feature = "nvrtc")]
        nvrtc()?;
        #[cfg(not(feature = "nvrtc"))]
        libdevice()?;
        let device = Device::cuda(0);
        device.identity().expect("a CUDA device says what it is");
        Ok(device)
    }

    /// Whether cubecl's runtime has TF32 on the device [`cuda`] opened.
    pub(crate) fn supports_tf32() -> bool {
        use cubecl::{prelude::Scalar, tf32};

        let client = cubecl::Device::Cuda(cubecl::cuda::CudaDevice::new(0)).client();
        client.properties().supports_type(tf32::elem_type_native())
    }

    /// Checks that NVRTC loads and that the headers it compiles cubecl's kernels with are where
    /// cubecl will look for them, as it would otherwise panic at the first kernel.
    #[cfg(feature = "nvrtc")]
    fn nvrtc() -> Result<(), Error> {
        use cubecl::cuda::install::{cuda_path, include_path};

        // SAFETY: as above.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(Error::NoNvrtc);
        }
        match cuda_path() {
            Some(_) if include_path().join("cuda_fp16.h").is_file() => Ok(()),
            Some(_) => Err(Error::NoCudaHeaders(include_path().display().to_string())),
            None => Err(Error::NoCudaHeaders(
                "CUDA_PATH, which is not set".to_owned(),
            )),
        }
    }

    /// Checks that libdevice is where cubecl's LLVM backend looks for it, as it would otherwise
    /// fail at the first kernel calling a math function, such as `exp`.
    #[cfg(not(feature = "nvrtc"))]
    fn libdevice() -> Result<(), Error> {
        let roots = match std::env::var_os("CUDA_PATH") {
            Some(path) => vec![path.into()],
            None => ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr"]
                .map(std::path::PathBuf::from)
                .to_vec(),
        };
        let libdevice = "nvvm/libdevice/libdevice.10.bc";
        if roots.iter().any(|root| root.join(libdevice).is_file()) {
            Ok(())
        } else {
            Err(Error::NoLibdevice)
        }
    }

    /// The CUDA versions a run depends on, each as CUDA numbers them, 12090 for 12.9.
    #[derive(Clone, Debug)]
    pub struct CudaVersions {
        /// The newest CUDA the driver supports.
        pub driver: i32,
        /// The CUDA the driver's functions were bound for in this build.
        pub bindings: u32,
        /// NVRTC's, where it loads.
        pub nvrtc: Option<i32>,
        /// What compiles the kernels: `nvrtc`, or `llvm`, to PTX the driver compiles further.
        pub compiler: &'static str,
    }

    /// The versions of the CUDA a run uses, once [`cuda`] has opened a device.
    ///
    /// # Errors
    ///
    /// If the driver does not say its version.
    pub fn cuda_versions() -> Result<CudaVersions, Error> {
        let mut driver = 0;
        // SAFETY: the driver writes its version to `driver`, which outlives the call.
        unsafe { sys::cuDriverGetVersion(&mut driver) }
            .result()
            .map_err(|e| Error::Cuda(e.to_string()))?;
        Ok(CudaVersions {
            driver,
            bindings: sys::CUDA_VERSION,
            nvrtc: nvrtc_version(),
            compiler: if cfg!(feature = "nvrtc") {
                "nvrtc"
            } else {
                "llvm"
            },
        })
    }

    fn nvrtc_version() -> Option<i32> {
        use cudarc::nvrtc::sys::{is_culib_present, nvrtcVersion};

        let (mut major, mut minor) = (0, 0);
        // SAFETY: the library is loaded only if it is present, and writes the version to
        // `major` and `minor`, which outlive the call.
        let loaded =
            unsafe { is_culib_present() && nvrtcVersion(&mut major, &mut minor).result().is_ok() };
        loaded.then_some(major * 1000 + minor * 10)
    }
}
