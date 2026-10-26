use std::fmt;

use ash::vk;

/// Why the runtime failed. Each names the Vulkan call or the decision that produced it, so a
/// message read in a test log is enough to find the cause.
#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader library could not be opened.
    Loader(ash::LoadingError),
    /// A Vulkan entry point returned an error code.
    Vk {
        call: &'static str,
        result: vk::Result,
    },
    /// No physical device met the runtime's requirements: one line per device seen, saying why
    /// it was passed over.
    NoDevice(Vec<String>),
    /// [`VALIDATION_VAR`](crate::VALIDATION_VAR) asked for the Khronos validation layer and the
    /// loader could not find it on `VK_LAYER_PATH`.
    ValidationUnavailable,
    /// A SPIR-V blob that is not a well-formed module header.
    Spirv(String),
    /// A request the device cannot satisfy: a subgroup size outside its range, a grid larger
    /// than its dispatch limits, a specialization constant given twice.
    Unsupported(String),
    /// No memory type both allowed by a buffer and acceptable to the runtime.
    NoMemoryType { type_bits: u32, size: u64 },
}

/// The runtime's result.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(source) => write!(
                f,
                "cannot open the Vulkan loader: {source}; NixOS keeps it off the library path, \
                 and the devshell puts it on LD_LIBRARY_PATH"
            ),
            Error::Vk { call, result } => write!(f, "{call} failed: {result}"),
            Error::NoDevice(reasons) => {
                write!(f, "no Vulkan device can run these kernels")?;
                for reason in reasons {
                    write!(f, "\n  {reason}")?;
                }
                Ok(())
            }
            Error::ValidationUnavailable => write!(
                f,
                "{} is set but VK_LAYER_KHRONOS_validation is not on VK_LAYER_PATH",
                crate::VALIDATION_VAR
            ),
            Error::Spirv(why) => write!(f, "not a SPIR-V module: {why}"),
            Error::Unsupported(why) => write!(f, "unsupported on this device: {why}"),
            Error::NoMemoryType { type_bits, size } => {
                write!(
                    f,
                    "no acceptable memory type among {type_bits:#b} for {size} bytes"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Loader(source) => Some(source),
            _ => None,
        }
    }
}

/// Attaches the name of the Vulkan entry point to an error code.
pub(crate) trait VkCall<T> {
    fn call(self, call: &'static str) -> Result<T>;
}

impl<T> VkCall<T> for std::result::Result<T, vk::Result> {
    fn call(self, call: &'static str) -> Result<T> {
        self.map_err(|result| Error::Vk { call, result })
    }
}
