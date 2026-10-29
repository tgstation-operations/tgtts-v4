use ash::vk;

/// Why the runtime failed. Each names the Vulkan call or the decision that produced it, so a
/// message read in a test log is enough to find the cause.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Vulkan loader library could not be opened.
    #[error(
        "cannot open the Vulkan loader: {0}; NixOS keeps it off the library path, \
         and the devshell puts it on LD_LIBRARY_PATH"
    )]
    Loader(#[source] ash::LoadingError),
    /// A Vulkan entry point returned an error code.
    #[error("{call} failed: {result}")]
    Vk {
        call: &'static str,
        result: vk::Result,
    },
    /// No physical device met the runtime's requirements: one line per device seen, saying why
    /// it was passed over.
    #[error("no Vulkan device can run these kernels{}", indented(.0))]
    NoDevice(Vec<String>),
    /// [`VALIDATION_VAR`](crate::VALIDATION_VAR) asked for the Khronos validation layer and the
    /// loader could not find it on `VK_LAYER_PATH`.
    #[error(
        "{} is set but VK_LAYER_KHRONOS_validation is not on VK_LAYER_PATH",
        crate::VALIDATION_VAR
    )]
    ValidationUnavailable,
    /// A SPIR-V blob that is not a well-formed module header.
    #[error("not a SPIR-V module: {0}")]
    Spirv(String),
    /// A request the device cannot satisfy: a subgroup size outside its range, a grid larger
    /// than its dispatch limits, a specialization constant given twice.
    #[error("unsupported on this device: {0}")]
    Unsupported(String),
    /// No memory type both allowed by a buffer and acceptable to the runtime.
    #[error("no acceptable memory type among {type_bits:#b} for {size} bytes")]
    NoMemoryType { type_bits: u32, size: u64 },
}

/// The runtime's result.
pub type Result<T> = std::result::Result<T, Error>;

/// Each of `reasons` on a line of its own, indented under the message.
fn indented(reasons: &[String]) -> String {
    reasons.iter().map(|r| format!("\n  {r}")).collect()
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
