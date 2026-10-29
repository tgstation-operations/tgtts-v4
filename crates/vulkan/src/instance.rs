use std::{
    ffi::{CStr, c_void},
    sync::Mutex,
};

use ash::vk;
use tracing::{error, warn};

use crate::{
    error::{Error, Result, VkCall},
    report::DeviceReport,
};

/// Set to `1`, attaches `VK_LAYER_KHRONOS_validation`, and logs and collects every warning and
/// error it reports ([`Instance::validation_messages`]). The loader finds the layer through
/// `VK_LAYER_PATH`; asking for it when it cannot be found is an error, never a silent run
/// without it.
pub const VALIDATION_VAR: &str = "VULKAN_VALIDATION";

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The Vulkan loader, opened at run time, an instance on it, and the validation messenger when
/// there is one.
pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub(crate) raw: ash::Instance,
    loader_version: u32,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    /// Written by the messenger callback through a raw pointer, so it is boxed to keep its
    /// address fixed, and dropped only after the messenger is destroyed.
    messages: Box<Mutex<Vec<String>>>,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("validation", &self.messenger.is_some())
            .finish_non_exhaustive()
    }
}

impl Instance {
    /// Opens the loader from the library path and creates a Vulkan 1.3 instance, with
    /// validation when [`VALIDATION_VAR`] is `1`.
    pub fn new() -> Result<Self> {
        // SAFETY: opening the loader runs its initializers; libvulkan is made to be dlopened, and
        // the entry points are resolved through vkGetInstanceProcAddr.
        let entry = unsafe { ash::Entry::load() }.map_err(Error::Loader)?;

        // SAFETY: a global command with no preconditions.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .call("vkEnumerateInstanceVersion")?
            .unwrap_or(vk::API_VERSION_1_0);

        let validation = std::env::var(VALIDATION_VAR).is_ok_and(|v| v == "1");
        let mut layers = Vec::new();
        let mut extensions = Vec::new();
        if validation {
            // SAFETY: a global command with no preconditions.
            let available = unsafe { entry.enumerate_instance_layer_properties() }
                .call("vkEnumerateInstanceLayerProperties")?;
            if !available
                .iter()
                .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
            {
                return Err(Error::ValidationUnavailable);
            }
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app = vk::ApplicationInfo::default()
            .application_name(c"tts-rs")
            .engine_name(c"tts-rs")
            .api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        // SAFETY: `info` and everything it points at outlive the call.
        let raw = unsafe { entry.create_instance(&info, None) }.call("vkCreateInstance")?;

        let mut instance = Instance {
            entry,
            raw,
            loader_version,
            messenger: None,
            messages: Box::new(Mutex::new(Vec::new())),
        };
        if validation {
            let utils = ash::ext::debug_utils::Instance::new(&instance.entry, &instance.raw);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(on_message))
                .user_data(
                    std::ptr::from_ref::<Mutex<Vec<String>>>(&instance.messages) as *mut c_void
                );
            // SAFETY: the user data points at the boxed sink, which `Drop` keeps alive until
            // after the messenger is destroyed.
            let messenger = unsafe { utils.create_debug_utils_messenger(&info, None) }
                .call("vkCreateDebugUtilsMessengerEXT")?;
            instance.messenger = Some((utils, messenger));
        }
        Ok(instance)
    }

    /// The instance-level API version the loader reports.
    pub fn loader_version(&self) -> u32 {
        self.loader_version
    }

    /// Whether the validation layer is attached.
    pub fn validating(&self) -> bool {
        self.messenger.is_some()
    }

    /// Every warning and error the validation layer has reported so far, each prefixed with its
    /// severity; empty when validation is off.
    pub fn validation_messages(&self) -> Vec<String> {
        self.messages.lock().map(|m| m.clone()).unwrap_or_default()
    }

    /// Every physical device the loader enumerates, with what it reports about itself.
    pub fn devices(&self) -> Result<Vec<(vk::PhysicalDevice, DeviceReport)>> {
        // SAFETY: the instance is live.
        let devices =
            unsafe { self.raw.enumerate_physical_devices() }.call("vkEnumeratePhysicalDevices")?;
        devices
            .into_iter()
            .map(|pd| DeviceReport::query(self, pd).map(|r| (pd, r)))
            .collect()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every child object is destroyed first: a device is destroyed by whatever owns
        // both it and this instance before that owner drops the instance, and the messenger is
        // destroyed here.
        unsafe {
            if let Some((utils, messenger)) = self.messenger.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}

/// The validation messenger's callback: logs the message, and appends it, tagged with its severity,
/// to the sink `user` points at. Never panics, since it is called from C.
unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes a valid callback-data pointer for the duration of the call, and
    // `user` is the boxed sink registered in `Instance::new`.
    let (data, sink) = unsafe { (&*data, &*(user as *const Mutex<Vec<String>>)) };
    // SAFETY: `p_message` is a valid NUL-terminated string or null.
    let text = unsafe { data.message_as_c_str() }
        .map(|m| m.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tag = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!("the validation layer reported: {text}");
        "error"
    } else {
        warn!("the validation layer reported: {text}");
        "warning"
    };
    if let Ok(mut messages) = sink.lock() {
        messages.push(format!("{tag}: {text}"));
    }
    vk::FALSE
}
