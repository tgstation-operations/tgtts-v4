//! Which device and which queue the runtime uses, decided from plain data so the rules are tested
//! without a GPU.
//!
//! A device must speak Vulkan 1.3, carry every extension in [`REQUIRED_EXTENSIONS`], support every
//! feature in [`Features`], have a compute queue that can write timestamps, and not be a CPU
//! implementation: Mesa's lavapipe would otherwise qualify, and run every kernel correctly and very
//! slowly. Among those that qualify a discrete GPU is preferred, then an integrated one, as burn's
//! Vulkan runtime chooses.

use std::ffi::CStr;

use ash::vk;

/// Device extensions the runtime cannot run without, beyond core Vulkan 1.3.
pub const REQUIRED_EXTENSIONS: &[&CStr] = &[ash::khr::cooperative_matrix::NAME];

/// The lowest device API version accepted: subgroup size control, synchronization2 and
/// maintenance4 are core from 1.3.
pub const MIN_API_VERSION: u32 = vk::API_VERSION_1_3;

/// The device features the runtime enables. Every one is required, and a device missing
/// any is passed over with the missing names in the reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// f16 arithmetic in shaders (Vulkan 1.2).
    pub shader_float16: bool,
    /// 16-bit values in storage buffers (Vulkan 1.1).
    pub storage_buffer_16bit_access: bool,
    /// A pipeline may require a subgroup size (Vulkan 1.3).
    pub subgroup_size_control: bool,
    /// A compute pipeline may require full subgroups (Vulkan 1.3).
    pub compute_full_subgroups: bool,
    /// `vkCmdPipelineBarrier2`, `vkCmdWriteTimestamp2` and `vkQueueSubmit2` (Vulkan 1.3).
    pub synchronization2: bool,
    /// Among others, the `LocalSizeId` execution mode: a workgroup size taken from
    /// specialization constants (Vulkan 1.3).
    pub maintenance4: bool,
    /// `VK_KHR_cooperative_matrix`: the subgroup matrix multiply-accumulate instructions.
    pub cooperative_matrix: bool,
    /// The Vulkan memory model (Vulkan 1.2), which SPIR-V requires of every module that
    /// declares `CooperativeMatrixKHR`.
    pub vulkan_memory_model: bool,
}

impl Features {
    /// Every feature set: what the runtime asks the device for.
    pub const ALL: Self = Self {
        shader_float16: true,
        storage_buffer_16bit_access: true,
        subgroup_size_control: true,
        compute_full_subgroups: true,
        synchronization2: true,
        maintenance4: true,
        cooperative_matrix: true,
        vulkan_memory_model: true,
    };

    /// Each feature by its Vulkan name, with whether it is set.
    pub fn list(&self) -> [(&'static str, bool); 8] {
        [
            ("shaderFloat16", self.shader_float16),
            ("storageBuffer16BitAccess", self.storage_buffer_16bit_access),
            ("subgroupSizeControl", self.subgroup_size_control),
            ("computeFullSubgroups", self.compute_full_subgroups),
            ("synchronization2", self.synchronization2),
            ("maintenance4", self.maintenance4),
            ("cooperativeMatrix", self.cooperative_matrix),
            ("vulkanMemoryModel", self.vulkan_memory_model),
        ]
    }

    /// The Vulkan names of the features not set.
    pub fn missing(&self) -> Vec<&'static str> {
        self.list()
            .into_iter()
            .filter_map(|(name, set)| (!set).then_some(name))
            .collect()
    }
}

/// One queue family, as far as the choice of a compute queue needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueFamily {
    pub flags: vk::QueueFlags,
    pub count: u32,
    /// Meaningful bits in a timestamp written on this family's queues; 0 means none.
    pub timestamp_valid_bits: u32,
}

/// The queue family compute work is submitted to: one that supports compute and timestamps,
/// preferring a family without graphics, the dedicated compute queue, as llama.cpp's Vulkan
/// backend does by default.
pub fn compute_queue_family(families: &[QueueFamily]) -> Option<u32> {
    let usable = |f: &QueueFamily| {
        f.flags.contains(vk::QueueFlags::COMPUTE) && f.count > 0 && f.timestamp_valid_bits > 0
    };
    let pick = families
        .iter()
        .position(|f| usable(f) && !f.flags.contains(vk::QueueFlags::GRAPHICS))
        .or_else(|| families.iter().position(usable))?;
    u32::try_from(pick).ok()
}

/// What device selection needs to know about one physical device.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    /// Required extensions the device does not advertise.
    pub missing_extensions: Vec<String>,
    pub features: Features,
    pub queue_families: Vec<QueueFamily>,
}

impl Candidate {
    /// Why this device cannot be used, or `None` when it can.
    pub fn rejection(&self) -> Option<String> {
        let mut why = Vec::new();
        if self.device_type == vk::PhysicalDeviceType::CPU {
            why.push("is a CPU implementation".to_string());
        }
        if self.api_version < MIN_API_VERSION {
            why.push(format!(
                "speaks Vulkan {}, not 1.3",
                version_string(self.api_version)
            ));
        }
        if !self.missing_extensions.is_empty() {
            why.push(format!("lacks {}", self.missing_extensions.join(", ")));
        }
        let features = self.features.missing();
        if !features.is_empty() {
            why.push(format!("lacks the features {}", features.join(", ")));
        }
        if compute_queue_family(&self.queue_families).is_none() {
            why.push("has no compute queue that writes timestamps".to_string());
        }
        (!why.is_empty()).then(|| format!("{}: {}", self.name, why.join("; ")))
    }
}

/// Rank of a device type among qualifying devices; lower is preferred.
fn preference(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    }
}

/// The index of the device to use, or one line per device saying why none qualified.
/// Ties keep the loader's enumeration order.
pub fn choose(candidates: &[Candidate]) -> std::result::Result<usize, Vec<String>> {
    candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.rejection().is_none())
        .min_by_key(|(i, c)| (preference(c.device_type), *i))
        .map(|(i, _)| i)
        .ok_or_else(|| {
            if candidates.is_empty() {
                vec!["the loader enumerated no physical devices".to_string()]
            } else {
                candidates.iter().filter_map(Candidate::rejection).collect()
            }
        })
}

/// `major.minor.patch` of a Vulkan API version number.
pub fn version_string(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags, timestamp_valid_bits: u32) -> QueueFamily {
        QueueFamily {
            flags,
            count: 1,
            timestamp_valid_bits,
        }
    }

    fn gpu(name: &str, device_type: vk::PhysicalDeviceType) -> Candidate {
        Candidate {
            name: name.to_string(),
            device_type,
            api_version: vk::make_api_version(0, 1, 4, 0),
            missing_extensions: Vec::new(),
            features: Features::ALL,
            queue_families: vec![
                family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 64),
                family(vk::QueueFlags::COMPUTE, 64),
            ],
        }
    }

    #[test]
    fn a_dedicated_compute_queue_is_preferred() {
        let families = gpu("a", vk::PhysicalDeviceType::INTEGRATED_GPU).queue_families;
        assert_eq!(compute_queue_family(&families), Some(1));
    }

    #[test]
    fn a_graphics_queue_serves_when_it_is_the_only_compute_queue() {
        let families = [
            family(vk::QueueFlags::TRANSFER, 64),
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 64),
        ];
        assert_eq!(compute_queue_family(&families), Some(1));
    }

    #[test]
    fn a_queue_without_timestamps_is_not_used() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 64),
            family(vk::QueueFlags::COMPUTE, 0),
        ];
        assert_eq!(compute_queue_family(&families), Some(0));
        assert_eq!(compute_queue_family(&families[1..]), None);
    }

    #[test]
    fn a_cpu_implementation_is_never_chosen() {
        let devices = [gpu("llvmpipe", vk::PhysicalDeviceType::CPU)];
        let reasons = choose(&devices).unwrap_err();
        assert_eq!(reasons, ["llvmpipe: is a CPU implementation"]);
    }

    #[test]
    fn a_discrete_gpu_is_preferred_and_ties_keep_enumeration_order() {
        let devices = [
            gpu("llvmpipe", vk::PhysicalDeviceType::CPU),
            gpu("radv", vk::PhysicalDeviceType::INTEGRATED_GPU),
            gpu("nvidia", vk::PhysicalDeviceType::DISCRETE_GPU),
            gpu("nvidia again", vk::PhysicalDeviceType::DISCRETE_GPU),
        ];
        assert_eq!(choose(&devices), Ok(2));
        assert_eq!(choose(&devices[..2]), Ok(1));
    }

    #[test]
    fn a_discrete_gpu_beside_an_apu_is_chosen_with_its_compute_only_queue() {
        // Queue families in the order NVIDIA's driver lists them: graphics, transfer only,
        // compute, video decode.
        let mut nvidia = gpu(
            "NVIDIA GeForce RTX 4070 Ti SUPER",
            vk::PhysicalDeviceType::DISCRETE_GPU,
        );
        let both = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        nvidia.queue_families = vec![
            family(both, 64),
            family(vk::QueueFlags::TRANSFER, 64),
            family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 64),
            family(vk::QueueFlags::VIDEO_DECODE_KHR, 64),
        ];
        let devices = [
            gpu("radv", vk::PhysicalDeviceType::INTEGRATED_GPU),
            nvidia.clone(),
            gpu("llvmpipe", vk::PhysicalDeviceType::CPU),
        ];
        assert_eq!(choose(&devices), Ok(1));
        assert_eq!(compute_queue_family(&nvidia.queue_families), Some(2));
    }

    #[test]
    fn every_reason_a_device_is_passed_over_is_reported() {
        let mut old = gpu("old", vk::PhysicalDeviceType::INTEGRATED_GPU);
        old.api_version = vk::make_api_version(0, 1, 2, 198);
        old.missing_extensions = vec!["VK_KHR_cooperative_matrix".to_string()];
        old.features.subgroup_size_control = false;
        old.features.maintenance4 = false;
        old.queue_families = vec![family(vk::QueueFlags::GRAPHICS, 64)];
        assert_eq!(
            choose(&[old]).unwrap_err(),
            [
                "old: speaks Vulkan 1.2.198, not 1.3; lacks VK_KHR_cooperative_matrix; lacks the features \
                 subgroupSizeControl, maintenance4; has no compute queue that writes timestamps"
            ]
        );
        assert_eq!(
            choose(&[]).unwrap_err(),
            ["the loader enumerated no physical devices"]
        );
    }

    #[test]
    fn missing_features_are_named_in_declaration_order() {
        assert!(Features::ALL.missing().is_empty());
        assert_eq!(
            Features::default().missing().len(),
            Features::ALL.list().len()
        );
        let mut f = Features::ALL;
        f.cooperative_matrix = false;
        f.shader_float16 = false;
        assert_eq!(f.missing(), ["shaderFloat16", "cooperativeMatrix"]);
    }
}
