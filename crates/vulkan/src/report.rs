//! What a physical device reports about itself: the numbers a kernel's shape and a roofline are
//! reasoned from, such as matrix-instruction shapes and types, subgroup sizes, compute limits, the
//! timestamp period, memory heaps and types, and the shader core's size.
//!
//! [`DeviceReport`] is plain data, so device selection ([`crate::select`]) runs on it, and its text
//! form is what the `properties` example prints. Values ash 0.38 has no name for, its headers being
//! Vulkan 1.3.281's, are declared here from the Vulkan 1.4 headers.

use std::{
    borrow::Cow,
    ffi::{CStr, c_void},
    fmt,
};

use ash::vk;

use crate::{
    error::{Result, VkCall},
    instance::Instance,
    kernel::required_subgroup_size_allowed,
    select::{Candidate, Features, QueueFamily, REQUIRED_EXTENSIONS, version_string},
};

/// `VK_KHR_shader_bfloat16`, newer than ash's headers.
const SHADER_BFLOAT16: &CStr = c"VK_KHR_shader_bfloat16";
const SHADER_BFLOAT16_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_141_000);

/// Extensions the report names when present, because a kernel or an instrument would use
/// them. The full list is in [`DeviceReport::extensions`].
const NOTABLE_EXTENSIONS: &[&str] = &[
    "VK_KHR_cooperative_matrix",
    "VK_NV_cooperative_matrix2",
    "VK_KHR_shader_bfloat16",
    "VK_EXT_shader_float8",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_shader_float_controls2",
    "VK_EXT_subgroup_size_control",
    "VK_EXT_memory_budget",
    "VK_AMD_device_coherent_memory",
    "VK_AMD_shader_core_properties",
    "VK_AMD_shader_core_properties2",
    "VK_KHR_pipeline_executable_properties",
    "VK_EXT_external_memory_host",
    "VK_KHR_shader_subgroup_extended_types",
];

/// `VkPhysicalDeviceShaderBfloat16FeaturesKHR`, declared here because ash 0.38 predates it.
#[repr(C)]
struct ShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

// SAFETY: the struct has the C layout the Vulkan 1.4 headers declare, begins with `sType`
// and `pNext`, and `VkPhysicalDeviceFeatures2` is a structure it may extend.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for ShaderBfloat16Features {}

/// One cooperative-matrix configuration: `result = A x B + C`, with A of `m x k`, B of
/// `k x n`, C and the result of `m x n`. Component types are raw `VkComponentTypeKHR`
/// values, named by [`component_type_name`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CooperativeMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub result: i32,
    pub saturating: bool,
    pub scope: i32,
}

/// One memory heap, with the `VK_EXT_memory_budget` figures when the device has them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryHeap {
    pub size: u64,
    pub flags: vk::MemoryHeapFlags,
    /// How much of the heap this process may use before allocations start to fail or
    /// degrade, as the driver estimates it now.
    pub budget: Option<u64>,
    /// How much of the heap this process uses now.
    pub usage: Option<u64>,
}

/// One memory type: its property flags and the heap it allocates from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryType {
    pub flags: vk::MemoryPropertyFlags,
    pub heap: u32,
}

/// The shader core's organization, from `VK_AMD_shader_core_properties(2)`: what turns a
/// clock rate into a peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderCore {
    pub shader_engines: u32,
    pub shader_arrays_per_engine: u32,
    pub compute_units_per_array: u32,
    pub simds_per_compute_unit: u32,
    pub wavefronts_per_simd: u32,
    pub wavefront_size: u32,
    pub vgprs_per_simd: u32,
    pub sgprs_per_simd: u32,
    pub active_compute_units: Option<u32>,
}

/// Float-control properties that decide whether a kernel's results can match a CPU
/// reference bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatControls {
    pub denorm_preserve_f16: bool,
    pub denorm_preserve_f32: bool,
    pub denorm_flush_f16: bool,
    pub denorm_flush_f32: bool,
    pub rounding_rte_f16: bool,
    pub rounding_rte_f32: bool,
    pub rounding_rtz_f16: bool,
    pub rounding_rtz_f32: bool,
}

/// Everything the runtime and its instruments read from one physical device.
#[derive(Clone, Debug)]
pub struct DeviceReport {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub driver_name: String,
    pub driver_info: String,
    pub conformance: String,

    /// The subgroup size a pipeline gets when it neither requires one nor allows it to vary.
    pub subgroup_size: u32,
    pub subgroup_stages: vk::ShaderStageFlags,
    pub subgroup_operations: vk::SubgroupFeatureFlags,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_compute_workgroup_subgroups: u32,
    /// Stages in which a pipeline may require a subgroup size.
    pub required_subgroup_size_stages: vk::ShaderStageFlags,

    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_shared_memory: u32,
    pub max_storage_buffer_range: u32,
    pub max_memory_allocation_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub non_coherent_atom_size: u64,
    pub min_memory_map_alignment: usize,

    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    pub timestamp_compute_and_graphics: bool,
    pub queue_families: Vec<QueueFamily>,

    pub heaps: Vec<MemoryHeap>,
    pub types: Vec<MemoryType>,

    /// Empty when the device lacks `VK_KHR_cooperative_matrix`.
    pub cooperative_matrix: Vec<CooperativeMatrixShape>,
    pub cooperative_matrix_stages: vk::ShaderStageFlags,
    /// The features the runtime requires, as the device reports them.
    pub features: Features,
    /// `shaderBFloat16Type`, `shaderBFloat16DotProduct`, `shaderBFloat16CooperativeMatrix`,
    /// when the device has `VK_KHR_shader_bfloat16`.
    pub bfloat16: Option<[bool; 3]>,
    pub float_controls: FloatControls,
    pub shader_core: Option<ShaderCore>,
    /// Every device extension, sorted.
    pub extensions: Vec<String>,
    /// The instance-level API version of the loader this report was read through.
    pub loader_version: u32,
}

fn c_string(chars: std::result::Result<&CStr, std::ffi::FromBytesUntilNulError>) -> String {
    chars
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn yes(b: vk::Bool32) -> bool {
    b == vk::TRUE
}

impl DeviceReport {
    /// Reads everything the report holds from `pd`.
    pub fn query(instance: &Instance, pd: vk::PhysicalDevice) -> Result<Self> {
        let raw = &instance.raw;
        // SAFETY: `pd` was enumerated from this live instance. The same holds for each
        // query below, where every structure in a pNext chain is one the device
        // advertises the extension for and outlives the call that fills it.
        let mut extensions: Vec<String> = unsafe { raw.enumerate_device_extension_properties(pd) }
            .call("vkEnumerateDeviceExtensionProperties")?
            .iter()
            .map(|e| c_string(e.extension_name_as_c_str()))
            .collect();
        extensions.sort();
        let has = |name: &CStr| extensions.iter().any(|e| e.as_bytes() == name.to_bytes());
        let has_coop = has(ash::khr::cooperative_matrix::NAME);
        let has_core = has(ash::amd::shader_core_properties::NAME);
        let has_core2 = has(ash::amd::shader_core_properties2::NAME);
        let has_budget = has(ash::ext::memory_budget::NAME);
        let has_bf16 = has(SHADER_BFLOAT16);

        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut pcoop = vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
        let mut pcore = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        let mut pcore2 = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut p11)
            .push_next(&mut p12)
            .push_next(&mut p13);
        if has_coop {
            props2 = props2.push_next(&mut pcoop);
        }
        if has_core {
            props2 = props2.push_next(&mut pcore);
        }
        if has_core2 {
            props2 = props2.push_next(&mut pcore2);
        }
        // SAFETY: as for the extension query above.
        unsafe { raw.get_physical_device_properties2(pd, &mut props2) };
        let p = props2.properties;

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut fcoop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut fbf16 = ShaderBfloat16Features {
            s_type: SHADER_BFLOAT16_FEATURES,
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        };
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if has_coop {
            features2 = features2.push_next(&mut fcoop);
        }
        if has_bf16 {
            features2 = features2.push_next(&mut fbf16);
        }
        // SAFETY: as above.
        unsafe { raw.get_physical_device_features2(pd, &mut features2) };

        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut memory2 = vk::PhysicalDeviceMemoryProperties2::default();
        if has_budget {
            memory2 = memory2.push_next(&mut budget);
        }
        // SAFETY: as above.
        unsafe { raw.get_physical_device_memory_properties2(pd, &mut memory2) };
        let memory = memory2.memory_properties;

        // SAFETY: as above.
        let queue_families = unsafe { raw.get_physical_device_queue_family_properties(pd) }
            .iter()
            .map(|q| QueueFamily {
                flags: q.queue_flags,
                count: q.queue_count,
                timestamp_valid_bits: q.timestamp_valid_bits,
            })
            .collect();

        let cooperative_matrix = if has_coop {
            let coop = ash::khr::cooperative_matrix::Instance::new(&instance.entry, raw);
            // SAFETY: as above; the device advertises the extension.
            unsafe { coop.get_physical_device_cooperative_matrix_properties(pd) }
                .call("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")?
                .iter()
                .map(|c| CooperativeMatrixShape {
                    m: c.m_size,
                    n: c.n_size,
                    k: c.k_size,
                    a: c.a_type.as_raw(),
                    b: c.b_type.as_raw(),
                    c: c.c_type.as_raw(),
                    result: c.result_type.as_raw(),
                    saturating: yes(c.saturating_accumulation),
                    scope: c.scope.as_raw(),
                })
                .collect()
        } else {
            Vec::new()
        };

        let heaps = memory.memory_heaps_as_slice();
        let report = DeviceReport {
            name: c_string(p.device_name_as_c_str()),
            device_type: p.device_type,
            vendor_id: p.vendor_id,
            device_id: p.device_id,
            api_version: p.api_version,
            driver_version: p.driver_version,
            driver_name: c_string(p12.driver_name_as_c_str()),
            driver_info: c_string(p12.driver_info_as_c_str()),
            conformance: {
                let c = p12.conformance_version;
                format!("{}.{}.{}.{}", c.major, c.minor, c.subminor, c.patch)
            },
            subgroup_size: p11.subgroup_size,
            subgroup_stages: p11.subgroup_supported_stages,
            subgroup_operations: p11.subgroup_supported_operations,
            min_subgroup_size: p13.min_subgroup_size,
            max_subgroup_size: p13.max_subgroup_size,
            max_compute_workgroup_subgroups: p13.max_compute_workgroup_subgroups,
            required_subgroup_size_stages: p13.required_subgroup_size_stages,
            max_workgroup_count: p.limits.max_compute_work_group_count,
            max_workgroup_size: p.limits.max_compute_work_group_size,
            max_workgroup_invocations: p.limits.max_compute_work_group_invocations,
            max_shared_memory: p.limits.max_compute_shared_memory_size,
            max_storage_buffer_range: p.limits.max_storage_buffer_range,
            max_memory_allocation_size: p11.max_memory_allocation_size,
            max_buffer_size: p13.max_buffer_size,
            min_storage_buffer_offset_alignment: p.limits.min_storage_buffer_offset_alignment,
            non_coherent_atom_size: p.limits.non_coherent_atom_size,
            min_memory_map_alignment: p.limits.min_memory_map_alignment,
            timestamp_period: p.limits.timestamp_period,
            timestamp_compute_and_graphics: yes(p.limits.timestamp_compute_and_graphics),
            queue_families,
            heaps: heaps
                .iter()
                .enumerate()
                .map(|(i, h)| MemoryHeap {
                    size: h.size,
                    flags: h.flags,
                    budget: has_budget.then(|| budget.heap_budget[i]),
                    usage: has_budget.then(|| budget.heap_usage[i]),
                })
                .collect(),
            types: memory
                .memory_types_as_slice()
                .iter()
                .map(|t| MemoryType {
                    flags: t.property_flags,
                    heap: t.heap_index,
                })
                .collect(),
            cooperative_matrix,
            cooperative_matrix_stages: pcoop.cooperative_matrix_supported_stages,
            features: Features {
                shader_float16: yes(f12.shader_float16),
                storage_buffer_16bit_access: yes(f11.storage_buffer16_bit_access),
                subgroup_size_control: yes(f13.subgroup_size_control),
                compute_full_subgroups: yes(f13.compute_full_subgroups),
                synchronization2: yes(f13.synchronization2),
                maintenance4: yes(f13.maintenance4),
                cooperative_matrix: has_coop && yes(fcoop.cooperative_matrix),
                vulkan_memory_model: yes(f12.vulkan_memory_model),
            },
            bfloat16: has_bf16.then(|| {
                [
                    yes(fbf16.shader_bfloat16_type),
                    yes(fbf16.shader_bfloat16_dot_product),
                    yes(fbf16.shader_bfloat16_cooperative_matrix),
                ]
            }),
            float_controls: FloatControls {
                denorm_preserve_f16: yes(p12.shader_denorm_preserve_float16),
                denorm_preserve_f32: yes(p12.shader_denorm_preserve_float32),
                denorm_flush_f16: yes(p12.shader_denorm_flush_to_zero_float16),
                denorm_flush_f32: yes(p12.shader_denorm_flush_to_zero_float32),
                rounding_rte_f16: yes(p12.shader_rounding_mode_rte_float16),
                rounding_rte_f32: yes(p12.shader_rounding_mode_rte_float32),
                rounding_rtz_f16: yes(p12.shader_rounding_mode_rtz_float16),
                rounding_rtz_f32: yes(p12.shader_rounding_mode_rtz_float32),
            },
            shader_core: has_core.then_some(ShaderCore {
                shader_engines: pcore.shader_engine_count,
                shader_arrays_per_engine: pcore.shader_arrays_per_engine_count,
                compute_units_per_array: pcore.compute_units_per_shader_array,
                simds_per_compute_unit: pcore.simd_per_compute_unit,
                wavefronts_per_simd: pcore.wavefronts_per_simd,
                wavefront_size: pcore.wavefront_size,
                vgprs_per_simd: pcore.vgprs_per_simd,
                sgprs_per_simd: pcore.sgprs_per_simd,
                active_compute_units: has_core2.then_some(pcore2.active_compute_unit_count),
            }),
            loader_version: instance.loader_version(),
            extensions,
        };
        Ok(report)
    }

    /// Whether a compute pipeline may require subgroups of `size` invocations.
    pub fn can_require_subgroup_size(&self, size: u32) -> bool {
        required_subgroup_size_allowed(
            size,
            self.features.subgroup_size_control,
            (self.min_subgroup_size, self.max_subgroup_size),
            self.required_subgroup_size_stages,
        )
    }

    /// The device as device selection sees it.
    pub fn candidate(&self) -> Candidate {
        Candidate {
            name: self.name.clone(),
            device_type: self.device_type,
            api_version: self.api_version,
            missing_extensions: REQUIRED_EXTENSIONS
                .iter()
                .map(|e| e.to_string_lossy().into_owned())
                .filter(|e| !self.extensions.contains(e))
                .collect(),
            features: self.features,
            queue_families: self.queue_families.clone(),
        }
    }
}

/// The name of a `VkComponentTypeKHR` value, from the Vulkan 1.4 headers; an unknown value
/// is printed as its number.
pub fn component_type_name(raw: i32) -> Cow<'static, str> {
    Cow::Borrowed(match raw {
        0 => "f16",
        1 => "f32",
        2 => "f64",
        3 => "s8",
        4 => "s16",
        5 => "s32",
        6 => "s64",
        7 => "u8",
        8 => "u16",
        9 => "u32",
        10 => "u64",
        1_000_141_000 => "bf16",
        1_000_491_000 => "s8-packed",
        1_000_491_001 => "u8-packed",
        1_000_491_002 => "f8e4m3",
        1_000_491_003 => "f8e5m2",
        other => return Cow::Owned(format!("component type {other}")),
    })
}

/// The name of a `VkScopeKHR` value.
pub fn scope_name(raw: i32) -> Cow<'static, str> {
    Cow::Borrowed(match raw {
        1 => "device",
        2 => "workgroup",
        3 => "subgroup",
        5 => "queue family",
        other => return Cow::Owned(format!("scope {other}")),
    })
}

/// A byte count in binary units, exact to two decimals: `512 MiB`, `31.25 GiB`.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if value.fract() == 0.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Mesa's driver version: `VK_MAKE_VERSION(major, minor, patch)` of the Mesa release.
pub fn mesa_version(v: u32) -> String {
    format!("{}.{}.{}", v >> 22, (v >> 12) & 0x3ff, v & 0xfff)
}

/// Memory property flags as their short Vulkan names joined by `|`.
pub fn memory_flags(flags: vk::MemoryPropertyFlags) -> String {
    let names = [
        (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
        (
            vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            "LAZILY_ALLOCATED",
        ),
        (vk::MemoryPropertyFlags::PROTECTED, "PROTECTED"),
        (
            vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD,
            "DEVICE_COHERENT_AMD",
        ),
        (
            vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
            "DEVICE_UNCACHED_AMD",
        ),
    ];
    let set: Vec<&str> = names
        .iter()
        .filter(|(f, _)| flags.contains(*f))
        .map(|(_, n)| *n)
        .collect();
    if set.is_empty() {
        "-".to_string()
    } else {
        set.join("|")
    }
}

fn device_type_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "other",
    }
}

fn on(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

impl fmt::Display for DeviceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "device      {} ({})",
            self.name,
            device_type_name(self.device_type)
        )?;
        writeln!(
            f,
            "ids         vendor {:#06x}, device {:#06x}",
            self.vendor_id, self.device_id
        )?;
        writeln!(
            f,
            "driver      {} -- {} (driver version {}, conformance {})",
            self.driver_name,
            self.driver_info,
            mesa_version(self.driver_version),
            self.conformance
        )?;
        writeln!(f, "API         {}", version_string(self.api_version))?;
        writeln!(
            f,
            "loader      instance API {}",
            version_string(self.loader_version)
        )?;

        writeln!(f, "\nsubgroups")?;
        writeln!(f, "  default size                 {}", self.subgroup_size)?;
        writeln!(
            f,
            "  size range                   {} to {}",
            self.min_subgroup_size, self.max_subgroup_size
        )?;
        writeln!(
            f,
            "  required-size stages         {:?}",
            self.required_subgroup_size_stages
        )?;
        writeln!(
            f,
            "  max subgroups per workgroup  {}",
            self.max_compute_workgroup_subgroups
        )?;
        writeln!(
            f,
            "  stages                       {:?}",
            self.subgroup_stages
        )?;
        writeln!(
            f,
            "  operations                   {:?}",
            self.subgroup_operations
        )?;

        writeln!(
            f,
            "\ncooperative matrix (stages {:?})",
            self.cooperative_matrix_stages
        )?;
        if self.cooperative_matrix.is_empty() {
            writeln!(f, "  none")?;
        }
        for c in &self.cooperative_matrix {
            writeln!(
                f,
                "  {}x{}x{}  A {:>4}  B {:>4}  C {:>4}  result {:>4}  {} scope{}",
                c.m,
                c.n,
                c.k,
                component_type_name(c.a),
                component_type_name(c.b),
                component_type_name(c.c),
                component_type_name(c.result),
                scope_name(c.scope),
                if c.saturating { ", saturating" } else { "" }
            )?;
        }

        writeln!(f, "\nfeatures the runtime requires")?;
        for (name, set) in self.features.list() {
            writeln!(f, "  {name:<28} {}", on(set))?;
        }
        match self.bfloat16 {
            Some([ty, dot, coop]) => writeln!(
                f,
                "  bfloat16                     type {}, dot product {}, cooperative matrix {}",
                on(ty),
                on(dot),
                on(coop)
            )?,
            None => writeln!(
                f,
                "  bfloat16                     no VK_KHR_shader_bfloat16"
            )?,
        }
        let fc = &self.float_controls;
        writeln!(
            f,
            "  denormals f16                preserve {}, flush {}",
            on(fc.denorm_preserve_f16),
            on(fc.denorm_flush_f16)
        )?;
        writeln!(
            f,
            "  denormals f32                preserve {}, flush {}",
            on(fc.denorm_preserve_f32),
            on(fc.denorm_flush_f32)
        )?;
        writeln!(
            f,
            "  rounding f16                 RTE {}, RTZ {}",
            on(fc.rounding_rte_f16),
            on(fc.rounding_rtz_f16)
        )?;
        writeln!(
            f,
            "  rounding f32                 RTE {}, RTZ {}",
            on(fc.rounding_rte_f32),
            on(fc.rounding_rtz_f32)
        )?;

        writeln!(f, "\ncompute limits")?;
        writeln!(
            f,
            "  workgroup count              {:?}",
            self.max_workgroup_count
        )?;
        writeln!(
            f,
            "  workgroup size               {:?}",
            self.max_workgroup_size
        )?;
        writeln!(
            f,
            "  workgroup invocations        {}",
            self.max_workgroup_invocations
        )?;
        writeln!(
            f,
            "  shared memory                {}",
            bytes(u64::from(self.max_shared_memory))
        )?;
        writeln!(
            f,
            "  storage buffer range         {}",
            bytes(u64::from(self.max_storage_buffer_range))
        )?;
        writeln!(
            f,
            "  memory allocation size       {}",
            bytes(self.max_memory_allocation_size)
        )?;
        writeln!(
            f,
            "  buffer size                  {}",
            bytes(self.max_buffer_size)
        )?;
        writeln!(
            f,
            "  storage offset alignment     {}",
            self.min_storage_buffer_offset_alignment
        )?;
        writeln!(
            f,
            "  non-coherent atom size       {}",
            self.non_coherent_atom_size
        )?;
        writeln!(
            f,
            "  memory map alignment         {}",
            self.min_memory_map_alignment
        )?;

        if let Some(core) = &self.shader_core {
            writeln!(f, "\nshader core")?;
            writeln!(
                f,
                "  {} engines x {} arrays x {} CUs, {} SIMDs per CU, {} waves per SIMD, native wave {}",
                core.shader_engines,
                core.shader_arrays_per_engine,
                core.compute_units_per_array,
                core.simds_per_compute_unit,
                core.wavefronts_per_simd,
                core.wavefront_size
            )?;
            writeln!(
                f,
                "  active CUs {}, VGPRs per SIMD {}, SGPRs per SIMD {}",
                core.active_compute_units
                    .map_or_else(|| "?".to_string(), |n| n.to_string()),
                core.vgprs_per_simd,
                core.sgprs_per_simd
            )?;
        }

        writeln!(
            f,
            "\ntimestamps  period {} ns per tick, compute and graphics {}",
            self.timestamp_period,
            on(self.timestamp_compute_and_graphics)
        )?;
        writeln!(f, "\nqueue families")?;
        for (i, q) in self.queue_families.iter().enumerate() {
            writeln!(
                f,
                "  {i}  {:?} x{}, timestamp bits {}",
                q.flags, q.count, q.timestamp_valid_bits
            )?;
        }

        writeln!(f, "\nmemory heaps")?;
        for (i, h) in self.heaps.iter().enumerate() {
            write!(f, "  {i}  {:>10}  {:?}", bytes(h.size), h.flags)?;
            if let (Some(budget), Some(usage)) = (h.budget, h.usage) {
                write!(f, "  (budget {}, in use {})", bytes(budget), bytes(usage))?;
            }
            writeln!(f)?;
        }
        writeln!(f, "memory types")?;
        for (i, t) in self.types.iter().enumerate() {
            writeln!(f, "  {i:>2}  heap {}  {}", t.heap, memory_flags(t.flags))?;
        }

        let notable: Vec<&str> = NOTABLE_EXTENSIONS
            .iter()
            .copied()
            .filter(|n| self.extensions.iter().any(|e| e == n))
            .collect();
        let absent: Vec<&str> = NOTABLE_EXTENSIONS
            .iter()
            .copied()
            .filter(|n| !self.extensions.iter().any(|e| e == n))
            .collect();
        writeln!(f, "\nextensions  {} in all", self.extensions.len())?;
        writeln!(f, "  present  {}", notable.join(" "))?;
        writeln!(
            f,
            "  absent   {}",
            if absent.is_empty() {
                "-".to_string()
            } else {
                absent.join(" ")
            }
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_types_are_named_from_the_1_4_headers() {
        assert_eq!(component_type_name(0), "f16");
        assert_eq!(component_type_name(1), "f32");
        assert_eq!(component_type_name(3), "s8");
        assert_eq!(component_type_name(5), "s32");
        assert_eq!(component_type_name(7), "u8");
        assert_eq!(component_type_name(1_000_141_000), "bf16");
        assert_eq!(component_type_name(1_000_491_002), "f8e4m3");
        assert_eq!(component_type_name(42), "component type 42");
        assert_eq!(scope_name(3), "subgroup");
        assert_eq!(scope_name(9), "scope 9");
    }

    #[test]
    fn byte_counts_read_in_binary_units() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(64 * 1024), "64 KiB");
        assert_eq!(bytes(512 << 20), "512 MiB");
        assert_eq!(bytes((31 << 30) + (1 << 28)), "31.25 GiB");
    }

    #[test]
    fn mesa_versions_unpack_from_vk_make_version() {
        // VK_MAKE_VERSION(26, 1, 8), as Mesa encodes its release.
        assert_eq!(mesa_version((26 << 22) | (1 << 12) | 8), "26.1.8");
    }

    #[test]
    fn memory_flags_print_by_name() {
        let f = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        assert_eq!(memory_flags(f), "DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT");
        assert_eq!(memory_flags(vk::MemoryPropertyFlags::empty()), "-");
    }
}
