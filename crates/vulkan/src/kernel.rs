//! A compute pipeline from a SPIR-V blob: its storage-buffer bindings, its
//! specialization constants, and the subgroup size it is compiled for.
//!
//! The subgroup size is the caller's to state. Left unstated, RADV picks one per shader (wave64, or
//! wave32 for a workgroup of 32 or fewer invocations when the module is SPIR-V 1.6), and a kernel
//! whose speed or numerics depend on the width, as every cooperative-matrix kernel's do, would be
//! compiled for whichever the heuristic chose.

use std::{ffi::CStr, sync::Arc};

use ash::vk;

use crate::{
    context::Inner,
    error::{Error, Result, VkCall},
    executable::{Executable, Representation, Statistic, fixed_string, stat_value},
    spirv,
};

/// The value of one specialization constant. Each occupies four bytes of the
/// specialization data, as its SPIR-V type does (a boolean is a `VkBool32`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl SpecValue {
    fn bytes(self) -> [u8; 4] {
        match self {
            SpecValue::Bool(b) => u32::from(b).to_ne_bytes(),
            SpecValue::U32(v) => v.to_ne_bytes(),
            SpecValue::I32(v) => v.to_ne_bytes(),
            SpecValue::F32(v) => v.to_ne_bytes(),
        }
    }
}

/// The map entries and data of a `VkSpecializationInfo` for `constants`, given as
/// (SpecId, value) pairs: the values packed back to back in the order given, each entry
/// pointing at its four bytes. A SpecId given twice is refused, since which value would
/// win is the driver's guess.
pub fn pack_specialization(
    constants: &[(u32, SpecValue)],
) -> Result<(Vec<vk::SpecializationMapEntry>, Vec<u8>)> {
    let mut entries = Vec::with_capacity(constants.len());
    let mut data = Vec::with_capacity(4 * constants.len());
    for (i, &(id, value)) in constants.iter().enumerate() {
        if constants[..i].iter().any(|&(seen, _)| seen == id) {
            return Err(Error::Unsupported(format!(
                "specialization constant {id} given twice"
            )));
        }
        entries.push(vk::SpecializationMapEntry {
            constant_id: id,
            offset: data.len() as u32,
            size: 4,
        });
        data.extend_from_slice(&value.bytes());
    }
    Ok((entries, data))
}

/// Whether a compute pipeline may require `size` invocations per subgroup, given the
/// device's subgroup size control feature, its size range and the stages that accept a
/// required size. Vulkan also requires a power of two.
pub fn required_subgroup_size_allowed(
    size: u32,
    feature: bool,
    range: (u32, u32),
    stages: vk::ShaderStageFlags,
) -> bool {
    feature
        && stages.contains(vk::ShaderStageFlags::COMPUTE)
        && size.is_power_of_two()
        && (range.0..=range.1).contains(&size)
}

/// How to build a [`Kernel`] from a SPIR-V module.
#[derive(Clone, Debug)]
pub struct KernelDesc<'a> {
    /// The entry point's name.
    pub entry: &'a CStr,
    /// Storage buffers the kernel binds: bindings 0 to n - 1 of descriptor set 0, in the
    /// order a dispatch passes them.
    pub storage_buffers: u32,
    /// The subgroup size the pipeline must run at, or `None` to leave it to the driver.
    pub subgroup_size: Option<u32>,
    /// Require every subgroup to be full (`VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`):
    /// the workgroup's width must then be a multiple of the subgroup size.
    pub full_subgroups: bool,
    /// (SpecId, value) pairs.
    pub constants: &'a [(u32, SpecValue)],
}

impl Default for KernelDesc<'_> {
    fn default() -> Self {
        KernelDesc {
            entry: c"main",
            storage_buffers: 0,
            subgroup_size: None,
            full_subgroups: false,
            constants: &[],
        }
    }
}

/// A compute pipeline and the layout it binds through.
pub struct Kernel {
    pub(crate) ctx: Arc<Inner>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) storage_buffers: u32,
    subgroup_size: Option<u32>,
    spirv: spirv::Header,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("storage_buffers", &self.storage_buffers)
            .field("subgroup_size", &self.subgroup_size)
            .field("spirv", &self.spirv)
            .finish_non_exhaustive()
    }
}

impl Kernel {
    pub(crate) fn new(ctx: &Arc<Inner>, spirv: &[u8], desc: &KernelDesc<'_>) -> Result<Self> {
        let report = &ctx.report;
        if let Some(size) = desc.subgroup_size
            && !required_subgroup_size_allowed(
                size,
                report.features.subgroup_size_control,
                (report.min_subgroup_size, report.max_subgroup_size),
                report.required_subgroup_size_stages,
            )
        {
            return Err(Error::Unsupported(format!(
                "a required subgroup size of {size} (the device allows {} to {} in {:?})",
                report.min_subgroup_size,
                report.max_subgroup_size,
                report.required_subgroup_size_stages
            )));
        }
        let (words, header) = spirv::words(spirv)?;
        let (entries, data) = pack_specialization(desc.constants)?;
        let device = &ctx.device;

        // Owned from here on: a failure below releases what exists through `Drop`
        // (destroying a null handle is a no-op).
        let mut kernel = Kernel {
            ctx: Arc::clone(ctx),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            storage_buffers: desc.storage_buffers,
            subgroup_size: desc.subgroup_size,
            spirv: header,
        };
        let bindings: Vec<_> = (0..desc.storage_buffers)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        // SAFETY: the device is live and each create-info outlives its call.
        kernel.set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .call("vkCreateDescriptorSetLayout")?;
        let set_layouts = [kernel.set_layout];
        // SAFETY: as above.
        kernel.layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .call("vkCreatePipelineLayout")?;

        // SAFETY: `words` is a SPIR-V module whose header was checked.
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .call("vkCreateShaderModule")?;
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(desc.subgroup_size.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(desc.entry)
            .specialization_info(&specialization);
        if desc.full_subgroups {
            stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }
        if desc.subgroup_size.is_some() {
            stage = stage.push_next(&mut required);
        }
        // Ask the driver to keep what it compiled, when it can say (`executable`).
        let capture = if ctx.executables.is_some() {
            vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
                | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let info = vk::ComputePipelineCreateInfo::default()
            .flags(capture)
            .stage(stage)
            .layout(kernel.layout);
        // SAFETY: every handle and pointer in `info` is live for the call; the module may
        // be destroyed once the pipeline exists.
        let created =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: the module is no longer needed, whatever the pipeline's fate.
        unsafe { device.destroy_shader_module(module, None) };
        kernel.pipeline = created
            .map_err(|(_, result)| result)
            .call("vkCreateComputePipelines")?[0];
        Ok(kernel)
    }

    /// The subgroup size the pipeline was required to run at, if any.
    pub fn subgroup_size(&self) -> Option<u32> {
        self.subgroup_size
    }

    /// The module's SPIR-V header.
    pub fn spirv(&self) -> spirv::Header {
        self.spirv
    }

    /// What the driver compiled this pipeline into: per executable, the subgroup size it
    /// runs at, the driver's statistics and its internal representations. An error when
    /// the device does not offer `VK_KHR_pipeline_executable_properties`.
    pub fn executables(&self) -> Result<Vec<Executable>> {
        let ext = self.ctx.executables.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "pipeline introspection (VK_KHR_pipeline_executable_properties)".to_string(),
            )
        })?;
        let info = vk::PipelineInfoKHR::default().pipeline(self.pipeline);
        // SAFETY: the pipeline is live and was created with the capture flags.
        let properties = unsafe { ext.get_pipeline_executable_properties(&info) }
            .call("vkGetPipelineExecutablePropertiesKHR")?;
        properties
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let which = vk::PipelineExecutableInfoKHR::default()
                    .pipeline(self.pipeline)
                    .executable_index(i as u32);
                // SAFETY: as above, for an executable index the driver listed.
                let statistics = unsafe { ext.get_pipeline_executable_statistics(&which) }
                    .call("vkGetPipelineExecutableStatisticsKHR")?
                    .iter()
                    .map(|s| Statistic {
                        name: fixed_string(&s.name),
                        description: fixed_string(&s.description),
                        value: stat_value(s.format, s.value),
                    })
                    .collect();
                Ok(Executable {
                    name: fixed_string(&p.name),
                    description: fixed_string(&p.description),
                    subgroup_size: p.subgroup_size,
                    statistics,
                    representations: internal_representations(ext, &which)?,
                })
            })
            .collect()
    }
}

/// The internal representations of one executable: counted, sized, then copied, since
/// each one's data is a buffer the caller allocates.
fn internal_representations(
    ext: &ash::khr::pipeline_executable_properties::Device,
    which: &vk::PipelineExecutableInfoKHR<'_>,
) -> Result<Vec<Representation>> {
    let get = ext
        .fp()
        .get_pipeline_executable_internal_representations_khr;
    const CALL: &str = "vkGetPipelineExecutableInternalRepresentationsKHR";
    let mut count = 0u32;
    // SAFETY: a count query, with no array.
    unsafe { get(ext.device(), which, &mut count, std::ptr::null_mut()) }
        .result()
        .call(CALL)?;
    let mut reps = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
    // SAFETY: `reps` holds `count` structures; with null data pointers the driver writes
    // each one's size.
    unsafe { get(ext.device(), which, &mut count, reps.as_mut_ptr()) }
        .result()
        .call(CALL)?;
    let mut data: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
    for (r, d) in reps.iter_mut().zip(&mut data) {
        r.p_data = d.as_mut_ptr().cast();
    }
    // SAFETY: each data pointer addresses `data_size` writable bytes, alive for the call.
    unsafe { get(ext.device(), which, &mut count, reps.as_mut_ptr()) }
        .result()
        .call(CALL)?;
    Ok(reps
        .iter()
        .zip(data)
        .map(|(r, d)| {
            let end = d.iter().position(|&b| b == 0).unwrap_or(d.len());
            Representation {
                name: fixed_string(&r.name),
                description: fixed_string(&r.description),
                text: (r.is_text == vk::TRUE)
                    .then(|| String::from_utf8_lossy(&d[..end]).into_owned()),
                bytes: d.len(),
            }
        })
        .collect())
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: dispatches wait for the GPU, so the pipeline is idle; null handles are
        // no-ops.
        unsafe {
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_pack_back_to_back_in_the_order_given() {
        let constants = [
            (0, SpecValue::U32(64)),
            (2, SpecValue::F32(-1.25)),
            (1, SpecValue::F32(2.5)),
            (7, SpecValue::Bool(true)),
            (9, SpecValue::I32(-3)),
        ];
        let (entries, data) = pack_specialization(&constants).unwrap();
        let ids: Vec<_> = entries
            .iter()
            .map(|e| (e.constant_id, e.offset, e.size))
            .collect();
        assert_eq!(
            ids,
            [(0, 0, 4), (2, 4, 4), (1, 8, 4), (7, 12, 4), (9, 16, 4)]
        );
        let word = |i: usize| {
            [
                data[4 * i],
                data[4 * i + 1],
                data[4 * i + 2],
                data[4 * i + 3],
            ]
        };
        assert_eq!(u32::from_ne_bytes(word(0)), 64);
        assert_eq!(f32::from_ne_bytes(word(1)), -1.25);
        assert_eq!(f32::from_ne_bytes(word(2)), 2.5);
        assert_eq!(u32::from_ne_bytes(word(3)), 1, "a boolean is a VkBool32");
        assert_eq!(i32::from_ne_bytes(word(4)), -3);
        assert_eq!(data.len(), 20);
    }

    #[test]
    fn no_constants_is_an_empty_specialization() {
        let (entries, data) = pack_specialization(&[]).unwrap();
        assert!(entries.is_empty() && data.is_empty());
    }

    #[test]
    fn a_constant_given_twice_is_refused() {
        let constants = [
            (1, SpecValue::F32(1.0)),
            (0, SpecValue::U32(64)),
            (1, SpecValue::F32(2.0)),
        ];
        assert!(
            matches!(pack_specialization(&constants), Err(Error::Unsupported(why)) if why.contains("1 given twice"))
        );
    }

    #[test]
    fn required_subgroup_sizes_must_be_powers_of_two_in_range_for_compute() {
        let compute = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;
        assert!(required_subgroup_size_allowed(32, true, (32, 64), compute));
        assert!(required_subgroup_size_allowed(64, true, (32, 64), compute));
        assert!(!required_subgroup_size_allowed(16, true, (32, 64), compute));
        assert!(!required_subgroup_size_allowed(
            128,
            true,
            (32, 64),
            compute
        ));
        assert!(!required_subgroup_size_allowed(48, true, (32, 64), compute));
        assert!(!required_subgroup_size_allowed(
            32,
            false,
            (32, 64),
            compute
        ));
        assert!(!required_subgroup_size_allowed(
            32,
            true,
            (32, 64),
            vk::ShaderStageFlags::FRAGMENT
        ));
    }
}
