use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{
    error::{Error, Result, VkCall},
    instance::Instance,
    kernel::{Kernel, KernelDesc},
    memory::Buffer,
    report::DeviceReport,
    select::{self, Features, REQUIRED_EXTENSIONS},
};

/// How long a dispatch may take before waiting for it is abandoned as a hung GPU.
const FENCE_TIMEOUT_NS: u64 = 60_000_000_000;

/// The number of timestamp ticks between two readings of a counter that keeps only its
/// low `valid_bits` bits (Vulkan's `timestampValidBits`), so the counter wrapping between
/// them does not read as a huge interval.
pub fn elapsed_ticks(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
}

/// Timestamp ticks in nanoseconds, at the device's `timestampPeriod` (nanoseconds per
/// tick, not necessarily whole).
pub fn ticks_to_ns(ticks: u64, period_ns: f32) -> f64 {
    ticks as f64 * f64::from(period_ns)
}

/// Checks a grid of workgroups against the device's per-dimension limit: every count at
/// least one and at most the limit.
pub fn check_groups(groups: [u32; 3], max: [u32; 3]) -> Result<()> {
    for (axis, (&n, &limit)) in groups.iter().zip(&max).enumerate() {
        if n == 0 || n > limit {
            return Err(Error::Unsupported(format!(
                "{n} workgroups along axis {axis} (the device allows 1 to {limit})"
            )));
        }
    }
    Ok(())
}

/// The number of workgroups of `width` invocations needed to cover `n` elements, one
/// invocation each.
pub fn groups_for(n: u64, width: u32) -> u64 {
    n.div_ceil(u64::from(width.max(1)))
}

/// What a dispatch measured: GPU time between a timestamp written before the kernel and
/// one written after it, each after all earlier work on the queue has finished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub ticks: u64,
    pub nanoseconds: f64,
}

/// The objects one submission uses, reused by every dispatch and guarded together, since
/// a queue and a command pool are externally synchronized.
struct Submit {
    queue: vk::Queue,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    timestamps: vk::QueryPool,
}

/// What every buffer and kernel keeps alive: the device, and the instance under it.
pub(crate) struct Inner {
    pub(crate) device: ash::Device,
    pub(crate) report: DeviceReport,
    /// `VK_KHR_pipeline_executable_properties`, when the device offers it.
    pub(crate) executables: Option<ash::khr::pipeline_executable_properties::Device>,
    queue_family: u32,
    submit: Mutex<Submit>,
    /// Dropped after `device` is destroyed in `Drop::drop`.
    instance: Instance,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: every buffer and kernel holds an `Arc<Inner>`, so none outlives this; the
        // queue is idled before anything is destroyed.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(s) = self.submit.get_mut() {
                self.device.destroy_query_pool(s.timestamps, None);
                self.device.destroy_fence(s.fence, None);
                self.device.destroy_command_pool(s.pool, None);
            }
            self.device.destroy_device(None);
        }
    }
}

/// The runtime's handle on the GPU: the chosen device with the runtime's features
/// enabled, and one compute queue.
#[derive(Clone)]
pub struct Context {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("device", &self.inner.report.name)
            .field("queue_family", &self.inner.queue_family)
            .field("instance", &self.inner.instance)
            .finish_non_exhaustive()
    }
}

/// The feature structures that enable exactly `f`, chained by the caller.
fn enabled(
    f: &Features,
) -> (
    vk::PhysicalDeviceVulkan11Features<'static>,
    vk::PhysicalDeviceVulkan12Features<'static>,
    vk::PhysicalDeviceVulkan13Features<'static>,
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static>,
) {
    (
        vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f.storage_buffer_16bit_access),
        vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(f.shader_float16)
            .vulkan_memory_model(f.vulkan_memory_model),
        vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(f.subgroup_size_control)
            .compute_full_subgroups(f.compute_full_subgroups)
            .synchronization2(f.synchronization2)
            .maintenance4(f.maintenance4),
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(f.cooperative_matrix),
    )
}

impl Context {
    /// Opens the loader, chooses the device ([`select::choose`]), and creates it with
    /// every feature in [`Features::ALL`] and every extension in [`REQUIRED_EXTENSIONS`].
    pub fn new() -> Result<Self> {
        let instance = Instance::new()?;
        let mut devices = instance.devices()?;
        let candidates: Vec<_> = devices.iter().map(|(_, r)| r.candidate()).collect();
        let chosen = select::choose(&candidates).map_err(Error::NoDevice)?;
        let (physical, report) = devices.swap_remove(chosen);
        let queue_family = select::compute_queue_family(&report.queue_families)
            .ok_or_else(|| Error::NoDevice(vec![format!("{}: no compute queue", report.name)]))?;

        // Optional: what the driver compiled each pipeline into (`executable`).
        let executables_ext = ash::khr::pipeline_executable_properties::NAME;
        let offers_executables = report
            .extensions
            .iter()
            .any(|e| e.as_bytes() == executables_ext.to_bytes())
            && {
                let mut f = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
                let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
                // SAFETY: `physical` came from this instance; the chain outlives the call.
                unsafe {
                    instance
                        .raw
                        .get_physical_device_features2(physical, &mut query)
                };
                f.pipeline_executable_info == vk::TRUE
            };

        let (mut f11, mut f12, mut f13, mut fcoop) = enabled(&Features::ALL);
        let mut fexec = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13)
            .push_next(&mut fcoop);
        if offers_executables {
            features = features.push_next(&mut fexec);
        }
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut extensions: Vec<_> = REQUIRED_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
        if offers_executables {
            extensions.push(executables_ext.as_ptr());
        }
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features);
        // SAFETY: `physical` came from this instance, which outlives the device (it is
        // owned by `Inner` and destroyed after it), and `info` outlives the call.
        let device =
            unsafe { instance.raw.create_device(physical, &info, None) }.call("vkCreateDevice")?;

        // The submission objects; on failure the device is destroyed before returning.
        let submit = (|| -> Result<Submit> {
            // SAFETY: the device is live; each create-info outlives its call.
            unsafe {
                let queue = device.get_device_queue(queue_family, 0);
                let pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(queue_family)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .call("vkCreateCommandPool")?;
                let commands = device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .call("vkAllocateCommandBuffers")?[0];
                let fence = device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .call("vkCreateFence")?;
                let timestamps = device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .call("vkCreateQueryPool")?;
                Ok(Submit {
                    queue,
                    pool,
                    commands,
                    fence,
                    timestamps,
                })
            }
        })();
        let submit = match submit {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: nothing created on the device survives a failed setup except
                // objects the device's destruction releases with it.
                unsafe { device.destroy_device(None) };
                return Err(e);
            }
        };
        let executables = offers_executables
            .then(|| ash::khr::pipeline_executable_properties::Device::new(&instance.raw, &device));
        Ok(Context {
            inner: Arc::new(Inner {
                device,
                report,
                executables,
                queue_family,
                submit: Mutex::new(submit),
                instance,
            }),
        })
    }

    /// What the chosen device reports about itself.
    pub fn report(&self) -> &DeviceReport {
        &self.inner.report
    }

    /// The queue family dispatches are submitted to.
    pub fn queue_family(&self) -> u32 {
        self.inner.queue_family
    }

    /// Whether pipelines keep what the driver compiled them into, for
    /// [`Kernel::executables`](crate::Kernel::executables): true when the device offers
    /// `VK_KHR_pipeline_executable_properties`.
    pub fn captures_executables(&self) -> bool {
        self.inner.executables.is_some()
    }

    /// Whether the Khronos validation layer is attached ([`VALIDATION_VAR`](crate::VALIDATION_VAR)).
    pub fn validating(&self) -> bool {
        self.inner.instance.validating()
    }

    /// Every warning and error the validation layer has reported so far.
    pub fn validation_messages(&self) -> Vec<String> {
        self.inner.instance.validation_messages()
    }

    /// A zero-filled storage buffer of `size` bytes, mapped for the CPU.
    pub fn buffer(&self, size: u64) -> Result<Buffer> {
        Buffer::new(&self.inner, size)
    }

    /// A compute pipeline from a SPIR-V module.
    pub fn kernel(&self, spirv: &[u8], desc: &KernelDesc<'_>) -> Result<Kernel> {
        Kernel::new(&self.inner, spirv, desc)
    }

    /// Runs `kernel` over a grid of `groups` workgroups with `buffers` bound in order,
    /// waits for it, and returns the GPU time between timestamps on either side of it.
    /// The buffers are borrowed mutably because the GPU may write any of them; when this
    /// returns, everything the kernel wrote is visible to the CPU.
    pub fn dispatch(
        &self,
        kernel: &Kernel,
        buffers: &mut [&mut Buffer],
        groups: [u32; 3],
    ) -> Result<Timing> {
        if buffers.len() != kernel.storage_buffers as usize {
            return Err(Error::Unsupported(format!(
                "{} buffers for a kernel that binds {}",
                buffers.len(),
                kernel.storage_buffers
            )));
        }
        check_groups(groups, self.inner.report.max_workgroup_count)?;
        let device = &self.inner.device;

        // A descriptor pool per dispatch: the simplest lifetime, and cheap next to any
        // kernel worth timing. Destroying the pool frees its set.
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(kernel.storage_buffers.max(1))];
        // SAFETY: the device is live; the create-info outlives the call.
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .call("vkCreateDescriptorPool")?;
        let result = self.record_and_run(pool, kernel, buffers, groups);
        // SAFETY: the dispatch has finished (or never started), so the set is idle.
        unsafe { device.destroy_descriptor_pool(pool, None) };
        result
    }

    fn record_and_run(
        &self,
        pool: vk::DescriptorPool,
        kernel: &Kernel,
        buffers: &[&mut Buffer],
        groups: [u32; 3],
    ) -> Result<Timing> {
        let device = &self.inner.device;
        let set_layouts = [kernel.set_layout];
        // SAFETY: the pool has room for one set of this layout.
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }
        .call("vkAllocateDescriptorSets")?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.raw)
                    .offset(0)
                    .range(b.size())]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: every buffer is live, and the set is not in use.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let submit = self
            .inner
            .submit
            .lock()
            .map_err(|_| Error::Unsupported("a poisoned queue lock".into()))?;
        let cmd = submit.commands;
        // Kernel writes made visible to the CPU once the fence signals.
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: the command buffer is not pending (every dispatch waits for its fence
        // before releasing the lock), and every object recorded is live until then.
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .call("vkResetCommandBuffer")?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .call("vkBeginCommandBuffer")?;
            device.cmd_reset_query_pool(cmd, submit.timestamps, 0, 2);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                submit.timestamps,
                0,
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                submit.timestamps,
                1,
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            device.end_command_buffer(cmd).call("vkEndCommandBuffer")?;

            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
            device
                .queue_submit2(submit.queue, &submits, submit.fence)
                .call("vkQueueSubmit2")?;
            // On a timeout the work is still pending, and this context must not be used
            // again.
            device
                .wait_for_fences(&[submit.fence], true, FENCE_TIMEOUT_NS)
                .call("vkWaitForFences")?;
            device.reset_fences(&[submit.fence]).call("vkResetFences")?;

            let mut stamps = [0u64; 2];
            device
                .get_query_pool_results(
                    submit.timestamps,
                    0,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .call("vkGetQueryPoolResults")?;
            let valid_bits = self.inner.report.queue_families[self.inner.queue_family as usize]
                .timestamp_valid_bits;
            let ticks = elapsed_ticks(stamps[0], stamps[1], valid_bits);
            Ok(Timing {
                ticks,
                nanoseconds: ticks_to_ns(ticks, self.inner.report.timestamp_period),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_ticks_survive_the_counter_wrapping() {
        assert_eq!(elapsed_ticks(100, 350, 64), 250);
        assert_eq!(elapsed_ticks(u64::MAX - 9, 5, 64), 15);
        // A 36-bit counter that wrapped between the readings.
        let top = (1u64 << 36) - 1;
        assert_eq!(elapsed_ticks(top - 2, 4, 36), 7);
        // Bits above the valid ones are not part of the reading.
        assert_eq!(
            elapsed_ticks(0xf000_0000_0000_0010, 0x0000_0000_0000_0020, 36),
            0x10
        );
    }

    #[test]
    fn ticks_convert_at_a_fractional_period() {
        // RADV on the Radeon 8060S reports 10.019036 ns per tick.
        let ns = ticks_to_ns(1_000_000, 10.019_036);
        assert!((ns - 10_019_036.0).abs() < 1.0, "{ns}");
        assert_eq!(ticks_to_ns(0, 10.019_036), 0.0);
        assert_eq!(ticks_to_ns(3, 1.0), 3.0);
    }

    #[test]
    fn grids_are_checked_against_the_device_limits() {
        let max = [u32::MAX, 65_535, 65_535];
        assert!(check_groups([1, 1, 1], max).is_ok());
        assert!(check_groups([u32::MAX, 65_535, 1], max).is_ok());
        assert!(
            matches!(check_groups([0, 1, 1], max), Err(Error::Unsupported(why)) if why.contains("axis 0"))
        );
        assert!(
            matches!(check_groups([1, 65_536, 1], max), Err(Error::Unsupported(why)) if why.contains("axis 1"))
        );
    }

    #[test]
    fn groups_cover_every_element() {
        assert_eq!(groups_for(0, 64), 0);
        assert_eq!(groups_for(1, 64), 1);
        assert_eq!(groups_for(64, 64), 1);
        assert_eq!(groups_for(65, 64), 2);
        assert_eq!(groups_for(100_003, 32), 3_126);
    }
}
