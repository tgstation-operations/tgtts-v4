//! A whole chain of dispatches recorded once into one command buffer and submitted as often as
//! needed: a model's step, whose shapes do not change from one step to the next, costs the host
//! one submission and one wait, not a recording per kernel.
//!
//! The recording owns the buffers it binds, so none of them can be freed, or touched by the host,
//! while the GPU may be using it: [`Recording::submit`] takes it mutably and waits for the GPU
//! before returning, [`Recording::start`] returns a [`Submission`] that holds it until waited for,
//! and the host reaches the buffers only through the recording between submissions. A value that
//! changes from one submission to the next, such as a position, lives in one of those buffers and
//! is written there before each submission, not baked into the commands. Buffers are bound through
//! descriptor sets written once, at their ranges as given; nothing in the commands is a raw device
//! address.
//!
//! Every submission is timed from a timestamp before its first dispatch to one after its last
//! ([`Recording::elapsed`]), which costs two timestamps a submission; a recording made timed also
//! times each dispatch ([`Recording::timings`]).

use std::{mem::ManuallyDrop, sync::Arc};

use ash::vk;

use crate::{
    context::{FENCE_TIMEOUT_NS, Inner, Timing, check_groups, elapsed_ticks, ticks_to_ns},
    error::{Error, Result, VkCall},
    kernel::Kernel,
    memory::Buffer,
};

/// A range of one of a recording's buffers, bound to one binding of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The buffer's index among those the recording owns.
    pub buffer: usize,
    pub offset: u64,
    /// `None` for the rest of the buffer from `offset`.
    pub size: Option<u64>,
}

impl Binding {
    /// The whole of buffer `buffer`.
    pub fn whole(buffer: usize) -> Self {
        Binding {
            buffer,
            offset: 0,
            size: None,
        }
    }

    /// `size` bytes of buffer `buffer` from `offset`.
    pub fn range(buffer: usize, offset: u64, size: u64) -> Self {
        Binding {
            buffer,
            offset,
            size: Some(size),
        }
    }
}

/// One dispatch of a recording.
#[derive(Clone, Debug)]
pub struct Dispatch {
    pub kernel: Arc<Kernel>,
    /// One per storage buffer the kernel binds, in binding order.
    pub bindings: Vec<Binding>,
    pub groups: [u32; 3],
    /// Wait for every earlier dispatch's writes to be visible before this one starts. The
    /// caller decides, knowing which dispatch reads what another wrote; the recording adds
    /// none of its own between dispatches.
    pub barrier: bool,
}

/// The bytes `binding` covers in a buffer of `buffer_size` bytes, checked: inside the buffer, at
/// an offset the device can bind, no larger than one binding may be, and not empty.
pub fn binding_range(
    binding: Binding,
    buffer_size: u64,
    offset_alignment: u64,
    max_range: u32,
) -> Result<u64> {
    let Binding { offset, size, .. } = binding;
    let size = size.unwrap_or(buffer_size.saturating_sub(offset));
    let end = offset.checked_add(size);
    if size == 0 || end.is_none_or(|end| end > buffer_size) {
        return Err(Error::Unsupported(format!(
            "{size} bytes from {offset} of a buffer of {buffer_size}"
        )));
    }
    if !offset.is_multiple_of(offset_alignment.max(1)) {
        return Err(Error::Unsupported(format!(
            "a binding at offset {offset}, not a multiple of {offset_alignment}"
        )));
    }
    if size > u64::from(max_range) {
        return Err(Error::Unsupported(format!(
            "a binding of {size} bytes (the device binds at most {max_range})"
        )));
    }
    Ok(size)
}

/// A chain of dispatches recorded into one command buffer, and the buffers it binds.
pub struct Recording {
    ctx: Arc<Inner>,
    buffers: Vec<Buffer>,
    /// Kept alive while the commands name their pipelines.
    kernels: Vec<Arc<Kernel>>,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
    /// One timestamp before the first dispatch, and one after each when timed, or else one after
    /// the last.
    timestamps: vk::QueryPool,
    timed: bool,
    /// Submitted, and not yet seen finished: the GPU may be using every buffer.
    pending: bool,
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recording")
            .field("buffers", &self.buffers.len())
            .field("dispatches", &self.kernels.len())
            .field("timed", &self.timed)
            .finish_non_exhaustive()
    }
}

impl Recording {
    pub(crate) fn new(
        ctx: &Arc<Inner>,
        buffers: Vec<Buffer>,
        dispatches: &[Dispatch],
        timed: bool,
    ) -> Result<Self> {
        let report = &ctx.report;
        // Every binding is checked before anything reaches the driver.
        let mut ranges = Vec::with_capacity(dispatches.len());
        for (i, d) in dispatches.iter().enumerate() {
            if !Arc::ptr_eq(&d.kernel.ctx, ctx) {
                return Err(Error::Unsupported(format!(
                    "dispatch {i}: a kernel of another context"
                )));
            }
            if d.bindings.len() != d.kernel.storage_buffers as usize {
                return Err(Error::Unsupported(format!(
                    "dispatch {i}: {} bindings for a kernel that binds {}",
                    d.bindings.len(),
                    d.kernel.storage_buffers
                )));
            }
            check_groups(d.groups, report.max_workgroup_count)?;
            let sizes = d
                .bindings
                .iter()
                .map(|b| {
                    let buffer = buffers.get(b.buffer).ok_or_else(|| {
                        Error::Unsupported(format!(
                            "dispatch {i}: buffer {} of {}",
                            b.buffer,
                            buffers.len()
                        ))
                    })?;
                    if !Arc::ptr_eq(&buffer.ctx, ctx) {
                        return Err(Error::Unsupported(format!(
                            "dispatch {i}: a buffer of another context"
                        )));
                    }
                    binding_range(
                        *b,
                        buffer.size(),
                        report.min_storage_buffer_offset_alignment,
                        report.max_storage_buffer_range,
                    )
                    .map_err(|e| Error::Unsupported(format!("dispatch {i}: {e}")))
                })
                .collect::<Result<Vec<u64>>>()?;
            ranges.push(sizes);
        }

        let device = &ctx.device;
        // Owned from here on: a failure below releases what exists through `Drop`
        // (destroying a null handle is a no-op).
        let mut recording = Recording {
            ctx: Arc::clone(ctx),
            buffers,
            kernels: dispatches.iter().map(|d| Arc::clone(&d.kernel)).collect(),
            pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptors: vk::DescriptorPool::null(),
            timestamps: vk::QueryPool::null(),
            timed,
            pending: false,
        };
        let count = u32::try_from(dispatches.len())
            .map_err(|_| Error::Unsupported("too many dispatches".into()))?;
        let bindings: u32 = dispatches.iter().map(|d| d.kernel.storage_buffers).sum();
        // SAFETY: the device is live and each create-info outlives its call.
        unsafe {
            recording.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family),
                    None,
                )
                .call("vkCreateCommandPool")?;
            recording.commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(recording.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .call("vkAllocateCommandBuffers")?[0];
            recording.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .call("vkCreateFence")?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(bindings.max(1))];
            recording.descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(count.max(1))
                        .pool_sizes(&sizes),
                    None,
                )
                .call("vkCreateDescriptorPool")?;
            recording.timestamps = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(recording.queries()),
                    None,
                )
                .call("vkCreateQueryPool")?;
        }

        let sets = if dispatches.is_empty() {
            Vec::new()
        } else {
            let layouts: Vec<_> = dispatches.iter().map(|d| d.kernel.set_layout).collect();
            // SAFETY: the pool has room for one set per dispatch, of as many bindings.
            unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(recording.descriptors)
                        .set_layouts(&layouts),
                )
            }
            .call("vkAllocateDescriptorSets")?
        };
        let infos: Vec<Vec<[vk::DescriptorBufferInfo; 1]>> = dispatches
            .iter()
            .zip(&ranges)
            .map(|(d, sizes)| {
                d.bindings
                    .iter()
                    .zip(sizes)
                    .map(|(b, &size)| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(recording.buffers[b.buffer].raw)
                            .offset(b.offset)
                            .range(size)]
                    })
                    .collect()
            })
            .collect();
        let writes: Vec<_> = sets
            .iter()
            .zip(&infos)
            .flat_map(|(&set, infos)| {
                infos.iter().enumerate().map(move |(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
            })
            .collect();
        // SAFETY: every buffer is live and owned by the recording; no set is in use.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        recording.record(dispatches, &sets)?;
        Ok(recording)
    }

    /// The timestamps a submission writes.
    fn queries(&self) -> u32 {
        if self.timed {
            self.kernels.len() as u32 + 1
        } else {
            2
        }
    }

    fn record(&self, dispatches: &[Dispatch], sets: &[vk::DescriptorSet]) -> Result<()> {
        let device = &self.ctx.device;
        let cmd = self.commands;
        let between = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )];
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: the command buffer is fresh, and every object recorded is owned or kept
        // alive by the recording.
        unsafe {
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .call("vkBeginCommandBuffer")?;
            // The previous submission's writes and reads, earlier in submission order, are
            // ordered before this one's.
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&between),
            );
            let pool = self.timestamps;
            device.cmd_reset_query_pool(cmd, pool, 0, self.queries());
            device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
            let mut bound = vk::Pipeline::null();
            for (i, (d, &set)) in dispatches.iter().zip(sets).enumerate() {
                if d.barrier && i > 0 {
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&between),
                    );
                }
                if d.kernel.pipeline != bound {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        d.kernel.pipeline,
                    );
                    bound = d.kernel.pipeline;
                }
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    d.kernel.layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_dispatch(cmd, d.groups[0], d.groups[1], d.groups[2]);
                if self.timed {
                    device.cmd_write_timestamp2(
                        cmd,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        pool,
                        i as u32 + 1,
                    );
                }
            }
            if !self.timed {
                device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            device.end_command_buffer(cmd).call("vkEndCommandBuffer")?;
        }
        Ok(())
    }

    /// The number of dispatches recorded.
    pub fn dispatches(&self) -> usize {
        self.kernels.len()
    }

    /// Buffer `i`, read in place between submissions.
    pub fn buffer(&self, i: usize) -> &Buffer {
        &self.buffers[i]
    }

    /// Buffer `i`, written in place between submissions.
    pub fn buffer_mut(&mut self, i: usize) -> &mut Buffer {
        &mut self.buffers[i]
    }

    /// Runs the recorded dispatches once and waits for them: when this returns, everything they
    /// wrote is visible to the host.
    pub fn submit(&mut self) -> Result<()> {
        self.start()?.wait()
    }

    /// Submits the recorded dispatches and returns without waiting for them.
    pub fn start(&mut self) -> Result<Submission<'_>> {
        {
            let submit = self
                .ctx
                .submit
                .lock()
                .map_err(|_| Error::Unsupported("a poisoned queue lock".into()))?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(self.commands)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
            // SAFETY: the command buffer is executable and not pending (a submission holds the
            // recording until it has been waited for), and the queue is held for the call.
            unsafe {
                self.ctx
                    .device
                    .queue_submit2(submit.queue, &submits, self.fence)
            }
            .call("vkQueueSubmit2")?;
        }
        self.pending = true;
        Ok(Submission { recording: self })
    }

    /// Waits for the last submission unless it was seen finished. On a timeout the work is still
    /// pending, and `pending` stays set.
    fn finish(&mut self) -> Result<()> {
        if !self.pending {
            return Ok(());
        }
        self.ctx.wait_for(self.fence)?;
        // SAFETY: the fence belongs to this recording, and has signaled.
        unsafe { self.ctx.device.reset_fences(&[self.fence]) }.call("vkResetFences")?;
        self.pending = false;
        Ok(())
    }

    /// The GPU time of each dispatch in the last submission, from the timestamp after the one
    /// before it (or after the start) to the timestamp after it; `None` if the recording is
    /// not timed. A barrier before a dispatch counts as its time.
    pub fn timings(&self) -> Result<Option<Vec<Timing>>> {
        if !self.timed {
            return Ok(None);
        }
        let stamps = self.stamps(0, self.queries())?;
        Ok(Some(
            stamps.windows(2).map(|w| self.timing(w[0], w[1])).collect(),
        ))
    }

    /// The GPU time of the last submission, from the timestamp before its first dispatch to the
    /// one after its last.
    pub fn elapsed(&self) -> Result<Timing> {
        let last = self.queries() - 1;
        let start = self.stamps(0, 1)?[0];
        let end = self.stamps(last, 1)?[0];
        Ok(self.timing(start, end))
    }

    /// `count` timestamps of the last submission from query `first`.
    fn stamps(&self, first: u32, count: u32) -> Result<Vec<u64>> {
        let mut stamps = vec![0u64; count as usize];
        // SAFETY: the pool holds the queries asked for, all written by the last submission, which
        // has finished.
        unsafe {
            self.ctx
                .device
                .get_query_pool_results(
                    self.timestamps,
                    first,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .call("vkGetQueryPoolResults")?;
        }
        Ok(stamps)
    }

    fn timing(&self, start: u64, end: u64) -> Timing {
        let report = &self.ctx.report;
        let valid_bits = report.queue_families[self.ctx.queue_family as usize].timestamp_valid_bits;
        let ticks = elapsed_ticks(start, end, valid_bits);
        Timing {
            ticks,
            nanoseconds: ticks_to_ns(ticks, report.timestamp_period),
        }
    }
}

/// A recording's dispatches, submitted and perhaps still running. It holds the recording, so the
/// host cannot reach a buffer the GPU may be using until the submission is waited for; dropping it
/// waits too.
#[must_use = "dropping a submission waits for it"]
pub struct Submission<'a> {
    recording: &'a mut Recording,
}

impl std::fmt::Debug for Submission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission")
            .field("pending", &self.recording.pending)
            .finish_non_exhaustive()
    }
}

impl Submission<'_> {
    /// Waits for the dispatches: when this returns, everything they wrote is visible to the host.
    pub fn wait(self) -> Result<()> {
        // A failed wait is not repeated when the submission is dropped.
        ManuallyDrop::new(self).recording.finish()
    }
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        // A failure leaves the recording pending, which its own drop then accounts for.
        let _ = self.recording.finish();
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        // SAFETY: a fence of this device; waiting has no other precondition.
        let idle = !self.pending
            || unsafe { device.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS) }.is_ok();
        if !idle {
            // The GPU may still read and write them: leaking the buffers and the commands is
            // the only safe end, freeing them would hand it memory reused by something else.
            std::mem::forget(std::mem::take(&mut self.buffers));
            std::mem::forget(std::mem::take(&mut self.kernels));
            return;
        }
        // SAFETY: the commands have finished, so nothing recorded is in use; null handles are
        // no-ops. Buffers and kernels are dropped after this, as fields.
        unsafe {
            device.destroy_query_pool(self.timestamps, None);
            device.destroy_descriptor_pool(self.descriptors, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binding_must_lie_inside_its_buffer() {
        let max = u32::MAX;
        assert_eq!(
            binding_range(Binding::whole(0), 1024, 4, max).unwrap(),
            1024
        );
        assert_eq!(
            binding_range(Binding::range(0, 256, 768), 1024, 4, max).unwrap(),
            768
        );
        assert_eq!(
            binding_range(
                Binding {
                    buffer: 0,
                    offset: 1000,
                    size: None
                },
                1024,
                4,
                max
            )
            .unwrap(),
            24
        );
        for bad in [
            Binding::range(0, 256, 769),
            Binding::range(0, 1024, 4),
            Binding::range(0, 0, 0),
            Binding::range(0, u64::MAX - 1, 8),
            Binding {
                buffer: 0,
                offset: 2048,
                size: None,
            },
        ] {
            assert!(
                matches!(binding_range(bad, 1024, 4, max), Err(Error::Unsupported(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn a_binding_must_be_aligned_and_no_larger_than_the_device_binds() {
        assert!(matches!(
            binding_range(Binding::range(0, 8, 8), 64, 16, u32::MAX),
            Err(Error::Unsupported(why)) if why.contains("multiple of 16")
        ));
        assert!(binding_range(Binding::range(0, 16, 8), 64, 16, u32::MAX).is_ok());
        assert!(matches!(
            binding_range(Binding::whole(0), 64, 4, 32),
            Err(Error::Unsupported(why)) if why.contains("at most 32")
        ));
    }
}
