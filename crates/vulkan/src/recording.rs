//! Chains of dispatches recorded once, each into one command buffer, and submitted as often as
//! needed: a model's step, whose shapes do not change from one step to the next, costs the host
//! one submission and one wait, not a recording per kernel. A recording holds several such
//! programs over the buffers they share, one running at a time.
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
//! ([`Recording::elapsed`]), which costs two timestamps a submission; a program recorded timed
//! also times each dispatch ([`Recording::timings`]).

use std::{mem::ManuallyDrop, sync::Arc};

use ash::vk;
use tracing::error;

use crate::{
    context::{FENCE_TIMEOUT_NS, Inner, Timing, check_groups, elapsed_ticks, ticks_to_ns},
    error::{Error, Result, VkCall},
    kernel::Kernel,
    memory::{Buffer, after_earlier_work, before_later_work},
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

/// Bytes copied between one of a recording's buffers and a buffer outside it
/// ([`Recording::copy_out`], [`Recording::copy_in`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// The buffer's index among those the recording owns, and the offset in it.
    pub buffer: usize,
    pub offset: u64,
    /// The offset in the buffer outside.
    pub other: u64,
    pub bytes: u64,
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

/// Chains of dispatches over one set of buffers, each recorded into a command buffer of its own
/// and submitted as often as needed, one at a time: the programs of a model over the weights and
/// state they share, such as its decode step and its prefill, or the stages of a pipeline. The
/// first program is recorded with the recording; more programs, and the buffers they need, are
/// added as they come.
pub struct Recording {
    ctx: Arc<Inner>,
    buffers: Vec<Buffer>,
    pool: vk::CommandPool,
    fence: vk::Fence,
    programs: Vec<Program>,
    /// Submitted, and not yet seen finished: the GPU may be using every buffer.
    pending: bool,
    /// The program the last submission ran, whose timestamps it has written.
    last: Option<usize>,
}

/// One chain of dispatches: its commands, the descriptor sets they bind through, and its
/// timestamps.
struct Program {
    /// Kept alive while the commands name their pipelines.
    kernels: Vec<Arc<Kernel>>,
    commands: vk::CommandBuffer,
    descriptors: vk::DescriptorPool,
    /// One timestamp before the first dispatch, and one after each when timed, or else one after
    /// the last.
    timestamps: vk::QueryPool,
    timed: bool,
}

impl Program {
    /// The timestamps a submission writes.
    fn queries(&self) -> u32 {
        if self.timed {
            self.kernels.len() as u32 + 1
        } else {
            2
        }
    }
}

impl std::fmt::Debug for Recording {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dispatches: Vec<usize> = self.programs.iter().map(|p| p.kernels.len()).collect();
        f.debug_struct("Recording")
            .field("buffers", &self.buffers.len())
            .field("dispatches", &dispatches)
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
        // Owned from here on: a failure below releases what exists through `Drop`
        // (destroying a null handle is a no-op).
        let mut recording = Recording {
            ctx: Arc::clone(ctx),
            buffers,
            pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
            programs: Vec::new(),
            pending: false,
            last: None,
        };
        let device = &ctx.device;
        // SAFETY: the device is live and each create-info outlives its call.
        unsafe {
            recording.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family),
                    None,
                )
                .call("vkCreateCommandPool")?;
            recording.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .call("vkCreateFence")?;
        }
        recording.add_program(dispatches, timed)?;
        Ok(recording)
    }

    /// Takes `buffer` among those programs bind, and returns its index.
    pub fn add_buffer(&mut self, buffer: Buffer) -> Result<usize> {
        if !Arc::ptr_eq(&buffer.ctx, &self.ctx) {
            return Err(Error::Unsupported("a buffer of another context".into()));
        }
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    /// Records `dispatches` as another program over the buffers, timing each dispatch when
    /// `timed`, and returns its index.
    pub fn add_program(&mut self, dispatches: &[Dispatch], timed: bool) -> Result<usize> {
        let program = self.program(dispatches, timed)?;
        self.programs.push(program);
        Ok(self.programs.len() - 1)
    }

    /// Records `dispatches` in place of program `program`, timing each dispatch when `timed`;
    /// first waits for the last submission if waiting for it failed.
    pub fn replace_program(
        &mut self,
        program: usize,
        dispatches: &[Dispatch],
        timed: bool,
    ) -> Result<()> {
        if program >= self.programs.len() {
            return Err(Error::Unsupported(format!(
                "program {program} of {}",
                self.programs.len()
            )));
        }
        self.finish()?;
        let fresh = self.program(dispatches, timed)?;
        let old = std::mem::replace(&mut self.programs[program], fresh);
        self.release(&old);
        if self.last == Some(program) {
            self.last = None;
        }
        Ok(())
    }

    /// `dispatches` recorded as a program over the buffers, every binding checked before anything
    /// reaches the driver.
    fn program(&self, dispatches: &[Dispatch], timed: bool) -> Result<Program> {
        let ctx = &self.ctx;
        let report = &ctx.report;
        let buffers = &self.buffers;
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
        let mut program = Program {
            kernels: dispatches.iter().map(|d| Arc::clone(&d.kernel)).collect(),
            commands: vk::CommandBuffer::null(),
            descriptors: vk::DescriptorPool::null(),
            timestamps: vk::QueryPool::null(),
            timed,
        };
        if let Err(e) = self.build(&mut program, dispatches, &ranges) {
            self.release(&program);
            return Err(e);
        }
        Ok(program)
    }

    /// Creates `program`'s objects and records its commands, each dispatch bound at `ranges`.
    fn build(
        &self,
        program: &mut Program,
        dispatches: &[Dispatch],
        ranges: &[Vec<u64>],
    ) -> Result<()> {
        let device = &self.ctx.device;
        let count = u32::try_from(dispatches.len())
            .map_err(|_| Error::Unsupported("too many dispatches".into()))?;
        let bindings: u32 = dispatches.iter().map(|d| d.kernel.storage_buffers).sum();
        // SAFETY: the device is live and each create-info outlives its call.
        unsafe {
            program.commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .call("vkAllocateCommandBuffers")?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(bindings.max(1))];
            program.descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(count.max(1))
                        .pool_sizes(&sizes),
                    None,
                )
                .call("vkCreateDescriptorPool")?;
            program.timestamps = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(program.queries()),
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
                        .descriptor_pool(program.descriptors)
                        .set_layouts(&layouts),
                )
            }
            .call("vkAllocateDescriptorSets")?
        };
        let infos: Vec<Vec<[vk::DescriptorBufferInfo; 1]>> = dispatches
            .iter()
            .zip(ranges)
            .map(|(d, sizes)| {
                d.bindings
                    .iter()
                    .zip(sizes)
                    .map(|(b, &size)| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(self.buffers[b.buffer].raw)
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
        // SAFETY: every buffer is live and owned by the recording; the sets are fresh.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.record(program, dispatches, &sets)
    }

    /// Destroys `program`'s objects, which no pending submission may be using; null handles
    /// are no-ops.
    fn release(&self, program: &Program) {
        let device = &self.ctx.device;
        // SAFETY: as above; the command buffer was allocated from this recording's pool.
        unsafe {
            device.destroy_query_pool(program.timestamps, None);
            device.destroy_descriptor_pool(program.descriptors, None);
            if program.commands != vk::CommandBuffer::null() {
                device.free_command_buffers(self.pool, &[program.commands]);
            }
        }
    }

    fn record(
        &self,
        program: &Program,
        dispatches: &[Dispatch],
        sets: &[vk::DescriptorSet],
    ) -> Result<()> {
        let device = &self.ctx.device;
        let cmd = program.commands;
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
            // The previous submission's writes and reads, of whichever program, earlier in
            // submission order, are ordered before this one's.
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&between),
            );
            let pool = program.timestamps;
            device.cmd_reset_query_pool(cmd, pool, 0, program.queries());
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
                if program.timed {
                    device.cmd_write_timestamp2(
                        cmd,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        pool,
                        i as u32 + 1,
                    );
                }
            }
            if !program.timed {
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

    /// The number of programs recorded.
    pub fn programs(&self) -> usize {
        self.programs.len()
    }

    /// The number of dispatches program `program` records.
    pub fn dispatches(&self, program: usize) -> usize {
        self.programs[program].kernels.len()
    }

    /// Buffer `i`, read in place between submissions.
    pub fn buffer(&self, i: usize) -> &Buffer {
        &self.buffers[i]
    }

    /// Buffer `i`, written in place between submissions.
    pub fn buffer_mut(&mut self, i: usize) -> &mut Buffer {
        &mut self.buffers[i]
    }

    /// The buffers, to be recorded again; the commands are freed.
    pub fn into_buffers(mut self) -> Result<Vec<Buffer>> {
        self.finish()?;
        Ok(std::mem::take(&mut self.buffers))
    }

    /// Copies each of `regions` of the buffers into `to`, in a submission of its own after the
    /// last, and waits for it.
    pub fn copy_out(&mut self, regions: &[Region], to: &mut Buffer) -> Result<()> {
        self.copy(regions, to, true)
    }

    /// Copies each of `regions` of `from` into the buffers, as [`copy_out`](Self::copy_out)
    /// copies them out.
    pub fn copy_in(&mut self, from: &Buffer, regions: &[Region]) -> Result<()> {
        self.copy(regions, from, false)
    }

    fn copy(&mut self, regions: &[Region], other: &Buffer, out: bool) -> Result<()> {
        if !Arc::ptr_eq(&other.ctx, &self.ctx) {
            return Err(Error::Unsupported("a buffer of another context".into()));
        }
        for r in regions {
            let inside =
                |offset: u64, size: u64| offset.checked_add(r.bytes).is_some_and(|end| end <= size);
            let held = self.buffers.get(r.buffer).map(Buffer::size);
            if r.bytes == 0
                || !held.is_some_and(|size| inside(r.offset, size))
                || !inside(r.other, other.size())
            {
                return Err(Error::Unsupported(format!(
                    "a copy of {r:?} with a buffer of {} bytes",
                    other.size()
                )));
            }
        }
        self.finish()?;
        let copies: Vec<_> = regions
            .iter()
            .map(|r| {
                let held = self.buffers[r.buffer].raw;
                let (from, to, src, dst) = match out {
                    true => (held, other.raw, r.offset, r.other),
                    false => (other.raw, held, r.other, r.offset),
                };
                let region = vk::BufferCopy::default()
                    .src_offset(src)
                    .dst_offset(dst)
                    .size(r.bytes);
                (from, to, region)
            })
            .collect();
        self.ctx.run_once(|device, cmd| {
            let before = after_earlier_work(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
            let after = [
                before_later_work()[0],
                vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ),
            ];
            // SAFETY: the buffers are live and were created for transfers, each region lies
            // inside both (checked above), and the recording's last submission has finished.
            unsafe {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&before),
                );
                for (from, to, region) in &copies {
                    device.cmd_copy_buffer(cmd, *from, *to, std::slice::from_ref(region));
                }
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&after),
                );
            }
        })
    }

    /// Runs the first program once and waits for it: when this returns, everything it wrote is
    /// visible to the host.
    pub fn submit(&mut self) -> Result<()> {
        self.submit_program(0)
    }

    /// Submits the first program and returns without waiting for it.
    pub fn start(&mut self) -> Result<Submission<'_>> {
        self.start_program(0)
    }

    /// Runs program `program` once and waits for it, as [`submit`](Self::submit) runs the
    /// first.
    pub fn submit_program(&mut self, program: usize) -> Result<()> {
        self.start_program(program)?.wait()
    }

    /// Submits program `program` and returns without waiting for it; first waits again for the
    /// last submission if waiting for it failed.
    pub fn start_program(&mut self, program: usize) -> Result<Submission<'_>> {
        let commands = self
            .programs
            .get(program)
            .ok_or_else(|| {
                Error::Unsupported(format!("program {program} of {}", self.programs.len()))
            })?
            .commands;
        self.finish()?;
        {
            let submit = self
                .ctx
                .submit
                .lock()
                .map_err(|_| Error::Unsupported("a poisoned queue lock".into()))?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(commands)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
            // SAFETY: the command buffer is executable and not pending (the last submission, of
            // whichever program, was seen finished above), and the queue is held for the call.
            unsafe {
                self.ctx
                    .device
                    .queue_submit2(submit.queue, &submits, self.fence)
            }
            .call("vkQueueSubmit2")?;
        }
        self.pending = true;
        self.last = Some(program);
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
    /// before it (or after the start) to the timestamp after it; `None` if its program is not
    /// timed. A barrier before a dispatch counts as its time.
    pub fn timings(&self) -> Result<Option<Vec<Timing>>> {
        let program = self.ran()?;
        if !program.timed {
            return Ok(None);
        }
        let stamps = self.stamps(0, program.queries())?;
        Ok(Some(
            stamps.windows(2).map(|w| self.timing(w[0], w[1])).collect(),
        ))
    }

    /// The GPU time of the last submission, from the timestamp before its first dispatch to the
    /// one after its last.
    pub fn elapsed(&self) -> Result<Timing> {
        let [start, end] = self.ends()?;
        Ok(self.timing(start, end))
    }

    /// The last submission's timestamps before its first dispatch and after its last, in ticks,
    /// which a [`Calibration`](crate::Calibration) places on the host's clock.
    pub fn ends(&self) -> Result<[u64; 2]> {
        let last = self.ran()?.queries() - 1;
        Ok([self.stamps(0, 1)?[0], self.stamps(last, 1)?[0]])
    }

    /// The program the last submission ran, once it has run to the end.
    fn ran(&self) -> Result<&Program> {
        // Waiting for a timestamp no submission has written, or one still running, could block
        // for good.
        match self.last {
            Some(program) if !self.pending => Ok(&self.programs[program]),
            _ => Err(Error::Unsupported(
                "the timestamps of a recording not run to the end".into(),
            )),
        }
    }

    /// `count` timestamps of the last submission from query `first`.
    fn stamps(&self, first: u32, count: u32) -> Result<Vec<u64>> {
        let program = self.ran()?;
        let mut stamps = vec![0u64; count as usize];
        // SAFETY: the pool holds the queries asked for, all written by the last submission, which
        // has finished.
        unsafe {
            self.ctx
                .device
                .get_query_pool_results(
                    program.timestamps,
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
            error!("dropped a recording whose submission did not finish: leaking its buffers");
            // The GPU may still read and write them: leaking the buffers and the commands is
            // the only safe end, freeing them would hand it memory reused by something else.
            std::mem::forget(std::mem::take(&mut self.buffers));
            std::mem::forget(std::mem::take(&mut self.programs));
            return;
        }
        for program in &self.programs {
            self.release(program);
        }
        // SAFETY: the commands have finished, so nothing recorded is in use; null handles are
        // no-ops. Buffers and kernels are dropped after this, as fields.
        unsafe {
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
