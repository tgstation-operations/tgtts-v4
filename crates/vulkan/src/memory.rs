//! Storage buffers, and the memory type they live in.
//!
//! Where a buffer lives is its [`Residence`], and each residence has a list of memory properties
//! in order of preference ([`Residence::preference`]), so the choice follows what the device
//! reports rather than who made it ([`choose_memory_type`]).
//!
//! What the GPU reads every step, weights and a model's working memory, is
//! [`Residence::Device`]: on a discrete GPU its own memory, preferring a type the host cannot map.
//! On an integrated GPU it is system memory the host does not cache, where the device offers a
//! type of it that is not device-local: an APU's device-local memory is the carve-out the firmware
//! set aside from the same system memory, 512 MiB on the Radeon 8060S here, which RADV fills with
//! device-local buffers before it places any elsewhere, and in which the display has to pin its
//! framebuffers. The host reaches it only through a staging buffer and a copy on the queue, a
//! chunk at a time ([`STAGING_BYTES`]).
//! What the host writes or reads every step, small inputs and outputs, is mapped for the buffer's
//! whole life and read and written in place: [`Residence::Upload`] in device-local memory the
//! host can map where a discrete GPU has it (through a resizable BAR), in memory the host does
//! not cache otherwise, [`Residence::Readback`] in host-cached memory where the device has that.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

use ash::vk;

use crate::{
    Element,
    context::Inner,
    error::{Error, Result, VkCall},
    report::{MemoryHeap, MemoryType},
};

type Flags = vk::MemoryPropertyFlags;

const fn all(flags: &[Flags]) -> Flags {
    let mut raw = 0;
    let mut i = 0;
    while i < flags.len() {
        raw |= flags[i].as_raw();
        i += 1;
    }
    Flags::from_raw(raw)
}

const DEVICE_LOCAL: Flags = Flags::DEVICE_LOCAL;
const MAPPABLE: Flags = all(&[Flags::HOST_VISIBLE, Flags::HOST_COHERENT]);

/// Properties no buffer's type may have: AMD's device-coherent, uncached memory bypasses the
/// GPU's caches on every access, which is for debugging markers, not for kernels.
pub const AVOID: Flags = all(&[Flags::DEVICE_COHERENT_AMD, Flags::DEVICE_UNCACHED_AMD]);

/// The most a staging buffer holds: a larger transfer is copied a chunk at a time, so moving a
/// model's weights pins no more host memory than this.
pub const STAGING_BYTES: u64 = 64 << 20;

/// Where a buffer lives, and so how the host reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Residence {
    /// Memory for the GPU alone, its own on a discrete GPU, filled and read through staging
    /// copies.
    Device,
    /// Mapped memory the host writes and the GPU reads. It is not host-cached, so the host's
    /// reads of it may be slow.
    Upload,
    /// Mapped memory the GPU writes and the host reads; staging buffers live here too.
    Readback,
}

impl Residence {
    /// The (wanted, unwanted) memory properties of this residence's type on a device of
    /// `device_type`, most preferred first. Every entry also refuses [`AVOID`].
    pub fn preference(self, device_type: vk::PhysicalDeviceType) -> &'static [(Flags, Flags)] {
        const DEVICE: &[(Flags, Flags)] = &[
            (DEVICE_LOCAL, Flags::HOST_VISIBLE),
            (DEVICE_LOCAL, Flags::empty()),
        ];
        const UPLOAD: &[(Flags, Flags)] = &[
            (all(&[DEVICE_LOCAL, MAPPABLE]), Flags::empty()),
            (MAPPABLE, Flags::empty()),
        ];
        const READBACK: &[(Flags, Flags)] = &[
            (all(&[MAPPABLE, Flags::HOST_CACHED]), Flags::empty()),
            (MAPPABLE, Flags::empty()),
        ];
        // Out of the carve-out, and out of memory the GPU would have to snoop the host's caches
        // for.
        const INTEGRATED_DEVICE: &[(Flags, Flags)] = &[
            (Flags::empty(), all(&[DEVICE_LOCAL, Flags::HOST_CACHED])),
            (DEVICE_LOCAL, Flags::HOST_VISIBLE),
            (DEVICE_LOCAL, Flags::empty()),
        ];
        const INTEGRATED_UPLOAD: &[(Flags, Flags)] = &[
            (MAPPABLE, all(&[DEVICE_LOCAL, Flags::HOST_CACHED])),
            (all(&[DEVICE_LOCAL, MAPPABLE]), Flags::empty()),
            (MAPPABLE, Flags::empty()),
        ];
        let integrated = device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
        match self {
            Residence::Device if integrated => INTEGRATED_DEVICE,
            Residence::Device => DEVICE,
            Residence::Upload if integrated => INTEGRATED_UPLOAD,
            Residence::Upload => UPLOAD,
            Residence::Readback => READBACK,
        }
    }

    /// Whether buffers of this residence are mapped for the host.
    pub fn mapped(self) -> bool {
        self != Residence::Device
    }
}

/// The index of the memory type a buffer of `size` bytes in `residence` is allocated from on a
/// device of `device_type`: for the first entry of [`Residence::preference`] any type matches,
/// the first type, in the device's order, that `type_bits` (the buffer's `memoryTypeBits`)
/// allows, has the entry's wanted properties and none of its unwanted ones nor of [`AVOID`], and
/// sits in a heap at least `size` large.
pub fn choose_memory_type(
    types: &[MemoryType],
    heaps: &[MemoryHeap],
    device_type: vk::PhysicalDeviceType,
    type_bits: u32,
    size: u64,
    residence: Residence,
) -> Option<u32> {
    residence
        .preference(device_type)
        .iter()
        .find_map(|&(wanted, unwanted)| {
            types.iter().enumerate().find_map(|(i, t)| {
                let allowed = i < 32 && type_bits & (1 << i) != 0;
                let fits = heaps.get(t.heap as usize).is_some_and(|h| h.size >= size);
                (allowed
                    && fits
                    && t.flags.contains(wanted)
                    && !t.flags.intersects(unwanted | AVOID))
                .then(|| u32::try_from(i).ok())
                .flatten()
            })
        })
}

/// A storage buffer whose element type its type leaves out: what a recording holds beside buffers
/// of other element types, and what an arena of tensors of several is. The host reaches it as a
/// [`Buffer`] of what it holds. One is one allocation; suballocation waits until a model's
/// hundreds of tensors make the driver's allocation-count limit matter.
pub struct Memory {
    pub(crate) ctx: Arc<Inner>,
    pub(crate) raw: vk::Buffer,
    memory: vk::DeviceMemory,
    /// The mapping, for a residence the host maps.
    ptr: Option<NonNull<u8>>,
    size: u64,
    memory_type: u32,
    residence: Residence,
}

// SAFETY: the mapped pointer is owned by this memory alone, and access through it follows
// Rust's borrow rules (`&self` reads, `&mut self` writes); the GPU writes only during a
// dispatch, which holds the memory mutably borrowed until the GPU has finished.
unsafe impl Send for Memory {}
// SAFETY: as above.
unsafe impl Sync for Memory {}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("size", &self.size)
            .field("residence", &self.residence)
            .field("memory_type", &self.memory_type)
            .finish_non_exhaustive()
    }
}

/// A storage buffer of elements of `E`, in the memory its [`Residence`] chooses, mapped for as
/// long as it lives when that is host memory: written and read an element at a time, at offsets
/// and in counts of elements.
///
/// What it holds is in its type:
///
/// ```compile_fail,E0308
/// # fn write(ctx: &vulkan::Context) -> vulkan::Result<()> {
/// let mut halves = ctx.buffer::<vulkan::F16>(4)?;
/// halves.write(&[1.0f32, 2.0, 3.0, 4.0])?;
/// # Ok(())
/// # }
/// ```
#[repr(transparent)]
pub struct Buffer<E> {
    memory: Memory,
    element: PhantomData<E>,
}

impl<E> Deref for Buffer<E> {
    type Target = Memory;

    fn deref(&self) -> &Memory {
        &self.memory
    }
}

impl<E> DerefMut for Buffer<E> {
    fn deref_mut(&mut self) -> &mut Memory {
        &mut self.memory
    }
}

impl<E> From<Buffer<E>> for Memory {
    fn from(buffer: Buffer<E>) -> Memory {
        buffer.memory
    }
}

impl<E: Element> std::fmt::Debug for Buffer<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("element", &E::TYPE)
            .field("len", &self.len())
            .field("memory", &self.memory)
            .finish()
    }
}

/// Accesses to any buffer by earlier dispatches and copies, ordered before a copy's (`dst_*`).
pub(crate) fn after_earlier_work(
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> [vk::MemoryBarrier2<'static>; 1] {
    [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)]
}

/// A copy's or a fill's writes, made visible to every later dispatch and copy.
pub(crate) fn before_later_work() -> [vk::MemoryBarrier2<'static>; 1] {
    [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(
            vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::TRANSFER_READ
                | vk::AccessFlags2::TRANSFER_WRITE,
        )]
}

impl Memory {
    /// Zero-filled memory of `size` bytes (more than zero) in `residence`.
    pub(crate) fn new(ctx: &Arc<Inner>, size: u64, residence: Residence) -> Result<Self> {
        if size == 0 {
            return Err(Error::Unsupported("a buffer of zero bytes".to_string()));
        }
        let device = &ctx.device;
        // Whole words, so that a fill reaches the last byte.
        let info = vk::BufferCreateInfo::default()
            .size(size.next_multiple_of(4))
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live and `info` is complete.
        let raw = unsafe { device.create_buffer(&info, None) }.call("vkCreateBuffer")?;
        // Owned from here on, so a failure below releases what exists through `Drop`
        // (destroying or freeing a null handle is a no-op).
        let mut buffer = Memory {
            ctx: Arc::clone(ctx),
            raw,
            memory: vk::DeviceMemory::null(),
            ptr: None,
            size,
            memory_type: u32::MAX,
            residence,
        };
        // SAFETY: `raw` is a live buffer of this device.
        let needs = unsafe { device.get_buffer_memory_requirements(raw) };
        let report = &ctx.report;
        buffer.memory_type = choose_memory_type(
            &report.types,
            &report.heaps,
            report.device_type,
            needs.memory_type_bits,
            needs.size,
            residence,
        )
        .ok_or(Error::NoMemoryType {
            type_bits: needs.memory_type_bits,
            size: needs.size,
        })?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(buffer.memory_type);
        // SAFETY: the type index was chosen among the buffer's allowed types.
        buffer.memory = unsafe { device.allocate_memory(&alloc, None) }.call("vkAllocateMemory")?;
        // SAFETY: fresh memory of the required size and an allowed type, bound at offset 0.
        unsafe { device.bind_buffer_memory(raw, buffer.memory, 0) }.call("vkBindBufferMemory")?;
        if !residence.mapped() {
            // Vulkan does not promise zeroed memory, and this constructor's contract is
            // zero-filled.
            ctx.run_once(|device, cmd| {
                // SAFETY: the buffer is live and was created for transfers.
                unsafe {
                    device.cmd_fill_buffer(cmd, raw, 0, vk::WHOLE_SIZE, 0);
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&before_later_work()),
                    );
                }
            })?;
            return Ok(buffer);
        }
        // SAFETY: the memory is host-visible (every mapped residence's type is) and not yet
        // mapped.
        let mapped = unsafe {
            device.map_memory(
                buffer.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .call("vkMapMemory")?;
        let ptr = NonNull::new(mapped.cast::<u8>()).ok_or(Error::Vk {
            call: "vkMapMemory",
            result: vk::Result::ERROR_MEMORY_MAP_FAILED,
        })?;
        // SAFETY: the mapping covers at least `size` bytes; zero-filled as above.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, size as usize) };
        buffer.ptr = Some(ptr);
        Ok(buffer)
    }

    /// The buffer's length in bytes, as created (the allocation may be larger).
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The index of the memory type the buffer was allocated from.
    pub fn memory_type(&self) -> u32 {
        self.memory_type
    }

    pub fn residence(&self) -> Residence {
        self.residence
    }

    /// Copies `bytes` in from byte `offset`: in place when the memory is mapped, through a
    /// staging buffer when it is not.
    ///
    /// # Safety
    ///
    /// The bytes from `offset` lie inside the memory.
    unsafe fn write_bytes(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let len = bytes.len();
        if let Some(ptr) = self.ptr {
            // SAFETY: the destination lies inside the mapping (the caller's promise), and cannot
            // overlap `bytes`, which is not mapped memory of this buffer while `&mut self` is
            // held.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    ptr.as_ptr().add(offset as usize),
                    len,
                )
            };
            return Ok(());
        }
        if len == 0 {
            return Ok(());
        }
        let mut staging = Memory::new(
            &self.ctx,
            (len as u64).min(STAGING_BYTES),
            Residence::Readback,
        )?;
        let mut done = 0;
        for chunk in bytes.chunks(STAGING_BYTES as usize) {
            // SAFETY: the staging memory is at least as large as a chunk.
            unsafe { staging.write_bytes(0, chunk)? };
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset + done)
                .size(chunk.len() as u64)];
            let (from, to) = (staging.raw, self.raw);
            self.ctx.run_once(|device, cmd| {
                let before = after_earlier_work(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                );
                // SAFETY: both buffers are live and were created for transfers, and the
                // region lies inside each.
                unsafe {
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&before),
                    );
                    device.cmd_copy_buffer(cmd, from, to, &region);
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&before_later_work()),
                    );
                }
            })?;
            done += chunk.len() as u64;
        }
        Ok(())
    }

    /// Copies the `len` bytes from byte `offset` out to `dst`: in place when the memory is mapped,
    /// through a staging buffer when it is not.
    ///
    /// # Safety
    ///
    /// The bytes from `offset` lie inside the memory, and `dst` is valid for writes of `len`
    /// bytes.
    unsafe fn read_bytes(&self, offset: u64, dst: *mut u8, len: usize) -> Result<()> {
        if let Some(ptr) = self.ptr {
            // SAFETY: the source lies inside the mapping and `dst` has room for `len` bytes (the
            // caller's promises), and the copy is bytewise, so the mapping's alignment does not
            // matter; the GPU does not write while `&self` is held outside a dispatch.
            unsafe { std::ptr::copy_nonoverlapping(ptr.as_ptr().add(offset as usize), dst, len) };
        } else if len > 0 {
            let staging = Memory::new(
                &self.ctx,
                (len as u64).min(STAGING_BYTES),
                Residence::Readback,
            )?;
            let staged = staging.ptr.expect("a staging buffer is mapped");
            let mut done = 0;
            while done < len {
                let n = (len - done).min(STAGING_BYTES as usize);
                let region = [vk::BufferCopy::default()
                    .src_offset(offset + done as u64)
                    .dst_offset(0)
                    .size(n as u64)];
                let (from, to) = (self.raw, staging.raw);
                self.ctx.run_once(|device, cmd| {
                    let before = after_earlier_work(
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_READ,
                    );
                    let to_host = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                    // SAFETY: as for `write_bytes`.
                    unsafe {
                        device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default().memory_barriers(&before),
                        );
                        device.cmd_copy_buffer(cmd, from, to, &region);
                        device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default().memory_barriers(&to_host),
                        );
                    }
                })?;
                // SAFETY: the copy has finished and its writes are visible to the host; `dst`
                // has room for `len` bytes, of which `done` are written.
                unsafe { std::ptr::copy_nonoverlapping(staged.as_ptr(), dst.add(done), n) };
                done += n;
            }
        }
        Ok(())
    }
}

impl<E: Element> Buffer<E> {
    /// `memory` as elements of `E`, as its user laid it out.
    pub(crate) fn of(memory: Memory) -> Self {
        Buffer {
            memory,
            element: PhantomData,
        }
    }

    /// [`of`](Self::of), borrowed.
    pub(crate) fn of_ref(memory: &Memory) -> &Self {
        // SAFETY: a `Buffer` is a transparent `Memory`, and any bytes are elements of `E`.
        unsafe { &*(memory as *const Memory).cast::<Self>() }
    }

    /// [`of`](Self::of), borrowed mutably.
    pub(crate) fn of_mut(memory: &mut Memory) -> &mut Self {
        // SAFETY: as for `of_ref`.
        unsafe { &mut *(memory as *mut Memory).cast::<Self>() }
    }

    /// The elements it holds: all of its whole ones.
    pub fn len(&self) -> usize {
        (self.memory.size / E::TYPE.bytes()) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The byte at which `count` elements from `at` start. Panics if they do not fit.
    fn range(&self, at: usize, count: usize) -> u64 {
        assert!(
            at.checked_add(count).is_some_and(|end| end <= self.len()),
            "{count} elements from {at} do not fit in a buffer of {}",
            self.len()
        );
        at as u64 * E::TYPE.bytes()
    }

    /// Copies `values` to the start of the buffer. Panics if they do not fit.
    pub fn write(&mut self, values: &[E]) -> Result<()> {
        self.write_at(0, values)
    }

    /// Copies `values` into the buffer from element `at`: in place when the buffer is mapped,
    /// through a staging buffer when it is not. Panics if they do not fit.
    pub fn write_at(&mut self, at: usize, values: &[E]) -> Result<()> {
        let offset = self.range(at, values.len());
        // SAFETY: `values` is readable elements, which have no padding (`Element`), so every byte
        // is initialized.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        };
        // SAFETY: the bytes lie inside (checked above).
        unsafe { self.memory.write_bytes(offset, bytes) }
    }

    /// The whole buffer, copied out.
    pub fn read(&self) -> Result<Vec<E>> {
        self.read_at(0, self.len())
    }

    /// `count` elements from element `at`, copied out: in place when the buffer is mapped,
    /// through a staging buffer when it is not. Panics if they do not fit.
    pub fn read_at(&self, at: usize, count: usize) -> Result<Vec<E>> {
        let offset = self.range(at, count);
        let mut out = Vec::<E>::with_capacity(count);
        let len = count * size_of::<E>();
        // SAFETY: the bytes lie inside (checked above), and `out` has room for them.
        unsafe {
            self.memory
                .read_bytes(offset, out.as_mut_ptr().cast::<u8>(), len)?
        };
        // SAFETY: `count` whole elements were written, and any bytes are elements (`Element`).
        unsafe { out.set_len(count) };
        Ok(out)
    }
}
impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: dispatches and copies wait for the GPU before returning, so the buffer is
        // idle; freeing the memory unmaps it.
        unsafe {
            self.ctx.device.destroy_buffer(self.raw, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DL: Flags = Flags::DEVICE_LOCAL;
    const HV: Flags = Flags::HOST_VISIBLE;
    const HC: Flags = Flags::HOST_COHERENT;
    const CACHED: Flags = Flags::HOST_CACHED;
    const AMD: Flags = AVOID;

    fn heap(size: u64) -> MemoryHeap {
        MemoryHeap {
            size,
            flags: vk::MemoryHeapFlags::empty(),
            budget: None,
            usage: None,
        }
    }

    fn ty(flags: Flags, heap: u32) -> MemoryType {
        MemoryType { flags, heap }
    }

    type Table = (Vec<MemoryType>, Vec<MemoryHeap>, vk::PhysicalDeviceType);

    /// The table RADV from Mesa 26.1.8 reports for the Radeon 8060S: heap 0 is a third of the
    /// GPU-addressable system memory, heap 1 the two thirds the driver marks device-local. Every
    /// device-local type allocates from the 512 MiB carve-out first, every other type from system
    /// memory.
    fn radv_gfx1151() -> Table {
        let heaps = vec![heap(11_289_620_480), heap(22_579_245_056)];
        let types = vec![
            ty(DL, 1),
            ty(DL, 1),
            ty(HV | HC, 0),
            ty(DL | HV | HC, 1),
            ty(DL | HV | HC, 1),
            ty(HV | HC | CACHED, 0),
            ty(HV | HC | CACHED, 0),
            ty(DL | AMD, 1),
            ty(HV | HC | AMD, 0),
            ty(DL | HV | HC | AMD, 1),
            ty(HV | HC | CACHED | AMD, 0),
        ];
        (types, heaps, vk::PhysicalDeviceType::INTEGRATED_GPU)
    }

    /// A discrete GPU of 16 GiB laid out as NVIDIA's driver lays one out: heap 0 its own memory,
    /// heap 1 system memory, heap 2 the window the host maps into the GPU's memory, `bar` bytes:
    /// 256 MiB without a resizable BAR, all of it with one.
    fn discrete(bar: u64) -> Table {
        let heaps = vec![heap(16 << 30), heap(32 << 30), heap(bar)];
        let types = vec![
            ty(Flags::empty(), 1),
            ty(DL, 0),
            ty(DL, 0),
            ty(HV | HC, 1),
            ty(HV | HC | CACHED, 1),
            ty(DL | HV | HC, 2),
        ];
        (types, heaps, vk::PhysicalDeviceType::DISCRETE_GPU)
    }

    fn choose(
        (types, heaps, device_type): &Table,
        type_bits: u32,
        size: u64,
        residence: Residence,
    ) -> Option<u32> {
        choose_memory_type(types, heaps, *device_type, type_bits, size, residence)
    }

    #[test]
    fn each_residence_takes_its_preferred_type_on_an_apu() {
        let radv = radv_gfx1151();
        // Write-combined system memory, out of the carve-out, for what the host does not read.
        assert_eq!(choose(&radv, u32::MAX, 1 << 20, Residence::Device), Some(2));
        assert_eq!(choose(&radv, u32::MAX, 1 << 20, Residence::Upload), Some(2));
        assert_eq!(
            choose(&radv, u32::MAX, 1 << 20, Residence::Readback),
            Some(5)
        );
        // Where the buffer's own allowed types leave no system memory, the carve-out's, narrowed
        // to the second such type.
        assert_eq!(
            choose(&radv, 1 << 4 | 1 << 1, 1 << 20, Residence::Upload),
            Some(4)
        );
    }

    #[test]
    fn an_apu_takes_the_carve_out_rather_than_memory_the_gpu_snoops() {
        let radv = radv_gfx1151();
        assert_eq!(
            choose(&radv, 1 << 5 | 1 << 1, 64, Residence::Device),
            Some(1)
        );
        assert_eq!(
            choose(&radv, 1 << 5 | 1 << 3, 64, Residence::Upload),
            Some(3)
        );
    }

    #[test]
    fn each_residence_takes_its_preferred_type_on_a_discrete_gpu() {
        for bar in [256 << 20, 16 << 30] {
            let gpu = discrete(bar);
            assert_eq!(choose(&gpu, u32::MAX, 1 << 30, Residence::Device), Some(1));
            assert_eq!(choose(&gpu, u32::MAX, 4096, Residence::Upload), Some(5));
            assert_eq!(choose(&gpu, u32::MAX, 4096, Residence::Readback), Some(4));
        }
        // Past the window without a resizable BAR, mapped memory is the host's.
        let small_bar = discrete(256 << 20);
        assert_eq!(
            choose(&small_bar, u32::MAX, 512 << 20, Residence::Upload),
            Some(3)
        );
        assert_eq!(
            choose(&discrete(16 << 30), u32::MAX, 512 << 20, Residence::Upload),
            Some(5)
        );
    }

    #[test]
    fn the_fallbacks_are_taken_in_order_and_amd_uncached_memory_is_never_used() {
        let radv = radv_gfx1151();
        let no_device_local_visible = !(1 << 3 | 1 << 4);
        assert_eq!(
            choose(&radv, no_device_local_visible, 64, Residence::Upload),
            Some(2)
        );
        let visible_device_local = 1 << 3 | 1 << 4;
        assert_eq!(
            choose(&radv, visible_device_local, 64, Residence::Device),
            Some(3)
        );
        let uncached = 1 << 2 | 1 << 3;
        assert_eq!(choose(&radv, uncached, 64, Residence::Readback), Some(2));
        let only_amd = 1 << 7 | 1 << 8 | 1 << 9 | 1 << 10;
        for residence in [Residence::Device, Residence::Upload, Residence::Readback] {
            assert_eq!(choose(&radv, only_amd, 64, residence), None);
        }
        let only_device_local = 1 << 0 | 1 << 1;
        assert_eq!(
            choose(&radv, only_device_local, 64, Residence::Upload),
            None
        );
        assert_eq!(choose(&radv, 1 << 5, 64, Residence::Device), None);
    }

    #[test]
    fn a_heap_smaller_than_the_buffer_is_passed_over() {
        let radv = radv_gfx1151();
        // Larger than heap 0, smaller than heap 1.
        assert_eq!(
            choose(&radv, 1 << 2 | 1 << 3, 12 << 30, Residence::Upload),
            Some(3)
        );
        assert_eq!(choose(&radv, 1 << 2, 12 << 30, Residence::Upload), None);
        assert_eq!(
            choose(&radv, u32::MAX, 12 << 30, Residence::Device),
            Some(0)
        );
        assert_eq!(choose(&radv, u32::MAX, 22 << 30, Residence::Device), None);
    }
}
