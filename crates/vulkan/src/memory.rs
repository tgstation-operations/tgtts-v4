//! Storage buffers, and the memory type they live in.
//!
//! On an APU such as the Radeon 8060S the GPU has no memory of its own: every heap the driver
//! reports is the same system memory, and which type a buffer lives in decides how the CPU maps it
//! and how the GPU's caches treat it, not where its bytes are. So every buffer is host-visible and
//! stays mapped for its whole life, the CPU writing inputs and reading results in place with no
//! staging copy, and the type is chosen by [`choose_memory_type`], preferring one the driver also
//! marks device-local ([`STORAGE_PREFERENCE`]). A discrete GPU offers such a type only through a
//! resizable BAR, and would want its own memory and staging copies instead.

use std::{ptr::NonNull, sync::Arc};

use ash::vk;

use crate::{
    context::Inner,
    error::{Error, Result, VkCall},
    report::{MemoryHeap, MemoryType},
};

/// The memory properties a storage buffer's type must have, most preferred first.
///
/// Device-local and host-visible first: the type the driver means for memory the GPU works in and
/// the CPU reaches directly. It is not host-cached, so the CPU's reads of it may be slow, a cost
/// for reading results back rather than for kernels. Plain host-visible memory is the fallback for
/// a device without such a type.
pub const STORAGE_PREFERENCE: [vk::MemoryPropertyFlags; 2] = [
    vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
            | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    ),
    vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    ),
];

/// Properties a storage buffer's type must not have: AMD's device-coherent, uncached
/// memory bypasses the GPU's caches on every access, which is for debugging markers, not
/// for kernels.
pub const STORAGE_AVOID: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD.as_raw(),
);

/// The index of the memory type a buffer of `size` bytes is allocated from: the first
/// type, in the device's order, that `type_bits` (the buffer's `memoryTypeBits`) allows,
/// has every property of the most preferred entry of [`STORAGE_PREFERENCE`] any allowed
/// type matches, has none of [`STORAGE_AVOID`], and sits in a heap at least `size` large.
pub fn choose_memory_type(
    types: &[MemoryType],
    heaps: &[MemoryHeap],
    type_bits: u32,
    size: u64,
) -> Option<u32> {
    STORAGE_PREFERENCE.iter().find_map(|wanted| {
        types.iter().enumerate().find_map(|(i, t)| {
            let allowed = i < 32 && type_bits & (1 << i) != 0;
            let fits = heaps.get(t.heap as usize).is_some_and(|h| h.size >= size);
            (allowed && fits && t.flags.contains(*wanted) && !t.flags.intersects(STORAGE_AVOID))
                .then(|| u32::try_from(i).ok())
                .flatten()
        })
    })
}

/// Element types a buffer may be read and written as: plain values with no padding and
/// no invalid bit patterns, so any bytes are a valid value.
///
/// # Safety
///
/// Implement only for types for which every bit pattern of their size is a valid value
/// and which have no padding.
pub unsafe trait Plain: Copy + 'static {}

// SAFETY: primitive integers and floats have no padding and accept every bit pattern.
unsafe impl Plain for u8 {}
// SAFETY: as above.
unsafe impl Plain for i8 {}
// SAFETY: as above.
unsafe impl Plain for u16 {}
// SAFETY: as above.
unsafe impl Plain for i16 {}
// SAFETY: as above.
unsafe impl Plain for u32 {}
// SAFETY: as above.
unsafe impl Plain for i32 {}
// SAFETY: as above.
unsafe impl Plain for u64 {}
// SAFETY: as above.
unsafe impl Plain for i64 {}
// SAFETY: as above.
unsafe impl Plain for f32 {}
// SAFETY: as above.
unsafe impl Plain for f64 {}

/// A storage buffer in host-visible memory, mapped for as long as it lives. One buffer is
/// one allocation; suballocation waits until a model's hundreds of tensors make the
/// driver's allocation-count limit matter.
pub struct Buffer {
    pub(crate) ctx: Arc<Inner>,
    pub(crate) raw: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: NonNull<u8>,
    size: u64,
    memory_type: u32,
}

// SAFETY: the mapped pointer is owned by this buffer alone, and access through it follows
// Rust's borrow rules (`&self` reads, `&mut self` writes); the GPU writes only during a
// dispatch, which holds the buffer mutably borrowed until the GPU has finished.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.size)
            .field("memory_type", &self.memory_type)
            .finish_non_exhaustive()
    }
}

impl Buffer {
    /// A zero-filled storage buffer of `size` bytes (more than zero).
    pub(crate) fn new(ctx: &Arc<Inner>, size: u64) -> Result<Self> {
        if size == 0 {
            return Err(Error::Unsupported("a buffer of zero bytes".to_string()));
        }
        let device = &ctx.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live and `info` is complete.
        let raw = unsafe { device.create_buffer(&info, None) }.call("vkCreateBuffer")?;
        // Owned from here on, so a failure below releases what exists through `Drop`
        // (destroying or freeing a null handle is a no-op).
        let mut buffer = Buffer {
            ctx: Arc::clone(ctx),
            raw,
            memory: vk::DeviceMemory::null(),
            ptr: NonNull::dangling(),
            size,
            memory_type: u32::MAX,
        };
        // SAFETY: `raw` is a live buffer of this device.
        let needs = unsafe { device.get_buffer_memory_requirements(raw) };
        let report = &ctx.report;
        buffer.memory_type = choose_memory_type(
            &report.types,
            &report.heaps,
            needs.memory_type_bits,
            needs.size,
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
        // SAFETY: the memory is host-visible (every preferred type is) and not yet mapped.
        let mapped = unsafe {
            device.map_memory(
                buffer.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .call("vkMapMemory")?;
        buffer.ptr = NonNull::new(mapped.cast::<u8>()).ok_or(Error::Vk {
            call: "vkMapMemory",
            result: vk::Result::ERROR_MEMORY_MAP_FAILED,
        })?;
        // SAFETY: the mapping covers at least `size` bytes. Vulkan does not promise zeroed
        // memory, and this constructor's contract is zero-filled.
        unsafe { std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, size as usize) };
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

    /// The buffer's bytes, read in place.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is live for `self`'s lifetime and covers `size` bytes; the
        // GPU does not write while `&self` is held outside a dispatch.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size as usize) }
    }

    /// The buffer's bytes, written in place.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `bytes`, and `&mut self` makes this the only access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize) }
    }

    /// Copies `values` to the start of the buffer. Panics if they do not fit.
    pub fn write<T: Plain>(&mut self, values: &[T]) {
        let len = std::mem::size_of_val(values);
        assert!(
            len as u64 <= self.size,
            "{len} bytes do not fit in a buffer of {}",
            self.size
        );
        // SAFETY: `values` is `len` readable bytes of a `Plain` type, the destination
        // holds at least `len` bytes, and the two cannot overlap (one is mapped memory).
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), self.ptr.as_ptr(), len)
        };
    }

    /// The whole buffer as values of `T`, copied out; trailing bytes short of a whole `T`
    /// are left out.
    pub fn read<T: Plain>(&self) -> Vec<T> {
        let n = self.size as usize / std::mem::size_of::<T>();
        let mut out = Vec::<T>::with_capacity(n);
        // SAFETY: `n` whole values fit in the mapping, `out` has room for them, the bytes
        // are a valid `T` whatever they are (`Plain`), and the copy is bytewise, so the
        // mapping's alignment does not matter.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                out.as_mut_ptr().cast::<u8>(),
                n * size_of::<T>(),
            );
            out.set_len(n);
        }
        out
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: dispatches wait for the GPU before returning, so the buffer is idle;
        // freeing the memory unmaps it.
        unsafe {
            self.ctx.device.destroy_buffer(self.raw, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const HV: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
    const HC: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_COHERENT;
    const CACHED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_CACHED;
    const AMD: vk::MemoryPropertyFlags = STORAGE_AVOID;

    fn heap(size: u64) -> MemoryHeap {
        MemoryHeap {
            size,
            flags: vk::MemoryHeapFlags::empty(),
            budget: None,
            usage: None,
        }
    }

    fn ty(flags: vk::MemoryPropertyFlags, heap: u32) -> MemoryType {
        MemoryType { flags, heap }
    }

    /// The table RADV from Mesa 26.1.8 reports for the Radeon 8060S: heap 0 is a third of the
    /// GPU-addressable system memory, heap 1 the two thirds the driver marks device-local.
    fn radv_gfx1151() -> (Vec<MemoryType>, Vec<MemoryHeap>) {
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
        (types, heaps)
    }

    #[test]
    fn device_local_host_visible_memory_is_preferred() {
        let (types, heaps) = radv_gfx1151();
        assert_eq!(
            choose_memory_type(&types, &heaps, u32::MAX, 1 << 20),
            Some(3)
        );
        // The buffer's own allowed types narrow the choice to the second such type.
        assert_eq!(
            choose_memory_type(&types, &heaps, 1 << 4 | 1 << 2, 1 << 20),
            Some(4)
        );
    }

    #[test]
    fn plain_host_visible_memory_is_the_fallback_and_amd_uncached_memory_is_never_used() {
        let (types, heaps) = radv_gfx1151();
        let no_device_local_visible = !(1 << 3 | 1 << 4);
        assert_eq!(
            choose_memory_type(&types, &heaps, no_device_local_visible, 64),
            Some(2)
        );
        let only_amd = 1 << 7 | 1 << 8 | 1 << 9 | 1 << 10;
        assert_eq!(choose_memory_type(&types, &heaps, only_amd, 64), None);
        let only_device_local = 1 << 0 | 1 << 1;
        assert_eq!(
            choose_memory_type(&types, &heaps, only_device_local, 64),
            None
        );
    }

    #[test]
    fn a_heap_smaller_than_the_buffer_is_passed_over() {
        let (types, heaps) = radv_gfx1151();
        // Larger than heap 0, smaller than heap 1.
        assert_eq!(
            choose_memory_type(&types, &heaps, 1 << 2 | 1 << 3, 12 << 30),
            Some(3)
        );
        assert_eq!(choose_memory_type(&types, &heaps, 1 << 2, 12 << 30), None);
        assert_eq!(choose_memory_type(&types, &heaps, u32::MAX, 22 << 30), None);
    }
}
