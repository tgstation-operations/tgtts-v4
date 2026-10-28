//! The GPU runtime on raw Vulkan: it loads SPIR-V compiled ahead of time, runs it at the subgroup
//! size the kernel was written for, and times it with GPU timestamps, with nothing between it and
//! the driver but `ash`'s bindings.
//!
//! In the order a caller meets it:
//!
//! - [`Context::new`] opens the Vulkan loader, chooses the device ([`select`]) and creates it with
//!   the features the kernels need: Vulkan 1.3, cooperative matrices, subgroup size control, f16
//!   arithmetic and storage, the Vulkan memory model, and a compute queue with timestamps; and
//!   `robustBufferAccess2` where the device has it.
//! - [`DeviceReport`] is what the device says about itself; the `properties` example prints it.
//! - [`Context::buffer_in`] makes a storage buffer where its [`Residence`] says: in memory for
//!   the GPU alone, which the host fills and reads through staging copies, or mapped for the host
//!   to write or read in place ([`memory`] says which memory and why); [`Context::buffer`] is
//!   the mapped kind the host writes.
//! - [`Context::kernel`] builds a compute pipeline from a SPIR-V blob with an explicit subgroup
//!   size and specialization constants ([`kernel`]), refusing one whose workgroup memory would
//!   pass the device's limit ([`spirv::workgroup_memory`]).
//! - [`Context::dispatch`] runs it, waits, and returns the GPU time from timestamps.
//! - [`Context::record`] records a chain of dispatches with the barriers between them into one
//!   command buffer, which [`Recording::submit`] runs as often as needed ([`recording`]), timing
//!   each submission, and each dispatch when asked; [`Recording::add_program`] records more such
//!   chains over the same buffers, the programs of one model, to run one at a time;
//!   [`Recording::copy_out`] and [`Recording::copy_in`] copy ranges of them to and from a buffer
//!   outside it on the GPU.
//! - [`Context::set_fence_spin`] has every wait for the GPU poll for a while before it blocks.
//! - [`Context::calibrate`] reads the GPU's timestamp counter beside the host's clock, which places
//!   a submission's ends ([`Recording::ends`]) on the host's timeline.
//! - [`Kernel::executables`] says what the driver compiled it into: the subgroup size it runs at,
//!   its registers and occupancy, and the machine code ([`executable`]).
//!
//! The decisions underneath, which device and queue, which memory type, how specialization
//! constants are packed and how timestamp ticks become nanoseconds, are plain functions tested
//! without a GPU. The tests that need one are built with the `device` feature:
//! `nix run .#device-tests` runs them under the validation layer.

mod context;
mod error;
pub mod executable;
mod instance;
pub mod kernel;
pub mod memory;
pub mod recording;
pub mod report;
pub mod select;
pub mod spirv;

pub use context::{
    Calibration, Context, Timing, check_groups, elapsed_ticks, groups_for, ticks_to_ns,
};
pub use error::{Error, Result};
pub use executable::Executable;
pub use instance::{Instance, VALIDATION_VAR};
pub use kernel::{Kernel, KernelDesc, SpecValue};
pub use memory::{Buffer, Residence};
pub use recording::{Binding, Dispatch, Recording, Region, Submission};
pub use report::DeviceReport;
