//! The GPU runtime on raw Vulkan: it loads SPIR-V compiled ahead of time, runs it at the subgroup
//! size the kernel was written for, and times it with GPU timestamps, with nothing between it and
//! the driver but `ash`'s bindings.
//!
//! In the order a caller meets it:
//!
//! - [`Context::new`] opens the Vulkan loader, chooses the device ([`select`]) and creates it with
//!   the features the kernels need: Vulkan 1.3, cooperative matrices, subgroup size control, f16
//!   arithmetic and storage, the Vulkan memory model, and a compute queue with timestamps.
//! - [`DeviceReport`] is what the device says about itself; the `properties` example prints it.
//! - [`Context::buffer`] makes a storage buffer the CPU reads and writes in place ([`memory`]
//!   says which memory and why).
//! - [`Context::kernel`] builds a compute pipeline from a SPIR-V blob with an explicit subgroup
//!   size and specialization constants ([`kernel`]).
//! - [`Context::dispatch`] runs it, waits, and returns the GPU time from timestamps.
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
pub mod report;
pub mod select;
pub mod spirv;

pub use context::{Context, Timing, check_groups, elapsed_ticks, groups_for, ticks_to_ns};
pub use error::{Error, Result};
pub use executable::Executable;
pub use instance::{Instance, VALIDATION_VAR};
pub use kernel::{Kernel, KernelDesc, SpecValue};
pub use memory::Buffer;
pub use report::DeviceReport;
