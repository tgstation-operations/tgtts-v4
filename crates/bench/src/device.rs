//! The device a run times, and what it was.

use std::path::Path;

use burn::tensor::{Device, FloatDType};
use serde::{Deserialize, Serialize};

/// The GPU a run timed, as it and its driver name themselves, and what autotune had chosen for
/// it before.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gpu {
    pub name: String,
    /// What cubecl compiles for on it.
    pub fingerprint: String,
    pub pci_bus_id: String,
    /// The Vulkan driver, `radv` for Mesa's on AMD.
    pub driver: String,
    /// Its version, `Mesa 26.1.8` for Mesa's.
    pub driver_info: String,
    pub autotune_cache: String,
    /// The size of autotune's database before the run: 0 where it chose every kernel anew, as
    /// the first repetitions of each shape then waited for it.
    pub autotune_cache_bytes_before: u64,
}

pub fn parse_dtype(name: &str) -> Result<FloatDType, String> {
    match name {
        "f32" => Ok(FloatDType::F32),
        "f16" => Ok(FloatDType::F16),
        "bf16" => Ok(FloatDType::BF16),
        other => Err(format!("unknown dtype {other}")),
    }
}

pub fn dtype_name(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F64 => "f64",
        FloatDType::F32 => "f32",
        FloatDType::Flex32 => "flex32",
        FloatDType::F16 => "f16",
        FloatDType::BF16 => "bf16",
    }
}

/// The device `name`, opened, and the GPU it is if it is one, with autotune keeping what it
/// chooses in `autotune_cache`.
pub fn open(name: &str, autotune_cache: &Path) -> Result<(Device, Option<Gpu>), String> {
    match name {
        "flex" => Ok((Device::flex(), None)),
        "vulkan" => vulkan(autotune_cache),
        other => Err(format!("no device {other}")),
    }
}

#[cfg(feature = "vulkan")]
fn vulkan(autotune_cache: &Path) -> Result<(Device, Option<Gpu>), String> {
    use burn::cubecl::{future::block_on, wgpu::wgpu};

    let cache_error = |e: std::io::Error| format!("{}: {e}", autotune_cache.display());
    std::fs::create_dir_all(autotune_cache).map_err(cache_error)?;
    let autotune_cache = &std::fs::canonicalize(autotune_cache).map_err(cache_error)?;
    let bytes_before = std::fs::metadata(autotune_cache.join("default.db")).map_or(0, |m| m.len());
    gpu::keep_autotune_in(autotune_cache).map_err(|e| e.to_string())?;
    let device = gpu::vulkan().map_err(|e| e.to_string())?;
    let identity = device.identity().expect("a Vulkan device says what it is");
    // wgpu reports the driver, which cubecl does not, from an instance of its own.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let info = adapters
        .iter()
        .map(wgpu::Adapter::get_info)
        .find(|info| info.name == identity.name);
    let info = info.ok_or_else(|| format!("wgpu lists no Vulkan adapter {}", identity.name))?;
    let gpu = Gpu {
        name: identity.name,
        fingerprint: identity.fingerprint,
        pci_bus_id: info.device_pci_bus_id,
        driver: info.driver,
        driver_info: info.driver_info,
        autotune_cache: autotune_cache.display().to_string(),
        autotune_cache_bytes_before: bytes_before,
    };
    Ok((device, Some(gpu)))
}

#[cfg(not(feature = "vulkan"))]
fn vulkan(_: &Path) -> Result<(Device, Option<Gpu>), String> {
    Err("this build has no device vulkan: build with --features vulkan".into())
}
