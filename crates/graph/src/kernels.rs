//! The kernels a graph dispatches: what each binds and how, and their pipelines, built from the
//! SPIR-V that `nix build .#glsl-kernels` compiles from `kernels/glsl`, or `nix build
//! .#gpu-kernels` from the MLIR of `kernels/gpu` with a manifest beside each, once per choice of
//! specialization constants.

use std::{
    collections::HashMap,
    ffi::CString,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::Deserialize;
use vulkan::{Context, Element, ElementType, F16, Kernel, KernelDesc};

use crate::{
    Error,
    graph::{Op, Pipeline},
};

/// Names the directory of compiled kernels; `nix run .#device-tests` sets it.
pub const KERNELS_VAR: &str = "GLSL_KERNELS";

/// Names the directory of the kernels compiled from MLIR; `nix run .#device-tests` sets it.
pub const MLIR_KERNELS_VAR: &str = "VULKAN_KERNELS";

/// How a kernel uses one of its bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// One of a kernel's bindings: how the kernel uses it, and the elements the data bound there are,
/// whatever the kernel loads them as, such as an f16 tensor eight halves to a uvec4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub access: Access,
    pub element: ElementType,
}

/// A binding of elements of `E` the kernel reads.
pub const fn read<E: Element>() -> Binding {
    Binding {
        access: Access::Read,
        element: E::TYPE,
    }
}

/// A binding of elements of `E` the kernel writes.
pub const fn write<E: Element>() -> Binding {
    Binding {
        access: Access::Write,
        element: E::TYPE,
    }
}

/// A kernel of `kernels/glsl`: its file's stem, and its bindings, in order.
#[derive(Debug)]
pub struct KernelInfo {
    pub name: &'static str,
    pub bindings: &'static [Binding],
}

/// What a kernel compiled from MLIR says of itself in the manifest beside its SPIR-V, as
/// `kernels/gpu/build.py` writes it. Such a kernel indexes its buffers by the shapes it was
/// compiled at and never checks a bound, so every dispatch of it must match its grid and bind
/// at least its bytes ([`Kernels::check`]).
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub entry_point: String,
    pub dispatch: [u32; 3],
    pub required_subgroup_size: Option<u32>,
    pub bindings: Vec<ManifestBinding>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestBinding {
    pub name: String,
    pub access: String,
    pub element: String,
    pub bytes: u64,
}

/// Why `manifest` does not describe `kernel`'s bindings: their access or their elements differ.
fn unlike(kernel: &KernelInfo, manifest: &Manifest) -> Option<String> {
    let access = |a: Access| match a {
        Access::Read => "read",
        Access::Write => "write",
    };
    let declared: Vec<(&str, &str)> = kernel
        .bindings
        .iter()
        .map(|b| (access(b.access), b.element.name()))
        .collect();
    let found: Vec<(&str, &str)> = manifest
        .bindings
        .iter()
        .map(|b| (b.access.as_str(), b.element.as_str()))
        .collect();
    (declared != found).then(|| format!("bindings {found:?}, declared {declared:?}"))
}

/// `rmsnorm.comp`: x, gain -> out.
pub static RMSNORM: KernelInfo = KernelInfo {
    name: "rmsnorm",
    bindings: &[read::<F16>(), read::<F16>(), write::<F16>()],
};

/// `gemv_qkv_rope.comp`: x, gain, W, cos, sin, position -> q, k cache, v cache.
pub static GEMV_QKV_ROPE: KernelInfo = KernelInfo {
    name: "gemv_qkv_rope",
    bindings: &[
        read::<F16>(),
        read::<F16>(),
        read::<F16>(),
        read::<f32>(),
        read::<f32>(),
        read::<u32>(),
        write::<F16>(),
        write::<F16>(),
        write::<F16>(),
    ],
};

/// `attention_split.comp`: q, k cache, v cache, position -> partial outputs, maxima and sums.
pub static ATTENTION_SPLIT: KernelInfo = KernelInfo {
    name: "attention_split",
    bindings: &[
        read::<F16>(),
        read::<F16>(),
        read::<F16>(),
        read::<u32>(),
        write::<f32>(),
        write::<f32>(),
    ],
};

/// `attention_combine.comp`: position, partial outputs, maxima and sums -> out.
pub static ATTENTION_COMBINE: KernelInfo = KernelInfo {
    name: "attention_combine",
    bindings: &[read::<u32>(), read::<f32>(), read::<f32>(), write::<F16>()],
};

/// `gemv_residual.comp`: x, W, residual -> out.
pub static GEMV_RESIDUAL: KernelInfo = KernelInfo {
    name: "gemv_residual",
    bindings: &[read::<F16>(), read::<F16>(), read::<F16>(), write::<F16>()],
};

/// `gemv_gate_up.comp`: x, gain, W -> out.
pub static GEMV_GATE_UP: KernelInfo = KernelInfo {
    name: "gemv_gate_up",
    bindings: &[read::<F16>(), read::<F16>(), read::<F16>(), write::<F16>()],
};

/// `stream_read.comp`: data -> one word per subgroup.
pub static STREAM_READ: KernelInfo = KernelInfo {
    name: "stream_read",
    bindings: &[read::<u32>(), write::<u32>()],
};

type Key = (&'static str, Vec<(u32, u32)>, Option<u32>);

/// The pipelines built so far, each once.
pub struct Kernels {
    ctx: Context,
    dir: PathBuf,
    mlir: Option<PathBuf>,
    built: HashMap<Key, Arc<Kernel>>,
    manifests: HashMap<&'static str, Option<Manifest>>,
}

impl Kernels {
    /// Pipelines from the SPIR-V in `dir`.
    pub fn new(ctx: &Context, dir: impl AsRef<Path>) -> Self {
        Kernels {
            ctx: ctx.clone(),
            dir: dir.as_ref().to_owned(),
            mlir: None,
            built: HashMap::new(),
            manifests: HashMap::new(),
        }
    }

    /// Pipelines also from the kernels compiled from MLIR in `dir`, those the first directory
    /// does not have.
    pub fn with_mlir(mut self, dir: impl AsRef<Path>) -> Self {
        self.mlir = Some(dir.as_ref().to_owned());
        self
    }

    /// Pipelines from the directory [`KERNELS_VAR`] names, and from the one
    /// [`MLIR_KERNELS_VAR`] names if it is set.
    pub fn from_env(ctx: &Context) -> Result<Self, Error> {
        let dir = std::env::var_os(KERNELS_VAR).ok_or_else(|| {
            Error::Kernels(format!(
                "{KERNELS_VAR} names no directory; `nix build .#glsl-kernels` builds one"
            ))
        })?;
        let kernels = Self::new(ctx, dir);
        Ok(match std::env::var_os(MLIR_KERNELS_VAR) {
            Some(mlir) => kernels.with_mlir(mlir),
            None => kernels,
        })
    }

    /// The manifest of `kernel` if it is one compiled from MLIR: none if the first directory
    /// has its SPIR-V.
    pub fn manifest(&mut self, kernel: &'static KernelInfo) -> Result<Option<&Manifest>, Error> {
        if !self.manifests.contains_key(kernel.name) {
            let glsl = self.dir.join(format!("{}.spv", kernel.name));
            let manifest = match &self.mlir {
                Some(dir) if !glsl.exists() => {
                    let path = dir.join(format!("{}.json", kernel.name));
                    let bad = |why: String| Error::Kernels(format!("{}: {why}", path.display()));
                    let json = std::fs::read(&path).map_err(|e| bad(e.to_string()))?;
                    let manifest: Manifest =
                        serde_json::from_slice(&json).map_err(|e| bad(e.to_string()))?;
                    if let Some(why) = unlike(kernel, &manifest) {
                        return Err(bad(why));
                    }
                    Some(manifest)
                }
                _ => None,
            };
            self.manifests.insert(kernel.name, manifest);
        }
        Ok(self.manifests[kernel.name].as_ref())
    }

    /// Checks `op` against its kernel's manifest, if it has one: its grid, and that each tensor
    /// it binds, of `bytes`, holds at least what the kernel reaches; and that it runs at the
    /// subgroup size the kernel was written for.
    pub fn check(&mut self, op: &Op, bytes: impl Fn(usize) -> u64) -> Result<(), Error> {
        let Some(manifest) = self.manifest(op.pipeline.kernel)? else {
            return Ok(());
        };
        let bad = |why: String| Err(Error::Shape(format!("{}: {why}", op.label)));
        if op.groups != manifest.dispatch {
            return bad(format!(
                "a grid of {:?}, compiled for {:?}",
                op.groups, manifest.dispatch
            ));
        }
        if op.pipeline.subgroup_size != manifest.required_subgroup_size {
            return bad(format!(
                "subgroups of {:?}, compiled for {:?}",
                op.pipeline.subgroup_size, manifest.required_subgroup_size
            ));
        }
        for (t, binding) in op.bindings.iter().zip(&manifest.bindings) {
            if bytes(t.0) < binding.bytes {
                return bad(format!(
                    "{} bytes bound as {}, which reaches {}",
                    bytes(t.0),
                    binding.name,
                    binding.bytes
                ));
            }
        }
        Ok(())
    }

    /// The pipeline `pipeline` describes, built on first use. Every kernel requires full
    /// subgroups when it requires a size.
    pub fn get(&mut self, pipeline: &Pipeline) -> Result<Arc<Kernel>, Error> {
        let key = pipeline.key();
        if let Some(kernel) = self.built.get(&key) {
            return Ok(Arc::clone(kernel));
        }
        let name = pipeline.kernel.name;
        let (path, entry) = match self.manifest(pipeline.kernel)? {
            Some(manifest) => {
                let entry = CString::new(manifest.entry_point.as_str())
                    .map_err(|e| Error::Kernels(format!("{name}: {e}")))?;
                let dir = self.mlir.as_ref().expect("a manifest is read from it");
                (dir.join(format!("{name}.spv")), entry)
            }
            None => (self.dir.join(format!("{name}.spv")), c"main".to_owned()),
        };
        let spirv =
            std::fs::read(&path).map_err(|e| Error::Kernels(format!("{}: {e}", path.display())))?;
        let desc = KernelDesc {
            entry: &entry,
            storage_buffers: pipeline.kernel.bindings.len() as u32,
            subgroup_size: pipeline.subgroup_size,
            full_subgroups: pipeline.subgroup_size.is_some(),
            constants: &pipeline.constants,
        };
        let kernel = Arc::new(self.ctx.kernel(&spirv, &desc)?);
        self.built.insert(key, Arc::clone(&kernel));
        Ok(kernel)
    }

    /// The number of distinct pipelines built.
    pub fn len(&self) -> usize {
        self.built.len()
    }

    pub fn is_empty(&self) -> bool {
        self.built.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(bindings: &[(&str, &str)]) -> Manifest {
        let bindings = bindings.iter().map(|&(access, element)| ManifestBinding {
            name: "t".into(),
            access: access.into(),
            element: element.into(),
            bytes: 64,
        });
        Manifest {
            entry_point: "main".into(),
            dispatch: [1, 1, 1],
            required_subgroup_size: None,
            bindings: bindings.collect(),
        }
    }

    #[test]
    fn a_manifest_of_other_bindings_than_the_kernel_declares_is_refused() {
        let gemm = crate::prefill::gemm(128, 2048, 2048, 64).unwrap();
        let built = [("read", "f16"), ("read", "f16"), ("write", "f32")];
        assert_eq!(unlike(gemm, &manifest(&built)), None);
        for other in [
            [("read", "f16"), ("read", "f16"), ("write", "f16")],
            [("read", "f16"), ("write", "f16"), ("write", "f32")],
        ] {
            assert!(unlike(gemm, &manifest(&other)).is_some(), "{other:?}");
        }
        assert!(unlike(gemm, &manifest(&built[..2])).is_some());
    }
}
