//! The kernels a graph dispatches: what each binds and how, and their pipelines, built from the
//! SPIR-V that `nix build .#glsl-kernels` compiles from `kernels/glsl`, once per choice of
//! specialization constants.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use vulkan::{Context, Kernel, KernelDesc};

use crate::{Error, graph::Pipeline};

/// Names the directory of compiled kernels; `nix run .#device-tests` sets it.
pub const KERNELS_VAR: &str = "GLSL_KERNELS";

/// How a kernel uses one of its bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// A kernel of `kernels/glsl`: its file's stem, and how it uses each binding, in order.
#[derive(Debug)]
pub struct KernelInfo {
    pub name: &'static str,
    pub access: &'static [Access],
}

use Access::{Read, Write};

/// `rmsnorm.comp`: x, gain -> out.
pub static RMSNORM: KernelInfo = KernelInfo {
    name: "rmsnorm",
    access: &[Read, Read, Write],
};

/// `gemv_qkv_rope.comp`: x, gain, W, cos, sin, position -> q, k cache, v cache.
pub static GEMV_QKV_ROPE: KernelInfo = KernelInfo {
    name: "gemv_qkv_rope",
    access: &[Read, Read, Read, Read, Read, Read, Write, Write, Write],
};

/// `attention_split.comp`: q, k cache, v cache, position -> partial outputs, maxima and sums.
pub static ATTENTION_SPLIT: KernelInfo = KernelInfo {
    name: "attention_split",
    access: &[Read, Read, Read, Read, Write, Write],
};

/// `attention_combine.comp`: position, partial outputs, maxima and sums -> out.
pub static ATTENTION_COMBINE: KernelInfo = KernelInfo {
    name: "attention_combine",
    access: &[Read, Read, Read, Write],
};

/// `gemv_residual.comp`: x, W, residual -> out.
pub static GEMV_RESIDUAL: KernelInfo = KernelInfo {
    name: "gemv_residual",
    access: &[Read, Read, Read, Write],
};

/// `gemv_gate_up.comp`: x, gain, W -> out.
pub static GEMV_GATE_UP: KernelInfo = KernelInfo {
    name: "gemv_gate_up",
    access: &[Read, Read, Read, Write],
};

/// `stream_read.comp`: data -> one word per subgroup.
pub static STREAM_READ: KernelInfo = KernelInfo {
    name: "stream_read",
    access: &[Read, Write],
};

type Key = (&'static str, Vec<(u32, u32)>, Option<u32>);

/// The pipelines built so far, each once.
pub struct Kernels {
    ctx: Context,
    dir: PathBuf,
    built: HashMap<Key, Arc<Kernel>>,
}

impl Kernels {
    /// Pipelines from the SPIR-V in `dir`.
    pub fn new(ctx: &Context, dir: impl AsRef<Path>) -> Self {
        Kernels {
            ctx: ctx.clone(),
            dir: dir.as_ref().to_owned(),
            built: HashMap::new(),
        }
    }

    /// Pipelines from the directory [`KERNELS_VAR`] names.
    pub fn from_env(ctx: &Context) -> Result<Self, Error> {
        let dir = std::env::var_os(KERNELS_VAR).ok_or_else(|| {
            Error::Kernels(format!(
                "{KERNELS_VAR} names no directory; `nix build .#glsl-kernels` builds one"
            ))
        })?;
        Ok(Self::new(ctx, dir))
    }

    /// The pipeline `pipeline` describes, built on first use. Every kernel requires full
    /// subgroups when it requires a size.
    pub fn get(&mut self, pipeline: &Pipeline) -> Result<Arc<Kernel>, Error> {
        let key = pipeline.key();
        if let Some(kernel) = self.built.get(&key) {
            return Ok(Arc::clone(kernel));
        }
        let path = self.dir.join(format!("{}.spv", pipeline.kernel.name));
        let spirv =
            std::fs::read(&path).map_err(|e| Error::Kernels(format!("{}: {e}", path.display())))?;
        let desc = KernelDesc {
            storage_buffers: pipeline.kernel.access.len() as u32,
            subgroup_size: pipeline.subgroup_size,
            full_subgroups: pipeline.subgroup_size.is_some(),
            constants: &pipeline.constants,
            ..KernelDesc::default()
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
