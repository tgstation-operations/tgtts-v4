//! Test support for the crates that take it as a dev-dependency.
//!
//! A crate is checked against what it replaces, upstream VoxCPM, the Python services or ffmpeg,
//! through fixtures: safetensors files that `nix run .#fixtures` writes from the generators in
//! `fixtures/`, each holding the inputs a generator ran that on and the outputs it gave.
//!
//! The GPU's tests on the real weights are held to goldens: what the GPU gave in each when it
//! was captured, on a tree whose output burn's CPU oracles had shown right, kept as JSON under
//! the directory `GOLDEN_DIR` names ([`golden`]).

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::{Serialize, de::DeserializeOwned};

/// Opens `tests/fixtures/<name>.safetensors` in the calling crate, as a [`Fixture`].
#[macro_export]
macro_rules! fixture {
    ($name:literal) => {
        $crate::Fixture::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/",
            $name,
            ".safetensors"
        ))
    };
}

/// A safetensors file, read whole: fit for fixtures, not for the checkpoint.
pub struct Fixture {
    path: PathBuf,
    bytes: Vec<u8>,
    metadata: BTreeMap<String, String>,
}

impl Fixture {
    /// Reads the file at `path`.
    ///
    /// # Panics
    ///
    /// If it cannot be read or is not a safetensors file.
    #[track_caller]
    pub fn open(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_owned();
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if let Err(e) = SafeTensors::deserialize(&bytes) {
            panic!("{}: not a safetensors file: {e}", path.display());
        }
        let (_, header) = SafeTensors::read_metadata(&bytes).expect("deserialized above");
        let metadata = header.metadata().clone().unwrap_or_default();
        Self {
            path,
            bytes,
            metadata: metadata.into_iter().collect(),
        }
    }

    /// Where it was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The string metadata in its header.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// The shape of the tensor `name` and its values as f32, widened if it is stored as bf16 or
    /// f16.
    ///
    /// # Panics
    ///
    /// If there is no such tensor, or it is not one of those floats.
    #[track_caller]
    pub fn floats(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let file = self.path.display();
        let (shape, dtype, bytes) = self.raw(name);
        let (halves, _) = bytes.as_chunks();
        let (words, _) = bytes.as_chunks();
        let values = match dtype {
            Dtype::F32 => words.iter().map(|&b| f32::from_le_bytes(b)).collect(),
            Dtype::F16 => halves
                .iter()
                .map(|&b| f16::from_le_bytes(b).into())
                .collect(),
            Dtype::BF16 => halves
                .iter()
                .map(|&b| bf16::from_le_bytes(b).into())
                .collect(),
            other => panic!("{file}: tensor {name:?} is {other:?}, not F32, F16 or BF16"),
        };
        (shape, values)
    }

    /// The shape of the integer tensor `name` and its values, such as the token ids a model is
    /// run on.
    ///
    /// # Panics
    ///
    /// If there is no such tensor, or it is not I64, the integer type torch gives indices.
    #[track_caller]
    pub fn ints(&self, name: &str) -> (Vec<usize>, Vec<i64>) {
        let file = self.path.display();
        let (shape, dtype, bytes) = self.raw(name);
        if dtype != Dtype::I64 {
            panic!("{file}: tensor {name:?} is {dtype:?}, not I64");
        }
        let (words, _) = bytes.as_chunks();
        let values = words.iter().map(|&b| i64::from_le_bytes(b)).collect();
        (shape, values)
    }

    #[track_caller]
    fn raw(&self, name: &str) -> (Vec<usize>, Dtype, Vec<u8>) {
        let file = self.path.display();
        let tensors = SafeTensors::deserialize(&self.bytes).expect("deserialized when opened");
        let view = tensors.tensor(name).unwrap_or_else(|_| {
            let mut names = tensors.names();
            names.sort_unstable();
            panic!("{file}: no tensor {name:?} among {names:?}")
        });
        (view.shape().to_vec(), view.dtype(), view.data().to_vec())
    }
}

/// The VoxCPM2 checkpoint directory that `VOXCPM2_DIR` names, for tests that need the real
/// weights. Such a test is `#[ignore]`d with a reason naming the variable, so the gate never
/// needs the checkpoint, and runs under `cargo test -- --ignored`.
///
/// # Panics
///
/// If the variable is unset or the directory has no `config.json`.
#[track_caller]
pub fn checkpoint_dir() -> PathBuf {
    let Some(dir) = std::env::var_os("VOXCPM2_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOXCPM2_DIR is not set; point it at a VoxCPM2 checkpoint directory");
    };
    // A test runs in its crate's directory, which a relative path is then taken from.
    let dir = std::path::absolute(dir).expect("absolute of a non-empty path");
    if !dir.join("config.json").is_file() {
        panic!("VOXCPM2_DIR: {} has no config.json", dir.display());
    }
    dir
}

/// The directory `GOLDEN_DIR` names, which holds the goldens and the inputs some tests take as
/// they are. A test that needs it is `#[ignore]`d with a reason naming the variable, as one that
/// needs the checkpoint is.
///
/// # Panics
///
/// If the variable is unset.
#[track_caller]
pub fn golden_dir() -> PathBuf {
    let Some(dir) = std::env::var_os("GOLDEN_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("GOLDEN_DIR is not set; point it at the goldens' directory");
    };
    std::path::absolute(dir).expect("absolute of a non-empty path")
}

/// Whether `GOLDEN_CAPTURE` is `1`, which has the tests capture their goldens rather than hold
/// the GPU to them.
pub fn capturing() -> bool {
    std::env::var_os("GOLDEN_CAPTURE").is_some_and(|value| value == "1")
}

/// The golden `name`, `<name>.json` under [`golden_dir`]: `capture`'s result, written there
/// first when [`capturing`], and read from there otherwise, never computed where it is missing.
/// A golden is captured at the device's default tuning on a tree whose output is known right,
/// and captured again when a change moves what the GPU computes on purpose.
///
/// A test holds each run to its golden within a tolerance that holds from one GPU to another,
/// whose subgroups and tunings round differently: no run is further from the golden than the two
/// are each from f32, so a tolerance is the largest error against f32 of any run on the GPUs
/// measured, plus the golden's own, a quarter over, as burn's CPU backend in f32 measured them
/// when the goldens were first captured.
///
/// # Panics
///
/// If the golden is missing or is not a `T`, or it cannot be written.
#[track_caller]
pub fn golden<T: Serialize + DeserializeOwned>(name: &str, capture: impl FnOnce() -> T) -> T {
    let path = golden_dir().join(format!("{name}.json"));
    if capturing() {
        let value = capture();
        std::fs::write(&path, serde_json::to_vec(&value).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        eprintln!("captured {}", path.display());
        return value;
    }
    let missing = "capture it with GOLDEN_CAPTURE=1 on a tree whose output is known right";
    read(&path, missing)
}

/// The input `name`, `<name>.json` under [`golden_dir`], which a test takes as it is: what
/// burn's CPU model computed for it, kept with the goldens since nothing here computes it. No
/// test writes one.
///
/// # Panics
///
/// If it is missing or is not a `T`.
#[track_caller]
pub fn golden_input<T: DeserializeOwned>(name: &str) -> T {
    let path = golden_dir().join(format!("{name}.json"));
    let missing = "burn's CPU model computed it, and it is kept with the goldens";
    read(&path, missing)
}

#[track_caller]
fn read<T: DeserializeOwned>(path: &Path, missing: &str) -> T {
    let json = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}; {missing}", path.display()));
    serde_json::from_slice(&json).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}
