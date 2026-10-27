//! Test support for the model crates, which take it as a dev-dependency.
//!
//! A model is checked against upstream VoxCPM through fixtures: safetensors files that
//! `nix run .#fixtures` writes from the generators in `fixtures/`, each holding an upstream
//! module's weights under their state_dict names, the inputs it was run on and the outputs it
//! gave.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use safetensors::{Dtype, SafeTensors};
use serde::{Serialize, de::DeserializeOwned};

/// The device every test runs on.
pub fn device() -> Device {
    Device::flex()
}

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

    /// Where it was read from, for loading its weights into a module through burn's store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The string metadata in its header.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// The tensor `name` on `device` as f32, widened if it is stored as bf16 or f16.
    ///
    /// # Panics
    ///
    /// If there is no such tensor, it is not of rank `D`, or it is not one of those floats.
    #[track_caller]
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Tensor<D> {
        let file = self.path.display();
        let (shape, dtype, bytes) = self.view(name, D);
        let dtype = match dtype {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            other => panic!("{file}: tensor {name:?} is {other:?}, not F32, F16 or BF16"),
        };
        let data = TensorData::from_bytes_vec(bytes, shape, dtype);
        Tensor::from_data(data.convert::<f32>(), device)
    }

    /// The integer tensor `name` on `device`, such as the token ids a model is run on.
    ///
    /// # Panics
    ///
    /// If there is no such tensor, it is not of rank `D`, or it is not I64, the integer type
    /// torch gives indices.
    #[track_caller]
    pub fn int_tensor<const D: usize>(&self, name: &str, device: &Device) -> Tensor<D, Int> {
        let file = self.path.display();
        let (shape, dtype, bytes) = self.view(name, D);
        if dtype != Dtype::I64 {
            panic!("{file}: tensor {name:?} is {dtype:?}, not I64");
        }
        Tensor::from_data(TensorData::from_bytes_vec(bytes, shape, DType::I64), device)
    }

    #[track_caller]
    fn view(&self, name: &str, rank: usize) -> (Vec<usize>, Dtype, Vec<u8>) {
        let file = self.path.display();
        let tensors = SafeTensors::deserialize(&self.bytes).expect("deserialized when opened");
        let view = tensors.tensor(name).unwrap_or_else(|_| {
            let mut names = tensors.names();
            names.sort_unstable();
            panic!("{file}: no tensor {name:?} among {names:?}")
        });
        let shape = view.shape();
        if shape.len() != rank {
            panic!("{file}: tensor {name:?} has shape {shape:?}, not rank {rank}");
        }
        (shape.to_vec(), view.dtype(), view.data().to_vec())
    }
}

/// Asserts that every element of `actual` is within `atol + rtol * |expected|` of `expected`,
/// the test `torch.testing.assert_close` applies. NaN is never close.
///
/// # Panics
///
/// If the shapes differ or any element is not close, with the shape, the largest absolute and
/// relative differences, and the index of the element furthest out of tolerance.
#[track_caller]
pub fn assert_close<const D: usize>(actual: Tensor<D>, expected: Tensor<D>, atol: f32, rtol: f32) {
    let (shape, expected_shape) = (actual.dims(), expected.dims());
    if shape != expected_shape {
        panic!("shapes differ: actual {shape:?}, expected {expected_shape:?}");
    }
    let values = |tensor: Tensor<D>| tensor.into_data().iter::<f32>().collect::<Vec<_>>();
    if let Some(report) = mismatch(&values(actual), &values(expected), &shape, atol, rtol) {
        panic!("{report}");
    }
}

fn mismatch(
    actual: &[f32],
    expected: &[f32],
    shape: &[usize],
    atol: f32,
    rtol: f32,
) -> Option<String> {
    let (mut failed, mut max_abs, mut max_rel) = (0, 0f64, 0f64);
    let mut worst = (0, f64::NEG_INFINITY);
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let (abs, rel, excess) = if a == e {
            (0.0, 0.0, f64::NEG_INFINITY)
        } else if a.is_finite() && e.is_finite() {
            let abs = (f64::from(a) - f64::from(e)).abs();
            let scale = f64::from(e).abs();
            let tolerance = f64::from(atol) + f64::from(rtol) * scale;
            (abs, abs / scale, abs - tolerance)
        } else {
            (f64::INFINITY, f64::INFINITY, f64::INFINITY)
        };
        if excess > 0.0 {
            failed += 1;
        }
        if excess > worst.1 {
            worst = (i, excess);
        }
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    if failed == 0 {
        return None;
    }
    let (i, _) = worst;
    Some(format!(
        "{failed} of {} elements not close (atol {atol:e}, rtol {rtol:e}) in shape {shape:?}\n\
         max abs diff {max_abs:.3e}, max rel diff {max_rel:.3e}\n\
         worst at {:?}: actual {}, expected {}",
        actual.len(),
        unravel(i, shape),
        actual[i],
        expected[i],
    ))
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for (axis, &len) in shape.iter().enumerate().rev() {
        index[axis] = flat % len;
        flat /= len;
    }
    index
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

/// The directory `ORACLE_DIR` names, where [`kept`] keeps what it computes, when it is set.
pub fn oracle_dir() -> Option<PathBuf> {
    std::env::var_os("ORACLE_DIR")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
}

/// `compute`'s result, kept as JSON under [`oracle_dir`] when it is set, by `name`: read from
/// there when a run before left it. A test keeps its CPU oracle so, to compute it once, or on
/// another host than the one whose GPU it runs on. A change to what computes it wants the file
/// removed.
///
/// # Panics
///
/// If the kept file is not a `T`, or the result cannot be kept.
#[track_caller]
pub fn kept<T: Serialize + DeserializeOwned>(name: &str, compute: impl FnOnce() -> T) -> T {
    let Some(dir) = oracle_dir() else {
        return compute();
    };
    let path = dir.join(format!("{name}.json"));
    if let Ok(json) = std::fs::read(&path) {
        eprintln!("the oracle from {}", path.display());
        return serde_json::from_slice(&json).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
    let value = compute();
    std::fs::write(&path, serde_json::to_vec(&value).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    value
}
