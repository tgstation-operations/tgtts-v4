//! Weights read from a safetensors file a tensor at a time, as f16: the header parsed on
//! opening, each tensor's bytes read from its offsets when asked for, so a 4.6 GB checkpoint is
//! never in memory whole.

use std::{
    collections::BTreeMap,
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    thread,
};

use half::{bf16, f16};
use serde::Deserialize;

use crate::Error;

#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// An open safetensors file.
#[derive(Debug)]
pub struct Checkpoint {
    path: PathBuf,
    file: File,
    /// Where the data after the header starts.
    data: u64,
    entries: BTreeMap<String, Entry>,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_owned();
        let bad = |why: String| Error::Checkpoint(format!("{}: {why}", path.display()));
        let file = File::open(&path).map_err(|e| bad(e.to_string()))?;
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0)
            .map_err(|e| bad(e.to_string()))?;
        let len = u64::from_le_bytes(len);
        if len > 100 << 20 {
            return Err(bad(format!("a header of {len} bytes")));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|e| bad(e.to_string()))?;
        let mut entries: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|e| bad(e.to_string()))?;
        entries.remove("__metadata__");
        let entries = entries
            .into_iter()
            .map(|(name, v)| {
                let entry: Entry =
                    serde_json::from_value(v).map_err(|e| bad(format!("{name}: {e}")))?;
                Ok((name, entry))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Checkpoint {
            path,
            file,
            data: 8 + len,
            entries,
        })
    }

    /// The shape of tensor `name`, if the file has it.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Tensor `name` as f16 bits, rounded to nearest even from bf16 or f32 as burn's cast
    /// does, and its shape.
    pub fn f16(&self, name: &str) -> Result<(Vec<usize>, Vec<u16>), Error> {
        let bad =
            |why: String| Error::Checkpoint(format!("{}: {name}: {why}", self.path.display()));
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| bad("no such tensor".into()))?;
        let elements: usize = entry.shape.iter().product();
        let width = match entry.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(bad(format!("{other} weights"))),
        };
        let [start, end] = entry.data_offsets;
        if end - start != (elements * width) as u64 {
            return Err(bad(format!(
                "{} bytes for {elements} {}",
                end - start,
                entry.dtype
            )));
        }
        let mut bytes = vec![0u8; (end - start) as usize];
        self.file
            .read_exact_at(&mut bytes, self.data + start)
            .map_err(|e| bad(e.to_string()))?;
        let mut out = vec![0u16; elements];
        let convert = |src: &[u8], dst: &mut [u16]| match entry.dtype.as_str() {
            "BF16" => {
                for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    let v = bf16::from_bits(u16::from_le_bytes(*s)).to_f32();
                    *d = f16::from_f32(v).to_bits();
                }
            }
            "F16" => {
                for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *d = u16::from_le_bytes(*s);
                }
            }
            _ => {
                for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                    let v = f32::from_le_bytes(*s);
                    *d = f16::from_f32(v).to_bits();
                }
            }
        };
        let threads = thread::available_parallelism().map_or(1, |t| t.get());
        let chunk = elements.div_ceil(threads).max(1 << 16);
        thread::scope(|s| {
            for (src, dst) in bytes.chunks(chunk * width).zip(out.chunks_mut(chunk)) {
                s.spawn(move || convert(src, dst));
            }
        });
        Ok((entry.shape.clone(), out))
    }

    /// The 2-D tensors `names` as f16, joined along their rows, as upstream's projections are
    /// joined to run as one; and the joined shape.
    pub fn rows_f16(&self, names: &[String]) -> Result<([usize; 2], Vec<u16>), Error> {
        let mut rows = 0;
        let mut columns = None;
        let mut out = Vec::new();
        for name in names {
            let (shape, values) = self.f16(name)?;
            let &[r, c] = shape.as_slice() else {
                return Err(Error::Checkpoint(format!(
                    "{name}: shape {shape:?}, not 2-D"
                )));
            };
            if *columns.get_or_insert(c) != c {
                return Err(Error::Checkpoint(format!(
                    "{name}: {c} columns, not {columns:?}"
                )));
            }
            rows += r;
            out.extend_from_slice(&values);
        }
        Ok(([rows, columns.unwrap_or(0)], out))
    }
}
