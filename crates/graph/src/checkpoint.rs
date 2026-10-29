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

use serde::Deserialize;
use vulkan::{BF16, F16};

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

    /// The entry of tensor `name`, and the bytes of each of its values, if its size is that of
    /// its shape.
    fn entry(&self, name: &str) -> Result<(&Entry, usize), Error> {
        let bad = |why: String| self.bad(name, why);
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
        Ok((entry, width))
    }

    fn bad(&self, name: &str, why: String) -> Error {
        Error::Checkpoint(format!("{}: {name}: {why}", self.path.display()))
    }

    /// Tensor `name` as f16, rounded to nearest even from bf16 or f32 ([`F16::from_bf16`],
    /// [`F16::from_f32`]), and its shape.
    pub fn f16(&self, name: &str) -> Result<(Vec<usize>, Vec<F16>), Error> {
        let (entry, width) = self.entry(name)?;
        let [start, end] = entry.data_offsets;
        let elements = (end - start) as usize / width;
        let mut bytes = vec![0u8; (end - start) as usize];
        self.file
            .read_exact_at(&mut bytes, self.data + start)
            .map_err(|e| self.bad(name, e.to_string()))?;
        let mut out = vec![F16::ZERO; elements];
        let convert = |src: &[u8], dst: &mut [F16]| convert(&entry.dtype, src, dst);
        let threads = thread::available_parallelism().map_or(1, |t| t.get());
        let chunk = elements.div_ceil(threads).max(1 << 16);
        thread::scope(|s| {
            for (src, dst) in bytes.chunks(chunk * width).zip(out.chunks_mut(chunk)) {
                s.spawn(move || convert(src, dst));
            }
        });
        Ok((entry.shape.clone(), out))
    }

    /// Rows `rows` of the 2-D tensor `name` as f16, converted as [`f16`](Self::f16) converts
    /// them, one after another: a lookup in an embedding table, which reads those rows alone.
    pub fn gather_f16(&self, name: &str, rows: &[usize]) -> Result<Vec<F16>, Error> {
        let (entry, width) = self.entry(name)?;
        let &[count, columns] = entry.shape.as_slice() else {
            return Err(self.bad(name, format!("shape {:?}, not 2-D", entry.shape)));
        };
        let mut bytes = vec![0u8; columns * width];
        let mut out = vec![F16::ZERO; rows.len() * columns];
        for (&row, dst) in rows.iter().zip(out.chunks_mut(columns)) {
            if row >= count {
                return Err(self.bad(name, format!("row {row} of {count}")));
            }
            let at = self.data + entry.data_offsets[0] + (row * bytes.len()) as u64;
            self.file
                .read_exact_at(&mut bytes, at)
                .map_err(|e| self.bad(name, e.to_string()))?;
            convert(&entry.dtype, &bytes, dst);
        }
        Ok(out)
    }

    /// The 2-D tensors `names` as f16, joined along their rows, as upstream's projections are
    /// joined to run as one; and the joined shape.
    pub fn rows_f16(&self, names: &[String]) -> Result<([usize; 2], Vec<F16>), Error> {
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

/// `src`, values of `dtype` as the checkpoint stores them, as f16 into `dst`.
fn convert(dtype: &str, src: &[u8], dst: &mut [F16]) {
    match dtype {
        "BF16" => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = F16::from_bf16(BF16::from_le_bytes(*s));
            }
        }
        "F16" => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = F16::from_le_bytes(*s);
            }
        }
        _ => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = F16::from_f32(f32::from_le_bytes(*s));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gather_reads_the_rows_the_whole_tensor_holds() {
        // Quarters, which bfloat16 holds exactly: the upper halves of their f32s.
        let rows: Vec<f32> = (0..12).map(|i| i as f32 / 4.0 - 1.5).collect();
        let data: Vec<u8> = rows
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"table":{{"dtype":"BF16","shape":[3,4],"data_offsets":[0,{}]}}}}"#,
            data.len()
        );
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header.as_bytes());
        file.extend(&data);
        let path = std::env::temp_dir().join(format!("gather-{}.safetensors", std::process::id()));
        std::fs::write(&path, file).unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        let (_, whole) = checkpoint.f16("table").unwrap();
        let gathered = checkpoint.gather_f16("table", &[2, 0, 2]).unwrap();
        std::fs::remove_file(&path).unwrap();
        let row = |r: usize| &whole[r * 4..(r + 1) * 4];
        assert_eq!(gathered, [row(2), row(0), row(2)].concat());
        assert!(matches!(
            checkpoint.gather_f16("table", &[3]),
            Err(Error::Checkpoint(_))
        ));
    }
}
