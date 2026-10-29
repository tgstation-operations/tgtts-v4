//! Weights read from a PyTorch checkpoint, the zip archive `torch.save` writes, as VoxCPM2's
//! `audiovae.pth` is: a pickle of a dict whose `state_dict` maps each parameter's name to a view
//! of a storage, and each storage an uncompressed entry of its own. The pickle is read as far as
//! such a file needs, the opcodes of protocol 2 that `torch.save` writes for dicts, tuples and
//! tensors, and contiguous f32 and i32 tensors; anything else is refused. A tensor's values are
//! read from the file when asked for.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{Error, vae::Parameter};

/// The bytes of a value, f32 and i32 alike.
const WIDTH: usize = 4;

/// An open checkpoint, its tensors listed.
#[derive(Debug)]
pub struct Pth {
    path: PathBuf,
    file: File,
    tensors: BTreeMap<String, Tensor>,
}

/// A tensor of the state dict, its values contiguous in the file from `offset`.
#[derive(Debug)]
struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    offset: u64,
}

#[derive(Clone, Copy, Debug)]
enum Dtype {
    F32,
    I32,
}

impl Pth {
    /// Opens the checkpoint at `path` and lists the tensors of its `state_dict`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_owned();
        let bad = |why: String| Error::Checkpoint(format!("{}: {why}", path.display()));
        let file = File::open(&path).map_err(|e| bad(e.to_string()))?;
        let tensors = Archive::open(&file)
            .and_then(|archive| archive.tensors())
            .map_err(bad)?;
        Ok(Pth {
            path,
            file,
            tensors,
        })
    }

    /// The parameter `name` as f32, in torch's layout, and its shape: a weight-normed
    /// convolution's `<conv>.weight` folded from its `weight_g` and `weight_v` as torch folds
    /// them.
    pub fn floats(&self, name: &str) -> Result<Parameter, Error> {
        if let Some(tensor) = self.tensors.get(name) {
            return Ok((tensor.shape.clone(), self.values(name, tensor)?));
        }
        let part = |suffix: &str| {
            let part = format!("{name}{suffix}");
            match self.tensors.get(&part) {
                Some(tensor) => Ok((part, tensor)),
                None => Err(self.bad(name, "no such tensor".into())),
            }
        };
        let ((g_name, g), (v_name, v)) = (part("_g")?, part("_v")?);
        let fits = g.shape.len() == v.shape.len()
            && g.shape.first() == v.shape.first()
            && g.shape.iter().skip(1).all(|&dim| dim == 1);
        if !fits {
            let why = format!(
                "{g_name} of shape {:?} does not fit {v_name} of shape {:?}",
                g.shape, v.shape
            );
            return Err(self.bad(name, why));
        }
        let weight = weight_norm(&self.values(&g_name, g)?, &self.values(&v_name, v)?);
        Ok((v.shape.clone(), weight))
    }

    /// The values of `tensor`, read from the file, as f32.
    fn values(&self, name: &str, tensor: &Tensor) -> Result<Vec<f32>, Error> {
        let elements: usize = tensor.shape.iter().product();
        let mut bytes = vec![0u8; elements * WIDTH];
        self.file
            .read_exact_at(&mut bytes, tensor.offset)
            .map_err(|e| self.bad(name, e.to_string()))?;
        let words = bytes.as_chunks::<WIDTH>().0.iter().copied();
        Ok(match tensor.dtype {
            Dtype::F32 => words.map(f32::from_le_bytes).collect(),
            Dtype::I32 => words.map(|word| i32::from_le_bytes(word) as f32).collect(),
        })
    }

    fn bad(&self, name: &str, why: String) -> Error {
        Error::Checkpoint(format!("{}: {name}: {why}", self.path.display()))
    }
}

/// `g * v / |v|`, with the norm taken over every axis of `v` but the first, as torch's
/// `_weight_norm(v, g, 0)` computes it: in f32, from a norm accumulated in f64.
fn weight_norm(g: &[f32], v: &[f32]) -> Vec<f32> {
    let row = v.len() / g.len();
    v.chunks(row)
        .zip(g)
        .flat_map(|(v, &g)| {
            let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let scale = g / norm as f32;
            v.iter().map(move |&x| x * scale)
        })
        .collect()
}

/// The zip archive, as far as its central directory.
struct Archive<'a> {
    file: &'a File,
    len: u64,
    entries: HashMap<String, Entry>,
}

/// An entry of the central directory.
struct Entry {
    method: u16,
    compressed: u64,
    size: u64,
    /// Where its local header starts.
    header: u64,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

impl<'a> Archive<'a> {
    fn open(file: &'a File) -> Result<Self, String> {
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut archive = Archive {
            file,
            len,
            entries: HashMap::new(),
        };
        // torch.save writes no comment, which would follow the end record.
        let end = len.checked_sub(22).ok_or("too short for a zip archive")?;
        let end = archive.read(end, 22)?;
        if u32_at(&end, 0) != 0x0605_4b50 {
            return Err("no zip end record at the end".into());
        }
        let (count, size, offset) = (u16_at(&end, 10), u32_at(&end, 12), u32_at(&end, 16));
        if count == u16::MAX || size == u32::MAX || offset == u32::MAX {
            return Err("a zip64 archive".into());
        }
        let directory = archive.read(offset.into(), size.into())?;
        let mut at = 0;
        for _ in 0..count {
            let truncated = "a truncated central directory";
            let fixed = directory.get(at..at + 46).ok_or(truncated)?;
            if u32_at(fixed, 0) != 0x0201_4b50 {
                return Err("a corrupt central directory".into());
            }
            let name_len = usize::from(u16_at(fixed, 28));
            let name = directory
                .get(at + 46..at + 46 + name_len)
                .ok_or(truncated)?;
            let name = String::from_utf8(name.to_vec()).map_err(|e| e.to_string())?;
            let entry = Entry {
                method: u16_at(fixed, 10),
                compressed: u32_at(fixed, 20).into(),
                size: u32_at(fixed, 24).into(),
                header: u32_at(fixed, 42).into(),
            };
            archive.entries.insert(name, entry);
            at += 46 + name_len + usize::from(u16_at(fixed, 30)) + usize::from(u16_at(fixed, 32));
        }
        Ok(archive)
    }

    /// `len` bytes at `offset`, refused past the end of the file.
    fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, String> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(format!("{len} bytes at {offset}, past the end of the file"));
        }
        let mut bytes = vec![0; len as usize];
        self.file
            .read_exact_at(&mut bytes, offset)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }

    /// Where the entry `name` starts and its size, which torch.save stores uncompressed.
    fn entry(&self, name: &str) -> Result<(u64, u64), String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("no entry {name}"))?;
        if entry.method != 0 || entry.compressed != entry.size {
            return Err(format!("{name} is compressed"));
        }
        let header = self.read(entry.header, 30)?;
        if u32_at(&header, 0) != 0x0403_4b50 {
            return Err(format!("{name} has no local header"));
        }
        let names = u64::from(u16_at(&header, 26)) + u64::from(u16_at(&header, 28));
        let start = entry.header + 30 + names;
        if start + entry.size > self.len {
            return Err(format!("{name} runs past the end of the file"));
        }
        Ok((start, entry.size))
    }

    /// The tensors of the pickle's `state_dict`, under its names.
    fn tensors(&self) -> Result<BTreeMap<String, Tensor>, String> {
        let pickles: Vec<&str> = self
            .entries
            .keys()
            .filter_map(|name| name.strip_suffix("/data.pkl"))
            .filter(|root| !root.contains('/'))
            .collect();
        let [root] = pickles[..] else {
            return Err(format!("{} entries <root>/data.pkl", pickles.len()));
        };
        if self.entries.contains_key(&format!("{root}/byteorder")) {
            let (start, size) = self.entry(&format!("{root}/byteorder"))?;
            if self.read(start, size)? != b"little" {
                return Err("not little-endian".into());
            }
        }
        let (start, size) = self.entry(&format!("{root}/data.pkl"))?;
        let Value::Dict(saved) = unpickle(&self.read(start, size)?)? else {
            return Err("the pickle is not of a dict".into());
        };
        let saved = saved.borrow();
        let state_dict = saved.iter().find_map(|(key, value)| {
            matches!(key, Value::Str(key) if &**key == "state_dict").then_some(value)
        });
        let Some(Value::Dict(state_dict)) = state_dict else {
            return Err("no state_dict".into());
        };
        let tensor = |(name, value): &(Value, Value)| {
            let (Value::Str(name), Value::Tensor(view)) = (name, value) else {
                return Err(format!(
                    "{name:?} in the state_dict, which is not a named tensor"
                ));
            };
            Ok((name.to_string(), self.tensor(root, name, view)?))
        };
        state_dict.borrow().iter().map(tensor).collect()
    }

    /// Where the values of `view`, the tensor `name`, are in the file.
    fn tensor(&self, root: &str, name: &str, view: &View) -> Result<Tensor, String> {
        let View {
            storage,
            offset,
            shape,
            stride,
        } = view;
        let (start, size) = self.entry(&format!("{root}/data/{}", storage.key))?;
        let len = storage.len;
        if len
            .checked_mul(WIDTH)
            .is_none_or(|bytes| bytes as u64 != size)
        {
            return Err(format!("{name}: {size} bytes of storage for {len} values"));
        }
        let contiguous = shape.iter().rev().scan(1, |stride: &mut usize, &dim| {
            let this = *stride;
            *stride = stride.checked_mul(dim)?;
            Some(this)
        });
        if !stride.iter().rev().copied().eq(contiguous) {
            return Err(format!("{name}: strides {stride:?} for shape {shape:?}"));
        }
        let elements = shape
            .iter()
            .try_fold(1, |n: usize, &dim| n.checked_mul(dim));
        let end = elements.and_then(|elements| elements.checked_add(*offset));
        if end.is_none_or(|end| end > len) {
            return Err(format!(
                "{name}: shape {shape:?} from {offset} of {len} values"
            ));
        }
        Ok(Tensor {
            dtype: storage.dtype,
            shape: shape.clone(),
            offset: start + (offset * WIDTH) as u64,
        })
    }
}

/// What the pickle builds, as far as a checkpoint uses it.
#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    /// What `torch.save` writes for a tensor's `requires_grad`.
    False,
    Str(Rc<str>),
    Tuple(Rc<[Value]>),
    /// A dict or an `OrderedDict`, which the memo shares and items are set into in place.
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    /// A class or a function, by its module and name.
    Global(Rc<str>, Rc<str>),
    Storage(Storage),
    Tensor(Rc<View>),
}

/// A storage, by its entry's name under `data/`, with its type and its length in values.
#[derive(Clone, Debug)]
struct Storage {
    dtype: Dtype,
    key: Rc<str>,
    len: usize,
}

/// A tensor as `_rebuild_tensor_v2` makes it, a view of a storage, in values.
#[derive(Debug)]
struct View {
    storage: Storage,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// The object the pickle `bytes` builds.
fn unpickle(bytes: &[u8]) -> Result<Value, String> {
    let mut input = Input { bytes, at: 0 };
    let mut stack = Vec::new();
    let mut marks = Vec::new();
    let mut memo = HashMap::new();
    let pop = |stack: &mut Vec<Value>| stack.pop().ok_or("a pop of an empty stack");
    let marked = |stack: &mut Vec<Value>, marks: &mut Vec<usize>| match marks.pop() {
        Some(mark) if mark <= stack.len() => Ok(stack.split_off(mark)),
        _ => Err("an opcode with no mark before it"),
    };
    let dict = |value: Option<&Value>| match value {
        Some(Value::Dict(dict)) => Ok(dict.clone()),
        _ => Err("items set into other than a dict"),
    };
    loop {
        let op = input.take::<1>()?[0];
        match op {
            // PROTO
            0x80 => {
                input.take::<1>()?;
            }
            // MARK
            b'(' => marks.push(stack.len()),
            // EMPTY_DICT
            b'}' => stack.push(Value::Dict(Rc::default())),
            // EMPTY_TUPLE
            b')' => stack.push(Value::Tuple(Rc::new([]))),
            // NEWFALSE
            0x89 => stack.push(Value::False),
            // BININT1, BININT2, BININT
            b'K' => stack.push(Value::Int(u8::from_le_bytes(input.take()?).into())),
            b'M' => stack.push(Value::Int(u16::from_le_bytes(input.take()?).into())),
            b'J' => stack.push(Value::Int(i32::from_le_bytes(input.take()?).into())),
            // BINUNICODE
            b'X' => {
                let len = u32::from_le_bytes(input.take()?);
                let text = std::str::from_utf8(input.slice(len as usize)?);
                stack.push(Value::Str(text.map_err(|e| e.to_string())?.into()));
            }
            // GLOBAL
            b'c' => {
                let module = input.line()?.into();
                stack.push(Value::Global(module, input.line()?.into()));
            }
            // BINPUT, LONG_BINPUT
            b'q' | b'r' => {
                let key = match op {
                    b'q' => u8::from_le_bytes(input.take()?).into(),
                    _ => u32::from_le_bytes(input.take()?),
                };
                memo.insert(key, stack.last().ok_or("a put of an empty stack")?.clone());
            }
            // BINGET, LONG_BINGET
            b'h' | b'j' => {
                let key = match op {
                    b'h' => u8::from_le_bytes(input.take()?).into(),
                    _ => u32::from_le_bytes(input.take()?),
                };
                let value = memo.get(&key).ok_or("a get of what was never put")?;
                stack.push(value.clone());
            }
            // TUPLE
            b't' => {
                let items = marked(&mut stack, &mut marks)?;
                stack.push(Value::Tuple(items.into()));
            }
            // TUPLE1, TUPLE2, TUPLE3
            0x85..=0x87 => {
                let len = usize::from(op - 0x84);
                let start = stack
                    .len()
                    .checked_sub(len)
                    .ok_or("a tuple of an empty stack")?;
                let items = stack.split_off(start);
                stack.push(Value::Tuple(items.into()));
            }
            // SETITEM
            b's' => {
                let (value, key) = (pop(&mut stack)?, pop(&mut stack)?);
                dict(stack.last())?.borrow_mut().push((key, value));
            }
            // SETITEMS
            b'u' => {
                let items = marked(&mut stack, &mut marks)?;
                if items.len() % 2 != 0 {
                    return Err("a key with no value".into());
                }
                let mut items = items.into_iter();
                let dict = dict(stack.last())?;
                while let (Some(key), Some(value)) = (items.next(), items.next()) {
                    dict.borrow_mut().push((key, value));
                }
            }
            // REDUCE
            b'R' => {
                let (args, callable) = (pop(&mut stack)?, pop(&mut stack)?);
                stack.push(reduce(&callable, &args)?);
            }
            // BINPERSID
            b'Q' => {
                let id = pop(&mut stack)?;
                stack.push(persistent(&id)?);
            }
            // BUILD: the attributes of an OrderedDict, such as a state dict's `_metadata`,
            // which no tensor needs.
            b'b' => {
                pop(&mut stack)?;
                dict(stack.last())?;
            }
            // STOP
            b'.' => return Ok(pop(&mut stack)?),
            _ => return Err(format!("pickle opcode {op:#04x}")),
        }
    }
}

/// The pickle's input, read from its start.
struct Input<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Input<'a> {
    fn slice(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let rest = &self.bytes[self.at..];
        let slice = rest.get(..len).ok_or("a truncated pickle")?;
        self.at += len;
        Ok(slice)
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], &'static str> {
        Ok(self.slice(N)?.try_into().expect("N bytes"))
    }

    fn line(&mut self) -> Result<&'a str, String> {
        let len = self.bytes[self.at..].iter().position(|&b| b == b'\n');
        let line = self.slice(len.ok_or("a truncated pickle")? + 1)?;
        std::str::from_utf8(&line[..line.len() - 1]).map_err(|e| e.to_string())
    }
}

/// What `callable` returns for `args`: an empty `OrderedDict`, or a tensor, whose
/// `requires_grad` and backward hooks do not change its values.
fn reduce(callable: &Value, args: &Value) -> Result<Value, String> {
    let (Value::Global(module, name), Value::Tuple(args)) = (callable, args) else {
        return Err(format!("a call of {callable:?}"));
    };
    match (&**module, &**name, &args[..]) {
        ("collections", "OrderedDict", []) => Ok(Value::Dict(Rc::default())),
        (
            "torch._utils",
            "_rebuild_tensor_v2",
            [
                Value::Storage(storage),
                Value::Int(offset),
                Value::Tuple(shape),
                Value::Tuple(stride),
                ..,
            ],
        ) => Ok(Value::Tensor(Rc::new(View {
            storage: storage.clone(),
            offset: index(*offset)?,
            shape: indices(shape)?,
            stride: indices(stride)?,
        }))),
        _ => Err(format!("a call of {module}.{name} on {args:?}")),
    }
}

/// The storage the persistent id `('storage', type, key, location, length)` names.
fn persistent(id: &Value) -> Result<Value, String> {
    let unknown = || format!("a persistent id {id:?}");
    let Value::Tuple(fields) = id else {
        return Err(unknown());
    };
    let [
        Value::Str(kind),
        Value::Global(module, name),
        Value::Str(key),
        _,
        Value::Int(len),
    ] = &fields[..]
    else {
        return Err(unknown());
    };
    let dtype = match (&**kind, &**module, &**name) {
        ("storage", "torch", "FloatStorage") => Dtype::F32,
        ("storage", "torch", "IntStorage") => Dtype::I32,
        _ => return Err(unknown()),
    };
    Ok(Value::Storage(Storage {
        dtype,
        key: key.clone(),
        len: index(*len)?,
    }))
}

fn index(value: i64) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("an index of {value}"))
}

fn indices(values: &[Value]) -> Result<Vec<usize>, String> {
    let index = |value: &Value| match value {
        Value::Int(value) => index(*value),
        _ => Err(format!("an index of {value:?}")),
    };
    values.iter().map(index).collect()
}
