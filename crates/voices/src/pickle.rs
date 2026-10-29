//! An evaluator for exactly the pickles the Python service's voice files hold.
//!
//! Those are protocol 2 pickles of dicts, strings, bytes (which protocol 2 writes as a call to
//! `_codecs.encode`) and `io.BytesIO` objects. This implements the 18 opcodes they use and knows
//! two globals, `_codecs.encode` and `_io.BytesIO`, by name; every other opcode or global is
//! refused with its offset. Nothing is imported or called: the only objects it makes are
//! [`Value`]s.

use std::{collections::BTreeMap, mem};

/// What a pickle evaluates to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// `None`.
    None,
    /// An int.
    Int(i64),
    /// A str.
    Str(String),
    /// A bytes, as `_codecs.encode(str, "latin1")` makes it.
    Bytes(Vec<u8>),
    /// A tuple.
    Tuple(Vec<Value>),
    /// A dict, whose keys here are always strings.
    Dict(BTreeMap<String, Value>),
    /// An `io.BytesIO` holding these bytes.
    BytesIo(Vec<u8>),
    /// An `io.BytesIO` made but not yet given its contents.
    EmptyBytesIo,
    /// One of the two globals, not called.
    Global(Global),
}

/// The globals a pickle may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Global {
    /// `_codecs.encode`.
    Encode,
    /// `_io.BytesIO`.
    BytesIo,
}

/// A pickle refused, at the offset of the opcode that was.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("pickle offset {offset}: {kind}")]
pub struct Error {
    /// Where the opcode is in the pickle.
    pub offset: usize,
    /// Why it was refused.
    pub kind: ErrorKind,
}

/// Why a pickle was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ErrorKind {
    /// An opcode this does not implement.
    #[error("opcode {0:#04x} is not supported")]
    Opcode(u8),
    /// A global other than the two this knows.
    #[error("global {module}.{name} is not allowed")]
    Global {
        /// The module it names.
        module: String,
        /// The name in the module.
        name: String,
    },
    /// The pickle ends inside the opcode, or before a STOP.
    #[error("truncated")]
    Truncated,
    /// The opcode is applied to what it cannot be, or the pickle is otherwise not one the
    /// voice files hold.
    #[error("{0}")]
    Invalid(String),
}

/// Evaluates `pickle`, refusing anything but what the voice files hold.
pub fn load(pickle: &[u8]) -> Result<Value, Error> {
    let mut machine = Machine {
        pickle,
        pos: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: BTreeMap::new(),
    };
    if pickle.first() != Some(&PROTO) {
        let kind = invalid("no PROTO at the start");
        return Err(Error { offset: 0, kind });
    }
    loop {
        let offset = machine.pos;
        let opcode = machine.take(1).map_err(|kind| Error { offset, kind })?[0];
        match machine.step(offset, opcode) {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {}
            Err(kind) => return Err(Error { offset, kind }),
        }
    }
}

const PROTO: u8 = 0x80;
const STOP: u8 = b'.';
const MARK: u8 = b'(';
const NONE: u8 = b'N';
const BININT1: u8 = b'K';
const BINUNICODE: u8 = b'X';
const EMPTY_TUPLE: u8 = b')';
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const EMPTY_DICT: u8 = b'}';
const SETITEMS: u8 = b'u';
const GLOBAL: u8 = b'c';
const REDUCE: u8 = b'R';
const NEWOBJ: u8 = 0x81;
const BUILD: u8 = b'b';
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const BINGET: u8 = b'h';

/// A memo entry. A dict or BytesIO is changed after it is put, and fetching it, or a tuple
/// holding it, would have to share it; the files never fetch one, so it is not kept.
enum Memo {
    Value(Value),
    Mutable,
}

struct Machine<'a> {
    pickle: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    /// The stack's length at each open MARK.
    marks: Vec<usize>,
    memo: BTreeMap<u32, Memo>,
}

fn invalid(reason: impl Into<String>) -> ErrorKind {
    ErrorKind::Invalid(reason.into())
}

impl<'a> Machine<'a> {
    /// Runs the opcode at `offset`, giving the pickle's value if it was STOP.
    fn step(&mut self, offset: usize, opcode: u8) -> Result<Option<Value>, ErrorKind> {
        match opcode {
            PROTO => {
                let protocol = self.take(1)?[0];
                if offset != 0 {
                    return Err(invalid("PROTO after the start"));
                }
                if protocol != 2 {
                    return Err(invalid(format!(
                        "protocol {protocol}, where only 2 is read"
                    )));
                }
            }
            STOP => {
                if !self.marks.is_empty() || self.stack.len() != 1 {
                    return Err(invalid("STOP with other than one value on the stack"));
                }
                if self.pos != self.pickle.len() {
                    return Err(invalid("bytes after STOP"));
                }
                return Ok(self.stack.pop());
            }
            MARK => self.marks.push(self.stack.len()),
            NONE => self.stack.push(Value::None),
            BININT1 => {
                let int = self.take(1)?[0];
                self.stack.push(Value::Int(int.into()));
            }
            BINUNICODE => {
                let len = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
                let bytes = self.take(len as usize)?;
                let string = str::from_utf8(bytes).map_err(|_| invalid("str is not UTF-8"))?;
                self.stack.push(Value::Str(string.to_owned()));
            }
            EMPTY_TUPLE => self.stack.push(Value::Tuple(Vec::new())),
            TUPLE2 | TUPLE3 => {
                let len = if opcode == TUPLE2 { 2 } else { 3 };
                if self.frame_len() < len {
                    return Err(invalid(format!(
                        "TUPLE{len} with fewer values on the stack"
                    )));
                }
                let items = self.stack.split_off(self.stack.len() - len);
                self.stack.push(Value::Tuple(items));
            }
            EMPTY_DICT => self.stack.push(Value::Dict(BTreeMap::new())),
            SETITEMS => {
                let items = self.pop_mark()?;
                if items.len() % 2 != 0 {
                    return Err(invalid("SETITEMS with a key and no value"));
                }
                let Some(Value::Dict(dict)) = self.top() else {
                    return Err(invalid("SETITEMS on what is not a dict"));
                };
                let mut items = items.into_iter();
                while let (Some(key), Some(value)) = (items.next(), items.next()) {
                    let Value::Str(key) = key else {
                        return Err(invalid(format!("dict key {}", describe(&key))));
                    };
                    if dict.contains_key(&key) {
                        return Err(invalid(format!("dict key {key:?} set twice")));
                    }
                    dict.insert(key, value);
                }
            }
            GLOBAL => {
                let module = self.line()?;
                let name = self.line()?;
                let global = match (module, name) {
                    (b"_codecs", b"encode") => Global::Encode,
                    (b"_io", b"BytesIO") => Global::BytesIo,
                    _ => {
                        return Err(ErrorKind::Global {
                            module: String::from_utf8_lossy(module).into_owned(),
                            name: String::from_utf8_lossy(name).into_owned(),
                        });
                    }
                };
                self.stack.push(Value::Global(global));
            }
            REDUCE => {
                let args = self.pop()?;
                let callable = self.pop()?;
                let Value::Global(Global::Encode) = callable else {
                    return Err(invalid(format!("REDUCE of {}", describe(&callable))));
                };
                let Value::Tuple(args) = args else {
                    return Err(invalid(format!("REDUCE with args {}", describe(&args))));
                };
                let [Value::Str(string), Value::Str(encoding)] = &args[..] else {
                    return Err(invalid("_codecs.encode of other than two strs"));
                };
                if encoding != "latin1" {
                    return Err(invalid(format!("_codecs.encode to {encoding:?}")));
                }
                let bytes = string
                    .chars()
                    .map(u8::try_from)
                    .collect::<Result<_, _>>()
                    .map_err(|_| invalid("_codecs.encode of a str beyond latin1"))?;
                self.stack.push(Value::Bytes(bytes));
            }
            NEWOBJ => {
                let args = self.pop()?;
                let class = self.pop()?;
                if class != Value::Global(Global::BytesIo) || args != Value::Tuple(Vec::new()) {
                    let reason = format!("NEWOBJ of {} with {}", describe(&class), describe(&args));
                    return Err(invalid(reason));
                }
                self.stack.push(Value::EmptyBytesIo);
            }
            BUILD => {
                let state = self.pop()?;
                let Some(object) = self.top() else {
                    return Err(invalid("BUILD with nothing to build"));
                };
                if *object != Value::EmptyBytesIo {
                    return Err(invalid(format!("BUILD of {}", describe(object))));
                }
                let Value::Tuple(mut state) = state else {
                    return Err(invalid(format!("BytesIO state {}", describe(&state))));
                };
                let [Value::Bytes(bytes), Value::Int(0), Value::None] = &mut state[..] else {
                    return Err(invalid("BytesIO state other than (bytes, 0, None)"));
                };
                *object = Value::BytesIo(mem::take(bytes));
            }
            BINPUT | LONG_BINPUT => {
                let index = if opcode == BINPUT {
                    self.take(1)?[0].into()
                } else {
                    u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"))
                };
                let Some(value) = self.top() else {
                    return Err(invalid("BINPUT with nothing on the stack"));
                };
                let entry = if is_mutable(value) {
                    Memo::Mutable
                } else {
                    Memo::Value(value.clone())
                };
                self.memo.insert(index, entry);
            }
            BINGET => {
                let index = self.take(1)?[0].into();
                let value = match self.memo.get(&index) {
                    Some(Memo::Value(value)) => value.clone(),
                    Some(Memo::Mutable) => {
                        return Err(invalid(format!("BINGET {index} of a dict or BytesIO")));
                    }
                    None => return Err(invalid(format!("BINGET {index}, which was never put"))),
                };
                self.stack.push(value);
            }
            _ => return Err(ErrorKind::Opcode(opcode)),
        }
        Ok(None)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ErrorKind> {
        let end = self.pos.checked_add(len).ok_or(ErrorKind::Truncated)?;
        let bytes = self.pickle.get(self.pos..end).ok_or(ErrorKind::Truncated)?;
        self.pos = end;
        Ok(bytes)
    }

    fn line(&mut self) -> Result<&'a [u8], ErrorKind> {
        let rest = &self.pickle[self.pos..];
        let len = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(ErrorKind::Truncated)?;
        self.pos += len + 1;
        Ok(&rest[..len])
    }

    /// How many values are on the stack above the last MARK: all an opcode but SETITEMS may
    /// take.
    fn frame_len(&self) -> usize {
        self.stack.len() - self.marks.last().copied().unwrap_or(0)
    }

    fn pop(&mut self) -> Result<Value, ErrorKind> {
        if self.frame_len() == 0 {
            return Err(invalid("stack underflow"));
        }
        Ok(self.stack.pop().expect("frame is not empty"))
    }

    fn top(&mut self) -> Option<&mut Value> {
        if self.frame_len() == 0 {
            return None;
        }
        self.stack.last_mut()
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>, ErrorKind> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| invalid("no MARK to pop to"))?;
        Ok(self.stack.split_off(mark))
    }
}

fn is_mutable(value: &Value) -> bool {
    match value {
        Value::Dict(_) | Value::BytesIo(_) | Value::EmptyBytesIo => true,
        Value::Tuple(items) => items.iter().any(is_mutable),
        _ => false,
    }
}

/// What a value is, for an error.
pub(crate) fn describe(value: &Value) -> &'static str {
    match value {
        Value::None => "None",
        Value::Int(_) => "an int",
        Value::Str(_) => "a str",
        Value::Bytes(_) => "a bytes",
        Value::Tuple(_) => "a tuple",
        Value::Dict(_) => "a dict",
        Value::BytesIo(_) | Value::EmptyBytesIo => "a BytesIO",
        Value::Global(Global::Encode) => "_codecs.encode",
        Value::Global(Global::BytesIo) => "_io.BytesIO",
    }
}
