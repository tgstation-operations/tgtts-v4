//! A SPIR-V blob as the words Vulkan takes, with its header checked before the driver sees
//! it: a blob read from a file or embedded with `include_bytes!` is bytes at any alignment,
//! and a truncated or foreign file should fail here with a reason, not inside the driver.
//! Also the workgroup memory a kernel takes once specialized ([`workgroup_memory`]), which a
//! pipeline must not ask past the device's limit: RADV's compiler divides by zero over it
//! rather than refuse it.

use std::{collections::HashMap, ffi::CStr};

use crate::{
    error::{Error, Result},
    kernel::SpecValue,
};

/// The SPIR-V magic number, as the first word of a module in the host's byte order.
pub const MAGIC: u32 = 0x0723_0203;

/// The newest SPIR-V a Vulkan 1.3 device must accept.
pub const MAX_VERSION: (u8, u8) = (1, 6);

/// What the five-word header of a SPIR-V module says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// SPIR-V version, (major, minor).
    pub version: (u8, u8),
    /// The tool that wrote the module, as registered with Khronos: the high 16 bits are
    /// the tool's id, the low 16 its own version.
    pub generator: u32,
    /// One more than the largest result id in the module.
    pub bound: u32,
}

/// The module's words in host byte order, and its header. A module written in the other
/// byte order (the magic number reads swapped) is swapped word by word, as the SPIR-V
/// specification allows a consumer to do.
pub fn words(bytes: &[u8]) -> Result<(Vec<u32>, Header)> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Spirv(format!(
            "{} bytes is not a whole number of words",
            bytes.len()
        )));
    }
    if bytes.len() < 20 {
        return Err(Error::Spirv(format!(
            "{} bytes is shorter than the 5-word header",
            bytes.len()
        )));
    }
    let mut words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_ne_bytes(*w))
        .collect();
    if words[0] == MAGIC.swap_bytes() {
        for w in &mut words {
            *w = w.swap_bytes();
        }
    } else if words[0] != MAGIC {
        return Err(Error::Spirv(format!("magic number {:#010x}", words[0])));
    }
    let v = words[1];
    let version = ((v >> 16) as u8, (v >> 8) as u8);
    if v & 0xff00_00ff != 0 || version.0 == 0 {
        return Err(Error::Spirv(format!("version word {v:#010x}")));
    }
    if version > MAX_VERSION {
        return Err(Error::Unsupported(format!(
            "SPIR-V {}.{}; Vulkan 1.3 accepts up to {}.{}",
            version.0, version.1, MAX_VERSION.0, MAX_VERSION.1
        )));
    }
    let header = Header {
        version,
        generator: words[2],
        bound: words[3],
    };
    Ok((words, header))
}

/// The bytes of workgroup memory entry point `entry` of the module `words` takes once
/// `constants`, (SpecId, value) pairs, specialize it: the most Vulkan lets an implementation
/// take for it (the specification's `workgroup-padding`, which
/// VUID-RuntimeSpirv-Workgroup-06530 holds to `maxComputeSharedMemorySize`). That is the
/// largest the entry point's variables in the Workgroup storage class come to laid out by the
/// standard storage buffer layout (std430) in any order, each at the smallest offset its
/// alignment allows after those before it, booleans as 32-bit integers, with nothing counted
/// after the last.
///
/// The variables are those the entry point's interface lists, which from SPIR-V 1.4 on is every
/// global it uses; every Workgroup variable of an older module. Explicitly laid out (Block)
/// workgroup memory, and array lengths from specialization constant operations other than
/// integer and logical arithmetic, are refused rather than guessed at.
pub fn workgroup_memory(
    words: &[u32],
    entry: &CStr,
    constants: &[(u32, SpecValue)],
) -> Result<u64> {
    let module = Module::parse(words)?;
    let mut active: Vec<u32> = Vec::new();
    let mut found = false;
    for (name, interface) in &module.entry_points {
        if name.as_slice() == entry.to_bytes() {
            found = true;
            active = interface.clone();
        }
    }
    if !found {
        return Err(Error::Spirv(format!("no entry point {entry:?}")));
    }
    let per_interface = words[1] >= 0x0001_0400;
    let mut eval = Eval {
        module: &module,
        constants,
        values: HashMap::new(),
    };
    let mut variables = Vec::new();
    for &(id, pointee) in &module.workgroup {
        if per_interface && !active.contains(&id) {
            continue;
        }
        if module.blocks.contains(&pointee) {
            return Err(Error::Unsupported(format!(
                "explicitly laid out workgroup memory (%{id})"
            )));
        }
        variables.push(eval.layout(pointee, 0)?);
    }
    Ok(most_padded(&variables))
}

/// What sizing workgroup memory reads of a module.
#[derive(Default)]
struct Module {
    /// Each entry point's name and interface.
    entry_points: Vec<(Vec<u8>, Vec<u32>)>,
    types: HashMap<u32, Type>,
    constants: HashMap<u32, Constant>,
    spec_ids: HashMap<u32, u32>,
    /// Types decorated Block.
    blocks: Vec<u32>,
    /// Each Workgroup variable, and the type it points to.
    workgroup: Vec<(u32, u32)>,
}

enum Type {
    Bool,
    /// Integer or float, by its width in bits.
    Scalar(u32),
    Vector(u32, u32),
    Matrix(u32, u32),
    /// An element type and the id of its length.
    Array(u32, u32),
    Struct(Vec<u32>),
    Other,
}

/// A constant's type, and its value.
struct Constant {
    ty: u32,
    value: Value,
}

enum Value {
    Literal(u64),
    /// A default the SpecId the module gives it may override.
    Spec(u64),
    /// An operation, and its operands.
    Op(u32, Vec<u32>),
}

const STORAGE_CLASS_WORKGROUP: u32 = 4;

impl Module {
    fn parse(words: &[u32]) -> Result<Self> {
        let mut m = Module::default();
        let mut pointers = HashMap::new();
        let mut variables = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let opcode = words[at] & 0xffff;
            if count == 0 || at + count > words.len() {
                return Err(Error::Spirv(format!(
                    "an instruction of {count} words at word {at}"
                )));
            }
            let ops = &words[at + 1..at + count];
            let operand = |i: usize| {
                ops.get(i).copied().ok_or_else(|| {
                    Error::Spirv(format!("opcode {opcode} at word {at} without operand {i}"))
                })
            };
            match opcode {
                // OpEntryPoint: model, function, name, interface.
                15 => {
                    let name: Vec<u8> = ops[2..]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .take_while(|&b| b != 0)
                        .collect();
                    let interface = ops[2 + (name.len() + 4) / 4..].to_vec();
                    m.entry_points.push((name, interface));
                }
                // OpDecorate: SpecId, Block.
                71 => match operand(1)? {
                    1 => {
                        m.spec_ids.insert(operand(0)?, operand(2)?);
                    }
                    2 => m.blocks.push(operand(0)?),
                    _ => {}
                },
                20 => {
                    m.types.insert(operand(0)?, Type::Bool);
                }
                21 | 22 => {
                    m.types.insert(operand(0)?, Type::Scalar(operand(1)?));
                }
                23 => {
                    m.types
                        .insert(operand(0)?, Type::Vector(operand(1)?, operand(2)?));
                }
                24 => {
                    m.types
                        .insert(operand(0)?, Type::Matrix(operand(1)?, operand(2)?));
                }
                28 => {
                    m.types
                        .insert(operand(0)?, Type::Array(operand(1)?, operand(2)?));
                }
                30 => {
                    m.types.insert(operand(0)?, Type::Struct(ops[1..].to_vec()));
                }
                32 => {
                    pointers.insert(operand(0)?, operand(2)?);
                }
                // The rest of the types, among them OpTypeRuntimeArray.
                19 | 25..=27 | 29 | 31 | 33..=39 => {
                    m.types.insert(operand(0)?, Type::Other);
                }
                // OpConstantTrue, OpConstantFalse, OpConstant; OpSpecConstantTrue,
                // OpSpecConstantFalse, OpSpecConstant; OpSpecConstantOp.
                41..=43 | 48..=50 | 52 => {
                    let value = match opcode {
                        41 | 42 => Value::Literal(u64::from(opcode == 41)),
                        43 => Value::Literal(literal(&ops[2..])),
                        48 | 49 => Value::Spec(u64::from(opcode == 48)),
                        50 => Value::Spec(literal(&ops[2..])),
                        _ => Value::Op(operand(2)?, ops[3..].to_vec()),
                    };
                    let ty = operand(0)?;
                    m.constants.insert(operand(1)?, Constant { ty, value });
                }
                // OpVariable: its pointer type, its id, its storage class.
                59 if operand(2)? == STORAGE_CLASS_WORKGROUP => {
                    variables.push((operand(1)?, operand(0)?));
                }
                _ => {}
            }
            at += count;
        }
        for (id, pointer) in variables {
            let Some(&pointee) = pointers.get(&pointer) else {
                return Err(Error::Spirv(format!("%{id} of no pointer type %{pointer}")));
            };
            m.workgroup.push((id, pointee));
        }
        Ok(m)
    }
}

/// A literal number of one word or two, low word first.
fn literal(words: &[u32]) -> u64 {
    let low = u64::from(words.first().copied().unwrap_or(0));
    let high = u64::from(words.get(1).copied().unwrap_or(0));
    low | high << 32
}

/// Specialization constants evaluated as a module's types need them.
struct Eval<'a> {
    module: &'a Module,
    constants: &'a [(u32, SpecValue)],
    values: HashMap<u32, u64>,
}

/// How deep types and constants may nest before the module is taken for a cyclic one.
const DEPTH: usize = 64;

impl Eval<'_> {
    /// The bits of constant `id`, as wide as its type: a boolean is 0 or 1.
    fn value(&mut self, id: u32, depth: usize) -> Result<u64> {
        if let Some(&v) = self.values.get(&id) {
            return Ok(v);
        }
        if depth > DEPTH {
            return Err(Error::Spirv(format!(
                "constants nested past {DEPTH} at %{id}"
            )));
        }
        let module = self.module;
        let Some(constant) = module.constants.get(&id) else {
            return Err(Error::Spirv(format!("%{id} is not a constant")));
        };
        let v = match &constant.value {
            Value::Literal(v) => *v,
            Value::Spec(default) => {
                let spec_id = self.module.spec_ids.get(&id);
                let given = self.constants.iter().find(|(c, _)| Some(c) == spec_id);
                match given {
                    Some((_, SpecValue::Bool(b))) => u64::from(*b),
                    Some((_, SpecValue::U32(v))) => u64::from(*v),
                    Some((_, SpecValue::I32(v))) => u64::from(*v as u32),
                    Some((_, SpecValue::F32(v))) => u64::from(v.to_bits()),
                    None => *default,
                }
            }
            Value::Op(op, operands) => {
                let mut args = Vec::with_capacity(operands.len());
                for &a in operands {
                    let ty = module.constants.get(&a).map_or(0, |c| c.ty);
                    args.push((self.value(a, depth + 1)?, self.width(ty, id)?));
                }
                spec_op(*op, &args, self.width(constant.ty, id)?).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "specialization constant operation {op} of %{id} on {args:?}"
                    ))
                })?
            }
        };
        self.values.insert(id, v);
        Ok(v)
    }

    /// The width in bits of scalar type `ty`, 1 for a boolean, of constant `of`.
    fn width(&self, ty: u32, of: u32) -> Result<u32> {
        match self.module.types.get(&ty) {
            Some(Type::Scalar(w)) => Ok(*w),
            Some(Type::Bool) => Ok(1),
            _ => Err(Error::Unsupported(format!(
                "specialization constant %{of} over a type other than a scalar"
            ))),
        }
    }

    /// The size and alignment of type `id` in the standard storage buffer layout.
    fn layout(&mut self, id: u32, depth: usize) -> Result<(u64, u64)> {
        if depth > DEPTH {
            return Err(Error::Spirv(format!("types nested past {DEPTH} at %{id}")));
        }
        let module = self.module;
        let unsupported = || Error::Unsupported(format!("workgroup memory of type %{id}"));
        Ok(match module.types.get(&id) {
            Some(Type::Bool) => (4, 4),
            Some(Type::Scalar(width)) => {
                let bytes = u64::from(*width / 8);
                (bytes, bytes)
            }
            Some(Type::Vector(element, count)) => {
                let (size, align) = self.layout(*element, depth + 1)?;
                let count = u64::from(*count);
                let align = match count {
                    2 => 2 * align,
                    3 | 4 => 4 * align,
                    _ => return Err(unsupported()),
                };
                (size * count, align)
            }
            Some(Type::Matrix(column, count)) => {
                let (size, align) = self.layout(*column, depth + 1)?;
                (size.next_multiple_of(align) * u64::from(*count), align)
            }
            Some(Type::Array(element, length)) => {
                let (size, align) = self.layout(*element, depth + 1)?;
                let length = self.value(*length, depth + 1)?;
                let bytes = size.next_multiple_of(align).checked_mul(length);
                (bytes.ok_or_else(unsupported)?, align)
            }
            Some(Type::Struct(members)) => {
                let (mut end, mut most) = (0u64, 1u64);
                for &member in members {
                    let (size, align) = self.layout(member, depth + 1)?;
                    end = end.next_multiple_of(align) + size;
                    most = most.max(align);
                }
                (end.next_multiple_of(most), most)
            }
            _ => return Err(unsupported()),
        })
    }
}

/// The value of OpSpecConstantOp's integer and logical operation `op` on `args`, each a value
/// and its width in bits (1 for a boolean), the result `width` bits wide; none for any other
/// operation.
fn spec_op(op: u32, args: &[(u64, u32)], width: u32) -> Option<u64> {
    let bits = |v: u64| {
        if width >= 64 {
            v
        } else {
            v & ((1 << width) - 1)
        }
    };
    let (a, a_width) = *args.first()?;
    let b = args.get(1).map_or(0, |&(b, _)| b);
    // An operand as signed, from its width, which an operation's integer operands share.
    let signed = |v: u64| {
        let shift = 64 - a_width.clamp(1, 64);
        ((v << shift) as i64) >> shift
    };
    let truth = |c: bool| Some(u64::from(c));
    let v = match op {
        // OpSConvert, OpUConvert: integers of one width, here.
        113 | 114 => a,
        126 => signed(a).wrapping_neg() as u64,
        128 => a.wrapping_add(b),
        130 => a.wrapping_sub(b),
        132 => a.wrapping_mul(b),
        134 => a.checked_div(b)?,
        135 => signed(a).checked_div(signed(b))? as u64,
        137 => a.checked_rem(b)?,
        138 => signed(a).checked_rem(signed(b))? as u64,
        139 => signed(a).checked_rem_euclid(signed(b))? as u64,
        // OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpLogicalNot.
        164 => return truth(a == b),
        165 => return truth(a != b),
        166 => return truth(a != 0 || b != 0),
        167 => return truth(a != 0 && b != 0),
        168 => return truth(a == 0),
        // OpSelect: condition, then, else.
        169 => {
            if a != 0 {
                b
            } else {
                args.get(2)?.0
            }
        }
        170 => return truth(a == b),
        171 => return truth(a != b),
        172 => return truth(a > b),
        173 => return truth(signed(a) > signed(b)),
        174 => return truth(a >= b),
        175 => return truth(signed(a) >= signed(b)),
        176 => return truth(a < b),
        177 => return truth(signed(a) < signed(b)),
        178 => return truth(a <= b),
        179 => return truth(signed(a) <= signed(b)),
        194 => a.checked_shr(b.try_into().ok()?)?,
        195 => signed(a).checked_shr(b.try_into().ok()?)? as u64,
        196 => a.checked_shl(b.try_into().ok()?)?,
        197 => a | b,
        198 => a ^ b,
        199 => a & b,
        200 => !a,
        _ => return None,
    };
    Some(bits(v))
}

/// The end of the last of `variables`, (size, alignment) each, placed in the order that pads
/// them most, each at the smallest offset its alignment allows after those before it. Past
/// eight variables, a bound on it: each placed at a multiple of the largest alignment.
fn most_padded(variables: &[(u64, u64)]) -> u64 {
    fn end(variables: &[(u64, u64)], left: &mut Vec<usize>, offset: u64) -> u64 {
        let mut most = offset;
        for k in 0..left.len() {
            let i = left.swap_remove(k);
            let (size, align) = variables[i];
            most = most.max(end(variables, left, offset.next_multiple_of(align) + size));
            left.push(i);
            let last = left.len() - 1;
            left.swap(k, last);
        }
        most
    }
    if variables.len() > 8 {
        let align = variables.iter().map(|&(_, a)| a).max().unwrap_or(1);
        return variables
            .iter()
            .map(|&(s, _)| s.next_multiple_of(align))
            .sum();
    }
    end(variables, &mut (0..variables.len()).collect(), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header-only module: magic, version 1.6, generator 7 (Khronos' assembler), bound 42,
    /// schema 0.
    fn module(version: u32) -> Vec<u8> {
        [MAGIC, version, 7 << 16, 42, 0]
            .iter()
            .flat_map(|w| w.to_ne_bytes())
            .collect()
    }

    #[test]
    fn a_header_is_read_into_host_order_words() {
        let (words, header) = words(&module(0x0001_0600)).unwrap();
        assert_eq!(words.len(), 5);
        assert_eq!(
            header,
            Header {
                version: (1, 6),
                generator: 7 << 16,
                bound: 42
            }
        );
    }

    #[test]
    fn a_module_in_the_other_byte_order_is_swapped() {
        let swapped: Vec<u8> = module(0x0001_0300)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[a, b, c, d]| [d, c, b, a])
            .collect();
        let (words, header) = words(&swapped).unwrap();
        assert_eq!(words[0], MAGIC);
        assert_eq!(header.version, (1, 3));
        assert_eq!(header.bound, 42);
    }

    #[test]
    fn malformed_blobs_are_refused_with_a_reason() {
        let short = &module(0x0001_0600)[..16];
        assert!(matches!(words(short), Err(Error::Spirv(why)) if why.contains("shorter")));
        let ragged = &module(0x0001_0600)[..19];
        assert!(matches!(words(ragged), Err(Error::Spirv(why)) if why.contains("whole number")));
        let mut foreign = module(0x0001_0600);
        foreign[0] ^= 1;
        assert!(matches!(words(&foreign), Err(Error::Spirv(why)) if why.contains("magic")));
        assert!(
            matches!(words(&module(0x0001_0601)), Err(Error::Spirv(why)) if why.contains("version"))
        );
    }

    #[test]
    fn a_module_newer_than_vulkan_1_3_accepts_is_unsupported() {
        assert!(matches!(
            words(&module(0x0001_0700)),
            Err(Error::Unsupported(_))
        ));
        assert!(words(&module(0x0001_0000)).is_ok());
    }

    /// The words of `tests/kernels/shared.spv`: `len` floats, a flag, and `2 rows` vec4 tiles
    /// when `wide`, one otherwise.
    fn shared() -> Vec<u32> {
        words(include_bytes!("../tests/kernels/shared.spv"))
            .unwrap()
            .0
    }

    #[test]
    fn workgroup_memory_is_its_variables_laid_out_in_the_order_that_pads_them_most() {
        let at = |constants: &[(u32, SpecValue)]| {
            workgroup_memory(&shared(), c"main", constants).unwrap()
        };
        // 16 floats and the flag, 68 bytes, before 16 tiles at the next multiple of 16.
        assert_eq!(at(&[]), 80 + 16 * 16);
        // 17 floats before the tiles end at 68 + 12 bytes of padding, and the flag after them.
        assert_eq!(at(&[(0, SpecValue::U32(17))]), 80 + 16 * 16 + 4);
        assert_eq!(at(&[(1, SpecValue::U32(4096))]), 80 + 8192 * 16);
        assert_eq!(at(&[(2, SpecValue::Bool(false))]), 80 + 16);
        assert!(matches!(
            workgroup_memory(&shared(), c"other", &[]),
            Err(Error::Spirv(why)) if why.contains("entry point")
        ));
    }

    #[test]
    fn the_padding_is_the_most_any_order_leaves() {
        assert_eq!(most_padded(&[]), 0);
        // Three floats then a vec4 leave four bytes before it; the vec4 first leaves none.
        assert_eq!(most_padded(&[(16, 16), (12, 4)]), 32);
        assert_eq!(most_padded(&[(4, 4), (16, 16), (4, 4)]), 36);
        // Past eight variables, each at a multiple of the largest alignment.
        let many = [(4, 4); 9];
        assert_eq!(most_padded(&many), 36);
        let mixed = [
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (16, 16),
        ];
        assert_eq!(most_padded(&mixed), 9 * 16);
    }

    #[test]
    fn specialization_constant_operations_are_integer_and_logical_arithmetic() {
        let word = |v: u64| (v, 32);
        assert_eq!(spec_op(134, &[word(2048), word(8)], 32), Some(256));
        assert_eq!(spec_op(130, &[word(0), word(1)], 32), Some(0xffff_ffff));
        // -1 is less than 0 signed, and greater unsigned.
        let minus_one = word(0xffff_ffff);
        assert_eq!(spec_op(177, &[minus_one, word(0)], 1), Some(1));
        assert_eq!(spec_op(176, &[minus_one, word(0)], 1), Some(0));
        assert_eq!(spec_op(135, &[minus_one, word(1)], 32), Some(0xffff_ffff));
        assert_eq!(spec_op(169, &[(0, 1), word(7), word(9)], 32), Some(9));
        assert_eq!(spec_op(134, &[word(1), word(0)], 32), None);
        // OpFMul, which a length could not come from.
        assert_eq!(spec_op(133, &[word(1), word(1)], 32), None);
    }

    #[test]
    fn the_committed_test_kernel_has_the_header_its_source_declares() {
        let (_, header) = words(include_bytes!("../tests/kernels/axpb.spv")).unwrap();
        assert_eq!(header.version, (1, 6));
        assert_eq!(header.bound, 42);
    }
}
