//! What the driver compiled a pipeline into, read back through
//! `VK_KHR_pipeline_executable_properties`: the subgroup size the code runs at, the
//! driver's statistics (registers, shared memory, code size, occupancy), and its internal
//! representations, the final machine code among them.
//!
//! This is the only witness of how a pipeline will run that does not depend on what it computes. A
//! kernel scheduled for 64-wide subgroups but compiled at 32 can still write the right numbers, two
//! 32-wide subgroups computing the same tile, so an output comparison cannot see the difference;
//! the subgroup size the driver reports can.
//!
//! The context enables the extension when the device offers it, and every pipeline is
//! then created asking the driver to keep statistics and internal representations. That
//! asks the driver to remember what it compiled, not to compile differently.

use std::{
    collections::BTreeMap,
    ffi::{CStr, c_char},
};

use ash::vk;

/// A statistic's value, in the format the driver gives it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
}

impl StatValue {
    /// The value as a number (a boolean as 0 or 1).
    pub fn as_f64(self) -> f64 {
        match self {
            StatValue::Bool(b) => f64::from(u8::from(b)),
            StatValue::I64(v) => v as f64,
            StatValue::U64(v) => v as f64,
            StatValue::F64(v) => v,
        }
    }
}

/// One statistic: RADV reports, among others, `SGPRs`, `VGPRs`, `Spilled SGPRs`,
/// `Spilled VGPRs`, `Code size`, `LDS size`, `Scratch size` and `Subgroups per SIMD`.
#[derive(Clone, Debug, PartialEq)]
pub struct Statistic {
    pub name: String,
    pub description: String,
    pub value: StatValue,
}

/// One internal representation: its name (RADV's include the NIR, the backend IR and the
/// final assembly) and, when the driver marks it as text, the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Representation {
    pub name: String,
    pub description: String,
    pub text: Option<String>,
    pub bytes: usize,
}

/// One executable of a pipeline: for a compute pipeline, its one shader as compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct Executable {
    pub name: String,
    pub description: String,
    /// The subgroup size the executable runs at.
    pub subgroup_size: u32,
    pub statistics: Vec<Statistic>,
    pub representations: Vec<Representation>,
}

impl Executable {
    /// The statistic named `name`, if the driver reports it.
    pub fn statistic(&self, name: &str) -> Option<StatValue> {
        self.statistics
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.value)
    }

    /// Every text representation, concatenated: where to look for an instruction
    /// whatever the driver names its stages.
    pub fn text(&self) -> String {
        self.representations
            .iter()
            .filter_map(|r| r.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A fixed-size, NUL-terminated `char` array from a Vulkan structure as a string.
pub(crate) fn fixed_string(chars: &[c_char]) -> String {
    // SAFETY: c_char and u8 have the same size and alignment, and every byte is a u8.
    let bytes = unsafe { std::slice::from_raw_parts(chars.as_ptr().cast::<u8>(), chars.len()) };
    CStr::from_bytes_until_nul(bytes)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// A statistic's value from its format and the union the driver filled.
pub(crate) fn stat_value(
    format: vk::PipelineExecutableStatisticFormatKHR,
    v: vk::PipelineExecutableStatisticValueKHR,
) -> StatValue {
    // SAFETY: the format names the union member the driver wrote.
    unsafe {
        match format {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => StatValue::Bool(v.b32 != 0),
            vk::PipelineExecutableStatisticFormatKHR::INT64 => StatValue::I64(v.i64),
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => StatValue::F64(v.f64),
            _ => StatValue::U64(v.u64),
        }
    }
}

/// How often each instruction whose mnemonic starts with `prefix` appears in `assembly`:
/// a mnemonic is a whitespace-separated word, so `v_wmma_f16_16x16x16_f16` in a line of
/// disassembly counts once, and the same word inside a comment counts too.
pub fn count_mnemonics(assembly: &str, prefix: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for word in assembly.split_whitespace() {
        let word = word.trim_end_matches([',', ';']);
        if word.starts_with(prefix) {
            *counts.entry(word.to_string()).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mnemonics_are_counted_by_word() {
        let asm = "\
            v_wmma_f16_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7] ; d0\n\
            s_waitcnt vmcnt(0)\n\
            v_wmma_f16_16x16x16_f16 v[24:31], v[8:15], v[32:39], v[24:31]\n\
            v_wmma_f32_16x16x16_f16, v[0:7]\n\
            ; not v_wmmaxyz\n";
        let counts = count_mnemonics(asm, "v_wmma_");
        assert_eq!(counts.len(), 2);
        assert_eq!(counts["v_wmma_f16_16x16x16_f16"], 2);
        assert_eq!(counts["v_wmma_f32_16x16x16_f16"], 1);
        assert!(count_mnemonics("s_endpgm", "v_wmma_").is_empty());
    }

    #[test]
    fn fixed_strings_stop_at_the_first_nul() {
        let mut chars = [0 as c_char; 8];
        for (c, b) in chars.iter_mut().zip(b"VGPRs") {
            *c = *b as c_char;
        }
        assert_eq!(fixed_string(&chars), "VGPRs");
        assert_eq!(fixed_string(&[0; 4]), "");
        // No terminator at all reads as empty rather than past the end.
        assert_eq!(fixed_string(&[b'a' as c_char; 3]), "");
    }

    #[test]
    fn statistics_read_the_member_their_format_names() {
        let v = vk::PipelineExecutableStatisticValueKHR { u64: 180 };
        assert_eq!(
            stat_value(vk::PipelineExecutableStatisticFormatKHR::UINT64, v),
            StatValue::U64(180)
        );
        let v = vk::PipelineExecutableStatisticValueKHR { b32: vk::TRUE };
        assert_eq!(
            stat_value(vk::PipelineExecutableStatisticFormatKHR::BOOL32, v),
            StatValue::Bool(true)
        );
        let v = vk::PipelineExecutableStatisticValueKHR { f64: 2.5 };
        assert_eq!(
            stat_value(vk::PipelineExecutableStatisticFormatKHR::FLOAT64, v).as_f64(),
            2.5
        );
        let v = vk::PipelineExecutableStatisticValueKHR { i64: -3 };
        assert_eq!(
            stat_value(vk::PipelineExecutableStatisticFormatKHR::INT64, v).as_f64(),
            -3.0
        );
    }
}
