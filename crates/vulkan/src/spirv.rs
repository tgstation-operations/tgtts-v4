//! A SPIR-V blob as the words Vulkan takes, with its header checked before the driver sees
//! it: a blob read from a file or embedded with `include_bytes!` is bytes at any alignment,
//! and a truncated or foreign file should fail here with a reason, not inside the driver.

use crate::error::{Error, Result};

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

    #[test]
    fn the_committed_test_kernel_has_the_header_its_source_declares() {
        let (_, header) = words(include_bytes!("../tests/kernels/axpb.spv")).unwrap();
        assert_eq!(header.version, (1, 6));
        assert_eq!(header.bound, 42);
    }
}
