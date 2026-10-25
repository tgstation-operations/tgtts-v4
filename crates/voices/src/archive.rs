//! The zip archives torch.save writes, as far as reading their records goes.
//!
//! torch stores every record uncompressed, so this reads nothing else: a compressed or
//! encrypted record, or an archive spanning disks, is refused. Each record is found through the
//! central directory and checked against its CRC-32.

const END: u32 = 0x0605_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
const END_LEN: usize = 22;
const CENTRAL_LEN: usize = 46;
const LOCAL_LEN: usize = 30;
/// The sizes follow the data, and the name is UTF-8: torch sets both.
const ALLOWED_FLAGS: u16 = 0x0808;

/// The archive's records, by name, in the central directory's order.
pub(crate) fn records(archive: &[u8]) -> Result<Vec<(&str, &[u8])>, String> {
    let end = archive
        .len()
        .checked_sub(END_LEN)
        .map(|at| &archive[at..])
        .filter(|end| u32_at(end, 0) == END && u16_at(end, 20) == 0)
        .ok_or("does not end in an end of central directory record without a comment")?;
    let entries = u16_at(end, 10);
    if u16_at(end, 4) != 0 || u16_at(end, 6) != 0 || u16_at(end, 8) != entries {
        return Err("spans more than one disk".to_owned());
    }
    let (size, offset) = (u32_at(end, 12) as usize, u32_at(end, 16) as usize);
    let directory = archive
        .get(offset..offset + size)
        .ok_or("central directory lies outside the archive")?;

    let mut records = Vec::with_capacity(entries.into());
    let mut at = 0;
    for _ in 0..entries {
        let entry = directory
            .get(at..at + CENTRAL_LEN)
            .filter(|entry| u32_at(entry, 0) == CENTRAL)
            .ok_or("central directory ends before its entries do")?;
        let name_len = u16_at(entry, 28) as usize;
        let rest_len = name_len + u16_at(entry, 30) as usize + u16_at(entry, 32) as usize;
        let name = directory
            .get(at + CENTRAL_LEN..at + CENTRAL_LEN + name_len)
            .ok_or("central directory ends inside a name")?;
        let name = str::from_utf8(name).map_err(|_| "a record's name is not UTF-8")?;
        at += CENTRAL_LEN + rest_len;

        let (flags, method) = (u16_at(entry, 8), u16_at(entry, 10));
        if flags & !ALLOWED_FLAGS != 0 {
            return Err(format!("record {name:?} has flags {flags:#06x}"));
        }
        if method != 0 {
            return Err(format!("record {name:?} is compressed (method {method})"));
        }
        let (crc, size) = (u32_at(entry, 16), u32_at(entry, 20) as usize);
        if u32_at(entry, 24) as usize != size {
            return Err(format!("record {name:?} is stored with two sizes"));
        }

        let local_at = u32_at(entry, 42) as usize;
        let local = archive
            .get(local_at..local_at + LOCAL_LEN)
            .filter(|local| u32_at(local, 0) == LOCAL)
            .ok_or_else(|| format!("record {name:?} has no local header"))?;
        let name_at = local_at + LOCAL_LEN;
        let local_name = archive.get(name_at..name_at + u16_at(local, 26) as usize);
        if u16_at(local, 8) != method || local_name != Some(name.as_bytes()) {
            return Err(format!("record {name:?} has a local header that disagrees"));
        }
        let data_at = name_at + name_len + u16_at(local, 28) as usize;
        let data = archive
            .get(data_at..data_at + size)
            .ok_or_else(|| format!("record {name:?} runs past the end of the archive"))?;
        if crc32fast::hash(data) != crc {
            return Err(format!("record {name:?} fails its CRC-32"));
        }
        if records.iter().any(|&(other, _)| other == name) {
            return Err(format!("record {name:?} appears twice"));
        }
        records.push((name, data));
    }
    if at != directory.len() {
        return Err("central directory runs on after its entries".to_owned());
    }
    Ok(records)
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}
