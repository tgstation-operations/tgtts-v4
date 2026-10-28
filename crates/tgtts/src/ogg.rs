//! Ogg pages, as RFC 3533 frames the packets of a logical bitstream: each page a header, a
//! table of up to 255 lacing values and the bytes they measure, the granule position of the last
//! packet that ends on it, and a CRC over all of it.

use std::{error, fmt};

/// A page that goes on with a packet the page before left unfinished.
const CONTINUED: u8 = 1;
/// The stream's first page.
const FIRST: u8 = 2;
/// The stream's last page.
const LAST: u8 = 4;
/// The lacing values a page holds at most.
const LACING: usize = 255;
/// The granule position of a page on which no packet ends.
const NO_GRANULE: u64 = u64::MAX;
const HEADER: usize = 27;

/// A logical bitstream, written a run of packets at a time.
#[derive(Clone, Debug)]
pub struct Writer {
    serial: u32,
    sequence: u32,
}

impl Writer {
    /// A bitstream of serial number `serial`.
    pub fn new(serial: u32) -> Self {
        Self {
            serial,
            sequence: 0,
        }
    }

    /// Appends to `out` the pages that hold `packets`, each with the granule position at its
    /// end, on as few pages as their lacing fits, the last ending with the last packet so that
    /// none waits for more. The first page begins the stream if `first`, and the last ends it if
    /// `last`. Nothing is written for no packets.
    pub fn write(&mut self, packets: &[(&[u8], u64)], first: bool, last: bool, out: &mut Vec<u8>) {
        // Each lacing value, with the granule position of the packet it ends, if it does.
        let mut lacing: Vec<(u8, Option<u64>)> = Vec::new();
        for &(packet, granule) in packets {
            lacing.extend(std::iter::repeat_n((255, None), packet.len() / 255));
            lacing.push(((packet.len() % 255) as u8, Some(granule)));
        }
        let mut body = packets
            .iter()
            .flat_map(|(packet, _)| packet.iter().copied());
        let pages = lacing.len().div_ceil(LACING);
        let mut continued = false;
        for (i, values) in lacing.chunks(LACING).enumerate() {
            let mut flags = 0;
            if continued {
                flags |= CONTINUED;
            }
            if first && i == 0 {
                flags |= FIRST;
            }
            if last && i == pages - 1 {
                flags |= LAST;
            }
            let granule = values.iter().rev().find_map(|&(_, g)| g);
            let size: usize = values.iter().map(|&(v, _)| usize::from(v)).sum();
            let start = out.len();
            out.extend_from_slice(b"OggS");
            out.push(0);
            out.push(flags);
            out.extend_from_slice(&granule.unwrap_or(NO_GRANULE).to_le_bytes());
            out.extend_from_slice(&self.serial.to_le_bytes());
            out.extend_from_slice(&self.sequence.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            out.push(values.len() as u8);
            out.extend(values.iter().map(|&(v, _)| v));
            out.extend(body.by_ref().take(size));
            let crc = crc(&out[start..]);
            out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
            self.sequence += 1;
            continued = values.last().is_some_and(|&(_, g)| g.is_none());
        }
    }
}

/// A page read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// Whether it goes on with a packet the page before left unfinished, begins the stream and
    /// ends it.
    pub continued: bool,
    pub first: bool,
    pub last: bool,
    /// The granule position, `None` where no packet ends on it.
    pub granule: Option<u64>,
    pub serial: u32,
    pub sequence: u32,
    pub lacing: Vec<u8>,
    pub body: Vec<u8>,
}

/// The pages of `data`, which holds whole pages and nothing else, each checked against its CRC.
pub fn pages(mut data: &[u8]) -> Result<Vec<Page>, Error> {
    let mut pages = Vec::new();
    while !data.is_empty() {
        let at = pages.len();
        let malformed = |what: &str| Error(format!("page {at}: {what}"));
        if data.len() < HEADER || &data[..4] != b"OggS" || data[4] != 0 {
            return Err(malformed("no page header"));
        }
        let count = usize::from(data[26]);
        let lacing = data
            .get(HEADER..HEADER + count)
            .ok_or(malformed("cut short"))?;
        let size: usize = lacing.iter().map(|&v| usize::from(v)).sum();
        let end = HEADER + count + size;
        let page = data.get(..end).ok_or(malformed("cut short"))?;
        let mut zeroed = page.to_vec();
        zeroed[22..26].fill(0);
        if crc(&zeroed).to_le_bytes() != page[22..26] {
            return Err(malformed("CRC"));
        }
        let word = |at: usize| u32::from_le_bytes(page[at..at + 4].try_into().expect("4 bytes"));
        let granule = u64::from_le_bytes(page[6..14].try_into().expect("8 bytes"));
        pages.push(Page {
            continued: page[5] & CONTINUED != 0,
            first: page[5] & FIRST != 0,
            last: page[5] & LAST != 0,
            granule: (granule != NO_GRANULE).then_some(granule),
            serial: word(14),
            sequence: word(18),
            lacing: lacing.to_vec(),
            body: page[HEADER + count..].to_vec(),
        });
        data = &data[end..];
    }
    Ok(pages)
}

/// The packets `pages` hold, in order, a packet that the last page leaves unfinished left out.
pub fn packets(pages: &[Page]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut packet = Vec::new();
    for page in pages {
        let mut body = page.body.as_slice();
        for &value in &page.lacing {
            let (bytes, rest) = body.split_at(usize::from(value));
            packet.extend_from_slice(bytes);
            body = rest;
            if value < 255 {
                packets.push(std::mem::take(&mut packet));
            }
        }
    }
    packets
}

/// Why bytes are not Ogg pages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ogg: {}", self.0)
    }
}

impl error::Error for Error {}

/// The CRC-32 of RFC 3533: polynomial 0x04c11db7, unreflected, from 0, with no final XOR.
fn crc(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0, |crc, &byte| {
        (crc << 8) ^ CRC_TABLE[usize::from((crc >> 24) as u8 ^ byte)]
    })
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04c1_1db7
            } else {
                r << 1
            };
            bit += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The CRC of the check string, as the CRC catalogue gives it for CRC-32/OGG.
    #[test]
    fn crc_is_ogg_s() {
        assert_eq!(crc(b"123456789"), 0x89a1_897f);
    }

    #[test]
    fn packets_come_back_as_written() {
        let sizes = [0, 1, 70000, 3, 254, 255, 256, 510];
        let written: Vec<Vec<u8>> = (sizes.iter().enumerate())
            .map(|(i, &n)| (0..n).map(|j| (i * 31 + j) as u8).collect())
            .collect();
        let with_granules: Vec<(&[u8], u64)> = (written.iter().enumerate())
            .map(|(i, p)| (p.as_slice(), 100 * i as u64))
            .collect();
        let mut writer = Writer::new(7);
        let mut out = Vec::new();
        writer.write(&with_granules[..2], true, false, &mut out);
        writer.write(&with_granules[2..], false, true, &mut out);
        let pages = pages(&out).unwrap();
        assert_eq!(packets(&pages), written);
        assert!(pages[0].first && !pages[0].last && pages.last().unwrap().last);
        let sequences: Vec<u32> = pages.iter().map(|p| p.sequence).collect();
        assert_eq!(sequences, (0..pages.len() as u32).collect::<Vec<_>>());
        assert!(
            pages
                .iter()
                .all(|p| p.serial == 7 && p.lacing.len() <= LACING)
        );
        // The 70000-byte packet fills a page of its own and ends on the next, which goes on with
        // it: the page it only passes through ends no packet.
        let spans: Vec<_> = pages.iter().map(|p| (p.continued, p.granule)).collect();
        assert_eq!(
            spans,
            [(false, Some(100)), (false, None), (true, Some(700))]
        );
    }

    #[test]
    fn refuses_a_page_whose_crc_is_wrong() {
        let mut out = Vec::new();
        Writer::new(1).write(&[(b"hello", 5)], true, true, &mut out);
        assert!(pages(&out).is_ok());
        out[30] ^= 1;
        assert_eq!(pages(&out), Err(Error("page 0: CRC".into())));
    }
}
