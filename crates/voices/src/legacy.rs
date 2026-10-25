//! The torch.save files the Python service kept its voices in, read without unpickling them.
//!
//! Each is a zip archive of `<stem>/data.pkl`, a protocol 2 pickle, and the records torch writes
//! beside it. A `.speaker_latent` pickles `{"latents": bytes, "text": str}`, the latents f32
//! little-endian `[frames, 64]`. A `.blips` pickles `{base: {number: {key: io.BytesIO}}}`, each
//! BytesIO a WAV. Neither holds a tensor, so [`pickle`] evaluates them whole.

use std::{collections::BTreeMap, fmt};

use crate::{
    BlipSet, LATENT_DIM, Latent, archive,
    pickle::{self, Value, describe},
};

const PICKLE: &str = "data.pkl";

/// The records torch.save writes beside the pickle, with what they hold where that is fixed.
const RECORDS: [(&str, Option<&[u8]>); 5] = [
    (".format_version", Some(b"1")),
    (".storage_alignment", Some(b"64")),
    ("byteorder", Some(b"little")),
    ("version", Some(b"3\n")),
    (".data/serialization_id", None),
];

/// Reads a `.speaker_latent` file, given whole.
pub fn read_latent(file: &[u8]) -> Result<Latent, Error> {
    let value = pickle::load(pickle_of(file)?)?;
    let dict = dict_of(&value, "the pickle")?;
    if !dict.keys().eq(["latents", "text"]) {
        let keys: Vec<_> = dict.keys().collect();
        let reason = format!("the pickle has the keys {keys:?}, not [\"latents\", \"text\"]");
        return Err(Error::Content(reason));
    }
    let Value::Bytes(latents) = &dict["latents"] else {
        let reason = format!("\"latents\" is {}, not a bytes", describe(&dict["latents"]));
        return Err(Error::Content(reason));
    };
    let Value::Str(text) = &dict["text"] else {
        let reason = format!("\"text\" is {}, not a str", describe(&dict["text"]));
        return Err(Error::Content(reason));
    };
    if latents.len() % (4 * LATENT_DIM) != 0 {
        let reason = format!(
            "\"latents\" is {} bytes, not whole frames of 64 f32",
            latents.len()
        );
        return Err(Error::Content(reason));
    }
    let values: Vec<f32> = latents
        .as_chunks()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    Ok(Latent {
        frames: values.as_chunks::<LATENT_DIM>().0.to_vec(),
        text: text.clone(),
    })
}

/// Reads a `.blips` file, given whole.
///
/// Each WAV must be the canonical 44-byte header of 16-bit mono PCM followed by its samples, so
/// that the samples and their rate, which are what is kept, give back every byte of it. All
/// must be at the same rate.
pub fn read_blips(file: &[u8]) -> Result<BlipSet, Error> {
    let value = pickle::load(pickle_of(file)?)?;
    let mut set = BlipSet::default();
    let mut rate = None;
    for (base, numbers) in dict_of(&value, "the pickle")? {
        for (number, keys) in dict_of(numbers, base)? {
            for (key, wav) in dict_of(keys, &format!("{base}/{number}"))? {
                let name = format!("{base}/{number}/{key}");
                let Value::BytesIo(wav) = wav else {
                    let reason = format!("{name} is {}, not a BytesIO", describe(wav));
                    return Err(Error::Content(reason));
                };
                let (sample_rate, samples) =
                    pcm(wav).map_err(|reason| Error::Content(format!("{name}: {reason}")))?;
                let rate = *rate.get_or_insert(sample_rate);
                if sample_rate != rate {
                    let reason =
                        format!("{name} is at {sample_rate} Hz, the samples before at {rate}");
                    return Err(Error::Content(reason));
                }
                set.samples
                    .insert((base.clone(), number.clone(), key.clone()), samples);
            }
        }
    }
    set.sample_rate = rate.ok_or_else(|| Error::Content("no samples".to_owned()))?;
    Ok(set)
}

/// The pickle in a torch.save archive that holds the records torch writes beside it and no
/// others: no tensor's storage, which these files never have.
fn pickle_of(file: &[u8]) -> Result<&[u8], Error> {
    let records = archive::records(file).map_err(Error::Archive)?;
    let prefix = records
        .first()
        .and_then(|(name, _)| name.split_once('/'))
        .map_or("", |(prefix, _)| prefix);
    let mut by_name = BTreeMap::new();
    for (name, data) in records {
        let record = name.strip_prefix(prefix).and_then(|n| n.strip_prefix('/'));
        let expected = record.is_some_and(|r| r == PICKLE || RECORDS.iter().any(|(n, _)| *n == r));
        let (Some(record), true) = (record, expected) else {
            return Err(Error::Archive(format!("unexpected record {name:?}")));
        };
        by_name.insert(record, data);
    }
    for (record, contents) in RECORDS.iter().copied().chain([(PICKLE, None)]) {
        let Some(&data) = by_name.get(record) else {
            return Err(Error::Archive(format!("no record \"{prefix}/{record}\"")));
        };
        if let Some(contents) = contents
            && data != contents
        {
            let reason = format!(
                "\"{prefix}/{record}\" holds {:?}, not {:?}",
                String::from_utf8_lossy(data),
                String::from_utf8_lossy(contents)
            );
            return Err(Error::Archive(reason));
        }
    }
    Ok(by_name[PICKLE])
}

fn dict_of<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>, Error> {
    match value {
        Value::Dict(dict) => Ok(dict),
        value => Err(Error::Content(format!(
            "{what} is {}, not a dict",
            describe(value)
        ))),
    }
}

/// The rate and samples of `wav`, if it is a WAV that they give back byte for byte.
fn pcm(wav: &[u8]) -> Result<(u32, Vec<i16>), String> {
    let refuse = || {
        let header = &wav[..wav.len().min(44)];
        format!("not 16-bit mono PCM after a 44-byte header: {header:02x?}")
    };
    let (header, data) = wav.split_at_checked(44).ok_or_else(refuse)?;
    let rate = u32::from_le_bytes(header[24..28].try_into().expect("four bytes"));
    if wav_header(rate, data.len()).as_deref() != Some(header) {
        return Err(refuse());
    }
    let samples = data.as_chunks().0.iter().map(|s| i16::from_le_bytes(*s));
    Ok((rate, samples.collect()))
}

/// The 44 bytes that begin a WAV of `data_len` bytes of 16-bit mono PCM at `rate` and nothing
/// else, if they can hold those numbers.
fn wav_header(rate: u32, data_len: usize) -> Option<Vec<u8>> {
    const FMT_LEN: u32 = 16;
    const PCM: u16 = 1;
    const MONO: u16 = 1;
    const SAMPLE_BYTES: u16 = 2;
    let data_len = u32::try_from(data_len)
        .ok()
        .filter(|len| len % u32::from(SAMPLE_BYTES) == 0)?;
    let header = [
        &b"RIFF"[..],
        &data_len.checked_add(36)?.to_le_bytes(),
        b"WAVEfmt ",
        &FMT_LEN.to_le_bytes(),
        &PCM.to_le_bytes(),
        &MONO.to_le_bytes(),
        &rate.to_le_bytes(),
        &rate.checked_mul(SAMPLE_BYTES.into())?.to_le_bytes(),
        &SAMPLE_BYTES.to_le_bytes(),
        &(8 * SAMPLE_BYTES).to_le_bytes(),
        b"data",
        &data_len.to_le_bytes(),
    ];
    Some(header.concat())
}

/// A voice file that could not be read, and why.
#[derive(Debug)]
pub enum Error {
    /// The zip archive is not one torch.save writes for these files.
    Archive(String),
    /// The pickle uses what [`pickle`] refuses.
    Pickle(pickle::Error),
    /// The pickle is not a latent or a blips set as the service wrote them.
    Content(String),
}

impl From<pickle::Error> for Error {
    fn from(error: pickle::Error) -> Self {
        Self::Pickle(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive(reason) => write!(f, "archive: {reason}"),
            Self::Pickle(error) => write!(f, "{PICKLE}: {error}"),
            Self::Content(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pickle(error) => Some(error),
            _ => None,
        }
    }
}
