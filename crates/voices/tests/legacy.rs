use std::collections::BTreeMap;

use voices::{
    legacy::{self, Error},
    pickle::{self, ErrorKind},
};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

const KEYS: [&str; 41] = [
    "0", "1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9", "Deska", "Gwah", "OK", "a", "b",
    "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
    "v", "w", "x", "y", "z",
];

#[test]
fn latent_keeps_every_bit_of_the_latents() {
    let latent = legacy::read_latent(&fixture("latent.speaker_latent")).unwrap();
    let mut expected: Vec<u8> = (0..=255).cycle().take(3 * 256).collect();
    for bits in [
        0x8000_0000_u32,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
        0x7fc0_0001,
    ] {
        expected.extend(bits.to_le_bytes());
    }
    expected.resize(4 * 256, 0);
    let bytes: Vec<u8> = latent
        .frames
        .as_flattened()
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    assert_eq!(latent.frames.len(), 4);
    assert_eq!(bytes, expected);
}

#[test]
fn latent_keeps_its_transcript() {
    let latent = legacy::read_latent(&fixture("latent.speaker_latent")).unwrap();
    assert_eq!(
        latent.text,
        "Said \"caf\u{e9}\", \u{2014} then \u{79c1}\u{306f}.\nA second line."
    );
}

#[test]
fn blips_hold_every_sample_where_it_was() {
    let set = legacy::read_blips(&fixture("blips.blips")).unwrap();
    let mut expected = BTreeMap::new();
    for (b, base) in ["male", "female"].into_iter().enumerate() {
        for n in 1..=4 {
            for (k, key) in KEYS.into_iter().enumerate() {
                let at = (1000 * b + 100 * n + k) as i16;
                let samples = [at, i16::MIN, i16::MAX][..k % 4].to_vec();
                expected.insert((base.to_owned(), n.to_string(), key.to_owned()), samples);
            }
        }
    }
    assert_eq!(set.sample_rate, 48000);
    assert_eq!(set.samples, expected);
}

#[test]
fn stereo_is_refused() {
    let error = legacy::read_blips(&fixture("stereo.blips")).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("male/1/b: not 16-bit mono PCM after a 44-byte header: [52, 49, 46, 46,"),
        "{error}"
    );
}

#[test]
fn samples_at_another_rate_are_refused() {
    let error = legacy::read_blips(&fixture("rates.blips")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "male/1/b is at 44100 Hz, the samples before at 48000"
    );
}

#[test]
fn ordered_dict_is_refused_by_its_global() {
    let error = legacy::read_latent(&fixture("ordered.speaker_latent")).unwrap_err();
    let Error::Pickle(error) = &error else {
        panic!("{error}");
    };
    let global = ErrorKind::Global {
        module: "collections".into(),
        name: "OrderedDict".into(),
    };
    assert_eq!(
        *error,
        pickle::Error {
            offset: 2,
            kind: global
        }
    );
    assert_eq!(
        error.to_string(),
        "pickle offset 2: global collections.OrderedDict is not allowed"
    );
}

#[test]
fn protocol_4_is_refused() {
    let error = legacy::read_latent(&fixture("protocol4.speaker_latent")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "data.pkl: pickle offset 0: protocol 4, where only 2 is read"
    );
}

#[test]
fn tensor_storage_is_refused() {
    let error = legacy::read_latent(&fixture("tensor.speaker_latent")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "archive: unexpected record \"tensor/data/0\""
    );
}

#[test]
fn compressed_record_is_refused() {
    let error = legacy::read_latent(&fixture("deflated.speaker_latent")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "archive: record \"deflated/data.pkl\" is compressed (method 8)"
    );
}

#[test]
fn damaged_record_fails_its_crc() {
    let mut file = fixture("latent.speaker_latent");
    let at = file.windows(4).position(|w| w == b"text").unwrap();
    file[at] = b'T';
    let error = legacy::read_latent(&file).unwrap_err();
    assert_eq!(
        error.to_string(),
        "archive: record \"latent/data.pkl\" fails its CRC-32"
    );
}

#[test]
fn truncated_archive_is_refused() {
    let file = fixture("latent.speaker_latent");
    let error = legacy::read_latent(&file[..file.len() - 1]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "archive: does not end in an end of central directory record without a comment"
    );
}

#[test]
fn latent_is_not_blips_nor_blips_a_latent() {
    let error = legacy::read_blips(&fixture("latent.speaker_latent")).unwrap_err();
    assert_eq!(error.to_string(), "latents is a bytes, not a dict");
    let error = legacy::read_latent(&fixture("blips.blips")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "the pickle has the keys [\"female\", \"male\"], not [\"latents\", \"text\"]"
    );
}
