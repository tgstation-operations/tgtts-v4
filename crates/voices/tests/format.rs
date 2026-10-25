use std::{
    fs,
    path::{Path, PathBuf},
};

use voices::{
    BlipSet, Blips, ErrorKind, Kind, LATENT_DIM, Latent, Voices, blips_path, latent_path,
    table_path, write_table,
};

/// A latent of one frame, 1.0 first and -0.0 last, and the transcript `Say "cafe".` with an
/// acute e and a newline, which the header has to escape.
fn latent() -> Latent {
    let mut frame = [0.0; LATENT_DIM];
    frame[0] = 1.0;
    frame[LATENT_DIM - 1] = -0.0;
    Latent {
        frames: vec![frame],
        text: "Say \"caf\u{e9}\".\n".to_owned(),
    }
}

fn latent_bytes() -> Vec<u8> {
    let mut bytes = Vec::from(112u64.to_le_bytes());
    bytes.extend(
        b"{\"__metadata__\":{\"text\":\"Say \\\"caf\xc3\xa9\\\".\\n\"},\
          \"latents\":{\"dtype\":\"F32\",\"shape\":[1,64],\"data_offsets\":[0,256]}}    ",
    );
    bytes.extend([0x00, 0x00, 0x80, 0x3f]);
    bytes.extend([0; 248]);
    bytes.extend([0x00, 0x00, 0x00, 0x80]);
    bytes
}

fn blip_set() -> BlipSet {
    let sample =
        |base: &str, number: &str, key: &str| (base.to_owned(), number.to_owned(), key.to_owned());
    BlipSet {
        sample_rate: 48000,
        samples: [
            (sample("male", "1", "a"), vec![1, -2]),
            (sample("female", "4", "Gwah"), vec![]),
            (sample("male", "1", "10"), vec![i16::MAX, i16::MIN]),
        ]
        .into(),
    }
}

fn blips_bytes() -> Vec<u8> {
    let mut bytes = Vec::from(232u64.to_le_bytes());
    bytes.extend(
        b"{\"__metadata__\":{\"sample_rate\":\"48000\"},\
          \"female/4/Gwah\":{\"dtype\":\"I16\",\"shape\":[0],\"data_offsets\":[0,0]},\
          \"male/1/10\":{\"dtype\":\"I16\",\"shape\":[2],\"data_offsets\":[0,4]},\
          \"male/1/a\":{\"dtype\":\"I16\",\"shape\":[2],\"data_offsets\":[4,8]}}      ",
    );
    bytes.extend([0xff, 0x7f, 0x00, 0x80, 0x01, 0x00, 0xfe, 0xff]);
    bytes
}

/// A fresh directory for one test.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("voices-{test}"));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(dir.join("latents")).unwrap();
    fs::create_dir_all(dir.join("blips")).unwrap();
    dir
}

/// A voices directory holding the latent and blips above under the stem "Test", which the voice
/// "Voice (Test)" names.
fn voices(test: &str) -> (PathBuf, Voices) {
    let dir = scratch(test);
    fs::write(table_path(&dir), r#"{"Voice (Test)": "Test"}"#).unwrap();
    fs::write(latent_path(&dir, "Test"), latent_bytes()).unwrap();
    fs::write(blips_path(&dir, "Test"), blips_bytes()).unwrap();
    let voices = Voices::open(&dir).unwrap();
    (dir, voices)
}

fn bits(latent: &Latent) -> Vec<u32> {
    latent
        .frames
        .as_flattened()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn latent_is_written_as_these_bytes() {
    let dir = scratch("latent-written");
    let path = latent_path(&dir, "Test");
    latent().write(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), latent_bytes());
}

#[test]
fn latent_reads_from_these_bytes() {
    let (_, voices) = voices("latent-read");
    let read = voices.latent("Voice (Test)").unwrap();
    assert_eq!(read.text, latent().text);
    assert_eq!(bits(&read), bits(&latent()));
    assert_eq!(read.frames[0][LATENT_DIM - 1].to_bits(), 0x8000_0000);
}

#[test]
fn blips_are_written_as_these_bytes() {
    let dir = scratch("blips-written");
    let path = blips_path(&dir, "Test");
    blip_set().write(&path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), blips_bytes());
}

#[test]
fn blips_read_from_these_bytes() {
    let (_, voices) = voices("blips-read");
    let blips = voices.blips("Voice (Test)").unwrap();
    assert_eq!(blips.sample_rate(), 48000);
    assert_eq!(blips.sample("male", "1", "a").unwrap(), [1, -2]);
    assert_eq!(
        blips.sample("male", "1", "10").unwrap(),
        [i16::MAX, i16::MIN]
    );
    assert!(blips.sample("female", "4", "Gwah").unwrap().is_empty());
}

#[test]
fn table_is_written_sorted_and_read_back() {
    let dir = scratch("table");
    let table = [("b", "stem b"), ("A", "stem a"), ("a", "stem a")]
        .map(|(name, stem)| (name.to_owned(), stem.to_owned()))
        .into();
    write_table(&dir, &table).unwrap();
    assert_eq!(
        fs::read_to_string(table_path(&dir)).unwrap(),
        "{\n  \"A\": \"stem a\",\n  \"a\": \"stem a\",\n  \"b\": \"stem b\"\n}\n"
    );
    let voices = Voices::open(&dir).unwrap();
    assert_eq!(voices.names().collect::<Vec<_>>(), ["A", "a", "b"]);
}

#[test]
fn unknown_voice_is_named() {
    let (_, voices) = voices("unknown");
    let error = voices.latent("Voice (Other)").unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::UnknownVoice));
    assert_eq!(
        error.to_string(),
        "voice \"Voice (Other)\": not in the table"
    );
}

#[test]
fn missing_files_are_named_with_their_voice() {
    let (dir, voices) = voices("missing");
    fs::remove_file(latent_path(&dir, "Test")).unwrap();
    fs::remove_file(blips_path(&dir, "Test")).unwrap();

    let error = voices.latent("Voice (Test)").unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::Missing(Kind::Latent)));
    let path = latent_path(&dir, "Test");
    assert_eq!(
        error.to_string(),
        format!("voice \"Voice (Test)\": {}: no latent", path.display())
    );

    let error = voices.blips("Voice (Test)").unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::Missing(Kind::Blips)));
    assert_eq!(error.voice(), Some("Voice (Test)"));
    assert_eq!(error.path(), Some(blips_path(&dir, "Test").as_path()));
}

#[test]
fn missing_sample_is_named_with_its_voice() {
    let (dir, voices) = voices("no-sample");
    let error = voices
        .blips("Voice (Test)")
        .unwrap()
        .sample("male", "2", "a")
        .unwrap_err();
    let path = blips_path(&dir, "Test");
    assert_eq!(
        error.to_string(),
        format!(
            "voice \"Voice (Test)\": {}: no blip male/2/a",
            path.display()
        )
    );
}

#[test]
fn a_slash_in_a_lookup_finds_no_other_sample() {
    let (_, voices) = voices("slash");
    let blips = voices.blips("Voice (Test)").unwrap();
    for parts in [
        ["male/1", "a", ""],
        ["male", "1/a", ""],
        ["", "male", "1/a"],
    ] {
        let [base, number, key] = parts;
        let error = blips.sample(base, number, key).unwrap_err();
        assert!(matches!(error.kind(), ErrorKind::NoSample(p) if *p == parts));
    }
}

#[test]
fn a_blip_named_with_a_slash_is_not_written() {
    let dir = scratch("write-slash");
    let mut set = blip_set();
    set.samples
        .insert(("male".into(), "1".into(), "a/b".into()), vec![0]);
    let error = set.write(&blips_path(&dir, "Test")).unwrap_err();
    assert!(
        error
            .to_string()
            .ends_with("blip [\"male\", \"1\", \"a/b\"] cannot be named <base>/<number>/<key>")
    );
}

#[test]
fn stem_that_is_not_a_file_name_is_refused() {
    let dir = scratch("stem");
    fs::write(table_path(&dir), r#"{"Voice (Test)": "../Test"}"#).unwrap();
    let error = Voices::open(&dir).unwrap_err();
    assert!(error.to_string().ends_with(
        "voices.json: voice \"Voice (Test)\" has the stem \"../Test\", which is not a file name"
    ));
}

#[test]
fn latent_of_the_wrong_shape_is_refused() {
    let dir = scratch("latent-shape");
    let path = latent_path(&dir, "Test");
    let mut bytes = latent_bytes();
    let header = &mut bytes[8..120];
    let at = header.windows(6).position(|w| w == b"[1,64]").unwrap();
    header[at..at + 6].copy_from_slice(b"[2,32]");
    fs::write(&path, bytes).unwrap();
    let error = Latent::read(&path).unwrap_err();
    assert!(
        error
            .to_string()
            .ends_with("\"latents\" is F32 [2, 32], not F32 [frames, 64]")
    );
}

#[test]
fn truncated_file_is_refused() {
    let dir = scratch("truncated");
    let path = blips_path(&dir, "Test");
    let bytes = blips_bytes();
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    let error = Blips::open(&path).unwrap_err();
    assert!(
        error
            .to_string()
            .ends_with("247 bytes, where the header describes 248")
    );
}

#[test]
fn latents_file_is_not_blips() {
    let dir = scratch("kinds");
    let path = blips_path(&dir, "Test");
    fs::write(&path, latent_bytes()).unwrap();
    let error = Blips::open(&path).unwrap_err();
    assert!(
        error
            .to_string()
            .ends_with("no \"sample_rate\" in the header's metadata")
    );
}
