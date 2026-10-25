use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use voices::{ErrorKind, Kind, Latent, Voices, latent_path, legacy};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn convert(source: &Path, mapping: &Path, out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_convert-voices"))
        .args([source, mapping, out])
        .output()
        .unwrap()
}

fn bits(latent: &Latent) -> Vec<u32> {
    latent
        .frames
        .as_flattened()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A speaker_latents directory with a voice "Voice (Test)" whose latent and blips convert, a
/// voice whose latent does not, a voice with no files, a stem in no voice and a file that is
/// no voice's.
fn source(test: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("voices-{test}"));
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    let source = root.join("speaker_latents");
    fs::create_dir_all(&source).unwrap();
    for (fixture_name, name) in [
        ("latent.speaker_latent", "Test.speaker_latent"),
        ("blips.blips", "Test.blips"),
        ("ordered.speaker_latent", "Ordered.speaker_latent"),
        ("latent.speaker_latent", "Unnamed.speaker_latent"),
    ] {
        fs::copy(fixture(fixture_name), source.join(name)).unwrap();
    }
    fs::write(source.join("notes.txt"), "").unwrap();
    let mapping = root.join("voice_mapping.json");
    let table = r#"{"Voice (Test)": "Test", "Voice (Ordered)": "Ordered", "Voice (Gone)": "Gone"}"#;
    fs::write(&mapping, table).unwrap();
    (source, mapping, root.join("voices"))
}

#[test]
fn every_file_converts_or_is_named_with_why_not() {
    let (source, mapping, out) = source("convert");
    let output = convert(&source, &mapping, &out);
    assert_eq!(output.status.code(), Some(1));
    let report = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = report.lines().collect();
    assert_eq!(lines.len(), 6, "{report}");
    assert_eq!(
        lines[0],
        "failed Ordered.speaker_latent: data.pkl: pickle offset 2: global \
         collections.OrderedDict is not allowed"
    );
    assert_eq!(
        lines[1],
        "skipped notes.txt: not a .speaker_latent or .blips file"
    );
    let bytes_in = 2 * fs::metadata(fixture("latent.speaker_latent"))
        .unwrap()
        .len()
        + fs::metadata(fixture("blips.blips")).unwrap().len();
    let converted = format!("converted 2 latent and 1 blips files of {bytes_in} bytes to ");
    assert!(lines[2].starts_with(&converted), "{}", lines[2]);
    assert!(lines[2].ends_with(" s; 1 failed"), "{}", lines[2]);
    assert_eq!(
        lines[3..],
        [
            "voice \"Voice (Gone)\" (stem \"Gone\") has no latent and no blips",
            "voice \"Voice (Ordered)\" (stem \"Ordered\") has no latent and no blips",
            "stem \"Unnamed\", in no voice, has a latent and no blips",
        ]
    );
}

#[test]
fn converted_voices_read_as_their_sources() {
    let (source, mapping, out) = source("converted");
    convert(&source, &mapping, &out);
    let voices = Voices::open(&out).unwrap();
    assert_eq!(
        voices.names().collect::<Vec<_>>(),
        ["Voice (Gone)", "Voice (Ordered)", "Voice (Test)"]
    );

    let expected = legacy::read_latent(&fs::read(fixture("latent.speaker_latent")).unwrap());
    let expected = expected.unwrap();
    let latent = voices.latent("Voice (Test)").unwrap();
    assert_eq!(bits(&latent), bits(&expected));
    assert_eq!(latent.text, expected.text);
    let unnamed = Latent::read(&latent_path(&out, "Unnamed")).unwrap();
    assert_eq!(bits(&unnamed), bits(&expected));

    let expected = legacy::read_blips(&fs::read(fixture("blips.blips")).unwrap()).unwrap();
    let blips = voices.blips("Voice (Test)").unwrap();
    assert_eq!(blips.sample_rate(), expected.sample_rate);
    for ((base, number, key), samples) in &expected.samples {
        assert_eq!(blips.sample(base, number, key).unwrap(), *samples);
    }

    let error = voices.latent("Voice (Ordered)").unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::Missing(Kind::Latent)));
}

#[test]
fn existing_output_is_left_alone() {
    let (source, mapping, out) = source("existing");
    fs::create_dir(&out).unwrap();
    let output = convert(&source, &mapping, &out);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.starts_with(&format!("convert-voices: {}: ", out.display())),
        "{error}"
    );
    assert_eq!(fs::read_dir(&out).unwrap().count(), 0);
}
