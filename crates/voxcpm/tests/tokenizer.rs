use std::{fs, path::PathBuf};

use safetensors::SafeTensors;
use serde_json::{Value, json};
use testkit::{checkpoint_dir, fixture};
use voices::Latent;
use voxcpm::{Error, Tokenizer};

fn tiny() -> Value {
    serde_json::from_str(include_str!("fixtures/voxcpm_tokenizer.json")).unwrap()
}

#[test]
fn encodes_as_upstream() {
    let tokenizer = Tokenizer::from_json(include_str!("fixtures/voxcpm_tokenizer.json")).unwrap();
    let fixture = fixture!("voxcpm_tokenizer");
    let texts: Vec<String> = serde_json::from_str(&fixture.metadata()["texts"]).unwrap();
    for (i, text) in texts.iter().enumerate() {
        let (_, expected) = fixture.ints(&format!("ids.{i}"));
        let ids: Vec<i64> = tokenizer.encode(text).into_iter().map(i64::from).collect();
        assert_eq!(ids, expected, "{text:?}");
    }
}

#[test]
fn what_is_not_implemented_is_refused() {
    let cases = [
        ("/pre_tokenizer", json!({"type": "Whitespace"})),
        ("/normalizer", Value::Null),
        ("/model/byte_fallback", json!(false)),
        ("/model/dropout", json!(0.1)),
        ("/model/type", json!("Unigram")),
        ("/added_tokens/3/normalized", json!(true)),
        ("/added_tokens/3/lstrip", json!(true)),
    ];
    for (pointer, value) in cases {
        let mut json = tiny();
        *json.pointer_mut(pointer).unwrap() = value;
        let error = Tokenizer::from_json(&json.to_string()).unwrap_err();
        assert!(matches!(error, Error::Tokenizer(_)), "{pointer}: {error}");
    }
}

#[test]
fn a_missing_byte_is_refused() {
    let mut json = tiny();
    json["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("<0xE9>");
    let error = Tokenizer::from_json(&json.to_string()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "tokenizer.json: \"<0xE9>\" is not in the vocabulary"
    );
}

/// The voices directory `VOICES_DIR` names.
fn voices_dir() -> PathBuf {
    let Some(dir) = std::env::var_os("VOICES_DIR").filter(|dir| !dir.is_empty()) else {
        panic!("VOICES_DIR is not set; point it at a voices directory");
    };
    std::path::absolute(dir).expect("absolute of a non-empty path")
}

/// Making the golden needs the checkpoint and the voices, so a run of every generator keeps it
/// as it is: this refuses one that such a run has left behind.
#[test]
fn checkpoint_golden_was_made_as_the_fixtures_were() {
    let golden = fixture!("voxcpm_tokenizer_checkpoint");
    let fixtures = fixture!("voxcpm_tokenizer");
    let regenerate = "regenerate it with VOXCPM2_DIR and VOICES_DIR set: \
                      nix run .#fixtures voxcpm_tokenizer_checkpoint";
    for key in ["torch", "upstream"] {
        assert_eq!(
            golden.metadata()[key],
            fixtures.metadata()[key],
            "the golden's {key} differs from the fixtures'; {regenerate}"
        );
    }
    let generator = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/voxcpm_tokenizer_checkpoint.py"
    );
    let crc32 = format!("{:08x}", crc32fast::hash(&fs::read(generator).unwrap()));
    assert_eq!(
        golden.metadata().get("generator_crc32"),
        Some(&crc32),
        "fixtures/voxcpm_tokenizer_checkpoint.py changed after making the golden; {regenerate}"
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and VOICES_DIR a voices directory"]
fn encodes_every_transcript_and_message_as_upstream() {
    let json = fs::read_to_string(checkpoint_dir().join("tokenizer.json")).unwrap();
    let tokenizer = Tokenizer::from_json(&json).unwrap();
    let golden = fixture!("voxcpm_tokenizer_checkpoint");
    let list = |key| -> Vec<String> { serde_json::from_str(&golden.metadata()[key]).unwrap() };
    let (stems, messages) = (list("stems"), list("messages"));
    for (i, message) in messages.iter().enumerate() {
        let (_, expected) = golden.ints(&format!("messages.{i}"));
        let ids: Vec<i64> = tokenizer
            .encode(message)
            .into_iter()
            .map(i64::from)
            .collect();
        assert_eq!(ids, expected, "{message:?}");
    }
    // Read past burn, which would hold the checksums as i32.
    let bytes = fs::read(golden.path()).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let rows = |name| -> Vec<i64> {
        let rows = file.tensor(name).unwrap();
        assert_eq!(rows.shape(), [stems.len(), 3]);
        let values = rows.data().as_chunks().0.iter();
        values.map(|value| i64::from_le_bytes(*value)).collect()
    };
    let (transcripts, prompts) = (rows("transcripts"), rows("prompts"));
    let voices = voices_dir();
    for (i, stem) in stems.iter().enumerate() {
        let text = Latent::read(&voices::latent_path(&voices, stem))
            .unwrap()
            .text;
        let prompt = format!("{text}{}", messages[i % messages.len()]);
        for (text, rows, what) in [
            (text, &transcripts, "transcript"),
            (prompt, &prompts, "prompt"),
        ] {
            let row = &rows[3 * i..3 * i + 3];
            let crc32 = i64::from(crc32fast::hash(text.as_bytes()));
            assert_eq!(
                crc32, row[0],
                "{stem}: the {what} is not the one the golden was made of"
            );
            let ids = tokenizer.encode(&text);
            let bytes: Vec<u8> = ids
                .iter()
                .flat_map(|&id| i64::from(id).to_le_bytes())
                .collect();
            let digest = (ids.len() as i64, i64::from(crc32fast::hash(&bytes)));
            assert_eq!(digest, (row[1], row[2]), "{stem}: the {what}'s ids {ids:?}");
        }
    }
    eprintln!(
        "{} messages, and {} transcripts alone and before a message, encoded as upstream",
        messages.len(),
        stems.len()
    );
}
