use std::{
    env, fs,
    path::{Path, PathBuf},
};

use safetensors::SafeTensors;
use voices::{Blips, Latent, Voices, blips_path, latent_path, legacy};

/// The directory the variable `var` names.
fn dir(var: &str) -> PathBuf {
    let Some(dir) = env::var_os(var).filter(|dir| !dir.is_empty()) else {
        panic!("{var} is not set");
    };
    // A test runs in its crate's directory, which a relative path is then taken from.
    let dir = std::path::absolute(dir).expect("absolute of a non-empty path");
    assert!(dir.is_dir(), "{var}: {} is not a directory", dir.display());
    dir
}

fn bits(latent: &Latent) -> Vec<u32> {
    latent
        .frames
        .as_flattened()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn tensors(path: &Path) -> usize {
    let bytes = fs::read(path).unwrap();
    SafeTensors::deserialize(&bytes).unwrap().len()
}

#[test]
#[ignore = "needs SPEAKER_LATENTS_DIR to name the Python service's speaker_latents and \
            VOICES_DIR the voices directory convert-voices made of it"]
fn every_voice_reads_as_its_source() {
    let (source, out) = (dir("SPEAKER_LATENTS_DIR"), dir("VOICES_DIR"));
    let mut files: Vec<_> = fs::read_dir(&source)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    let (mut latents, mut blips, mut samples) = (0, 0, 0);
    for path in &files {
        let name = path.file_name().unwrap().to_str().unwrap();
        match name.rsplit_once('.') {
            Some((stem, "speaker_latent")) => {
                let expected = legacy::read_latent(&fs::read(path).unwrap()).unwrap();
                let latent = Latent::read(&latent_path(&out, stem)).unwrap();
                assert_eq!(bits(&latent), bits(&expected), "{name}");
                assert_eq!(latent.text, expected.text, "{name}");
                latents += 1;
            }
            Some((stem, "blips")) => {
                let expected = legacy::read_blips(&fs::read(path).unwrap()).unwrap();
                let converted = blips_path(&out, stem);
                let set = Blips::open(&converted).unwrap();
                assert_eq!(set.sample_rate(), expected.sample_rate, "{name}");
                assert_eq!(tensors(&converted), expected.samples.len(), "{name}");
                for ((base, number, key), pcm) in &expected.samples {
                    let sample = set.sample(base, number, key).unwrap();
                    assert_eq!(sample, *pcm, "{name}: {base}/{number}/{key}");
                }
                samples += expected.samples.len();
                blips += 1;
            }
            _ => {}
        }
    }
    let voices = Voices::open(&out).unwrap();
    for voice in voices.names() {
        voices.latent(voice).unwrap();
        voices.blips(voice).unwrap();
    }
    let named = voices.names().count();
    eprintln!("{latents} latents, {blips} blips ({samples} samples) and {named} voices read");
}
