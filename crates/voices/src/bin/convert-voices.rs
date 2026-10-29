//! Converts the Python service's voice files into a voices directory.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Display,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use argh::FromArgs;
use voices::{Kind, Voices, blips_path, latent_path, legacy, table_path, write_table};

/// Converts every \<stem\>.speaker_latent and \<stem\>.blips file in SPEAKER_LATENTS, as the
/// Python service reads them, into the voices directory OUT, which must not exist, with
/// VOICE_MAPPING, the service's voice_mapping.json, as its table. Each file that cannot be
/// converted is named with the reason, and the table is checked against the files: a voice
/// without a latent or blips is named, and so is a stem in no voice. Exits 1 if any file failed.
#[derive(FromArgs)]
struct Args {
    /// the Python service's directory of voice files
    #[argh(positional, arg_name = "SPEAKER_LATENTS")]
    source: PathBuf,
    /// the Python service's voice_mapping.json
    #[argh(positional, arg_name = "VOICE_MAPPING")]
    mapping: PathBuf,
    /// the voices directory to make
    #[argh(positional, arg_name = "OUT")]
    out: PathBuf,
}

fn main() -> ExitCode {
    let Args {
        source,
        mapping,
        out,
    } = argh::from_env();
    match convert(&source, &mapping, &out) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("convert-voices: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Converts what it can, reporting as it goes, and says whether every voice file converted.
fn convert(source: &Path, mapping: &Path, out: &Path) -> Result<bool, String> {
    let start = Instant::now();
    let json = fs::read(mapping).map_err(at(mapping))?;
    let table: BTreeMap<String, String> = serde_json::from_slice(&json).map_err(at(mapping))?;
    let mut files = fs::read_dir(source)
        .and_then(|dir| {
            dir.map(|entry| entry.map(|e| e.path()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(at(source))?;
    files.sort();
    for dir in [out, &out.join("latents"), &out.join("blips")] {
        fs::create_dir(dir).map_err(at(dir))?;
    }

    let (mut latents, mut blips) = (BTreeSet::new(), BTreeSet::new());
    let (mut failed, mut bytes_in, mut bytes_out) = (0, 0, 0);
    for path in &files {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let (stem, kind) = match name.rsplit_once('.') {
            Some((stem, "speaker_latent")) => (stem, Kind::Latent),
            Some((stem, "blips")) => (stem, Kind::Blips),
            _ => {
                println!("skipped {name}: not a .speaker_latent or .blips file");
                continue;
            }
        };
        match convert_file(path, kind, out, stem) {
            Ok((read, wrote)) => {
                bytes_in += read;
                bytes_out += wrote;
                match kind {
                    Kind::Latent => latents.insert(stem.to_owned()),
                    Kind::Blips => blips.insert(stem.to_owned()),
                };
            }
            Err(e) => {
                println!("failed {name}: {e}");
                failed += 1;
            }
        }
    }

    write_table(out, &table).map_err(|e| e.to_string())?;
    Voices::open(out).map_err(|e| e.to_string())?;
    bytes_out += fs::metadata(table_path(out)).map_err(at(out))?.len();
    println!(
        "converted {} latent and {} blips files of {bytes_in} bytes to {bytes_out} bytes in \
         {:.1} s; {failed} failed",
        latents.len(),
        blips.len(),
        start.elapsed().as_secs_f64(),
    );

    let has = |stem: &str| match (latents.contains(stem), blips.contains(stem)) {
        (true, true) => "a latent and blips",
        (true, false) => "a latent and no blips",
        (false, true) => "blips and no latent",
        (false, false) => "no latent and no blips",
    };
    for (voice, stem) in &table {
        if !(latents.contains(stem) && blips.contains(stem)) {
            println!("voice {voice:?} (stem {stem:?}) has {}", has(stem));
        }
    }
    let named: BTreeSet<&String> = table.values().collect();
    for stem in latents.union(&blips).filter(|stem| !named.contains(stem)) {
        println!("stem {stem:?}, in no voice, has {}", has(stem));
    }
    Ok(failed == 0)
}

/// Converts one file, giving the bytes read and written.
fn convert_file(path: &Path, kind: Kind, out: &Path, stem: &str) -> Result<(u64, u64), String> {
    let file = fs::read(path).map_err(|e| e.to_string())?;
    let written = match kind {
        Kind::Latent => {
            let written = latent_path(out, stem);
            let latent = legacy::read_latent(&file).map_err(|e| e.to_string())?;
            latent.write(&written).map_err(|e| e.to_string())?;
            written
        }
        Kind::Blips => {
            let written = blips_path(out, stem);
            let set = legacy::read_blips(&file).map_err(|e| e.to_string())?;
            set.write(&written).map_err(|e| e.to_string())?;
            written
        }
    };
    let wrote = fs::metadata(&written).map_err(at(&written))?.len();
    Ok((file.len() as u64, wrote))
}

fn at<E: Display>(path: &Path) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}
