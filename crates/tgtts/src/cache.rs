//! The speech cache: what the model said for a voice and text, kept on disk as it left the model
//! service, up to five takes of each; past that a request is given one of them at random, as the
//! old frontend did.
//!
//! A take is written whole under a temporary name and linked into place under a name no other
//! take holds, so concurrent requests neither see a partial file nor keep more than five.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use dsp::segment::Segment;
use rand::{Rng, RngExt};

use crate::wav;

/// How many takes of a voice and text are made before they are reused.
pub const TAKES: usize = 5;

/// A cache directory.
#[derive(Clone, Debug)]
pub struct Cache {
    dir: PathBuf,
}

/// What the cache holds of a voice and text.
#[derive(Clone, Debug, PartialEq)]
pub struct Lookup {
    /// How many takes it holds, up to [`TAKES`].
    pub takes: usize,
    /// One of them, once it holds all of them.
    pub take: Option<Segment>,
}

impl Cache {
    /// The cache in `dir`, created if it does not exist.
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Where the takes of `text` in `voice` are kept.
    ///
    /// The old frontend hashed the voice and the text run together, so that one voice's text
    /// could name another's; the voice's length goes first here, which no two pairs share with
    /// the same bytes after it.
    pub fn entry(&self, voice: &str, text: &str) -> PathBuf {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(voice.len() as u64).to_le_bytes());
        hasher.update(voice.as_bytes());
        hasher.update(text.as_bytes());
        self.dir.join(hasher.finalize().to_hex().as_str())
    }

    /// How many takes of `text` in `voice` have been made, and once all [`TAKES`] have, one of
    /// them drawn from `rng`; while fewer have, another is to be made.
    pub fn lookup(&self, voice: &str, text: &str, rng: &mut impl Rng) -> io::Result<Lookup> {
        let entry = self.entry(voice, text);
        let taken = takes(&entry)?;
        let take = if taken.len() < TAKES {
            None
        } else {
            Some(read(&taken[rng.random_range(0..taken.len())])?)
        };
        Ok(Lookup {
            takes: taken.len(),
            take,
        })
    }

    /// Keeps `speech` as a take of `text` in `voice`, unless all [`TAKES`] are kept already.
    /// Says whether it was kept.
    pub fn store(
        &self,
        voice: &str,
        text: &str,
        speech: &Segment,
        rng: &mut impl Rng,
    ) -> io::Result<bool> {
        let entry = self.entry(voice, text);
        fs::create_dir_all(&entry)?;
        let temporary = entry.join(format!(".{:016x}.tmp", rng.random::<u64>()));
        fs::write(
            &temporary,
            wav::write_pcm16(speech.rate, 1, &speech.samples),
        )?;
        let mut kept = Ok(false);
        for take in 0..TAKES {
            match fs::hard_link(&temporary, take_path(&entry, take)) {
                Ok(()) => {
                    kept = Ok(true);
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => {
                    kept = Err(e);
                    break;
                }
            }
        }
        fs::remove_file(&temporary)?;
        kept
    }
}

fn take_path(entry: &Path, take: usize) -> PathBuf {
    entry.join(format!("{take}.wav"))
}

/// The takes kept in `entry`.
fn takes(entry: &Path) -> io::Result<Vec<PathBuf>> {
    let mut taken = Vec::new();
    for take in 0..TAKES {
        let path = take_path(entry, take);
        match fs::metadata(&path) {
            Ok(_) => taken.push(path),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(taken)
}

fn read(path: &Path) -> io::Result<Segment> {
    let invalid = |reason: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {reason}", path.display()),
        )
    };
    let audio = wav::read(&fs::read(path)?).map_err(|e| invalid(e.to_string()))?;
    if audio.channels != 1 {
        return Err(invalid(format!("{} channels, not 1", audio.channels)));
    }
    Ok(Segment::new(audio.rate, audio.to_pcm().samples))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc, thread};

    use super::*;

    fn speech(n: i16) -> Segment {
        Segment::new(48000, vec![n, -n, n])
    }

    #[test]
    fn keys_cannot_collide() {
        let cache = Cache::open(tempfile::tempdir().unwrap().path()).unwrap();
        assert_ne!(cache.entry("ab", "c"), cache.entry("a", "bc"));
        assert_ne!(cache.entry("", "x"), cache.entry("x", ""));
        assert_eq!(cache.entry("a", "bc"), cache.entry("a", "bc"));
    }

    #[test]
    fn makes_five_takes_then_reuses_them() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let mut rng = rand::rng();
        for (made, n) in (1..=5).enumerate() {
            let lookup = cache.lookup("v", "hello", &mut rng).unwrap();
            let none = Lookup {
                takes: made,
                take: None,
            };
            assert_eq!(lookup, none);
            assert!(cache.store("v", "hello", &speech(n), &mut rng).unwrap());
        }
        assert!(!cache.store("v", "hello", &speech(6), &mut rng).unwrap());
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let lookup = cache.lookup("v", "hello", &mut rng).unwrap();
            assert_eq!(lookup.takes, TAKES);
            let take = lookup.take.unwrap();
            assert_eq!(take.rate, 48000);
            seen.insert(take.samples[0]);
        }
        assert_eq!(seen, HashSet::from([1, 2, 3, 4, 5]));
        let entry = cache.entry("v", "hello");
        let mut names: Vec<_> = fs::read_dir(&entry)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["0.wav", "1.wav", "2.wav", "3.wav", "4.wav"]);
        assert_eq!(
            fs::read(entry.join("0.wav")).unwrap(),
            wav::write_pcm16(48000, 1, &[1, -1, 1])
        );
        let other = cache.lookup("v", "other", &mut rng).unwrap();
        assert_eq!((other.takes, other.take), (0, None));
    }

    #[test]
    fn concurrent_stores_keep_five() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(Cache::open(dir.path()).unwrap());
        let threads: Vec<_> = (1..=16)
            .map(|n| {
                let cache = cache.clone();
                thread::spawn(move || cache.store("v", "t", &speech(n), &mut rand::rng()).unwrap())
            })
            .collect();
        let kept = threads.into_iter().map(|t| t.join().unwrap());
        let kept = kept.filter(|&k| k).count();
        assert_eq!(kept, TAKES);
        assert_eq!(fs::read_dir(cache.entry("v", "t")).unwrap().count(), TAKES);
    }

    #[test]
    fn a_corrupt_take_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let mut rng = rand::rng();
        for n in 1..=5 {
            cache.store("v", "t", &speech(n), &mut rng).unwrap();
        }
        for take in 0..TAKES {
            fs::write(cache.entry("v", "t").join(format!("{take}.wav")), b"RIFF").unwrap();
        }
        let error = cache.lookup("v", "t", &mut rng).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
