//! How loud each voice comes out of the model, which streamed speech is brought to the service's
//! loudness by: a gain known before the speech is, where the whole-file routes measure each line
//! once they have all of it.
//!
//! A voice's level is the median integrated loudness of lines it has spoken, as the model gave
//! them, before anything levels them. `tgtts-gains` writes it, a voice at a time, as JSON:
//! `{"Man (Ace Detective)": {"loudness_lufs": -38.2, "lines": 12}, ...}`.

use std::{collections::BTreeMap, fs, io, path::Path};

use serde::{Deserialize, Serialize};

/// The voices' levels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Levels {
    voices: BTreeMap<String, Level>,
}

/// A voice's level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    /// The median integrated loudness of its lines, in LUFS.
    pub loudness_lufs: f64,
    /// The lines it is the median of.
    pub lines: usize,
}

impl Levels {
    /// The levels in the file at `path`.
    pub fn read(path: &Path) -> io::Result<Self> {
        let json = fs::read(path)?;
        let levels: Self = serde_json::from_slice(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if let Some((voice, _)) = (levels.voices.iter()).find(|(_, l)| !l.loudness_lufs.is_finite())
        {
            let reason = format!("{voice:?} has no finite loudness");
            return Err(io::Error::new(io::ErrorKind::InvalidData, reason));
        }
        Ok(levels)
    }

    /// Writes the levels to `path`, whole or not at all: to a file beside it first, then renamed
    /// over it.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut partial = path.as_os_str().to_owned();
        partial.push(".partial");
        let json = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        fs::write(&partial, json)?;
        fs::rename(&partial, path)
    }

    /// Sets `voice`'s level.
    pub fn set(&mut self, voice: &str, level: Level) {
        self.voices.insert(voice.to_owned(), level);
    }

    /// `voice`'s level, if it has one.
    pub fn get(&self, voice: &str) -> Option<Level> {
        self.voices.get(voice).copied()
    }

    /// `voice`'s loudness, or for a voice without a level the median of the voices', or none
    /// where no voice has one.
    pub fn loudness(&self, voice: &str) -> Option<f64> {
        if let Some(level) = self.voices.get(voice) {
            return Some(level.loudness_lufs);
        }
        let all: Vec<f64> = self.voices.values().map(|l| l.loudness_lufs).collect();
        median(&all)
    }

    /// How many voices have a level.
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// Whether no voice has one.
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }
}

/// The median of `values`, the mean of the middle two of an even number, or none of none.
pub fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(sorted[n / 2]),
        _ => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_without_a_level_gets_the_median_of_theirs() {
        let mut levels = Levels::default();
        assert_eq!(levels.loudness("A"), None);
        for (voice, loudness) in [("A", -40.0), ("B", -30.0), ("C", -32.0)] {
            let level = Level {
                loudness_lufs: loudness,
                lines: 12,
            };
            levels.set(voice, level);
        }
        assert_eq!(levels.loudness("B"), Some(-30.0));
        assert_eq!(levels.loudness("Z"), Some(-32.0));
        assert_eq!(median(&[1.0, 4.0, 2.0, 3.0]), Some(2.5));
    }

    #[test]
    fn reads_what_it_writes_and_refuses_what_is_not_a_level() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("levels.json");
        let mut levels = Levels::default();
        let level = Level {
            loudness_lufs: -38.25,
            lines: 3,
        };
        levels.set("Man (Test)", level);
        levels.write(&path).unwrap();
        assert_eq!(Levels::read(&path).unwrap(), levels);
        let json = fs::read_to_string(&path).unwrap();
        assert!(json.contains(r#""Man (Test)": {"#), "{json}");
        fs::write(&path, r#"{"A": {"loudness_lufs": "loud", "lines": 1}}"#).unwrap();
        assert!(Levels::read(&path).is_err());
        fs::write(&path, r#"{"A": {"loudness_lufs": -1e999, "lines": 1}}"#).unwrap();
        assert!(Levels::read(&path).is_err());
    }
}
