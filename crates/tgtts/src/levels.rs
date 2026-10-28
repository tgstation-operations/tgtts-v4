//! How loud each voice comes out of the model, which streamed speech is brought to the service's
//! loudness by: a gain known before the speech is, where the whole-file routes measure each line
//! once they have all of it.
//!
//! A voice's level is the median integrated loudness of the same lines spoken in it, as the
//! model gave them, before anything levels them, each with the silence the old frontend put
//! after it, as the whole-file routes measure a line: on a short line that silence moves the
//! measure by decibels. [`measure`] measures them, a voice at a time, for `tgtts-gains`, which
//! writes them as JSON: `{"Man (Ace Detective)": {"loudness_lufs": -38.2, "lines": 12}, ...}`.

use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    io::{self, Write},
    path::Path,
    sync::{Mutex, mpsc},
    thread,
    time::Instant,
};

use dsp::segment::Segment;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use voices::Voices;

use crate::{
    synth::{self, Synthesizer},
    wav,
};

/// The lines each voice speaks: of the lengths the game's lines run to, from a word to a long
/// sentence, said, asked and shouted.
pub const LINES: [&str; 12] = [
    "Hello there, could someone please come to medbay? I need help.",
    "Yes.",
    "Get away from me!",
    "The captain wants everyone on the bridge in five minutes.",
    "Does anyone know where the toolboxes are kept?",
    "I think the engine is about to blow, we need to leave right now.",
    "Security, there is a man with a toolbox in the armory.",
    "Thank you, doctor. I feel much better now.",
    "What is going on out there?",
    "Please state the nature of your medical emergency.",
    "Cargo, we need more plasma sheets for research, as soon as you can.",
    "I am not a changeling, I swear! Ask anyone who knows me.",
];

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

/// A line measured.
struct Measured {
    line: usize,
    /// Its integrated loudness with the silence after it, or none for silence.
    loudness: Option<f64>,
    seconds: f64,
    patches: usize,
}

/// Measures each voice of `voices` that `levels` has no level of on `model`, up to `requests`
/// lines at once, setting its level and writing the table to `path` as each is done. Each line's
/// loudness goes to `log` as a line of JSON as it is measured, and each voice's first line to
/// `keep` as 16-bit WAV, named by the voice, where it is given.
///
/// # Errors
///
/// If the table or the log cannot be written.
pub fn measure(
    model: &dyn Synthesizer,
    voices: &Voices,
    levels: &mut Levels,
    path: &Path,
    requests: usize,
    log: &mut dyn Write,
    keep: Option<&Path>,
) -> io::Result<()> {
    let names: Vec<String> = (voices.names())
        .filter(|name| levels.get(name).is_none())
        .map(str::to_owned)
        .collect();
    info!(
        voices = names.len(),
        done = levels.len(),
        requests,
        "measuring"
    );
    let tail = Segment::silent(dsp::speech::TAIL, dsp::RATE).samples.len();
    let started = Instant::now();
    let work: Mutex<VecDeque<(usize, usize)>> = Mutex::new(
        (0..names.len())
            .flat_map(|voice| (0..LINES.len()).map(move |line| (voice, line)))
            .collect(),
    );
    let (measured, received) = mpsc::channel();
    thread::scope(|scope| {
        for _ in 0..requests {
            let (work, measured, names) = (&work, measured.clone(), &names);
            scope.spawn(move || {
                loop {
                    let next = work.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                    let Some((voice, line)) = next else {
                        return;
                    };
                    let name = &names[voice];
                    let said =
                        (voices.latent(name).map_err(synth::Error::from)).and_then(|latent| {
                            let text = synth::prepare(LINES[line]).expect("a line has letters");
                            model.synthesize(&text, &latent, line as u64)
                        });
                    let speech = match said {
                        Ok(speech) => speech,
                        Err(e) => {
                            warn!(voice = %name, line, "{e}");
                            let _ = measured.send((voice, None));
                            continue;
                        }
                    };
                    if let (Some(keep), 0) = (keep, line) {
                        let file = keep.join(format!("{}.wav", file_name(name)));
                        let pcm = synth::to_segment(&speech.samples);
                        if let Err(e) =
                            fs::write(&file, wav::write_pcm16(pcm.rate, 1, &pcm.samples))
                        {
                            warn!("{}: {e}", file.display());
                        }
                    }
                    let mut samples: Vec<f64> =
                        speech.samples.iter().map(|&s| f64::from(s)).collect();
                    samples.resize(samples.len() + tail, 0.0);
                    let loudness = dsp::loudness::integrated(&samples, dsp::RATE);
                    let _ = measured.send((
                        voice,
                        Some(Measured {
                            line,
                            loudness: loudness.is_finite().then_some(loudness),
                            seconds: speech.samples.len() as f64 / f64::from(dsp::RATE),
                            patches: speech.patches,
                        }),
                    ));
                }
            });
        }
        drop(measured);
        // Each voice's lines as they come, until it has them all.
        let mut said: BTreeMap<usize, Vec<Option<f64>>> = BTreeMap::new();
        let mut done = 0;
        for (voice, measured) in received {
            let loudness = match measured {
                Some(m) => {
                    let line = serde_json::json!({
                        "voice": names[voice],
                        "line": m.line,
                        "loudness_lufs": m.loudness,
                        "seconds": m.seconds,
                        "patches": m.patches,
                    });
                    writeln!(log, "{line}")?;
                    m.loudness
                }
                None => None,
            };
            let lines = said.entry(voice).or_default();
            lines.push(loudness);
            if lines.len() < LINES.len() {
                continue;
            }
            let loud: Vec<f64> = lines.iter().flatten().copied().collect();
            said.remove(&voice);
            let name = &names[voice];
            done += 1;
            let Some(median) = median(&loud) else {
                warn!(voice = %name, "said nothing to measure");
                continue;
            };
            let level = Level {
                loudness_lufs: median,
                lines: loud.len(),
            };
            levels.set(name, level);
            levels.write(path)?;
            info!(
                voice = %name,
                loudness_lufs = median,
                lines = loud.len(),
                done,
                left = names.len() - done,
                seconds = started.elapsed().as_secs_f64(),
                "measured"
            );
        }
        Ok(())
    })
}

/// `name` as a file's name: letters, digits, `-` and `_` kept, anything else `_`.
fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        f32::consts::TAU,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use voices::Latent;

    use super::*;
    use crate::synth::Speech;

    /// A model that says a tone as long as the text, as loud as its voice's reference text says,
    /// and fails on a voice whose reference text is not a number.
    #[derive(Default)]
    struct Tone {
        asked: AtomicUsize,
        at_once: AtomicUsize,
        most: AtomicUsize,
    }

    impl Synthesizer for Tone {
        fn synthesize(&self, text: &str, voice: &Latent, _: u64) -> Result<Speech, synth::Error> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            let at_once = self.at_once.fetch_add(1, Ordering::SeqCst) + 1;
            self.most.fetch_max(at_once, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(5));
            self.at_once.fetch_sub(1, Ordering::SeqCst);
            let amplitude: f32 = voice.text.parse()?;
            let samples = (0..1200 * text.len())
                .map(|i| amplitude * (TAU * 220.0 * i as f32 / 48000.0).sin())
                .collect();
            Ok(Speech {
                samples,
                patches: text.len(),
                start: None,
            })
        }
    }

    /// Voices whose model says tones at `amplitudes`, by name.
    fn voices(dir: &Path, amplitudes: &[(&str, &str)]) -> Voices {
        let table = (amplitudes.iter())
            .map(|(name, _)| (name.to_string(), name.to_lowercase()))
            .collect();
        voices::write_table(dir, &table).unwrap();
        fs::create_dir_all(dir.join("latents")).unwrap();
        for (name, amplitude) in amplitudes {
            let latent = Latent {
                frames: vec![[0.0; voices::LATENT_DIM]; 4],
                text: amplitude.to_string(),
            };
            let path = voices::latent_path(dir, &name.to_lowercase());
            latent.write(&path).unwrap();
        }
        Voices::open(dir).unwrap()
    }

    #[test]
    fn measures_each_voice_once_as_the_median_of_its_lines() {
        let dir = tempfile::tempdir().unwrap();
        let amplitudes = [("Loud", "0.5"), ("Quiet", "0.125"), ("Broken", "loud")];
        let voices = voices(dir.path(), &amplitudes);
        let (path, keep) = (dir.path().join("levels.json"), dir.path().join("kept"));
        fs::create_dir_all(&keep).unwrap();
        let (tone, mut levels, mut log) = (Tone::default(), Levels::default(), Vec::new());
        measure(&tone, &voices, &mut levels, &path, 4, &mut log, Some(&keep)).unwrap();
        assert_eq!(tone.asked.load(Ordering::SeqCst), 3 * LINES.len());
        assert!(tone.most.load(Ordering::SeqCst) > 1, "one line at a time");
        let read = Levels::read(&path).unwrap();
        assert_eq!(read.len(), levels.len());
        for (voice, level) in &levels.voices {
            let read = read.get(voice).unwrap();
            assert!(
                (read.loudness_lufs - level.loudness_lufs).abs() < 1e-9,
                "{voice}"
            );
        }
        // Loudness goes with the level of a signal decibel for decibel, the gates with it.
        let (loud, quiet) = (levels.get("Loud").unwrap(), levels.get("Quiet").unwrap());
        assert_eq!((loud.lines, quiet.lines), (LINES.len(), LINES.len()));
        let apart = loud.loudness_lufs - quiet.loudness_lufs;
        assert!(
            (apart - 20.0 * 4f64.log10()).abs() < 1e-9,
            "{apart} dB apart"
        );
        assert_eq!(levels.get("Broken"), None);
        // The level is the median of the lines logged.
        let logged: Vec<f64> = (String::from_utf8(log).unwrap().lines())
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|line| line["voice"] == "Loud")
            .map(|line| line["loudness_lufs"].as_f64().unwrap())
            .collect();
        assert_eq!(median(&logged), Some(loud.loudness_lufs));
        let kept = crate::wav::read(&fs::read(keep.join("Loud.wav")).unwrap()).unwrap();
        assert_eq!(
            kept.frames(),
            1200 * synth::prepare(LINES[0]).unwrap().len()
        );

        // A run that finds a table goes on with the voices it does not have.
        let mut resumed = Levels::default();
        resumed.set("Loud", loud);
        let tone = Tone::default();
        measure(
            &tone,
            &voices,
            &mut resumed,
            &path,
            4,
            &mut Vec::new(),
            None,
        )
        .unwrap();
        assert_eq!(tone.asked.load(Ordering::SeqCst), 2 * LINES.len());
        assert_eq!(resumed, levels);
    }

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
