//! How loud each voice comes out of the model, which streamed speech is brought to the service's
//! loudness by: a gain known before the speech is, where the whole-file routes measure each line
//! once they have all of it.
//!
//! A voice's level is the median integrated loudness of the same lines spoken in it, as the
//! model gave them, before anything levels them, each with the silence the old frontend put
//! after it, as the whole-file routes measure a line: on a short line that silence moves the
//! measure by decibels. [`Measure`] measures them, a voice at a time, for `tgtts-gains`, which
//! writes them as JSON: `{"Man (Ace Detective)": {"loudness_lufs": -38.2, "lines": 12}, ...}`,
//! with the peaks the radio scales the voice's lines by as they stream, which a radio on a stream
//! cannot measure.

use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    io::{self, Write},
    path::Path,
    sync::{Mutex, mpsc},
    thread,
    time::Instant,
};

use dsp::{
    quantity::{self, Gain, QuarterTones, Ratio},
    radio,
    segment::Segment,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use voices::Voices;

use crate::{
    service::Loudness,
    synth::{self, Synthesizer},
    wav::{self, Audio},
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

/// How much more than its voice's gain the whole-file path may raise a line by, in dB: 7189 of
/// the 7197 lines 600 voices said in a gains run came within 12 dB under their voice's level.
pub const MARGIN: Gain = Gain::new(12.0);

/// The most the whole-file path raises speech in a voice without a level by, in dB: more than
/// 589 of those 600 voices' levels allow with [`MARGIN`].
pub const MOST_GAIN: Gain = Gain::new(24.0);

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
    /// The peaks the radio scales its lines by, as they stream, where they were measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radio: Option<RadioPeaks>,
}

/// The medians of the peaks the radio scales a voice's lines by, as [`radio::Peaks`] has them,
/// of each line brought to the service's loudness by the voice's level, as it streams: what the
/// radio on a stream scales by, which cannot measure them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadioPeaks {
    pub narrowed: f64,
    pub driven: f64,
    pub output: f64,
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
    pub fn loudness(&self, voice: &str) -> Option<quantity::Loudness> {
        if let Some(level) = self.voices.get(voice) {
            return Some(quantity::Loudness::new(level.loudness_lufs));
        }
        let all: Vec<f64> = self.voices.values().map(|l| l.loudness_lufs).collect();
        median(&all).map(quantity::Loudness::new)
    }

    /// The most the whole-file path raises speech in `voice` by to bring it to `target`: the gain
    /// that brings the voice's level there and [`MARGIN`], or for a voice without a level,
    /// [`MOST_GAIN`].
    pub fn most_gain(&self, voice: &str, target: quantity::Loudness) -> Gain {
        (self.voices.get(voice)).map_or(MOST_GAIN, |level| {
            Gain::between(quantity::Loudness::new(level.loudness_lufs), target) + MARGIN
        })
    }

    /// The peaks the radio scales `voice`'s streamed speech by, or for a voice without them the
    /// medians of the voices', or none where no voice has them.
    pub fn radio(&self, voice: &str) -> Option<radio::Peaks> {
        let peaks = match self.voices.get(voice).and_then(|l| l.radio) {
            Some(peaks) => peaks,
            None => {
                let all: Vec<RadioPeaks> = self.voices.values().filter_map(|l| l.radio).collect();
                let of = |peak: fn(&RadioPeaks) -> f64| {
                    median(&all.iter().map(peak).collect::<Vec<_>>())
                };
                RadioPeaks {
                    narrowed: of(|p| p.narrowed)?,
                    driven: of(|p| p.driven)?,
                    output: of(|p| p.output)?,
                }
            }
        };
        Some(radio::Peaks {
            narrowed: peaks.narrowed as f32,
            driven: peaks.driven,
            output: peaks.output,
        })
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

/// The `q`th quantile of `values`, interpolated linearly between the two nearest of them in order,
/// as numpy's `quantile` has it by default, or none of none.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len().checked_sub(1)?;
    let at = q.clamp(0.0, 1.0) * last as f64;
    let (below, above) = (at.floor() as usize, at.ceil() as usize);
    Some(sorted[below] + (sorted[above] - sorted[below]) * (at - below as f64))
}

/// What the lines [`Measure::run`] logs say, for reading: of each measure, its median, its 10th
/// and 90th percentiles, and its least and most, over the lines that have it.
///
/// # Errors
///
/// If a line is not JSON.
pub fn report(log: &str) -> Result<String, serde_json::Error> {
    let lines: Vec<serde_json::Value> = (log.lines())
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let of = |field: &str| -> Vec<f64> {
        lines
            .iter()
            .filter_map(|line| line[field].as_f64())
            .collect()
    };
    let mut report = String::new();
    for (field, what) in [
        ("loudness_lufs", "the model's loudness, LUFS"),
        ("streamed_lufs", "the speech as it streams, LUFS"),
        (
            "radio_level_db",
            "the streamed radio's loudness less today's, dB",
        ),
        (
            "radio_drive_db",
            "the streamed radio's drive less today's, dB",
        ),
        ("seconds", "the model's speech, seconds"),
        ("patches", "its patches"),
    ] {
        let values = of(field);
        let at = |q: f64| quantile(&values, q).map_or("-".to_owned(), |v| format!("{v:.2}"));
        report += &format!(
            "{what} over {} lines: median {}, p10 {}, p90 {}, min {}, max {}\n",
            values.len(),
            at(0.5),
            at(0.1),
            at(0.9),
            at(0.0),
            at(1.0)
        );
    }
    let clipped = of("radio_clipped");
    let most = clipped.iter().copied().fold(0.0, f64::max);
    report += &format!(
        "streamed radio lines with a clipped sample: {} of {}, at most {most} samples\n",
        clipped.iter().filter(|&&n| n > 0.0).count(),
        clipped.len()
    );
    Ok(report)
}

/// A line measured: the model's samples and their loudness with the silence after them, or none
/// for silence; and the loudness of today's radio of it, as the whole-file path makes it.
struct Measured {
    line: usize,
    samples: Vec<f32>,
    loudness: Option<quantity::Loudness>,
    patches: usize,
    radio_lufs: quantity::Loudness,
}

/// What [`Measure::run`] measures with.
pub struct Measure<'a> {
    pub model: &'a dyn Synthesizer,
    pub voices: &'a Voices,
    /// The lines asked for at once.
    pub requests: usize,
    /// The loudness the service brings speech to.
    pub loudness: Loudness,
    /// The radio's clicks and static.
    pub radio: &'a radio::Assets,
    /// Where each voice's first line is kept as 16-bit WAV, named by the voice, if anywhere.
    pub keep: Option<&'a Path>,
    /// When to stop asking for lines, if ever: the lines asked for are measured, and a voice
    /// left short is measured again by the next run.
    pub until: Option<Instant>,
}

impl Measure<'_> {
    /// Measures each voice that `levels` has no level of, setting its level and writing the
    /// table to `path` as each is done.
    ///
    /// A voice's lines are measured as the model says them, then once it has said them all,
    /// brought to the service's loudness by its level, as they stream, and put on the radio, at
    /// the medians of their peaks, as the radio streams; each line then goes to `log` as a line
    /// of JSON: its loudness, how loud it streams, where the whole-file path brings each line to
    /// the target, and how far the streamed radio of it is from today's, in loudness and in how
    /// hard it is driven.
    ///
    /// # Errors
    ///
    /// If the table or the log cannot be written.
    pub fn run(&self, levels: &mut Levels, path: &Path, log: &mut dyn Write) -> io::Result<()> {
        let names: Vec<String> = (self.voices.names())
            .filter(|name| levels.get(name).is_none())
            .map(str::to_owned)
            .collect();
        info!(
            voices = names.len(),
            done = levels.len(),
            requests = self.requests,
            "measuring"
        );
        let started = Instant::now();
        let work: Mutex<VecDeque<(usize, usize)>> = Mutex::new(
            (0..names.len())
                .flat_map(|voice| (0..LINES.len()).map(move |line| (voice, line)))
                .collect(),
        );
        let (measured, received) = mpsc::channel();
        thread::scope(|scope| {
            for _ in 0..self.requests {
                let (work, measured, names) = (&work, measured.clone(), &names);
                scope.spawn(move || {
                    loop {
                        if self.until.is_some_and(|until| Instant::now() >= until) {
                            return;
                        }
                        let next = work.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                        let Some((voice, line)) = next else {
                            return;
                        };
                        let _ = measured.send((voice, self.line(&names[voice], line)));
                    }
                });
            }
            drop(measured);
            // Each voice's lines as they come, until it has them all.
            let mut said: BTreeMap<usize, Vec<Option<Measured>>> = BTreeMap::new();
            let mut done = 0;
            for (voice, measured) in received {
                let lines = said.entry(voice).or_default();
                lines.push(measured);
                if lines.len() < LINES.len() {
                    continue;
                }
                let lines = said.remove(&voice).expect("the voice's lines");
                let name = &names[voice];
                done += 1;
                let Some(level) = self.level(name, lines, log)? else {
                    warn!(voice = %name, "said nothing to measure");
                    continue;
                };
                levels.set(name, level);
                levels.write(path)?;
                info!(
                    voice = %name,
                    loudness_lufs = level.loudness_lufs,
                    lines = level.lines,
                    done,
                    left = names.len() - done,
                    seconds = started.elapsed().as_secs_f64(),
                    "measured"
                );
            }
            Ok(())
        })
    }

    /// Line `line` in `voice`, measured, or none where the model could not say it.
    fn line(&self, voice: &str, line: usize) -> Option<Measured> {
        let said = (self.voices.latent(voice).map_err(synth::Error::from)).and_then(|latent| {
            let text = synth::prepare(LINES[line]).expect("a line has letters");
            self.model.synthesize(&text, &latent, line as u64)
        });
        let speech = match said {
            Ok(speech) => speech,
            Err(e) => {
                warn!(voice, line, "{e}");
                return None;
            }
        };
        let pcm = synth::to_segment(&speech.samples);
        if let (Some(keep), 0) = (self.keep, line) {
            let file = keep.join(format!("{}.wav", file_name(voice)));
            if let Err(e) = fs::write(&file, wav::write_pcm16(pcm.rate, 1, &pcm.samples)) {
                warn!("{}: {e}", file.display());
            }
        }
        let mut samples: Vec<f64> = speech.samples.iter().map(|&s| f64::from(s)).collect();
        samples.resize(samples.len() + tail(), 0.0);
        let loudness = dsp::loudness::integrated(&samples, dsp::RATE);
        // Today's: the speech leveled, finished and brought to the loudness whole, then put on
        // the radio whole.
        let finished = dsp::speech::finish(&synth::level(&pcm), QuarterTones::ZERO);
        let (target, ceiling) = (self.loudness.target, self.loudness.ceiling);
        let whole = dsp::loudness::normalize(&finished, target, ceiling, MOST_GAIN);
        let radio = radio::radio(&mono(&whole.segment.samples), self.radio);
        Some(Measured {
            line,
            samples: speech.samples,
            loudness: loudness.value().is_finite().then_some(loudness),
            patches: speech.patches,
            radio_lufs: pcm_lufs(&radio.samples),
        })
    }

    /// The level of a voice from its `lines`, logged, or none where it said nothing.
    fn level(
        &self,
        voice: &str,
        lines: Vec<Option<Measured>>,
        log: &mut dyn Write,
    ) -> io::Result<Option<Level>> {
        let mut lines: Vec<Measured> = lines.into_iter().flatten().collect();
        lines.retain(|m| m.loudness.is_some());
        lines.sort_by_key(|m| m.line);
        let loud: Vec<f64> = (lines.iter())
            .filter_map(|m| m.loudness)
            .map(quantity::Loudness::value)
            .collect();
        let Some(loudness) = median(&loud).map(quantity::Loudness::new) else {
            return Ok(None);
        };
        // Each line as it streams: at the voice's gain and under the limiter.
        let gain = Ratio::of(Gain::between(loudness, self.loudness.target)).value();
        let streamed: Vec<Vec<i16>> = (lines.iter())
            .map(|m| {
                let mut x: Vec<f64> = m.samples.iter().map(|&s| f64::from(s) * gain).collect();
                x.resize(x.len() + tail(), 0.0);
                let limited = dsp::limiter::limit(&x, dsp::RATE, self.loudness.ceiling).samples;
                let samples = limited.iter().map(|&s| s as f32).collect();
                Audio {
                    rate: dsp::RATE,
                    channels: 1,
                    samples,
                }
                .to_pcm()
                .samples
            })
            .collect();
        let peaks: Vec<radio::Peaks> = (streamed.iter())
            .map(|pcm| radio::peaks(&mono(pcm), self.radio))
            .collect();
        let of =
            |peak: fn(&radio::Peaks) -> f64| median(&peaks.iter().map(peak).collect::<Vec<_>>());
        let radio = RadioPeaks {
            narrowed: of(|p| f64::from(p.narrowed)).expect("a line"),
            driven: of(|p| p.driven).expect("a line"),
            output: of(|p| p.output).expect("a line"),
        };
        let fixed = radio::Peaks {
            narrowed: radio.narrowed as f32,
            driven: radio.driven,
            output: radio.output,
        };
        for ((m, pcm), peaks) in lines.iter().zip(&streamed).zip(&peaks) {
            let mut stream = radio::Stream::new(self.radio, fixed, dsp::RATE, 1);
            let mut out = Vec::new();
            stream.process(pcm, &mut out);
            stream.finish(&mut out);
            let line = serde_json::json!({
                "voice": voice,
                "line": m.line,
                "loudness_lufs": m.loudness.map(quantity::Loudness::value),
                "seconds": m.samples.len() as f64 / f64::from(dsp::RATE),
                "patches": m.patches,
                "streamed_lufs": pcm_lufs(pcm).value(),
                "radio_level_db": Gain::between(m.radio_lufs, pcm_lufs(&out)).value(),
                "radio_drive_db": 20.0 * (f64::from(peaks.narrowed) / radio.narrowed).log10(),
                "radio_clipped": out.iter().filter(|&&s| s == i16::MAX || s <= -i16::MAX).count(),
            });
            writeln!(log, "{line}")?;
        }
        Ok(Some(Level {
            loudness_lufs: loudness.value(),
            lines: loud.len(),
            radio: Some(radio),
        }))
    }
}

/// The silence the old frontend put after speech, in samples.
fn tail() -> usize {
    Segment::silent(dsp::speech::TAIL, dsp::RATE).samples.len()
}

fn mono(samples: &[i16]) -> radio::Pcm {
    radio::Pcm {
        rate: dsp::RATE,
        channels: 1,
        samples: samples.to_vec(),
    }
}

/// The integrated loudness of 16-bit `samples`.
fn pcm_lufs(samples: &[i16]) -> quantity::Loudness {
    let x: Vec<f64> = samples.iter().map(|&s| f64::from(s) / 32768.0).collect();
    dsp::loudness::integrated(&x, dsp::RATE)
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
            let samples = (0..480 * text.len())
                .map(|i| amplitude * (TAU * 220.0 * i as f32 / 48000.0).sin())
                .collect();
            Ok(Speech {
                samples,
                patches: text.len(),
                start: None,
            })
        }
    }

    /// Clicks and static at the service's rate.
    fn radio() -> radio::Assets {
        let pcm = |frames: usize, amplitude: f64| radio::Pcm {
            rate: dsp::RATE,
            channels: 1,
            samples: (0..frames)
                .map(|i| ((i as f64 * 0.37).sin() * amplitude * 32767.0) as i16)
                .collect(),
        };
        radio::Assets::new(&pcm(2400, 0.1), &pcm(2400, 0.05), &pcm(48000, 0.4))
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
        let radio = radio();
        let measure = |model| Measure {
            model,
            voices: &voices,
            requests: 4,
            loudness: Loudness {
                target: quantity::Loudness::new(-16.0),
                ceiling: dsp::loudness::CEILING,
            },
            radio: &radio,
            keep: Some(&keep),
            until: None,
        };
        measure(&tone).run(&mut levels, &path, &mut log).unwrap();
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
        // Brought to the loudness by their levels, the two voices stream the same speech, which
        // the radio scales by the same peaks.
        let (a, b) = (loud.radio.unwrap(), quiet.radio.unwrap());
        for (a, b) in [
            (a.narrowed, b.narrowed),
            (a.driven, b.driven),
            (a.output, b.output),
        ] {
            assert!(a > 0.0 && (a - b).abs() < 1e-6, "{a} and {b}");
        }
        // The level is the median of the lines logged, each with its radio's.
        let logged: Vec<serde_json::Value> = (String::from_utf8(log).unwrap().lines())
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|line: &serde_json::Value| line["voice"] == "Loud")
            .collect();
        let loudness: Vec<f64> = (logged.iter())
            .map(|line| line["loudness_lufs"].as_f64().unwrap())
            .collect();
        assert_eq!(median(&loudness), Some(loud.loudness_lufs));
        assert!(logged.iter().all(|line| line["radio_level_db"].is_f64()));
        // Under the limiter, a line streams as loud as it is plus its voice's gain, so the
        // median line at the target.
        let streamed: Vec<f64> = (logged.iter())
            .map(|line| line["streamed_lufs"].as_f64().unwrap())
            .collect();
        let streamed = median(&streamed).unwrap();
        assert!((streamed + 16.0).abs() < 0.01, "{streamed} LUFS");
        let kept = crate::wav::read(&fs::read(keep.join("Loud.wav")).unwrap()).unwrap();
        assert_eq!(kept.frames(), 480 * synth::prepare(LINES[0]).unwrap().len());

        // A run out of time asks for nothing more, and sets no level of a voice left short.
        let tone = Tone::default();
        let mut late = Levels::default();
        let out_of_time = Measure {
            until: Some(Instant::now()),
            ..measure(&tone)
        };
        out_of_time
            .run(&mut late, &dir.path().join("late.json"), &mut Vec::new())
            .unwrap();
        assert_eq!((tone.asked.load(Ordering::SeqCst), late.len()), (0, 0));

        // A run that finds a table goes on with the voices it does not have.
        let mut resumed = Levels::default();
        resumed.set("Loud", loud);
        let tone = Tone::default();
        measure(&tone)
            .run(&mut resumed, &path, &mut Vec::new())
            .unwrap();
        assert_eq!(tone.asked.load(Ordering::SeqCst), 2 * LINES.len());
        assert_eq!(resumed, levels);
    }

    /// The quantiles of a known distribution, 1 to 101 dB down in steps of 1 dB and out of
    /// order, with as many digits as text sorts apart from numbers.
    #[test]
    fn reports_the_quantiles_of_the_lines() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[3.0, 1.0, 2.0, 4.0], 0.5), Some(2.5));
        assert_eq!(quantile(&[3.0, 1.0, 2.0, 4.0], 0.1), Some(1.3));
        let log: String = (0..101)
            .map(|i| (i * 37) % 101 + 1)
            .map(|db| {
                let line = serde_json::json!({
                    "loudness_lufs": -f64::from(db),
                    "radio_clipped": usize::from(db % 25 == 0) * 7,
                });
                format!("{line}\n")
            })
            .collect();
        let report = report(&log).unwrap();
        let first = report.lines().next().unwrap();
        assert_eq!(
            first,
            "the model's loudness, LUFS over 101 lines: median -51.00, p10 -91.00, p90 -11.00, \
             min -101.00, max -1.00"
        );
        assert!(report.contains("the model's speech, seconds over 0 lines: median -,"));
        assert!(report.ends_with("a clipped sample: 4 of 101, at most 7 samples\n"));
        assert!(super::report("not JSON").is_err());
    }

    #[test]
    fn a_voice_without_a_level_gets_the_median_of_theirs() {
        let mut levels = Levels::default();
        assert_eq!(levels.loudness("A"), None);
        for (voice, loudness) in [("A", -40.0), ("B", -30.0), ("C", -32.0)] {
            let level = Level {
                loudness_lufs: loudness,
                lines: 12,
                radio: None,
            };
            levels.set(voice, level);
        }
        assert_eq!(levels.loudness("B"), Some(quantity::Loudness::new(-30.0)));
        assert_eq!(levels.loudness("Z"), Some(quantity::Loudness::new(-32.0)));
        assert_eq!(median(&[1.0, 4.0, 2.0, 3.0]), Some(2.5));
    }

    #[test]
    fn the_whole_file_path_raises_a_voice_by_its_gain_and_the_margin_at_most() {
        let mut levels = Levels::default();
        let level = Level {
            loudness_lufs: -30.0,
            lines: 12,
            radio: None,
        };
        levels.set("A", level);
        let target = quantity::Loudness::new(-16.0);
        assert_eq!(levels.most_gain("A", target), Gain::new(14.0) + MARGIN);
        assert_eq!(levels.most_gain("B", target), MOST_GAIN);
    }

    #[test]
    fn reads_what_it_writes_and_refuses_what_is_not_a_level() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("levels.json");
        let mut levels = Levels::default();
        let level = Level {
            loudness_lufs: -38.25,
            lines: 3,
            radio: None,
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
