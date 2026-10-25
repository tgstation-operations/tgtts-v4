//! The blips: a voice's recorded syllables strung together, one per other letter of the text, in
//! place of speech.
//!
//! Each blip is detuned at random, lowered in volume and faded, then crossfaded onto the last, by
//! 150 ms or as much of that as both have, and the whole brought to -25 dBFS with its peak held
//! under -5 dBFS. [`sentence`] is what the old
//! blips service made of the text it was sent; [`message`] is what its frontend made of a message,
//! a sentence at a time.

use std::{collections::HashMap, error, fmt};

use voices::Blips;

use crate::{
    Random,
    float::{from_pcm, to_pcm},
    pitch::pitch_shift,
    segment::{self, Segment},
    sentences,
};

/// How far blips overlap, in ms.
const CROSSFADE: i64 = 150;
/// The silence after each sentence of a message, in ms.
const SENTENCE_GAP: u32 = 250;

/// What made the blips fail, where the old service failed the request.
#[derive(Debug)]
pub enum Error {
    /// The voice has no sample the text needs, or it could not be read.
    Voice(voices::Error),
    /// pydub would have refused an operation on the samples.
    Segment(segment::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Voice(e) => e.fmt(f),
            Self::Segment(e) => e.fmt(f),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Voice(e) => Some(e),
            Self::Segment(e) => Some(e),
        }
    }
}

impl From<voices::Error> for Error {
    fn from(e: voices::Error) -> Self {
        Self::Voice(e)
    }
}

impl From<segment::Error> for Error {
    fn from(e: segment::Error) -> Self {
        Self::Segment(e)
    }
}

/// The blips of `text` in the voice whose samples `blips` holds, at its `base` and `number`,
/// `pitch` quarter tones up, as the old frontend made them: the blips of each of pysbd's
/// sentences that makes a sound, each followed by 250 ms of silence.
pub fn message(
    text: &str,
    blips: &Blips,
    base: &str,
    number: &str,
    pitch: i32,
    random: &mut impl Random,
) -> Result<Segment, Error> {
    let mut samples = Vec::new();
    for text in sentences::segment(text) {
        let sound = sentence(&text, blips, base, number, pitch, random)?;
        if !sound.samples.is_empty() {
            samples.extend(sound.samples);
            samples.extend(Segment::silent(SENTENCE_GAP, blips.sample_rate()).samples);
        }
    }
    Ok(Segment::new(blips.sample_rate(), samples))
}

/// The blips of `text` as the old blips service made them, `pitch` quarter tones up.
///
/// Text with no letter or digit makes no sound. Each letter or digit at an even index
/// makes the blip of its lowercase, detuned by up to 0.6 semitones either way, a capital by 1.6
/// semitones more and up to 1.05 either way, and lowered by 6 dB, a capital by 3; `?` and `!`
/// make the `Deska` and `Gwah` blips, detuned by up to 1.2 semitones, lowered by 4.4 dB and
/// stripped of their silence; nothing else makes a sound. Each blip is faded in over 3 ms and out
/// over 8. Each detuning is a draw from `random`.
pub fn sentence(
    text: &str,
    blips: &Blips,
    base: &str,
    number: &str,
    pitch: i32,
    random: &mut impl Random,
) -> Result<Segment, Error> {
    let rate = blips.sample_rate();
    if !text.contains(char::is_alphanumeric) {
        return Ok(Segment::new(rate, Vec::new()));
    }
    let mut samples = HashMap::new();
    let mut sample = |key: String| -> Result<Vec<f32>, Error> {
        if !samples.contains_key(&key) {
            let pcm = blips.sample(base, number, &key)?;
            samples.insert(key.clone(), from_pcm(&pcm));
        }
        Ok(samples[&key].clone())
    };
    let mut result = Segment::new(rate, Vec::new());
    for (i, letter) in text.chars().enumerate() {
        let blip = if letter.is_alphanumeric() {
            if i % 2 != 0 {
                continue;
            }
            let upper = letter.is_uppercase();
            let (base_pitch, random_pitch) = if upper { (1.6, 0.15) } else { (0.0, 0.0) };
            let mut blip = sample(letter.to_lowercase().collect())?;
            blip = detune(blip, rate, base_pitch, 0.2 + random_pitch, random);
            if pitch != 0 {
                blip = pitch_shift(&blip, rate, f64::from(pitch), 24);
            }
            let blip = change_volume(
                &Segment::new(rate, to_pcm(&blip)),
                if upper { 0.7 } else { 0.5 },
            );
            // The old service also strips this blip of its silence, then uses it unstripped.
            fade(&blip)?
        } else if letter == '?' || letter == '!' {
            let blip = sample(if letter == '?' { "Deska" } else { "Gwah" }.to_owned())?;
            let blip = detune(blip, rate, 0.0, 0.2 + 0.2, random);
            fade(&change_volume(&Segment::new(rate, to_pcm(&blip)), 0.6).strip_silence()?)?
        } else {
            continue;
        };
        let crossfade = CROSSFADE.min(result.len_ms()).min(blip.len_ms());
        result = result.append(&blip, crossfade)?;
    }
    let result = result.apply_gain(-25.0 - result.dbfs());
    let peak = result.max_dbfs();
    Ok(if peak > -5.0 {
        result.apply_gain(-5.0 - peak)
    } else {
        result
    })
}

/// The old service's detuning: `base_pitch` semitones, and a draw of up to `3 * spread`
/// semitones either way.
fn detune(
    blip: Vec<f32>,
    rate: u32,
    base_pitch: f64,
    spread: f64,
    random: &mut impl Random,
) -> Vec<f32> {
    let cents = (0.0 + base_pitch) * 100.0 + (random.random() * (300.0 + 300.0) - 300.0) * spread;
    let semitones = cents / 100.0;
    if semitones == 0.0 {
        blip
    } else {
        pitch_shift(&blip, rate, semitones, 12)
    }
}

/// `blip` faded in over 3 ms and out over 8, or over all of it where it is shorter: pydub's fade
/// out of a shorter segment repeats its first frames.
fn fade(blip: &Segment) -> Result<Segment, Error> {
    let len = blip.len_ms();
    if len == 0 {
        return Ok(blip.clone());
    }
    Ok(blip.fade_in(3.min(len))?.fade_out(8.min(len))?)
}

/// The old service's `change_volume`: a gain of the amplitude ratio `multiplier`, through dB.
fn change_volume(segment: &Segment, multiplier: f64) -> Segment {
    segment.apply_gain(20.0 * multiplier.log10())
}
