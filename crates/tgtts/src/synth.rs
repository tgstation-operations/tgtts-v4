//! Speech from the model, and what the old model service did to the text before it and to the
//! audio after it.

use std::{error, time::Duration};

use dsp::{
    RATE,
    loudness::FLOOR,
    quantity::{Gain, QuarterTones},
    segment::Segment,
};
use voices::Latent;

/// An error from a synthesizer, which the service reports and fails the request on.
pub type Error = Box<dyn error::Error + Send + Sync>;

/// Where a stream's audio goes, a run of samples at a time.
pub type Sink<'a> = dyn FnMut(&[f32]) + 'a;

/// A text-to-speech model, speaking at the service's rate.
pub trait Synthesizer: Send + Sync + 'static {
    /// `text` spoken in `voice`, drawn from `seed`.
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, Error>;

    /// The same speech, its samples given to `audio` as they are made, a run at a time, in
    /// order, all of them before it comes back whole. A model that cannot give them sooner
    /// gives them all at the end.
    fn stream(
        &self,
        text: &str,
        voice: &Latent,
        seed: u64,
        audio: &mut Sink<'_>,
    ) -> Result<Speech, Error> {
        let speech = self.synthesize(text, voice, seed)?;
        audio(&speech.samples);
        Ok(speech)
    }
}

/// What a synthesizer said.
#[derive(Clone, Debug, PartialEq)]
pub struct Speech {
    /// Mono samples at [`RATE`], full scale at 1.
    pub samples: Vec<f32>,
    /// How many patches the model generated for them.
    pub patches: usize,
    /// How the model came to their first patch, where it says.
    pub start: Option<Start>,
}

/// How a synthesizer came to a speech's first patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Start {
    /// From when it was asked for the speech.
    pub waited: Duration,
    /// The positions of the prompt.
    pub prompt: usize,
    /// Those that ran through the model, the rest coming from what it kept of the voice.
    pub prefilled: usize,
}

/// The text the old model service gave the model for `text`, or `None` for text it made silence
/// of instead: text with no letter or digit.
///
/// It puts `". "` after text that does not end in `.`, `?` or `!`, trailing space and all, and
/// capitalizes a lowercase first letter. Its own check saw only capitals and digits, so it made
/// silence of text in lowercase.
pub fn prepare(text: &str) -> Option<String> {
    if !text.contains(|c: char| c.is_ascii_alphanumeric()) {
        return None;
    }
    let mut text = text.to_owned();
    if !text.ends_with(['.', '?', '!']) {
        text.push_str(". ");
    }
    let mut chars = text.chars();
    let first = chars.next().expect("the text has a letter or digit");
    if first.is_alphabetic() && !first.is_uppercase() {
        text = first.to_uppercase().chain(chars).collect();
    }
    Some(text)
}

/// `samples` as 16-bit PCM, as the old model service wrote them with soundfile:
/// libsndfile scales by 32767 and rounds in single precision. The audio VAE ends in a tanh, so
/// that nothing reaches full scale; past it a sample is clipped.
pub fn to_segment(samples: &[f32]) -> Segment {
    let pcm = samples
        .iter()
        .map(|&s| (s * 32767.0).round_ties_even() as i16)
        .collect();
    Segment::new(RATE, pcm)
}

/// The level the model's speech is brought to first: its peak at -0.1 dBFS, as pydub's
/// `effects.normalize` at its default headroom brought it in the old model service, so that 16
/// bits hold as much of it as they can; the service brings it to its loudness after.
/// That service also computed a -25 dBFS loudness and a -5 dBFS cap, then threw both away.
///
/// Speech quieter than [`FLOOR`] with the silence after it is kept as it is, which the service
/// serves as the silence it nearly is: leveled, its loudest click would be at full scale.
pub fn level(speech: &Segment) -> Segment {
    let finished = dsp::speech::finish(speech, QuarterTones::ZERO);
    let samples: Vec<f64> = (finished.samples.iter())
        .map(|&s| f64::from(s) / 32768.0)
        .collect();
    if dsp::loudness::integrated(&samples, speech.rate) < FLOOR {
        return speech.clone();
    }
    speech.normalize(Gain::new(0.1))
}

#[cfg(test)]
mod tests {
    use dsp::quantity::SampleRate;

    use super::*;

    #[test]
    fn prepares_text_as_the_old_service() {
        assert_eq!(prepare("hello there").as_deref(), Some("Hello there. "));
        assert_eq!(prepare("Hello there!").as_deref(), Some("Hello there!"));
        assert_eq!(prepare("what?").as_deref(), Some("What?"));
        assert_eq!(prepare("4 cats.").as_deref(), Some("4 cats."));
        assert_eq!(prepare("'quoted'").as_deref(), Some("'quoted'. "));
        assert_eq!(prepare("hi ").as_deref(), Some("Hi . "));
    }

    #[test]
    fn makes_silence_of_text_without_letters_or_digits() {
        assert_eq!(prepare(""), None);
        assert_eq!(prepare("...?! ,-'&"), None);
        assert_eq!(prepare("x").as_deref(), Some("X. "));
        assert_eq!(prepare("7").as_deref(), Some("7. "));
    }

    #[test]
    fn writes_pcm_as_libsndfile() {
        let segment = to_segment(&[0.0, 1.0, -1.0, 0.5, 1.5 / 32767.0, 2.5 / 32767.0, 2.0]);
        assert_eq!(segment.rate, RATE);
        assert_eq!(segment.samples, [0, 32767, -32767, 16384, 2, 2, 32767]);
    }

    #[test]
    fn levels_the_peak_at_a_tenth_of_a_decibel_down() {
        let speech = Segment::new(SampleRate::new(48000), [0, 1000, -2000, 500].repeat(2400));
        assert_eq!(level(&speech).max(), 32393);
        let silence = Segment::new(SampleRate::new(48000), vec![0; 4]);
        assert_eq!(level(&silence), silence);
    }

    #[test]
    fn keeps_what_is_quieter_than_the_floor_as_it_is() {
        // A patch of noise a few dozen steps high, which leveled would come up by 56 dB.
        let noise: Vec<i16> = (0..7680)
            .map(|i: i32| ((i * 7919) % 97 - 48) as i16)
            .collect();
        let noise = Segment::new(SampleRate::new(48000), noise);
        assert_eq!(level(&noise), noise);
        let louder = Segment::new(
            SampleRate::new(48000),
            noise.samples.iter().map(|&s| s * 4).collect(),
        );
        assert_eq!(level(&louder).max(), 32393);
    }
}
