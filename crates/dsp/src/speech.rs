//! What the old frontend did to the model's speech before the game's filters.

use crate::{
    float::{from_pcm, to_pcm},
    pitch::pitch_shift,
    segment::Segment,
};

/// The silence after the speech, in ms.
pub const TAIL: u32 = 250;

/// `speech` with 250 ms of silence after it, then, unless `pitch` is 0, all of it shifted by
/// `pitch` quarter tones, as the old frontend finished the model's speech. The shift can take
/// the peaks of speech leveled to -0.1 dBFS past full scale, which the old frontend's 16 bits
/// clipped: such a shift is brought down to full scale first, which the loudness after makes up.
pub fn finish(speech: &Segment, pitch: i32) -> Segment {
    let mut samples = speech.samples.clone();
    samples.extend(Segment::silent(TAIL, speech.rate).samples);
    if pitch != 0 {
        let mut shifted = pitch_shift(&from_pcm(&samples), speech.rate, f64::from(pitch), 24);
        let peak = shifted.iter().fold(0f32, |m, s| m.max(s.abs()));
        if peak > 1.0 {
            shifted.iter_mut().for_each(|s| *s /= peak);
        }
        samples = to_pcm(&shifted);
    }
    Segment::new(speech.rate, samples)
}
