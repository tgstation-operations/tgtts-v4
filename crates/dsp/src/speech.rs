//! What the old frontend did to the model's speech before the game's filters.

use crate::{
    float::{from_pcm, to_pcm},
    pitch::pitch_shift,
    segment::Segment,
};

/// The silence after the speech, in ms.
const TAIL: u32 = 250;

/// `speech` with 250 ms of silence after it, then, unless `pitch` is 0, all of it shifted by
/// `pitch` quarter tones, as the old frontend finished the model's speech.
pub fn finish(speech: &Segment, pitch: i32) -> Segment {
    let mut samples = speech.samples.clone();
    samples.extend(Segment::silent(TAIL, speech.rate).samples);
    if pitch != 0 {
        let shifted = pitch_shift(&from_pcm(&samples), speech.rate, f64::from(pitch), 24);
        samples = to_pcm(&shifted);
    }
    Segment::new(speech.rate, samples)
}
