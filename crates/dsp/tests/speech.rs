//! The finishing of speech against the old frontend's.

mod common;

use common::{Tensors, assert_pcm_close};
use dsp::{segment::Segment, speech};

/// The pitch shift's: a sample one off now and then, at most 7 of 19200 here.
const MAX_DIFFERENCE: i32 = 1;
const MAX_DIFFERING: f64 = 0.01;

#[test]
fn finishes_as_the_old_frontend() {
    let fx = Tensors::new(&testkit::fixture!("dsp_speech"));
    let input = Segment::new(48000, fx.pcm("input"));
    for pitch in fx.ints("pitches") {
        let actual = speech::finish(&input, pitch as i32);
        if pitch == 0 {
            let bytes: Vec<u8> = actual
                .samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            assert_eq!(actual.samples.len() as i64, fx.int("0/output_len"));
            assert_eq!(i64::from(crc32fast::hash(&bytes)), fx.int("0/output_crc32"));
        } else {
            let expected = fx.pcm(&format!("{pitch}/output"));
            let name = format!("pitch {pitch}");
            assert_pcm_close(
                &name,
                &actual.samples,
                &expected,
                MAX_DIFFERENCE,
                MAX_DIFFERING,
            );
        }
    }
}
