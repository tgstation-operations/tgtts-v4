//! The resample module against librosa, to the bit: both drive the same libsoxr the same way.

mod common;

use common::{Tensors, first_difference};
use dsp::resample::{Resampler, resample};

fn same<T: Copy + Into<f64>>(actual: &[T], expected: &[T]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &e)| a.into().to_bits() == e.into().to_bits())
}

#[test]
fn matches_librosa() {
    let fx = Tensors::new(&testkit::fixture!("dsp_resample"));
    for case in 0..fx.int("cases") {
        let name = |what: &str| format!("{case}/{what}");
        let [from, to] = fx.f64s(&name("rates"))[..] else {
            panic!("{case}: rates")
        };
        if fx.int(&name("f64")) == 0 {
            let x = fx.f32s(&name("input"));
            let (actual, expected) = (resample(&x, from, to), fx.f32s(&name("output")));
            let diff = first_difference(&actual, &expected);
            assert!(
                same(&actual, &expected),
                "{case}, {from} to {to} Hz: {diff}"
            );
        } else {
            let x = fx.f64s(&name("input"));
            let (actual, expected) = (resample(&x, from, to), fx.f64s(&name("output")));
            let diff = first_difference(&actual, &expected);
            assert!(
                same(&actual, &expected),
                "{case}, {from} to {to} Hz: {diff}"
            );
        }
    }
}

#[test]
fn matches_librosa_across_chunks() {
    let fx = Tensors::new(&testkit::fixture!("dsp_resample"));
    let y = resample(&fx.f32s("long/input"), 8000.0, 48000.0);
    assert_eq!(y.len() as i64, fx.int("long/output_len"));
    let bytes: Vec<u8> = y.iter().flat_map(|s| s.to_le_bytes()).collect();
    assert_eq!(
        i64::from(crc32fast::hash(&bytes)),
        fx.int("long/output_crc32")
    );
}

/// A stream through libsoxr gives what `resample` does whole, however the input is cut up, to
/// the bit, and never more samples than `resample` keeps.
#[test]
fn a_stream_gives_what_the_whole_does() {
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let pitch = 48000.0 / 2f64.powf(-3.0 / 24.0);
    for (from, to) in [
        (48000.0, 8000.0),
        (8000.0, 48000.0),
        (pitch, 48000.0),
        (44100.0, 48000.0),
    ] {
        for len in [0, 1, 999, 48000] {
            let x: Vec<f32> = (0..len)
                .map(|_| (next() as f32 / u32::MAX as f32) - 0.5)
                .collect();
            let whole = resample(&x, from, to);
            for chunk in [1, 7, 4096, usize::MAX] {
                let mut resampler = Resampler::new(from, to);
                let mut streamed = Vec::new();
                for piece in x.chunks(chunk.min(x.len().max(1))) {
                    resampler.process(piece, &mut streamed);
                }
                resampler.finish(&mut streamed);
                assert!(streamed.len() <= whole.len(), "{from} to {to}, {len}");
                streamed.resize(whole.len(), 0.0);
                assert!(
                    same(&streamed, &whole),
                    "{from} to {to}, {len} in chunks of {chunk}: {}",
                    first_difference(&streamed, &whole)
                );
            }
        }
    }
}
