//! The resample module against librosa, to the bit: both drive the same libsoxr the same way.

mod common;

use common::{Tensors, first_difference};
use dsp::resample::resample;

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
