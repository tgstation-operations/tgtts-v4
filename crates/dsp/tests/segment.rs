//! The segment module against pydub, to the bit.

mod common;

use common::{Tensors, first_difference};
use dsp::{
    quantity::{Gain, Millis, SampleRate},
    segment::{Error, Segment},
};

const CROSSFADE_TOO_LONG: i64 = 1;
const TOO_MANY_MISSING_FRAMES: i64 = 2;
const EMPTY_FADE: i64 = 3;

#[track_caller]
fn check(fx: &Tensors, name: &str, actual: Result<Segment, Error>) {
    let error = fx.int(&format!("{name}/error"));
    match (actual, error) {
        (Ok(actual), 0) => {
            let expected = fx.pcm(name);
            assert!(
                actual.samples == expected,
                "{name}: {}",
                first_difference(&actual.samples, &expected)
            );
        }
        (Err(Error::CrossfadeTooLong { .. }), CROSSFADE_TOO_LONG)
        | (Err(Error::TooManyMissingFrames(_)), TOO_MANY_MISSING_FRAMES)
        | (Err(Error::EmptyFade), EMPTY_FADE) => {}
        (actual, error) => panic!("{name}: {actual:?} where pydub gave error {error}"),
    }
}

#[test]
fn matches_pydub() {
    let fx = Tensors::new(&testkit::fixture!("dsp_pydub"));
    let segments: Vec<Segment> = (0..fx.int("cases"))
        .map(|case| {
            Segment::new(
                SampleRate::new(fx.int(&format!("{case}/rate")) as u32),
                fx.pcm(&format!("{case}/input")),
            )
        })
        .collect();
    for (case, seg) in segments.iter().enumerate() {
        let name = |op: &str| format!("{case}/{op}");
        assert_eq!(seg.len_ms(), Millis::new(fx.int(&name("len_ms"))), "{case}");
        assert_eq!(i64::from(seg.rms()), fx.int(&name("rms")), "{case}");
        assert_eq!(i64::from(seg.max()), fx.int(&name("max")), "{case}");
        assert_eq!(
            seg.dbfs().value().to_bits(),
            fx.f64s(&name("dbfs"))[0].to_bits(),
            "{case}"
        );
        assert_eq!(
            seg.max_dbfs().value().to_bits(),
            fx.f64s(&name("max_dbfs"))[0].to_bits(),
            "{case}"
        );
        assert_eq!(
            seg.leading_silence(),
            Ok(Millis::new(fx.int(&name("leading_silence")))),
            "{case}"
        );
        check(&fx, &name("strip_silence"), seg.strip_silence());
        for (k, db) in fx.f64s(&name("gain")).into_iter().enumerate() {
            check(
                &fx,
                &name(&format!("gain/{k}")),
                Ok(seg.apply_gain(Gain::new(db))),
            );
        }
        for (k, headroom) in fx.f64s(&name("normalize")).into_iter().enumerate() {
            check(
                &fx,
                &name(&format!("normalize/{k}")),
                Ok(seg.normalize(Gain::new(headroom))),
            );
        }
        for (k, duration) in fx.ints(&name("fade")).into_iter().enumerate() {
            let duration = Millis::new(duration);
            check(&fx, &name(&format!("fade_in/{k}")), seg.fade_in(duration));
            check(&fx, &name(&format!("fade_out/{k}")), seg.fade_out(duration));
        }
        for (k, row) in fx.int_rows(&name("slice")).into_iter().enumerate() {
            let start = (row[0] != 0).then_some(Millis::new(row[1]));
            let end = (row[2] != 0).then_some(Millis::new(row[3]));
            check(&fx, &name(&format!("slice/{k}")), seg.slice(start, end));
        }
        for (k, row) in fx.int_rows(&name("append")).into_iter().enumerate() {
            let other = &segments[row[0] as usize];
            check(
                &fx,
                &name(&format!("append/{k}")),
                seg.append(other, Millis::new(row[1])),
            );
        }
    }
    for duration in [250, 1, 33] {
        for rate in [48000, 44100, 11025, 3001] {
            let silent = Segment::silent(Millis::new(duration), SampleRate::new(rate));
            assert!(silent.samples.iter().all(|&s| s == 0));
            let frames = fx.int(&format!("silent/{duration}/{rate}"));
            assert_eq!(
                silent.samples.len() as i64,
                frames,
                "silent({duration}, {rate})"
            );
        }
    }
}

#[test]
fn reverses() {
    let seg = Segment::new(SampleRate::new(8000), vec![1, -2, 3]);
    assert_eq!(seg.reverse().samples, [3, -2, 1]);
}
