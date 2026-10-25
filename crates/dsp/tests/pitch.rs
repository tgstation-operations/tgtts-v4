//! The pitch shift against librosa's, and the conversions around it against the old service's.

mod common;

use common::Tensors;
use dsp::{float, pitch::pitch_shift};

/// librosa inverts its float32 spectrogram in float32, through pocketfft, where this inverts
/// in float64 and rounds: with that step taken in float64 in librosa too, the two agree to the
/// bit. As it is, they differed by up to 6.0e-7 on these signals, the loudest of which peaks at
/// 1.09.
const MAX_ERROR: f64 = 1e-6;
/// The SNR of the same differences was 134 dB or more.
const MIN_SNR_DB: f64 = 120.0;
/// Off by so little, a sample can still truncate to the next 16-bit value, more often the louder
/// it is: at most 13 of 3000 samples did, in the case clipped at full scale.
const MAX_PCM_DIFFERING: f64 = 0.01;

#[test]
fn matches_librosa() {
    let fx = Tensors::new(&testkit::fixture!("dsp_pitch"));
    for case in 0..fx.int("cases") {
        let name = |what: &str| format!("{case}/{what}");
        let input = float::from_pcm(&fx.pcm(&name("input")));
        let steps = fx.f64s(&name("steps"))[0];
        let bins = fx.int(&name("bins_per_octave")) as u32;
        let actual = pitch_shift(&input, 48000, steps, bins);
        let expected = fx.f32s(&name("output"));
        assert_eq!(actual.len(), expected.len(), "{case}");

        let (mut signal, mut noise, mut max_error) = (0.0, 0.0, 0f64);
        for (&a, &e) in actual.iter().zip(&expected) {
            let error = f64::from(a) - f64::from(e);
            signal += f64::from(e) * f64::from(e);
            noise += error * error;
            max_error = max_error.max(error.abs());
        }
        let snr = 10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10();
        assert!(max_error <= MAX_ERROR, "{case}: max error {max_error:e}");
        assert!(snr >= MIN_SNR_DB, "{case}: SNR {snr:.1} dB");

        // The conversion is the old service's to the bit, as the test below has it.
        let (actual, expected) = (float::to_pcm(&actual), float::to_pcm(&expected));
        let differing = actual.iter().zip(&expected).filter(|(a, e)| a != e).count();
        let worst = (actual.iter().zip(&expected))
            .map(|(a, e)| a.wrapping_sub(*e).unsigned_abs())
            .max();
        assert!(worst <= Some(1), "{case}: 16-bit samples {worst:?} apart");
        let fraction = differing as f64 / expected.len() as f64;
        assert!(
            fraction <= MAX_PCM_DIFFERING,
            "{case}: {differing} samples differ"
        );
    }
}

#[test]
fn converts_as_the_old_service() {
    let fx = Tensors::new(&testkit::fixture!("dsp_pitch"));
    let pcm: Vec<i16> = fx.ints("convert/pcm").iter().map(|&s| s as i16).collect();
    let floats: Vec<u32> = float::from_pcm(&pcm).iter().map(|f| f.to_bits()).collect();
    let expected: Vec<u32> = fx
        .f32s("convert/float")
        .iter()
        .map(|f| f.to_bits())
        .collect();
    assert_eq!(floats, expected);

    let input = fx.ints("convert/float32/input");
    let x: Vec<f32> = input.iter().map(|&b| f32::from_bits(b as u32)).collect();
    let expected: Vec<i16> = fx
        .ints("convert/float32/pcm")
        .iter()
        .map(|&s| s as i16)
        .collect();
    assert_eq!(float::to_pcm(&x), expected, "float32");

    let input = fx.ints("convert/float64/input");
    let x: Vec<f64> = input.iter().map(|&b| f64::from_bits(b as u64)).collect();
    let expected: Vec<i16> = fx
        .ints("convert/float64/pcm")
        .iter()
        .map(|&s| s as i16)
        .collect();
    assert_eq!(float::to_pcm(&x), expected, "float64");
}
