//! librosa's `effects.pitch_shift`: a phase-vocoder time stretch, resampled back to the length it
//! started at.
//!
//! librosa keeps the spectrogram in complex64, the vocoder's phase in float32 and the signal it
//! inverts to in float32, while it computes most steps in between in float64; each step here
//! rounds where librosa's does. The phase grows by up to 1600 radians a frame, so its float32
//! rounding is far from negligible: kept in float64 instead, the result agreed with librosa's to
//! 85 dB on a blip and 58 dB on 3 s of speech, rather than 134 dB on both.

use std::f64::consts::PI;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::resample::resample;

const N_FFT: usize = 2048;
const HOP: usize = N_FFT / 4;
const BINS: usize = N_FFT / 2 + 1;

/// `librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)`:
/// `y` at `sr` Hz shifted by `n_steps` steps of `1 / bins_per_octave` octave, keeping its
/// length.
///
/// # Panics
///
/// If `bins_per_octave` is 0.
pub fn pitch_shift(y: &[f32], sr: u32, n_steps: f64, bins_per_octave: u32) -> Vec<f32> {
    assert!(bins_per_octave > 0, "bins_per_octave of 0");
    if y.is_empty() {
        return Vec::new();
    }
    let rate = 2f64.powf(-n_steps / f64::from(bins_per_octave));
    let stretched = time_stretch(y, rate);
    let mut shifted = resample(&stretched, f64::from(sr) / rate, f64::from(sr));
    shifted.resize(y.len(), 0.0);
    shifted
}

/// `librosa.effects.time_stretch(y, rate=rate)`.
fn time_stretch(y: &[f32], rate: f64) -> Vec<f32> {
    let window = hann();
    let mut fft = FftPlanner::new();
    let spectrum = stft(y, &window, &mut fft);
    let stretched = phase_vocoder(&spectrum, rate);
    let len = (y.len() as f64 / rate).round_ties_even() as usize;
    istft(&stretched, &window, len, &mut fft)
}

/// scipy's periodic Hann window of `N_FFT`, `0.5 + 0.5 cos` over `linspace(-pi, pi)`.
fn hann() -> Vec<f64> {
    let step = (PI - -PI) / N_FFT as f64;
    (0..N_FFT)
        .map(|n| 0.5 + 0.5 * (n as f64 * step + -PI).cos())
        .collect()
}

type Column = Vec<Complex<f32>>;

/// `librosa.stft(y)`: centered frames of the zero-padded signal, windowed and transformed in
/// float64, stored as complex64.
fn stft(y: &[f32], window: &[f64], planner: &mut FftPlanner<f64>) -> Vec<Column> {
    let fft = planner.plan_fft_forward(N_FFT);
    let mut padded = vec![0f32; N_FFT / 2];
    padded.extend_from_slice(y);
    padded.resize(padded.len() + N_FFT / 2, 0.0);
    let frames = 1 + (padded.len() - N_FFT) / HOP;
    let mut buffer = vec![Complex::default(); N_FFT];
    (0..frames)
        .map(|t| {
            let frame = &padded[t * HOP..t * HOP + N_FFT];
            for ((b, &w), &s) in buffer.iter_mut().zip(window).zip(frame) {
                *b = Complex::new(w * f64::from(s), 0.0);
            }
            fft.process(&mut buffer);
            buffer[..BINS]
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect()
        })
        .collect()
}

/// `librosa.phase_vocoder(D, rate=rate)`.
fn phase_vocoder(spectrum: &[Column], rate: f64) -> Vec<Column> {
    let frames = spectrum.len();
    // numpy's rfftfreq at sr = 2 pi, times the hop.
    let d = 1.0 / (2.0 * PI);
    let val = 1.0 / (N_FFT as f64 * d);
    let phi_advance: Vec<f64> = (0..BINS).map(|k| HOP as f64 * (k as f64 * val)).collect();
    let mut phase: Vec<f32> = spectrum[0].iter().map(|c| angle(*c)).collect();
    let zeros = vec![Complex::default(); BINS];
    let column = |i: usize| spectrum.get(i).unwrap_or(&zeros);
    let steps = (frames as f64 / rate).ceil() as usize;
    (0..steps)
        .map(|t| {
            let step = t as f64 * rate;
            let (c0, c1) = (column(step as usize), column(step as usize + 1));
            let alpha = step % 1.0;
            let mut out = Vec::with_capacity(BINS);
            for k in 0..BINS {
                let mag = (1.0 - alpha) * f64::from(abs(c0[k])) + alpha * f64::from(abs(c1[k]));
                let (sin, cos) = (phase[k].sin(), phase[k].cos());
                out.push(Complex::new(
                    (f64::from(cos) * mag) as f32,
                    (f64::from(sin) * mag) as f32,
                ));
                let dphase = f64::from(angle(c1[k]) - angle(c0[k])) - phi_advance[k];
                let dphase = dphase - 2.0 * PI * (dphase / (2.0 * PI)).round_ties_even();
                phase[k] = (f64::from(phase[k]) + (phi_advance[k] + dphase)) as f32;
            }
            out
        })
        .collect()
}

/// `librosa.istft(D, length=len)` for complex64 `D`: each frame inverted to float32, windowed in
/// float64 and overlap-added in float32, then divided by the summed squared window.
fn istft(
    spectrum: &[Column],
    window: &[f64],
    len: usize,
    planner: &mut FftPlanner<f64>,
) -> Vec<f32> {
    let ifft = planner.plan_fft_inverse(N_FFT);
    let frames = spectrum.len().min((len + 2 * (N_FFT / 2)).div_ceil(HOP));
    // librosa overlap-adds the first two frames into a buffer of their own, which the signal is
    // then cut from, and the rest into the signal: as one buffer padded by half a frame, but
    // with the first two frames whole.
    let padded_len = (N_FFT / 2 + len).max(N_FFT + HOP);
    let mut padded = vec![0f32; padded_len];
    let mut buffer = vec![Complex::default(); N_FFT];
    for (t, column) in spectrum[..frames].iter().enumerate() {
        // scipy's irfft takes only the real parts of the first and last bins.
        buffer[0] = Complex::new(f64::from(column[0].re), 0.0);
        buffer[N_FFT / 2] = Complex::new(f64::from(column[N_FFT / 2].re), 0.0);
        for k in 1..N_FFT / 2 {
            let c = Complex::new(f64::from(column[k].re), f64::from(column[k].im));
            buffer[k] = c;
            buffer[N_FFT - k] = c.conj();
        }
        ifft.process(&mut buffer);
        let start = t * HOP;
        let end = if t < 2 { padded_len } else { N_FFT / 2 + len };
        let n = N_FFT.min(end.saturating_sub(start));
        for i in 0..n {
            let sample = (buffer[i].re / N_FFT as f64) as f32;
            let windowed = window[i] * f64::from(sample);
            padded[start + i] = (f64::from(padded[start + i]) + windowed) as f32;
        }
    }
    let mut y = padded[N_FFT / 2..N_FFT / 2 + len].to_vec();
    let sum = window_sumsquare(window, frames);
    for (i, s) in y.iter_mut().enumerate() {
        let w = sum.get(N_FFT / 2 + i).copied().unwrap_or(0.0);
        if w > f32::MIN_POSITIVE {
            *s /= w;
        }
    }
    y
}

/// `librosa.filters.window_sumsquare` for `frames` frames, summed in float32.
fn window_sumsquare(window: &[f64], frames: usize) -> Vec<f32> {
    let len = N_FFT + HOP * (frames - 1);
    let mut sum = vec![0f32; len];
    for t in 0..frames {
        let start = t * HOP;
        let n = N_FFT.min(len - start);
        for (s, &w) in sum[start..start + n].iter_mut().zip(window) {
            *s = (f64::from(*s) + w * w) as f32;
        }
    }
    sum
}

/// numpy's `abs` of a complex64: the larger part times `sqrt(1 + r * r)`, `r` the smaller over the
/// larger, with the `1 + r * r` fused.
fn abs(c: Complex<f32>) -> f32 {
    let (re, im) = (c.re.abs(), c.im.abs());
    let (larger, smaller) = (re.max(im), re.min(im));
    if larger == 0.0 {
        return 0.0;
    }
    let ratio = smaller / larger;
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

/// numpy's `angle` of a complex64: libm's `atan2f`.
fn angle(c: Complex<f32>) -> f32 {
    c.im.atan2(c.re)
}

#[cfg(test)]
mod tests {
    use safetensors::SafeTensors;

    use super::*;
    use crate::float::from_pcm;

    /// The steps before the inverse transform, which librosa takes in float64 where this does,
    /// against librosa's to the bit, where the tolerance of the whole would hide a misrounding.
    #[test]
    fn stft_and_vocoder_match_librosa() {
        let fixture = testkit::fixture!("dsp_pitch");
        let bytes = std::fs::read(fixture.path()).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let data = |name: &str| tensors.tensor(name).unwrap().data().to_vec();
        let floats = |name: &str| -> Vec<f32> {
            data(name)
                .as_chunks()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let bits = |columns: &[Column]| -> Vec<u32> {
            columns
                .iter()
                .flatten()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect()
        };
        let expected =
            |name: &str| -> Vec<u32> { floats(name).iter().map(|f| f.to_bits()).collect() };

        let input: Vec<i16> = floats("steps/input").iter().map(|&s| s as i16).collect();
        let rate = f64::from_le_bytes(data("steps/rate").as_chunks().0[0]);
        let spectrum = stft(&from_pcm(&input), &hann(), &mut FftPlanner::new());
        assert!(bits(&spectrum) == expected("steps/stft"), "stft");
        let columns: Vec<Column> = floats("steps/stft")
            .as_chunks::<2>()
            .0
            .chunks(BINS)
            .map(|column| {
                column
                    .iter()
                    .map(|&[re, im]| Complex::new(re, im))
                    .collect()
            })
            .collect();
        let stretched = phase_vocoder(&columns, rate);
        assert!(
            bits(&stretched) == expected("steps/phase_vocoder"),
            "phase vocoder"
        );
    }
}
