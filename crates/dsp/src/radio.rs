//! The radio effect: speech narrowed to a radio's band, overdriven, with static under it and mic
//! clicks around it, as the old radio service made it.

use std::{error, f64::consts::PI, fmt};

use rustfft::num_complex::Complex;

use crate::{float::to_pcm, resample::resample, segment::Segment};

/// The rate the old service resampled speech to after 8 kHz, and declared every radio message to
/// be at, whatever its parts were at.
pub(crate) const OUTPUT_RATE: u32 = 48000;

/// Interleaved 16-bit PCM, as a WAV file holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcm {
    /// Frames per second.
    pub rate: u32,
    /// Samples per frame.
    pub channels: u16,
    /// The samples, a frame's channels in turn.
    pub samples: Vec<i16>,
}

/// The sounds the radio puts around and under speech, mono and matched to a rate as the old
/// service's `load_and_match` read them from their WAV files.
#[derive(Clone, Debug)]
pub struct Assets {
    rate: u32,
    click_on: Vec<f64>,
    click_off: Vec<f64>,
    noise: Vec<f64>,
}

impl Assets {
    /// The mic clicks and the static, matched to `rate`: each read as float64 and averaged over
    /// its channels, then resampled to `rate` as librosa does.
    pub fn new(click_on: &Pcm, click_off: &Pcm, noise: &Pcm, rate: u32) -> Self {
        let matched = |clip: &Pcm| {
            let mono = to_mono(clip);
            if clip.rate == rate {
                mono
            } else {
                resample(&mono, f64::from(clip.rate), f64::from(rate))
            }
        };
        Self {
            rate,
            click_on: matched(click_on),
            click_off: matched(click_off),
            noise: matched(noise),
        }
    }

    /// The rate these are matched to.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The click before the speech.
    pub fn click_on(&self) -> &[f64] {
        &self.click_on
    }

    /// The click after the speech.
    pub fn click_off(&self) -> &[f64] {
        &self.click_off
    }

    /// The static.
    pub fn noise(&self) -> &[f64] {
        &self.noise
    }
}

/// soundfile's float64 of a WAV's samples, averaged over its channels as numpy's `mean` does.
fn to_mono(clip: &Pcm) -> Vec<f64> {
    let channels = usize::from(clip.channels.max(1));
    clip.samples
        .chunks(channels)
        .map(|frame| {
            let sum = frame
                .iter()
                .fold(0.0, |sum, &s| sum + f64::from(s) / 32768.0);
            sum / channels as f64
        })
        .collect()
}

/// Why the radio failed, where the old service failed the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// There was no speech, whose peak numpy could not take.
    Empty,
    /// The speech, of this many samples once resampled, outlasts the static, which numpy could
    /// not add to it.
    LongerThanStatic(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no speech"),
            Self::LongerThanStatic(len) => {
                write!(f, "speech of {len} samples outlasts the static")
            }
        }
    }
}

impl error::Error for Error {}

/// `speech` over the radio, as the old radio service made it: resampled to 8 kHz and on to
/// 48 kHz, band-passed to 300 to 3000 Hz, overdriven, with static under it, then between the
/// mic clicks, with 0.1 s of static before the second, and declared to be at 48 kHz.
///
/// The old service filtered at the speech's own rate, though it had resampled it to 48 kHz,
/// and matched the clicks and static to the speech's rate, though it declared the whole to be
/// at 48 kHz: so it did with speech at another rate, as the silicon voices' 44.1 kHz. It took
/// a frame of two channels as their mean, and the samples of more as they were interleaved.
///
/// # Panics
///
/// If `assets` are not matched to `speech`'s rate.
pub fn radio(speech: &Pcm, assets: &Assets) -> Result<Segment, Error> {
    assert_eq!(assets.rate, speech.rate, "assets for another rate");
    let narrowed = narrow(speech)?;
    finish(narrowed, assets, speech.rate)
}

/// The radio's processing of the speech, up to the static: float32 through the resampling and
/// the first normalization, float64 from the filter on, as numpy has it.
pub(crate) fn narrow(speech: &Pcm) -> Result<Vec<f64>, Error> {
    let scaled = speech.samples.iter().map(|&s| f32::from(s) / 32768.0);
    let samples: Vec<f32> = if speech.channels == 2 {
        let scaled: Vec<f32> = scaled.collect();
        scaled
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[l, r]| (l + r) / 2.0)
            .collect()
    } else {
        scaled.collect()
    };
    let x = resample(&samples, f64::from(speech.rate), 8000.0);
    let x = resample(&x, 8000.0, f64::from(OUTPUT_RATE));
    let peak = x.iter().fold(None, |m: Option<f32>, v| {
        Some(m.map_or(v.abs(), |m| m.max(v.abs())))
    });
    let peak = peak.ok_or(Error::Empty)?;
    let x: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(if peak > 0.0 { v / peak * 0.5 } else { v }))
        .collect();
    let (b, a) = bandpass(speech.rate);
    let x: Vec<f64> = lfilter(&b, &a, &x)
        .iter()
        .map(|v| (2.0 * v).tanh())
        .collect();
    Ok(normalize(&x, 0.7))
}

/// The rest of the radio: the static under the speech, the clicks and 0.1 s of static, at the
/// speech's `rate`, around it.
pub(crate) fn finish(speech: Vec<f64>, assets: &Assets, rate: u32) -> Result<Segment, Error> {
    let noise = (assets.noise.get(..speech.len())).ok_or(Error::LongerThanStatic(speech.len()))?;
    let tail = &noise[..((f64::from(rate) * 0.1) as usize).min(noise.len())];
    let mut output = assets.click_on.clone();
    output.extend(speech.iter().zip(noise).map(|(s, n)| s + n * 0.2));
    output.extend_from_slice(tail);
    output.extend_from_slice(&assets.click_off);
    Ok(Segment::new(OUTPUT_RATE, to_pcm(&normalize(&output, 0.9))))
}

/// The old service's `normalize`: `x` scaled so its peak is `peak`, unless it is silent.
fn normalize(x: &[f64], peak: f64) -> Vec<f64> {
    let m = x.iter().fold(0f64, |m, v| m.max(v.abs()));
    if m > 0.0 {
        x.iter().map(|v| v / m * peak).collect()
    } else {
        x.to_vec()
    }
}

/// `scipy.signal.butter(4, [300 / nyq, 3000 / nyq], btype="band")` at `rate`, through the same
/// steps: the analog prototype's poles moved to the band and the bilinear transform.
fn bandpass(rate: u32) -> ([f64; 9], [f64; 9]) {
    let nyq = f64::from(rate) * 0.5;
    let fs = 2.0;
    let warped = [300.0 / nyq, 3000.0 / nyq].map(|w| 2.0 * fs * (PI * w / fs).tan());
    let prototype = [-3.0, -1.0, 1.0, 3.0].map(|m| -Complex::new(0.0, PI * m / 8.0).exp());
    let bw = warped[1] - warped[0];
    let wo = (warped[0] * warped[1]).sqrt();
    let lowpass = prototype.map(|p| p * bw / 2.0);
    let shift = |p: Complex<f64>| (p * p - wo * wo).sqrt();
    let poles: Vec<Complex<f64>> = (lowpass.iter().map(|&p| p + shift(p)))
        .chain(lowpass.iter().map(|&p| p - shift(p)))
        .collect();
    let gain = bw.powf(4.0);
    // Four zeros at the origin go to 1, the four at infinity to -1.
    let fs2 = 2.0 * fs;
    let digital: Vec<Complex<f64>> = poles.iter().map(|&p| (fs2 + p) / (fs2 - p)).collect();
    let denominator = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |product, &p| product * (fs2 - p));
    let gain = gain * (Complex::new(fs2.powi(4), 0.0) / denominator).re;
    let zeros = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0].map(|z| Complex::new(z, 0.0));
    let b = poly(&zeros).map(|c| gain * c.re);
    let a = poly(&digital).map(|c| c.re);
    (b, a)
}

/// The coefficients of the monic polynomial with roots `roots`, highest power first.
fn poly(roots: &[Complex<f64>]) -> [Complex<f64>; 9] {
    let mut coefficients = [Complex::new(0.0, 0.0); 9];
    coefficients[0] = Complex::new(1.0, 0.0);
    for (n, &root) in roots.iter().enumerate() {
        for i in (1..=n + 1).rev() {
            coefficients[i] -= root * coefficients[i - 1];
        }
    }
    coefficients
}

/// `scipy.signal.lfilter(b, a, x)`: the direct form II transposed, as scipy computes it.
fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let b: Vec<f64> = b.iter().map(|v| v / a0).collect();
    let a: Vec<f64> = a.iter().map(|v| v / a0).collect();
    let n = b.len();
    let mut z = vec![0.0; n - 1];
    x.iter()
        .map(|&x| {
            let y = z[0] + b[0] * x;
            for k in 0..n - 2 {
                z[k] = z[k + 1] + x * b[k + 1] - y * a[k + 1];
            }
            z[n - 2] = x * b[n - 1] - y * a[n - 1];
            y
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use safetensors::SafeTensors;

    use super::*;

    /// The design's steps are scipy's, but its complex arithmetic is num-complex's and its
    /// polynomials are multiplied out here rather than through numpy's dot products: the
    /// coefficients agreed to within 2.1e-15 of the largest.
    #[test]
    fn bandpass_matches_scipy() {
        let fixture = testkit::fixture!("dsp_radio");
        let bytes = std::fs::read(fixture.path()).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let doubles = |name: &str| -> Vec<f64> {
            let data = tensors.tensor(name).unwrap().data().to_vec();
            data.as_chunks()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect()
        };
        for rate in [8000, 44100, 48000] {
            let (b, a) = bandpass(rate);
            for (name, actual) in [("b", b), ("a", a)] {
                let expected = doubles(&format!("butter/{rate}/{name}"));
                let scale = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
                for (i, (x, e)) in actual.iter().zip(&expected).enumerate() {
                    let error = (x - e).abs() / scale;
                    assert!(error < 1e-13, "{rate} Hz {name}[{i}]: {x:e} for {e:e}");
                }
            }
        }
    }
}
