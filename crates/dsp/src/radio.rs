//! The radio effect: speech narrowed to a radio's band, overdriven, with static under it and mic
//! clicks around it, as the old radio service made it.

use std::{
    error,
    f64::consts::PI,
    fmt, iter,
    ops::{Add, Div},
};

use rustfft::num_complex::Complex;

use crate::{RATE, float::to_pcm, resample::resample, segment::Segment};

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

/// The sounds the radio puts around and under speech, mono and matched to the service's rate as
/// the old service's `load_and_match` read them from their WAV files.
#[derive(Clone, Debug)]
pub struct Assets {
    click_on: Vec<f64>,
    click_off: Vec<f64>,
    noise: Vec<f64>,
}

impl Assets {
    /// The mic clicks and the static, matched to the service's rate: each read as float64 and
    /// averaged over its channels, then resampled as librosa does.
    pub fn new(click_on: &Pcm, click_off: &Pcm, noise: &Pcm) -> Self {
        let matched = |clip: &Pcm| {
            let mono = to_mono(clip);
            if clip.rate == RATE {
                mono
            } else {
                resample(&mono, f64::from(clip.rate), f64::from(RATE))
            }
        };
        Self {
            click_on: matched(click_on),
            click_off: matched(click_off),
            noise: matched(noise),
        }
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

/// Each frame of `pcm` scaled to [-1, 1) and averaged over its channels in the precision of `T`,
/// as numpy's `mean` does: the sum of up to 256 channels is exact in either, whatever its order.
fn to_mono<T>(pcm: &Pcm) -> Vec<T>
where
    T: Copy + From<i16> + From<u16> + Add<Output = T> + Div<Output = T>,
{
    let channels = pcm.channels.max(1);
    pcm.samples
        .chunks_exact(usize::from(channels))
        .map(|frame| {
            let scaled = frame.iter().map(|&s| T::from(s) / T::from(32768u16));
            let sum = scaled
                .reduce(|sum, s| sum + s)
                .expect("a frame has a sample");
            sum / T::from(channels)
        })
        .collect()
}

/// Why the radio failed, where the old service failed the request.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// There was no speech, whose peak numpy could not take.
    Empty,
    /// A span to duck, starting at this many seconds, starts past the end of the speech, where
    /// numpy could not fade it.
    SpanPastEnd(f64),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no speech"),
            Self::SpanPastEnd(start) => write!(f, "span at {start} s starts past the speech"),
        }
    }
}

impl error::Error for Error {}

/// `speech` over the radio, as the old radio service made it: its channels averaged, resampled to
/// 8 kHz and on to the service's 48 kHz, band-passed to 300 to 3000 Hz, overdriven, with the
/// static looped under it and going on for 0.1 s after it, then between the mic clicks.
pub fn radio(speech: &Pcm, assets: &Assets) -> Result<Segment, Error> {
    let narrowed = narrow(speech)?;
    finish(narrowed, assets)
}

/// The radio's processing of the speech, up to the static: float32 through the resampling and
/// the first normalization, float64 from the filter on, as numpy has it.
pub(crate) fn narrow(speech: &Pcm) -> Result<Vec<f64>, Error> {
    let samples: Vec<f32> = to_mono(speech);
    let x = resample(&samples, f64::from(speech.rate), 8000.0);
    let x = resample(&x, 8000.0, f64::from(RATE));
    let peak = x.iter().fold(None, |m: Option<f32>, v| {
        Some(m.map_or(v.abs(), |m| m.max(v.abs())))
    });
    let peak = peak.ok_or(Error::Empty)?;
    let x: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(if peak > 0.0 { v / peak * 0.5 } else { v }))
        .collect();
    let (b, a) = bandpass();
    let x: Vec<f64> = lfilter(&b, &a, &x)
        .iter()
        .map(|v| (2.0 * v).tanh())
        .collect();
    Ok(normalize(&x, 0.7))
}

/// The rest of the radio: the static under the speech and 0.1 s past it, and the clicks around
/// them.
pub(crate) fn finish(speech: Vec<f64>, assets: &Assets) -> Result<Segment, Error> {
    let mut noise = looped(&assets.noise);
    let mut output = assets.click_on.clone();
    output.extend(speech.iter().zip(&mut noise).map(|(s, n)| s + n * 0.2));
    output.extend(noise.take((f64::from(RATE) * 0.1) as usize));
    output.extend_from_slice(&assets.click_off);
    Ok(Segment::new(RATE, to_pcm(&normalize(&output, 0.9))))
}

/// The static from its start, looped for as long as it is taken; none where there is none.
pub(crate) fn looped(noise: &[f64]) -> impl Iterator<Item = f64> + '_ {
    noise.iter().copied().cycle().chain(iter::repeat(0.0))
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

/// `scipy.signal.butter(4, [300 / nyq, 3000 / nyq], btype="band")` at the service's rate, through
/// the same steps: the analog prototype's poles moved to the band and the bilinear transform.
fn bandpass() -> ([f64; 9], [f64; 9]) {
    let nyq = f64::from(RATE) * 0.5;
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
        let (b, a) = bandpass();
        for (name, actual) in [("b", b), ("a", a)] {
            let expected = doubles(&format!("butter/{name}"));
            let scale = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
            for (i, (x, e)) in actual.iter().zip(&expected).enumerate() {
                let error = (x - e).abs() / scale;
                assert!(error < 1e-13, "{name}[{i}]: {x:e} for {e:e}");
            }
        }
    }
}
