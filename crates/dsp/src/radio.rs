//! The radio effect: speech narrowed to a radio's band, overdriven, with static under it and mic
//! clicks around it, as the old radio service made it.

use std::{
    f64::consts::PI,
    iter,
    ops::{Add, Div},
    sync::Arc,
};

use rustfft::num_complex::Complex;

use crate::{
    RATE,
    float::{Float, to_pcm},
    quantity::SampleRate,
    resample::{Resampler, resample},
    segment::Segment,
};

/// Interleaved 16-bit PCM, as a WAV file holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcm {
    /// Frames per second.
    pub rate: SampleRate,
    /// Samples per frame.
    pub channels: u16,
    /// The samples, a frame's channels in turn.
    pub samples: Vec<i16>,
}

/// The sounds the radio puts around and under speech, mono and matched to the service's rate as
/// the old service's `load_and_match` read them from their WAV files, shared by its clones.
#[derive(Clone, Debug)]
pub struct Assets {
    click_on: Arc<[f64]>,
    click_off: Arc<[f64]>,
    noise: Arc<[f64]>,
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
            click_on: matched(click_on).into(),
            click_off: matched(click_off).into(),
            noise: matched(noise).into(),
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
    mono(&pcm.samples, pcm.channels)
}

/// Frames of `samples` in `channels` channels, as [`to_mono`] averages them.
fn mono<T>(samples: &[i16], channels: u16) -> Vec<T>
where
    T: Copy + From<i16> + From<u16> + Add<Output = T> + Div<Output = T>,
{
    let channels = channels.max(1);
    samples
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

/// `speech` over the radio, as the old radio service made it: its channels averaged, resampled to
/// 8 kHz and on to the service's 48 kHz, band-passed to 300 to 3000 Hz, overdriven, with the
/// static looped under it and going on for 0.1 s after it, then between the mic clicks.
pub fn radio(speech: &Pcm, assets: &Assets) -> Segment {
    finish(narrow(speech), assets)
}

/// The radio's processing of the speech, up to the static: float32 through the resampling and
/// the first normalization, float64 from the filter on, as numpy has it.
pub(crate) fn narrow(speech: &Pcm) -> Vec<f64> {
    narrowed(speech).0
}

/// [`narrow`]'s speech, and the peaks it scaled by: that of the speech resampled, and that of
/// the speech band-passed and overdriven.
fn narrowed(speech: &Pcm) -> (Vec<f64>, f32, f64) {
    let samples: Vec<f32> = to_mono(speech);
    let x = resample(&samples, f64::from(speech.rate), 8000.0);
    let x = resample(&x, 8000.0, f64::from(RATE));
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let x: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(if peak > 0.0 { v / peak * 0.5 } else { v }))
        .collect();
    let (b, a) = bandpass();
    let x: Vec<f64> = lfilter(&b, &a, &x)
        .iter()
        .map(|v| (2.0 * v).tanh())
        .collect();
    let driven = peak_of(&x);
    (normalize(&x, 0.7), peak, driven)
}

/// The rest of the radio: the static under the speech and 0.1 s past it, and the clicks around
/// them.
pub(crate) fn finish(speech: Vec<f64>, assets: &Assets) -> Segment {
    finished(speech, assets).0
}

/// [`finish`]'s radio, and the peak it scaled by, of the whole before it was scaled.
fn finished(speech: Vec<f64>, assets: &Assets) -> (Segment, f64) {
    let mut noise = looped(&assets.noise);
    let mut output = assets.click_on.to_vec();
    output.extend(speech.iter().zip(&mut noise).map(|(s, n)| s + n * 0.2));
    output.extend(noise.take(TAIL));
    output.extend_from_slice(&assets.click_off);
    let peak = peak_of(&output);
    (Segment::new(RATE, to_pcm(&normalize(&output, 0.9))), peak)
}

/// The static after the speech, in samples: 0.1 s.
const TAIL: usize = (RATE.value() / 10) as usize;

/// The peaks [`radio`] scales speech by, each to a level of its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peaks {
    /// The speech's, resampled to 8 kHz and back, which it brings to 0.5.
    pub narrowed: f32,
    /// The speech's band-passed and overdriven, which it brings to 0.7.
    pub driven: f64,
    /// The whole's, the clicks and the static with the speech, which it brings to 0.9.
    pub output: f64,
}

/// The peaks [`radio`] scales `speech` by.
pub fn peaks(speech: &Pcm, assets: &Assets) -> Peaks {
    let (narrowed, peak, driven) = self::narrowed(speech);
    let (_, output) = finished(narrowed, assets);
    Peaks {
        narrowed: peak,
        driven,
        output,
    }
}

/// [`radio`] on a stream, the speech in a chunk at a time, scaling by the peaks it is given
/// rather than those of the whole, which it has not got: given those [`peaks`] measures, it
/// gives [`radio`]'s samples to the bit. Each of its steps is causal but its resamplings, which
/// keep it behind the speech by libsoxr's filters, and the click before the speech goes out with
/// the first of it.
pub struct Stream {
    assets: Assets,
    peaks: Peaks,
    rate: SampleRate,
    channels: u16,
    /// From the speech's rate to 8 kHz, where it is not 8 kHz, and back up.
    down: Option<Resampler<f32>>,
    up: Option<Resampler<f32>>,
    /// The speech's frames taken, and the samples each resampling has given.
    taken: usize,
    narrowed: usize,
    widened: usize,
    /// The band-pass's coefficients and state.
    b: [f64; 9],
    a: [f64; 9],
    z: [f64; 8],
    /// The static's samples used.
    noise: usize,
    started: bool,
}

impl Stream {
    /// The radio of speech at `rate` Hz in `channels` channels, with `assets`' clicks and
    /// static, scaling by `peaks`.
    pub fn new(assets: &Assets, peaks: Peaks, rate: SampleRate, channels: u16) -> Self {
        let (b, a) = bandpass();
        Self {
            assets: assets.clone(),
            peaks,
            rate,
            channels,
            down: (rate.value() != 8000).then(|| Resampler::new(f64::from(rate), 8000.0)),
            up: Some(Resampler::new(8000.0, f64::from(RATE))),
            taken: 0,
            narrowed: 0,
            widened: 0,
            b: b.map(|v| v / a[0]),
            a: a.map(|v| v / a[0]),
            z: [0.0; 8],
            noise: 0,
            started: false,
        }
    }

    /// Takes the frames of `speech`, interleaved, and appends to `output` the radio it has made.
    pub fn process(&mut self, speech: &[i16], output: &mut Vec<i16>) {
        let mono: Vec<f32> = mono(speech, self.channels);
        self.taken += mono.len();
        let narrow = match &mut self.down {
            Some(down) => {
                let mut narrow = Vec::new();
                down.process(&mono, &mut narrow);
                narrow
            }
            None => mono,
        };
        self.narrowed += narrow.len();
        let mut wide = Vec::new();
        let up = self.up.as_mut().expect("resampling until the speech ends");
        up.process(&narrow, &mut wide);
        self.widened += wide.len();
        self.speech(&wide, output);
    }

    /// Appends to `output` the rest of the radio, as the speech ends: what the resamplings hold,
    /// cut or padded to [`radio`]'s lengths, the static after it and the click.
    pub fn finish(mut self, output: &mut Vec<i16>) {
        let mut narrow = Vec::new();
        let narrowed = match self.down.take() {
            Some(down) => {
                down.finish(&mut narrow);
                (self.taken as f64 * (8000.0 / f64::from(self.rate))).ceil() as usize
            }
            None => self.taken,
        };
        narrow.resize(narrowed - self.narrowed, 0.0);
        let widened = (narrowed as f64 * (f64::from(RATE) / 8000.0)).ceil() as usize;
        let mut wide = Vec::new();
        let mut up = self.up.take().expect("resampling until the speech ends");
        up.process(&narrow, &mut wide);
        up.finish(&mut wide);
        wide.resize(widened - self.widened, 0.0);
        self.speech(&wide, output);
        self.start(output);
        let tail: Vec<f64> = (0..TAIL).map(|_| self.next_noise()).collect();
        self.scaled(&tail, output);
        let click = self.assets.click_off.clone();
        self.scaled(&click, output);
    }

    /// The speech resampled, through the band-pass, the overdrive and the static under it.
    fn speech(&mut self, wide: &[f32], output: &mut Vec<i16>) {
        if wide.is_empty() {
            return;
        }
        self.start(output);
        let (narrowed, driven) = (self.peaks.narrowed, self.peaks.driven);
        let mut out = Vec::with_capacity(wide.len());
        for &v in wide {
            let x = f64::from(if narrowed > 0.0 {
                v / narrowed * 0.5
            } else {
                v
            });
            let (b, a, z) = (&self.b, &self.a, &mut self.z);
            let y = z[0] + b[0] * x;
            for k in 0..7 {
                z[k] = z[k + 1] + x * b[k + 1] - y * a[k + 1];
            }
            z[7] = x * b[8] - y * a[8];
            let y = (2.0 * y).tanh();
            let y = if driven > 0.0 { y / driven * 0.7 } else { y };
            out.push(y + self.next_noise() * 0.2);
        }
        self.scaled(&out, output);
    }

    /// The click before the speech, unless it has gone out.
    fn start(&mut self, output: &mut Vec<i16>) {
        if !self.started {
            self.started = true;
            let click = self.assets.click_on.clone();
            self.scaled(&click, output);
        }
    }

    fn next_noise(&mut self) -> f64 {
        let noise = &self.assets.noise;
        let n = if noise.is_empty() {
            0.0
        } else {
            noise[self.noise % noise.len()]
        };
        self.noise += 1;
        n
    }

    /// `x` scaled as the whole is, to 16 bits.
    fn scaled(&self, x: &[f64], output: &mut Vec<i16>) {
        let peak = self.peaks.output;
        output.extend(
            x.iter()
                .map(|&v| if peak > 0.0 { v / peak * 0.9 } else { v }.to_pcm()),
        );
    }
}

/// The static from its start, looped for as long as it is taken; none where there is none.
pub(crate) fn looped(noise: &[f64]) -> impl Iterator<Item = f64> + '_ {
    noise.iter().copied().cycle().chain(iter::repeat(0.0))
}

/// The largest magnitude of `x`.
fn peak_of(x: &[f64]) -> f64 {
    x.iter().fold(0f64, |m, v| m.max(v.abs()))
}

/// The old service's `normalize`: `x` scaled so its peak is `peak`, unless it is silent.
fn normalize(x: &[f64], peak: f64) -> Vec<f64> {
    let m = peak_of(x);
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
