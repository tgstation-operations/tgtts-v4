//! Effects on the spectrum, frame by frame: a short-time Fourier transform run on a stream, the
//! fish tongue's swinging filter, and the formants moved against a pitch shift.

use std::{f64::consts::PI, sync::Arc};

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{Processor, RATE};

/// A short-time Fourier transform on a stream: Hann-windowed frames of `size` every `hop`,
/// each spectrum changed in turn, windowed again and overlap-added, the output starting where
/// the input does.
struct Stft {
    size: usize,
    hop: usize,
    window: Vec<f64>,
    /// What the windows applied twice sum to, which the output is divided by.
    scale: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    /// The input not yet past, after the frame less a hop of zeros put before the stream, so
    /// that every sample of the stream has as many frames over it.
    input: Vec<f64>,
    /// The overlap-add of the frames put down, from the first sample not yet given.
    sum: Vec<f64>,
    skip: usize,
    stream: usize,
    given: usize,
    /// The frames put down.
    frames: usize,
}

impl Stft {
    fn new(size: usize, hop: usize) -> Self {
        let window: Vec<f64> = (0..size)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / size as f64).cos())
            .collect();
        let scale = (0..size / hop)
            .map(|j| window[j * hop].powi(2))
            .sum::<f64>();
        let mut planner = FftPlanner::new();
        Self {
            size,
            hop,
            window,
            scale,
            forward: planner.plan_fft_forward(size),
            inverse: planner.plan_fft_inverse(size),
            input: vec![0.0; size - hop],
            sum: vec![0.0; size],
            skip: size - hop,
            stream: 0,
            given: 0,
            frames: 0,
        }
    }

    /// Puts down the frame at the start of the input, its spectrum changed by `change`, which
    /// is given the frame's index and its bins, the first half and the middle, and gives the hop
    /// it finishes, up to `limit` in all.
    fn frame(
        &mut self,
        output: &mut Vec<f32>,
        limit: usize,
        change: &mut impl FnMut(usize, &mut [Complex<f64>]),
    ) {
        let size = self.size;
        let mut spectrum: Vec<Complex<f64>> = self.input[..size]
            .iter()
            .zip(&self.window)
            .map(|(&x, &w)| Complex::new(x * w, 0.0))
            .collect();
        self.forward.process(&mut spectrum);
        change(self.frames, &mut spectrum[..=size / 2]);
        for k in 1..size / 2 {
            spectrum[size - k] = spectrum[k].conj();
        }
        spectrum[0].im = 0.0;
        spectrum[size / 2].im = 0.0;
        self.inverse.process(&mut spectrum);
        let scale = 1.0 / (self.scale * size as f64);
        for ((s, c), &w) in self.sum.iter_mut().zip(&spectrum).zip(&self.window) {
            *s += c.re * w * scale;
        }
        let finished: Vec<f64> = self.sum.drain(..self.hop).collect();
        self.sum.resize(size, 0.0);
        self.input.drain(..self.hop);
        self.frames += 1;
        let drop = self.skip.min(finished.len());
        self.skip -= drop;
        let take = (finished.len() - drop).min(limit - self.given);
        output.extend(finished[drop..drop + take].iter().map(|&y| y as f32));
        self.given += take;
    }

    fn process(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
        mut change: impl FnMut(usize, &mut [Complex<f64>]),
    ) {
        self.input.extend(input.iter().map(|&x| f64::from(x)));
        self.stream += input.len();
        while self.input.len() >= self.size {
            self.frame(output, usize::MAX, &mut change);
        }
    }

    fn finish(
        &mut self,
        output: &mut Vec<f32>,
        mut change: impl FnMut(usize, &mut [Complex<f64>]),
    ) {
        while self.given < self.stream {
            self.input.resize(self.input.len().max(self.size), 0.0);
            self.frame(output, self.stream, &mut change);
        }
    }
}

/// The fish tongue's filter as ffmpeg's afftfilt makes it of the game's expression: each frame
/// of 1024 samples, a quarter frame apart, multiplied bin by bin by
/// `1 - x^2 + 2 exp(-ln(1 + x)^2 / 2) / sqrt(2 pi)`, `x` the bin's frequency over a cutoff set
/// once a frame from a sine that starts at the center and goes down first.
///
/// That is about +5 dB well under the cutoff, a notch at 1.27 times it, and above it a treble
/// boost that grows 12 dB an octave: the gain goes negative there and is never clamped.
pub struct Wobble {
    stft: Stft,
    center: f64,
    swing: f64,
    rate: f64,
}

/// The frame and hop of ffmpeg's afftfilt as the game sets it, at the rate the effects run at.
const WOBBLE_FRAME: usize = 1024;
const WOBBLE_HOP: usize = WOBBLE_FRAME / 4;

impl Wobble {
    /// The cutoff at `center - swing * sin(2 pi rate t)` Hz, `t` the time of each frame.
    pub fn new(center: f64, swing: f64, rate: f64) -> Self {
        Self {
            stft: Stft::new(WOBBLE_FRAME, WOBBLE_HOP),
            center,
            swing,
            rate,
        }
    }

    fn change(center: f64, swing: f64, rate: f64) -> impl FnMut(usize, &mut [Complex<f64>]) {
        move |frame, bins| {
            let time = (frame * WOBBLE_HOP) as f64 / RATE;
            let cutoff = center - swing * (2.0 * PI * rate * time).sin();
            for (k, bin) in bins.iter_mut().enumerate() {
                // afftfilt reads each bin at its upper edge.
                let x = (k as f64 + 0.5) * RATE / WOBBLE_FRAME as f64 / cutoff;
                let bell = (-(x + 1.0).ln().powi(2) / 2.0).exp() / (2.0 * PI).sqrt();
                *bin *= 1.0 - x * x + 2.0 * bell;
            }
        }
    }
}

impl Processor for Wobble {
    fn latency(&self) -> usize {
        WOBBLE_FRAME
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let change = Self::change(self.center, self.swing, self.rate);
        self.stft.process(input, output, change);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let change = Self::change(self.center, self.swing, self.rate);
        self.stft.finish(output, change);
    }
}

/// The STFT's frame for [`Formants`]: fine enough in frequency to tell the harmonics of a low
/// voice from its envelope.
const FORMANT_FRAME: usize = 2048;
/// The cepstral lifter's cutoff, in samples: 1 ms, well under the period of the highest voice,
/// so that the envelope is smooth over its harmonics.
const LIFTER: usize = 48;
/// The most [`Formants`] raises or lowers a frequency, in nepers of amplitude: 24 dB, so that
/// the valleys of the envelope are not raised into noise.
const MOST: f64 = 24.0 / 8.685_889_638;
/// What rubberband's formant=preserved takes off a voice beyond what keeping its frames' energy
/// would, which the game's fish tongue is played at today: 2 dB on average over the sample
/// voices, within 3 dB of each, in dB.
const FORMANT_LOSS: f64 = -2.0;

/// Each frame's spectral envelope moved the opposite way to a pitch shift of `ratio`: the
/// spectrum multiplied by `E(ratio * f) / E(f)`, `E` the frame's envelope by cepstral smoothing,
/// so that once the shift has scaled the frequencies by the ratio, the envelope is where it was.
/// Each frame keeps its energy, less the 2 dB rubberband takes off.
pub struct Formants {
    stft: Stft,
    ratio: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Formants {
    /// Moving the envelope against a pitch shift by `ratio`.
    pub fn new(ratio: f64) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            stft: Stft::new(FORMANT_FRAME, FORMANT_FRAME / 4),
            ratio,
            forward: planner.plan_fft_forward(FORMANT_FRAME),
            inverse: planner.plan_fft_inverse(FORMANT_FRAME),
        }
    }

    fn change(
        ratio: f64,
        forward: &Arc<dyn Fft<f64>>,
        inverse: &Arc<dyn Fft<f64>>,
    ) -> impl FnMut(usize, &mut [Complex<f64>]) {
        let (forward, inverse) = (forward.clone(), inverse.clone());
        move |_, bins| {
            // The real cepstrum of the frame, cut at the lifter: its log envelope.
            let mut cepstrum: Vec<Complex<f64>> = (0..FORMANT_FRAME)
                .map(|k| {
                    let bin = bins[k.min(FORMANT_FRAME - k)];
                    Complex::new(bin.norm().max(1e-9).ln(), 0.0)
                })
                .collect();
            inverse.process(&mut cepstrum);
            for (q, c) in cepstrum.iter_mut().enumerate() {
                let quefrency = q.min(FORMANT_FRAME - q);
                *c = if quefrency < LIFTER {
                    *c / FORMANT_FRAME as f64
                } else {
                    Complex::default()
                };
            }
            forward.process(&mut cepstrum);
            let envelope: Vec<f64> = cepstrum[..bins.len()].iter().map(|c| c.re).collect();
            let last = envelope.len() - 1;
            let energy = |bins: &[Complex<f64>]| bins.iter().map(Complex::norm_sqr).sum::<f64>();
            let before = energy(bins);
            for (k, bin) in bins.iter_mut().enumerate() {
                let at = (k as f64 * ratio).min(last as f64);
                let (i, t) = (at as usize, at.fract());
                let moved = envelope[i] + t * (envelope[(i + 1).min(last)] - envelope[i]);
                *bin *= (moved - envelope[k]).clamp(-MOST, MOST).exp();
            }
            let after = energy(bins).max(f64::MIN_POSITIVE);
            let scale = (before / after).sqrt() * 10f64.powf(FORMANT_LOSS / 20.0);
            bins.iter_mut().for_each(|bin| *bin *= scale);
        }
    }
}

impl Processor for Formants {
    fn latency(&self) -> usize {
        FORMANT_FRAME
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let change = Self::change(self.ratio, &self.forward, &self.inverse);
        self.stft.process(input, output, change);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let change = Self::change(self.ratio, &self.forward, &self.inverse);
        self.stft.finish(output, change);
    }
}
