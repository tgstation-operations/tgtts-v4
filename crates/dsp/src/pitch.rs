//! librosa's `effects.pitch_shift`: a phase-vocoder time stretch, resampled back to the length it
//! started at, whole or on a stream.
//!
//! librosa keeps the spectrogram in complex64, the vocoder's phase in float32 and the signal it
//! inverts to in float32, while it computes most steps in between in float64; each step here
//! rounds where librosa's does. The phase grows by up to 1600 radians a frame, so its float32
//! rounding is far from negligible: kept in float64 instead, the result agreed with librosa's to
//! 85 dB on a blip and 58 dB on 3 s of speech, rather than 134 dB on both.

use std::{collections::VecDeque, f64::consts::PI, sync::Arc};

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::resample::Resampler;

const N_FFT: usize = 2048;
const HOP: usize = N_FFT / 4;
const BINS: usize = N_FFT / 2 + 1;
/// The padding librosa's centered frames put before the signal, and after.
const HALF: usize = N_FFT / 2;

/// `librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)`:
/// `y` at `sr` Hz shifted by `n_steps` steps of `1 / bins_per_octave` octave, keeping its
/// length.
///
/// # Panics
///
/// If `bins_per_octave` is 0.
pub fn pitch_shift(y: &[f32], sr: u32, n_steps: f64, bins_per_octave: u32) -> Vec<f32> {
    let mut shifter = Shifter::new(sr, n_steps, bins_per_octave);
    let mut shifted = Vec::with_capacity(y.len());
    shifter.process(y, &mut shifted);
    shifter.finish(&mut shifted);
    shifted
}

type Column = Vec<Complex<f32>>;

/// [`pitch_shift`] on a stream: the same samples, to the bit, a chunk at a time, behind the
/// input by what its frames, the vocoder and libsoxr's filter hold back, in all 57 ms for an
/// octave up and 112 ms for an octave down, in 10 ms chunks.
///
/// Each step of librosa's but the last is local or causal: a column of the STFT needs the input
/// half a frame past its center; a frame of the phase vocoder, the columns either side of where
/// it falls, and the phase of the frame before; the inverse, the frames over each sample; and
/// libsoxr's output does not depend on how its input comes. Only the end depends on the length
/// of the whole, where the vocoder's columns run out, the inverse's last frames and its length
/// are cut, and the resampled signal is cut or padded to the input's length: those wait for
/// [`finish`](Self::finish).
pub struct Shifter {
    /// The time stretch, librosa's `rate`.
    rate: f64,
    sr: u32,
    window: Vec<f64>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    buffer: Vec<Complex<f64>>,
    phi_advance: Vec<f64>,
    /// The input as the STFT's frames read it, after [`HALF`] zeros, from `input_start`.
    input: Vec<f32>,
    input_start: usize,
    taken: usize,
    /// The STFT's columns made, those still needed kept from `columns_start`.
    columns: VecDeque<Column>,
    columns_start: usize,
    made: usize,
    /// The vocoder's phase, once the first column is made.
    phase: Option<Vec<f32>>,
    /// The vocoder's frames made, each overlap-added into `signal` as it is made.
    frames: usize,
    /// The inverse's signal, as librosa's buffer padded by [`HALF`] holds it, from
    /// `signal_start`.
    signal: VecDeque<f32>,
    signal_start: usize,
    /// The samples of the stretched signal resampled, and the resampler, where the stretch
    /// needs one.
    stretched: usize,
    resampler: Option<Resampler<f32>>,
    /// What the resampler has given and the shifter has not, from sample `given` of its output.
    resampled: VecDeque<f32>,
    given: usize,
}

impl Shifter {
    /// A shifter of audio at `sr` Hz by `n_steps` steps of `1 / bins_per_octave` octave.
    ///
    /// # Panics
    ///
    /// If `bins_per_octave` is 0.
    pub fn new(sr: u32, n_steps: f64, bins_per_octave: u32) -> Self {
        assert!(bins_per_octave > 0, "bins_per_octave of 0");
        let rate = 2f64.powf(-n_steps / f64::from(bins_per_octave));
        let (from, to) = resampling(sr, rate);
        let mut planner = FftPlanner::new();
        // numpy's rfftfreq at sr = 2 pi, times the hop.
        let d = 1.0 / (2.0 * PI);
        let val = 1.0 / (N_FFT as f64 * d);
        Self {
            rate,
            sr,
            window: hann(),
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
            buffer: vec![Complex::default(); N_FFT],
            phi_advance: (0..BINS).map(|k| HOP as f64 * (k as f64 * val)).collect(),
            input: vec![0.0; HALF],
            input_start: 0,
            taken: 0,
            columns: VecDeque::new(),
            columns_start: 0,
            made: 0,
            phase: None,
            frames: 0,
            signal: VecDeque::new(),
            signal_start: 0,
            stretched: 0,
            resampler: (from != to).then(|| Resampler::new(from, to)),
            resampled: VecDeque::new(),
            given: 0,
        }
    }

    /// Takes `input` and appends to `output` the samples it has finished.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.input.extend_from_slice(input);
        self.taken += input.len();
        while self.made * HOP + N_FFT <= self.input_start + self.input.len() {
            self.analyze();
        }
        while self.made > (self.frames as f64 * self.rate) as usize + 1 {
            let frame = self.vocode(self.made);
            self.add(&frame);
        }
        let done = (self.frames * HOP).saturating_sub(HALF);
        let mut stretched = Vec::new();
        self.stretch(done, usize::MAX, &mut stretched);
        self.resample(&stretched);
        // The resampled signal is cut or padded to its length at the end, past which it never
        // gives a sample: that length is within half the stretch of the input's.
        let safe = self
            .taken
            .saturating_sub((self.rate / 2.0).ceil() as usize + 2);
        self.give(safe, output);
    }

    /// Appends to `output` the samples it still holds, as the stream ends.
    pub fn finish(mut self, output: &mut Vec<f32>) {
        let len = self.taken;
        if len == 0 {
            return;
        }
        self.input.resize(self.input.len() + HALF, 0.0);
        let columns = 1 + len / HOP;
        while self.made < columns {
            self.analyze();
        }
        let steps = (columns as f64 / self.rate).ceil() as usize;
        let len_out = (len as f64 / self.rate).round_ties_even() as usize;
        let frames = steps.min((len_out + 2 * HALF).div_ceil(HOP));
        while self.frames < frames {
            let frame = self.vocode(columns);
            self.add(&frame);
        }
        let mut stretched = Vec::new();
        self.stretch(len_out, frames, &mut stretched);
        self.resample(&stretched);
        if let Some(resampler) = self.resampler.take() {
            let mut rest = Vec::new();
            resampler.finish(&mut rest);
            self.resampled.extend(rest);
        }
        // librosa's resample cuts or pads to its length, and pitch_shift to the input's.
        let (from, to) = resampling(self.sr, self.rate);
        let kept = if from == to {
            len_out
        } else {
            (len_out as f64 * (to / from)).ceil() as usize
        };
        self.resampled.truncate(kept - self.given);
        self.resampled.resize(kept - self.given, 0.0);
        self.resampled.resize(len - self.given, 0.0);
        output.extend(self.resampled.drain(..));
    }

    /// Makes the next column of the STFT, from the frame of the input centered on it: windowed
    /// and transformed in float64, stored as complex64.
    fn analyze(&mut self) {
        let at = self.made * HOP - self.input_start;
        let frame = &self.input[at..at + N_FFT];
        for ((b, &w), &s) in self.buffer.iter_mut().zip(&self.window).zip(frame) {
            *b = Complex::new(w * f64::from(s), 0.0);
        }
        self.forward.process(&mut self.buffer);
        let column: Column = self.buffer[..BINS]
            .iter()
            .map(|c| Complex::new(c.re as f32, c.im as f32))
            .collect();
        if self.phase.is_none() {
            self.phase = Some(column.iter().map(|c| angle(*c)).collect());
        }
        self.columns.push_back(column);
        self.made += 1;
        // The next column's frame starts a hop on.
        let spent = self.made * HOP - self.input_start;
        self.input.drain(..spent);
        self.input_start += spent;
    }

    /// Makes the vocoder's next frame from the columns either side of where it falls, those
    /// from `columns` on being zeros.
    fn vocode(&mut self, columns: usize) -> Column {
        let step = self.frames as f64 * self.rate;
        let at = step as usize;
        let zeros = vec![Complex::default(); BINS];
        let column = |i: usize| {
            if i < columns {
                &self.columns[i - self.columns_start]
            } else {
                &zeros
            }
        };
        let (c0, c1) = (column(at), column(at + 1));
        let alpha = step % 1.0;
        let phase = self.phase.as_mut().expect("the first column is made");
        let mut out = Vec::with_capacity(BINS);
        for k in 0..BINS {
            let mag = (1.0 - alpha) * f64::from(abs(c0[k])) + alpha * f64::from(abs(c1[k]));
            let (sin, cos) = (phase[k].sin(), phase[k].cos());
            out.push(Complex::new(
                (f64::from(cos) * mag) as f32,
                (f64::from(sin) * mag) as f32,
            ));
            let dphase = f64::from(angle(c1[k]) - angle(c0[k])) - self.phi_advance[k];
            let dphase = dphase - 2.0 * PI * (dphase / (2.0 * PI)).round_ties_even();
            phase[k] = (f64::from(phase[k]) + (self.phi_advance[k] + dphase)) as f32;
        }
        self.frames += 1;
        // The next frame falls at or after its column.
        let next = ((self.frames as f64 * self.rate) as usize).min(self.made);
        while self.columns_start < next {
            self.columns.pop_front();
            self.columns_start += 1;
        }
        out
    }

    /// Overlap-adds the vocoder's frame just made into the signal: inverted to float32,
    /// windowed in float64 and added in float32.
    fn add(&mut self, column: &Column) {
        // scipy's irfft takes only the real parts of the first and last bins.
        self.buffer[0] = Complex::new(f64::from(column[0].re), 0.0);
        self.buffer[N_FFT / 2] = Complex::new(f64::from(column[N_FFT / 2].re), 0.0);
        for k in 1..N_FFT / 2 {
            let c = Complex::new(f64::from(column[k].re), f64::from(column[k].im));
            self.buffer[k] = c;
            self.buffer[N_FFT - k] = c.conj();
        }
        self.inverse.process(&mut self.buffer);
        let start = (self.frames - 1) * HOP;
        let end = start + N_FFT - self.signal_start;
        if self.signal.len() < end {
            self.signal.resize(end, 0.0);
        }
        for i in 0..N_FFT {
            let sample = (self.buffer[i].re / N_FFT as f64) as f32;
            let windowed = self.window[i] * f64::from(sample);
            let at = start + i - self.signal_start;
            self.signal[at] = (f64::from(self.signal[at]) + windowed) as f32;
        }
    }

    /// Appends to `out` the stretched signal's samples up to `end`, each divided by the squared
    /// window summed in float32 over the first `frames` frames over it, as librosa's
    /// `window_sumsquare` sums it, where that sum is not too small to divide by.
    fn stretch(&mut self, end: usize, frames: usize, out: &mut Vec<f32>) {
        while self.stretched < end {
            let p = HALF + self.stretched;
            let first = (p + 1).saturating_sub(N_FFT).div_ceil(HOP);
            let last = (p / HOP).min(frames.saturating_sub(1));
            let mut sum = 0f32;
            for t in first..=last {
                if t < frames {
                    let w = self.window[p - t * HOP];
                    sum = (f64::from(sum) + w * w) as f32;
                }
            }
            let mut sample = self
                .signal
                .get(p - self.signal_start)
                .copied()
                .unwrap_or(0.0);
            if sum > f32::MIN_POSITIVE {
                sample /= sum;
            }
            out.push(sample);
            self.stretched += 1;
        }
        // Kept from the next sample out, or from where the next frame adds, if that is before.
        let spent = (HALF + self.stretched)
            .min(self.frames * HOP)
            .saturating_sub(self.signal_start)
            .min(self.signal.len());
        self.signal.drain(..spent);
        self.signal_start += spent;
    }

    fn resample(&mut self, stretched: &[f32]) {
        match &mut self.resampler {
            Some(resampler) => {
                let mut out = Vec::new();
                resampler.process(stretched, &mut out);
                self.resampled.extend(out);
            }
            None => self.resampled.extend(stretched),
        }
    }

    /// Appends to `output` the resampled samples up to `end`, of those there are.
    fn give(&mut self, end: usize, output: &mut Vec<f32>) {
        let end = end.min(self.given + self.resampled.len());
        if end > self.given {
            output.extend(self.resampled.drain(..end - self.given));
            self.given = end;
        }
    }
}

/// The rates librosa resamples a stretch by `rate` back from and to, at `sr` Hz.
fn resampling(sr: u32, rate: f64) -> (f64, f64) {
    (f64::from(sr) / rate, f64::from(sr))
}

/// scipy's periodic Hann window of `N_FFT`, `0.5 + 0.5 cos` over `linspace(-pi, pi)`.
fn hann() -> Vec<f64> {
    let step = (PI - -PI) / N_FFT as f64;
    (0..N_FFT)
        .map(|n| 0.5 + 0.5 * (n as f64 * step + -PI).cos())
        .collect()
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

    /// librosa's `stft(y)`, as the shifter makes its columns.
    fn stft(y: &[f32]) -> Vec<Column> {
        let mut shifter = Shifter::new(48000, 0.0, 24);
        shifter.input.extend_from_slice(y);
        shifter.input.resize(shifter.input.len() + HALF, 0.0);
        while shifter.made < 1 + y.len() / HOP {
            shifter.analyze();
        }
        shifter.columns.into()
    }

    /// librosa's `phase_vocoder(D, rate=rate)`, as the shifter makes its frames.
    fn phase_vocoder(spectrum: &[Column], rate: f64) -> Vec<Column> {
        let mut shifter = Shifter::new(48000, 0.0, 24);
        shifter.rate = rate;
        shifter.columns = spectrum.iter().cloned().collect();
        shifter.made = spectrum.len();
        shifter.phase = Some(spectrum[0].iter().map(|c| angle(*c)).collect());
        let steps = (spectrum.len() as f64 / rate).ceil() as usize;
        (0..steps).map(|_| shifter.vocode(spectrum.len())).collect()
    }

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
        let spectrum = stft(&from_pcm(&input));
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
