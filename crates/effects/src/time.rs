//! Effects on time: tempo, by a phase vocoder or by splicing, WSOLA; tape speed, by a
//! windowed-sinc resampler; and pitch, the two together, with the formants kept by a spectral
//! envelope if asked.
//!
//! The vocoder sounds as rubberband does, smooth with a little phasiness, and the splicing as
//! ffmpeg's atempo, itself WSOLA, crisp with now and then a doubled syllable: each is used where
//! the game used the one it sounds like. Each gives `round(n / ratio)` samples for `n` in, `n`
//! for pitch.

use std::{
    f64::consts::PI,
    sync::{Arc, LazyLock},
};

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{Processor, Series, schema::Method, spectral::Formants};

/// The length of WSOLA's frames: two periods of a low voice, short enough that a syllable's
/// changes do not smear.
const FRAME: usize = 1536;
/// How far apart the frames are put down, half a frame, where the Hann windows sum to one.
const HOP: usize = FRAME / 2;
/// How far either way a frame is moved to line up with the last one's continuation: more than
/// half the period of a low voice, so that some offset always lines the periods up.
const SEEK: usize = 480;
/// The step of the coarse search, which the fine one then searches either side of.
const COARSE: usize = 4;

/// The tempo times a ratio, the pitch kept, by the waveform-similarity overlap-add of Verhelst
/// and Roelands, "An overlap-add technique based on waveform similarity (WSOLA) for high quality
/// time-scale modification of speech" (ICASSP 1993).
///
/// Frames are taken from the input at `ratio` times the hop they are put down at, each moved by
/// up to 10 ms to where it best matches how the frame before it would have gone on.
#[derive(Clone, Debug)]
pub struct Splice {
    ratio: f64,
    window: Vec<f64>,
    /// The input from `start`, after `SEEK + HOP` zeros put before the stream, so that the first
    /// frame can be centred on its first sample and moved back.
    input: Vec<f64>,
    start: usize,
    /// The input's length with the zeros, and the stream's.
    taken: usize,
    stream: usize,
    /// The next frame, and where the last one was taken from.
    frame: usize,
    last: Option<usize>,
    /// The second half of the last frame, which the next is added to.
    tail: Vec<f64>,
    /// The output still to drop, the half frame before the stream's start, and the output given.
    skip: usize,
    given: usize,
}

impl Splice {
    /// The tempo times `ratio`.
    pub fn new(ratio: f64) -> Self {
        let pad = SEEK + HOP;
        Self {
            ratio,
            window: (0..FRAME)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / FRAME as f64).cos())
                .collect(),
            input: vec![0.0; pad],
            start: 0,
            taken: pad,
            stream: 0,
            frame: 0,
            last: None,
            tail: vec![0.0; HOP],
            skip: HOP,
            given: 0,
        }
    }

    /// Where frame `k` would be taken from with no search: its middle at the input the middle of
    /// where it is put down stands for.
    fn nominal(&self, k: usize) -> usize {
        SEEK + (k as f64 * HOP as f64 * self.ratio).round() as usize
    }

    /// The input from `i` on, which it still holds.
    fn from(&self, i: usize) -> &[f64] {
        &self.input[i - self.start..]
    }

    /// The offset of frame `k` that best matches the continuation of the frame before it.
    fn search(&self, k: usize) -> usize {
        let nominal = self.nominal(k);
        let Some(last) = self.last else {
            return nominal;
        };
        let target = &self.from(last + HOP)[..HOP];
        let low = nominal - SEEK;
        let span = &self.from(low)[..2 * SEEK + HOP];
        let mut energy = Vec::with_capacity(span.len() + 1);
        energy.push(0.0);
        for x in span {
            energy.push(energy[energy.len() - 1] + x * x);
        }
        let score = |d: usize| {
            let power: f64 = energy[d + HOP] - energy[d];
            dot(&span[d..d + HOP], target) / power.max(1e-18).sqrt()
        };
        let best = |offsets: &mut dyn Iterator<Item = usize>| {
            offsets
                .map(|d| (score(d), d))
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .expect("the range is not empty")
                .1
        };
        let coarse = best(&mut (0..=2 * SEEK).step_by(COARSE));
        let fine =
            best(&mut (coarse.saturating_sub(COARSE - 1)..=(coarse + COARSE - 1).min(2 * SEEK)));
        low + fine
    }

    /// The input frame `k` and its search need.
    fn needs(&self, k: usize) -> usize {
        let continuation = self.last.map_or(0, |last| last + 2 * HOP);
        (self.nominal(k) + SEEK + FRAME).max(continuation)
    }

    /// Puts down every frame the input has, then gives what is finished.
    fn run(&mut self, output: &mut Vec<f32>, limit: Option<usize>) {
        while self.needs(self.frame) <= self.taken {
            if limit.is_some_and(|limit| self.given >= limit) {
                break;
            }
            let from = self.search(self.frame);
            let frame = &self.from(from)[..FRAME];
            let finished: Vec<f64> = (0..HOP)
                .map(|i| self.tail[i] + self.window[i] * frame[i])
                .collect();
            let tail: Vec<f64> = (0..HOP)
                .map(|i| self.window[HOP + i] * frame[HOP + i])
                .collect();
            self.tail = tail;
            self.last = Some(from);
            self.frame += 1;
            let drop = self.skip.min(finished.len());
            self.skip -= drop;
            let mut finished = &finished[drop..];
            if let Some(limit) = limit {
                finished = &finished[..finished.len().min(limit - self.given)];
            }
            self.given += finished.len();
            output.extend(finished.iter().map(|&y| y as f32));
        }
        // Keeps the input from the earliest the next frame can reach.
        let keep = self.nominal(self.frame) - SEEK;
        let keep = self.last.map_or(keep, |last| keep.min(last + HOP));
        if keep > self.start {
            self.input
                .drain(..(keep - self.start).min(self.input.len()));
            self.start = keep;
        }
    }
}

impl Processor for Splice {
    fn latency(&self) -> usize {
        FRAME + 2 * SEEK
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.input.extend(input.iter().map(|&x| f64::from(x)));
        self.taken += input.len();
        self.stream += input.len();
        self.run(output, None);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let total = (self.stream as f64 / self.ratio).round() as usize;
        while self.given < total {
            let zeros = self.needs(self.frame).saturating_sub(self.taken).max(HOP);
            self.input.resize(self.input.len() + zeros, 0.0);
            self.taken += zeros;
            self.run(output, Some(total));
        }
    }
}

/// The dot product of `a` and `b`, summed in four lanes, which the compiler can vectorize.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0; 4];
    let (chunks, rest) = a.as_chunks::<4>();
    let (others, other_rest) = b.as_chunks::<4>();
    for (x, y) in chunks.iter().zip(others) {
        for j in 0..4 {
            lanes[j] += x[j] * y[j];
        }
    }
    let rest: f64 = rest.iter().zip(other_rest).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f64>() + rest
}

/// The half-width of the resampler's kernel at the input's rate, in zero crossings.
const CROSSINGS: usize = 16;
/// The points of the kernel's table between zero crossings.
const RESOLUTION: usize = 512;
/// The resampler's cutoff, as a fraction of the lower Nyquist frequency, leaving room for the
/// kernel's transition band.
const CUTOFF: f64 = 0.95;

/// sinc times a Kaiser window, beta 8.6, over `CROSSINGS` zero crossings each side, tabulated
/// from the middle out.
static KERNEL: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let beta: f64 = 8.6;
    (0..=CROSSINGS * RESOLUTION + 1)
        .map(|i| {
            let t = i as f64 / RESOLUTION as f64;
            let sinc = if i == 0 {
                1.0
            } else {
                (PI * t).sin() / (PI * t)
            };
            let r = (t / CROSSINGS as f64).min(1.0);
            sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta)
        })
        .collect()
});

/// The modified Bessel function of the first kind, of order 0, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
    }
    sum
}

/// Tape speed times a ratio: the input read at `ratio` samples a sample, through a windowed sinc
/// that, speeding up, also takes out what would fold back over the Nyquist frequency.
#[derive(Clone, Debug)]
pub struct Speed {
    ratio: f64,
    /// The kernel's scale: the cutoff, as a fraction of the input's Nyquist frequency.
    scale: f64,
    /// The kernel's half-width, in input samples.
    reach: usize,
    input: Vec<f64>,
    start: usize,
    stream: usize,
    given: usize,
}

impl Speed {
    /// Tape speed times `ratio`.
    pub fn new(ratio: f64) -> Self {
        let scale = CUTOFF * ratio.recip().min(1.0);
        Self {
            ratio,
            scale,
            reach: (CROSSINGS as f64 / scale).ceil() as usize,
            input: Vec::new(),
            start: 0,
            stream: 0,
            given: 0,
        }
    }

    fn sample(&self, position: f64) -> f64 {
        let first = (position - self.reach as f64).ceil().max(0.0) as usize;
        let last = (position + self.reach as f64).floor() as usize;
        let mut sum = 0.0;
        for i in first..=last {
            let Some(&x) = self.input.get(i.wrapping_sub(self.start)) else {
                continue;
            };
            let at = (position - i as f64).abs() * self.scale * RESOLUTION as f64;
            let j = at as usize;
            if j >= CROSSINGS * RESOLUTION {
                continue;
            }
            let frac = at - j as f64;
            sum += x * (KERNEL[j] + frac * (KERNEL[j + 1] - KERNEL[j]));
        }
        sum * self.scale
    }

    /// Gives every output whose kernel the input covers, up to `total`.
    fn run(&mut self, output: &mut Vec<f32>, covered: usize, total: usize) {
        while self.given < total {
            let position = self.given as f64 * self.ratio;
            if position as usize + self.reach >= covered {
                break;
            }
            output.push(self.sample(position) as f32);
            self.given += 1;
        }
        let keep = (self.given as f64 * self.ratio) as usize;
        let keep = keep.saturating_sub(self.reach + 1);
        if keep > self.start {
            self.input
                .drain(..(keep - self.start).min(self.input.len()));
            self.start = keep;
        }
    }
}

impl Processor for Speed {
    fn latency(&self) -> usize {
        self.reach + 1
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.input.extend(input.iter().map(|&x| f64::from(x)));
        self.stream += input.len();
        self.run(output, self.stream, usize::MAX);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let total = (self.stream as f64 / self.ratio).round() as usize;
        self.run(output, usize::MAX, total);
    }
}

/// The tempo times `ratio` by `method`.
pub fn tempo(ratio: f64, method: Method) -> Box<dyn Processor> {
    match method {
        Method::Vocoder => Box::new(Vocoder::new(ratio)),
        Method::Splice => Box::new(Splice::new(ratio)),
    }
}

/// The pitch shifted, the duration kept: the tempo and the speed changed by the shift in turn,
/// which moves the formants with the pitch unless [`Formants`] has moved them the other way
/// first.
pub struct Pitch {
    series: Series,
    stream: usize,
    given: usize,
}

impl Pitch {
    /// The pitch shifted by `semitones` by `method`, the formants kept where they are if
    /// `keep_formants`.
    ///
    /// Splicing comes after the speed, as the game's `asetrate` and `atempo` came in ffmpeg,
    /// and the vocoder before it, as rubberband shifts.
    pub fn new(semitones: f64, keep_formants: bool, method: Method) -> Self {
        let ratio = 2f64.powf(semitones / 12.0);
        let mut stages: Vec<Box<dyn Processor>> = Vec::new();
        if keep_formants {
            stages.push(Box::new(Formants::new(ratio)));
        }
        match method {
            Method::Vocoder => {
                stages.push(tempo(ratio.recip(), method));
                stages.push(Box::new(Speed::new(ratio)));
            }
            Method::Splice => {
                stages.push(Box::new(Speed::new(ratio)));
                stages.push(tempo(ratio.recip(), method));
            }
        }
        Self {
            series: Series(stages),
            stream: 0,
            given: 0,
        }
    }
}

impl Processor for Pitch {
    fn latency(&self) -> usize {
        self.series.latency()
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.stream += input.len();
        let before = output.len();
        self.series.process(input, output);
        output.truncate(before + (self.stream - self.given).min(output.len() - before));
        self.given += output.len() - before;
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let before = output.len();
        self.series.finish(output);
        output.resize(before + (self.stream - self.given), 0.0);
        self.given = self.stream;
    }
}

/// The phase vocoder's frame: two periods of the lowest voices at the finest, as rubberband's
/// standard window is at 48 kHz.
const VOCODER_FRAME: usize = 2048;
/// How far apart its frames are put down: a quarter frame, where Hann windows applied twice
/// sum to a constant.
const VOCODER_HOP: usize = VOCODER_FRAME / 4;
const VOCODER_BINS: usize = VOCODER_FRAME / 2 + 1;
/// What rubberband takes off a voice it stretches, which the game's voices are played at today:
/// 1.5 to 2 dB on the sample voices once the stretch is a quarter octave, less below it.
const VOCODER_LOSS: f64 = -1.7;

/// The tempo times a ratio, the pitch kept, by a phase vocoder with identity phase locking, as
/// Laroche and Dolson's "Improved phase vocoder time-scale modification of audio" (IEEE Trans.
/// Speech and Audio Processing, 1999) has it: frames taken at `ratio` times the hop they are put
/// down at, each bin's phase advanced by the frequency its own advance measures, and the bins
/// around each peak of the spectrum kept in the phase they had relative to the peak. Its level
/// comes down as rubberband's does, by 1.7 dB once the stretch is a quarter octave.
pub struct Vocoder {
    ratio: f64,
    gain: f64,
    window: Vec<f64>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    /// The input from `start`, after zeros put before the stream so that a frame's middle stands
    /// for where it is put down.
    input: Vec<f64>,
    start: usize,
    taken: usize,
    stream: usize,
    /// The next frame, and the start and the phases of the last one taken, and the phases it was
    /// put down with.
    frame: usize,
    last: Option<(usize, Vec<f64>, Vec<f64>)>,
    sum: Vec<f64>,
    skip: usize,
    given: usize,
}

impl Vocoder {
    /// The tempo times `ratio`.
    pub fn new(ratio: f64) -> Self {
        let mut planner = FftPlanner::new();
        // So that the middle of frame k, put down at k hops less the skip, stands for the input
        // at `ratio` times that: the skip a frame less a hop, as the overlap-add needs.
        let skip = VOCODER_FRAME - VOCODER_HOP;
        let half = (VOCODER_FRAME / 2) as f64;
        let pad = (half - ratio * (half - skip as f64)).round() as usize;
        let loss = VOCODER_LOSS * (ratio.log2().abs() / 0.25).min(1.0);
        Self {
            ratio,
            gain: 10f64.powf(loss / 20.0),
            window: (0..VOCODER_FRAME)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / VOCODER_FRAME as f64).cos())
                .collect(),
            forward: planner.plan_fft_forward(VOCODER_FRAME),
            inverse: planner.plan_fft_inverse(VOCODER_FRAME),
            input: vec![0.0; pad],
            start: 0,
            taken: pad,
            stream: 0,
            frame: 0,
            last: None,
            sum: vec![0.0; VOCODER_FRAME],
            skip,
            given: 0,
        }
    }

    /// Where frame `k` is taken from.
    fn at(&self, k: usize) -> usize {
        (k as f64 * VOCODER_HOP as f64 * self.ratio).round() as usize
    }

    fn put_down(&mut self, output: &mut Vec<f32>, limit: usize) {
        let from = self.at(self.frame);
        let mut spectrum: Vec<Complex<f64>> = self.input[from - self.start..][..VOCODER_FRAME]
            .iter()
            .zip(&self.window)
            .map(|(&x, &w)| Complex::new(x * w, 0.0))
            .collect();
        self.forward.process(&mut spectrum);
        let magnitude: Vec<f64> = spectrum[..VOCODER_BINS].iter().map(|c| c.norm()).collect();
        let phase: Vec<f64> = spectrum[..VOCODER_BINS].iter().map(|c| c.arg()).collect();
        let put = match &self.last {
            None => phase.clone(),
            Some((before, analysed, put)) => {
                let advance = (from - before) as f64;
                // Each peak's phase advanced by the frequency its own advance measures, and the
                // bins nearest it kept in the phase they have relative to it.
                let peaks: Vec<usize> = (0..VOCODER_BINS)
                    .filter(|&k| {
                        (k == 0 || magnitude[k] > magnitude[k - 1])
                            && (k + 1 == VOCODER_BINS || magnitude[k] >= magnitude[k + 1])
                    })
                    .collect();
                let mut locked = vec![0.0; VOCODER_BINS];
                for (i, &peak) in peaks.iter().enumerate() {
                    let omega = 2.0 * PI * peak as f64 / VOCODER_FRAME as f64;
                    let expected = analysed[peak] + omega * advance;
                    let deviation = wrap(phase[peak] - expected);
                    let frequency = omega + deviation / advance.max(1.0);
                    let peak_phase = put[peak] + frequency * VOCODER_HOP as f64;
                    let low = if i == 0 {
                        0
                    } else {
                        (peaks[i - 1] + peak) / 2 + 1
                    };
                    let high = peaks
                        .get(i + 1)
                        .map_or(VOCODER_BINS, |&next| (peak + next) / 2 + 1);
                    for k in low..high {
                        locked[k] = peak_phase + phase[k] - phase[peak];
                    }
                }
                locked
            }
        };
        for k in 0..VOCODER_BINS {
            spectrum[k] = Complex::from_polar(magnitude[k], put[k]);
        }
        for k in 1..VOCODER_FRAME / 2 {
            spectrum[VOCODER_FRAME - k] = spectrum[k].conj();
        }
        spectrum[0].im = 0.0;
        spectrum[VOCODER_FRAME / 2].im = 0.0;
        self.inverse.process(&mut spectrum);
        // Hann windows a quarter frame apart, applied twice, sum to 1.5.
        let scale = self.gain / (1.5 * VOCODER_FRAME as f64);
        for ((s, c), &w) in self.sum.iter_mut().zip(&spectrum).zip(&self.window) {
            *s += c.re * w * scale;
        }
        self.last = Some((from, phase, put.iter().map(|&p| wrap(p)).collect()));
        self.frame += 1;
        let finished: Vec<f64> = self.sum.drain(..VOCODER_HOP).collect();
        self.sum.resize(VOCODER_FRAME, 0.0);
        let drop = self.skip.min(finished.len());
        self.skip -= drop;
        let take = (finished.len() - drop).min(limit - self.given);
        output.extend(finished[drop..drop + take].iter().map(|&y| y as f32));
        self.given += take;
        let keep = self.at(self.frame).min(from);
        if keep > self.start {
            self.input.drain(..keep - self.start);
            self.start = keep;
        }
    }

    fn run(&mut self, output: &mut Vec<f32>, limit: usize) {
        while self.given < limit && self.at(self.frame) + VOCODER_FRAME <= self.taken {
            self.put_down(output, limit);
        }
    }
}

/// `phase` wrapped to [-pi, pi).
fn wrap(phase: f64) -> f64 {
    phase - 2.0 * PI * ((phase + PI) / (2.0 * PI)).floor()
}

impl Processor for Vocoder {
    fn latency(&self) -> usize {
        2 * VOCODER_FRAME
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.input.extend(input.iter().map(|&x| f64::from(x)));
        self.taken += input.len();
        self.stream += input.len();
        self.run(output, usize::MAX);
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        let total = (self.stream as f64 / self.ratio).round() as usize;
        while self.given < total {
            let zeros = (self.at(self.frame) + VOCODER_FRAME)
                .saturating_sub(self.taken)
                .max(VOCODER_HOP);
            self.input.resize(self.input.len() + zeros, 0.0);
            self.taken += zeros;
            self.run(output, total);
        }
    }
}
