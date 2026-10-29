//! pydub's `AudioSegment`, as far as the old service uses it, on mono 16-bit PCM.
//!
//! pydub positions everything in milliseconds, rounds a segment's length to whole ones and turns
//! positions into frames through floating point, then works on the samples with audioop's integer
//! arithmetic. Each operation here takes the same steps, so that it gives the same samples:
//! frames a slice drops or pads with silence where milliseconds and frames disagree included.

use std::ops::Range;

use crate::quantity::{Gain, Level, Millis, Ratio, SampleRate};

/// The gain a fade starts or ends at, pydub's `db_to_float(-120)`.
const FADE_FLOOR_DB: Gain = Gain::new(-120.0);

/// Full scale for 16-bit samples, as pydub's `max_possible_amplitude` has it.
const FULL_SCALE: f64 = 32768.0;

/// Mono 16-bit PCM at a sample rate, as a pydub `AudioSegment` of that format holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// Frames per second.
    pub rate: SampleRate,
    /// The samples.
    pub samples: Vec<i16>,
}

/// Something pydub raises, where the old service then failed the request.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A crossfade longer than the segment of this length it would fade out.
    #[error(
        "crossfade of {} ms is longer than a segment of {} ms",
        .crossfade.value(),
        .len.value()
    )]
    CrossfadeTooLong {
        /// The crossfade.
        crossfade: Millis,
        /// The length of the segment it is longer than.
        len: Millis,
    },
    /// A slice that pydub would have to pad with more than 2 ms of silence.
    #[error("slice would pad {0} missing frames with silence")]
    TooManyMissingFrames(i64),
    /// A fade of at most 100 ms over no frames, whose gain step pydub divides by zero for.
    #[error("fade over no frames")]
    EmptyFade,
}

impl Segment {
    /// A segment of `samples` at `rate` frames per second.
    pub fn new(rate: SampleRate, samples: Vec<i16>) -> Self {
        Self { rate, samples }
    }

    /// `AudioSegment.silent(duration, rate)`: `duration` of zeros.
    pub fn silent(duration: Millis, rate: SampleRate) -> Self {
        let frames = (f64::from(rate) * (duration.value() as f64 / 1000.0)) as usize;
        Self::new(rate, vec![0; frames])
    }

    /// `len(segment)`: the length in ms, rounded half to even.
    pub fn len_ms(&self) -> Millis {
        let ms = 1000.0 * (self.samples.len() as f64 / f64::from(self.rate));
        Millis::new(ms.round_ties_even() as i64)
    }

    /// `segment.rms`: audioop's root mean square, truncated to an integer.
    pub fn rms(&self) -> u32 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum_squares = self
            .samples
            .iter()
            .fold(0.0, |sum, &s| sum + f64::from(s) * f64::from(s));
        (sum_squares / self.samples.len() as f64).sqrt() as u32
    }

    /// `segment.max`: the largest magnitude of a sample.
    pub fn max(&self) -> u32 {
        self.samples
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .map_or(0, u32::from)
    }

    /// `segment.dBFS`: the level of the RMS against full scale, `-inf` for silence.
    pub fn dbfs(&self) -> Level {
        Level::from_full_scale(ratio_to_db(Ratio::new(f64::from(self.rms()) / FULL_SCALE)))
    }

    /// `segment.max_dBFS`: the level of the peak against full scale, `-inf` for silence.
    pub fn max_dbfs(&self) -> Level {
        Level::from_full_scale(ratio_to_db(Ratio::new(f64::from(self.max()) / FULL_SCALE)))
    }

    /// `segment.apply_gain(gain)`: every sample raised by `gain`, clipped and floored as audioop
    /// does.
    pub fn apply_gain(&self, gain: Gain) -> Self {
        self.spawn(mul(&self.samples, Ratio::of(gain).value()))
    }

    /// `pydub.effects.normalize(segment, headroom)`: the peak brought to `headroom` below full
    /// scale, silence left as it is.
    pub fn normalize(&self, headroom: Gain) -> Self {
        let peak = self.max();
        if peak == 0 {
            return self.clone();
        }
        let target_peak = FULL_SCALE * Ratio::of(-headroom).value();
        self.apply_gain(ratio_to_db(Ratio::new(target_peak / f64::from(peak))))
    }

    /// `segment.reverse()`.
    pub fn reverse(&self) -> Self {
        self.spawn(self.samples.iter().rev().copied().collect())
    }

    /// `segment[start:end]`, positions in ms: `None` for the start or the end, negative counting
    /// back from the end.
    pub fn slice(&self, start: Option<Millis>, end: Option<Millis>) -> Result<Self, Error> {
        let len = self.len_ms();
        let start = start.unwrap_or(Millis::ZERO).min(len);
        let end = end.unwrap_or(len).min(len);
        self.frames(self.position(start), self.position(end))
    }

    /// `segment[ms]`: the frames of one millisecond.
    fn ms(&self, ms: Millis) -> Result<Self, Error> {
        self.frames(self.position(ms), self.position(ms + Millis::new(1)))
    }

    /// pydub's `_parse_position`: the frame a position in ms falls on.
    fn position(&self, ms: Millis) -> i64 {
        let ms = ms.value();
        let ms = if ms < 0 {
            self.len_ms().value() - ms.abs()
        } else {
            ms
        };
        self.frame_count(ms as f64) as i64
    }

    fn frame_count(&self, ms: f64) -> f64 {
        ms * (f64::from(self.rate) / 1000.0)
    }

    /// The frames from `start` to `end` as pydub cuts them from its bytes, padded with silence
    /// where the end lies past the data.
    fn frames(&self, start: i64, end: i64) -> Result<Self, Error> {
        let mut data = self.samples[py_slice(self.samples.len(), start, end)].to_vec();
        let missing = end - start - data.len() as i64;
        if missing != 0 {
            if missing as f64 > self.frame_count(2.0) {
                return Err(Error::TooManyMissingFrames(missing));
            }
            // pydub pads with copies of the first frame silenced, so with nothing if there is none.
            if !data.is_empty() && missing > 0 {
                data.resize(data.len() + missing as usize, 0);
            }
        }
        Ok(self.spawn(data))
    }

    /// `segment.get_frame(index)`, which indexes pydub's bytes as Python slices them.
    fn frame(&self, index: i64) -> &[i16] {
        &self.samples[py_slice(self.samples.len(), index, index + 1)]
    }

    /// `segment.fade_in(duration)`.
    ///
    /// # Panics
    ///
    /// If `duration` is not positive, which pydub refuses.
    pub fn fade_in(&self, duration: Millis) -> Result<Self, Error> {
        assert!(duration > Millis::ZERO, "fade of {} ms", duration.value());
        self.fade(
            Gain::ZERO,
            FADE_FLOOR_DB,
            Millis::ZERO.min(self.len_ms()),
            duration,
        )
    }

    /// `segment.fade_out(duration)`.
    ///
    /// # Panics
    ///
    /// If `duration` is not positive, which pydub refuses.
    pub fn fade_out(&self, duration: Millis) -> Result<Self, Error> {
        assert!(duration > Millis::ZERO, "fade of {} ms", duration.value());
        let end = self.len_ms();
        self.fade(FADE_FLOOR_DB, Gain::ZERO, end - duration, duration)
    }

    /// `segment.fade(to_gain, from_gain, start, end)` once pydub has resolved its arguments to a
    /// start and a duration.
    fn fade(
        &self,
        to_gain: Gain,
        from_gain: Gain,
        start: Millis,
        duration: Millis,
    ) -> Result<Self, Error> {
        let end = start + duration;
        let from_power = Ratio::of(from_gain).value();
        let mut output = self.slice(None, Some(start))?.samples;
        if from_gain != Gain::ZERO {
            output = mul(&output, from_power);
        }
        let gain_delta = Ratio::of(to_gain).value() - from_power;
        // Longer fades step the gain once per ms, shorter ones once per frame.
        if duration > Millis::new(100) {
            let scale_step = gain_delta / duration.value() as f64;
            for i in 0..duration.value() {
                let volume = from_power + scale_step * i as f64;
                output.extend(mul(&self.ms(start + Millis::new(i))?.samples, volume));
            }
        } else {
            let start_frame = self.frame_count(start.value() as f64);
            let end_frame = self.frame_count(end.value() as f64);
            let fade_frames = end_frame - start_frame;
            if fade_frames == 0.0 {
                return Err(Error::EmptyFade);
            }
            let scale_step = gain_delta / fade_frames;
            for i in 0..fade_frames as i64 {
                let volume = from_power + scale_step * i as f64;
                output.extend(mul(self.frame((start_frame + i as f64) as i64), volume));
            }
        }
        let mut after = self.slice(Some(end), None)?.samples;
        if to_gain != Gain::ZERO {
            after = mul(&after, Ratio::of(to_gain).value());
        }
        output.extend(after);
        Ok(self.spawn(output))
    }

    /// `segment.append(other, crossfade)`: `other` after this, the last `crossfade` of this fading
    /// out as the first of `other` fade in.
    ///
    /// # Panics
    ///
    /// If the rates differ, which pydub would resample to the higher.
    pub fn append(&self, other: &Self, crossfade: Millis) -> Result<Self, Error> {
        assert_eq!(self.rate, other.rate, "appending across sample rates");
        if crossfade == Millis::ZERO {
            return Ok(self.spawn([&self.samples[..], &other.samples].concat()));
        }
        for len in [self.len_ms(), other.len_ms()] {
            if crossfade > len {
                return Err(Error::CrossfadeTooLong { crossfade, len });
            }
        }
        let tail = self.slice(Some(-crossfade), None)?;
        let tail = tail.fade(FADE_FLOOR_DB, Gain::ZERO, Millis::ZERO, tail.len_ms())?;
        let head = other.slice(None, Some(crossfade))?;
        let head = head.fade(Gain::ZERO, FADE_FLOOR_DB, Millis::ZERO, head.len_ms())?;
        let mut output = self.slice(None, Some(-crossfade))?.samples;
        // `tail * head` overlays head, looped, on `tail[0:]`, with audioop's clipping add.
        let tail = tail.slice(Some(Millis::ZERO), None)?.samples;
        let looped = head.samples.iter().cycle().chain(std::iter::repeat(&0));
        output.extend(tail.iter().zip(looped).map(|(&a, &b)| a.saturating_add(b)));
        output.extend(other.slice(Some(crossfade), None)?.samples);
        Ok(self.spawn(output))
    }

    /// pydub's `detect_leading_silence` at its defaults: where the first 10 ms chunk no quieter
    /// than -50 dBFS starts, or the length if there is none.
    pub fn leading_silence(&self) -> Result<Millis, Error> {
        const THRESHOLD: Level = Level::new(-50.0);
        const CHUNK: Millis = Millis::new(10);
        let len = self.len_ms();
        let mut trim = Millis::ZERO;
        while trim < len && self.slice(Some(trim), Some(trim + CHUNK))?.dbfs() < THRESHOLD {
            trim += CHUNK;
        }
        Ok(trim.min(len))
    }

    /// The segment with its leading and trailing silence cut, as the old service's
    /// `strip_silence` does it: the leading off, then the leading of the reversed rest.
    pub fn strip_silence(&self) -> Result<Self, Error> {
        let trim_leading = |s: &Self| s.slice(Some(s.leading_silence()?), None);
        let leading = trim_leading(self)?;
        Ok(trim_leading(&leading.reverse())?.reverse())
    }

    fn spawn(&self, samples: Vec<i16>) -> Self {
        Self::new(self.rate, samples)
    }
}

/// pydub's `ratio_to_db`: the gain of the amplitude ratio `ratio`, `-inf` for zero, through the
/// natural logarithm where [`Gain::of`] takes the common one.
pub fn ratio_to_db(ratio: Ratio) -> Gain {
    let ratio = ratio.value();
    if ratio == 0.0 {
        return Gain::new(f64::NEG_INFINITY);
    }
    // Python's math.log(ratio, 10) divides the two natural logarithms.
    Gain::new(20.0 * (ratio.ln() / 10f64.ln()))
}

/// audioop's `mul`: each sample scaled by `factor`, clipped to the 16-bit range and floored.
fn mul(samples: &[i16], factor: f64) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| fbound(f64::from(s) * factor))
        .collect()
}

/// audioop's `fbound` for 16-bit samples. Its C cast of a NaN gives `INT_MIN`, whose low 16 bits
/// are the 0 that Rust's saturating cast gives.
fn fbound(value: f64) -> i16 {
    let value = if value > 32767.0 {
        32767.0
    } else if value < -32767.0 {
        -32768.0
    } else {
        value
    };
    value.floor() as i16
}

/// The range a Python slice `[start:end]` takes from a sequence of `len`.
fn py_slice(len: usize, start: i64, end: i64) -> Range<usize> {
    let len = len as i64;
    let clamp = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
    let (start, end) = (clamp(start), clamp(end));
    start as usize..end.max(start) as usize
}
