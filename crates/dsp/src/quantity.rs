//! The quantities the audio path carries, each a newtype with only the relations that mean
//! something: levels in decibels and the gains between them, ratios of amplitudes, times in
//! milliseconds, sample rates, counts of samples and of the model's patches, and pitch in quarter
//! tones.
//!
//! A relation between two kinds is an associated function on the type of its result, named for
//! what it is made from, and its doc says why it holds. Each takes the steps of the arithmetic it
//! names, in that order and with that rounding, which the service's replies depend on to the bit.
//! Within one kind, addition and subtraction are operators where a sum means something: two gains
//! one after the other, or two durations end to end. A level has none, since two loudnesses do
//! not add and a gain does not add to one: the gain between two is [`Gain::between`]. Scaling by
//! a dimensionless factor is named.
//!
//! A quantity is made from its number with `new` where the number comes in, read from a file or
//! a request or measured from samples, and gives it back with `value` where arithmetic on the
//! samples takes it, which stays on bare numbers.
//!
//! What the types refuse, as a compile error: a gain added to a loudness,
//!
//! ```compile_fail,E0369
//! use dsp::quantity::{Gain, Loudness};
//! let _ = Loudness::new(-23.0) + Gain::new(6.0);
//! ```
//!
//! and silence with its duration and its rate the wrong way round:
//!
//! ```compile_fail,E0308
//! use dsp::{RATE, segment::Segment, speech::TAIL};
//! let _ = Segment::silent(RATE, TAIL);
//! ```

use std::ops::{Add, AddAssign, Neg, Sub};

/// A quantity over one number, with a name and its number, and nothing else until a relation
/// gives it more.
macro_rules! quantity {
    ($(#[$attr:meta])* $name:ident($number:ty)) => {
        $(#[$attr])*
        pub struct $name($number);

        impl $name {
            /// The quantity of `value`, in its unit.
            pub const fn new(value: $number) -> Self {
                Self(value)
            }

            /// Its number, in its unit.
            pub const fn value(self) -> $number {
                self.0
            }
        }
    };
}

/// The operators of a kind whose values add: its zero, and addition, subtraction and negation.
macro_rules! linear {
    ($name:ident, $zero:literal) => {
        impl $name {
            /// The zero of this kind.
            pub const ZERO: Self = Self($zero);
        }

        impl Add for $name {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl Sub for $name {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                *self = *self + other;
            }
        }
    };
}

quantity!(
    /// An integrated loudness, in LUFS, as BS.1770's meter reads it: a level, which a [`Gain`]
    /// moves.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    Loudness(f64)
);

quantity!(
    /// A true peak, in dBTP, full scale at 0: the largest magnitude of the waveform between the
    /// samples as well as on them, as BS.1770's meter reads it. A kind of its own and not a
    /// [`Level`], though both count decibels from full scale: the waveform between two samples
    /// can stand decibels over both, so a ceiling on the one does not hold the other.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    TruePeak(f64)
);

quantity!(
    /// A level of the samples themselves, in dBFS, full scale at 0: of their root mean square or
    /// of the largest of them, as pydub measures both.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    Level(f64)
);

quantity!(
    /// A change of level, in dB: up where positive, down where negative.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    Gain(f64)
);

quantity!(
    /// A ratio of amplitudes: what a gain multiplies samples by, or an amplitude against full
    /// scale.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    Ratio(f64)
);

quantity!(
    /// A time in whole milliseconds, as pydub positions everything: a duration, or a position in
    /// a segment, from its start, or from its end where negative, as pydub slices.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    Millis(i64)
);

quantity!(
    /// A rate of samples, in Hz: the frames a second of audio holds.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    SampleRate(u32)
);

quantity!(
    /// A count of samples of one channel: a length of audio at its rate.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    Samples(usize)
);

quantity!(
    /// A count of the model's patches, the steps it makes speech in, each decoded to the same
    /// count of samples.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    Patches(usize)
);

quantity!(
    /// A shift of pitch in quarter tones, up where positive, as the game asks for one.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    QuarterTones(i32)
);

linear!(Gain, 0.0);
linear!(Millis, 0);

impl Gain {
    /// The gain that brings audio at a loudness of `from` to `to`: `to - from`. A gain raises
    /// the mean square of every block the meter averages by itself, and the relative gate with
    /// them, so it raises the integrated loudness by itself too, as long as the absolute gate
    /// passes the same blocks.
    pub fn between(from: Loudness, to: Loudness) -> Self {
        Self(to.0 - from.0)
    }

    /// The gain of an amplitude ratio: twenty times its common logarithm, a decibel being a
    /// tenth of a bel of power, and power going as the square of amplitude.
    pub fn of(ratio: Ratio) -> Self {
        Self(20.0 * ratio.0.log10())
    }

    /// The gain divided by a dimensionless factor.
    pub fn divided_by(self, factor: f64) -> Self {
        Self(self.0 / factor)
    }

    /// The lesser of two, as [`f64::min`] has it.
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// The size of the change, up or down.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Ratio {
    /// The factor a gain multiplies amplitudes by: ten to the gain over twenty, the inverse of
    /// [`Gain::of`].
    pub fn of(gain: Gain) -> Self {
        Self(10f64.powf(gain.0 / 20.0))
    }

    /// The amplitude a level stands at, against full scale: as [`Ratio::of`] a gain's, since a
    /// level in dBFS is the gain to it from full scale.
    pub fn of_level(level: Level) -> Self {
        Self(10f64.powf(level.0 / 20.0))
    }

    /// The amplitude a true peak stands at, against full scale: as [`Ratio::of`] a gain's,
    /// since a true peak in dBTP is the gain to it from full scale.
    pub fn of_true_peak(peak: TruePeak) -> Self {
        Self(10f64.powf(peak.0 / 20.0))
    }
}

impl Level {
    /// The level a gain from full scale comes to: the gain's number, since a level in dBFS is
    /// the gain to it from full scale.
    pub fn from_full_scale(gain: Gain) -> Self {
        Self(gain.0)
    }
}

impl Millis {
    /// The duration of `count` of these one after another.
    pub fn times(self, count: usize) -> Self {
        Self(self.0 * count as i64)
    }
}

impl QuarterTones {
    /// No shift.
    pub const ZERO: Self = Self(0);
}

impl From<SampleRate> for f64 {
    /// The rate as the float librosa and libsoxr take, exactly.
    fn from(rate: SampleRate) -> Self {
        Self::from(rate.0)
    }
}

impl Samples {
    /// The samples `duration` spans at `rate`, cut to a whole number: `duration * rate / 1000`
    /// in whole numbers, the product first, a rate being samples a second and a second a
    /// thousand milliseconds.
    ///
    /// # Panics
    ///
    /// If `duration` is negative.
    pub fn over(duration: Millis, rate: SampleRate) -> Self {
        let ms = usize::try_from(duration.0).expect("a duration is not negative");
        Self(ms * rate.0 as usize / 1000)
    }
}

impl Patches {
    /// The patches it takes to make `samples`, a patch decoding to `per_patch` of them: the last
    /// counted whole, since the model makes whole patches.
    pub fn covering(samples: Samples, per_patch: Samples) -> Self {
        Self(samples.0.div_ceil(per_patch.0))
    }
}
