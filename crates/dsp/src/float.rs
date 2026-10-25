//! The old service's conversions between 16-bit PCM and the floats librosa and scipy work on.

/// `audiosegment_to_numpy`: 16-bit samples as float32 in [-1, 1), divided by 32768.
pub fn from_pcm(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| f32::from(s) / 32768.0).collect()
}

/// `numpy_to_audiosegment`: floats scaled by 32767, not 32768, in their own precision, and cast
/// to 16 bits as numpy casts on x86-64. The cast truncates to 32 bits first, so a sample past
/// full scale wraps around rather than clipping, and one past the 32-bit range becomes 0.
pub fn to_pcm<F: Float>(samples: &[F]) -> Vec<i16> {
    samples.iter().map(|&s| s.to_pcm()).collect()
}

/// A float the old service turned into 16-bit samples.
pub trait Float: Copy + sealed::Sealed {
    /// The sample `numpy_to_audiosegment` makes of this one.
    fn to_pcm(self) -> i16;
}

impl Float for f32 {
    fn to_pcm(self) -> i16 {
        cast(f64::from(self * 32767.0))
    }
}

impl Float for f64 {
    fn to_pcm(self) -> i16 {
        cast(self * 32767.0)
    }
}

/// x86-64's `cvttsd2si` to 32 bits, which gives `i32::MIN` for what it cannot represent, then
/// the low 16 bits.
fn cast(value: f64) -> i16 {
    let value = value.trunc();
    if (-2147483648.0..2147483648.0).contains(&value) {
        value as i32 as i16
    } else {
        0
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
