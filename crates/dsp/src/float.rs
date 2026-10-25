//! The old service's conversions between 16-bit PCM and the floats librosa and scipy work on.

/// `audiosegment_to_numpy`: 16-bit samples as float32 in [-1, 1), divided by 32768.
pub fn from_pcm(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| f32::from(s) / 32768.0).collect()
}

/// `numpy_to_audiosegment`: floats scaled by 32767, not 32768, in their own precision, and cast
/// to 16 bits toward zero, as numpy casts, but held at full scale where numpy's cast wraps
/// around, and a NaN made 0.
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
        (self * 32767.0) as i16
    }
}

impl Float for f64 {
    fn to_pcm(self) -> i16 {
        (self * 32767.0) as i16
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
