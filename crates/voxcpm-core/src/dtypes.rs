/// A float dtype a part of the model can be held and computed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

/// The dtype each part of the model is held and computed in: a [`Dtype`] by default, and burn's
/// own in the `voxcpm` crate, which runs the model on burn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dtypes<T = Dtype> {
    /// The base and residual LMs, and the projections and heads around them.
    pub lm: T,
    pub local_encoder: T,
    pub local_dit: T,
    /// The audio VAE's decoder, which `voxcpm`'s `TextToSpeech` loads beside the model.
    pub audio_decoder: T,
}

impl<T> Dtypes<T> {
    /// Each part's dtype as `f` gives it.
    pub fn map<U>(self, f: impl Fn(T) -> U) -> Dtypes<U> {
        Dtypes {
            lm: f(self.lm),
            local_encoder: f(self.local_encoder),
            local_dit: f(self.local_dit),
            audio_decoder: f(self.audio_decoder),
        }
    }
}
