//! Signal processing for the service, on PCM in memory: what the Python service did with pydub,
//! librosa and scipy, reproduced closely enough that a voice sounds as it did.

pub mod blips;
pub mod float;
pub mod pitch;
pub mod resample;
pub mod segment;
pub mod sentences;

/// The random draws the old service made with Python's `random` module, which the service makes
/// from a generator of its own and a test replays.
pub trait Random {
    /// `random.random()`: a draw from [0, 1).
    fn random(&mut self) -> f64;
}
