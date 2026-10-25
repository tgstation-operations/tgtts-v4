//! Signal processing for the service, on PCM in memory: what the Python service did with pydub,
//! librosa and scipy, reproduced closely enough that a voice sounds as it did.

pub mod blips;
pub mod float;
pub mod gibberish;
pub mod limiter;
pub mod loudness;
pub mod pitch;
pub mod radio;
pub mod resample;
pub mod segment;
pub mod sentences;
pub mod speech;

/// The rate the service works at, in Hz: the model's, every voice's, and the one the game's
/// filters are written for.
pub const RATE: u32 = 48000;

/// The random draws the old service made with Python's `random` module, which the service makes
/// from a generator of its own and a test replays.
pub trait Random {
    /// `random.random()`: a draw from [0, 1).
    fn random(&mut self) -> f64;

    /// `random._randbelow(n)`: a draw from `0..n`, which `random.sample` makes.
    fn below(&mut self, n: usize) -> usize;
}
