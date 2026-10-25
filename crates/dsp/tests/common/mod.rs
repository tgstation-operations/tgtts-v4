//! Reading the dsp fixtures, which hold PCM as F32 and doubles as the I64 of their bits, since
//! the generators' safetensors writer has neither I16 nor F64. The tensors are read from the
//! file's bytes: testkit's readers go through burn, whose integers are 32-bit.

#![allow(dead_code)]

use safetensors::{Dtype, SafeTensors};

/// A fixture's tensors, read whole.
pub struct Tensors {
    name: String,
    bytes: Vec<u8>,
}

impl Tensors {
    /// The fixture `testkit::fixture!` opened.
    pub fn new(fixture: &testkit::Fixture) -> Self {
        let path = fixture.path();
        Self {
            name: path.display().to_string(),
            bytes: std::fs::read(path).expect("testkit read it"),
        }
    }

    /// Whether there is a tensor `name`.
    pub fn contains(&self, name: &str) -> bool {
        let tensors = SafeTensors::deserialize(&self.bytes).expect("testkit checked it");
        tensors.tensor(name).is_ok()
    }

    #[track_caller]
    fn raw(&self, name: &str, dtype: Dtype) -> (Vec<usize>, Vec<u8>) {
        let tensors = SafeTensors::deserialize(&self.bytes).expect("testkit checked it");
        let view = tensors
            .tensor(name)
            .unwrap_or_else(|e| panic!("{}: {name}: {e}", self.name));
        assert_eq!(view.dtype(), dtype, "{}: {name}", self.name);
        (view.shape().to_vec(), view.data().to_vec())
    }

    /// The integer tensor `name`, flattened.
    #[track_caller]
    pub fn ints(&self, name: &str) -> Vec<i64> {
        let (_, bytes) = self.raw(name, Dtype::I64);
        bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect()
    }

    /// The first element of the integer tensor `name`.
    #[track_caller]
    pub fn int(&self, name: &str) -> i64 {
        self.ints(name)[0]
    }

    /// The rows of the 2-D integer tensor `name`.
    #[track_caller]
    pub fn int_rows(&self, name: &str) -> Vec<Vec<i64>> {
        let (shape, _) = self.raw(name, Dtype::I64);
        let values = self.ints(name);
        values
            .chunks(shape[1].max(1))
            .map(<[i64]>::to_vec)
            .collect()
    }

    /// The doubles stored as the bits of the integer tensor `name`.
    #[track_caller]
    pub fn f64s(&self, name: &str) -> Vec<f64> {
        self.ints(name)
            .into_iter()
            .map(|bits| f64::from_bits(bits as u64))
            .collect()
    }

    /// The float tensor `name`, flattened.
    #[track_caller]
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let (_, bytes) = self.raw(name, Dtype::F32);
        bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// The 16-bit PCM stored as the F32 tensor `name`.
    #[track_caller]
    pub fn pcm(&self, name: &str) -> Vec<i16> {
        self.f32s(name)
            .into_iter()
            .map(|s| {
                assert!(
                    s == s.trunc() && (-32768.0..=32767.0).contains(&s),
                    "{name}: {s}"
                );
                s as i16
            })
            .collect()
    }
}

/// Where `actual` first differs from `expected`, for a failure message that shows it.
pub fn first_difference<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T]) -> String {
    match actual.iter().zip(expected).position(|(a, e)| a != e) {
        Some(i) => format!(
            "lengths {} and {}, first difference at {i}: {:?} != {:?}",
            actual.len(),
            expected.len(),
            actual[i],
            expected[i]
        ),
        None => format!("lengths {} and {}", actual.len(), expected.len()),
    }
}

/// Asserts that 16-bit `actual` is `expected` but for samples at most `max` apart, no more than
/// `fraction` of them.
#[track_caller]
pub fn assert_pcm_close(name: &str, actual: &[i16], expected: &[i16], max: i32, fraction: f64) {
    assert_eq!(actual.len(), expected.len(), "{name}: lengths");
    let differences: Vec<i32> = (actual.iter().zip(expected))
        .map(|(a, e)| (i32::from(*a) - i32::from(*e)).abs())
        .collect();
    let worst = differences.iter().copied().max().unwrap_or(0);
    assert!(worst <= max, "{name}: samples {worst} apart");
    let differing = differences.iter().filter(|&&d| d != 0).count();
    let share = differing as f64 / expected.len().max(1) as f64;
    assert!(
        share <= fraction,
        "{name}: {differing} of {} samples differ",
        expected.len()
    );
}

/// The rate, channels and samples of a 16-bit PCM WAV file.
pub fn read_wav(path: &std::path::Path) -> (u32, u16, Vec<i16>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{}: not a WAV",
        path.display()
    );
    let (mut format, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match &bytes[at..at + 4] {
            b"fmt " => format = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + len % 2;
    }
    let (format, data) = (format.unwrap(), data.unwrap());
    let word = |i: usize| u16::from_le_bytes([format[i], format[i + 1]]);
    assert_eq!(
        (word(0), word(14)),
        (1, 16),
        "{}: not 16-bit PCM",
        path.display()
    );
    let rate = u32::from_le_bytes(format[4..8].try_into().unwrap());
    let samples = data
        .as_chunks()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b))
        .collect();
    (rate, word(2), samples)
}
