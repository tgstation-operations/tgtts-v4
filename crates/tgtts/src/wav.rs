//! WAV as the service reads and writes it: 16-bit PCM and 32-bit float, which is what ffmpeg is
//! given and gives back, what the cache keeps and what the assets are.

use dsp::{
    quantity::{Level, SampleRate},
    radio::Pcm,
    segment::Segment,
};

const PCM: u16 = 1;
const FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xfffe;
/// The size ffmpeg gives the RIFF and data chunks when it writes to a pipe and cannot go back to
/// fill them in.
const UNKNOWN_SIZE: u32 = u32::MAX;

/// Interleaved float samples, full scale at 1, as ffmpeg's filters leave them.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    /// Frames per second.
    pub rate: SampleRate,
    /// Samples per frame.
    pub channels: u16,
    /// The samples, a frame's channels in turn.
    pub samples: Vec<f32>,
}

impl Audio {
    /// `segment` as ffmpeg reads 16-bit PCM: each sample divided by 32768.
    pub fn from_segment(segment: &Segment) -> Self {
        Self {
            rate: segment.rate,
            channels: 1,
            samples: segment.samples.iter().map(|&s| from_i16(s)).collect(),
        }
    }

    /// The samples as ffmpeg writes floats as 16-bit PCM: scaled by 32768, rounded and clipped.
    pub fn to_pcm(&self) -> Pcm {
        Pcm {
            rate: self.rate,
            channels: self.channels,
            samples: self.samples.iter().map(|&s| to_i16(s)).collect(),
        }
    }

    /// The number of frames.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// The length in seconds.
    pub fn seconds(&self) -> f64 {
        self.frames() as f64 / f64::from(self.rate)
    }

    /// The RMS level of all the samples, in dB from full scale: 0 for a full-scale square wave,
    /// -3 for a full-scale sine, and negative infinity for digital silence or no samples at all.
    pub fn rms_dbfs(&self) -> Level {
        let energy: f64 = self.samples.iter().map(|&s| f64::from(s).powi(2)).sum();
        if energy == 0.0 {
            return Level::new(f64::NEG_INFINITY);
        }
        Level::new(10.0 * (energy / self.samples.len() as f64).log10())
    }
}

fn from_i16(sample: i16) -> f32 {
    f32::from(sample) / 32768.0
}

fn to_i16(sample: f32) -> i16 {
    (sample * 32768.0)
        .round_ties_even()
        .clamp(-32768.0, 32767.0) as i16
}

/// A WAV file of 16-bit PCM.
pub fn write_pcm16(rate: SampleRate, channels: u16, samples: &[i16]) -> Vec<u8> {
    let data = samples.iter().flat_map(|s| s.to_le_bytes());
    write(PCM, 16, rate.value(), channels, samples.len() * 2, data)
}

/// A WAV file of `audio` as 32-bit floats.
pub fn write_float(audio: &Audio) -> Vec<u8> {
    let data = audio.samples.iter().flat_map(|s| s.to_le_bytes());
    let len = audio.samples.len() * 4;
    write(FLOAT, 32, audio.rate.value(), audio.channels, len, data)
}

fn write(
    format: u16,
    bits: u16,
    rate: u32,
    channels: u16,
    len: usize,
    data: impl Iterator<Item = u8>,
) -> Vec<u8> {
    let block_align = channels * bits / 8;
    let len = u32::try_from(len).expect("a WAV file holds under 4 GiB of samples");
    let mut bytes = Vec::with_capacity(44 + len as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + len).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(format.to_le_bytes());
    bytes.extend(channels.to_le_bytes());
    bytes.extend(rate.to_le_bytes());
    bytes.extend((rate * u32::from(block_align)).to_le_bytes());
    bytes.extend(block_align.to_le_bytes());
    bytes.extend(bits.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(len.to_le_bytes());
    bytes.extend(data);
    bytes
}

/// Reads a WAV file of 16-bit PCM or 32-bit floats, including one ffmpeg wrote to a pipe, whose
/// sizes it left unknown.
pub fn read(bytes: &[u8]) -> Result<Audio, Error> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::NotWav);
    }
    let mut format = None;
    let mut rest = &bytes[12..];
    while rest.len() >= 8 {
        let id = &rest[..4];
        let size = u32::from_le_bytes(rest[4..8].try_into().expect("four bytes"));
        let body = &rest[8..];
        if id == b"data" {
            let (format, channels, rate, bits) = format.ok_or(Error::NoFormat)?;
            let data = match usize::try_from(size) {
                Ok(size) if size != UNKNOWN_SIZE as usize => body.get(..size).unwrap_or(body),
                _ => body,
            };
            let samples = match (format, bits) {
                (PCM, 16) => data
                    .as_chunks()
                    .0
                    .iter()
                    .map(|b| from_i16(i16::from_le_bytes(*b)))
                    .collect(),
                (FLOAT, 32) => data
                    .as_chunks()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                _ => return Err(Error::Unsupported { format, bits }),
            };
            if channels == 0 || rate == 0 {
                return Err(Error::NoFormat);
            }
            return Ok(Audio {
                rate: SampleRate::new(rate),
                channels,
                samples,
            });
        }
        let size = size as usize;
        if id == b"fmt " {
            let fmt = body.get(..size.max(16)).ok_or(Error::NoFormat)?;
            let u16_at = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
            let mut tag = u16_at(0);
            // WAVE_FORMAT_EXTENSIBLE carries the format in the first two bytes of its GUID.
            if tag == EXTENSIBLE {
                tag = if fmt.len() >= 26 { u16_at(24) } else { 0 };
            }
            let rate = u32::from_le_bytes(fmt[4..8].try_into().expect("four bytes"));
            format = Some((tag, u16_at(2), rate, u16_at(14)));
        }
        // Chunks are padded to an even length.
        let Some(next) = body.get(size + size % 2..) else {
            break;
        };
        rest = next;
    }
    Err(Error::NoData)
}

/// Why a WAV file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// It does not start as a WAV file does.
    #[error("not a WAV file")]
    NotWav,
    /// It has no format chunk before its data, or the format chunk is short or empty.
    #[error("WAV file without a usable format chunk")]
    NoFormat,
    /// Its samples are in this format tag at this many bits, which are neither 16-bit PCM nor
    /// 32-bit floats.
    #[error("WAV samples of format {format} at {bits} bits")]
    Unsupported {
        /// The format tag.
        format: u16,
        /// Bits per sample.
        bits: u16,
    },
    /// It has no data chunk.
    #[error("WAV file without data")]
    NoData,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_round_trips() {
        let samples = [0, 1, -1, i16::MAX, i16::MIN, 12345];
        let audio = read(&write_pcm16(SampleRate::new(22050), 2, &samples)).unwrap();
        assert_eq!((audio.rate, audio.channels), (SampleRate::new(22050), 2));
        assert_eq!(audio.to_pcm().samples, samples);
    }

    #[test]
    fn floats_round_trip() {
        let audio = Audio {
            rate: SampleRate::new(48000),
            channels: 1,
            samples: vec![0.0, 0.5, -1.0, 1.5, f32::MIN_POSITIVE],
        };
        assert_eq!(read(&write_float(&audio)).unwrap(), audio);
    }

    #[test]
    fn floats_clip_to_pcm() {
        let audio = Audio {
            rate: SampleRate::new(8000),
            channels: 1,
            samples: vec![1.0, -1.0, 2.0, -2.0, 0.5 / 32768.0, 1.5 / 32768.0],
        };
        assert_eq!(audio.to_pcm().samples, [32767, -32768, 32767, -32768, 0, 2]);
    }

    #[test]
    fn measures_the_rms_level() {
        let audio = |samples: Vec<f32>| Audio {
            rate: SampleRate::new(48000),
            channels: 2,
            samples,
        };
        let sine: Vec<f32> = (0..48000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        assert!((audio(sine).rms_dbfs().value() + 3.0103).abs() < 1e-4);
        assert_eq!(
            audio(vec![0.5, -0.5, 0.5, -0.5]).rms_dbfs().value(),
            20.0 * 0.5f64.log10()
        );
        let lsb = audio(vec![1.0 / 32768.0, -1.0 / 32768.0])
            .rms_dbfs()
            .value();
        assert!((lsb + 90.309).abs() < 1e-3, "{lsb}");
        assert_eq!(
            audio(vec![0.0; 12000]).rms_dbfs().value(),
            f64::NEG_INFINITY
        );
        assert_eq!(audio(Vec::new()).rms_dbfs().value(), f64::NEG_INFINITY);
    }

    /// The header ffmpeg 8.1 writes for pcm_f32le to a pipe: unknown sizes, an 18-byte format
    /// chunk and a LIST chunk before the data.
    #[test]
    fn reads_what_ffmpeg_writes_to_a_pipe() {
        let mut bytes = b"RIFF\xff\xff\xff\xffWAVEfmt \x12\0\0\0\x03\0\x01\0\x80\xbb\0\0\0\xee\x02\0\x04\0\x20\0\0\0".to_vec();
        bytes.extend(b"LIST\x1a\0\0\0INFOISFT\x0e\0\0\0Lavf62.12.102\0");
        bytes.extend(b"data\xff\xff\xff\xff");
        bytes.extend(0.25f32.to_le_bytes());
        bytes.extend((-0.5f32).to_le_bytes());
        let audio = read(&bytes).unwrap();
        assert_eq!((audio.rate, audio.channels), (SampleRate::new(48000), 1));
        assert_eq!(audio.samples, [0.25, -0.5]);
    }

    #[test]
    fn refuses_other_formats() {
        let mut bytes = write_pcm16(SampleRate::new(8000), 1, &[0; 4]);
        bytes[34] = 8;
        assert_eq!(read(&bytes), Err(Error::Unsupported { format: 1, bits: 8 }));
        assert_eq!(read(b"RIFF\0\0\0\0WAVE"), Err(Error::NoData));
        assert_eq!(read(b"OggS"), Err(Error::NotWav));
    }
}
