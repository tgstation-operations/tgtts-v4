//! Ogg Vorbis as the old frontend encoded what the game plays, in process: libvorbis set up as
//! ffmpeg's `-b:a 64k` set it up, aiming at an average of 64 kb/s without managing the bitrate.

use std::{
    error, fmt,
    num::{NonZeroU8, NonZeroU32},
};

use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder, VorbisError};

use crate::wav::Audio;

/// The bitrate the old frontend asked for, in bits a second.
const BITRATE: NonZeroU32 = NonZeroU32::new(64000).expect("not zero");
/// The frames given to the encoder at a time, which libvorbis's documentation calls reasonable.
const BLOCK: usize = 1024;

/// `audio` encoded as Ogg Vorbis, its last granule position its length in frames.
pub fn encode(audio: &Audio) -> Result<Vec<u8>, Error> {
    let format = || Error::Format(audio.rate.value(), audio.channels);
    let rate = NonZeroU32::new(audio.rate.value()).ok_or_else(format)?;
    let channels = u8::try_from(audio.channels)
        .ok()
        .and_then(NonZeroU8::new)
        .ok_or_else(format)?;
    // One stream in a file needs no serial of its own.
    let mut builder = VorbisEncoderBuilder::new_with_serial(rate, channels, Vec::new(), 0);
    builder.bitrate_management_strategy(VorbisBitrateManagementStrategy::Vbr {
        target_bitrate: BITRATE,
    });
    let mut encoder = builder.build().map_err(Error::Vorbis)?;
    let count = usize::from(channels.get());
    for block in audio.samples.chunks(BLOCK * count) {
        let planar: Vec<Vec<f32>> = (0..count)
            .map(|channel| block.iter().skip(channel).step_by(count).copied().collect())
            .collect();
        encoder.encode_audio_block(&planar).map_err(Error::Vorbis)?;
    }
    encoder.finish().map_err(Error::Vorbis)
}

/// Why audio could not be encoded.
#[derive(Debug)]
pub enum Error {
    /// It has no frames a second or no channels, or more channels than Vorbis takes.
    Format(u32, u16),
    /// libvorbis or libogg failed.
    Vorbis(VorbisError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(rate, channels) => {
                write!(f, "no Vorbis at {rate} Hz in {channels} channels")
            }
            Self::Vorbis(e) => write!(f, "encoding Vorbis: {e}"),
        }
    }
}

impl error::Error for Error {}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use dsp::quantity::SampleRate;

    use super::*;

    fn sine(rate: u32, channels: u16, frames: usize) -> Audio {
        let samples = (0..frames * usize::from(channels))
            .map(|i| 0.5 * (TAU * 440.0 * (i / usize::from(channels)) as f32 / rate as f32).sin())
            .collect();
        Audio {
            rate: SampleRate::new(rate),
            channels,
            samples,
        }
    }

    /// The granule position of the last Ogg page: the frames a decoder should give.
    fn last_granule(ogg: &[u8]) -> i64 {
        let at = ogg
            .windows(4)
            .rposition(|w| w == b"OggS")
            .expect("an Ogg page");
        i64::from_le_bytes(ogg[at + 6..at + 14].try_into().unwrap())
    }

    /// The Vorbis identification header's channels, rate and nominal bitrate.
    fn vorbis_header(ogg: &[u8]) -> (u8, u32, i32) {
        let at = ogg
            .windows(7)
            .position(|w| w == b"\x01vorbis")
            .expect("an identification header");
        let header = &ogg[at + 7..];
        let rate = u32::from_le_bytes(header[5..9].try_into().unwrap());
        let nominal = i32::from_le_bytes(header[13..17].try_into().unwrap());
        (header[4], rate, nominal)
    }

    #[test]
    fn encodes_vorbis_at_64k_declaring_the_length() {
        for (rate, channels, frames) in [(48000, 1, 24000), (44100, 2, 12345), (48000, 1, 1)] {
            let ogg = encode(&sine(rate, channels, frames)).unwrap();
            assert_eq!(&ogg[..4], b"OggS");
            assert_eq!(vorbis_header(&ogg), (channels as u8, rate, 64000));
            assert_eq!(last_granule(&ogg), frames as i64);
        }
    }

    #[test]
    fn refuses_what_vorbis_cannot_hold() {
        let none = Audio {
            rate: SampleRate::new(48000),
            channels: 0,
            samples: Vec::new(),
        };
        assert!(matches!(encode(&none), Err(Error::Format(48000, 0))));
    }
}
