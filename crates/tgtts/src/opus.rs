//! Ogg Opus as the streamed routes serve it, RFC 7845's encapsulation of what libopus encodes:
//! the audio in 20 ms frames as it comes, each run of frames on pages of its own, so that a
//! player has the audio as soon as it is encoded, and the stream's length exact by the last
//! page's granule position.

use std::{
    error,
    ffi::{CStr, c_char, c_int},
    fmt,
    ptr::NonNull,
};

use crate::ogg::{self, Writer};

/// The rate Opus encodes and decodes at here, the service's.
pub const RATE: u32 = 48000;
/// The samples per channel of a frame, 20 ms: the longest of Opus's usual frames, the fewest
/// bits spent on each frame's own overhead.
const FRAME: usize = 960;
/// The most bytes a packet can take, as libopus's documentation advises allowing.
const MAX_PACKET: usize = 4000;
/// The most samples per channel a packet decodes to, 120 ms.
const MAX_DECODED: usize = 5760;

const OPUS_OK: c_int = 0;
const OPUS_APPLICATION_AUDIO: c_int = 2049;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_GET_LOOKAHEAD_REQUEST: c_int = 4027;

#[repr(C)]
struct OpusEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
struct OpusDecoder {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;
    fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_encoder_destroy(st: *mut OpusEncoder);
    fn opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int) -> *mut OpusDecoder;
    fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_decoder_destroy(st: *mut OpusDecoder);
    fn opus_strerror(error: c_int) -> *const c_char;
    fn opus_get_version_string() -> *const c_char;
}

/// A libopus encoder.
struct Encoder(NonNull<OpusEncoder>);

// SAFETY: an encoder's state is its own memory, which libopus touches only in calls given it.
unsafe impl Send for Encoder {}

impl Encoder {
    fn new(channels: u8, bitrate: u32) -> Result<Self, Error> {
        let mut error = OPUS_OK;
        // SAFETY: the arguments are in range or refused with an error; the encoder is
        // destroyed when dropped.
        let encoder = unsafe {
            opus_encoder_create(
                RATE as i32,
                c_int::from(channels),
                OPUS_APPLICATION_AUDIO,
                &mut error,
            )
        };
        let encoder = Self(NonNull::new(encoder).ok_or(Error::Opus(error))?);
        let bitrate = i32::try_from(bitrate).map_err(|_| Error::Opus(-1))?;
        // SAFETY: the request takes an opus_int32.
        check(unsafe { opus_encoder_ctl(encoder.0.as_ptr(), OPUS_SET_BITRATE_REQUEST, bitrate) })?;
        Ok(encoder)
    }

    /// The samples per channel the encoder holds back, which a decoder's output starts with.
    fn lookahead(&mut self) -> Result<u16, Error> {
        let mut lookahead: i32 = 0;
        // SAFETY: the request takes a pointer to an opus_int32, which outlives the call.
        check(unsafe {
            opus_encoder_ctl(
                self.0.as_ptr(),
                OPUS_GET_LOOKAHEAD_REQUEST,
                &raw mut lookahead,
            )
        })?;
        u16::try_from(lookahead).map_err(|_| Error::Opus(-1))
    }

    /// The packet of a frame of interleaved samples.
    fn encode(&mut self, frame: &[f32], channels: usize) -> Result<Vec<u8>, Error> {
        debug_assert_eq!(frame.len(), FRAME * channels);
        let mut packet = vec![0; MAX_PACKET];
        // SAFETY: `frame` holds a frame's samples for each channel, and `packet` the bytes it
        // is said to.
        let size = unsafe {
            opus_encode_float(
                self.0.as_ptr(),
                frame.as_ptr(),
                FRAME as c_int,
                packet.as_mut_ptr(),
                MAX_PACKET as i32,
            )
        };
        packet.truncate(usize::try_from(size).map_err(|_| Error::Opus(size))?);
        Ok(packet)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: made by opus_encoder_create and destroyed only here.
        unsafe { opus_encoder_destroy(self.0.as_ptr()) }
    }
}

/// An Ogg Opus stream being encoded: audio in, pages out, as soon as each 20 ms frame is whole.
pub struct Stream {
    encoder: Encoder,
    ogg: Writer,
    channels: usize,
    pre_skip: u64,
    /// Samples waiting for a whole frame, interleaved.
    pending: Vec<f32>,
    /// The samples per channel encoded so far, frames of padding at the end included.
    encoded: u64,
    /// The samples per channel of audio taken.
    taken: u64,
}

impl Stream {
    /// A stream of `channels` channels, one or two, at [`RATE`], aiming at `bitrate` bits a
    /// second, its pages numbered as bitstream `serial`; and the pages of its headers, which
    /// come before any audio.
    pub fn new(channels: u8, bitrate: u32, serial: u32) -> Result<(Self, Vec<u8>), Error> {
        if !(1..=2).contains(&channels) {
            return Err(Error::Channels(channels));
        }
        let mut encoder = Encoder::new(channels, bitrate)?;
        let pre_skip = encoder.lookahead()?;
        let mut head = b"OpusHead".to_vec();
        head.push(1);
        head.push(channels);
        head.extend_from_slice(&pre_skip.to_le_bytes());
        head.extend_from_slice(&RATE.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0);
        // SAFETY: libopus's version is a static C string.
        let vendor = unsafe { CStr::from_ptr(opus_get_version_string()) }.to_bytes();
        let mut tags = b"OpusTags".to_vec();
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes());
        let mut ogg = Writer::new(serial);
        let mut pages = Vec::new();
        // Each header on a page of its own, as RFC 7845 has it.
        ogg.write(&[(&head, 0)], true, false, &mut pages);
        ogg.write(&[(&tags, 0)], false, false, &mut pages);
        let stream = Self {
            encoder,
            ogg,
            channels: usize::from(channels),
            pre_skip: u64::from(pre_skip),
            pending: Vec::new(),
            encoded: 0,
            taken: 0,
        };
        Ok((stream, pages))
    }

    /// Takes `samples`, interleaved, and gives the pages of the frames they complete, none if
    /// they complete none.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<u8>, Error> {
        self.taken += (samples.len() / self.channels) as u64;
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / (FRAME * self.channels) * FRAME * self.channels;
        let frames: Vec<f32> = self.pending.drain(..whole).collect();
        self.pages(&frames, false)
    }

    /// Ends the stream, giving the pages of the audio it still holds. The last frame is padded
    /// with silence, and so are as many frames after as it takes for a decoder's output, which
    /// lags by the encoder's lookahead, to reach the audio's end; the last page's granule
    /// position, the pre-skip and the audio's length, trims the padding off.
    pub fn finish(mut self) -> Result<Vec<u8>, Error> {
        let frames = (self.taken + self.pre_skip).div_ceil(FRAME as u64);
        let padded = (frames - self.encoded / FRAME as u64) as usize * FRAME * self.channels;
        let mut rest = std::mem::take(&mut self.pending);
        rest.resize(padded, 0.0);
        self.pages(&rest, true)
    }

    fn pages(&mut self, samples: &[f32], last: bool) -> Result<Vec<u8>, Error> {
        let mut packets = Vec::new();
        for frame in samples.chunks_exact(FRAME * self.channels) {
            self.encoded += FRAME as u64;
            packets.push((self.encoder.encode(frame, self.channels)?, self.encoded));
        }
        if last && let Some((_, granule)) = packets.last_mut() {
            *granule = self.pre_skip + self.taken;
        }
        let packets: Vec<(&[u8], u64)> = packets.iter().map(|(p, g)| (p.as_slice(), *g)).collect();
        let mut pages = Vec::new();
        self.ogg.write(&packets, false, last, &mut pages);
        Ok(pages)
    }
}

/// `samples`, interleaved in `channels` channels at [`RATE`], encoded whole as Ogg Opus at
/// `bitrate`, as bitstream `serial`.
pub fn encode(samples: &[f32], channels: u8, bitrate: u32, serial: u32) -> Result<Vec<u8>, Error> {
    let (mut stream, mut ogg) = Stream::new(channels, bitrate, serial)?;
    ogg.extend(stream.push(samples)?);
    ogg.extend(stream.finish()?);
    Ok(ogg)
}

/// What [`decode`] made of a stream.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub channels: u8,
    /// The samples at [`RATE`], interleaved.
    pub samples: Vec<f32>,
}

/// The audio of the Ogg Opus stream `ogg`, of mapping family 0, decoded by libopus and trimmed as
/// its header and its last page say: the pre-skip off the start, and the end past the last
/// page's granule position, or nothing off the end where it has no last page yet.
pub fn decode(ogg: &[u8]) -> Result<Decoded, Error> {
    let pages = ogg::pages(ogg).map_err(Error::Ogg)?;
    let packets = ogg::packets(&pages);
    let head = packets
        .first()
        .filter(|p| p.len() >= 19 && p.starts_with(b"OpusHead"));
    let head = head.ok_or(Error::Malformed("no OpusHead"))?;
    let channels = head[9];
    if !(1..=2).contains(&channels) || head[18] != 0 {
        return Err(Error::Channels(channels));
    }
    let pre_skip = usize::from(u16::from_le_bytes([head[10], head[11]]));
    let mut error = OPUS_OK;
    // SAFETY: the arguments are in range or refused with an error; the decoder is destroyed
    // below.
    let decoder = unsafe { opus_decoder_create(RATE as i32, c_int::from(channels), &mut error) };
    let decoder = NonNull::new(decoder).ok_or(Error::Opus(error))?;
    let count = usize::from(channels);
    let mut samples = Vec::new();
    let mut buffer = vec![0f32; MAX_DECODED * count];
    let mut result = Ok(());
    for packet in packets.iter().skip(2) {
        // SAFETY: `packet` and `buffer` hold the lengths they are said to.
        let decoded = unsafe {
            opus_decode_float(
                decoder.as_ptr(),
                packet.as_ptr(),
                packet.len() as i32,
                buffer.as_mut_ptr(),
                MAX_DECODED as c_int,
                0,
            )
        };
        match usize::try_from(decoded) {
            Ok(n) => samples.extend_from_slice(&buffer[..n * count]),
            Err(_) => {
                result = Err(Error::Opus(decoded));
                break;
            }
        }
    }
    // SAFETY: made by opus_decoder_create and destroyed only here.
    unsafe { opus_decoder_destroy(decoder.as_ptr()) };
    result?;
    if let Some(end) = pages.last().filter(|p| p.last).and_then(|p| p.granule) {
        samples.truncate(usize::try_from(end).unwrap_or(usize::MAX) * count);
    }
    samples.drain(..(pre_skip * count).min(samples.len()));
    Ok(Decoded { channels, samples })
}

/// Why audio could not be encoded or decoded.
#[derive(Debug)]
pub enum Error {
    /// libopus failed with this code.
    Opus(i32),
    /// Opus here takes one or two channels.
    Channels(u8),
    Ogg(ogg::Error),
    Malformed(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Opus(code) => {
                // SAFETY: libopus's errors are static C strings, for any code.
                let reason = unsafe { CStr::from_ptr(opus_strerror(*code)) };
                write!(f, "Opus: {}", reason.to_string_lossy())
            }
            Self::Channels(n) => write!(f, "no Opus stream here in {n} channels"),
            Self::Ogg(e) => e.fmt(f),
            Self::Malformed(what) => write!(f, "Ogg Opus: {what}"),
        }
    }
}

impl error::Error for Error {}

fn check(code: c_int) -> Result<(), Error> {
    if code == OPUS_OK {
        Ok(())
    } else {
        Err(Error::Opus(code))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        f32::consts::TAU,
        io::Write,
        process::{Command, Stdio},
    };

    use super::*;

    fn sine(frames: usize, channels: usize) -> Vec<f32> {
        (0..frames * channels)
            .map(|i| 0.5 * (TAU * 440.0 * (i / channels) as f32 / RATE as f32).sin())
            .collect()
    }

    /// The RMS of `x` against `y`, over that of `y`, in dB.
    fn error_db(x: &[f32], y: &[f32]) -> f64 {
        let energy =
            |v: &mut dyn Iterator<Item = f32>| v.map(|s| f64::from(s).powi(2)).sum::<f64>();
        let noise = energy(&mut x.iter().zip(y).map(|(a, b)| a - b));
        10.0 * (noise / energy(&mut y.iter().copied())).log10()
    }

    #[test]
    fn gives_back_as_many_samples_as_it_took_however_they_came() {
        for channels in [1, 2] {
            for frames in [0, 1, 959, 960, 961, 48000, 48000 + 312] {
                let audio = sine(frames, channels);
                let ogg = encode(&audio, channels as u8, 64000, 9).unwrap();
                let decoded = decode(&ogg).unwrap();
                assert_eq!(decoded.channels, channels as u8);
                assert_eq!(decoded.samples.len(), audio.len(), "{frames} in {channels}");
            }
        }
    }

    #[test]
    fn decodes_what_it_encodes_in_time_and_near_the_waveform() {
        let audio = sine(RATE as usize, 1);
        let decoded = decode(&encode(&audio, 1, 64000, 1).unwrap()).unwrap();
        // A tone at 64 kb/s comes back within a few percent, and in step: a sample's lag
        // would put it far off.
        let error = error_db(&decoded.samples[4800..43200], &audio[4800..43200]);
        assert!(error < -20.0, "{error} dB");
    }

    #[test]
    fn puts_each_frame_out_as_soon_as_it_is_whole() {
        let (mut stream, headers) = Stream::new(1, 64000, 3).unwrap();
        let pages = ogg::pages(&headers).unwrap();
        assert_eq!(pages.len(), 2);
        assert!(pages[0].first && pages[0].lacing.len() == 1 && pages[1].lacing.len() == 1);
        assert!(stream.push(&[0.1; 959]).unwrap().is_empty());
        let page = ogg::pages(&stream.push(&[0.1; 1]).unwrap()).unwrap();
        assert_eq!((page.len(), page[0].granule), (1, Some(960)));
        let pages = ogg::pages(&stream.push(&[0.1; 960 * 3]).unwrap()).unwrap();
        assert_eq!(ogg::packets(&pages).len(), 3);
        assert_eq!(pages.last().unwrap().granule, Some(960 * 4));
        let end = ogg::pages(&stream.finish().unwrap()).unwrap();
        let last = end.last().unwrap();
        assert!(last.last);
        // The encoder's lookahead and four frames of audio.
        assert_eq!(last.granule, Some(312 + 960 * 4));
    }

    /// ffmpeg's own Opus decoder, not libopus, reads the stream as `decode` does: the same
    /// length, trimmed at both ends, and the same waveform.
    #[test]
    fn ffmpeg_reads_it_as_it_is_meant() {
        let audio = sine(RATE as usize + 123, 1);
        let ogg = encode(&audio, 1, 64000, 42).unwrap();
        let mut ffmpeg = Command::new("ffmpeg")
            .args([
                "-v", "error", "-f", "ogg", "-i", "pipe:0", "-f", "f32le", "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        ffmpeg.stdin.take().unwrap().write_all(&ogg).unwrap();
        let output = ffmpeg.wait_with_output().unwrap();
        assert!(output.status.success());
        let theirs: Vec<f32> = (output.stdout.as_chunks().0.iter())
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let ours = decode(&ogg).unwrap().samples;
        assert_eq!(theirs.len(), audio.len());
        let error = error_db(&theirs, &ours);
        assert!(error < -60.0, "{error} dB");
    }
}
