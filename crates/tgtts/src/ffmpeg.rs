//! ffmpeg, which runs the game's filter graphs and makes the Vorbis the game plays, with the
//! arguments the old frontend gave it.
//!
//! The old frontend filtered and encoded in one run and handed its radio services the Ogg, which
//! they decoded again. Here the filtered audio comes back as float PCM, which the radio routes
//! take as it is, and a second run encodes it: the encoder is given the same samples as before.

use std::{
    error,
    ffi::OsString,
    fmt, io,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
};

use tokio::{io::AsyncWriteExt, process::Command};

use crate::wav::{self, Audio};

/// How much of ffmpeg's error output an error keeps: the end, where it says what failed.
const STDERR_TAIL: usize = 2048;

/// The ffmpeg program to run.
#[derive(Clone, Debug)]
pub struct Ffmpeg {
    program: PathBuf,
}

impl Ffmpeg {
    /// The ffmpeg at `program`, a path or a name to look up on `PATH`.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
        }
    }

    /// The WAV file `input` through the filter graph `graph`, with the files `inputs` as the
    /// graph's further inputs, as float PCM at the rate and in the channels the graph gives.
    pub async fn filter(
        &self,
        input: Vec<u8>,
        graph: &str,
        inputs: &[&Path],
    ) -> Result<Audio, Error> {
        let mut args: Vec<OsString> = ["-f", "wav", "-i", "pipe:0"].map(OsString::from).into();
        for path in inputs {
            args.extend(["-i".into(), path.into()]);
        }
        args.extend(["-filter_complex".into(), graph.into()]);
        args.extend(["-c:a", "pcm_f32le", "-f", "wav", "pipe:1"].map(OsString::from));
        let output = self.run(&args, input).await?;
        wav::read(&output).map_err(Error::Output)
    }

    /// `audio` encoded as the old frontend encoded what the game plays: Vorbis at 64 kb/s in Ogg.
    pub async fn encode(&self, audio: &Audio) -> Result<Vec<u8>, Error> {
        let args = [
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-c:a",
            "libvorbis",
            "-b:a",
            "64k",
            "-f",
            "ogg",
            "pipe:1",
        ]
        .map(OsString::from);
        self.run(&args, wav::write_float(audio)).await
    }

    /// Runs ffmpeg with `args`, `input` on its standard input, and gives its standard output.
    /// ffmpeg is killed if the future is dropped.
    async fn run(&self, args: &[OsString], input: Vec<u8>) -> Result<Vec<u8>, Error> {
        let mut child = Command::new(&self.program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::Spawn(self.program.clone(), e))?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        // ffmpeg can fill its output pipe before it has read all of its input, so the input is
        // written as the output is read. A write that fails because ffmpeg gave up is reported
        // by its exit status.
        let write = async move {
            let _ = stdin.write_all(&input).await;
        };
        let ((), output) = tokio::join!(write, child.wait_with_output());
        let output = output.map_err(Error::Io)?;
        if !output.status.success() {
            let stderr = &output.stderr[output.stderr.len().saturating_sub(STDERR_TAIL)..];
            return Err(Error::Failed {
                status: output.status,
                stderr: String::from_utf8_lossy(stderr).trim().to_owned(),
            });
        }
        Ok(output.stdout)
    }
}

/// The filters the game's own filter graphs use: all a graph from the game may name, since an
/// admin can set any string as an object's voice filter, and other filters can read files or
/// make audio without end.
const ALLOWED: [&str; 25] = [
    "acompressor",
    "acontrast",
    "acrusher",
    "aecho",
    "aemphasis",
    "afftfilt",
    "aformat",
    "alimiter",
    "amix",
    "anoisesrc",
    "aphaser",
    "aresample",
    "asetrate",
    "asplit",
    "atempo",
    "crystalizer",
    "equalizer",
    "highpass",
    "highshelf",
    "lowpass",
    "lowshelf",
    "rubberband",
    "tremolo",
    "vibrato",
    "volume",
];

/// Why `graph` is not run, if it is not: a filter the game's own graphs do not use, or quoting
/// left open, which ffmpeg would read otherwise than the check does.
pub fn refusal(graph: &str) -> Option<String> {
    let Some(names) = filters(graph) else {
        return Some("a quote left open".to_owned());
    };
    names
        .into_iter()
        .find(|name| !ALLOWED.contains(name))
        .map(|name| format!("the filter {name:?}"))
}

/// The names of the filters in `graph`, an ffmpeg filter graph: each filter of each chain, after
/// the link labels before it and up to its arguments, or `None` if a quote is left open. As in
/// ffmpeg's `av_get_token`, a backslash outside quotes escapes the character after it, and inside
/// them is a backslash.
fn filters(graph: &str) -> Option<Vec<&str>> {
    let mut descriptions = Vec::new();
    let (mut start, mut quoted, mut escaped) = (0, false, false);
    for (i, c) in graph.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if !quoted => escaped = true,
            '\'' => quoted = !quoted,
            ',' | ';' if !quoted => {
                descriptions.push(&graph[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if quoted {
        return None;
    }
    descriptions.push(&graph[start..]);
    let names = descriptions.into_iter().filter_map(|description| {
        let mut rest = description.trim_start();
        while let Some(label) = rest.strip_prefix('[') {
            rest = label
                .split_once(']')
                .map_or("", |(_, after)| after)
                .trim_start();
        }
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        (end > 0).then(|| &rest[..end])
    });
    Some(names.collect())
}

/// Why ffmpeg gave nothing to use.
#[derive(Debug)]
pub enum Error {
    /// The program could not be started.
    Spawn(PathBuf, io::Error),
    /// Its pipes failed.
    Io(io::Error),
    /// It exited with this status, having written this at the end of its error output.
    Failed {
        /// How it exited.
        status: ExitStatus,
        /// The end of what it wrote to its standard error.
        stderr: String,
    },
    /// What it wrote is not a WAV file that can be read.
    Output(wav::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(program, e) => write!(f, "running {}: {e}", program.display()),
            Self::Io(e) => write!(f, "talking to ffmpeg: {e}"),
            Self::Failed { status, stderr } => write!(f, "ffmpeg {status}: {stderr}"),
            Self::Output(e) => write!(f, "ffmpeg's output: {e}"),
        }
    }
}

impl error::Error for Error {}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    fn ffmpeg() -> Ffmpeg {
        Ffmpeg::new("ffmpeg")
    }

    fn sine(rate: u32, frames: usize) -> Audio {
        let samples = (0..frames)
            .map(|i| 0.5 * (TAU * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        Audio {
            rate,
            channels: 1,
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

    /// The Vorbis identification header's channels, rate and nominal bitrate: what `-b:a 64k`
    /// asked libvorbis for.
    fn vorbis_header(ogg: &[u8]) -> (u8, u32, i32) {
        let at = ogg
            .windows(7)
            .position(|w| w == b"\x01vorbis")
            .expect("an identification header");
        let header = &ogg[at + 7..];
        let channels = header[4];
        let rate = u32::from_le_bytes(header[5..9].try_into().unwrap());
        let nominal = i32::from_le_bytes(header[13..17].try_into().unwrap());
        (channels, rate, nominal)
    }

    #[tokio::test]
    async fn encodes_vorbis_at_64k_declaring_the_length() {
        for (rate, frames) in [(48000, 24000), (44100, 12345), (48000, 1)] {
            let audio = sine(rate, frames);
            let ogg = ffmpeg().encode(&audio).await.unwrap();
            assert_eq!(&ogg[..4], b"OggS");
            assert_eq!(vorbis_header(&ogg), (1, rate, 64000));
            assert_eq!(last_granule(&ogg), frames as i64);
        }
    }

    #[tokio::test]
    async fn filters_to_float_at_the_graphs_rate() {
        let input = wav::write_pcm16(48000, 1, &sine(48000, 4800).to_pcm().samples);
        let louder = ffmpeg()
            .filter(input.clone(), "volume=2", &[])
            .await
            .unwrap();
        assert_eq!(
            (louder.rate, louder.channels, louder.frames()),
            (48000, 1, 4800)
        );
        let peak = louder.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!((peak - 1.0).abs() < 1e-3, "peak {peak}");

        let resampled = ffmpeg()
            .filter(input, "aresample=44100,pan=stereo|c0=c0|c1=c0", &[])
            .await
            .unwrap();
        assert_eq!((resampled.rate, resampled.channels), (44100, 2));
    }

    #[tokio::test]
    async fn gives_the_graph_its_further_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let impulse = dir.path().join("impulse.wav");
        std::fs::write(&impulse, wav::write_pcm16(48000, 1, &[16384, 0, 0, 0])).unwrap();
        let input = wav::write_pcm16(48000, 1, &sine(48000, 480).to_pcm().samples);
        let graph = "[0] [1] afir=dry=1:wet=1";
        let output = ffmpeg().filter(input, graph, &[&impulse]).await.unwrap();
        assert_eq!((output.rate, output.channels), (48000, 1));
        assert!(output.frames() >= 480);
    }

    #[test]
    fn finds_the_filters_a_graph_names() {
        let names = |graph| filters(graph).unwrap();
        assert_eq!(
            names(
                "[0:a] asplit [out0][out2]; [out0] asetrate=48000*0.9,aresample=48000 [p0]; [p0][0][out2] amix=inputs=3"
            ),
            ["asplit", "asetrate", "aresample", "amix"]
        );
        assert_eq!(
            names(
                "rubberband=pitch='ifnot(0,2-(0+if(1,4))/16,1)':formant=preserved,highpass=f=1000"
            ),
            ["rubberband", "highpass"]
        );
        assert_eq!(
            names(r"volume=1\,amovie=x,volume@loud=2"),
            ["volume", "volume"]
        );
        // Inside quotes a backslash is a backslash, so the quote after it closes them.
        assert_eq!(names(r"volume='\',amovie=x"), ["volume", "amovie"]);
        assert_eq!(filters("volume='2,amovie=x"), None);
    }

    #[test]
    fn refuses_filters_the_game_does_not_use() {
        assert_eq!(refusal("lowpass=f=750,volume=2"), None);
        assert_eq!(
            refusal("[0:a] amovie=/etc/passwd [m]; [0:a][m] amix"),
            Some("the filter \"amovie\"".to_owned())
        );
        assert_eq!(refusal("volume='2"), Some("a quote left open".to_owned()));
    }

    #[tokio::test]
    async fn reports_what_ffmpeg_said() {
        let input = wav::write_pcm16(48000, 1, &[0; 480]);
        let Err(Error::Failed { status, stderr }) =
            ffmpeg().filter(input, "nosuchfilter=1", &[]).await
        else {
            panic!("a filter that does not exist ran");
        };
        assert!(!status.success());
        assert!(stderr.contains("nosuchfilter"), "{stderr}");

        let missing = Ffmpeg::new("/nonexistent/ffmpeg");
        let result = missing.encode(&sine(48000, 10)).await;
        assert!(matches!(result, Err(Error::Spawn(..))));
    }
}
