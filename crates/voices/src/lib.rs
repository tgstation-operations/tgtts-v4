//! The voices the service speaks in, as it keeps them on disk.
//!
//! A voices directory holds the table of voices the game can choose from and, under each voice's
//! stem (the name its files go by), the voice's latent and blips:
//!
//! - `voices.json`: the table, a JSON object from the name the game shows to the stem.
//! - `latents/<stem>.safetensors`: the voice's reference audio as AudioVAE latents, the F32
//!   tensor `latents` of shape `[frames, 64]`, with what is said in it as the header's `text`.
//! - `blips/<stem>.safetensors`: the voice's blip samples, an I16 tensor of mono PCM per sample,
//!   named `<base>/<number>/<key>`, all at the rate the header gives as `sample_rate`.
//!
//! A latent is read whole when it is asked for. A blips file is opened per voice and its samples
//! are read one at a time, so that no voice's set, about 14 MB, has to be held in memory.
//!
//! [`legacy`] reads the torch.save files the Python service kept its voices in.

mod archive;
pub mod legacy;
pub mod pickle;

use std::{
    collections::{BTreeMap, HashMap},
    error, fmt,
    fs::{self, File},
    io::{self, Read},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use safetensors::{
    Dtype,
    tensor::{Metadata, TensorView},
};

/// The width of a latent frame: AudioVAE's latent dimension.
pub const LATENT_DIM: usize = 64;

const TABLE: &str = "voices.json";
const LATENTS: &str = "latents";
const BLIPS: &str = "blips";
const TEXT: &str = "text";
const SAMPLE_RATE: &str = "sample_rate";

/// A voices directory and its table.
#[derive(Debug)]
pub struct Voices {
    dir: PathBuf,
    table: BTreeMap<String, String>,
}

impl Voices {
    /// Opens the voices directory `dir` by reading its table. The files a voice names are only
    /// read when the voice is asked for.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, Error> {
        let dir = dir.into();
        let path = table_path(&dir);
        let bytes = fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let table: BTreeMap<String, String> =
            serde_json::from_slice(&bytes).map_err(|e| Error::invalid(&path, e.to_string()))?;
        if let Some((voice, stem)) = table.iter().find(|(_, stem)| !is_file_name(stem)) {
            let reason = format!("voice {voice:?} has the stem {stem:?}, which is not a file name");
            return Err(Error::invalid(&path, reason));
        }
        Ok(Self { dir, table })
    }

    /// The names the game shows, sorted, as `/tts-voices` serves them.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.table.keys().map(String::as_str)
    }

    /// Reads `voice`'s latent.
    pub fn latent(&self, voice: &str) -> Result<Latent, Error> {
        let path = latent_path(&self.dir, self.stem(voice)?);
        Latent::read(&path).map_err(|e| e.for_voice(voice, Kind::Latent))
    }

    /// Opens `voice`'s blips.
    pub fn blips(&self, voice: &str) -> Result<Blips, Error> {
        let path = blips_path(&self.dir, self.stem(voice)?);
        let mut blips = Blips::open(&path).map_err(|e| e.for_voice(voice, Kind::Blips))?;
        blips.voice = Some(voice.to_owned());
        Ok(blips)
    }

    fn stem(&self, voice: &str) -> Result<&str, Error> {
        self.table
            .get(voice)
            .map(String::as_str)
            .ok_or_else(|| Error {
                voice: Some(voice.to_owned()),
                path: None,
                kind: ErrorKind::UnknownVoice,
            })
    }
}

/// Where the voices directory `dir` keeps its table.
pub fn table_path(dir: &Path) -> PathBuf {
    dir.join(TABLE)
}

/// Where the voices directory `dir` keeps the latent of the stem `stem`.
pub fn latent_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(LATENTS).join(format!("{stem}.safetensors"))
}

/// Where the voices directory `dir` keeps the blips of the stem `stem`.
pub fn blips_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(BLIPS).join(format!("{stem}.safetensors"))
}

/// Writes `table`, from the name the game shows to the stem, as the voices directory `dir`'s
/// table.
pub fn write_table(dir: &Path, table: &BTreeMap<String, String>) -> Result<(), Error> {
    let path = table_path(dir);
    let mut json = serde_json::to_string_pretty(table).expect("a map of strings serializes");
    json.push('\n');
    fs::write(&path, json).map_err(|e| Error::io(&path, e))
}

fn is_file_name(stem: &str) -> bool {
    !stem.is_empty() && stem != "." && stem != ".." && !stem.contains(['/', '\0'])
}

/// A voice's reference audio, which the model is prompted with to speak in that voice.
#[derive(Debug, Clone)]
pub struct Latent {
    /// The reference audio as AudioVAE latents, a frame per 40 ms.
    pub frames: Vec<[f32; LATENT_DIM]>,
    /// What is said in the reference audio.
    pub text: String,
}

impl Latent {
    /// Reads the latent file at `path`.
    pub fn read(path: &Path) -> Result<Self, Error> {
        let file = Tensors::open(path)?;
        let text = file.metadata(TEXT)?.to_owned();
        let Some((dtype, shape, range)) = file.tensor(LATENTS) else {
            return Err(file.invalid(format!("no tensor {LATENTS:?}")));
        };
        if dtype != Dtype::F32 || shape.len() != 2 || shape[1] != LATENT_DIM {
            let reason = format!("{LATENTS:?} is {dtype:?} {shape:?}, not F32 [frames, 64]");
            return Err(file.invalid(reason));
        }
        let bytes = file.read(range)?;
        let values: Vec<f32> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let frames = values.as_chunks::<LATENT_DIM>().0.to_vec();
        Ok(Self { frames, text })
    }

    /// Writes the latent to `path`, replacing any file there.
    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let bytes: Vec<u8> = self
            .frames
            .as_flattened()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let shape = vec![self.frames.len(), LATENT_DIM];
        let view = TensorView::new(Dtype::F32, shape, &bytes).expect("shape matches the data");
        let metadata = HashMap::from([(TEXT.to_owned(), self.text.clone())]);
        write(path, [(LATENTS.to_owned(), view)], metadata)
    }
}

/// A voice's blips file, open for reading its samples one at a time.
#[derive(Debug)]
pub struct Blips {
    voice: Option<String>,
    file: Tensors,
    sample_rate: u32,
}

impl Blips {
    /// Opens the blips file at `path`, reading its header.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = Tensors::open(path)?;
        let rate = file.metadata(SAMPLE_RATE)?;
        let Ok(sample_rate) = rate.parse() else {
            return Err(file.invalid(format!("{SAMPLE_RATE} {rate:?} is not a rate in Hz")));
        };
        for (name, info) in file.header.tensors() {
            let parts: Vec<&str> = name.split('/').collect();
            if parts.len() != 3
                || parts.contains(&"")
                || info.dtype != Dtype::I16
                || info.shape.len() != 1
            {
                let reason = format!(
                    "tensor {name:?} is {:?} {:?}, not an I16 [samples] named \
                     <base>/<number>/<key>",
                    info.dtype, info.shape
                );
                return Err(file.invalid(reason));
            }
        }
        Ok(Self {
            voice: None,
            file,
            sample_rate,
        })
    }

    /// The rate every sample is at, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reads the sample at `base` (the game's `blip_base`, "male" or "female"), `number` (its
    /// `blip_number`, "1" to "4") and `key`, as mono 16-bit PCM.
    pub fn sample(&self, base: &str, number: &str, key: &str) -> Result<Vec<i16>, Error> {
        // Every name splits into exactly three parts, as open checks, so a part that holds a
        // '/' makes a name of more, which matches nothing rather than another sample.
        let parts = [base, number, key];
        let Some((_, _, range)) = self.file.tensor(&parts.join("/")) else {
            return Err(Error {
                voice: self.voice.clone(),
                path: Some(self.file.path.clone()),
                kind: ErrorKind::NoSample(parts.map(str::to_owned)),
            });
        };
        let bytes = self.file.read(range).map_err(|mut e| {
            e.voice.clone_from(&self.voice);
            e
        })?;
        Ok(bytes
            .as_chunks()
            .0
            .iter()
            .map(|s| i16::from_le_bytes(*s))
            .collect())
    }
}

/// A voice's blips in memory, as a converter has them before writing them out.
#[derive(Debug, Clone, Default)]
pub struct BlipSet {
    /// The rate every sample is at, in Hz.
    pub sample_rate: u32,
    /// Mono 16-bit PCM, by base, number and key.
    pub samples: BTreeMap<(String, String, String), Vec<i16>>,
}

impl BlipSet {
    /// Writes the set to `path`, replacing any file there.
    ///
    /// A base, number or key that is empty or holds a `/` is refused, since the sample could
    /// not be found by it again.
    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let mut tensors = Vec::with_capacity(self.samples.len());
        for ((base, number, key), samples) in &self.samples {
            let parts = [base, number, key];
            if parts
                .iter()
                .any(|part| part.is_empty() || part.contains('/'))
            {
                let reason = format!("blip {parts:?} cannot be named <base>/<number>/<key>");
                return Err(Error::invalid(path, reason));
            }
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            tensors.push((parts.map(String::as_str).join("/"), samples.len(), bytes));
        }
        let views = tensors.iter().map(|(name, len, bytes)| {
            let view = TensorView::new(Dtype::I16, vec![*len], bytes).expect("shape matches");
            (name.clone(), view)
        });
        let metadata = HashMap::from([(SAMPLE_RATE.to_owned(), self.sample_rate.to_string())]);
        write(path, views, metadata)
    }
}

/// A safetensors file of which only the header is held, its tensors read as they are asked for.
#[derive(Debug)]
struct Tensors {
    path: PathBuf,
    file: File,
    header: Metadata,
    data_start: u64,
}

impl Tensors {
    fn open(path: &Path) -> Result<Self, Error> {
        let io = |e| Error::io(path, e);
        let mut file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();
        let mut len = [0; 8];
        file.read_exact(&mut len).map_err(io)?;
        let len = u64::from_le_bytes(len);
        // Checked before allocating, so that a corrupt length is not taken at its word.
        if len > file_len - 8 {
            let reason = format!("header of {len} bytes in a file of {file_len}");
            return Err(Error::invalid(path, reason));
        }
        let mut header = vec![0; len as usize];
        file.read_exact(&mut header).map_err(io)?;
        let header: Metadata = serde_json::from_slice(&header)
            .map_err(|e| Error::invalid(path, format!("header: {e}")))?;
        let data_start = 8 + len;
        if data_start + header.data_len() as u64 != file_len {
            let reason = format!(
                "{file_len} bytes, where the header describes {}",
                data_start + header.data_len() as u64
            );
            return Err(Error::invalid(path, reason));
        }
        Ok(Self {
            path: path.to_owned(),
            file,
            header,
            data_start,
        })
    }

    fn metadata(&self, key: &str) -> Result<&str, Error> {
        let value = self.header.metadata().as_ref().and_then(|m| m.get(key));
        value
            .map(String::as_str)
            .ok_or_else(|| self.invalid(format!("no {key:?} in the header's metadata")))
    }

    fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], (usize, usize))> {
        let info = self.header.info(name)?;
        Some((info.dtype, &info.shape, info.data_offsets))
    }

    fn read(&self, (start, end): (usize, usize)) -> Result<Vec<u8>, Error> {
        let mut bytes = vec![0; end - start];
        self.file
            .read_exact_at(&mut bytes, self.data_start + start as u64)
            .map_err(|e| Error::io(&self.path, e))?;
        Ok(bytes)
    }

    fn invalid(&self, reason: String) -> Error {
        Error::invalid(&self.path, reason)
    }
}

fn write<'a>(
    path: &Path,
    tensors: impl IntoIterator<Item = (String, TensorView<'a>)>,
    metadata: HashMap<String, String>,
) -> Result<(), Error> {
    // serialize_to_file would leave the file readable by its owner alone.
    let bytes = safetensors::serialize(tensors, Some(metadata))
        .map_err(|e| Error::invalid(path, e.to_string()))?;
    fs::write(path, bytes).map_err(|e| Error::io(path, e))
}

/// Which of a voice's files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Its latent.
    Latent,
    /// Its blips.
    Blips,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Kind::Latent => "latent",
            Kind::Blips => "blips",
        })
    }
}

/// A voice or file that could not be read, naming the voice and the file where it concerns one.
#[derive(Debug)]
pub struct Error {
    voice: Option<String>,
    path: Option<PathBuf>,
    kind: ErrorKind,
}

/// What went wrong.
#[derive(Debug)]
pub enum ErrorKind {
    /// The table has no voice of that name.
    UnknownVoice,
    /// The voice's file of this kind does not exist.
    Missing(Kind),
    /// The blips have no sample at this base, number and key.
    NoSample([String; 3]),
    /// The file could not be read or written.
    Io(io::Error),
    /// The file is not in the format, for this reason.
    Invalid(String),
}

impl Error {
    fn io(path: &Path, error: io::Error) -> Self {
        Self {
            voice: None,
            path: Some(path.to_owned()),
            kind: ErrorKind::Io(error),
        }
    }

    fn invalid(path: &Path, reason: String) -> Self {
        Self {
            voice: None,
            path: Some(path.to_owned()),
            kind: ErrorKind::Invalid(reason),
        }
    }

    fn for_voice(mut self, voice: &str, kind: Kind) -> Self {
        self.voice = Some(voice.to_owned());
        if matches!(&self.kind, ErrorKind::Io(e) if e.kind() == io::ErrorKind::NotFound) {
            self.kind = ErrorKind::Missing(kind);
        }
        self
    }

    /// The voice, if the error came from looking one up.
    pub fn voice(&self) -> Option<&str> {
        self.voice.as_deref()
    }

    /// The file, if the error concerns one.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// What went wrong.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(voice) = &self.voice {
            write!(f, "voice {voice:?}: ")?;
        }
        if let Some(path) = &self.path {
            write!(f, "{}: ", path.display())?;
        }
        match &self.kind {
            ErrorKind::UnknownVoice => f.write_str("not in the table"),
            ErrorKind::Missing(kind) => write!(f, "no {kind}"),
            ErrorKind::NoSample(parts) => write!(f, "no blip {}", parts.join("/")),
            ErrorKind::Io(e) => write!(f, "{e}"),
            ErrorKind::Invalid(reason) => f.write_str(reason),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io(e) => Some(e),
            _ => None,
        }
    }
}
