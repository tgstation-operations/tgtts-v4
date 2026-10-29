//! The service's configuration: a TOML file of the keys below, any of which the environment
//! variable `TGTTS_<KEY>`, the key in capitals, overrides.
//!
//! - `listen`: the address to serve on, `127.0.0.1:5002` by default.
//! - `token_file`: a file holding the token the game sends in its `Authorization` header.
//! - `voices_dir`: the voices directory.
//! - `assets_dir`: the radio's clicks and static and the silicon effect's impulse responses.
//! - `checkpoint_dir`: the VoxCPM2 checkpoint.
//! - `cache_dir`: where the speech cache is kept.
//! - `glsl_kernels_dir`, `mlir_kernels_dir`: the kernels the model runs on the GPU, as `nix build
//!   .#glsl-kernels` and `nix build .#gpu-kernels` compile them.
//! - `metrics_db`: the SQLite database the metrics go to, none by default.
//! - `ffmpeg`: the ffmpeg to run, `ffmpeg` on the `PATH` by default.
//! - `synthesis_limit`: syntheses run at once, 16 by default, as the old model service ran.
//! - `prompt_cache_bytes`: the bytes on the GPU the prompt cache keeps of the voices, 512 MiB by
//!   default.
//! - `cpu_limit`: other work on the CPU run at once, one per CPU by default.
//! - `queue_timeout_seconds`: how long a request waits for the model, 5 by default.
//! - `radio_wait_seconds`: how long a radio route waits for the route that makes what it puts on
//!   the radio to begin, 10 by default.
//! - `radio_wait_ceiling_seconds`: how long a radio route waits in all once that route has begun,
//!   30 by default, as long as the game waits for any route by default.
//! - `loudness_target_lufs`: the integrated loudness speech and blips are brought to before the
//!   game's filters, -16 by default.
//! - `loudness_ceiling_dbtp`: the true peak they are held under, at most 0, EBU R128's -1 by
//!   default.
//! - `voice_levels_file`: how loud each voice comes out of the model, as `tgtts-gains` writes
//!   it, by which `POST /stream` streams speech as it is made; without it, that speech is made
//!   whole first.
//! - `babble_cap`: whether a request's speech is cut at 3 s and 0.5 s for each character of its
//!   text, where the model talks on past its text, rather than at its 256th patch, 41 s in;
//!   `false` by default.
//! - `log_level`: `DEBUG`, `INFO`, `WARNING` or `ERROR`, `INFO` by default, of the service's own
//!   events; the crates it uses log at it but never finer than `WARN`.

use std::{
    collections::HashMap,
    fmt, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    thread,
    time::Duration,
};

use dsp::quantity::{self, TruePeak};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::Targets;

use crate::{
    load,
    service::{Loudness, parse_level},
};

const KEYS: [&str; 21] = [
    "listen",
    "token_file",
    "voices_dir",
    "assets_dir",
    "checkpoint_dir",
    "cache_dir",
    "glsl_kernels_dir",
    "mlir_kernels_dir",
    "metrics_db",
    "ffmpeg",
    "synthesis_limit",
    "prompt_cache_bytes",
    "cpu_limit",
    "queue_timeout_seconds",
    "radio_wait_seconds",
    "radio_wait_ceiling_seconds",
    "loudness_target_lufs",
    "loudness_ceiling_dbtp",
    "voice_levels_file",
    "babble_cap",
    "log_level",
];

/// The service's configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub listen: SocketAddr,
    pub token_file: PathBuf,
    pub voices_dir: PathBuf,
    pub assets_dir: PathBuf,
    pub checkpoint_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub glsl_kernels_dir: PathBuf,
    pub mlir_kernels_dir: PathBuf,
    pub metrics_db: Option<PathBuf>,
    pub ffmpeg: PathBuf,
    pub synthesis_limit: usize,
    pub prompt_cache_bytes: usize,
    pub cpu_limit: usize,
    pub queue_timeout: Duration,
    pub radio_wait: Duration,
    pub radio_wait_ceiling: Duration,
    pub loudness: Loudness,
    pub voice_levels_file: Option<PathBuf>,
    pub babble_cap: bool,
    pub log_level: LevelFilter,
}

impl Config {
    /// Reads the configuration from the TOML file at `file`, if there is one, and the
    /// environment that `var` reads.
    pub fn load(file: Option<&Path>, var: impl Fn(&str) -> Option<String>) -> Result<Self, Error> {
        let mut values = HashMap::new();
        if let Some(file) = file {
            let text =
                fs::read_to_string(file).map_err(|e| Error(format!("{}: {e}", file.display())))?;
            let table: toml::Table = text
                .parse()
                .map_err(|e| Error(format!("{}: {e}", file.display())))?;
            for (key, value) in table {
                if !KEYS.contains(&key.as_str()) {
                    return Err(Error(format!("{}: no setting {key:?}", file.display())));
                }
                let value = match value {
                    toml::Value::String(s) => s,
                    toml::Value::Integer(n) => n.to_string(),
                    toml::Value::Float(x) => x.to_string(),
                    toml::Value::Boolean(b) => b.to_string(),
                    other => return Err(Error(format!("{key}: {other} is not a setting's value"))),
                };
                values.insert(key, value);
            }
        }
        for key in KEYS {
            if let Some(value) = var(&format!("TGTTS_{}", key.to_uppercase())) {
                values.insert(key.to_owned(), value);
            }
        }
        let settings = Settings(values);
        let cpus = thread::available_parallelism().map_or(1, |n| n.get());
        Ok(Self {
            listen: settings
                .parse("listen")?
                .unwrap_or(SocketAddr::from(([127, 0, 0, 1], 5002))),
            token_file: settings.required("token_file")?,
            voices_dir: settings.required("voices_dir")?,
            assets_dir: settings.required("assets_dir")?,
            checkpoint_dir: settings.required("checkpoint_dir")?,
            cache_dir: settings.required("cache_dir")?,
            glsl_kernels_dir: settings.required("glsl_kernels_dir")?,
            mlir_kernels_dir: settings.required("mlir_kernels_dir")?,
            metrics_db: settings.parse("metrics_db")?,
            ffmpeg: settings.parse("ffmpeg")?.unwrap_or_else(|| "ffmpeg".into()),
            synthesis_limit: settings.limit("synthesis_limit")?.unwrap_or(16),
            prompt_cache_bytes: settings
                .parse("prompt_cache_bytes")?
                .unwrap_or(load::PROMPT_CACHE_BYTES),
            cpu_limit: settings.limit("cpu_limit")?.unwrap_or(cpus),
            queue_timeout: settings
                .seconds("queue_timeout_seconds")?
                .unwrap_or(Duration::from_secs(5)),
            radio_wait: settings
                .seconds("radio_wait_seconds")?
                .unwrap_or(Duration::from_secs(10)),
            radio_wait_ceiling: settings
                .seconds("radio_wait_ceiling_seconds")?
                .unwrap_or(Duration::from_secs(30)),
            loudness: Loudness {
                target: quantity::Loudness::new(
                    settings.decibels("loudness_target_lufs")?.unwrap_or(-16.0),
                ),
                ceiling: match settings.decibels("loudness_ceiling_dbtp")? {
                    Some(ceiling) if ceiling > 0.0 => {
                        let reason = format!("loudness_ceiling_dbtp: {ceiling} is over full scale");
                        return Err(Error(reason));
                    }
                    ceiling => ceiling.map_or(dsp::loudness::CEILING, TruePeak::new),
                },
            },
            voice_levels_file: settings.parse("voice_levels_file")?,
            babble_cap: settings.parse("babble_cap")?.unwrap_or(false),
            log_level: match settings.0.get("log_level") {
                None => LevelFilter::INFO,
                Some(name) => parse_level(&name.to_uppercase())
                    .ok_or_else(|| Error(format!("log_level: no level {name:?}")))?,
            },
        })
    }
}

/// The settings given, as text.
struct Settings(HashMap<String, String>);

impl Settings {
    fn parse<T: FromStr>(&self, key: &str) -> Result<Option<T>, Error>
    where
        T::Err: fmt::Display,
    {
        self.0
            .get(key)
            .map(|value| {
                value
                    .parse()
                    .map_err(|e| Error(format!("{key}: {value:?}: {e}")))
            })
            .transpose()
    }

    fn required<T: FromStr>(&self, key: &str) -> Result<T, Error>
    where
        T::Err: fmt::Display,
    {
        self.parse(key)?
            .ok_or_else(|| Error(format!("{key} is not set")))
    }

    fn limit(&self, key: &str) -> Result<Option<usize>, Error> {
        match self.parse(key)? {
            Some(0) => Err(Error(format!("{key}: 0 would never run anything"))),
            limit => Ok(limit),
        }
    }

    fn decibels(&self, key: &str) -> Result<Option<f64>, Error> {
        match self.parse::<f64>(key)? {
            Some(db) if !db.is_finite() => Err(Error(format!("{key}: {db} is not a level"))),
            db => Ok(db),
        }
    }

    fn seconds(&self, key: &str) -> Result<Option<Duration>, Error> {
        self.parse::<f64>(key)?
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds).map_err(|e| Error(format!("{key}: {e}")))
            })
            .transpose()
    }
}

/// What is logged at `level`: the service's own events at it, and the crates' it uses at it but
/// never finer than `WARN`.
pub fn log_filter(level: LevelFilter) -> Targets {
    Targets::new()
        .with_default(level.min(LevelFilter::WARN))
        .with_target("tgtts", level)
}

/// A configuration that could not be read, and why.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Error(String);

/// Reads the token from `path`: the file's text, without the whitespace around it.
pub fn read_token(path: &Path) -> Result<String, Error> {
    let text = fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let token = text.trim();
    if token.is_empty() {
        return Err(Error(format!("{}: no token", path.display())));
    }
    Ok(token.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUIRED: &str = r#"
        token_file = "/run/secrets/tgtts-token"
        voices_dir = "/var/lib/tgtts/voices"
        assets_dir = "/var/lib/tgtts/assets"
        checkpoint_dir = "/var/lib/tgtts/VoxCPM2"
        cache_dir = "/var/cache/tgtts"
        glsl_kernels_dir = "/k/glsl"
        mlir_kernels_dir = "/k/mlir"
    "#;

    fn load(toml: &str, env: &[(&str, &str)]) -> Result<Config, Error> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tgtts.toml");
        fs::write(&path, toml).unwrap();
        let env: HashMap<String, String> = env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Config::load(Some(&path), |key| env.get(key).cloned())
    }

    #[test]
    fn defaults_what_is_not_given() {
        let config = load(REQUIRED, &[]).unwrap();
        let cpus = thread::available_parallelism().unwrap().get();
        assert_eq!(
            config,
            Config {
                listen: "127.0.0.1:5002".parse().unwrap(),
                token_file: "/run/secrets/tgtts-token".into(),
                voices_dir: "/var/lib/tgtts/voices".into(),
                assets_dir: "/var/lib/tgtts/assets".into(),
                checkpoint_dir: "/var/lib/tgtts/VoxCPM2".into(),
                cache_dir: "/var/cache/tgtts".into(),
                glsl_kernels_dir: "/k/glsl".into(),
                mlir_kernels_dir: "/k/mlir".into(),
                metrics_db: None,
                ffmpeg: "ffmpeg".into(),
                synthesis_limit: 16,
                prompt_cache_bytes: 1 << 29,
                cpu_limit: cpus,
                queue_timeout: Duration::from_secs(5),
                radio_wait: Duration::from_secs(10),
                radio_wait_ceiling: Duration::from_secs(30),
                loudness: Loudness {
                    target: quantity::Loudness::new(-16.0),
                    ceiling: TruePeak::new(-1.0),
                },
                voice_levels_file: None,
                babble_cap: false,
                log_level: LevelFilter::INFO,
            }
        );
    }

    #[test]
    fn reads_every_key_and_lets_the_environment_override_it() {
        let toml = format!(
            "{REQUIRED}
            listen = \"0.0.0.0:5011\"
            metrics_db = \"/var/lib/tgtts/metrics.sqlite\"
            ffmpeg = \"/usr/bin/ffmpeg\"
            synthesis_limit = 4
            prompt_cache_bytes = 536870912
            cpu_limit = 8
            queue_timeout_seconds = 2.5
            radio_wait_seconds = 10
            radio_wait_ceiling_seconds = 45
            loudness_target_lufs = -18
            loudness_ceiling_dbtp = -1.5
            voice_levels_file = \"/var/lib/tgtts/levels.json\"
            log_level = \"debug\"
            babble_cap = true
            "
        );
        let config = load(&toml, &[]).unwrap();
        assert_eq!(config.listen, "0.0.0.0:5011".parse().unwrap());
        assert_eq!(
            config.metrics_db.as_deref(),
            Some(Path::new("/var/lib/tgtts/metrics.sqlite"))
        );
        assert_eq!(config.ffmpeg, Path::new("/usr/bin/ffmpeg"));
        assert_eq!(
            config.voice_levels_file.as_deref(),
            Some(Path::new("/var/lib/tgtts/levels.json"))
        );
        assert_eq!((config.synthesis_limit, config.cpu_limit), (4, 8));
        assert_eq!(config.prompt_cache_bytes, 1 << 29);
        assert!(config.babble_cap);
        assert_eq!(config.queue_timeout, Duration::from_millis(2500));
        assert_eq!(config.radio_wait_ceiling, Duration::from_secs(45));
        let loudness = Loudness {
            target: quantity::Loudness::new(-18.0),
            ceiling: TruePeak::new(-1.5),
        };
        assert_eq!(config.loudness, loudness);
        assert_eq!(config.log_level, LevelFilter::DEBUG);

        let env = [
            ("TGTTS_LISTEN", "[::1]:6000"),
            ("TGTTS_SYNTHESIS_LIMIT", "2"),
            ("TGTTS_LOG_LEVEL", "WARNING"),
            ("TGTTS_TOKEN_FILE", "/etc/token"),
            ("TGTTS_LOUDNESS_TARGET_LUFS", "-14.5"),
        ];
        let config = load(&toml, &env).unwrap();
        assert_eq!(config.listen, "[::1]:6000".parse().unwrap());
        assert_eq!(config.synthesis_limit, 2);
        assert_eq!(config.log_level, LevelFilter::WARN);
        assert_eq!(config.token_file, Path::new("/etc/token"));
        assert_eq!(config.loudness.target, quantity::Loudness::new(-14.5));
    }

    #[test]
    fn may_come_from_the_environment_alone() {
        let env: HashMap<&str, &str> = [
            ("TGTTS_TOKEN_FILE", "/t"),
            ("TGTTS_VOICES_DIR", "/v"),
            ("TGTTS_ASSETS_DIR", "/a"),
            ("TGTTS_CHECKPOINT_DIR", "/c"),
            ("TGTTS_CACHE_DIR", "/k"),
            ("TGTTS_GLSL_KERNELS_DIR", "/g"),
            ("TGTTS_MLIR_KERNELS_DIR", "/m"),
        ]
        .into();
        let config = Config::load(None, |key| env.get(key).map(|v| (*v).to_owned())).unwrap();
        assert_eq!(config.checkpoint_dir, Path::new("/c"));
    }

    #[test]
    fn refuses_what_it_cannot_use() {
        let error = |toml: &str, env: &[(&str, &str)]| load(toml, env).unwrap_err().0;
        assert_eq!(error("", &[]), "token_file is not set");
        let unknown = error(&format!("{REQUIRED}\ntoken = \"hunter2\""), &[]);
        assert!(unknown.ends_with("no setting \"token\""), "{unknown}");
        // One device is left, which no setting names.
        let gone = error(&format!("{REQUIRED}\ndevice = \"rt\""), &[]);
        assert!(gone.ends_with("no setting \"device\""), "{gone}");
        assert!(error(REQUIRED, &[("TGTTS_LISTEN", "localhost")]).starts_with("listen:"));
        assert!(error(REQUIRED, &[("TGTTS_CPU_LIMIT", "0")]).starts_with("cpu_limit:"));
        assert!(error(REQUIRED, &[("TGTTS_RADIO_WAIT_SECONDS", "-1")]).starts_with("radio_wait"));
        assert!(error(REQUIRED, &[("TGTTS_LOG_LEVEL", "loud")]).starts_with("log_level:"));
        let babble = error(REQUIRED, &[("TGTTS_BABBLE_CAP", "1")]);
        assert!(babble.starts_with("babble_cap:"), "{babble}");
        let kernels = REQUIRED.replace("mlir_kernels_dir = \"/k/mlir\"", "");
        assert_eq!(error(&kernels, &[]), "mlir_kernels_dir is not set");
        for (key, value) in [
            ("TGTTS_LOUDNESS_TARGET_LUFS", "NaN"),
            ("TGTTS_LOUDNESS_TARGET_LUFS", "-inf"),
            ("TGTTS_LOUDNESS_CEILING_DBTP", "0.5"),
            ("TGTTS_LOUDNESS_CEILING_DBTP", "loud"),
        ] {
            let error = error(REQUIRED, &[(key, value)]);
            assert!(error.starts_with("loudness_"), "{value}: {error}");
        }
        assert!(error("x = [", &[]).contains("tgtts.toml"));
    }

    #[test]
    fn logs_the_crates_it_uses_no_finer_than_warn() {
        use tracing::Level;

        let enabled =
            |filter: &Targets, target: &str, level: Level| filter.would_enable(target, &level);
        let debug = log_filter(LevelFilter::DEBUG);
        assert!(enabled(&debug, "tgtts::rt", Level::DEBUG));
        assert!(!enabled(&debug, "graph::generator", Level::INFO));
        assert!(enabled(&debug, "vulkan::context", Level::WARN));
        let warn = log_filter(LevelFilter::WARN);
        assert!(!enabled(&warn, "tgtts::rt", Level::INFO));
        let error = log_filter(LevelFilter::ERROR);
        assert!(!enabled(&error, "vulkan::context", Level::WARN));
    }

    #[test]
    fn reads_the_token_without_its_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        fs::write(&path, "s3cret\n").unwrap();
        assert_eq!(read_token(&path).unwrap(), "s3cret");
        fs::write(&path, " \n").unwrap();
        assert!(read_token(&path).is_err());
        assert!(read_token(&dir.path().join("missing")).is_err());
    }
}
