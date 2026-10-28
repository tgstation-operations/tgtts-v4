//! The device the model runs on, as the configuration names it.

use std::{
    fmt,
    path::Path,
    str::FromStr,
    sync::atomic::{AtomicUsize, Ordering},
};

use burn::tensor::{Device, FloatDType};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{
    filter::{LevelFilter, Targets},
    layer::{Context, Layer},
};
use voxcpm_core::{Dtype, Dtypes};

/// What is logged at `level`: the service's own events at it, the crates' it uses at it but
/// never finer than `WARN`, as pliron logs every kernel cubecl compiles at `INFO` after each of
/// its passes, gigabytes in minutes; and the kernels autotune tunes, which cubecl logs at `INFO`
/// and [`Tunings`] counts, whatever the level.
pub fn log_filter(level: LevelFilter) -> Targets {
    Targets::new()
        .with_default(level.min(LevelFilter::WARN))
        .with_target("tgtts", level)
        .with_target("cubecl_runtime::tune", LevelFilter::INFO)
}

/// The kernels autotune has tuned in this process, as [`Tunings`] counts them.
static TUNED: AtomicUsize = AtomicUsize::new(0);

/// How many kernels autotune has tuned in this process, as the layer [`Tunings`] has seen
/// cubecl log them: none where it is not installed or the log level is past `INFO`.
pub fn tuned() -> usize {
    TUNED.load(Ordering::Relaxed)
}

/// Counts the kernels autotune tunes, by the line cubecl logs at `INFO` as it starts on each,
/// for [`tuned`]: one tuned while requests wait holds them up for as long as it times its
/// candidates.
#[derive(Debug, Default)]
pub struct Tunings;

impl<S: Subscriber> Layer<S> for Tunings {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        struct Message(bool);
        impl Visit for Message {
            fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
                if field.name() == "message" {
                    let text = format!("{value:?}");
                    self.0 = text.starts_with("Tuning ") && text.contains("AutotuneKey");
                }
            }
        }
        let mut message = Message(false);
        event.record(&mut message);
        if message.0 {
            TUNED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A device the configuration can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The CPU, through burn's Flex backend.
    Cpu,
    /// The GPU wgpu prefers among the Vulkan devices, or the one `CUBECL_WGPU_DEFAULT_DEVICE`
    /// names.
    Vulkan,
    /// The first CUDA device, or the first `CUDA_VISIBLE_DEVICES` names.
    Cuda,
    /// tts-rs's own runtime on the Vulkan GPU the `vulkan` crate chooses, rather than burn.
    Rt,
}

impl Kind {
    /// Whether autotune chooses its kernels for each shape as it first runs, by timing them,
    /// which the first request of each shape would wait for.
    pub fn autotunes(self) -> bool {
        matches!(self, Self::Vulkan | Self::Cuda)
    }

    /// What each part of the model runs in by default: f32 on the CPU, and on a GPU f16 but for
    /// the audio decoder, whose first transposed convolution sums in f16 there and takes its
    /// audio to 46 dB of upstream's, where the CPU in f16 reaches 56 dB. The rt device runs in
    /// those alone, storing f16 and computing in f32.
    pub fn dtypes(self) -> Dtypes {
        let (f32, f16) = (Dtype::F32, Dtype::F16);
        match self {
            Self::Cpu => Dtypes {
                lm: f32,
                local_encoder: f32,
                local_dit: f32,
                audio_decoder: f32,
            },
            Self::Vulkan | Self::Cuda | Self::Rt => Dtypes {
                lm: f16,
                local_encoder: f16,
                local_dit: f16,
                audio_decoder: f32,
            },
        }
    }
}

impl FromStr for Kind {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "vulkan" => Ok(Self::Vulkan),
            "cuda" => Ok(Self::Cuda),
            "rt" => Ok(Self::Rt),
            _ => Err("not cpu, vulkan, cuda or rt".to_owned()),
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "cpu",
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
            Self::Rt => "rt",
        })
    }
}

/// A dtype as the configuration names it.
pub fn parse_dtype(name: &str) -> Result<Dtype, String> {
    match name {
        "f32" => Ok(Dtype::F32),
        "f16" => Ok(Dtype::F16),
        "bf16" => Ok(Dtype::BF16),
        _ => Err("not f32, f16 or bf16".to_owned()),
    }
}

/// `dtype` as burn names it.
pub fn float_dtype(dtype: Dtype) -> FloatDType {
    match dtype {
        Dtype::F32 => FloatDType::F32,
        Dtype::F16 => FloatDType::F16,
        Dtype::BF16 => FloatDType::BF16,
    }
}

/// What the model is opened on.
#[derive(Debug)]
pub enum Opened {
    /// burn's device.
    Burn(Device),
    /// tts-rs's own runtime, which opens its GPU as the model loads.
    #[cfg(feature = "rt")]
    Rt,
}

/// Opens the device of `kind`, with autotune keeping its results in `autotune_dir` on a GPU,
/// where cubecl would keep them without one, and checks that it computes in each of `dtypes`.
///
/// # Errors
///
/// If the build cannot open such a device, it is not what it should be, or it does not compute
/// in one of the dtypes.
pub fn open(kind: Kind, dtypes: &Dtypes, autotune_dir: Option<&Path>) -> Result<Opened, String> {
    let device = match kind {
        Kind::Cpu => Device::flex(),
        Kind::Vulkan => vulkan(autotune_dir)?,
        Kind::Cuda => cuda(autotune_dir)?,
        Kind::Rt => return rt(dtypes),
    };
    let parts = [
        ("lm_dtype", dtypes.lm),
        ("local_encoder_dtype", dtypes.local_encoder),
        ("local_dit_dtype", dtypes.local_dit),
        ("audio_decoder_dtype", dtypes.audio_decoder),
    ];
    for (key, dtype) in parts {
        if !device.supports_dtype(float_dtype(dtype)) {
            return Err(format!(
                "{key}: the {kind} device does not compute in {dtype:?}"
            ));
        }
    }
    Ok(Opened::Burn(device))
}

#[cfg(feature = "vulkan")]
fn vulkan(autotune_dir: Option<&Path>) -> Result<Device, String> {
    if let Some(dir) = autotune_dir {
        gpu::keep_autotune_in(dir).map_err(|e| e.to_string())?;
    }
    gpu::vulkan().map_err(|e| e.to_string())
}

#[cfg(not(feature = "vulkan"))]
fn vulkan(_: Option<&Path>) -> Result<Device, String> {
    Err("device vulkan: this build has no Vulkan; build tgtts with --features vulkan".to_owned())
}

#[cfg(feature = "cuda")]
fn cuda(autotune_dir: Option<&Path>) -> Result<Device, String> {
    if let Some(dir) = autotune_dir {
        gpu::keep_autotune_in(dir).map_err(|e| e.to_string())?;
    }
    gpu::cuda().map_err(|e| e.to_string())
}

#[cfg(not(feature = "cuda"))]
fn cuda(_: Option<&Path>) -> Result<Device, String> {
    Err("device cuda: this build has no CUDA; build tgtts with --features cuda".to_owned())
}

#[cfg(feature = "rt")]
fn rt(dtypes: &Dtypes) -> Result<Opened, String> {
    if *dtypes != Kind::Rt.dtypes() {
        return Err(format!(
            "the rt device computes in its own dtypes alone, {:?}",
            Kind::Rt.dtypes()
        ));
    }
    Ok(Opened::Rt)
}

#[cfg(not(feature = "rt"))]
fn rt(_: &Dtypes) -> Result<Opened, String> {
    Err("device rt: this build has no rt; build tgtts with --features rt".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_the_crates_it_uses_no_finer_than_warn() {
        use tracing::Level;

        let enabled =
            |filter: &Targets, target: &str, level: Level| filter.would_enable(target, &level);
        let debug = log_filter(LevelFilter::DEBUG);
        assert!(enabled(&debug, "tgtts::model", Level::DEBUG));
        assert!(!enabled(&debug, "pliron::pass", Level::INFO));
        assert!(enabled(&debug, "cubecl_runtime::server", Level::WARN));
        assert!(enabled(&debug, "cubecl_runtime::tune::tuner", Level::INFO));
        let warn = log_filter(LevelFilter::WARN);
        assert!(!enabled(&warn, "tgtts::model", Level::INFO));
        assert!(enabled(&warn, "cubecl_runtime::tune::tuner", Level::INFO));
        let error = log_filter(LevelFilter::ERROR);
        assert!(!enabled(&error, "cubecl_runtime::server", Level::WARN));
    }

    #[test]
    fn counts_the_kernels_autotune_tunes() {
        use tracing_subscriber::layer::SubscriberExt;

        let before = tuned();
        let subscriber = tracing_subscriber::registry().with(Tunings);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("Tuning MatmulAutotuneKey - Definition: m: 128, n: 2048, k: 1024");
            tracing::info!("Tuning ConvAutotuneKey {{ kernel_size: [7] }}");
            tracing::info!("Tuning done");
            tracing::info!("Load autotune cache ...");
        });
        assert_eq!(tuned() - before, 2);
    }

    #[test]
    fn names_devices_and_dtypes() {
        assert_eq!("cpu".parse(), Ok(Kind::Cpu));
        assert_eq!("vulkan".parse(), Ok(Kind::Vulkan));
        assert_eq!("cuda".parse(), Ok(Kind::Cuda));
        assert_eq!("rt".parse(), Ok(Kind::Rt));
        assert!("rocm".parse::<Kind>().is_err());
        assert_eq!(parse_dtype("bf16"), Ok(Dtype::BF16));
        assert!(parse_dtype("f64").is_err());
    }

    #[test]
    fn opens_the_cpu_in_every_dtype() {
        let bf16 = Dtypes {
            lm: Dtype::BF16,
            ..Kind::Cpu.dtypes()
        };
        for dtypes in [Kind::Cpu.dtypes(), Kind::Vulkan.dtypes(), bf16] {
            assert!(matches!(
                open(Kind::Cpu, &dtypes, None),
                Ok(Opened::Burn(_))
            ));
        }
    }

    #[cfg(feature = "rt")]
    #[test]
    fn opens_rt_in_its_own_dtypes_alone() {
        assert!(matches!(
            open(Kind::Rt, &Kind::Rt.dtypes(), None),
            Ok(Opened::Rt)
        ));
        let error = open(Kind::Rt, &Kind::Cpu.dtypes(), None).unwrap_err();
        assert!(error.contains("its own dtypes"), "{error}");
    }

    #[cfg(not(feature = "rt"))]
    #[test]
    fn refuses_rt_in_a_build_without_it() {
        let error = open(Kind::Rt, &Kind::Rt.dtypes(), None);
        assert!(error.unwrap_err().contains("--features rt"));
    }

    #[cfg(not(feature = "vulkan"))]
    #[test]
    fn refuses_vulkan_in_a_build_without_it() {
        let error = open(Kind::Vulkan, &Kind::Vulkan.dtypes(), None);
        assert!(error.unwrap_err().contains("--features vulkan"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn refuses_cuda_in_a_build_without_it() {
        let error = open(Kind::Cuda, &Kind::Cuda.dtypes(), None);
        assert!(error.unwrap_err().contains("--features cuda"));
    }
}
