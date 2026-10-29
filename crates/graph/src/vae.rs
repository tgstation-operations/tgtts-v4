//! VoxCPM2's audio VAE decoder, streamed a chunk of latent frames at a time: latent frames in,
//! `decoder_rates.product()` samples of audio per frame out, each causal convolution carrying its
//! last inputs from one chunk into the next, so that chunked audio is the whole decode's.
//!
//! A chunk is a step recorded once for at most [`VaeShape::chunk`] frames and submitted for every
//! chunk, its length read from a buffer the host writes before each: the depthwise convolution on
//! the latents and the pointwise one to `decoder_dim` channels; per block, the transposed
//! convolution with the block's condition and Snake before it, and three residual units, each a
//! depthwise convolution between two Snakes and a pointwise convolution with the residual; then
//! Snake, the convolution to one channel and tanh. The pointwise and transposed convolutions are
//! GEMMs over the chunk's steps; what a convolution keeps for the next chunk is a ring of its
//! last inputs, which a dispatch after it writes once it has read them.
//!
//! A decoder decodes several streams, a chunk of one of them at a time: each convolution keeps a
//! ring for each stream, and the host says which stream a chunk is of in the buffer it writes
//! before each, so that each stream comes out bit for bit as it does alone.
//!
//! Everything is stored in f32: storing activations and weights in f16 took the audio from 115 dB
//! of the CPU's f32 decode to 59, to save less than a sixth of a chunk's time.

use std::iter;

use serde::Deserialize;
use vulkan::{Buffer, Context, Residence, SpecValue, Timing};

use crate::{
    Error,
    graph::{Bindings, Graph, Kind, Pipeline, Tensor},
    kernels::{KernelInfo, Kernels, read, write},
    step::{Region, Step},
};

/// `vae_gemm.comp`: chunk, x, ring, prologue, W, bias, residual -> out.
pub static VAE_GEMM: KernelInfo = KernelInfo {
    name: "vae_gemm",
    bindings: &[
        read::<u32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        write::<f32>(),
    ],
};

/// `vae_depthwise.comp`: chunk, x, ring, prologue, w, bias, epilogue -> out.
pub static VAE_DEPTHWISE: KernelInfo = KernelInfo {
    name: "vae_depthwise",
    bindings: &[
        read::<u32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        write::<f32>(),
    ],
};

/// `vae_keep.comp`: chunk, x -> ring, with the prologue.
pub static VAE_KEEP: KernelInfo = KernelInfo {
    name: "vae_keep",
    bindings: &[read::<u32>(), read::<f32>(), write::<f32>(), read::<f32>()],
};

/// `vae_out.comp`: chunk, x, ring, prologue, w, bias -> audio.
pub static VAE_OUT: KernelInfo = KernelInfo {
    name: "vae_out",
    bindings: &[
        read::<u32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        read::<f32>(),
        write::<f32>(),
    ],
};

/// The width of every workgroup; no kernel here needs a subgroup size.
const WORKGROUP: u32 = 256;

/// The columns of A each step of the GEMM takes, fixed in the kernel.
const GEMM_BK: usize = 16;

/// The channels and rows a workgroup of the depthwise convolution takes, fixed in the kernel.
const DEPTHWISE_COLUMNS: usize = 32;
const DEPTHWISE_ROWS: usize = 64;

/// The kernel of every convolution but the pointwise and transposed ones.
const KERNEL: usize = 7;

/// What the decoder reads of VoxCPM2's `audio_vae_config`, with upstream's defaults for what the
/// JSON leaves out; the encoder's fields are ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct VaeConfig {
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub out_sample_rate: u32,
    pub sr_bin_boundaries: Option<Vec<u32>>,
    #[serde(default = "upstream_true")]
    pub depthwise: bool,
    #[serde(default)]
    pub use_noise_block: bool,
    #[serde(default = "upstream_cond_type")]
    pub cond_type: String,
    #[serde(default)]
    pub cond_out_layer: bool,
}

fn upstream_true() -> bool {
    true
}

fn upstream_cond_type() -> String {
    "scale_bias".into()
}

/// The shapes a step is recorded at.
#[derive(Clone, Debug, PartialEq)]
pub struct VaeShape {
    pub latent: usize,
    /// Channels of the first block; each block halves them.
    pub channels: usize,
    pub rates: Vec<usize>,
    /// The sample-rate bucket the output rate falls in, and how many there are.
    pub bucket: usize,
    pub buckets: usize,
    /// The most frames a chunk holds.
    pub chunk: usize,
    /// The streams it decodes.
    pub streams: usize,
}

impl VaeShape {
    /// The decoder `config` describes, of `streams` streams in chunks of at most `chunk` frames.
    /// Refuses what VoxCPM2 does not use.
    pub fn new(config: &VaeConfig, chunk: usize, streams: usize) -> Result<Self, Error> {
        let unsupported = |why: String| Err(Error::Shape(why));
        if !config.depthwise
            || config.use_noise_block
            || config.cond_out_layer
            || !matches!(config.cond_type.as_str(), "scale_bias" | "scale_bias_init")
        {
            return unsupported(format!("audio VAE {config:?}"));
        }
        let Some(boundaries) = config.sr_bin_boundaries.as_ref().filter(|b| b.is_sorted()) else {
            return unsupported(format!("sr_bin_boundaries {:?}", config.sr_bin_boundaries));
        };
        let rates = &config.decoder_rates;
        if rates.is_empty() || rates.contains(&0) || config.decoder_dim >> rates.len() == 0 {
            return unsupported(format!(
                "decoder_rates {rates:?} from {} channels",
                config.decoder_dim
            ));
        }
        if chunk == 0 || streams == 0 {
            return unsupported(format!("chunks of {chunk} frames of {streams} streams"));
        }
        Ok(VaeShape {
            latent: config.latent_dim,
            channels: config.decoder_dim,
            rates: rates.clone(),
            bucket: boundaries.partition_point(|&b| b < config.out_sample_rate),
            buckets: boundaries.len() + 1,
            chunk,
            streams,
        })
    }

    /// Samples of audio per latent frame.
    pub fn hop(&self) -> usize {
        self.rates.iter().product()
    }
}

/// How a constant of the step is filled from the checkpoint's tensor of that name.
#[derive(Clone, Debug)]
pub enum Fill {
    /// As it is, in f32.
    Plain(String),
    /// A pointwise convolution's weight `[out][in][1]` as the GEMM's W `[in][out]`.
    Pointwise(String),
    /// A transposed convolution's weight `[in][out][2 stride]` as the GEMM's W
    /// `[2 in][stride out]`: the previous step's tap above the current step's.
    Transposed(String, usize),
    /// A bias `[out]` repeated for each of a transposed convolution's `stride` outputs.
    Repeated(String, usize),
    /// The last convolution's weight `[1][in][KERNEL]` as `[KERNEL][in]`.
    Output(String),
    /// Per channel: the scale and bias of the condition named (1 and 0 without), Snake's alpha
    /// named, and the reciprocal of alpha + 1e-9.
    Snake {
        alpha: String,
        condition: Option<String>,
    },
}

/// The step's graph, the tensors the host writes and reads, and where each constant comes from.
#[derive(Clone, Debug)]
pub struct VaeGraph {
    pub graph: Graph,
    pub shape: VaeShape,
    /// frames_before, frames and the stream.
    pub chunk: Tensor<u32>,
    /// `[chunk][latent]`.
    pub latents: Tensor<f32>,
    /// `[chunk * hop]` samples.
    pub audio: Tensor<f32>,
    /// Each convolution's ring of its last inputs, `[streams][rows][channels]`, and how many rows
    /// it holds.
    pub rings: Vec<(Tensor<f32>, usize)>,
    pub constants: Vec<(Tensor<f32>, Fill)>,
}

/// A tensor of activations: `[up * frames][channels]`, of a chunk of frames.
#[derive(Clone, Copy, Debug)]
struct Activations {
    id: Tensor<f32>,
    up: usize,
    channels: usize,
}

/// A keep held back until one more op is recorded: it waits for a barrier after its
/// convolution, which reads the ring it writes, and the op after the convolution needs one anyway.
struct Keep {
    x: Activations,
    ring: Tensor<f32>,
    reach: usize,
    prologue: Option<Tensor<f32>>,
    label: String,
}

/// The graph as it is built, op by op.
struct Builder {
    g: Graph,
    shape: VaeShape,
    chunk: Tensor<u32>,
    /// Bound where a kernel's binding goes unused.
    unused: Tensor<f32>,
    rings: Vec<(Tensor<f32>, usize)>,
    constants: Vec<(Tensor<f32>, Fill)>,
    keep: Option<Keep>,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

impl Builder {
    fn activations(&mut self, name: String, up: usize, channels: usize) -> Activations {
        let shape = [self.shape.chunk * up, channels];
        let id = self.g.tensor(name, &shape, Kind::Intermediate);
        Activations { id, up, channels }
    }

    fn constant(&mut self, shape: &[usize], fill: Fill) -> Tensor<f32> {
        let name = match &fill {
            Fill::Plain(n)
            | Fill::Pointwise(n)
            | Fill::Transposed(n, _)
            | Fill::Repeated(n, _)
            | Fill::Output(n)
            | Fill::Snake { alpha: n, .. } => n.clone(),
        };
        let id = self.g.tensor(name, shape, Kind::Constant);
        self.constants.push((id, fill));
        id
    }

    fn snake(&mut self, alpha: String, condition: Option<String>, channels: usize) -> Tensor<f32> {
        self.constant(&[channels, 4], Fill::Snake { alpha, condition })
    }

    fn ring(&mut self, name: String, reach: usize, channels: usize) -> Tensor<f32> {
        let shape = [self.shape.streams, reach, channels];
        let id = self.g.tensor(name, &shape, Kind::State);
        self.rings.push((id, reach));
        id
    }

    /// The constants every kernel here takes, for a convolution over `x` reaching `reach` back.
    fn pipeline(
        &self,
        kernel: &'static KernelInfo,
        x: Activations,
        reach: usize,
        prologue: bool,
        more: &[(u32, SpecValue)],
    ) -> Pipeline {
        let constants = [
            (0, SpecValue::U32(WORKGROUP)),
            (1, u(x.up)),
            (2, u(x.channels)),
            (3, u(reach)),
            (4, SpecValue::Bool(prologue)),
            (5, u(self.shape.streams)),
        ];
        Pipeline {
            kernel,
            constants: constants.iter().chain(more).copied().collect(),
            subgroup_size: None,
        }
    }

    fn op(&mut self, label: String, pipeline: Pipeline, bindings: impl Bindings, groups: [u32; 3]) {
        self.g.op(label, pipeline, bindings, groups);
        if let Some(keep) = self.keep.take() {
            self.emit_keep(keep);
        }
    }

    /// Queues the keep of `ring` from `x` after the op that follows.
    fn keep(
        &mut self,
        label: String,
        x: Activations,
        ring: Tensor<f32>,
        reach: usize,
        prologue: Option<Tensor<f32>>,
    ) {
        if let Some(keep) = self.keep.replace(Keep {
            x,
            ring,
            reach,
            prologue,
            label,
        }) {
            self.emit_keep(keep);
        }
    }

    fn emit_keep(&mut self, keep: Keep) {
        let pipeline = self.pipeline(&VAE_KEEP, keep.x, keep.reach, keep.prologue.is_some(), &[]);
        let groups = (keep.reach * keep.x.channels).div_ceil(WORKGROUP as usize) as u32;
        let bindings = (
            self.chunk,
            keep.x.id,
            keep.ring,
            keep.prologue.unwrap_or(self.unused),
        );
        self.g.op(keep.label, pipeline, bindings, [groups, 1, 1]);
    }

    /// A depthwise causal convolution of kernel 7 over `x`, of the weights under `prefix`, with
    /// Snake before and after where given.
    fn depthwise(
        &mut self,
        label: &str,
        x: Activations,
        prefix: &str,
        dilation: usize,
        snakes: Option<(String, String)>,
    ) -> Result<Activations, Error> {
        let (c, reach) = (x.channels, (KERNEL - 1) * dilation);
        if !c.is_multiple_of(DEPTHWISE_COLUMNS) {
            return Err(Error::Shape(format!(
                "{label}: {c} channels in tiles of {DEPTHWISE_COLUMNS}"
            )));
        }
        let w = self.constant(&[c, KERNEL], Fill::Plain(format!("{prefix}.weight")));
        let bias = self.constant(&[c], Fill::Plain(format!("{prefix}.bias")));
        let (prologue, epilogue) = match snakes {
            Some((before, after)) => (
                Some(self.snake(before, None, c)),
                Some(self.snake(after, None, c)),
            ),
            None => (None, None),
        };
        let ring = self.ring(format!("{label}.ring"), reach, c);
        let out = self.activations(format!("{label}.out"), x.up, c);
        let pipeline = self.pipeline(
            &VAE_DEPTHWISE,
            x,
            reach,
            prologue.is_some(),
            &[(6, SpecValue::Bool(epilogue.is_some()))],
        );
        let groups = [
            (c / DEPTHWISE_COLUMNS) as u32,
            (self.shape.chunk * x.up).div_ceil(DEPTHWISE_ROWS) as u32,
            1,
        ];
        let bindings = (
            self.chunk,
            x.id,
            ring,
            prologue.unwrap_or(self.unused),
            w,
            bias,
            epilogue.unwrap_or(self.unused),
            out.id,
        );
        self.op(label.into(), pipeline, bindings, groups);
        self.keep(format!("{label}.keep"), x, ring, reach, prologue);
        Ok(out)
    }

    /// The GEMM `out = A W + bias (+ residual)` over `x`, A's rows `taps` steps of `x` each.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        label: &str,
        x: Activations,
        taps: usize,
        (w, bias): (Tensor<f32>, Tensor<f32>),
        n: usize,
        prologue: Option<Tensor<f32>>,
        residual: Option<Activations>,
        out: Activations,
    ) -> Result<(), Error> {
        let rows = self.shape.chunk * x.up;
        let (wn, tn) = (16, if n.is_multiple_of(64) { 4 } else { 2 });
        let tm = if rows <= 32 { 2 } else { 4 };
        let (bm, bn) = (tm * (WORKGROUP as usize / wn), tn * wn);
        if !n.is_multiple_of(bn) || !x.channels.is_multiple_of(GEMM_BK) {
            return Err(Error::Shape(format!(
                "{label}: {} by {n} in tiles of {GEMM_BK} by {bn}",
                x.channels
            )));
        }
        let reach = taps - 1;
        let ring = (reach > 0).then(|| self.ring(format!("{label}.ring"), reach, x.channels));
        let pipeline = self.pipeline(
            &VAE_GEMM,
            x,
            reach,
            prologue.is_some(),
            &[
                (6, u(taps)),
                (7, u(n)),
                (8, SpecValue::Bool(residual.is_some())),
                (9, u(tm)),
                (10, u(tn)),
                (11, u(wn)),
            ],
        );
        let bindings = (
            self.chunk,
            x.id,
            ring.unwrap_or(self.unused),
            prologue.unwrap_or(self.unused),
            w,
            bias,
            residual.map_or(self.unused, |r| r.id),
            out.id,
        );
        let groups = [(n / bn) as u32, rows.div_ceil(bm) as u32, 1];
        self.op(label.into(), pipeline, bindings, groups);
        if let Some(ring) = ring {
            self.keep(format!("{label}.keep"), x, ring, reach, prologue);
        }
        Ok(())
    }

    /// A pointwise convolution of the weights under `prefix` from `x` to `channels`.
    fn pointwise(
        &mut self,
        label: &str,
        x: Activations,
        prefix: &str,
        channels: usize,
        residual: Option<Activations>,
    ) -> Result<Activations, Error> {
        let w = self.constant(
            &[x.channels, channels],
            Fill::Pointwise(format!("{prefix}.weight")),
        );
        let bias = self.constant(&[channels], Fill::Plain(format!("{prefix}.bias")));
        let out = self.activations(format!("{label}.out"), x.up, channels);
        self.gemm(label, x, 1, (w, bias), channels, None, residual, out)?;
        Ok(out)
    }

    /// Decoder block `i` over `x`: its condition and Snake, the transposed convolution
    /// upsampling by `stride`, and three residual units.
    fn block(&mut self, i: usize, x: Activations, stride: usize) -> Result<Activations, Error> {
        let at = 2 + i;
        let (c_in, c) = (x.channels, x.channels / 2);
        let prefix = format!("model.{at}.block");
        let prologue = self.snake(
            format!("{prefix}.0.alpha"),
            Some(format!("sr_cond_model.{at}")),
            c_in,
        );
        let w = self.constant(
            &[2 * c_in, stride * c],
            Fill::Transposed(format!("{prefix}.1.weight"), stride),
        );
        let bias = self.constant(
            &[stride * c],
            Fill::Repeated(format!("{prefix}.1.bias"), stride),
        );
        let label = format!("block{i}");
        let mut y = self.activations(format!("{label}.upsample.out"), x.up * stride, c);
        self.gemm(
            &format!("{label}.upsample"),
            x,
            2,
            (w, bias),
            stride * c,
            Some(prologue),
            None,
            y,
        )?;
        for (unit, dilation) in [(2, 1), (3, 3), (4, 9)] {
            let unit_prefix = format!("{prefix}.{unit}.block");
            let label = format!("{label}.unit{unit}");
            let h = self.depthwise(
                &format!("{label}.depthwise"),
                y,
                &format!("{unit_prefix}.1"),
                dilation,
                Some((
                    format!("{unit_prefix}.0.alpha"),
                    format!("{unit_prefix}.2.alpha"),
                )),
            )?;
            y = self.pointwise(
                &format!("{label}.pointwise"),
                h,
                &format!("{unit_prefix}.3"),
                c,
                Some(y),
            )?;
        }
        Ok(y)
    }
}

/// The decoder's step of `shape`.
pub fn vae_graph(shape: &VaeShape) -> Result<VaeGraph, Error> {
    let mut g = Graph::default();
    let chunk = g.tensor("chunk", &[3], Kind::Param);
    let unused = g.tensor("unused", &[4], Kind::Constant);
    let latents = g.tensor("latents", &[shape.chunk, shape.latent], Kind::Input);
    let audio = g.tensor("audio", &[shape.chunk * shape.hop()], Kind::Output);
    let mut b = Builder {
        g,
        shape: shape.clone(),
        chunk,
        unused,
        rings: Vec::new(),
        constants: Vec::new(),
        keep: None,
    };
    let latents_in = Activations {
        id: latents,
        up: 1,
        channels: shape.latent,
    };
    let h = b.depthwise("in.depthwise", latents_in, "model.0", 1, None)?;
    let mut x = b.pointwise("in.pointwise", h, "model.1", shape.channels, None)?;
    for (i, &stride) in shape.rates.iter().enumerate() {
        x = b.block(i, x, stride)?;
    }
    let last = shape.rates.len() + 2;
    let c = x.channels;
    let prologue = b.snake(format!("model.{last}.alpha"), None, c);
    let w = b.constant(
        &[KERNEL, c],
        Fill::Output(format!("model.{}.weight", last + 1)),
    );
    let bias = b.constant(&[1], Fill::Plain(format!("model.{}.bias", last + 1)));
    let reach = KERNEL - 1;
    let ring = b.ring("out.ring".into(), reach, c);
    let pipeline = b.pipeline(&VAE_OUT, x, reach, true, &[]);
    let groups = (shape.chunk * x.up).div_ceil(WORKGROUP as usize) as u32;
    b.op(
        "out".into(),
        pipeline,
        (chunk, x.id, ring, prologue, w, bias, audio),
        [groups, 1, 1],
    );
    b.emit_keep(Keep {
        x,
        ring,
        reach,
        prologue: Some(prologue),
        label: "out.keep".into(),
    });
    Ok(VaeGraph {
        graph: b.g,
        shape: shape.clone(),
        chunk,
        latents,
        audio,
        rings: b.rings,
        constants: b.constants,
    })
}

/// A parameter's values in torch's layout, and its shape, by its name under the decoder.
pub type Parameter = (Vec<usize>, Vec<f32>);

/// The audio VAE's decoder on the GPU, with its weights, and the streams it is decoding.
#[derive(Debug)]
pub struct VaeDecoder {
    step: Step,
    tensors: VaeGraph,
    /// The frames each stream has decoded, modulo `period`.
    frames_before: Vec<u32>,
    /// A multiple of every ring's length.
    period: u32,
    /// Zeros, as many as a stream's largest ring holds.
    zeros: Buffer<f32>,
}

impl VaeDecoder {
    /// Records the step of `shape`, with every weight zero until [`load`](Self::load) and every
    /// stream at its start.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        shape: &VaeShape,
        timed: bool,
    ) -> Result<Self, Error> {
        let tensors = vae_graph(shape)?;
        let step = Step::new(ctx, kernels, &tensors.graph, timed)?;
        let period = tensors
            .rings
            .iter()
            .fold(1, |period, &(_, reach)| lcm(period, reach as u32));
        let ring = |&(t, _): &(Tensor<f32>, usize)| tensors.graph.info(t).len();
        let largest = tensors.rings.iter().map(ring).max().unwrap_or(0);
        let zeros = ctx.buffer_in(largest / shape.streams, Residence::Device)?;
        Ok(VaeDecoder {
            step,
            frames_before: vec![0; shape.streams],
            tensors,
            period,
            zeros,
        })
    }

    /// Writes the weights, `parameter` giving each by its name under the decoder (upstream's
    /// state_dict names less `decoder.`), weight norm folded, as
    /// [`Pth::floats`](crate::pth::Pth::floats) gives them.
    pub fn load(
        &mut self,
        mut parameter: impl FnMut(&str) -> Result<Parameter, Error>,
    ) -> Result<(), Error> {
        let VaeShape { bucket, .. } = self.tensors.shape;
        for (t, fill) in self.tensors.constants.clone() {
            let values: Vec<f32> = match &fill {
                Fill::Plain(name) => parameter(name)?.1,
                Fill::Pointwise(name) => {
                    let (shape, w) = parameter(name)?;
                    let [out, inputs, 1] = shape[..] else {
                        return Err(bad_shape(name, &shape));
                    };
                    (0..inputs * out)
                        .map(|i| w[i % out * inputs + i / out])
                        .collect()
                }
                Fill::Transposed(name, stride) => {
                    let (shape, w) = parameter(name)?;
                    let s = *stride;
                    let [inputs, out, kernel] = shape[..] else {
                        return Err(bad_shape(name, &shape));
                    };
                    if kernel != 2 * s {
                        return Err(bad_shape(name, &shape));
                    }
                    // Row (tap, i), column (j, o): the previous step's tap is kernel index
                    // j + s, the current step's j.
                    let mut values = Vec::with_capacity(w.len());
                    for tap in 0..2 {
                        for i in 0..inputs {
                            for j in 0..s {
                                let k = j + if tap == 0 { s } else { 0 };
                                values.extend((0..out).map(|o| w[(i * out + o) * kernel + k]));
                            }
                        }
                    }
                    values
                }
                Fill::Repeated(name, stride) => parameter(name)?.1.repeat(*stride),
                Fill::Output(name) => {
                    let (shape, w) = parameter(name)?;
                    let [1, inputs, KERNEL] = shape[..] else {
                        return Err(bad_shape(name, &shape));
                    };
                    (0..KERNEL * inputs)
                        .map(|i| w[i % inputs * KERNEL + i / inputs])
                        .collect()
                }
                Fill::Snake { alpha, condition } => {
                    let alpha = parameter(alpha)?.1;
                    let c = alpha.len();
                    let (scale, bias) = match condition {
                        Some(prefix) => {
                            let mut row = |what: &str| -> Result<Vec<f32>, Error> {
                                let name = format!("{prefix}.{what}_embed.weight");
                                let (shape, w) = parameter(&name)?;
                                if shape != [self.tensors.shape.buckets, c] {
                                    return Err(bad_shape(&name, &shape));
                                }
                                Ok(w[bucket * c..(bucket + 1) * c].to_vec())
                            };
                            (row("scale")?, row("bias")?)
                        }
                        None => (vec![1.0; c], vec![0.0; c]),
                    };
                    let values = (0..c).flat_map(|i| {
                        let a = alpha[i];
                        [scale[i], bias[i], a, 1.0 / (a + 1e-9)]
                    });
                    values.collect()
                }
            };
            let tensor = self.tensors.graph.info(t);
            if values.len() != tensor.len() {
                return Err(Error::Checkpoint(format!(
                    "{}: {} values for {}",
                    tensor.name,
                    values.len(),
                    tensor.len()
                )));
            }
            self.step.write(t, &values)?;
        }
        Ok(())
    }

    /// Decodes stream `stream`'s next latent frames, `[frames][latent]` flattened, into their
    /// audio, `hop` samples a frame, as many chunks as it takes.
    pub fn push(&mut self, stream: usize, latents: &[f32]) -> Result<Vec<f32>, Error> {
        let VaeShape {
            latent,
            chunk,
            streams,
            ..
        } = self.tensors.shape;
        if !latents.len().is_multiple_of(latent) || stream >= streams {
            return Err(Error::Shape(format!(
                "{} latent values in frames of {latent} of stream {stream} of {streams}",
                latents.len()
            )));
        }
        let hop = self.tensors.shape.hop();
        let mut audio = Vec::with_capacity(latents.len() / latent * hop);
        let mut padded = vec![0.0; chunk * latent];
        for frames in latents.chunks(chunk * latent) {
            let count = frames.len() / latent;
            padded[..frames.len()].copy_from_slice(frames);
            padded[frames.len()..].fill(0.0);
            self.step.write(self.tensors.latents, &padded)?;
            let before = &mut self.frames_before[stream];
            self.step
                .write(self.tensors.chunk, &[*before, count as u32, stream as u32])?;
            self.step.submit()?;
            let samples: Vec<f32> = self.step.read(self.tensors.audio)?;
            audio.extend_from_slice(&samples[..count * hop]);
            *before = (*before + count as u32) % self.period;
        }
        Ok(audio)
    }

    /// Starts stream `stream` anew: its ring of every convolution back to zeros.
    pub fn reset(&mut self, stream: usize) -> Result<(), Error> {
        let streams = self.tensors.shape.streams;
        if stream >= streams {
            return Err(Error::Shape(format!("stream {stream} of {streams}")));
        }
        let regions: Vec<Region<f32>> = self
            .tensors
            .rings
            .iter()
            .map(|&(ring, _)| {
                let len = self.tensors.graph.info(ring).len() / streams;
                Region {
                    tensor: ring,
                    offset: stream * len,
                    other: 0,
                    len,
                }
            })
            .collect();
        self.step.copy_in(&self.zeros, &regions)?;
        self.frames_before[stream] = 0;
        Ok(())
    }

    /// Each dispatch's label and GPU time in the last chunk, when recorded timed.
    pub fn timings(&self) -> Result<Option<Vec<(&str, Timing)>>, Error> {
        self.step.timings()
    }

    pub fn shape(&self) -> &VaeShape {
        &self.tensors.shape
    }

    /// The recorded step's dispatches, barriers and arena.
    pub fn graph_step(&self) -> &Step {
        &self.step
    }
}

fn bad_shape(name: &str, shape: &[usize]) -> Error {
    Error::Checkpoint(format!("{name}: shape {shape:?}"))
}

fn lcm(a: u32, b: u32) -> u32 {
    let gcd = iter::successors(Some((a, b)), |&(a, b)| (b != 0).then(|| (b, a % b)))
        .last()
        .map_or(1, |(a, _)| a);
    a / gcd * b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{barriers, plan},
        kernels::Access,
    };

    fn shape(chunk: usize, streams: usize) -> VaeShape {
        let config = serde_json::json!({
            "encoder_dim": 128,
            "encoder_rates": [2, 5, 8, 8],
            "latent_dim": 64,
            "decoder_dim": 2048,
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000,
            "out_sample_rate": 48000
        });
        VaeShape::new(&serde_json::from_value(config).unwrap(), chunk, streams).unwrap()
    }

    #[test]
    fn voxcpm2s_decoder_writes_1920_samples_a_frame_in_the_top_bucket() {
        let s = shape(25, 1);
        assert_eq!(s.hop(), 1920);
        assert_eq!((s.bucket, s.buckets), (3, 4));
    }

    #[test]
    fn each_ring_is_kept_after_its_convolution_has_read_it() {
        let g = vae_graph(&shape(25, 1)).unwrap();
        let graph = &g.graph;
        let needs_barrier = barriers(graph, &plan(graph));
        for &(ring, _) in &g.rings {
            let uses: Vec<(usize, Access)> = graph
                .ops
                .iter()
                .enumerate()
                .flat_map(|(i, op)| {
                    let access = op.pipeline.kernel.bindings.iter().map(|b| b.access);
                    let bound = op.bindings.iter().zip(access);
                    bound
                        .filter(|&(&t, _)| t == ring.id())
                        .map(move |(_, a)| (i, a))
                })
                .collect();
            let name = &graph.info(ring).name;
            let [(read, Access::Read), (kept, Access::Write)] = uses[..] else {
                panic!("{name}: {uses:?}");
            };
            assert_eq!(graph.ops[kept].pipeline.kernel.name, "vae_keep", "{name}");
            assert!(
                needs_barrier[read + 1..=kept].contains(&true),
                "{name} is kept while it may be read"
            );
        }
    }

    #[test]
    fn the_rings_carry_what_a_decode_stream_keeps_for_each_stream() {
        for streams in [1, 3] {
            let g = vae_graph(&shape(25, streams)).unwrap();
            // 1 + 6 blocks of a transposed convolution and three residual units + 1.
            assert_eq!(g.rings.len(), 26);
            let values: usize = g.rings.iter().map(|&(t, _)| g.graph.info(t).len()).sum();
            // As the audiovae crate's DecodeStream counted what it kept of a batch item.
            assert_eq!(values, streams * 161_856);
            assert_eq!(g.graph.ops.len(), 3 + 6 * 11 + 2);
            let declared = (5, SpecValue::U32(streams as u32));
            assert!(
                g.graph
                    .ops
                    .iter()
                    .all(|op| op.pipeline.constants.contains(&declared))
            );
        }
    }

    #[test]
    fn every_gemm_covers_its_outputs_once() {
        for chunk in [1, 25] {
            let g = vae_graph(&shape(chunk, 1)).unwrap();
            for op in g
                .graph
                .ops
                .iter()
                .filter(|op| op.pipeline.kernel.name == "vae_gemm")
            {
                let constant = |id: u32| match op.pipeline.constants.iter().find(|c| c.0 == id) {
                    Some(&(_, SpecValue::U32(v))) => v as usize,
                    other => panic!("{}: constant {id} is {other:?}", op.label),
                };
                let (up, n, tm, tn, wn) = (
                    constant(1),
                    constant(7),
                    constant(9),
                    constant(10),
                    constant(11),
                );
                let (bm, bn) = (tm * (WORKGROUP as usize / wn), tn * wn);
                let rows = chunk * up;
                let [x, y, 1] = op.groups.map(|g| g as usize) else {
                    panic!("{}: {:?}", op.label, op.groups);
                };
                assert_eq!(x * bn, n, "{}", op.label);
                assert!(y * bm >= rows && (y - 1) * bm < rows, "{}", op.label);
            }
        }
    }

    #[test]
    fn what_voxcpm2_does_not_use_is_refused() {
        let base = serde_json::json!({
            "latent_dim": 64,
            "decoder_dim": 2048,
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "out_sample_rate": 48000
        });
        let with = |field: &str, value: serde_json::Value| {
            let mut config = base.clone();
            config[field] = value;
            VaeShape::new(&serde_json::from_value(config).unwrap(), 25, 1)
        };
        let refused = [
            ("depthwise", serde_json::json!(false)),
            ("use_noise_block", serde_json::json!(true)),
            ("cond_type", serde_json::json!("concat")),
            ("cond_out_layer", serde_json::json!(true)),
            ("sr_bin_boundaries", serde_json::json!(null)),
            ("sr_bin_boundaries", serde_json::json!([30000, 20000])),
            ("decoder_rates", serde_json::json!([])),
            ("decoder_dim", serde_json::json!(32)),
        ];
        for (field, value) in refused {
            assert!(
                matches!(with(field, value.clone()), Err(Error::Shape(_))),
                "{field}: {value}"
            );
        }
        assert!(with("cond_type", serde_json::json!("scale_bias_init")).is_ok());
        let config = serde_json::from_value(base).unwrap();
        assert!(matches!(
            VaeShape::new(&config, 25, 0),
            Err(Error::Shape(_))
        ));
    }

    #[test]
    fn the_period_is_a_multiple_of_every_ring() {
        assert_eq!(lcm(1, 6), 6);
        assert_eq!([6, 1, 6, 18, 54].into_iter().fold(1, lcm), 54);
    }
}
