//! A prompt run into a slot of the patch step's caches, as voxcpm's `run_prompt` runs it, all but
//! its last position, which the step of the first patch after it runs as the patch before
//! ([`patch`](crate::patch)): a prompt that ends in audio, as the service's do.
//!
//! A position is a text token or a patch of audio latents. A token's embedding is its row of the
//! base LM's embedding table, read from the checkpoint on the host: the table is 73448 rows of
//! 2048 values, 300 MB in f16, of which a prompt reads a few dozen rows, so a gather on the GPU
//! would keep it resident to save a copy of 4 KiB a row. A patch's embedding is the local
//! encoder's and enc_to_lm_proj's, a submission for each distinct patch of the prompt: the
//! service's prompts hold their voice's patches twice.
//!
//! Then a submission for each run of up to `max_rows` positions: the base LM over the
//! embeddings, the scalar quantizer over its outputs at the audio positions, fusion_concat_proj
//! of those beside the audio embeddings, zero at text, and the residual LM over that, both LMs
//! writing their keys and values into the patch step's caches. The layers between the LMs read
//! each weight once for a block of rows (`linear_rows.comp`). Both are programs over the patch
//! step's memory ([`Step::add`](crate::step::Step::add)).
//!
//! A [`PromptCache`] keeps what a prompt in a voice computes that the next prompts in the voice
//! share: its patches' embeddings, and the LMs' keys and values over the positions it begins
//! with, which a prompt that begins with them copies back into the caches rather than run.

use std::{
    collections::{HashMap, hash_map::Entry},
    iter,
};

use vulkan::{Buffer, Context, F16, Residence, SpecValue};

use crate::{
    Error,
    checkpoint::Checkpoint,
    decode::{LayerTensors, Row, Tuning},
    graph::{Graph, Kind, Pipeline, Tensor, TensorId},
    kernels::{KernelInfo, Kernels, read, write},
    lm_side::{Act, Linear, Stack, append_linear, splice},
    patch::{PatchGraph, PatchStep, transposed},
    prefill::{StackTensors, TILE, append_prefill},
    step::{Program, Region},
};

/// `linear_rows.comp`: run, x0, x1, W, bias -> out.
pub static LINEAR_ROWS: KernelInfo = KernelInfo {
    name: "linear_rows",
    bindings: &[
        read::<u32>(),
        read::<F16>(),
        read::<F16>(),
        read::<F16>(),
        read::<F16>(),
        write::<F16>(),
    ],
};

/// `select_rows.comp`: run, flags, a, b -> out.
pub static SELECT_ROWS: KernelInfo = KernelInfo {
    name: "select_rows",
    bindings: &[
        read::<u32>(),
        read::<u32>(),
        read::<F16>(),
        read::<F16>(),
        write::<F16>(),
    ],
};

/// The width of the rows kernels' workgroups.
const WORKGROUP: u32 = 256;

/// The rows a workgroup of linear_rows takes, and the columns of them it stages at a time.
const BLOCK: usize = 32;
const STAGED: usize = 512;

/// The lanes linear_rows sums a row of W with, fixed in gemm.glsl.
const CLUSTER: usize = 16;

/// The base LM's embedding table in the checkpoint.
const EMBED_TOKENS: &str = "base_lm.embed_tokens.weight";

/// The programs a prompt runs over the patch step's memory, all over the same tensors.
#[derive(Clone, Debug)]
pub struct PromptGraphs {
    /// Embeds a patch.
    pub embed: Graph,
    /// `programs[i]` runs `(i + 1) TILE` positions.
    pub programs: Vec<Graph>,
    pub tensors: PromptTensors,
}

/// The tensors of a prompt's programs that the host writes and reads.
#[derive(Clone, Copy, Debug)]
pub struct PromptTensors {
    pub max_rows: usize,
    /// The patch to embed, laid out as the patch step's `cond`, and its embedding `[1][hidden]`.
    pub patch: Tensor<f32>,
    pub embedding: Tensor<F16>,
    /// The first position and the length of the run, and the slot of the caches it goes to.
    pub run: Tensor<u32>,
    /// The positions' embeddings, `[max_rows][hidden]`.
    pub x: Tensor<F16>,
    /// The audio positions' embeddings, laid out as `x`, zero at text.
    pub audio_x: Tensor<F16>,
    /// Whether each position is audio, one to a position.
    pub audio: Tensor<u32>,
}

fn u(v: usize) -> SpecValue {
    SpecValue::U32(v as u32)
}

/// The stack `stack` of the LM side, where `ids` puts its tensors in the patch step's graph.
fn stack_layers(stack: &Stack, ids: &[TensorId]) -> (Vec<LayerTensors>, Tensor<F16>) {
    (
        stack.layers.iter().map(|l| l.moved(ids)).collect(),
        stack.norm.moved(ids),
    )
}

/// The prompt's programs over `patch`'s memory, for runs of up to `max_rows` positions, their
/// kernels at `tuning`'s subgroup size.
pub fn prompt_graphs(
    patch: &PatchGraph,
    max_rows: usize,
    tuning: &Tuning,
) -> Result<PromptGraphs, Error> {
    let side = &patch.side;
    let s = &side.shape;
    let hidden = s.base.hidden;
    let layer = |l: Linear| l.moved(&patch.side_ids);
    let mut g = patch.graph.clone();
    g.ops.clear();

    let encoder = &patch.encoder;
    let encoders = |t: TensorId| encoder.graph.info(t).shape.as_slice();
    let input = g.tensor::<f32>("embed.patch", encoders(encoder.patch.id()), Kind::Input);
    let enc = g.tensor::<F16>("embed.enc", encoders(encoder.out.id()), Kind::Intermediate);
    let embedding = g.tensor("embed.out", &[1, hidden], Kind::Output);
    let weights = encoder.graph.tensors.iter().enumerate();
    let shared: Vec<(TensorId, TensorId)> = weights
        .filter(|(_, t)| t.kind == Kind::Constant)
        .map(|(i, _)| (TensorId(i), patch.encoder_ids[i]))
        .chain([
            (encoder.patch.id(), input.id()),
            (encoder.out.id(), enc.id()),
        ])
        .collect();
    splice(&mut g, &encoder.graph, "embed.", &shared);
    append_linear(
        &mut g,
        "embed.enc_to_lm_proj",
        layer(side.heads.enc_to_lm_proj),
        &[enc],
        Act::None,
        (embedding, 0),
        tuning,
        s.levels,
    )?;
    let embed_ops = g.ops.len();

    let run = g.tensor("prompt.run", &[3], Kind::Param);
    let x = g.tensor("prompt.x", &[max_rows, hidden], Kind::Input);
    let audio_x = g.tensor("prompt.audio_x", &[max_rows, hidden], Kind::Input);
    let audio = g.tensor("prompt.audio", &[max_rows], Kind::Input);
    let mut between = |name: &str, n: usize| g.tensor(name, &[max_rows, n], Kind::Intermediate);
    let base_out = between("prompt.base_lm.out", hidden);
    let levels = between("prompt.fsq_layer.levels", s.latent);
    let quantized = between("prompt.fsq_layer.out", hidden);
    let lm_hidden = between("prompt.lm_hidden", hidden);
    let residual_in = between("prompt.residual_lm.in", hidden);
    let residual_out = between("prompt.residual_lm.out", hidden);
    let (base_layers, base_norm) = stack_layers(&side.base, &patch.side_ids);
    let (residual_layers, residual_norm) = stack_layers(&side.residual, &patch.side_ids);
    let stack = |shape, layers, norm| StackTensors {
        shape,
        slots: side.batch.slots,
        layers,
        norm,
        cos: side.cos.moved(&patch.side_ids),
        sin: side.sin.moved(&patch.side_ids),
    };
    let base = stack(&s.base, &base_layers[..], base_norm);
    let residual = stack(&s.residual, &residual_layers[..], residual_norm);

    let subgroup = tuning.subgroup;
    let pipeline = |kernel, constants: &[(u32, SpecValue)]| Pipeline {
        kernel,
        constants: [
            (0, SpecValue::U32(subgroup)),
            (1, SpecValue::U32(WORKGROUP)),
        ]
        .iter()
        .chain(constants)
        .copied()
        .collect(),
        subgroup_size: Some(subgroup),
    };
    let check = |ok: bool, why: String| if ok { Ok(()) } else { Err(Error::Shape(why)) };
    // `layer` over each row of the inputs `x` joined, into `out`.
    let rows_linear = |g: &mut Graph,
                       label: &str,
                       layer: Linear,
                       x: &[Tensor<F16>],
                       act: Act,
                       out: Tensor<F16>,
                       rows: usize|
     -> Result<(), Error> {
        let width = |t: Tensor<F16>| g.info(t).shape[1];
        let (k0, k1) = (width(x[0]), x.get(1).map_or(0, |&t| width(t)));
        let n = width(out);
        let k = k0 + k1;
        let staged = STAGED.min(k);
        check(
            k0.is_multiple_of(8)
                && k.is_multiple_of(staged)
                && staged.is_multiple_of(8 * CLUSTER)
                && n.is_multiple_of(WORKGROUP as usize / CLUSTER)
                && rows.is_multiple_of(BLOCK),
            format!("{label}: {k0} and {k1} columns into {n} over {rows} rows"),
        )?;
        let pipeline = pipeline(
            &LINEAR_ROWS,
            &[
                (2, u(k0)),
                (3, u(k1)),
                (4, u(staged)),
                (5, u(BLOCK)),
                (6, u(n)),
                (7, SpecValue::U32(act as u32)),
                (8, SpecValue::F32(s.levels as f32)),
            ],
        );
        let groups = [
            (n / (WORKGROUP as usize / CLUSTER)) as u32,
            (rows / BLOCK) as u32,
            1,
        ];
        let bindings = (run, x[0], x[x.len() - 1], layer.weight, layer.bias, out);
        g.op(label, pipeline, bindings, groups);
        Ok(())
    };
    let select = pipeline(&SELECT_ROWS, &[(2, u(hidden))]);

    let mut programs = Vec::with_capacity(max_rows / TILE);
    for tiles in 1..=max_rows / TILE {
        let rows = tiles * TILE;
        let mut p = g.clone();
        let io = |x, out| (run, x, out);
        append_prefill(
            &mut p,
            "prompt.base_lm",
            base,
            rows,
            max_rows,
            tuning,
            io(x, base_out),
        )?;
        let h = &side.heads;
        rows_linear(
            &mut p,
            "fsq_layer.in_proj",
            layer(h.fsq_in_proj),
            &[base_out],
            Act::Levels,
            levels,
            rows,
        )?;
        rows_linear(
            &mut p,
            "fsq_layer.out_proj",
            layer(h.fsq_out_proj),
            &[levels],
            Act::None,
            quantized,
            rows,
        )?;
        p.op(
            "select_rows",
            select.clone(),
            (run, audio, quantized, base_out, lm_hidden),
            [rows as u32, 1, 1],
        );
        rows_linear(
            &mut p,
            "fusion_concat_proj",
            layer(h.fusion_concat_proj),
            &[lm_hidden, audio_x],
            Act::None,
            residual_in,
            rows,
        )?;
        append_prefill(
            &mut p,
            "prompt.residual_lm",
            residual,
            rows,
            max_rows,
            tuning,
            io(residual_in, residual_out),
        )?;
        p.ops.drain(..embed_ops);
        programs.push(p);
    }
    let Some(first) = programs.first() else {
        return Err(Error::Shape(format!("programs of up to {max_rows} rows")));
    };
    let mut embed = first.clone();
    embed.ops = g.ops;
    Ok(PromptGraphs {
        embed,
        programs,
        tensors: PromptTensors {
            max_rows,
            patch: input,
            embedding,
            run,
            x,
            audio_x,
            audio,
        },
    })
}

/// A prompt's positions, laid out as voxcpm's `Prompt` lays them out.
#[derive(Clone, Copy, Debug)]
pub struct Prompt<'a> {
    /// Each position's token, read where it is text.
    pub tokens: &'a [u32],
    /// Each position's patch, `[patch_size][feat_dim]`, read where it is audio.
    pub patches: &'a [f32],
    /// Whether each position is audio.
    pub audio: &'a [bool],
}

impl Prompt<'_> {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// What a generator keeps of the voices for the next prompts in them, up to a number of bytes:
/// for each voice, the embeddings of its prompts'
/// patches, on the host, which the service's prompts hold twice; and on the GPU, the LMs' keys
/// and values over the first prompt in it up to its last run of audio, the voice as reference
/// audio and the text after it. The LMs are causal, so a prompt that begins with the same
/// positions has the same keys and values over them, and runs only the positions after those.
/// A voice is told by the bits of its latent frames, and the voices used longest ago make room
/// for others.
#[derive(Debug)]
pub struct PromptCache {
    ctx: Context,
    budget: usize,
    bytes: usize,
    clock: u64,
    voices: HashMap<Vec<u32>, Voice>,
}

#[derive(Debug, Default)]
struct Voice {
    /// The embedding of each patch as the base LM takes it, `[hidden]`, by the patch's bits.
    embeddings: HashMap<Vec<u32>, Vec<F16>>,
    prefix: Option<Prefix>,
    /// When it was last used, on the cache's clock.
    used: u64,
}

/// The positions a prompt began with, and the LMs' keys and values over them.
#[derive(Debug)]
struct Prefix {
    tokens: Vec<u32>,
    /// Each position's patch's bits, `[len][patch]`.
    patches: Vec<u32>,
    audio: Vec<bool>,
    /// Laid out as [`cache_regions`] lays out as many positions as it holds.
    kept: Buffer<F16>,
}

impl Prefix {
    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn bytes(&self) -> usize {
        Self::bytes_of(
            self.kept.size(),
            self.patches.len() / self.len(),
            self.len(),
        )
    }

    /// The bytes of a prefix of `len` positions of patches of `patch` values, whose keys and
    /// values take `kept`.
    fn bytes_of(kept: u64, patch: usize, len: usize) -> usize {
        kept as usize + (4 + 1 + 4 * patch) * len
    }

    /// The positions `prompt` begins with that this holds.
    fn shared(&self, prompt: Prompt<'_>) -> usize {
        let patch = self.patches.len() / self.len();
        (0..self.len().min(prompt.len()))
            .take_while(|&p| {
                let at = p * patch..(p + 1) * patch;
                let bits = prompt.patches[at.clone()].iter().map(|v| v.to_bits());
                self.audio[p] == prompt.audio[p]
                    && match prompt.audio[p] {
                        true => bits.eq(self.patches[at].iter().copied()),
                        false => self.tokens[p] == prompt.tokens[p],
                    }
            })
            .count()
    }
}

impl Voice {
    fn bytes(&self) -> usize {
        let embeddings = self.embeddings.iter();
        let embeddings: usize = embeddings.map(|(k, v)| 4 * k.len() + 2 * v.len()).sum();
        embeddings + self.prefix.as_ref().map_or(0, Prefix::bytes)
    }
}

impl PromptCache {
    /// An empty cache on `ctx`'s GPU that keeps at most `budget` bytes, on the host and the GPU
    /// together.
    pub fn new(ctx: &Context, budget: usize) -> Self {
        PromptCache {
            ctx: ctx.clone(),
            budget,
            bytes: 0,
            clock: 0,
            voices: HashMap::new(),
        }
    }

    /// The bytes it keeps.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The number of voices it keeps.
    pub fn voices(&self) -> usize {
        self.voices.len()
    }

    /// The voice `key` with what is kept of it, which is no longer kept until it is
    /// [`put`](Self::put) back.
    fn take(&mut self, key: &[u32]) -> Voice {
        self.clock += 1;
        let Some(voice) = self.voices.remove(key) else {
            return Voice::default();
        };
        self.bytes -= 4 * key.len() + voice.bytes();
        voice
    }

    /// Keeps `voice` as `key`, without its prefix if only that way it fits, and not at all if
    /// it does not fit alone; the voices used longest ago are dropped to make room.
    fn put(&mut self, key: Vec<u32>, mut voice: Voice) {
        let bytes = |voice: &Voice| 4 * key.len() + voice.bytes();
        if bytes(&voice) > self.budget {
            voice.prefix = None;
        }
        if bytes(&voice) > self.budget {
            return;
        }
        while self.bytes + bytes(&voice) > self.budget {
            let (oldest, _) = self
                .voices
                .iter()
                .min_by_key(|(_, voice)| voice.used)
                .expect("the bytes kept are some voice's");
            let oldest = oldest.clone();
            let dropped = self.voices.remove(&oldest).expect("found");
            self.bytes -= 4 * oldest.len() + dropped.bytes();
        }
        voice.used = self.clock;
        self.bytes += bytes(&voice);
        self.voices.insert(key, voice);
    }
}

/// Where the LMs' keys and values over positions `0..len` lie in slot `slot` of `graph`'s caches,
/// laid out as `rope_rows.comp` writes them, `[slots][kv_heads][capacity][head_dim]` in f16 for
/// each layer's keys and values: for each in turn, each head's `len` positions, from `held`
/// positions apart in the buffer outside.
fn cache_regions(graph: &PatchGraph, slot: usize, len: usize, held: usize) -> Vec<Region<F16>> {
    let side = &graph.side;
    let stacks = [
        (&side.base, &side.shape.base),
        (&side.residual, &side.shape.residual),
    ];
    let mut regions = Vec::new();
    let mut other = 0;
    for (stack, shape) in stacks {
        let (row, heads) = (shape.head_dim, shape.kv_heads);
        for layer in &stack.layers {
            for cache in [layer.k_cache, layer.v_cache] {
                for head in slot * heads..(slot + 1) * heads {
                    regions.push(Region {
                        tensor: cache.moved(&graph.side_ids),
                        offset: head * shape.capacity * row,
                        other,
                        len: len * row,
                    });
                    other += held * row;
                }
            }
        }
    }
    regions
}

/// The prompt's programs on the GPU, recorded over a patch step's memory.
#[derive(Debug)]
pub struct Prompter {
    embed: Program,
    /// `programs[i]` runs `(i + 1) TILE` positions.
    programs: Vec<Program>,
    tensors: PromptTensors,
    hidden: usize,
    patch_size: usize,
}

impl Prompter {
    /// Records the prompt's programs over `step`'s memory, for runs of up to `max_rows`
    /// positions at once, their kernels at `tuning`'s subgroup size; timed per dispatch when
    /// `timed`.
    pub fn new(
        ctx: &Context,
        kernels: &mut Kernels,
        step: &mut PatchStep,
        max_rows: usize,
        tuning: &Tuning,
        timed: bool,
    ) -> Result<Self, Error> {
        let (graph, recorded) = step.parts_mut();
        let graphs = prompt_graphs(graph, max_rows, tuning)?;
        let hidden = graph.side.shape.base.hidden;
        let patch_size = graph.sampler.patch_size;
        let all: Vec<&Graph> = iter::once(&graphs.embed).chain(&graphs.programs).collect();
        let mut programs = recorded.add(ctx, kernels, &all, timed)?;
        let embed = programs.remove(0);
        Ok(Prompter {
            embed,
            programs,
            tensors: graphs.tensors,
            hidden,
            patch_size,
        })
    }

    /// The most positions one submission runs.
    pub fn max_rows(&self) -> usize {
        self.tensors.max_rows
    }

    /// The embedding of `patch`, `[patch_size][feat_dim]`, as the base LM takes it.
    pub fn embed(&self, step: &mut PatchStep, patch: &[f32]) -> Result<Vec<F16>, Error> {
        let (_, recorded) = step.parts_mut();
        let cond = transposed(patch, self.patch_size);
        recorded.write(self.tensors.patch, &cond)?;
        recorded.run(self.embed)?;
        recorded.read(self.tensors.embedding)
    }

    /// Runs `prompt` but its last position into slot `slot` of `step`'s caches from position 0,
    /// the tokens' embeddings read from `checkpoint`. The step of the first patch after it runs
    /// the last, whose patch it takes as the patch before.
    pub fn run(
        &self,
        step: &mut PatchStep,
        slot: usize,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
    ) -> Result<(), Error> {
        let (x, audio_x) = self.embeddings(step, checkpoint, prompt)?;
        let at = Row { position: 0, slot };
        self.run_rows(step, at, &x, &audio_x, &prompt.audio[..prompt.len() - 1])
    }

    /// Runs `prompt` as [`run`](Self::run) does, in the voice whose latent frames are `voice`:
    /// from the positions it begins with that `cache` keeps of the voice, and with the patches'
    /// embeddings it keeps; and keeps there what the next prompts in the voice can use. Gives
    /// the positions that run through the LMs, the prompt's last among them.
    pub fn run_cached(
        &self,
        step: &mut PatchStep,
        slot: usize,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
        voice: &[f32],
        cache: &mut PromptCache,
    ) -> Result<usize, Error> {
        check(prompt)?;
        let len = prompt.len();
        let key: Vec<u32> = voice.iter().map(|v| v.to_bits()).collect();
        let mut kept = cache.take(&key);
        let prefix = kept.prefix.as_ref();
        let shared = prefix
            .map_or(0, |prefix| prefix.shared(prompt))
            .min(len - 1);
        let (x, audio_x) =
            self.embeddings_from(step, checkpoint, prompt, shared, &mut kept.embeddings)?;
        if slot >= step.slots() {
            return Err(Error::Shape(format!("slot {slot} of {}", step.slots())));
        }
        if let Some(prefix) = kept.prefix.as_ref().filter(|_| shared > 0) {
            let (graph, recorded) = step.parts_mut();
            let regions = cache_regions(graph, slot, shared, prefix.len());
            recorded.copy_in(&prefix.kept, &regions)?;
        }
        let at = Row {
            position: shared,
            slot,
        };
        self.run_rows(step, at, &x, &audio_x, &prompt.audio[shared..len - 1])?;
        let held = prompt
            .audio
            .iter()
            .rposition(|&a| !a)
            .map_or(0, |text| text + 1);
        if kept.prefix.is_none() && held > 0 {
            let (graph, recorded) = step.parts_mut();
            let regions = cache_regions(graph, slot, held, held);
            let values: usize = regions.iter().map(|r| r.len).sum();
            let patch = prompt.patches.len() / len;
            let bytes = (values * size_of::<F16>()) as u64;
            let prefix = Prefix::bytes_of(bytes, patch, held);
            if 4 * key.len() + kept.bytes() + prefix <= cache.budget {
                let mut buffer = cache.ctx.buffer_in(values, Residence::Device)?;
                recorded.copy_out(&regions, &mut buffer)?;
                let patches = &prompt.patches[..held * patch];
                kept.prefix = Some(Prefix {
                    tokens: prompt.tokens[..held].to_vec(),
                    patches: patches.iter().map(|v| v.to_bits()).collect(),
                    audio: prompt.audio[..held].to_vec(),
                    kept: buffer,
                });
            }
        }
        cache.put(key, kept);
        Ok(len - shared)
    }

    /// The embeddings of `prompt`'s positions but its last, `[len - 1][hidden]`, and those of its
    /// audio positions, zero at text: a token's read from `checkpoint`, a patch's from the GPU,
    /// once for each distinct patch.
    pub fn embeddings(
        &self,
        step: &mut PatchStep,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
    ) -> Result<(Vec<F16>, Vec<F16>), Error> {
        check(prompt)?;
        self.embeddings_from(step, checkpoint, prompt, 0, &mut HashMap::new())
    }

    /// The embeddings of `prompt`'s positions from `from` but its last, as
    /// [`embeddings`](Self::embeddings) gives those from the first: a patch's from `known`
    /// where it holds it, and else from the GPU, and kept there.
    fn embeddings_from(
        &self,
        step: &mut PatchStep,
        checkpoint: &Checkpoint,
        prompt: Prompt<'_>,
        from: usize,
        known: &mut HashMap<Vec<u32>, Vec<F16>>,
    ) -> Result<(Vec<F16>, Vec<F16>), Error> {
        let (len, hidden) = (prompt.len(), self.hidden);
        let patch = prompt.patches.len() / len;
        let ran = from..len - 1;
        let mut x = vec![F16::ZERO; ran.len() * hidden];
        let mut audio_x = vec![F16::ZERO; ran.len() * hidden];
        let row = |p: usize| (p - from) * hidden..(p - from + 1) * hidden;
        let text: Vec<usize> = ran.clone().filter(|&p| !prompt.audio[p]).collect();
        let tokens: Vec<usize> = text.iter().map(|&p| prompt.tokens[p] as usize).collect();
        let rows = checkpoint.gather_f16(EMBED_TOKENS, &tokens)?;
        for (&p, embedding) in text.iter().zip(rows.chunks(hidden)) {
            x[row(p)].copy_from_slice(embedding);
        }
        for p in ran.filter(|&p| prompt.audio[p]) {
            let values = &prompt.patches[p * patch..(p + 1) * patch];
            let key: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let embedding = match known.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(self.embed(step, values)?),
            };
            x[row(p)].copy_from_slice(embedding);
            audio_x[row(p)].copy_from_slice(embedding);
        }
        Ok((x, audio_x))
    }

    /// Runs the positions from `at` on whose embeddings are `x`, `[len][hidden]`, those of audio
    /// as `audio` says, through both LMs into `step`'s caches: `audio_x` holds the embeddings of
    /// the audio positions, zero at text. A run of more than [`max_rows`](Self::max_rows)
    /// positions runs in pieces.
    pub fn run_rows(
        &self,
        step: &mut PatchStep,
        at: Row,
        x: &[F16],
        audio_x: &[F16],
        audio: &[bool],
    ) -> Result<(), Error> {
        let (hidden, max_rows) = (self.hidden, self.tensors.max_rows);
        let len = audio.len();
        let Row {
            position: start,
            slot,
        } = at;
        if x.len() != len * hidden
            || audio_x.len() != x.len()
            || start + len > step.capacity()
            || slot >= step.slots()
        {
            return Err(Error::Shape(format!(
                "{} and {} values for {len} positions from {start} in slot {slot} of {} of {}",
                x.len(),
                audio_x.len(),
                step.slots(),
                step.capacity()
            )));
        }
        let (_, recorded) = step.parts_mut();
        let t = &self.tensors;
        let mut padded = vec![F16::ZERO; max_rows * hidden];
        let mut padded_audio = vec![F16::ZERO; max_rows * hidden];
        let mut flags = vec![0u32; max_rows];
        for first in (0..len).step_by(max_rows) {
            let rows = (len - first).min(max_rows);
            let piece = first * hidden..(first + rows) * hidden;
            padded[..rows * hidden].copy_from_slice(&x[piece.clone()]);
            padded_audio[..rows * hidden].copy_from_slice(&audio_x[piece]);
            let audio = &audio[first..first + rows];
            for (flag, &a) in flags.iter_mut().zip(audio) {
                *flag = u32::from(a);
            }
            recorded.write(t.run, &[(start + first) as u32, rows as u32, slot as u32])?;
            recorded.write(t.x, &padded)?;
            recorded.write(t.audio_x, &padded_audio)?;
            recorded.write(t.audio, &flags)?;
            recorded.run(self.programs[rows.div_ceil(TILE) - 1])?;
        }
        Ok(())
    }
}

/// Refuses a prompt the prompter cannot run: one that does not end in audio, or whose patches
/// and flags are not one for each position.
fn check(prompt: Prompt<'_>) -> Result<(), Error> {
    let len = prompt.len();
    let patch = prompt.patches.len() / len.max(1);
    if len == 0
        || prompt.audio.len() != len
        || prompt.patches.len() != len * patch
        || !prompt.audio[len - 1]
    {
        return Err(Error::Shape(format!(
            "a prompt of {len} tokens, {} values of patches and {} audio flags, which must end \
             in audio",
            prompt.patches.len(),
            prompt.audio.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{kernels::Access, patch};

    fn constant(op: &crate::graph::Op, id: u32) -> usize {
        match op.pipeline.constants.iter().find(|c| c.0 == id) {
            Some(&(_, SpecValue::U32(v))) => v as usize,
            other => panic!("{}: constant {id} is {other:?}", op.label),
        }
    }

    #[test]
    fn the_programs_run_over_the_patch_steps_weights_and_caches_and_tensors_of_their_own() {
        for subgroup in crate::decode::SUBGROUPS {
            let patch = patch::tests::graph(subgroup);
            let p = prompt_graphs(&patch, 2 * TILE, &Tuning::for_subgroup(subgroup)).unwrap();
            let shape = |g: &Graph| -> Vec<(u64, Kind)> {
                g.tensors.iter().map(|t| (t.bytes(), t.kind)).collect()
            };
            assert_eq!(p.programs.len(), 2);
            assert!(p.programs.iter().all(|g| shape(g) == shape(&p.embed)));
            assert_eq!(p.embed.ops.len(), patch.encoder.graph.ops.len() + 1);
            let held = &patch.graph.tensors;
            for g in iter::once(&p.embed).chain(&p.programs) {
                for op in &g.ops {
                    let theirs = op.bindings.iter().filter(|t| t.0 < held.len());
                    let arena = theirs.filter(|t| held[t.0].kind.in_arena());
                    assert_eq!(
                        arena.count(),
                        0,
                        "{} binds the patch step's arena",
                        op.label
                    );
                }
            }
            let caches = |stack: &Stack| -> Vec<TensorId> {
                let at = |l: &LayerTensors| l.k_cache.moved(&patch.side_ids).id();
                stack.layers.iter().map(at).collect()
            };
            let mut keys = caches(&patch.side.base);
            keys.extend(caches(&patch.side.residual));
            for (tiles, g) in p.programs.iter().enumerate() {
                let rows = (tiles + 1) * TILE;
                let written: Vec<TensorId> = g
                    .ops
                    .iter()
                    .flat_map(|op| op.bindings.iter().zip(op.pipeline.kernel.bindings))
                    .filter(|&(t, b)| b.access == Access::Write && keys.contains(t))
                    .map(|(&t, _)| t)
                    .collect();
                assert_eq!(written, keys, "{rows} rows");
                let linear = g
                    .ops
                    .iter()
                    .filter(|op| op.pipeline.kernel.name == "linear_rows");
                let mut layers = 0;
                for op in linear {
                    let n = constant(op, 6);
                    assert_eq!(op.groups[0] as usize * WORKGROUP as usize / CLUSTER, n);
                    assert_eq!(op.groups[1] as usize * BLOCK, rows, "{}", op.label);
                    layers += 1;
                }
                assert_eq!(layers, 3);
            }
        }
    }

    #[test]
    fn the_keys_and_values_kept_are_each_caches_first_positions_in_the_slot_held_apart() {
        let (slots, slot) = (4, 2);
        let patch = patch::tests::batch_graph(64, 1, slots);
        let (len, held) = (5, 7);
        let regions = cache_regions(&patch, slot, len, held);
        let side = &patch.side;
        let caches: Vec<Tensor<F16>> = [&side.base, &side.residual]
            .into_iter()
            .flat_map(|stack| &stack.layers)
            .flat_map(|l| [l.k_cache, l.v_cache])
            .map(|t| t.moved(&patch.side_ids))
            .collect();
        let s = &side.shape.base;
        let row = s.head_dim;
        assert_eq!(regions.len(), caches.len() * s.kv_heads);
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(r.tensor, caches[i / s.kv_heads]);
            let head = patch.graph.info(r.tensor).len() / (slots * s.kv_heads);
            assert_eq!(r.offset, (slot * s.kv_heads + i % s.kv_heads) * head);
            assert_eq!(r.len, len * row);
            assert_eq!(r.other, i * held * row);
        }
    }
}
