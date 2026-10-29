//! The prefill on the GPU, on VoxCPM2's real stacks over a real prompt, held to its golden: the
//! prompt run at once, then decode steps from where it left the cache. The keys and values it
//! leaves in the cache are held, and the outputs over the prompt and the steps. Built with the
//! `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint, `GOLDEN_DIR` the goldens
//! and `GLSL_KERNELS` and `VULKAN_KERNELS` the compiled kernels, as `nix run .#device-tests` sets
//! the last two. Each tolerance is set as testkit's `golden` says.
#![cfg(feature = "device")]

use std::{collections::BTreeMap, fs, time::Instant};

use graph::{
    Error,
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, SUBGROUPS, Shape, Tuning},
    kernels::Kernels,
    prefill::{Prefill, TILE},
};
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use testkit::{Fixture, checkpoint_dir, golden, golden_input};
use vulkan::Context;

fn context() -> Context {
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    if !ctx.validating() {
        eprintln!("the validation layer is off for this run; `nix run .#device-tests` turns it on");
    }
    ctx
}

fn assert_no_validation_messages(ctx: &Context) {
    let messages = ctx.validation_messages();
    assert!(
        messages.is_empty(),
        "the validation layer reported:\n{}",
        messages.join("\n")
    );
}

/// Token ids upstream's tokenizer gave for the game's lines, in order, the first `len` of them.
fn prompt_ids(len: usize) -> Vec<i64> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_tokenizer_checkpoint.safetensors"
    );
    let fixture = Fixture::open(path);
    let mut ids = Vec::with_capacity(len);
    for i in 0.. {
        ids.extend(fixture.ints(&format!("messages.{i}")).1);
        if ids.len() >= len {
            break;
        }
    }
    ids.truncate(len);
    ids
}

/// How far one run's outputs are from another's.
#[derive(Clone, Copy)]
struct Distance {
    max_abs: f64,
    /// RMS of the differences over RMS of the reference.
    relative_rms: f64,
}

impl Distance {
    fn within(self, tolerance: Distance) -> bool {
        self.max_abs <= tolerance.max_abs && self.relative_rms <= tolerance.relative_rms
    }
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |d| {:.2e}, relative RMS {:.2e}",
            self.max_abs, self.relative_rms
        )
    }
}

fn distance(got: &[f32], want: &[f32]) -> Distance {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut err2, mut ref2) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        let d = (g - w).abs();
        max_abs = max_abs.max(if d.is_finite() { d } else { f64::INFINITY });
        err2 += d * d;
        ref2 += w * w;
    }
    Distance {
        max_abs,
        relative_rms: (err2 / ref2).sqrt(),
    }
}

fn config() -> Value {
    let dir = checkpoint_dir();
    serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn f16s(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
}

/// The decode steps run after the prompt: several, since at any one position which of two runs
/// rounds worse is luck.
const STEPS: usize = 8;

/// What a run of a stack gives over a prompt and the decode steps after it: the outputs after the
/// final norm, `[len + STEPS][hidden]`, and every layer's keys then values over the prompt,
/// `[layers][2][kv_heads][len][head_dim]`.
#[derive(Serialize, Deserialize)]
struct Run {
    outputs: Vec<f32>,
    cache: Vec<f32>,
}

/// One of VoxCPM2's causal stacks: its config as the decode step reads it, and the prefix of its
/// weights' names.
struct Stack {
    gpu: LmConfig,
    prefix: &'static str,
}

/// The base LM's first `layers` layers.
fn base_lm(layers: usize) -> Stack {
    let mut gpu: LmConfig = serde_json::from_value(config()["lm_config"].clone()).unwrap();
    gpu.num_hidden_layers = layers;
    Stack {
        gpu,
        prefix: "base_lm.",
    }
}

/// The residual LM, as `config.json` derives it from the base LM.
fn residual_lm() -> Stack {
    let json = config();
    let layers = json["residual_lm_num_layers"].as_u64().unwrap() as usize;
    let no_rope = json["residual_lm_no_rope"].as_bool().unwrap();
    let base = base_lm(layers);
    Stack {
        gpu: LmConfig {
            no_rope,
            ..base.gpu
        },
        prefix: "residual_lm.",
    }
}

/// The base LM's embeddings of `len` tokens of the game's lines, rounded to f16, `[len][hidden]`
/// as f16 bits.
fn embedded(len: usize) -> Vec<u16> {
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    let ids: Vec<usize> = prompt_ids(len).iter().map(|&id| id as usize).collect();
    checkpoint
        .gather_f16("base_lm.embed_tokens.weight", &ids)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The golden of `stack` over the hidden states `x`, the prompt `runs` and `max_rows` as the
/// test runs them: its run at the device's default tuning.
fn golden_of(stack: &Stack, x: &[u16], runs: &[usize], max_rows: usize) -> Run {
    let len: usize = runs.iter().sum();
    let name = format!(
        "prefill-{}{}-{len}",
        stack.prefix, stack.gpu.num_hidden_layers
    );
    golden(&name, || gpu(stack, x, runs, 0, None, max_rows))
}

/// The GPU's run over `stack` and the hidden states `x`: the prompt in pieces of `runs`
/// positions, each a prefill from where the last left off, all `shift` positions on in the cache,
/// then the decode steps. The prefill's kernels at `subgroup` (the device's default when `None`),
/// in programs of up to `max_rows` rows.
fn gpu(
    stack: &Stack,
    x: &[u16],
    runs: &[usize],
    shift: usize,
    subgroup: Option<u32>,
    max_rows: usize,
) -> Run {
    let config = &stack.gpu;
    let (hidden, layers) = (config.hidden_size, config.num_hidden_layers);
    let len: usize = runs.iter().sum();
    assert_eq!(x.len(), (len + STEPS) * hidden);
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let capacity = (len + STEPS + shift).next_multiple_of(32);
    let shape = Shape::new(config, layers, capacity).unwrap();
    let start = Instant::now();
    let tuning = match subgroup {
        Some(s) => Tuning::for_subgroup(s),
        None => Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}")),
    };
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, stack.prefix)
        .unwrap_or_else(|e| panic!("{e}"));
    let prefill = Prefill::new(&ctx, &mut kernels, &mut decoder, max_rows, &tuning, true)
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{}{layers} layers recorded and loaded on the GPU, the prefill's kernels at {tuning:?}: \
         {:.1} s, {} pipelines",
        stack.prefix,
        start.elapsed().as_secs_f64(),
        kernels.len(),
    );

    let mut outputs = Vec::with_capacity((len + STEPS) * hidden);
    let mut done = 0;
    for &run in runs {
        let start = Instant::now();
        let out = prefill
            .run(
                &mut decoder,
                shift + done,
                &x[done * hidden..(done + run) * hidden],
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let host = start.elapsed().as_secs_f64() * 1e3;
        let timings = decoder.timings().unwrap().expect("a timed prefill");
        let mut by_kernel = BTreeMap::<&str, f64>::new();
        for (label, t) in &timings {
            *by_kernel.entry(label).or_default() += t.nanoseconds / 1e6;
        }
        let longest = timings.iter().map(|(_, t)| t.nanoseconds / 1e6);
        eprintln!(
            "positions {} to {} prefilled: {host:.1} ms on the host, {:.1} ms on the GPU, the \
             longest dispatch {:.2} ms; by kernel {by_kernel:.2?} (smoke, not a measurement)",
            shift + done,
            shift + done + run - 1,
            by_kernel.values().sum::<f64>(),
            longest.fold(0.0, f64::max),
        );
        outputs.extend(f16s(&out));
        done += run;
    }
    let s = decoder.shape().clone();
    let (tensors, step) = decoder.parts_mut();
    let mut cache = Vec::with_capacity(layers * 2 * s.kv_heads * len * s.head_dim);
    for layer in &tensors.layers {
        for t in [layer.k_cache, layer.v_cache] {
            let held: Vec<u16> = step.read(t).unwrap_or_else(|e| panic!("{e}"));
            for head in held.chunks(s.capacity * s.head_dim) {
                cache.extend(f16s(&head[shift * s.head_dim..(shift + len) * s.head_dim]));
            }
        }
    }
    for (t, x) in x.chunks(hidden).enumerate().skip(len) {
        let out = decoder.step(shift + t, x).unwrap_or_else(|e| panic!("{e}"));
        outputs.extend(f16s(&out));
    }
    assert_no_validation_messages(&ctx);
    Run { outputs, cache }
}

/// A stack's tolerances: of the outputs, over the prompt and the steps after it, and of the keys
/// and values the prompt left in the cache.
struct Tolerance {
    outputs: Distance,
    cache: Distance,
}

/// Whether `got` is within `tolerance` of `golden`, over the outputs and the keys and values.
fn check(name: &str, got: &Run, golden: &Run, tolerance: &Tolerance) -> bool {
    let outputs = distance(&got.outputs, &golden.outputs);
    let cache = distance(&got.cache, &golden.cache);
    eprintln!("{name} against the golden: outputs {outputs:?}; keys and values {cache:?}");
    outputs.within(tolerance.outputs) && cache.within(tolerance.cache)
}

#[test]
fn a_prefill_runs_on_the_decoder_it_was_recorded_for_alone() {
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    // A layer of the base LM's shape, its weights zero.
    let shape = Shape {
        hidden: 2048,
        inner: 6144,
        heads: 16,
        kv_heads: 2,
        head_dim: 128,
        layers: 1,
        capacity: 256,
        eps: 1e-5,
        rope_theta: 10_000.0,
        rope_factors: vec![1.0; 64],
        rope: true,
    };
    let tuning = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let decoder = |kernels: &mut Kernels| {
        Decoder::new(&ctx, kernels, &shape, &tuning, false).unwrap_or_else(|e| panic!("{e}"))
    };
    let (mut ours, mut other) = (decoder(&mut kernels), decoder(&mut kernels));
    let prefill = Prefill::new(&ctx, &mut kernels, &mut ours, TILE, &tuning, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let x = vec![0u16; 100 * shape.hidden];
    assert!(matches!(
        prefill.run(&mut other, 0, &x),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        prefill.run(&mut ours, 200, &x),
        Err(Error::Shape(_))
    ));
    let out = prefill
        .run(&mut ours, 0, &x)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.len(), x.len());
    assert!(out.iter().all(|&h| h == 0), "zero weights give zero");
    assert_no_validation_messages(&ctx);
}

/// One layer's tolerances: against burn's CPU in f32, the largest errors and relative RMS errors
/// of any run on the Radeon 8060S and the RTX 4070 Ti SUPER were 3.04e-2 and 3.26e-4 over the
/// outputs and 4.12e-3 and 2.53e-4 over the keys and values, and the golden's 3.04e-2 and
/// 2.49e-4, and 4.12e-3 and 2.53e-4.
const ONE_LAYER: Tolerance = Tolerance {
    outputs: Distance {
        max_abs: 7.6e-2,
        relative_rms: 7.2e-4,
    },
    cache: Distance {
        max_abs: 1.03e-2,
        relative_rms: 6.4e-4,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn one_layer_prefills_as_its_golden() {
    // Five tiles of rows, run as two prefills that each end inside a tile, the second from where
    // the first left the cache.
    let len = 600;
    let runs = [333, 267];
    let stack = base_lm(1);
    let x = embedded(len + STEPS);
    let want = golden_of(&stack, &x, &runs, 3 * TILE);
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(&stack, &x, &runs, 0, Some(subgroup), 3 * TILE);
        let name = format!("wave{subgroup}");
        assert!(check(&name, &got, &want, &ONE_LAYER), "{name}");
    }
    // Defect: the prompt one slot on in the cache, so rotated by the next position's angle.
    let shifted = gpu(&stack, &x, &[len], 1, None, 5 * TILE);
    assert!(
        !check("positions one slot on", &shifted, &want, &ONE_LAYER),
        "positions one slot on pass"
    );
}

/// The base LM's tolerances: against f32, any run's were 1.74e-1 and 2.09e-3 over the outputs
/// and 4.80e-2 and 1.07e-3 over the keys and values, and the golden's 1.74e-1 and 1.82e-3, and
/// 4.80e-2 and 1.07e-3.
const BASE_LM: Tolerance = Tolerance {
    outputs: Distance {
        max_abs: 4.4e-1,
        relative_rms: 4.9e-3,
    },
    cache: Distance {
        max_abs: 1.2e-1,
        relative_rms: 2.7e-3,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_base_lm_prefills_as_its_golden() {
    let len = 200;
    let stack = base_lm(28);
    let x = embedded(len + STEPS);
    let want = golden_of(&stack, &x, &[len], 2 * TILE);
    let got = gpu(&stack, &x, &[len], 0, None, 2 * TILE);
    assert!(check("the base LM", &got, &want, &BASE_LM));
}

/// The residual LM's tolerances: against f32, any run's were 6.33e-2 and 9.75e-4 over the
/// outputs and 1.94e-2 and 5.94e-4 over the keys and values, and the golden's 5.55e-2 and
/// 9.74e-4, and 1.94e-2 and 5.94e-4.
const RESIDUAL_LM: Tolerance = Tolerance {
    outputs: Distance {
        max_abs: 1.5e-1,
        relative_rms: 2.5e-3,
    },
    cache: Distance {
        max_abs: 4.9e-2,
        relative_rms: 1.5e-3,
    },
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_residual_lm_prefills_as_its_golden() {
    // Run as two prefills, of two tiles' rows and of one.
    let len = 300;
    let runs = [200, 100];
    let stack = residual_lm();
    assert!(stack.gpu.no_rope, "VoxCPM2's residual LM goes without RoPE");
    // What VoxCPM2 feeds its residual LM at positions of text, rounded to f16: fusion_concat_proj
    // over the base LM's outputs, beside audio embeddings that are zero there, as burn's CPU
    // computed it in f32.
    let x: Vec<u16> = golden_input(&format!("input-prefill-residual_lm.8-{len}"));
    let want = golden_of(&stack, &x, &runs, 2 * TILE);
    let got = gpu(&stack, &x, &runs, 0, None, 2 * TILE);
    assert!(check("the residual LM", &got, &want, &RESIDUAL_LM));
}
