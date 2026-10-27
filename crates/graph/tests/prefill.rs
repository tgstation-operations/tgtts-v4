//! The prefill on the GPU against minicpm4's stack on burn's CPU backend, on VoxCPM2's real base
//! LM over a real prompt: the prompt run at once on both, then decode steps on both from where it
//! left the cache. The keys and values it leaves in the cache and the outputs over it are
//! compared, then the steps' outputs. Built with the `device` feature, and run with `VOXCPM2_DIR`
//! naming the checkpoint and `GLSL_KERNELS` and `VULKAN_KERNELS` the compiled kernels, as `nix run
//! .#device-tests` sets them.
#![cfg(feature = "device")]

use std::{collections::BTreeMap, fs, time::Instant};

use burn::tensor::{FloatDType, Int, Tensor, TensorData};
use graph::{
    Error,
    checkpoint::Checkpoint,
    decode::{Decoder, LmConfig, SUBGROUPS, Shape, Tuning},
    kernels::Kernels,
    prefill::{Prefill, TILE},
};
use half::f16;
use minicpm4::{Config, Mask, Model};
use serde_json::Value;
use testkit::{Fixture, checkpoint_dir, device};
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
        let message = fixture.int_tensor::<1>(&format!("messages.{i}"), &device());
        ids.extend(message.into_data().iter::<i64>());
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
    /// The largest |a - b| / (atol + rtol |b|).
    worst: f64,
    /// RMS of the differences over RMS of the reference.
    relative_rms: f64,
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max |d| {:.2e}, worst at {:.2} of the tolerance, relative RMS {:.2e}",
            self.max_abs, self.worst, self.relative_rms
        )
    }
}

fn distance(got: &[f32], want: &[f32]) -> Distance {
    let (atol, rtol) = SERVED;
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut worst, mut err2, mut ref2) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        worst = worst.max(d / (atol + rtol * w.abs()));
        if !d.is_finite() {
            max_abs = f64::INFINITY;
            worst = f64::INFINITY;
        }
        err2 += d * d;
        ref2 += w * w;
    }
    Distance {
        max_abs,
        worst,
        relative_rms: (err2 / ref2).sqrt(),
    }
}

/// The served f16 tolerance of crates/gpu's real-weight tests, set for 8 layers.
const SERVED: (f64, f64) = (2e-2, 4e-3);

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
struct Run {
    outputs: Vec<f32>,
    cache: Vec<f32>,
}

/// One of VoxCPM2's causal stacks: its config as minicpm4 reads it and as the decode step does,
/// and the prefix of its weights' names.
struct Stack {
    cpu: Config,
    gpu: LmConfig,
    prefix: &'static str,
}

/// The base LM's first `layers` layers.
fn base_lm(layers: usize) -> Stack {
    let lm = &config()["lm_config"];
    let mut cpu: Config = serde_json::from_value(lm.clone()).unwrap();
    cpu.num_hidden_layers = layers;
    let mut gpu: LmConfig = serde_json::from_value(lm.clone()).unwrap();
    gpu.num_hidden_layers = layers;
    Stack {
        cpu,
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
        cpu: base.cpu.residual_lm(layers, no_rope),
        gpu: LmConfig {
            no_rope,
            ..base.gpu
        },
        prefix: "residual_lm.",
    }
}

/// The base LM's embeddings of `len` tokens of the game's lines, `[len][hidden]`.
fn embeddings(model: &Model, len: usize) -> Vec<f32> {
    let ids = Tensor::<2, Int>::from_data(TensorData::new(prompt_ids(len), [1, len]), &device());
    model.embed(ids).into_data().iter::<f32>().collect()
}

fn rounded(x: &[f32]) -> Vec<u16> {
    x.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
}

/// minicpm4 on the CPU over `stack`, in f32 and in f16, over `x16`, hidden states
/// `[len + STEPS][hidden]` as f16 bits: the first `len` as a prompt at once, then a step at a
/// time.
fn oracle(stack: &Stack, x16: &[u16], len: usize) -> (Run, Run) {
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let config = &stack.cpu;
    let (hidden, layers) = (config.hidden_size, config.num_hidden_layers);
    let total = len + STEPS;
    assert_eq!(x16.len(), total * hidden);
    let start = Instant::now();
    let run = |dtype: FloatDType| -> Run {
        let model = Model::load(config, &weights, stack.prefix, dtype, &cpu)
            .unwrap_or_else(|e| panic!("{e}"));
        let x = Tensor::<3>::from_data(TensorData::new(f16s(x16), [1, total, hidden]), &cpu)
            .cast(dtype);
        let mut cache = model.cache(1, total);
        let prompt = model.forward_cached(x.clone().narrow(1, 0, len), &mut cache);
        let mut held = Vec::new();
        for l in 0..layers {
            let (keys, values) = cache.layer(l);
            for t in [keys, values] {
                let t = t.narrow(2, 0, len).cast(FloatDType::F32);
                held.extend(t.into_data().iter::<f32>());
            }
        }
        let mut outputs = vec![prompt];
        for t in len..total {
            outputs.push(model.forward_cached(x.clone().narrow(1, t, 1), &mut cache));
        }
        let outputs = Tensor::cat(outputs, 1).cast(FloatDType::F32);
        Run {
            outputs: outputs.into_data().iter::<f32>().collect(),
            cache: held,
        }
    };
    let want = run(FloatDType::F32);
    let burn_f16 = run(FloatDType::F16);
    eprintln!(
        "{}{layers} layers over {len} positions on the CPU, f32 and f16: {:.1} s",
        stack.prefix,
        start.elapsed().as_secs_f64()
    );
    (want, burn_f16)
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

/// Checks that the GPU stays as close to f32 as f16 arithmetic on the CPU does: over the
/// prompt's outputs and over the keys and values it left in the cache, its largest error and its
/// relative RMS error within 1.25 times burn's f16; and the decode steps', where burn's f16
/// drifts on the CPU, within 1.25 times what burn's f16 reached over the prompt, as the decode
/// step's own test holds them. Returns the GPU's distance from f32 over the prompt's outputs,
/// and the margins.
fn check(want: &Run, burn_f16: &Run, got: &Run, len: usize) -> (Distance, [f64; 6]) {
    let split = len * 2048;
    let ours = distance(&got.outputs[..split], &want.outputs[..split]);
    let burn = distance(&burn_f16.outputs[..split], &want.outputs[..split]);
    let ours_cache = distance(&got.cache, &want.cache);
    let burn_cache = distance(&burn_f16.cache, &want.cache);
    let ours_step = distance(&got.outputs[split..], &want.outputs[split..]);
    let burn_step = distance(&burn_f16.outputs[split..], &want.outputs[split..]);
    let margin = [
        ours.max_abs / burn.max_abs,
        ours.relative_rms / burn.relative_rms,
        ours_cache.max_abs / burn_cache.max_abs,
        ours_cache.relative_rms / burn_cache.relative_rms,
        ours_step.max_abs / burn.max_abs,
        ours_step.relative_rms / burn.relative_rms,
    ];
    eprintln!(
        "prompt: GPU {ours:?}; burn f16 {burn:?}\nkeys and values: GPU {ours_cache:?}; burn f16 \
         {burn_cache:?}\ndecode steps: GPU {ours_step:?}; burn f16 {burn_step:?}\nthe GPU's \
         against burn's f16: {margin:.2?}"
    );
    (ours, margin)
}

fn within(margin: [f64; 6]) -> bool {
    margin.iter().all(|&m| m <= 1.25)
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

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_layer_prefills_as_minicpm4_does_on_the_cpu() {
    // Five tiles of rows, run as two prefills that each end inside a tile, the second from where
    // the first left the cache.
    let len = 600;
    let runs = [333, 267];
    let stack = base_lm(1);
    let model = Model::load(
        &stack.cpu,
        checkpoint_dir().join("model.safetensors"),
        stack.prefix,
        FloatDType::F32,
        &device(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let x = rounded(&embeddings(&model, len + STEPS));
    drop(model);
    let (want, burn_f16) = oracle(&stack, &x, len);
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(&stack, &x, &runs, 0, Some(subgroup), 3 * TILE);
        let (ours, margin) = check(&want, &burn_f16, &got, len);
        assert!(within(margin), "wave{subgroup}: {margin:?}");
        // One layer is well inside the served tolerance, which was set for eight.
        assert!(ours.worst <= 1.0, "wave{subgroup}: {ours:?}");
    }
    // Defect: the prompt one slot on in the cache, so rotated by the next position's angle.
    let shifted = gpu(&stack, &x, &[len], 1, None, 5 * TILE);
    let (_, margin) = check(&want, &burn_f16, &shifted, len);
    assert!(!within(margin), "positions one slot on pass: {margin:?}");
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_base_lm_prefills_as_minicpm4_does_on_the_cpu() {
    let len = 200;
    let stack = base_lm(28);
    let model = Model::load(
        &stack.cpu,
        checkpoint_dir().join("model.safetensors"),
        stack.prefix,
        FloatDType::F32,
        &device(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let x = rounded(&embeddings(&model, len + STEPS));
    drop(model);
    let (want, burn_f16) = oracle(&stack, &x, len);
    let got = gpu(&stack, &x, &[len], 0, None, 2 * TILE);
    let (_, margin) = check(&want, &burn_f16, &got, len);
    assert!(within(margin), "{margin:?}");
}

/// What VoxCPM2 feeds its residual LM at positions of text: `fusion_concat_proj` over the base
/// LM's outputs beside audio embeddings that are zero there, on the CPU in f32 over the base
/// LM's f32 outputs over `len` tokens.
fn residual_inputs(len: usize) -> Vec<f32> {
    let weights = checkpoint_dir().join("model.safetensors");
    let cpu = device();
    let stack = base_lm(28);
    let hidden = stack.cpu.hidden_size;
    let model = Model::load(&stack.cpu, &weights, stack.prefix, FloatDType::F32, &cpu)
        .unwrap_or_else(|e| panic!("{e}"));
    let x = Tensor::<3>::from_data(
        TensorData::new(embeddings(&model, len), [1, len, hidden]),
        &cpu,
    );
    let outputs = model.forward(x, Mask::Causal).reshape([len, hidden]);
    drop(model);
    let checkpoint = Checkpoint::open(&weights).unwrap_or_else(|e| panic!("{e}"));
    let read = |name: &str| {
        let (shape, bits) = checkpoint.f16(name).unwrap_or_else(|e| panic!("{e}"));
        (shape, f16s(&bits))
    };
    let (shape, w) = read("fusion_concat_proj.weight");
    assert_eq!(shape, [hidden, 2 * hidden]);
    let (_, b) = read("fusion_concat_proj.bias");
    let w = Tensor::<2>::from_data(TensorData::new(w, [hidden, 2 * hidden]), &cpu);
    let b = Tensor::<1>::from_data(TensorData::new(b, [hidden]), &cpu);
    let projected = outputs.matmul(w.narrow(1, 0, hidden).transpose()) + b.unsqueeze();
    projected.into_data().iter::<f32>().collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_residual_lm_prefills_as_minicpm4_does_on_the_cpu() {
    // Run as two prefills, of two tiles' rows and of one.
    let len = 300;
    let runs = [200, 100];
    let stack = residual_lm();
    assert!(stack.gpu.no_rope, "VoxCPM2's residual LM goes without RoPE");
    let x = rounded(&residual_inputs(len + STEPS));
    let (want, burn_f16) = oracle(&stack, &x, len);
    let got = gpu(&stack, &x, &runs, 0, None, 2 * TILE);
    let (_, margin) = check(&want, &burn_f16, &got, len);
    assert!(within(margin), "{margin:?}");
}
