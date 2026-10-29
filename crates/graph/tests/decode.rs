//! The decode step on the GPU, on VoxCPM2's real base LM over a real prompt a position at a
//! time, held to its golden, at every subgroup size the device can require. Built with the
//! `device` feature, and run with `VOXCPM2_DIR` naming the checkpoint, `GOLDEN_DIR` the goldens
//! and `GLSL_KERNELS` the compiled kernels, as `nix run .#device-tests` sets the last. The GPU
//! also runs the prompt as several requests at once, a row each, each row at its own position in
//! its own slot of the cache, and each comes out bit for bit as it does alone. Each tolerance is
//! set as testkit's `golden` says.
#![cfg(feature = "device")]

use std::{fs, time::Instant};

use graph::{
    checkpoint::Checkpoint,
    decode::{Batch, Decoder, LmConfig, Row, SUBGROUPS, Shape, Tuning},
    kernels::Kernels,
};
use serde_json::Value;
use testkit::{Fixture, checkpoint_dir, golden};
use vulkan::{Context, F16};

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

/// Token ids upstream's tokenizer gave for two of the game's lines, 79 in all: positions in three
/// of attention's chunks of 32.
fn prompt_ids() -> Vec<i64> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../voxcpm/tests/fixtures/voxcpm_tokenizer_checkpoint.safetensors"
    );
    let fixture = Fixture::open(path);
    ["messages.15", "messages.24"]
        .iter()
        .flat_map(|name| fixture.ints(name).1)
        .collect()
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

/// A stack of VoxCPM2's: its base LM; its residual LM, which goes without RoPE; or, as a defect,
/// the residual LM rotated as the base LM is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stack {
    Base,
    Residual,
    RotatedResidual,
}

impl Stack {
    fn prefix(self) -> &'static str {
        match self {
            Stack::Base => "base_lm.",
            _ => "residual_lm.",
        }
    }
}

fn config() -> Value {
    let dir = checkpoint_dir();
    serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

/// The prompt's embeddings in the base LM's table, rounded to f16: `[positions][hidden]`.
fn embeddings() -> Vec<F16> {
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    let ids: Vec<usize> = prompt_ids().iter().map(|&id| id as usize).collect();
    checkpoint
        .gather_f16("base_lm.embed_tokens.weight", &ids)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The golden of `layers` of `stack` over `x`: their outputs at the device's default tuning.
fn golden_of(stack: Stack, layers: usize, x: &[F16]) -> Vec<f32> {
    golden(&format!("decode-{stack:?}-{layers}"), || {
        gpu(stack, layers, x, 0, None)
    })
}

/// The GPU's outputs for the same layers of `stack` over the embeddings `x`, every position one at
/// a time, each at its position plus `shift` in the cache, with the kernels at `subgroup` (the
/// device's default when `None`).
fn gpu(stack: Stack, layers: usize, x: &[F16], shift: usize, subgroup: Option<u32>) -> Vec<f32> {
    gpu_rows(stack, layers, x, shift, subgroup, 1).remove(0)
}

/// [`gpu`]'s outputs for each of `rows` requests run at once, cut as [`Tuning::for_batch`] cuts
/// them: request r from step 2 r on, in slot `rows - 1 - r` of the cache, in a row that changes
/// from step to step, the rows without one spare.
fn gpu_rows(
    stack: Stack,
    layers: usize,
    x: &[F16],
    shift: usize,
    subgroup: Option<u32>,
    rows: usize,
) -> Vec<Vec<f32>> {
    let json = config();
    let hidden = json["lm_config"]["hidden_size"].as_u64().unwrap() as usize;
    let len = x.len() / hidden;
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let config = LmConfig {
        no_rope: stack == Stack::Residual,
        ..serde_json::from_value(json["lm_config"].clone()).unwrap()
    };
    let shape = Shape::new(&config, layers, (len + shift).next_multiple_of(32)).unwrap();
    let start = Instant::now();
    let subgroup = subgroup.unwrap_or_else(|| {
        let device = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
        device.subgroup
    });
    let tuning = Tuning::for_batch(subgroup, rows);
    let batch = Batch { rows, slots: rows };
    let mut decoder = Decoder::with_batch(&ctx, &mut kernels, &shape, &tuning, batch, false)
        .unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, stack.prefix())
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "{layers} layers in {rows} rows recorded and loaded on the GPU at {tuning:?}: {:.1} s, \
         {} dispatches, {} barriers, {} pipelines, an arena of {} KiB",
        start.elapsed().as_secs_f64(),
        decoder.graph_step().dispatches(),
        decoder.graph_step().barriers(),
        kernels.len(),
        decoder.graph_step().arena_bytes() / 1024
    );
    let mut got = vec![Vec::with_capacity(len * hidden); rows];
    for t in 0..len + 2 * (rows - 1) {
        let mut placed = vec![None; rows];
        let mut input = vec![F16::ZERO; rows * hidden];
        for r in 0..rows {
            let Some(i) = t.checked_sub(2 * r).filter(|&i| i < len) else {
                continue;
            };
            let row = (r + t) % rows;
            placed[row] = Some(Row {
                position: i + shift,
                slot: rows - 1 - r,
            });
            input[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&x[i * hidden..(i + 1) * hidden]);
        }
        let out = decoder
            .step_rows(&placed, &input)
            .unwrap_or_else(|e| panic!("{e}"));
        for r in (0..rows).filter(|r| (2 * r..2 * r + len).contains(&t)) {
            let row = (r + t) % rows;
            let out = &out[row * hidden..(row + 1) * hidden];
            got[r].extend(out.iter().map(|h| h.to_f32()));
        }
    }
    assert_no_validation_messages(&ctx);
    got
}

/// How far `got` is from `golden` over every position. Errors are compared over many positions:
/// in a deep stack they gather in a few outlier channels, and at any one position which of two
/// runs rounds those worse is luck.
fn check(name: &str, got: &[f32], golden: &[f32]) -> Distance {
    let d = distance(got, golden);
    eprintln!("{name}: against the golden {d:?}");
    d
}

/// One layer's tolerance: against burn's CPU in f32, the largest error and relative RMS error of
/// any run on the Radeon 8060S and the RTX 4070 Ti SUPER were 3.44e-2 and 3.53e-4, and the
/// golden's 3.44e-2 and 3.49e-4.
const ONE_LAYER: Distance = Distance {
    max_abs: 8.6e-2,
    relative_rms: 8.8e-4,
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn one_layer_decodes_as_its_golden() {
    let x = embeddings();
    let want = golden_of(Stack::Base, 1, &x);
    // At every subgroup size the device can require: 32, as on NVIDIA, also on the Radeon.
    let report = context().report().clone();
    for subgroup in SUBGROUPS
        .into_iter()
        .filter(|&s| report.can_require_subgroup_size(s))
    {
        let got = gpu(Stack::Base, 1, &x, 0, Some(subgroup));
        let d = check(&format!("wave{subgroup}"), &got, &want);
        assert!(d.within(ONE_LAYER), "wave{subgroup}: {d:?}");
        // Three requests at once, each at its own position: each as it is alone.
        let rows = gpu_rows(Stack::Base, 1, &x, 0, Some(subgroup), 3);
        for (r, row) in rows.iter().enumerate() {
            assert!(
                row == &got,
                "wave{subgroup}: row {r} of 3 is not as it is alone"
            );
        }
    }
    // Defect: every position one slot on in the cache, so rotated by the next position's angle.
    let shifted = gpu(Stack::Base, 1, &x, 1, None);
    let d = check("positions one slot on", &shifted, &want);
    assert!(!d.within(ONE_LAYER), "positions one slot on pass: {d:?}");
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn one_layer_recorded_again_with_its_rows_cut_otherwise_decodes_the_same() {
    let json = config();
    let config: LmConfig = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let ctx = context();
    let mut kernels = Kernels::from_env(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let positions = 40;
    let shape = Shape::new(&config, 1, 64).unwrap();
    let base = Tuning::for_device(ctx.report()).unwrap_or_else(|e| panic!("{e}"));
    let mut decoder =
        Decoder::new(&ctx, &mut kernels, &shape, &base, false).unwrap_or_else(|e| panic!("{e}"));
    let checkpoint = Checkpoint::open(checkpoint_dir().join("model.safetensors"))
        .unwrap_or_else(|e| panic!("{e}"));
    decoder
        .load(&checkpoint, "base_lm.")
        .unwrap_or_else(|e| panic!("{e}"));
    // Hidden states of about the embeddings' scale, from a fixed xorshift sequence.
    let mut state = 0x9e37_79b9_u32;
    let x: Vec<F16> = (0..positions * shape.hidden)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            F16::from_f64((f64::from(state) / f64::from(u32::MAX) - 0.5) * 0.2)
        })
        .collect();
    let run = |decoder: &mut Decoder| -> Vec<F16> {
        x.chunks(shape.hidden)
            .enumerate()
            .flat_map(|(t, x)| decoder.step(t, x).unwrap_or_else(|e| panic!("{e}")))
            .collect()
    };
    let want = run(&mut decoder);
    // Rows to a subgroup, loads ahead and the workgroup width decide where and when each row is
    // summed, not in what order: the norms are summed in the order of the default's workgroup.
    let cuts = [
        (
            Tuning {
                workgroup: 4 * base.workgroup,
                ..base
            },
            false,
        ),
        (
            Tuning {
                qkv_pairs: 2,
                gate_up_pairs: 2,
                residual_rows: 2,
                ..base
            },
            false,
        ),
        (Tuning { ahead: 64, ..base }, false),
        (base, true),
    ];
    for (tuning, timed) in cuts {
        decoder = decoder
            .retuned(&ctx, &mut kernels, &tuning, timed)
            .unwrap_or_else(|e| panic!("{e}"));
        let got = run(&mut decoder);
        let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(
            differ,
            0,
            "{tuning:?}, timed {timed}: {differ} of {} differ",
            want.len()
        );
    }
    eprintln!("{positions} positions alike at {base:?} and recorded again as {cuts:?}");
    assert_no_validation_messages(&ctx);
}

/// The base LM's tolerance: against f32, any run's were 1.81e-1 and 2.00e-3, and the golden's
/// 1.53e-1 and 1.58e-3.
const BASE_LM: Distance = Distance {
    max_abs: 4.2e-1,
    relative_rms: 4.5e-3,
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_base_lm_decodes_as_its_golden() {
    let x = embeddings();
    let want = golden_of(Stack::Base, 28, &x);
    let got = gpu(Stack::Base, 28, &x, 0, None);
    let d = check("the base LM", &got, &want);
    assert!(d.within(BASE_LM), "{d:?}");
    // Sixteen requests at once, each at its own position: each as it is alone.
    let rows = gpu_rows(Stack::Base, 28, &x, 0, None, 16);
    for (r, row) in rows.iter().enumerate() {
        assert!(row == &got, "row {r} of 16 is not as it is alone");
    }
}

/// The residual LM's tolerance: against f32, any run's were 3.55e-2 and 1.17e-3, and the
/// golden's 3.30e-2 and 1.15e-3.
const RESIDUAL_LM: Distance = Distance {
    max_abs: 8.7e-2,
    relative_rms: 2.9e-3,
};

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint and GOLDEN_DIR the goldens"]
fn the_residual_lm_decodes_without_rope_as_its_golden() {
    let x = embeddings();
    let want = golden_of(Stack::Residual, 8, &x);
    let got = gpu(Stack::Residual, 8, &x, 0, None);
    let d = check("the residual LM", &got, &want);
    assert!(d.within(RESIDUAL_LM), "{d:?}");
    let rows = gpu_rows(Stack::Residual, 8, &x, 0, None, 4);
    for (r, row) in rows.iter().enumerate() {
        assert!(row == &got, "row {r} of 4 is not as it is alone");
    }
    // Defect: the residual LM rotated by RoPE.
    let rotated = gpu(Stack::RotatedResidual, 8, &x, 0, None);
    let d = check("the residual LM rotated", &rotated, &want);
    assert!(
        !d.within(RESIDUAL_LM),
        "the residual LM rotated passes: {d:?}"
    );
}
