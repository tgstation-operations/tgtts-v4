//! minicpm4's fixtures and real-weight golden on the GPU, in f32 and in f16.
#![cfg(feature = "vulkan")]

mod common;

use std::fs;

use burn::tensor::{FloatDType, Tensor};
use common::fixture;
use minicpm4::{Config, LocalShape, Mask, Model};
use serde_json::Value;
use testkit::{Fixture, assert_close, checkpoint_dir};

/// A dtype, and how far its outputs may be from upstream's in f32: `atol + rtol * |expected|`.
#[derive(Clone, Copy)]
struct Precision {
    dtype: FloatDType,
    atol: f32,
    rtol: f32,
}

// As on the CPU: these outputs, of magnitude under 4.3, came within 7.2e-7 of upstream's.
const F32: Precision = Precision {
    dtype: FloatDType::F32,
    atol: 2e-6,
    rtol: 1e-6,
};
// Inputs and weights rounded to f16 and computed in it, but for RMSNorm and RoPE: these came
// within 3.7e-3 of upstream's, and at most 7.4e-4 beyond four f16 epsilons of each.
const F16: Precision = Precision {
    dtype: FloatDType::F16,
    atol: 1.5e-3,
    rtol: 4e-3,
};

/// The tiny VoxCPM2 config minicpm4's fixture was generated at, and the stacks derived from it.
struct Tiny {
    lm: Config,
    residual_lm: Config,
    encoder: Config,
}

fn tiny() -> Tiny {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../minicpm4/tests/fixtures/minicpm4.json"
    );
    let json: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let lm: Config = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let count = |value: &Value| value.as_u64().unwrap() as usize;
    let encoder = &json["encoder_config"];
    let shape = LocalShape {
        hidden_dim: count(&encoder["hidden_dim"]),
        ffn_dim: count(&encoder["ffn_dim"]),
        num_heads: count(&encoder["num_heads"]),
        num_layers: count(&encoder["num_layers"]),
        kv_channels: encoder["kv_channels"].as_u64().map(|n| n as usize),
    };
    Tiny {
        residual_lm: lm.residual_lm(
            count(&json["residual_lm_num_layers"]),
            json["residual_lm_no_rope"].as_bool().unwrap(),
        ),
        encoder: lm.local(&shape),
        lm,
    }
}

fn check(actual: Tensor<3>, expected: Tensor<3>, precision: Precision) {
    let actual = actual.cast(FloatDType::F32);
    assert_close(actual, expected, precision.atol, precision.rtol);
}

/// Runs the stack under `prefix` over `input` at once, against upstream's output.
fn attends_as_upstream(
    config: &Config,
    prefix: &str,
    input: &str,
    mask: Mask,
    precision: Precision,
) {
    let (fixture, device) = (fixture("minicpm4", "minicpm4"), gpu::device());
    let model = Model::load(config, fixture.path(), prefix, precision.dtype, &device).unwrap();
    let x = fixture.tensor::<3>(input, &device).cast(precision.dtype);
    let output = match mask {
        Mask::Causal => format!("output.{prefix}causal"),
        Mask::Bidirectional => format!("output.{prefix}bidirectional"),
    };
    check(
        model.forward(x, mask),
        fixture.tensor(&output, &device),
        precision,
    );
}

/// Prefills the first 3 of the 7 positions of `input`, then runs 2 more at once and the last
/// 2 one at a time, where upstream stepped through all 4 one at a time.
fn decodes_as_upstream(config: &Config, prefix: &str, input: &str, precision: Precision) {
    let (fixture, device) = (fixture("minicpm4", "minicpm4"), gpu::device());
    let model = Model::load(config, fixture.path(), prefix, precision.dtype, &device).unwrap();
    let x = fixture.tensor::<3>(input, &device).cast(precision.dtype);
    let mut cache = model.cache(2, 7);
    let mut run = |start, len| model.forward_cached(x.clone().narrow(1, start, len), &mut cache);
    let prefill = run(0, 3);
    let steps = Tensor::cat(vec![run(3, 2), run(5, 1), run(6, 1)], 1);
    let causal = fixture.tensor::<3>(&format!("output.{prefix}causal"), &device);
    check(prefill, causal.narrow(1, 0, 3), precision);
    let expected = fixture.tensor(&format!("output.{prefix}steps"), &device);
    check(steps, expected, precision);
}

/// Rows prefilled alone to 3 and 5 positions run on together, a position at a time, then the
/// shorter alone with the other row holding nothing, its last two positions at once: each
/// against upstream's output at the same positions.
fn rows_go_on_as_upstream(config: &Config, prefix: &str, input: &str, precision: Precision) {
    let (fixture, device) = (fixture("minicpm4", "minicpm4"), gpu::device());
    let model = Model::load(config, fixture.path(), prefix, precision.dtype, &device).unwrap();
    let x = fixture.tensor::<3>(input, &device).cast(precision.dtype);
    let causal = fixture.tensor::<3>(&format!("output.{prefix}causal"), &device);
    let at = |x: &Tensor<3>, rows: &[(usize, usize)], len| {
        let rows = rows
            .iter()
            .map(|&(row, start)| x.clone().narrow(0, row, 1).narrow(1, start, len));
        Tensor::cat(rows.collect(), 0)
    };
    let mut batch = model.cache(2, 7);
    for (row, len) in [(0, 3), (1, 5)] {
        let mut alone = model.cache(1, 7);
        model.forward_cached(at(&x, &[(row, 0)], len), &mut alone);
        batch.insert(row, &alone, 0);
    }
    let out = model.forward_cached(at(&x, &[(0, 3), (1, 5)], 1), &mut batch);
    check(out, at(&causal, &[(0, 3), (1, 5)], 1), precision);
    batch.clear(1);
    let out = model.forward_cached(at(&x, &[(0, 4), (1, 0)], 1), &mut batch);
    check(out.narrow(0, 0, 1), at(&causal, &[(0, 4)], 1), precision);
    let out = model.forward_cached(at(&x, &[(0, 5), (1, 0)], 2), &mut batch);
    check(out.narrow(0, 0, 1), at(&causal, &[(0, 5)], 2), precision);
}

#[test]
fn base_lm_rows_of_different_lengths_go_on_as_upstream_in_f32() {
    rows_go_on_as_upstream(&tiny().lm, "base_lm.", "output.base_lm.embeds", F32);
}

#[test]
fn base_lm_rows_of_different_lengths_go_on_as_upstream_in_f16() {
    rows_go_on_as_upstream(&tiny().lm, "base_lm.", "output.base_lm.embeds", F16);
}

#[test]
fn residual_lm_rows_of_different_lengths_go_on_as_upstream_in_f32() {
    let lm = tiny().residual_lm;
    rows_go_on_as_upstream(&lm, "residual_lm.", "input.residual_lm.x", F32);
}

#[test]
fn residual_lm_rows_of_different_lengths_go_on_as_upstream_in_f16() {
    let lm = tiny().residual_lm;
    rows_go_on_as_upstream(&lm, "residual_lm.", "input.residual_lm.x", F16);
}

fn base_lm_attends(precision: Precision) {
    let input = "output.base_lm.embeds";
    attends_as_upstream(&tiny().lm, "base_lm.", input, Mask::Causal, precision);
}

fn residual_lm_attends(precision: Precision) {
    let input = "input.residual_lm.x";
    attends_as_upstream(
        &tiny().residual_lm,
        "residual_lm.",
        input,
        Mask::Causal,
        precision,
    );
}

fn local_encoder_attends(precision: Precision) {
    let (prefix, input) = ("feat_encoder.encoder.", "input.feat_encoder.encoder.x");
    attends_as_upstream(
        &tiny().encoder,
        prefix,
        input,
        Mask::Bidirectional,
        precision,
    );
}

#[test]
fn base_lm_attends_as_upstream_in_f32() {
    base_lm_attends(F32);
}

#[test]
fn base_lm_attends_as_upstream_in_f16() {
    base_lm_attends(F16);
}

#[test]
fn residual_lm_attends_as_upstream_in_f32() {
    residual_lm_attends(F32);
}

#[test]
fn residual_lm_attends_as_upstream_in_f16() {
    residual_lm_attends(F16);
}

#[test]
fn local_encoder_attends_as_upstream_in_f32() {
    local_encoder_attends(F32);
}

#[test]
fn local_encoder_attends_as_upstream_in_f16() {
    local_encoder_attends(F16);
}

#[test]
fn base_lm_decodes_as_upstream_in_f32() {
    decodes_as_upstream(&tiny().lm, "base_lm.", "output.base_lm.embeds", F32);
}

#[test]
fn base_lm_decodes_as_upstream_in_f16() {
    decodes_as_upstream(&tiny().lm, "base_lm.", "output.base_lm.embeds", F16);
}

#[test]
fn residual_lm_decodes_as_upstream_in_f32() {
    decodes_as_upstream(
        &tiny().residual_lm,
        "residual_lm.",
        "input.residual_lm.x",
        F32,
    );
}

#[test]
fn residual_lm_decodes_as_upstream_in_f16() {
    decodes_as_upstream(
        &tiny().residual_lm,
        "residual_lm.",
        "input.residual_lm.x",
        F16,
    );
}

#[test]
fn base_lm_embeds_tokens_exactly() {
    let (fixture, device) = (fixture("minicpm4", "minicpm4"), gpu::device());
    let model = Model::load(
        &tiny().lm,
        fixture.path(),
        "base_lm.",
        FloatDType::F32,
        &device,
    );
    let embeds = model
        .unwrap()
        .embed(fixture.int_tensor("input.base_lm.ids", &device));
    let expected = fixture.tensor("output.base_lm.embeds", &device);
    assert_close(embeds, expected, 0.0, 0.0);
}

/// The residual LM on the real weights at once and a step at a time, against upstream's in f32.
fn residual_lm_runs_and_decodes_as_upstream(precision: Precision) {
    let (dir, device) = (checkpoint_dir(), gpu::device());
    let json: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let lm: Config = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let config = lm.residual_lm(
        json["residual_lm_num_layers"].as_u64().unwrap() as usize,
        json["residual_lm_no_rope"].as_bool().unwrap(),
    );
    let weights = dir.join("model.safetensors");
    let model = Model::load(&config, weights, "residual_lm.", precision.dtype, &device).unwrap();

    let golden: Fixture = fixture("minicpm4", "minicpm4_residual_lm");
    let x = golden
        .tensor::<3>("input.residual_lm.x", &device)
        .cast(precision.dtype);
    let expected = golden.tensor::<3>("output.residual_lm.causal", &device);
    check(
        model.forward(x.clone(), Mask::Causal),
        expected.clone(),
        precision,
    );

    let mut cache = model.cache(2, 8);
    let mut decoded = vec![model.forward_cached(x.clone().narrow(1, 0, 5), &mut cache)];
    for t in 5..8 {
        decoded.push(model.forward_cached(x.clone().narrow(1, t, 1), &mut cache));
    }
    check(Tensor::cat(decoded, 1), expected, precision);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn residual_lm_on_the_real_weights_runs_and_decodes_as_upstream_in_f32() {
    // As on the CPU: these, of RMS 1.2 and magnitude up to 20, came within 7.2e-5 of upstream's.
    let precision = Precision {
        atol: 5e-5,
        rtol: 1e-5,
        ..F32
    };
    residual_lm_runs_and_decodes_as_upstream(precision);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn residual_lm_on_the_real_weights_runs_and_decodes_as_upstream_in_f16() {
    // Through 8 layers: these came within 2.5e-2 of upstream's, and at most 1.1e-2 beyond four
    // f16 epsilons of each.
    let precision = Precision { atol: 2e-2, ..F16 };
    residual_lm_runs_and_decodes_as_upstream(precision);
}
