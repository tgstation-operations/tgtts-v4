use burn::tensor::{FloatDType, Tensor};
use minicpm4::{Config, Error, KvCache, LocalShape, Mask, Model};
use serde_json::Value;
use testkit::{Fixture, assert_close, device, fixture};

// f32 here and in upstream, summing in other orders and rounding transcendentals an ulp
// apart: these outputs, of magnitude under 4, came within 9.6e-7 (4 ulps) of upstream's.
const ATOL: f32 = 2e-6;
const RTOL: f32 = 1e-6;

/// The tiny VoxCPM2 config the fixture was generated at, and the stacks derived from it.
struct Tiny {
    lm: Config,
    residual_lm: Config,
    encoder: Config,
}

fn tiny() -> Tiny {
    let json: Value = serde_json::from_str(include_str!("fixtures/minicpm4.json")).unwrap();
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

fn load(config: &Config, fixture: &Fixture, prefix: &str) -> Result<Model, Error> {
    Model::load(config, fixture.path(), prefix, FloatDType::F32, &device())
}

#[test]
fn base_lm_embeds_tokens_as_upstream() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().lm, &fixture, "base_lm.").unwrap();
    let embeds = model.embed(fixture.int_tensor("input.base_lm.ids", &device));
    // A lookup copies rows of the table, so nothing may differ.
    let expected = fixture.tensor("output.base_lm.embeds", &device);
    assert_close(embeds, expected, 0.0, 0.0);
}

#[test]
fn base_lm_attends_causally_with_rope_as_upstream() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().lm, &fixture, "base_lm.").unwrap();
    let x = fixture.tensor("output.base_lm.embeds", &device);
    let expected = fixture.tensor("output.base_lm.causal", &device);
    assert_close(model.forward(x, Mask::Causal), expected, ATOL, RTOL);
}

#[test]
fn residual_lm_attends_causally_without_rope_as_upstream() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().residual_lm, &fixture, "residual_lm.").unwrap();
    let x = fixture.tensor("input.residual_lm.x", &device);
    let expected = fixture.tensor("output.residual_lm.causal", &device);
    assert_close(model.forward(x, Mask::Causal), expected, ATOL, RTOL);
}

#[test]
fn local_encoder_attends_bidirectionally_with_rope_as_upstream() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().encoder, &fixture, "feat_encoder.encoder.").unwrap();
    let x = fixture.tensor("input.feat_encoder.encoder.x", &device);
    let expected = fixture.tensor("output.feat_encoder.encoder.bidirectional", &device);
    assert_close(model.forward(x, Mask::Bidirectional), expected, ATOL, RTOL);
}

/// Prefills the first 3 of the 7 positions of `input`, then runs 2 more at once and the last
/// 2 one at a time, where upstream stepped through all 4 one at a time.
fn decodes_as_upstream(config: &Config, prefix: &str, input: &str) {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(config, &fixture, prefix).unwrap();
    let x = fixture.tensor::<3>(input, &device);
    let mut cache = model.cache(2, 7);
    let mut run = |start, len| model.forward_cached(x.clone().narrow(1, start, len), &mut cache);
    let prefill = run(0, 3);
    let steps = Tensor::cat(vec![run(3, 2), run(5, 1), run(6, 1)], 1);
    let causal = fixture.tensor::<3>(&format!("output.{prefix}causal"), &device);
    assert_close(prefill, causal.narrow(1, 0, 3), ATOL, RTOL);
    let expected = fixture.tensor(&format!("output.{prefix}steps"), &device);
    assert_close(steps, expected, ATOL, RTOL);
}

#[test]
fn base_lm_decodes_as_upstream() {
    decodes_as_upstream(&tiny().lm, "base_lm.", "output.base_lm.embeds");
}

#[test]
fn residual_lm_decodes_as_upstream() {
    decodes_as_upstream(&tiny().residual_lm, "residual_lm.", "input.residual_lm.x");
}

/// Row `row` of `x` run alone through a cache of its own, in runs of the given lengths.
fn alone(model: &Model, x: &Tensor<3>, row: usize, runs: &[usize]) -> Vec<Tensor<3>> {
    let mut cache = model.cache(1, 7);
    let x = x.clone().narrow(0, row, 1);
    let mut start = 0;
    let mut outputs = vec![];
    for &len in runs {
        outputs.push(model.forward_cached(x.clone().narrow(1, start, len), &mut cache));
        start += len;
    }
    outputs
}

/// A cache of one row holding row `row` of `x` up to `len`.
fn prefilled(model: &Model, x: &Tensor<3>, row: usize, len: usize) -> KvCache {
    let mut cache = model.cache(1, 7);
    model.forward_cached(x.clone().narrow(0, row, 1).narrow(1, 0, len), &mut cache);
    cache
}

/// Positions `start..start + len` of each row of `x`, from the row each start is given for.
fn positions(x: &Tensor<3>, starts: &[(usize, usize)], len: usize) -> Tensor<3> {
    let rows = starts
        .iter()
        .map(|&(row, start)| x.clone().narrow(0, row, 1).narrow(1, start, len));
    Tensor::cat(rows.collect(), 0)
}

// Each row's products and its own keys are those it has alone, and the keys of longer rows are
// masked out as exact zeros: at this size every row comes out bit for bit as it does alone.
fn assert_as_alone(batch: Tensor<3>, row: usize, alone: &Tensor<3>) {
    assert_close(batch.narrow(0, row, 1), alone.clone(), 0.0, 0.0);
}

/// Rows prefilled alone to 3 and 5 positions, then run on together: a position at a time, the
/// longer row then holding nothing, and the shorter one's last two positions at once.
fn rows_go_on_as_alone(config: &Config, prefix: &str, input: &str) {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(config, &fixture, prefix).unwrap();
    let x = fixture.tensor::<3>(input, &device);
    let first = alone(&model, &x, 0, &[3, 1, 1, 2]);
    let second = alone(&model, &x, 1, &[5, 1]);

    let mut batch = model.cache(2, 7);
    batch.insert(0, &prefilled(&model, &x, 0, 3), 0);
    batch.insert(1, &prefilled(&model, &x, 1, 5), 0);
    assert_eq!((batch.len(0), batch.len(1)), (Some(3), Some(5)));
    let out = model.forward_cached(positions(&x, &[(0, 3), (1, 5)], 1), &mut batch);
    assert_as_alone(out.clone(), 0, &first[1]);
    assert_as_alone(out, 1, &second[1]);

    batch.clear(1);
    let out = model.forward_cached(positions(&x, &[(0, 4), (1, 0)], 1), &mut batch);
    assert_as_alone(out, 0, &first[2]);
    assert_eq!((batch.len(0), batch.len(1)), (Some(5), None));
    let out = model.forward_cached(positions(&x, &[(0, 5), (1, 0)], 2), &mut batch);
    assert_as_alone(out, 0, &first[3]);
    assert_eq!((batch.len(0), batch.len(1)), (Some(7), None));
}

#[test]
fn base_lm_rows_of_different_lengths_go_on_as_alone() {
    rows_go_on_as_alone(&tiny().lm, "base_lm.", "output.base_lm.embeds");
}

#[test]
fn residual_lm_rows_of_different_lengths_go_on_as_alone() {
    rows_go_on_as_alone(&tiny().residual_lm, "residual_lm.", "input.residual_lm.x");
}

/// The prefix of a sequence prefilled in runs of 2 and 2, and the sequence prefilled in runs of
/// 2 and 3: a prefill of other runs would round otherwise, by up to 2.4e-7 here.
#[test]
fn a_sequence_goes_on_from_a_copy_of_its_prefix() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().lm, &fixture, "base_lm.").unwrap();
    let x = fixture.tensor::<3>("output.base_lm.embeds", &device);
    let whole = alone(&model, &x, 1, &[2, 3]);
    let prefix = prefilled(&model, &x, 1, 2).prefix(0, 2);
    assert_eq!(
        (prefix.rows(), prefix.capacity(), prefix.len(0)),
        (1, 2, Some(2))
    );

    let mut cache = model.cache(1, 7);
    cache.insert(0, &prefix, 0);
    let out = model.forward_cached(x.clone().narrow(0, 1, 1).narrow(1, 2, 3), &mut cache);
    assert_close(out, whole[1].clone(), 0.0, 0.0);

    // A longer sequence cut back, as a shared prefix ends where two prompts part.
    let mut longer = prefilled(&model, &x, 1, 2);
    model.forward_cached(x.clone().narrow(0, 0, 1).narrow(1, 2, 2), &mut longer);
    let mut cache = model.cache(1, 7);
    cache.insert(0, &longer, 0);
    cache.truncate(0, 2);
    let out = model.forward_cached(x.clone().narrow(0, 1, 1).narrow(1, 2, 3), &mut cache);
    assert_close(out, whole[1].clone(), 0.0, 0.0);
}

/// A row whose sequence went NaN, cleared and taken by a shorter one, beside a row long enough
/// that attention reaches past the new sequence into what the old one left.
#[test]
fn a_sequence_sees_nothing_of_what_its_row_held_before() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().lm, &fixture, "base_lm.").unwrap();
    let x = fixture.tensor::<3>("output.base_lm.embeds", &device);
    let first = alone(&model, &x, 0, &[2, 1]);
    let mut batch = model.cache(2, 7);
    let mut diverged = model.cache(1, 7);
    let nan = x
        .clone()
        .narrow(0, 0, 1)
        .narrow(1, 0, 6)
        .mul_scalar(f32::NAN);
    model.forward_cached(nan, &mut diverged);
    batch.insert(0, &diverged, 0);
    batch.clear(0);
    batch.insert(0, &prefilled(&model, &x, 0, 2), 0);
    batch.insert(1, &prefilled(&model, &x, 1, 5), 0);
    let out = model.forward_cached(positions(&x, &[(0, 2), (1, 5)], 1), &mut batch);
    assert_as_alone(out, 0, &first[1]);
}

#[test]
fn reshaping_keeps_the_rows_it_is_told_to() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().lm, &fixture, "base_lm.").unwrap();
    let x = fixture.tensor::<3>("output.base_lm.embeds", &device);
    let first = alone(&model, &x, 0, &[2, 1]);
    let second = alone(&model, &x, 1, &[4, 1]);
    let mut batch = model.cache(2, 5);
    batch.insert(0, &prefilled(&model, &x, 0, 2), 0);
    batch.insert(1, &prefilled(&model, &x, 1, 4), 0);

    batch.reshape(&[1, 0], 3, 7);
    assert_eq!((batch.rows(), batch.capacity()), (3, 7));
    assert_eq!(
        [batch.len(0), batch.len(1), batch.len(2)],
        [Some(4), Some(2), None]
    );
    let out = model.forward_cached(positions(&x, &[(1, 4), (0, 2), (0, 0)], 1), &mut batch);
    assert_as_alone(out.clone(), 0, &second[1]);
    assert_as_alone(out, 1, &first[1]);

    batch.reshape(&[1], 1, 3);
    assert_eq!(
        (batch.rows(), batch.capacity(), batch.len(0)),
        (1, 3, Some(3))
    );
}

#[test]
#[should_panic(expected = "KV cache is full: 6 positions and 2 more are past its 7")]
fn a_full_cache_is_refused() {
    let (fixture, device) = (fixture!("minicpm4"), device());
    let model = load(&tiny().residual_lm, &fixture, "residual_lm.").unwrap();
    let x = fixture.tensor::<3>("input.residual_lm.x", &device);
    let mut cache = model.cache(2, 7);
    model.forward_cached(x.clone().narrow(1, 0, 6), &mut cache);
    model.forward_cached(x.narrow(1, 0, 2), &mut cache);
}

#[test]
fn a_stack_needs_every_weight() {
    let error = load(&tiny().lm, &fixture!("minicpm4"), "residual_lm.").unwrap_err();
    assert!(matches!(error, Error::Load(_)), "{error}");
}

fn refusal(edit: impl FnOnce(&mut Config)) -> String {
    let mut config = tiny().lm;
    edit(&mut config);
    let error = load(&config, &fixture!("minicpm4"), "base_lm.").unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    error.to_string()
}

#[test]
fn mup_is_refused() {
    let error = refusal(|config| config.use_mup = true);
    assert_eq!(error, "unsupported MiniCPM-4 config: use_mup");
}

#[test]
fn mup_is_assumed_where_unset_as_upstream_assumes_it() {
    let mut json: Value = serde_json::from_str(include_str!("fixtures/minicpm4.json")).unwrap();
    json["lm_config"].as_object_mut().unwrap().remove("use_mup");
    let config: Config = serde_json::from_value(json["lm_config"].take()).unwrap();
    assert!(config.use_mup);
}

#[test]
fn unknown_field_is_refused() {
    let mut json: Value = serde_json::from_str(include_str!("fixtures/minicpm4.json")).unwrap();
    json["lm_config"]["attention_bias"] = Value::Bool(false);
    let error = serde_json::from_value::<Config>(json["lm_config"].take()).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("unknown field `attention_bias`"),
        "{error}"
    );
}

#[test]
fn ungrouped_key_value_heads_are_refused() {
    let error = refusal(|config| config.num_key_value_heads = 3);
    assert_eq!(
        error,
        "unsupported MiniCPM-4 config: 8 attention heads shared among 3 key-value heads"
    );
}

#[test]
fn rope_other_than_longrope_is_refused() {
    let error = refusal(|config| config.rope_scaling.kind = "dynamic".into());
    assert_eq!(
        error,
        "unsupported MiniCPM-4 config: rope_scaling of type \"dynamic\""
    );
}

#[test]
fn stretched_longrope_is_refused() {
    let error = refusal(|config| config.max_position_embeddings = 64);
    assert_eq!(
        error,
        "unsupported MiniCPM-4 config: LongRoPE stretched from 32 positions to 64"
    );
}

#[test]
fn short_factors_must_cover_the_head() {
    let error = refusal(|config| config.kv_channels = Some(10));
    assert_eq!(
        error,
        "unsupported MiniCPM-4 config: 4 RoPE short factors for heads 10 wide"
    );
}
