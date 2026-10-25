use burn::tensor::{FloatDType, Tensor};
use minicpm4::{Config, Error, LocalShape, Mask, Model};
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
