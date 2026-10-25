use std::{fs, path::Path};

use burn::tensor::{DType, FloatDType, Tensor};
use minicpm4::{Config, LocalShape, Mask, Model};
use serde_json::Value;
use testkit::{assert_close, checkpoint_dir, device, fixture};

// f32 here and in upstream, summing 2048 to 6144 terms in other orders through 8 layers: that
// moved these outputs, of RMS 1.2 and magnitude up to 20, by up to 2.5e-5.
const ATOL: f32 = 5e-5;
const RTOL: f32 = 1e-5;

fn config_json(dir: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn count(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}

/// Making the golden needs the checkpoint, so a run of every generator leaves it as it is:
/// this refuses one that such a run has left behind.
#[test]
fn real_weight_golden_was_made_as_the_fixtures_were() {
    let (golden, fixtures) = (fixture!("minicpm4_residual_lm"), fixture!("minicpm4"));
    let golden = golden.metadata();
    let regenerate = "with VOXCPM2_DIR set, regenerate it: nix run .#fixtures minicpm4_residual_lm";
    for key in ["torch", "upstream"] {
        assert_eq!(
            golden[key],
            fixtures.metadata()[key],
            "the golden's {key} differs from the fixtures'; {regenerate}"
        );
    }
    let generator = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/minicpm4_residual_lm.py"
    );
    let crc32 = format!("{:08x}", crc32fast::hash(&fs::read(generator).unwrap()));
    assert_eq!(
        golden.get("generator_crc32"),
        Some(&crc32),
        "fixtures/minicpm4_residual_lm.py changed after making the golden; {regenerate}"
    );
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn residual_lm_on_the_real_weights_runs_and_decodes_as_upstream() {
    let (dir, device) = (checkpoint_dir(), device());
    let json = config_json(&dir);
    let lm: Config = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let config = lm.residual_lm(
        count(&json["residual_lm_num_layers"]),
        json["residual_lm_no_rope"].as_bool().unwrap(),
    );
    let weights = dir.join("model.safetensors");
    let model = Model::load(&config, weights, "residual_lm.", FloatDType::F32, &device).unwrap();

    let golden = fixture!("minicpm4_residual_lm");
    let x = golden.tensor::<3>("input.residual_lm.x", &device);
    let expected = golden.tensor::<3>("output.residual_lm.causal", &device);
    let causal = model.forward(x.clone(), Mask::Causal);
    assert_close(causal, expected.clone(), ATOL, RTOL);

    let mut cache = model.cache(2, 8);
    let mut decoded = vec![model.forward_cached(x.clone().narrow(1, 0, 5), &mut cache)];
    for t in 5..8 {
        decoded.push(model.forward_cached(x.clone().narrow(1, t, 1), &mut cache));
    }
    assert_close(Tensor::cat(decoded, 1), expected, ATOL, RTOL);
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn every_stack_loads_from_the_real_checkpoint_and_runs_in_bf16() {
    let (dir, device) = (checkpoint_dir(), device());
    let json = config_json(&dir);
    let lm: Config = serde_json::from_value(json["lm_config"].clone()).unwrap();
    let local = |stack: &Value| {
        lm.local(&LocalShape {
            hidden_dim: count(&stack["hidden_dim"]),
            ffn_dim: count(&stack["ffn_dim"]),
            num_heads: count(&stack["num_heads"]),
            num_layers: count(&stack["num_layers"]),
            kv_channels: stack["kv_channels"].as_u64().map(|n| n as usize),
        })
    };
    let residual_lm = lm.residual_lm(
        count(&json["residual_lm_num_layers"]),
        json["residual_lm_no_rope"].as_bool().unwrap(),
    );
    let stacks = [
        (lm.clone(), "base_lm.", Mask::Causal),
        (residual_lm, "residual_lm.", Mask::Causal),
        (
            local(&json["encoder_config"]),
            "feat_encoder.encoder.",
            Mask::Bidirectional,
        ),
        (
            local(&json["dit_config"]),
            "feat_decoder.estimator.decoder.",
            Mask::Bidirectional,
        ),
    ];
    let weights = dir.join("model.safetensors");
    for (config, prefix, mask) in stacks {
        let model = Model::load(&config, &weights, prefix, FloatDType::BF16, &device)
            .unwrap_or_else(|e| panic!("{prefix}: {e}"));
        let x = Tensor::<3>::ones([1, 3, config.hidden_size], (&device, DType::BF16));
        let out = model.forward(x, mask);
        assert_eq!(out.dtype(), DType::BF16, "{prefix}");
        let out: Vec<f32> = out.into_data().iter::<f32>().collect();
        assert!(out.iter().all(|x| x.is_finite()), "{prefix}");
    }
}
