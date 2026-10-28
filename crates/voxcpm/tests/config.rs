use audiovae::AudioVaeConfig;
use serde_json::{Value, json};
use testkit::checkpoint_dir;
use voxcpm::{CfmConfig, Config};
use voxcpm_core::Error;

fn tiny() -> Value {
    serde_json::from_str(include_str!("fixtures/voxcpm.json")).unwrap()
}

fn parse(json: &Value) -> Result<Config, Error> {
    Config::from_json(&json.to_string())
}

#[test]
fn stacks_are_derived_from_the_base_lm_as_upstream_derives_them() {
    let config = parse(&tiny()).unwrap();
    let lm = &config.lm_config;

    let residual_lm = config.residual_lm();
    assert_eq!(residual_lm.num_hidden_layers, 1);
    assert!(residual_lm.no_rope);
    assert_eq!(residual_lm.vocab_size, 0);
    assert_eq!(residual_lm.hidden_size, lm.hidden_size);

    let encoder = config.local_encoder();
    let dit = config.local_dit();
    for (stack, ffn_dim) in [(&encoder, 24), (&dit, 20)] {
        assert_eq!(stack.hidden_size, 16);
        assert_eq!(stack.intermediate_size, ffn_dim);
        assert_eq!(stack.num_attention_heads, 4);
        assert_eq!(stack.num_hidden_layers, 2);
        assert_eq!(stack.kv_channels, Some(8));
        assert_eq!(stack.vocab_size, 0);
        assert!(!stack.no_rope);
        assert_eq!(stack.rope_scaling, lm.rope_scaling);
    }

    let vae = config.audio_vae();
    assert_eq!((vae.latent_dim, vae.decoder_dim), (8, 16));
}

#[test]
fn missing_fields_take_upstreams_defaults() {
    let json = json!({
        "lm_config": tiny()["lm_config"],
        "encoder_config": {},
        "dit_config": {"cfm_config": {}},
    });
    let config = parse(&json).unwrap();
    assert_eq!(config.architecture, None);
    assert_eq!((config.patch_size, config.feat_dim), (4, 64));
    assert_eq!(config.residual_lm_num_layers, 8);
    assert!(!config.residual_lm_no_rope);
    assert_eq!(config.scalar_quantization_latent_dim, 512);
    assert_eq!(config.scalar_quantization_scale, 9);
    assert_eq!(config.max_length, 8192);
    assert_eq!((&*config.device, &*config.dtype), ("cuda", "bfloat16"));
    assert_eq!(config.audio_vae(), AudioVaeConfig::default());
    let encoder = &config.encoder_config;
    let shape = (encoder.hidden_dim, encoder.ffn_dim, encoder.num_heads);
    assert_eq!(
        (shape, encoder.num_layers, encoder.kv_channels),
        ((1024, 4096, 16), 4, None)
    );
    let dit = &config.dit_config;
    let shape = (dit.hidden_dim, dit.ffn_dim, dit.num_heads);
    assert_eq!(
        (shape, dit.num_layers, dit.kv_channels),
        ((1024, 4096, 16), 4, None)
    );
    assert!(!dit.dit_mean_mode && !dit.mean_mode);
    let cfm = &dit.cfm_config;
    assert_eq!(*cfm, CfmConfig::default());
    assert_eq!((cfm.sigma_min, &*cfm.solver), (1e-6, "euler"));
    assert_eq!(cfm.inference_cfg_rate, 1.0);
}

#[test]
fn null_audio_vae_config_is_upstreams_default() {
    let mut json = tiny();
    json["audio_vae_config"] = Value::Null;
    json["feat_dim"] = json!(64);
    assert_eq!(parse(&json).unwrap().audio_vae(), AudioVaeConfig::default());
}

#[test]
fn unknown_fields_are_refused() {
    let cases = [
        ("", "dit_mean_mode"),
        ("/encoder_config", "num_kv_heads"),
        ("/dit_config", "concat_hidden"),
        ("/dit_config/cfm_config", "sway_sampling_coef"),
    ];
    for (pointer, field) in cases {
        let mut json = tiny();
        json.pointer_mut(pointer).unwrap()[field] = json!(true);
        let error = parse(&json).unwrap_err();
        assert!(matches!(error, Error::Config(_)), "{field}: {error}");
        let expected = format!("unknown field `{field}`");
        assert!(error.to_string().contains(&expected), "{error}");
    }
}

#[test]
fn unimplemented_values_are_refused() {
    let cases = [
        ("/architecture", json!("voxcpm"), "architecture \"voxcpm\""),
        (
            "/dit_config/dit_mean_mode",
            json!(true),
            "dit_config.dit_mean_mode: true",
        ),
        (
            "/dit_config/mean_mode",
            json!(true),
            "dit_config.mean_mode: true",
        ),
        (
            "/dit_config/cfm_config/solver",
            json!("heun"),
            "solver \"heun\"",
        ),
        (
            "/feat_dim",
            json!(64),
            "feat_dim 64 where the audio VAE's latents have 8 channels",
        ),
    ];
    for (pointer, value, message) in cases {
        let mut json = tiny();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        json.pointer_mut(parent).unwrap()[field] = value;
        match parse(&json) {
            Err(Error::Unsupported(what)) => assert_eq!(what, message),
            other => panic!("{pointer}: expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn the_checkpoints_config_reads() {
    let json = std::fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let config = Config::from_json(&json).unwrap();
    assert_eq!(config.architecture.as_deref(), Some("voxcpm2"));
    assert_eq!((config.patch_size, config.feat_dim), (4, 64));
    assert_eq!(config.local_dit().num_hidden_layers, 12);
    assert_eq!(config.residual_lm().num_hidden_layers, 8);
    assert_eq!(config.audio_vae(), AudioVaeConfig::default());
}
