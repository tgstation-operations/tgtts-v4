use voxcpm_core::AudioVaeConfig;

use crate::Error;

/// What building an encoder or a decoder refuses of an [`AudioVaeConfig`]: a trait, the config
/// being `voxcpm_core`'s.
pub(crate) trait Check {
    /// Refuses what the encoder needs and this port does not implement, or what cannot be
    /// built at all.
    fn check_encoder(&self) -> Result<(), Error>;

    /// Refuses what the decoder needs and this port does not implement, or what cannot be
    /// built at all, and gives the sample-rate bucket boundaries.
    fn check_decoder(&self) -> Result<&[u32], Error>;
}

impl Check for AudioVaeConfig {
    fn check_encoder(&self) -> Result<(), Error> {
        check_shared(self)?;
        check_rates("encoder_rates", &self.encoder_rates)?;
        if self.encoder_dim == 0 {
            return Err(Error::Unsupported("encoder_dim: 0".into()));
        }
        Ok(())
    }

    fn check_decoder(&self) -> Result<&[u32], Error> {
        check_shared(self)?;
        check_rates("decoder_rates", &self.decoder_rates)?;
        if self.decoder_dim >> self.decoder_rates.len() == 0 {
            return Err(Error::Unsupported(format!(
                "decoder_dim: {} cannot be halved {} times",
                self.decoder_dim,
                self.decoder_rates.len()
            )));
        }
        if self.use_noise_block {
            return Err(Error::Unsupported("use_noise_block: true".into()));
        }
        if !matches!(self.cond_type.as_str(), "scale_bias" | "scale_bias_init") {
            return Err(Error::Unsupported(format!(
                "cond_type: {:?}",
                self.cond_type
            )));
        }
        if self.cond_out_layer {
            return Err(Error::Unsupported("cond_out_layer: true".into()));
        }
        let Some(boundaries) = &self.sr_bin_boundaries else {
            return Err(Error::Unsupported("sr_bin_boundaries: null".into()));
        };
        if !boundaries.is_sorted() {
            return Err(Error::Unsupported(format!(
                "sr_bin_boundaries: {boundaries:?} is not ascending"
            )));
        }
        Ok(boundaries)
    }
}

fn check_shared(config: &AudioVaeConfig) -> Result<(), Error> {
    if !config.depthwise {
        return Err(Error::Unsupported("depthwise: false".into()));
    }
    if config.latent_dim == 0 {
        return Err(Error::Unsupported("latent_dim: 0".into()));
    }
    Ok(())
}

fn check_rates(field: &str, rates: &[usize]) -> Result<(), Error> {
    if rates.is_empty() || rates.contains(&0) {
        return Err(Error::Unsupported(format!("{field}: {rates:?}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKPOINT: &str = r#"{
        "encoder_dim": 128,
        "encoder_rates": [2, 5, 8, 8],
        "latent_dim": 64,
        "decoder_dim": 2048,
        "decoder_rates": [8, 6, 5, 2, 2, 2],
        "sr_bin_boundaries": [20000, 30000, 40000],
        "sample_rate": 16000,
        "out_sample_rate": 48000
    }"#;

    fn with(field: &str, value: &str) -> Result<AudioVaeConfig, serde_json::Error> {
        let mut json: serde_json::Value = serde_json::from_str(CHECKPOINT).unwrap();
        json[field] = serde_json::from_str(value).unwrap();
        serde_json::from_value(json)
    }

    fn unsupported<T: std::fmt::Debug>(result: Result<T, Error>) -> String {
        match result {
            Err(Error::Unsupported(message)) => message,
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_config_is_upstreams_default() {
        let config: AudioVaeConfig = serde_json::from_str(CHECKPOINT).unwrap();
        assert_eq!(config, AudioVaeConfig::default());
        config.check_encoder().unwrap();
        assert_eq!(config.check_decoder().unwrap(), [20000, 30000, 40000]);
    }

    #[test]
    fn unknown_field_is_refused() {
        let error = with("noise_scale", "0.5").unwrap_err().to_string();
        assert!(error.starts_with("unknown field `noise_scale`"), "{error}");
    }

    #[test]
    fn unimplemented_values_are_refused() {
        let cases = [
            ("depthwise", "false", "depthwise: false"),
            ("use_noise_block", "true", "use_noise_block: true"),
            ("cond_type", r#""concat""#, r#"cond_type: "concat""#),
            ("cond_out_layer", "true", "cond_out_layer: true"),
            ("sr_bin_boundaries", "null", "sr_bin_boundaries: null"),
            (
                "sr_bin_boundaries",
                "[30000, 20000]",
                "sr_bin_boundaries: [30000, 20000] is not ascending",
            ),
            ("decoder_rates", "[]", "decoder_rates: []"),
            (
                "decoder_dim",
                "32",
                "decoder_dim: 32 cannot be halved 6 times",
            ),
        ];
        for (field, value, message) in cases {
            let config = with(field, value).unwrap();
            assert_eq!(unsupported(config.check_decoder()), message);
        }
        let cases = [
            ("depthwise", "false", "depthwise: false"),
            ("encoder_rates", "[2, 0]", "encoder_rates: [2, 0]"),
            ("encoder_dim", "0", "encoder_dim: 0"),
        ];
        for (field, value, message) in cases {
            let config = with(field, value).unwrap();
            assert_eq!(unsupported(config.check_encoder()), message);
        }
    }

    #[test]
    fn scale_bias_init_conditions_as_scale_bias_does() {
        with("cond_type", r#""scale_bias_init""#)
            .unwrap()
            .check_decoder()
            .unwrap();
    }
}
