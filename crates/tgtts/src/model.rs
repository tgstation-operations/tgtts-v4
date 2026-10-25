//! VoxCPM2 as the service's [`Synthesizer`].

use std::{fs, path::Path};

use burn::tensor::Device;
use dsp::RATE;
use voices::Latent;
use voxcpm::{Config, Dtypes, Params, SeededNoise, TextToSpeech};

use crate::synth::{self, Speech, Synthesizer};

/// VoxCPM2, synthesizing with the same parameters for every request.
#[derive(Debug)]
pub struct VoxCpm {
    tts: TextToSpeech,
    params: Params,
    samples_per_patch: usize,
}

impl VoxCpm {
    /// Loads the VoxCPM2 checkpoint in `dir`, each part in its dtype from `dtypes`, on
    /// `device`, to synthesize with `params`, refusing one whose audio is not at the service's
    /// rate before it loads the weights.
    pub fn load(
        dir: &Path,
        dtypes: &Dtypes,
        device: &Device,
        params: Params,
    ) -> Result<Self, voxcpm::Error> {
        let path = dir.join("config.json");
        let json = fs::read_to_string(&path).map_err(|e| voxcpm::Error::Io(path, e))?;
        let config = Config::from_json(&json)?;
        let rate = config.audio_vae().out_sample_rate;
        if rate != RATE {
            let reason = format!("out_sample_rate {rate} where the service works at {RATE} Hz");
            return Err(voxcpm::Error::Unsupported(reason));
        }
        let hop: usize = config.audio_vae().decoder_rates.iter().product();
        Ok(Self {
            tts: TextToSpeech::load(dir, dtypes, device)?,
            params,
            samples_per_patch: config.patch_size * hop,
        })
    }
}

impl Synthesizer for VoxCpm {
    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        let mut noise = SeededNoise::new(seed);
        let samples = self.tts.synthesize(text, voice, &self.params, &mut noise)?;
        Ok(Speech {
            patches: samples.len() / self.samples_per_patch,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::FloatDType;
    use serde_json::Value;

    use super::*;

    /// voxcpm's tiny config, at `rate`, as the checkpoint `dir` holds it, with nothing else.
    fn checkpoint(dir: &Path, rate: u32) {
        let json = include_str!("../../voxcpm/tests/fixtures/voxcpm.json");
        let mut config: Value = serde_json::from_str(json).unwrap();
        config["audio_vae_config"]["out_sample_rate"] = rate.into();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    fn load(dir: &Path) -> voxcpm::Error {
        let f32 = FloatDType::F32;
        let dtypes = Dtypes {
            lm: f32,
            local_encoder: f32,
            local_dit: f32,
            audio_decoder: f32,
        };
        VoxCpm::load(dir, &dtypes, &Device::flex(), Params::default()).unwrap_err()
    }

    #[test]
    fn refuses_a_model_at_another_rate() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(dir.path(), 44100);
        let error = load(dir.path()).to_string();
        assert!(error.contains("out_sample_rate 44100"), "{error}");
        // At the service's rate it goes on, to what the tiny config fails on in TextToSpeech.
        checkpoint(dir.path(), RATE);
        let error = load(dir.path()).to_string();
        assert!(!error.contains("out_sample_rate"), "{error}");
    }
}
