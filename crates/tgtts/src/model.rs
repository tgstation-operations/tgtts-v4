//! VoxCPM2 as the service's [`Synthesizer`].

use std::{fs, path::Path};

use burn::tensor::Device;
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
    /// `device`, to synthesize with `params`.
    pub fn load(
        dir: &Path,
        dtypes: &Dtypes,
        device: &Device,
        params: Params,
    ) -> Result<Self, voxcpm::Error> {
        let path = dir.join("config.json");
        let json = fs::read_to_string(&path).map_err(|e| voxcpm::Error::Io(path, e))?;
        let config = Config::from_json(&json)?;
        let hop: usize = config.audio_vae().decoder_rates.iter().product();
        Ok(Self {
            tts: TextToSpeech::load(dir, dtypes, device)?,
            params,
            samples_per_patch: config.patch_size * hop,
        })
    }
}

impl Synthesizer for VoxCpm {
    fn sample_rate(&self) -> u32 {
        self.tts.out_sample_rate()
    }

    fn synthesize(&self, text: &str, voice: &Latent, seed: u64) -> Result<Speech, synth::Error> {
        let mut noise = SeededNoise::new(seed);
        let samples = self.tts.synthesize(text, voice, &self.params, &mut noise)?;
        Ok(Speech {
            patches: samples.len() / self.samples_per_patch,
            samples,
        })
    }
}
