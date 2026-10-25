use audiovae::{AudioVaeConfig, Weights};
use burn::store::SafetensorsStore;
use testkit::fixture;

/// The config fixtures/audiovae.py builds its decoder and encoder from.
pub fn tiny() -> AudioVaeConfig {
    AudioVaeConfig {
        encoder_dim: 2,
        encoder_rates: vec![2, 5],
        latent_dim: 4,
        decoder_dim: 32,
        decoder_rates: vec![3, 2, 2],
        out_sample_rate: 24000,
        ..AudioVaeConfig::default()
    }
}

pub fn weights() -> Weights {
    let mut store = SafetensorsStore::from_file(fixture!("audiovae").path());
    Weights::from_store(&mut store).unwrap()
}
