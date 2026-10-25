use voxcpm::Config;

/// The tiny config the fixtures were generated at.
pub fn tiny() -> Config {
    Config::from_json(include_str!("../fixtures/voxcpm.json")).unwrap()
}
