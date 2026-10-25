use std::path::Path;

use testkit::Fixture;

/// The fixture `name` of the workspace's crate `krate`.
pub fn fixture(krate: &str, name: &str) -> Fixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(krate);
    Fixture::open(dir.join(format!("tests/fixtures/{name}.safetensors")))
}
