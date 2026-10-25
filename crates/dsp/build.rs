fn main() {
    // The resample module calls libsoxr.
    if let Err(e) = pkg_config::Config::new()
        .atleast_version("0.1.3")
        .probe("soxr")
    {
        panic!("{e}");
    }
}
