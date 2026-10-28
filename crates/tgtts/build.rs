fn main() {
    // The opus module calls libopus.
    if let Err(e) = pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("opus")
    {
        panic!("{e}");
    }
}
