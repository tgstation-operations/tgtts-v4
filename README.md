# tts-rs

Text-to-speech for /tg/station, in Rust, to replace the Python services the
game's TTS subsystem talks to today.

## Working on it

Every tool comes from the Nix flake: `direnv allow` once, or `nix develop`.

- `nix run .#fmt` formats the tree: rustfmt, nixfmt and ruff.
- `nix run .#audit` is the gate: formatting, ASCII, the Nix lints, the vendored
  upstreams' pins, clippy with warnings denied, rustdoc and the full test suite.
- `nix run .#fixtures [NAME...]` runs the generators `fixtures/<NAME>.py`, all
  of them by default, which write what upstream VoxCPM computes for the tests to
  compare against. A rerun writes the same bytes.
- Tests that need the real weights are ignored by default: point `VOXCPM2_DIR`
  at a VoxCPM2 checkpoint and run `cargo test -- --ignored`.
- The gate runs nothing on a GPU. On a Vulkan GPU, `cargo test -p gpu
  --features vulkan` runs the models' fixtures there, and with
  `-- --include-ignored` their goldens on the real weights and the service end
  to end, which also need `VOICES_DIR` and `ASSETS_DIR`.
- `nix run .#device-tests` runs the Vulkan runtime's tests on the GPU, under
  the Khronos validation layer, the start of a GPU path of its own; the burn
  code stays the oracle it is checked against. Its kernels are MLIR compiled
  ahead of time to SPIR-V, from `kernels/gpu`: `nix build .#gpu-kernels`, which
  builds LLVM first where the store does not have it. The decode step's kernels
  are GLSL, from `kernels/glsl`: `nix build .#glsl-kernels`.
- `nix run .#bench` times release builds of the hot paths on the real data
  (`VOXCPM2_DIR`, `VOICES_DIR` for a converted voices directory and `ASSETS_DIR`
  for the radio's sounds) and writes each run as JSON;
  `nix run .#bench -- --help` tells how to compare two, and how to time a GPU.
- `vendor/` pins the upstreams the GPU kernels and their toolchain follow, as
  git submodules that nothing builds from; `vendor/README.md` lists them.

## Running the service

`tgtts` serves the routes the game calls, in one process: the model, the blips,
the radio and the game's voice effects, or the ffmpeg filters it sent before
them. It reads a TOML file named as its argument, and `TGTTS_<KEY>` in the
environment overrides any key; the keys are listed in
`crates/tgtts/src/config.rs`. The token is read from a file, so that a secret
manager can provide it. To serve locally, write a `tgtts.toml` such as

```toml
token_file = "token"
voices_dir = "/path/to/voices"
assets_dir = "/path/to/assets"
checkpoint_dir = "/path/to/VoxCPM2"
cache_dir = "cache"
metrics_db = "metrics.sqlite"
```

and run `cargo run --release -p tgtts -- tgtts.toml`, adding
`--features burn/rayon,burn/simd` on the CPU, where they make the model several
times faster. On a Vulkan GPU, build with `--features vulkan` and add

```toml
device = "vulkan"
autotune_cache_dir = "autotune"
```

where autotune keeps the kernels it chose. `/health-check` answers 503 until the
model has loaded and warmed up: a minute after the first start, half a minute
once autotune has kept its choices.
Point the game's `TTS_HTTP_URL` at `http://127.0.0.1:5002` with
the token as `TTS_HTTP_TOKEN`. The service runs ffmpeg for the filter strings
the game sent before effects, which needs rubberband, as the devshell's has.
The end-to-end test runs on the real model with `VOXCPM2_DIR`, `VOICES_DIR` and
`ASSETS_DIR` set:
`cargo test --release -p tgtts --test real -- --ignored`.

Entering the devshell points `core.hooksPath` at `.githooks/`. The pre-commit
hook runs the gate over the staged tree, and post-checkout seeds a new
worktree's `target/` as a copy-on-write clone of the main checkout's.
