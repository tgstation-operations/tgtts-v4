# tts-rs

Text-to-speech for /tg/station, in Rust, to replace the Python services the
game's TTS subsystem talks to today.

## Working on it

Every tool comes from the Nix flake: `direnv allow` once, or `nix develop`.

- `nix run .#fmt` formats the tree: rustfmt, nixfmt and ruff.
- `nix run .#audit` is the gate: formatting, ASCII, the Nix lints, the vendored
  upstreams' pins, that tgtts builds nothing of burn, clippy with warnings
  denied, rustdoc and the full test suite.
- `nix run .#fixtures [NAME...]` runs the generators `fixtures/<NAME>.py`, all
  of them by default, which write what upstream VoxCPM computes for the tests to
  compare against. A rerun writes the same bytes.
- Tests that need the real weights are ignored by default: point `VOXCPM2_DIR`
  at a VoxCPM2 checkpoint and run `cargo test -- --ignored`.
- The gate runs nothing on a GPU.
- `nix run .#device-tests` runs the Vulkan runtime's tests on the GPU, under
  the Khronos validation layer, the start of a GPU path of its own; the burn
  code stays the oracle it is checked against. Its kernels are MLIR compiled
  ahead of time to SPIR-V, from `kernels/gpu`: `nix build .#gpu-kernels`, which
  builds LLVM first where the store does not have it. The decode step's kernels
  are GLSL, from `kernels/glsl`: `nix build .#glsl-kernels`. With
  `-- --include-ignored` and `VOXCPM2_DIR`, the `graph` crate's decode step runs
  on the real weights against minicpm4 on the CPU.
- `vendor/` pins the upstreams the GPU kernels and their toolchain follow, as
  git submodules that nothing builds from; `vendor/README.md` lists them.

## Running the service

`tgtts` serves the routes the game calls, in one process: the model, the blips,
the radio and the game's voice effects, or the ffmpeg filters it sent before
them. It reads a TOML file named as its argument, and `TGTTS_<KEY>` in the
environment overrides any key; the keys are listed in
`crates/tgtts/src/config.rs`. The token is read from a file, so that a secret
manager can provide it. The model runs on tts-rs's own runtime on a Vulkan GPU.
To serve locally, write a `tgtts.toml` such as

```toml
token_file = "token"
voices_dir = "/path/to/voices"
assets_dir = "/path/to/assets"
checkpoint_dir = "/path/to/VoxCPM2"
cache_dir = "cache"
metrics_db = "metrics.sqlite"
glsl_kernels_dir = "/path/to/glsl-kernels"
mlir_kernels_dir = "/path/to/gpu-kernels"
```

with the kernels `nix build .#glsl-kernels` and `nix build .#gpu-kernels` build,
and run `cargo run --release -p tgtts -- tgtts.toml`. `/health-check` answers 503
until the model has loaded and warmed up. `nix build .#tgtts` builds the service
as the servers run it, its wrapper naming the kernels and ffmpeg.
Point the game's `TTS_HTTP_URL` at `http://127.0.0.1:5002` with
the token as `TTS_HTTP_TOKEN`. The service runs ffmpeg for the filter strings
the game sent before effects, which needs rubberband, as the devshell's has.
The end-to-end tests run on the GPU with `VOXCPM2_DIR`, `VOICES_DIR`,
`ASSETS_DIR` and `GOLDEN_DIR` set and `GLSL_KERNELS` and `VULKAN_KERNELS` naming
the kernels: `cargo test --release -p tgtts --test rt -- --ignored`.

Entering the devshell points `core.hooksPath` at `.githooks/`. The pre-commit
hook runs the gate over the staged tree, and post-checkout seeds a new
worktree's `target/` as a copy-on-write clone of the main checkout's.
