# tts-rs

Text-to-speech for /tg/station, in Rust, to replace the Python services the
game's TTS subsystem talks to today.

## Working on it

Every tool comes from the Nix flake: `direnv allow` once, or `nix develop`.

- `nix run .#fmt` formats the tree: rustfmt, nixfmt and ruff.
- `nix run .#audit` is the gate: formatting, ASCII, the Nix lints, clippy with
  warnings denied, rustdoc and the full test suite.
- `nix run .#fixtures [NAME...]` runs the generators `fixtures/<NAME>.py`, all
  of them by default, which write what upstream VoxCPM computes for the tests to
  compare against. A rerun writes the same bytes.
- Tests that need the real weights are ignored by default: point `VOXCPM2_DIR`
  at a VoxCPM2 checkpoint and run `cargo test -- --ignored`.
- `nix run .#bench` times release builds of the hot paths on the real data
  (`VOXCPM2_DIR`, `VOICES_DIR` for a converted voices directory and `ASSETS_DIR`
  for the radio's sounds) and writes each run as JSON;
  `nix run .#bench -- --help` tells how to compare two.

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
times faster. Point the game's `TTS_HTTP_URL` at `http://127.0.0.1:5002` with
the token as `TTS_HTTP_TOKEN`. The service runs ffmpeg for the filter strings
the game sent before effects, which needs rubberband, as the devshell's has.
The end-to-end test runs on the real model with `VOXCPM2_DIR`, `VOICES_DIR` and
`ASSETS_DIR` set:
`cargo test --release -p tgtts --test real -- --ignored`.

Entering the devshell points `core.hooksPath` at `.githooks/`. The pre-commit
hook runs the gate over the staged tree, and post-checkout seeds a new
worktree's `target/` as a copy-on-write clone of the main checkout's.
