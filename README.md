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

Entering the devshell points `core.hooksPath` at `.githooks/`. The pre-commit
hook runs the gate over the staged tree, and post-checkout seeds a new
worktree's `target/` as a copy-on-write clone of the main checkout's.
