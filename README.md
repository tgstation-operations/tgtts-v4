# tts-rs

Text-to-speech for /tg/station, in Rust, to replace the Python services the
game's TTS subsystem talks to today.

## Working on it

Every tool comes from the Nix flake: `direnv allow` once, or `nix develop`.

- `nix run .#fmt` formats the tree: rustfmt and nixfmt.
- `nix run .#audit` is the gate: formatting, ASCII, the Nix lints, clippy with
  warnings denied, rustdoc and the full test suite.

Entering the devshell points `core.hooksPath` at `.githooks/`. The pre-commit
hook runs the gate over the staged tree, and post-checkout seeds a new
worktree's `target/` as a copy-on-write clone of the main checkout's.
