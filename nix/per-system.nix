{ pkgs }:
let
  rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
  # The game's voice filters use rubberband, which only ffmpeg-full builds in by default.
  ffmpeg = pkgs.ffmpeg-headless.override { withRubberband = true; };
  tracelLlvm = import ./tracel-llvm.nix { inherit pkgs; };

  scripts = rec {
    fmt = import ./scripts/fmt.nix { inherit pkgs rustToolchain; };
    audit = import ./scripts/audit.nix {
      inherit
        pkgs
        rustToolchain
        fmt
        ffmpeg
        tracelLlvm
        ;
    };
    bench = import ./scripts/bench.nix { inherit pkgs rustToolchain; };
  };
in
{
  # fixtures stays out of the devshell, whose closure would then hold torch.
  packages = scripts // {
    fixtures = import ./scripts/fixtures.nix { inherit pkgs ffmpeg; };
    cuda-trial = import ./cuda-trial.nix {
      inherit
        pkgs
        rustToolchain
        ffmpeg
        tracelLlvm
        ;
    };
  };
  devShell = import ./devshell.nix {
    inherit
      pkgs
      rustToolchain
      scripts
      ffmpeg
      ;
  };
}
