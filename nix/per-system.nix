{ pkgs, llvm-project }:
let
  rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
  # The game's voice filters use rubberband, which only ffmpeg-full builds in by default.
  ffmpeg = pkgs.ffmpeg-headless.override { withRubberband = true; };
  mlir = import ./mlir.nix { inherit pkgs llvm-project; };
  gpuKernels = import ./gpu-kernels.nix { inherit pkgs mlir; };
  glslKernels = import ./glsl-kernels.nix { inherit pkgs; };

  scripts = rec {
    fmt = import ./scripts/fmt.nix { inherit pkgs rustToolchain; };
    audit = import ./scripts/audit.nix {
      inherit
        pkgs
        rustToolchain
        fmt
        ffmpeg
        ;
    };
  };
in
{
  # fixtures stays out of the devshell, whose closure would then hold torch, and device-tests,
  # which would make entering it build LLVM where the store does not have it.
  packages = scripts // {
    inherit mlir;
    gpu-kernels = gpuKernels;
    glsl-kernels = glslKernels;
    device-tests = import ./scripts/device-tests.nix {
      inherit
        pkgs
        rustToolchain
        gpuKernels
        glslKernels
        ;
    };
    fixtures = import ./scripts/fixtures.nix { inherit pkgs ffmpeg; };
    tgtts = import ./tgtts.nix {
      inherit
        pkgs
        rustToolchain
        ffmpeg
        gpuKernels
        glslKernels
        ;
    };
    vk-trial = import ./vk-trial.nix {
      inherit
        pkgs
        rustToolchain
        gpuKernels
        glslKernels
        ;
    };
    rt-trial = import ./rt-trial.nix {
      inherit
        pkgs
        rustToolchain
        ffmpeg
        gpuKernels
        glslKernels
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
