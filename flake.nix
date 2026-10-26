{
  description = "Text-to-speech for /tg/station";

  inputs = {
    # The release the servers run, so a deployment that makes this follow its
    # own nixpkgs moves nothing.
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # The LLVM commit vendor/mlir-aie and vendor/mlir-air name in their
    # utils/clone-llvm.sh, from the fork they clone it from: ROCm's amd-staging
    # carries it and upstream's main does not. The gate's vendor-pins stage
    # holds the three together.
    llvm-project = {
      url = "github:ROCm/llvm-project/bdf04ebb32d7a5dce853f9b399dcbd3ed11a5aa7";
      flake = false;
    };
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      llvm-project,
      ...
    }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import ./nix/per-system.nix {
              inherit llvm-project;
              pkgs = import nixpkgs {
                inherit system;
                overlays = [ rust-overlay.overlays.default ];
                # The CUDA a GPU build runs with: NVRTC, the headers it compiles
                # kernels with, and libdevice, which comes with nvcc.
                config.allowUnfreePredicate =
                  pkg:
                  builtins.elem (nixpkgs.lib.getName pkg) [
                    "cuda_cccl"
                    "cuda_cudart"
                    "cuda_nvcc"
                    "cuda_nvrtc"
                  ];
              };
            }
          )
        );
    in
    {
      packages = forAllSystems (s: s.packages);
      devShells = forAllSystems (s: {
        default = s.devShell;
      });
    };
}
