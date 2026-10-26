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
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import ./nix/per-system.nix {
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
