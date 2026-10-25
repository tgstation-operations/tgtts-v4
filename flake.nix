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
