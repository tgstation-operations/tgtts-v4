{ pkgs }:
let
  rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;

  scripts = rec {
    fmt = import ./scripts/fmt.nix { inherit pkgs rustToolchain; };
    audit = import ./scripts/audit.nix { inherit pkgs rustToolchain fmt; };
  };
in
{
  packages = scripts;
  devShell = import ./devshell.nix { inherit pkgs rustToolchain scripts; };
}
