{ pkgs }:
let
  rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;

  scripts = rec {
    fmt = import ./scripts/fmt.nix { inherit pkgs rustToolchain; };
    audit = import ./scripts/audit.nix { inherit pkgs rustToolchain fmt; };
  };
in
{
  # fixtures stays out of the devshell, whose closure would then hold torch.
  packages = scripts // {
    fixtures = import ./scripts/fixtures.nix { inherit pkgs; };
  };
  devShell = import ./devshell.nix { inherit pkgs rustToolchain scripts; };
}
