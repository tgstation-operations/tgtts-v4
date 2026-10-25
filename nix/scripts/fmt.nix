{ pkgs, rustToolchain }:
pkgs.writeShellApplication {
  name = "fmt";
  runtimeInputs = [
    pkgs.findutils
    pkgs.git
    pkgs.nixfmt
    rustToolchain
  ];
  text = ''
    check=()
    case "''${1:-}" in
      "") ;;
      --check) check=(--check) ;;
      *)
        echo "usage: fmt [--check]" >&2
        exit 2
        ;;
    esac

    cd "$(git rev-parse --show-toplevel)"
    status=0
    cargo fmt --all -- "''${check[@]}" || status=1
    git ls-files -z '*.nix' | xargs -0 -r nixfmt "''${check[@]}" || status=1
    exit "$status"
  '';
}
