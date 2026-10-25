{
  pkgs,
  rustToolchain,
  fmt,
}:
pkgs.writeShellApplication {
  name = "audit";
  runtimeInputs = [
    fmt
    rustToolchain
    # cargo links through cc, which a bare `nix run` shell does not have.
    pkgs.stdenv.cc
    pkgs.coreutils
    pkgs.deadnix
    pkgs.findutils
    pkgs.git
    pkgs.gnugrep
    pkgs.statix
  ];
  text = ''
    # A copy of the gate on a shell's PATH goes stale when this file changes;
    # the store path shows which one ran.
    echo "audit: $0" >&2
    cd "$(git rev-parse --show-toplevel)"
    export CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true CARGO_BUILD_FINGERPRINT=content \
      CARGO_UNSTABLE_RUSTDOC_DEPINFO=true

    failed=()
    stage() {
      local name=$1
      shift
      echo "==> $name" >&2
      "$@" || failed+=("$name")
    }

    # git runs a hook only if it is executable, and skips it silently if not.
    hooks() {
      ! git ls-files -s .githooks | grep -v '^100755 '
    }
    ascii() {
      ! LC_ALL=C git grep -I -n -P '[\x80-\xFF]'
    }
    nix_files() {
      git ls-files -z '*.nix'
    }
    deadnix_files() {
      nix_files | xargs -0 deadnix --fail
    }
    statix_files() {
      local file status=0
      while IFS= read -r -d "" file; do
        statix check "$file" || status=1
      done < <(nix_files)
      return "$status"
    }

    stage hooks hooks
    stage fmt fmt --check
    stage ascii ascii
    stage deadnix deadnix_files
    stage statix statix_files
    stage clippy cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    stage rustdoc env RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
    stage test cargo test --locked --workspace --all-features

    if [ "''${#failed[@]}" -gt 0 ]; then
      echo "audit: FAILED: ''${failed[*]}" >&2
      exit 1
    fi
    echo "audit: passed" >&2
  '';
}
