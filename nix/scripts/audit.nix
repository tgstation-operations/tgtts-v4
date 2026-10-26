{
  pkgs,
  rustToolchain,
  fmt,
  ffmpeg,
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
    pkgs.pkg-config
    pkgs.statix
    # The service's tests run it.
    ffmpeg
  ];
  text = ''
    # A copy of the gate on a shell's PATH goes stale when this file changes;
    # the store path shows which one ran.
    echo "audit: $0" >&2
    cd "$(git rev-parse --show-toplevel)"
    export CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true CARGO_BUILD_FINGERPRINT=content \
      CARGO_UNSTABLE_RUSTDOC_DEPINFO=true
    # The C libraries the devshell's buildInputs give, which a bare `nix run` lacks.
    export PKG_CONFIG_PATH=${pkgs.lib.getDev pkgs.soxr}/lib/pkgconfig

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
    # Each crate's default features, those of the CPU: a GPU feature compiles
    # wgpu into every crate it reaches, and its tests need the GPU. The GPU
    # code is linted on its own.
    stage clippy cargo clippy --locked --workspace --all-targets -- -D warnings
    stage clippy-vulkan cargo clippy --locked -p gpu -p tgtts --all-targets \
      --features gpu/vulkan,tgtts/vulkan -- -D warnings
    stage rustdoc env RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
    stage test cargo test --locked --workspace

    if [ "''${#failed[@]}" -gt 0 ]; then
      echo "audit: FAILED: ''${failed[*]}" >&2
      exit 1
    fi
    echo "audit: passed" >&2
  '';
}
