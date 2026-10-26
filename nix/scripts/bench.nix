{ pkgs, rustToolchain }:
pkgs.writeShellApplication {
  name = "bench";
  runtimeInputs = [
    rustToolchain
    # cargo links through cc, which a bare `nix run` shell does not have.
    pkgs.stdenv.cc
    pkgs.coreutils
    pkgs.git
    pkgs.gnugrep
    pkgs.gnused
    pkgs.pkg-config
  ];
  text = ''
    # bench [CARGO_ARGS... --] [BENCH_ARGS...]: arguments before a `--` go to cargo, such as
    # `--features burn/rayon`, which tries one of burn's features without changing the
    # workspace's.
    cd "$(git rev-parse --show-toplevel)"
    # As the devshell sets them, or alternating with it would rebuild everything each time.
    export CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true CARGO_BUILD_FINGERPRINT=content \
      CARGO_UNSTABLE_RUSTDOC_DEPINFO=true
    # The C libraries the devshell's buildInputs give, which a bare `nix run` lacks.
    export PKG_CONFIG_PATH=${pkgs.lib.getDev pkgs.soxr}/lib/pkgconfig
    # As the devshell has it: wgpu opens the Vulkan loader at run time.
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    cargo_args=()
    for arg; do
      if [ "$arg" = -- ]; then
        while [ "$1" != -- ]; do
          cargo_args+=("$1")
          shift
        done
        shift
        break
      fi
    done

    # cargo tree takes neither a profile nor --release.
    profile=(--release)
    tree_args=()
    skip=false
    for arg in "''${cargo_args[@]}"; do
      if "$skip"; then
        skip=false
        continue
      fi
      case "$arg" in
        --profile) profile=() skip=true ;;
        --profile=* | --release) profile=() ;;
        *) tree_args+=("$arg") ;;
      esac
    done

    features=$(
      cargo tree --locked -p bench -e normal "''${tree_args[@]}" --prefix none -f '{p} [{f}]' |
        grep -E '^(burn-cubecl|burn-flex|gemm|gemm-common|pulp) ' | sed 's/ (\*)$//' | sort -u
    ) || features=unknown
    export BENCH_CARGO_ARGS="''${cargo_args[*]}" BENCH_RESOLVED_FEATURES="$features"
    exec cargo run --locked "''${profile[@]}" -p bench "''${cargo_args[@]}" -- "$@"
  '';
}
