{
  pkgs,
  rustToolchain,
  fmt,
  ffmpeg,
  tracelLlvm,
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
    pkgs.gnused
    pkgs.jq
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
    export PKG_CONFIG_PATH=${pkgs.lib.getDev pkgs.libopus}/lib/pkgconfig:${pkgs.lib.getDev pkgs.soxr}/lib/pkgconfig

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

    # A file of a vendored upstream at the commit this tree pins it to, from the objects of
    # whichever checkout of this repository fetched it, so a worktree and the pre-commit hook's
    # tree find the main checkout's. Fails, saying how to fetch it, rather than pass blind.
    vendored_file() {
      local path=$1 file=$2 sha dir common
      sha=$(git ls-files -s -- "$path" | cut -d' ' -f2)
      common=$(git rev-parse --path-format=absolute --git-common-dir)
      for dir in "$(git rev-parse --path-format=absolute --git-path "modules/$path")" \
        "$common/modules/$path" "$common"/worktrees/*/"modules/$path"; do
        if [ -n "$sha" ] && git --git-dir="$dir" cat-file -e "$sha^{commit}" 2>/dev/null; then
          git --git-dir="$dir" show "$sha:$file"
          return
        fi
      done
      echo "audit: $path is not fetched; run: git submodule update --init --filter=blob:none $path" >&2
      return 1
    }
    gitlink() {
      git ls-files -s -- "$1" | cut -d' ' -f2
    }
    check_pin() {
      local what=$1 want=$2 have=$3
      if [ -n "$want" ] && [ "$want" = "$have" ]; then
        echo "    $what: $have" >&2
      else
        echo "audit: $what disagree: the upstream names '$want', this tree pins '$have'" >&2
        return 1
      fi
    }
    # A Peano wheel's version names its llvm-aie commit by a hash prefix after the `+`, at least
    # seven digits, so that a mangled pin reads as empty and fails rather than matching.
    peano_pin() {
      sed -n 's/.*llvm-aie==[0-9.]*+\([0-9a-f]\{7,\}\).*/\1/p'
    }
    # The NPU's version set is mlir-air's, and the LLVM the kernels compile with is the one it
    # names: every check runs, so the verdict names each disagreement.
    vendor_pins() {
      local llvm aie peano aie_peano air_peano iron_aie iron_peano status=0
      llvm=$(jq -r '.nodes["llvm-project"].locked.rev' flake.lock)
      aie=$(gitlink vendor/mlir-aie)
      peano=$(gitlink vendor/llvm-aie)
      check_pin "llvm-project (flake.lock) and vendor/mlir-aie's LLVM_PROJECT_COMMIT" \
        "$(vendored_file vendor/mlir-aie utils/clone-llvm.sh | sed -n 's/^LLVM_PROJECT_COMMIT=//p')" \
        "$llvm" || status=1
      check_pin "llvm-project (flake.lock) and vendor/mlir-air's LLVM commithash" \
        "$(vendored_file vendor/mlir-air utils/clone-llvm.sh | sed -n 's/^export commithash=//p')" \
        "$llvm" || status=1
      check_pin "vendor/mlir-aie and vendor/mlir-air's mlir-aie HASH" \
        "$(vendored_file vendor/mlir-air utils/clone-mlir-aie.sh | sed -n 's/^export HASH=//p')" \
        "$aie" || status=1
      aie_peano=$(vendored_file vendor/mlir-aie utils/peano-requirements.txt | peano_pin)
      check_pin "vendor/llvm-aie and vendor/mlir-aie's Peano wheel" \
        "$aie_peano" "''${peano:0:''${#aie_peano}}" || status=1
      air_peano=$(vendored_file vendor/mlir-air utils/build-mlir-air-using-wheels.sh | peano_pin)
      check_pin "vendor/llvm-aie and vendor/mlir-air's Peano wheel" \
        "$air_peano" "''${peano:0:''${#air_peano}}" || status=1
      iron_aie=$(vendored_file vendor/IRON requirements.txt |
        sed -n 's/^mlir_aie==.*+g\([0-9a-f]\{7,\}\).*/\1/p')
      check_pin "vendor/mlir-aie and vendor/IRON's mlir_aie wheel" \
        "$iron_aie" "''${aie:0:''${#iron_aie}}" || status=1
      iron_peano=$(vendored_file vendor/IRON requirements.txt | peano_pin)
      check_pin "vendor/llvm-aie and vendor/IRON's Peano wheel" \
        "$iron_peano" "''${peano:0:''${#iron_peano}}" || status=1
      return "$status"
    }

    stage hooks hooks
    stage fmt fmt --check
    stage ascii ascii
    stage deadnix deadnix_files
    stage statix statix_files
    stage vendor-pins vendor_pins
    # Each crate's default features, those of the CPU: a GPU feature compiles
    # wgpu into every crate it reaches, and its tests need the GPU. The GPU
    # code is linted on its own.
    stage clippy cargo clippy --locked --workspace --all-targets -- -D warnings
    stage clippy-vulkan cargo clippy --locked -p gpu -p tgtts -p bench --all-targets \
      --features gpu/vulkan,tgtts/vulkan,tgtts/rt,bench/vulkan -- -D warnings
    # cubecl's CUDA runtime links the LLVM its build script finds there.
    stage clippy-cuda env XDG_DATA_HOME=${tracelLlvm} cargo clippy --locked -p gpu -p tgtts \
      -p bench --all-targets --features gpu/nvrtc,tgtts/cuda,bench/cuda -- -D warnings
    stage rustdoc env RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
    stage test cargo test --locked --workspace

    if [ "''${#failed[@]}" -gt 0 ]; then
      echo "audit: FAILED: ''${failed[*]}" >&2
      exit 1
    fi
    echo "audit: passed" >&2
  '';
}
