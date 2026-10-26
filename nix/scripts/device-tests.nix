{
  pkgs,
  rustToolchain,
  gpuKernels,
}:
pkgs.writeShellApplication {
  name = "device-tests";
  runtimeInputs = [
    rustToolchain
    # cargo links through cc, which a bare `nix run` shell does not have.
    pkgs.stdenv.cc
    pkgs.coreutils
    pkgs.diffutils
    pkgs.git
    pkgs.spirv-tools
  ];
  text = ''
    # device-tests [CARGO_TEST_ARGS...]: the vulkan crate's tests on the GPU, the GPU kernels
    # among them, each failing on any message of the validation layer.
    cd "$(git rev-parse --show-toplevel)"
    export CARGO_UNSTABLE_CHECKSUM_FRESHNESS=true CARGO_BUILD_FINGERPRINT=content \
      CARGO_UNSTABLE_RUSTDOC_DEPINFO=true
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export VULKAN_VALIDATION=1
    export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
    export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
    export VULKAN_KERNELS=${gpuKernels}

    # A hand-written kernel's blob is what its assembly assembles to, and valid.
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    while IFS= read -r -d "" source; do
      spirv-as --target-env vulkan1.3 "$source" -o "$scratch/assembled.spv"
      if ! cmp -s "$scratch/assembled.spv" "''${source%.spvasm}.spv"; then
        echo "device-tests: ''${source%.spvasm}.spv is not what $source assembles to" >&2
        exit 1
      fi
      spirv-val --target-env vulkan1.3 "''${source%.spvasm}.spv"
    done < <(git ls-files -z 'crates/*.spvasm')

    # One test at a time: they share the GPU, and time it.
    cargo test --locked --release -p vulkan --features device -- --test-threads 1 "$@"
  '';
}
