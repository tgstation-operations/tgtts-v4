{
  pkgs,
  rustToolchain,
  ffmpeg,
  gpuKernels,
  glslKernels,
}:
# What a GPU host runs of tts-rs before it serves, built release: the vulkan and graph crates'
# device tests and tgtts's rt test as test-<name>, tgtts and tgtts-gains, stream_timing, which
# times a running tgtts's streams, the graph crate's timing examples and vulkan's properties.
# Each program loads the Vulkan loader from the store, which finds the host driver's ICD under
# /run/opengl-driver, as NixOS puts it, and runs ffmpeg from the store; the tests find their
# fixtures in the store. Each runs the store's kernels unless the environment names others:
# TGTTS_GLSL_KERNELS_DIR and TGTTS_MLIR_KERNELS_DIR, which override the configuration file tgtts
# and tgtts-gains read, and the tests' GLSL_KERNELS and VULKAN_KERNELS. VULKAN_VALIDATION=1
# attaches the validation layer, which VK_LAYER_PATH names.
let
  inherit (pkgs) lib;

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
in
pkgs.stdenv.mkDerivation {
  pname = "gpu-trial";
  version = "0.1.0";
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.lock
      ../Cargo.toml
      ../crates
    ];
  };
  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../Cargo.lock;
    allowBuiltinFetchGit = true;
  };
  nativeBuildInputs = [
    rustPlatform.cargoSetupHook
    rustToolchain
    pkgs.jq
    pkgs.makeWrapper
    pkgs.pkg-config
    pkgs.removeReferencesTo
  ];
  buildInputs = [
    pkgs.libopus
    pkgs.soxr
  ];
  dontConfigure = true;
  # The tests read their fixtures from beside their sources, by the paths they were compiled
  # at, so the sources are built where they stay.
  buildPhase = ''
    runHook preBuild
    mkdir -p $out
    cp -r . $out/src
    # The vendored sources are configured in the build directory, which is no parent of $out.
    export CARGO_HOME=$NIX_BUILD_TOP/.cargo CARGO_TARGET_DIR=$NIX_BUILD_TOP/target
    cd $out/src
    cargo build --release --frozen -p tgtts --bins --example stream_timing
    cargo build --release --frozen -p vulkan -p graph --examples
    {
      cargo test --release --frozen -p vulkan -p graph --features vulkan/device,graph/device \
        --no-run --message-format=json
      cargo test --release --frozen -p tgtts --test rt --no-run --message-format=json
    } | jq -r 'select(.target.kind == ["test"]) | .executable // empty' >$NIX_BUILD_TOP/tests
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/libexec
    release=$CARGO_TARGET_DIR/release
    cp $release/{tgtts,tgtts-gains} \
      $release/examples/{stream_timing,decode_timing,generator_timing,lm_side_timing,local_timing,prefill_timing,vae_timing,properties} \
      $out/libexec/
    while read -r test; do
      name=$(basename "$test")
      cp "$test" "$out/libexec/test-''${name%-*}"
    done <$NIX_BUILD_TOP/tests
    # Panic locations in the standard library name the toolchain's sources.
    remove-references-to -t ${rustToolchain} $out/libexec/*
    for program in $out/libexec/*; do
      makeWrapper "$program" "$out/bin/$(basename "$program")" \
        --prefix LD_LIBRARY_PATH : ${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib \
        --prefix PATH : ${ffmpeg}/bin \
        --set-default TGTTS_GLSL_KERNELS_DIR ${glslKernels} \
        --set-default TGTTS_MLIR_KERNELS_DIR ${gpuKernels} \
        --set-default GLSL_KERNELS ${glslKernels} \
        --set-default VULKAN_KERNELS ${gpuKernels} \
        --set-default VK_LAYER_PATH ${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d \
        --set-default VK_KHRONOS_VALIDATION_VALIDATE_SYNC true
    done
    runHook postInstall
  '';

  meta.license = lib.licenses.agpl3Only;
}
