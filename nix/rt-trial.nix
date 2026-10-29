{
  pkgs,
  rustToolchain,
  ffmpeg,
  gpuKernels,
  glslKernels,
}:
# tgtts and tgtts-gains, the test of its model on the real weights as test-rt, and stream_timing,
# which times a running tgtts's streams, built release for a first run on an NVIDIA host. Each
# program loads the Vulkan loader from the store, which finds the host driver's ICD under
# /run/opengl-driver, as NixOS puts it, and runs ffmpeg from the store. Each
# runs the store's kernels unless the environment names others: TGTTS_GLSL_KERNELS_DIR and
# TGTTS_MLIR_KERNELS_DIR, which override the configuration file tgtts and tgtts-gains read, and
# the test's GLSL_KERNELS and VULKAN_KERNELS.
# VULKAN_VALIDATION=1 attaches the validation layer, which VK_LAYER_PATH names.
let
  inherit (pkgs) lib;

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
in
pkgs.stdenv.mkDerivation {
  pname = "rt-trial";
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
  buildPhase = ''
    runHook preBuild
    cargo build --release --frozen -p tgtts --bins --example stream_timing
    cargo test --release --frozen -p tgtts --test rt --no-run --message-format=json |
      jq -r 'select(.target.kind == ["test"]) | .executable // empty' >$NIX_BUILD_TOP/tests
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/libexec
    cp target/release/tgtts target/release/tgtts-gains target/release/examples/stream_timing \
      $out/libexec/
    cp "$(cat $NIX_BUILD_TOP/tests)" $out/libexec/test-rt
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
