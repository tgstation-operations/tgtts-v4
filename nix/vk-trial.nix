{
  pkgs,
  rustToolchain,
  glslKernels,
}:
# The vulkan and graph crates' device tests and their examples, decode_timing among them, built
# release for a first run on an NVIDIA host. Each program loads the Vulkan loader from the store,
# which finds the host driver's ICD under /run/opengl-driver, as NixOS puts it; the tests find
# their fixtures in the store and the GLSL kernels where GLSL_KERNELS names them, and
# VULKAN_VALIDATION=1 attaches the validation layer, which VK_LAYER_PATH names.
let
  inherit (pkgs) lib;

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
in
pkgs.stdenv.mkDerivation {
  pname = "vk-trial";
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
    pkgs.removeReferencesTo
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
    cargo build --release --frozen -p vulkan -p graph --examples
    cargo test --release --frozen -p vulkan -p graph --features vulkan/device,graph/device \
      --no-run --message-format=json |
      jq -r 'select(.target.kind == ["test"]) | .executable // empty' >$NIX_BUILD_TOP/tests
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/libexec
    cp $CARGO_TARGET_DIR/release/examples/{decode_timing,lm_side_timing,properties,vae_timing} $out/libexec/
    while read -r test; do
      name=$(basename "$test")
      name=''${name%-*}
      # The cooperative-matrix GEMMs are compiled for RADV's 64-wide subgroups.
      [ "$name" = gemm ] && continue
      cp "$test" "$out/libexec/test-$name"
    done <$NIX_BUILD_TOP/tests
    # Panic locations in the standard library name the toolchain's sources.
    remove-references-to -t ${rustToolchain} $out/libexec/*
    for program in $out/libexec/*; do
      makeWrapper "$program" "$out/bin/$(basename "$program")" \
        --prefix LD_LIBRARY_PATH : ${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib \
        --set-default GLSL_KERNELS ${glslKernels} \
        --set-default VK_LAYER_PATH ${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d \
        --set-default VK_KHRONOS_VALIDATION_VALIDATE_SYNC true
    done
    runHook postInstall
  '';

  meta.license = lib.licenses.agpl3Only;
}
