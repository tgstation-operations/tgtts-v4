{
  pkgs,
  rustToolchain,
  ffmpeg,
  gpuKernels,
  glslKernels,
}:
# tgtts and tgtts-gains as the servers run them. Each loads the Vulkan loader from the store, which
# finds the host driver's ICD under /run/opengl-driver, as NixOS puts it, runs ffmpeg from the
# store, and runs the store's kernels unless TGTTS_GLSL_KERNELS_DIR and TGTTS_MLIR_KERNELS_DIR,
# which override the configuration file, name others.
let
  inherit (pkgs) lib;

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };
in
pkgs.stdenv.mkDerivation {
  pname = "tgtts";
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
    cargo build --release --frozen -p tgtts --bins
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/libexec
    cp target/release/tgtts target/release/tgtts-gains $out/libexec/
    # Panic locations in the standard library name the toolchain's sources.
    remove-references-to -t ${rustToolchain} $out/libexec/*
    for program in $out/libexec/*; do
      makeWrapper "$program" "$out/bin/$(basename "$program")" \
        --prefix LD_LIBRARY_PATH : ${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib \
        --prefix PATH : ${ffmpeg}/bin \
        --set-default TGTTS_GLSL_KERNELS_DIR ${glslKernels} \
        --set-default TGTTS_MLIR_KERNELS_DIR ${gpuKernels}
    done
    runHook postInstall
  '';

  meta.license = lib.licenses.agpl3Only;
}
