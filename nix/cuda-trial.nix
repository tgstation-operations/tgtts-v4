{
  pkgs,
  rustToolchain,
  ffmpeg,
  tracelLlvm,
}:
# tgtts, bench and the gpu crate's tests built for CUDA, for a first run on an
# NVIDIA host: once with cubecl's default kernel compiler, LLVM to PTX, under
# llvm/, and once with NVRTC compiling its CUDA C++, under nvrtc/. Each program
# finds the host driver's libcuda in /run/opengl-driver/lib, as NixOS puts it,
# and the tests find their fixtures in the store.
let
  inherit (pkgs) lib;
  # 12.9: cubecl-cuda enables what it has for 12.8 and later, and a driver
  # supporting any CUDA from 12.9 up, 13.2 for the 595 series, runs it.
  cuda = pkgs.cudaPackages_12_9;
  cudaVersion = "12090";

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };

  # CUDA_PATH, laid out as a CUDA installation: the libdevice LLVM links the
  # kernels' math functions from, and the headers NVRTC compiles them with.
  cudaPath = pkgs.runCommand "cuda-path-${cuda.cudaMajorMinorVersion}" { } ''
    mkdir -p $out/include $out/nvvm
    cp -r ${cuda.cuda_cudart}/include/. ${cuda.cuda_cccl}/include/. $out/include/
    cp -r ${cuda.cuda_nvcc}/include/crt $out/include/
    cp -r ${cuda.cuda_nvcc}/nvvm/libdevice $out/nvvm/
  '';

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.lock
      ../Cargo.toml
      ../crates
    ];
  };

  build =
    {
      compiler,
      features,
      env,
    }:
    pkgs.stdenv.mkDerivation {
      pname = "tts-rs-cuda-${compiler}";
      version = "0.1.0";
      inherit src;
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
      buildInputs = [ pkgs.soxr ];
      # cubecl-cuda's LLVM, and the CUDA cudarc binds the driver's functions
      # for, which it would otherwise take from nvcc or its newest.
      XDG_DATA_HOME = tracelLlvm;
      CUDARC_CUDA_VERSION = cudaVersion;
      dontConfigure = true;
      # The tests read their fixtures from beside their sources, by the paths
      # they were compiled at, so the sources are built where they stay.
      buildPhase = ''
        runHook preBuild
        mkdir -p $out
        cp -r . $out/src
        # The vendored sources are configured in the build directory, which is
        # no parent of $out.
        export CARGO_HOME=$NIX_BUILD_TOP/.cargo CARGO_TARGET_DIR=$NIX_BUILD_TOP/target
        cd $out/src
        cargo build --release --frozen -p tgtts -p bench \
          --features tgtts/cuda,bench/cuda${features}
        cargo test --release --frozen -p gpu --features cuda${features} --no-run \
          --message-format=json |
          jq -r 'select(.target.kind == ["test"]) | .executable // empty' >$NIX_BUILD_TOP/tests
        runHook postBuild
      '';
      installPhase = ''
        runHook preInstall
        mkdir -p $out/bin $out/libexec
        cp $CARGO_TARGET_DIR/release/{tgtts,bench} $out/libexec/
        while read -r test; do
          name=$(basename "$test")
          cp "$test" "$out/libexec/gpu-test-''${name%-*}"
        done <$NIX_BUILD_TOP/tests
        # Panic locations in the standard library name the toolchain's sources.
        remove-references-to -t ${rustToolchain} $out/libexec/*
        for program in $out/libexec/*; do
          makeWrapper "$program" "$out/bin/$(basename "$program")" \
            --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib \
            --prefix PATH : ${ffmpeg}/bin --set CUDA_PATH ${cudaPath} ${env}
        done
        runHook postInstall
      '';
    };

  llvm = build {
    compiler = "llvm";
    features = "";
    env = "";
  };
  nvrtc = build {
    compiler = "nvrtc";
    features = ",gpu/nvrtc";
    env = "--prefix LD_LIBRARY_PATH : ${cuda.cuda_nvrtc.lib}/lib";
  };
in
pkgs.runCommand "cuda-trial" { } ''
  mkdir $out
  ln -s ${llvm} $out/llvm
  ln -s ${nvrtc} $out/nvrtc
''
// {
  inherit
    llvm
    nvrtc
    cudaPath
    ;
}
