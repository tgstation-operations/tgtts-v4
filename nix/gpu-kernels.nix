{ pkgs, mlir }:
# The GPU kernels, compiled ahead of time from kernels/gpu to SPIR-V for RADV on the
# Radeon 8060S: per kernel, <name>.spv, the manifest <name>.json a runtime loads it by, and
# the disassembly <name>.spvasm; ir/ holds each one's instantiated source and final
# spirv.module, and checks/ the CPU oracle's log. The build fails unless every module passes
# spirv-val, matches its manifest and multiplies at RADV's one cooperative-matrix shape, and
# unless the GEMM's linalg op, lowered for the CPU instead, reproduces its reference bit for
# bit. Whether the kernels run right on the GPU is the device tests'.
#
# A package rather than part of the gate: a machine without the MLIR build would compile LLVM.
let
  inherit (pkgs) lib;
  python = pkgs.python3.withPackages (ps: [ ps.numpy ]);
  # M x N x K: the device tests' small shapes, and the base LM's up projection over a 446
  # position prefill, padded to the schedule's 128.
  gemms = [
    "128 128 32"
    "256 384 512"
    "512 6144 2048"
  ];
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "gpu-kernels";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = ../kernels/gpu;
  };

  nativeBuildInputs = [
    mlir
    python
    pkgs.spirv-tools
  ];

  buildPhase = ''
    runHook preBuild
    export PYTHONPATH=${mlir}/python_packages/mlir_core
    mkdir -p $out/checks
    for shape in ${lib.escapeShellArgs gemms}; do
      read -r m n k <<<"$shape"
      for acc in f16 f32; do
        python3 kernels/gpu/build.py --out $out gemm --m "$m" --n "$n" --k "$k" --acc "$acc"
      done
    done
    for acc in f16 f32; do
      python3 kernels/gpu/cpu_oracle.py --mlir-lib ${mlir}/lib gemm kernels/gpu/gemm.mlir \
        --entry gemm --acc "$acc" -m 64 -n 96 -k 1536
    done 2>&1 | tee $out/checks/cpu-oracle.log
    runHook postBuild
  '';

  dontInstall = true;

  meta.license = lib.licenses.agpl3Only;
}
