{ pkgs, llvm-project }:
# LLVM and MLIR at the commit vendor/mlir-aie and vendor/mlir-air build
# against, which compiles the GPU kernels to SPIR-V and is the MLIR the NPU's
# compiler builds on later. The configuration is mlir-aie's
# (utils/build-llvm.sh): the host target only, MLIR the only extra project,
# assertions on, RTTI off, and the Python bindings on, which the kernels'
# build drives MLIR through and IRON needs, and which would otherwise mean
# rebuilding LLVM to add.
#
# It links with the wrapped default linker rather than lld as that script
# prefers: nixpkgs' lld is unwrapped, so -fuse-ld=lld skips the wrapper's RPATH
# handling, and the freshly built tablegen cannot load zlib at the first
# generated header.
let
  python = pkgs.python3.withPackages (ps: [
    ps.nanobind
    ps.pybind11
    ps.numpy
    ps.pyyaml
  ]);
in
pkgs.stdenv.mkDerivation {
  pname = "mlir";
  version = "24.0.0-unstable-bdf04ebb";
  src = llvm-project;
  sourceRoot = "source/llvm";

  nativeBuildInputs = [
    pkgs.cmake
    pkgs.ninja
    python
  ];
  buildInputs = [ pkgs.zlib ];

  cmakeFlags = [
    "-DLLVM_ENABLE_PROJECTS=mlir"
    "-DLLVM_TARGETS_TO_BUILD=host"
    "-DLLVM_BUILD_EXAMPLES=OFF"
    "-DLLVM_OPTIMIZED_TABLEGEN=OFF"
    "-DLLVM_ENABLE_OCAMLDOC=OFF"
    "-DLLVM_ENABLE_BINDINGS=OFF"
    "-DLLVM_ENABLE_RTTI=OFF"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_INCLUDE_TESTS=OFF"
    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    "-DPython3_EXECUTABLE=${python}/bin/python3"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON"
    "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON"
    "-DCMAKE_C_VISIBILITY_PRESET=hidden"
    "-DCMAKE_CXX_VISIBILITY_PRESET=hidden"
  ];

  meta.license = [
    pkgs.lib.licenses.asl20
    pkgs.lib.licenses.llvm-exception
  ];
}
