# Vendored upstreams

The projects the GPU kernels and their toolchain read or build against, each a
git submodule pinned to one commit. Nothing builds from them: Nix does not see
submodules in this flake's source, and a build that needs an upstream takes it
as a flake input at the same commit. A clone does not need them to build or
test; `git submodule update --init --recursive` fetches them all, about 14 GB,
most of it the LLVM forks IREE and iree-amd-aie nest.

| path | upstream | license | commit | why |
| -- | -- | -- | -- | -- |
| `llama.cpp` | <https://github.com/ggml-org/llama.cpp> | MIT | `b64739ea` (release b9190) | its Vulkan matmul shaders, the nearest reference for the GPU kernels |
| `mlir-aie` | <https://github.com/Xilinx/mlir-aie> | Apache-2.0 WITH LLVM-exception, some files under the others in its `LICENSES/` | `35fa6b16` (main, 2026-10-09) | names the LLVM commit the toolchain is pinned to; the NPU compiler |
| `IRON` | <https://github.com/amd/IRON> | Apache-2.0, some files CC0-1.0 | `03613839` (iron-next, 2026-10-10) | NPU operators, built on mlir-aie at this set |
| `llvm-aie` | <https://github.com/Xilinx/llvm-aie> | Apache-2.0 WITH LLVM-exception | `94b320c0` (the Peano build mlir-aie and mlir-air pin) | the compiler for the NPU's cores |
| `xdna-driver` | <https://github.com/amd/xdna-driver> | per file: Apache-2.0 (userspace), GPL-2.0 (driver), MIT; `LICENSE.amdnpu` covers firmware, not in the tree | `e2fad18d` (main, 2026-09-25) | XRT's plugin for the NPU's driver |
| `mlir-air` | <https://github.com/Xilinx/mlir-air> | MIT | `416edaec` (main, 2026-10-10) | names the same LLVM, mlir-aie and Peano; maps loops onto the NPU |
| `iree` | <https://github.com/iree-org/iree> | Apache-2.0 | `3fd498c0` (rc20261010, 2026-10-09) | an MLIR pipeline to cooperative-matrix SPIR-V that runs, the kernels' reference |
| `iree-amd-aie` | <https://github.com/nod-ai/iree-amd-aie> | Apache-2.0 | `fddfec1b` (main, 2026-06-04) | IREE's NPU plugin |

The NPU's are one version set, mlir-air's: mlir-air names the LLVM, mlir-aie
and Peano commits it builds with (`utils/clone-llvm.sh`,
`utils/clone-mlir-aie.sh`, `utils/build-mlir-air-using-wheels.sh`), mlir-aie
names the same LLVM and Peano, and IRON's `iron-next`, unlike its `devel`,
requires that mlir-aie and Peano. A bump moves the set together.

Vendored code is not edited, formatted or linted here; a change goes to a fork
the submodule then points at.
