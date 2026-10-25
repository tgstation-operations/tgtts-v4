{ pkgs }:
let
  voxcpm = pkgs.fetchFromGitHub {
    owner = "OpenBMB";
    repo = "VoxCPM";
    rev = "cce58a5b59303c9bd63d12f23afe6a49a4a80c59";
    hash = "sha256-A49XnDjrGYsU1QDwjMjjoT0YgWC54VFzgBpNjWl/t9U=";
  };
  python = pkgs.python3.withPackages (ps: [
    ps.einops
    ps.librosa
    ps.numpy
    ps.pydantic
    ps.pydub
    ps.pysbd
    ps.safetensors
    ps.scipy
    ps.soundfile
    ps.torch
  ]);
in
pkgs.writeShellApplication {
  name = "fixtures";
  runtimeInputs = [
    pkgs.git
    python
  ];
  text = ''
    cd "$(git rev-parse --show-toplevel)"
    export VOXCPM_SRC=${voxcpm} VOXCPM_REV=${voxcpm.rev}
    # Otherwise importing common leaves fixtures/__pycache__ in the tree.
    export PYTHONDONTWRITEBYTECODE=1
    # torch, oneDNN and OpenBLAS pick kernels for the host's instruction set, and
    # oneDNN's AVX-512 ones round differently from its AVX2 ones: pinned to AVX2,
    # any x86-64 host since Haswell writes the same bytes.
    export ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 OPENBLAS_CORETYPE=Haswell

    if [ "$#" -eq 0 ]; then
      for generator in fixtures/*.py; do
        name=$(basename "$generator" .py)
        [ "$name" = common ] || set -- "$@" "$name"
      done
    fi
    for name; do
      if [ "$name" = common ] || [ ! -f "fixtures/$name.py" ]; then
        echo "fixtures: no generator fixtures/$name.py" >&2
        exit 2
      fi
    done
    for name; do
      echo "==> $name" >&2
      python3 "fixtures/$name.py"
    done
  '';
}
