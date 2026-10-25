{ pkgs, ffmpeg }:
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
    ps.transformers
  ]);
in
pkgs.writeShellApplication {
  name = "fixtures";
  runtimeInputs = [
    ffmpeg
    pkgs.git
    python
  ];
  text = ''
    cd "$(git rev-parse --show-toplevel)"
    export VOXCPM_SRC=${voxcpm} VOXCPM_REV=${voxcpm.rev}
    # Otherwise importing common leaves fixtures/__pycache__ in the tree.
    export PYTHONDONTWRITEBYTECODE=1
    # torch, oneDNN, OpenBLAS and numpy pick kernels for the host's instruction set,
    # and the AVX-512 ones of oneDNN and numpy round differently from their AVX2
    # ones: pinned to AVX2, any x86-64 host since Haswell writes the same bytes.
    export ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 OPENBLAS_CORETYPE=Haswell \
      NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"

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
