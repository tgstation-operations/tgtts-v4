"""Shared by the fixture generators, which `nix run .#fixtures` runs.

A generator builds an upstream module at a tiny config with seeded random weights, runs it
on seeded random inputs, and saves the weights under their state_dict names, the inputs and
the outputs to crates/<crate>/tests/fixtures/<generator>.safetensors.
"""

import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = Path(os.environ["VOXCPM_SRC"])
REVISION = os.environ["VOXCPM_REV"]

DTYPES = {
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
}


def upstream(module):
    """Import voxcpm.<module> from the pinned upstream source.

    The package's __init__ imports .core, which needs librosa, transformers and more, while
    the modules need only torch, einops, pydantic and numpy: voxcpm is registered as a bare
    package, so that its __init__ never runs.
    """
    if "voxcpm" not in sys.modules:
        spec = importlib.machinery.ModuleSpec("voxcpm", None, is_package=True)
        spec.submodule_search_locations = [str(UPSTREAM / "src" / "voxcpm")]
        sys.modules["voxcpm"] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"voxcpm.{module}")


class Fixture:
    """One generator's run, seeded when it is made."""

    def __init__(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        # A thread count taken from the machine would let a rerun elsewhere differ in the
        # last bits.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def save(self, crate, tensors):
        name = Path(sys.modules["__main__"].__file__).stem
        crate_dir = ROOT / "crates" / crate
        if not (crate_dir / "Cargo.toml").is_file():
            sys.exit(f"{name}: no crate {crate}")
        path = crate_dir / "tests" / "fixtures" / f"{name}.safetensors"
        metadata = {
            "generator": f"fixtures/{name}.py",
            "seed": str(self.seed),
            "torch": torch.__version__,
            "upstream": f"OpenBMB/VoxCPM@{REVISION}",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialize(tensors, metadata))
        check(path, tensors, metadata)
        size = path.stat().st_size
        print(f"{path.relative_to(ROOT)}: {len(tensors)} tensors, {size} bytes")


def raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def serialize(tensors, metadata):
    """The safetensors encoding, with its header in a fixed order.

    The library writes the metadata in a different order on each run, so a regenerated
    fixture would not be byte-identical to the last. Like the library, this lays the
    tensors out widest element first, so that each one is aligned to its element.
    """
    header = {"__metadata__": metadata}
    chunks = []
    offset = 0
    for name in sorted(tensors, key=lambda name: (-tensors[name].element_size(), name)):
        tensor = tensors[name]
        data = raw(tensor).numpy().tobytes()
        header[name] = {
            "dtype": DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(chunks)


def check(path, tensors, metadata):
    """Read the file back through the safetensors library."""
    with safe_open(path, "pt") as file:
        assert file.metadata() == metadata
    loaded = load_file(path)
    assert loaded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype, name
        assert loaded[name].shape == tensor.shape, name
        assert torch.equal(raw(loaded[name]), raw(tensor)), name
