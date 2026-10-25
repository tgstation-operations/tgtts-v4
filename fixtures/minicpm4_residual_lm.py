"""VoxCPM2's residual LM on the real weights, which VOXCPM2_DIR names.

Without them this leaves the golden as it is, and the minicpm4 tests refuse it once it was
made by another torch or upstream than the fixtures beside it, or by another version of this
file.
"""

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

import common

# openbmb/VoxCPM2 at 32279effe8c19989596f05d353d1447f51d9e915, the snapshot prod serves.
CHECKPOINT_SHA256 = "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"

if not os.environ.get("VOXCPM2_DIR"):
    print("VOXCPM2_DIR is not set: the golden is left as it is", file=sys.stderr)
    sys.exit()
checkpoint = Path(os.environ["VOXCPM2_DIR"])
weights = checkpoint / "model.safetensors"
with weights.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
if digest != CHECKPOINT_SHA256:
    sys.exit(
        f"{weights} has sha256 {digest}, not that of the checkpoint the golden is of"
    )

fixture = common.Fixture(seed=0)
minicpm4 = common.upstream("modules.minicpm4")
config = json.loads((checkpoint / "config.json").read_text())

lm_config = minicpm4.MiniCPM4Config.model_validate(config["lm_config"])
residual_lm_config = lm_config.model_copy(deep=True)
residual_lm_config.num_hidden_layers = config["residual_lm_num_layers"]
residual_lm_config.vocab_size = 0
residual_lm_config.no_rope = config["residual_lm_no_rope"]

residual_lm = minicpm4.MiniCPMModel(residual_lm_config).eval()
prefix = "residual_lm."
with safe_open(weights, "pt") as file:
    names = [name for name in file.keys() if name.startswith(prefix)]
    state = {name.removeprefix(prefix): file.get_tensor(name) for name in names}
residual_lm.load_state_dict(state)

with torch.no_grad():
    x = torch.randn(2, 8, residual_lm_config.hidden_size)
    causal, _ = residual_lm(x, is_causal=True)

fixture.save(
    "minicpm4",
    {"input.residual_lm.x": x, "output.residual_lm.causal": causal},
    checkpoint=f"model.safetensors sha256:{digest}",
    generator_crc32=f"{zlib.crc32(Path(__file__).read_bytes()):08x}",
)
