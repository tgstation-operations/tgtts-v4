"""VoxCPM2's local encoder at the tiny config in crates/voxcpm/tests/fixtures, on random
patches, with its weights under the checkpoint's names."""

import torch

import common

fixture = common.Fixture(seed=0)
model = common.voxcpm2(
    (common.ROOT / "crates/voxcpm/tests/fixtures/voxcpm.json").read_text()
)
encoder = model.feat_encoder
tensors = {
    f"feat_encoder.{name}": parameter.data
    for name, parameter in encoder.named_parameters()
}
with torch.no_grad():
    patches = torch.randn(2, 3, model.patch_size, model.feat_dim)
    tensors["input.patches"] = patches
    tensors["output.embeddings"] = encoder(patches)

fixture.save("voxcpm", tensors)
