"""VoxCPM2's local DiT at the tiny config in crates/voxcpm/tests/fixtures, with its weights
under the checkpoint's names.

The first row is at the start of the flow with no step, as the sampler calls it, and the second
has the hidden states zeroed, as the sampler's unconditional half has them.
"""

import torch

import common

fixture = common.Fixture(seed=0)
model = common.voxcpm2(
    (common.ROOT / "crates/voxcpm/tests/fixtures/voxcpm.json").read_text()
)
dit = model.feat_decoder.estimator
tensors = {
    f"feat_decoder.estimator.{name}": parameter.data
    for name, parameter in dit.named_parameters()
}
with torch.no_grad():
    rows, hidden = 3, model.config.dit_config.hidden_dim
    inputs = {
        "x": torch.randn(rows, model.feat_dim, model.patch_size),
        "mu": torch.randn(rows, 2 * hidden),
        "t": torch.rand(rows),
        "cond": torch.randn(rows, model.feat_dim, model.patch_size),
        "dt": torch.rand(rows),
    }
    inputs["t"][0], inputs["dt"][0] = 1.0, 0.0
    inputs["mu"][1] = 0.0
    tensors.update({f"input.{name}": value for name, value in inputs.items()})
    tensors["output.velocity"] = dit(**inputs)

fixture.save("voxcpm", tensors)
