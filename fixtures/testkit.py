"""Values that testkit's own test reads back and checks against literals."""

import torch

import common

fixture = common.Fixture(seed=0)
minicpm4 = common.upstream("modules.minicpm4.model")
fixture.save(
    "testkit",
    {
        "f32": torch.tensor([[1.5, -2.25, 0.1], [3.0, 1e-3, -0.0]]),
        "bf16": torch.tensor([1.0, -2.5, 0.1, 3.14159], dtype=torch.bfloat16),
        "f16": torch.tensor([0.1, -65504.0, 6e-8], dtype=torch.float16),
        "i64": torch.tensor([7, -1]),
        "rms_layernorm": minicpm4.rms_layernorm(
            torch.tensor([[3.0, 4.0]]), torch.tensor([1.0, 2.0]), eps=0.0
        ),
    },
)
