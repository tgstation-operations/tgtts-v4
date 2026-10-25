"""VoxCPM2's sampler, upstream's UnifiedCFM, at the tiny config in crates/voxcpm/tests/fixtures,
with its local DiT's weights under the checkpoint's names.

Each case records the standard normal noise upstream drew, and the noise scaled by the
temperature and the times it stepped through, as its Euler solver was given them. The first
case samples as the Python service did; the second has CFG-Zero* hold two steps rather than one.
"""

import torch

import common

fixture = common.Fixture(seed=0)
model = common.voxcpm2(
    (common.ROOT / "crates/voxcpm/tests/fixtures/voxcpm.json").read_text()
)
cfm = model.feat_decoder
tensors = {
    f"feat_decoder.estimator.{name}": parameter.data
    for name, parameter in cfm.estimator.named_parameters()
}

solve_euler = cfm.solve_euler
solved = {}


def capture(**kwargs):
    solved.update(kwargs)
    return solve_euler(**kwargs)


cfm.solve_euler = capture

cases = {"service": (10, 1.0, 2.0), "long": (50, 0.7, 1.5)}
with torch.no_grad():
    for name, (timesteps, temperature, cfg_value) in cases.items():
        rows, hidden = 2, model.config.dit_config.hidden_dim
        mu = torch.randn(rows, 2 * hidden)
        cond = torch.randn(rows, model.feat_dim, model.patch_size)
        state = torch.get_rng_state()
        noise = torch.randn(rows, model.feat_dim, model.patch_size)
        torch.set_rng_state(state)
        patches = cfm(
            mu,
            timesteps,
            model.patch_size,
            cond,
            temperature=temperature,
            cfg_value=cfg_value,
        )
        assert torch.equal(solved["x"], noise * temperature)
        sampling = torch.tensor([timesteps, temperature, cfg_value])
        tensors.update(
            {
                f"input.{name}.mu": mu,
                f"input.{name}.cond": cond,
                f"input.{name}.noise": noise,
                f"input.{name}.sampling": sampling,
                f"input.{name}.x": solved["x"],
                f"input.{name}.t_span": solved["t_span"],
                f"output.{name}.patches": patches,
            }
        )

fixture.save("voxcpm", tensors)
