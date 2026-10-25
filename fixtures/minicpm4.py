"""MiniCPM-4's stacks at the tiny VoxCPM2 config in crates/minicpm4/tests/fixtures.

The base LM, the residual LM and the local encoder are derived from its lm_config as
VoxCPM2Model derives them, and saved under the checkpoint's prefixes.
"""

import json

import torch

import common

fixture = common.Fixture(seed=0)
minicpm4 = common.upstream("modules.minicpm4")
config = json.loads(
    (common.ROOT / "crates/minicpm4/tests/fixtures/minicpm4.json").read_text()
)

lm_config = minicpm4.MiniCPM4Config.model_validate(config["lm_config"])
residual_lm_config = lm_config.model_copy(deep=True)
residual_lm_config.num_hidden_layers = config["residual_lm_num_layers"]
residual_lm_config.vocab_size = 0
residual_lm_config.no_rope = config["residual_lm_no_rope"]
encoder_config = lm_config.model_copy(deep=True)
encoder_config.hidden_size = config["encoder_config"]["hidden_dim"]
encoder_config.intermediate_size = config["encoder_config"]["ffn_dim"]
encoder_config.num_attention_heads = config["encoder_config"]["num_heads"]
encoder_config.num_hidden_layers = config["encoder_config"]["num_layers"]
encoder_config.kv_channels = config["encoder_config"]["kv_channels"]
encoder_config.vocab_size = 0

tensors = {}


def stack(prefix, config):
    model = minicpm4.MiniCPMModel(config).eval()
    for name, param in model.named_parameters():
        # Norm weights start at one, where a norm that ignored them would pass.
        if name.endswith("norm.weight"):
            param.data.uniform_(0.5, 1.5)
        tensors[prefix + name] = param.data
    return model


with torch.no_grad():
    base_lm = stack("base_lm.", lm_config)
    ids = torch.randint(lm_config.vocab_size, (2, 7))
    embeds = base_lm.embed_tokens(ids)
    tensors["input.base_lm.ids"] = ids
    tensors["output.base_lm.embeds"] = embeds
    tensors["output.base_lm.causal"], _ = base_lm(embeds, is_causal=True)

    residual_lm = stack("residual_lm.", residual_lm_config)
    x = torch.randn(2, 7, residual_lm_config.hidden_size)
    tensors["input.residual_lm.x"] = x
    tensors["output.residual_lm.causal"], _ = residual_lm(x, is_causal=True)

    encoder = stack("feat_encoder.encoder.", encoder_config)
    x = torch.randn(3, 5, encoder_config.hidden_size)
    tensors["input.feat_encoder.encoder.x"] = x
    hidden, _ = encoder(x, is_causal=False)
    tensors["output.feat_encoder.encoder.bidirectional"] = hidden

fixture.save("minicpm4", tensors)
