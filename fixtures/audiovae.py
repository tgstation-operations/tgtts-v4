"""The audio VAE's building blocks, its decoder and its encoder, at tiny sizes with random
weights.

Each module's weights are saved under its name, and what it was given and gave back under
input.<name> and output.<name>.
"""

import torch

import common

fixture = common.Fixture(seed=0)
vae = common.upstream("modules.audiovae.audio_vae_v2")


def randomize(module):
    """Replace upstream's initial values, under several of which (Snake alphas of 1, conditions
    that scale by 1 and add 0, g equal to |v|) a wrong port would still pass."""
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("alpha"):
                magnitude = torch.empty_like(parameter).uniform_(0.2, 2.0)
                parameter.copy_(magnitude * torch.randn_like(parameter).sign())
            elif name.endswith("weight_g"):
                parameter.uniform_(0.5, 1.5)
            elif name.endswith("weight_v"):
                parameter.normal_()
            elif name.endswith("scale_embed.weight"):
                parameter.normal_(mean=1.0, std=0.3)
            elif name.endswith("bias_embed.weight"):
                parameter.normal_(std=0.3)
            elif name.endswith("bias"):
                parameter.normal_(std=0.1)
            else:
                raise ValueError(f"no random value for {name}")
    return module


tensors = {}


def save(name, module, inputs, output):
    tensors.update(
        {f"{name}.{key}": value for key, value in module.state_dict().items()}
    )
    tensors.update({f"input.{name}.{key}": value for key, value in inputs.items()})
    tensors[f"output.{name}"] = output.detach()


def block(name, module, x):
    module = randomize(module)
    save(name, module, {"x": x}, module(x))


# The encoder's downsampling, with an odd stride.
block(
    "conv",
    vae.WNCausalConv1d(3, 4, kernel_size=10, stride=5, padding=3, output_padding=1),
    torch.randn(2, 3, 20),
)
block(
    "conv_transpose_odd",
    vae.WNCausalTransposeConv1d(
        4, 3, kernel_size=10, stride=5, padding=3, output_padding=1
    ),
    torch.randn(2, 4, 6),
)
block(
    "conv_transpose_even",
    vae.WNCausalTransposeConv1d(
        4, 3, kernel_size=4, stride=2, padding=1, output_padding=0
    ),
    torch.randn(2, 4, 6),
)
block(
    "residual", vae.CausalResidualUnit(4, dilation=9, groups=4), torch.randn(2, 4, 40)
)

# Alphas at the extremes of VoxCPM2's, negative and near zero included, on large inputs.
snake = vae.Snake1d(4)
with torch.no_grad():
    snake.alpha.copy_(torch.tensor([-1.71, 2.04e-5, 1.0, 3.75]).view(1, 4, 1))
x = 20 * torch.randn(2, 4, 16)
save("snake", snake, {"x": x}, snake(x))

condition = randomize(vae.SampleRateConditionLayer(4, sr_bin_buckets=4))
x, buckets = torch.randn(3, 4, 5), torch.tensor([0, 3, 1])
save("sr_cond", condition, {"x": x, "buckets": buckets}, condition(x, buckets))

# out_sample_rate falls in bucket 1 of 0 to 3, where taking the last bucket for the default
# would not.
config = vae.AudioVAEConfig(
    encoder_dim=2,
    encoder_rates=[2, 5],
    latent_dim=4,
    decoder_dim=32,
    decoder_rates=[3, 2, 2],
    out_sample_rate=24000,
)
model = vae.AudioVAE(config)
randomize(model.decoder)
z = torch.randn(2, 4, 8)
decoder = model.decoder.state_dict()
# common writes no I32.
decoder["sr_bin_boundaries"] = decoder["sr_bin_boundaries"].long()
tensors.update({f"decoder.{key}": value for key, value in decoder.items()})
tensors["input.decode.z"] = z
tensors["output.decode"] = model.decode(z).detach()
# Upstream's streaming decoder, a frame at a time at first.
with model.streaming_decode() as stream:
    streamed = [stream.decode_chunk(chunk) for chunk in z.split([1, 1, 2, 4], dim=2)]
tensors["output.decode_streamed"] = torch.cat(streamed, dim=2).detach()

randomize(model.encoder)
# Upstream pads 97 samples to whole frames of 10, and encodes 10 frames.
audio = 0.3 * torch.randn(2, 1, 97)
tensors.update(
    {f"encoder.{key}": value for key, value in model.encoder.state_dict().items()}
)
tensors["input.encode.audio"] = audio
tensors["output.encode"] = model.encode(audio, config.sample_rate).detach()

fixture.save("audiovae", tensors)
