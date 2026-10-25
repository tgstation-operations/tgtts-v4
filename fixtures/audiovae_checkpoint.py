"""What upstream's audio VAE computes with VoxCPM2's own weights, for the crate's ignored
tests: a few frames of a real voice's latents decoded, and that speech encoded again.

It needs the checkpoint and a voices directory, which a run of every generator need not
have: without VOXCPM2_DIR and VOICES_DIR it keeps the golden as it is. The crate's tests
refuse a golden made with another upstream or torch than the other fixtures, or by another
version of this file.
"""

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file

import common

checkpoint, voices = os.environ.get("VOXCPM2_DIR"), os.environ.get("VOICES_DIR")
if not checkpoint or not voices:
    print(
        "audiovae_checkpoint: kept the golden; making it needs VOXCPM2_DIR and VOICES_DIR",
        file=sys.stderr,
    )
    sys.exit()

fixture = common.Fixture(seed=0)
vae = common.upstream("modules.audiovae.audio_vae_v2")
checkpoint = Path(checkpoint)
config = json.loads((checkpoint / "config.json").read_text())["audio_vae_config"]
config = vae.AudioVAEConfig(**config)
model = vae.AudioVAE(config)
weights = checkpoint / "audiovae.pth"
model.load_state_dict(torch.load(weights, "cpu", weights_only=True)["state_dict"])

voice = "Adachi"
latents = load_file(Path(voices) / "latents" / f"{voice}.safetensors")["latents"]
# Speech, past the quiet start.
frames = slice(4, 12)
z = latents[frames].T.unsqueeze(0).contiguous()

with torch.no_grad():
    audio = model.decode(z)
    # Averaged down to 16 kHz, and cut short of whole frames so that the encoder pads it.
    clip = torch.nn.functional.avg_pool1d(audio, 3)[..., :-100].contiguous()
    mu = model.encode(clip, config.sample_rate)

with weights.open("rb") as file:
    checkpoint_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
fixture.save(
    "audiovae",
    {
        "input.decode": z,
        "output.decode": audio,
        "input.encode": clip,
        "output.encode": mu,
    },
    checkpoint=f"audiovae.pth sha256:{checkpoint_sha256}",
    voice=f"{voice}, frames {frames.start} to {frames.stop}",
    generator_crc32=f"{zlib.crc32(Path(__file__).read_bytes()):08x}",
)
