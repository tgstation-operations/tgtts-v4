"""VoxCPM2 on its own weights, which VOXCPM2_DIR names, speaking a game message in a voice from
VOICES_DIR as the Python service had it spoken, in f32: each patch upstream generates until its
stop head says to stop, the noise it was sampled from and the stop head's logits, and the start
and end of the audio decoded from the patches.

The voice is kept out of the golden but for checksums. The tokenizer runs tokenizer.json as
transformers 4 runs it.

Without both directories this leaves the golden as it is. The graph crate's GPU tests read it,
the voice it names from VOICES_DIR, and its patches and their noise, which they take as inputs.
"""

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

import common

# openbmb/VoxCPM2 at 32279effe8c19989596f05d353d1447f51d9e915, the snapshot prod serves.
CHECKPOINT_SHA256 = "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"
VOICE = "Baraka"
# "Security, report to the bridge" as tg_tts_service.py cleaned it up.
TEXT = "Security, report to the bridge. "
SEED = 0
# The service's max_generate_length.
MAX_PATCHES = 256
# Audio kept from each end: two patches of 48 kHz samples.
KEPT = 2 * 4 * 1920

checkpoint, voices = os.environ.get("VOXCPM2_DIR"), os.environ.get("VOICES_DIR")
if not checkpoint or not voices:
    print(
        "voxcpm_checkpoint: kept the golden; making it needs VOXCPM2_DIR and VOICES_DIR",
        file=sys.stderr,
    )
    sys.exit()
checkpoint, voices = Path(checkpoint), Path(voices)
weights = checkpoint / "model.safetensors"
with weights.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
if digest != CHECKPOINT_SHA256:
    sys.exit(
        f"{weights} has sha256 {digest}, not that of the checkpoint the golden is of"
    )

fixture = common.Fixture(seed=0)
voxcpm2 = common.upstream("model.voxcpm2")
config = voxcpm2.VoxCPMConfig.model_validate_json(
    (checkpoint / "config.json").read_text()
)
config.dtype = "float32"
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(checkpoint / "tokenizer.json"), unk_token="<unk>"
)
vae = voxcpm2.AudioVAEV2(config.audio_vae_config)
state = torch.load(checkpoint / "audiovae.pth", "cpu", weights_only=True)
vae.load_state_dict(state["state_dict"])
model = voxcpm2.VoxCPM2Model(config, tokenizer, vae, device="cpu")
missing, unexpected = model.load_state_dict(load_file(weights), strict=False)
assert not unexpected and all(name.startswith("audio_vae.") for name in missing)
model.eval()

path = voices / "latents" / f"{VOICE}.safetensors"
latents = load_file(path)["latents"]
with safe_open(path, "pt") as file:
    transcript = file.metadata()["text"]
model._encode_wav = lambda *args, **kwargs: latents.view(
    -1, model.patch_size, model.feat_dim
)

prompt = {}
inference = model._inference


def record_prompt(text, *args, **kwargs):
    prompt["tokens"] = text[0].tolist()
    return inference(text, *args, **kwargs)


model._inference = record_prompt
steps = common.record_steps(model)
with torch.no_grad():
    audio = model.generate(
        TEXT,
        prompt_text=transcript,
        prompt_wav_path="voice",
        reference_wav_path="voice",
        min_len=-1,
        max_len=MAX_PATCHES,
        inference_timesteps=10,
        cfg_value=2.0,
        seed=SEED,
    )[0]

stops = torch.cat(steps["stop_logits"]).argmax(dim=-1).tolist()
assert stops.index(1) == len(stops) - 1, f"the stop head says {stops}"
tokens = torch.tensor(prompt["tokens"], dtype=torch.int64)
fixture.save(
    "voxcpm",
    {
        "input.noise": torch.cat(steps["noise"]),
        "output.patches": torch.cat(steps["patches"]),
        "output.stop_logits": torch.cat(steps["stop_logits"]),
        "output.audio.start": audio[:KEPT].contiguous(),
        "output.audio.end": audio[-KEPT:].contiguous(),
    },
    checkpoint=f"model.safetensors sha256:{digest}",
    voice=VOICE,
    voice_crc32=f"{zlib.crc32(latents.numpy().tobytes()):08x}",
    transcript_crc32=f"{zlib.crc32(transcript.encode()):08x}",
    text=TEXT,
    prompt_tokens=json.dumps(
        [len(tokens), zlib.crc32(tokens.numpy().astype("<i8").tobytes())]
    ),
    audio_samples=str(len(audio)),
    generator_crc32=f"{zlib.crc32(Path(__file__).read_bytes()):08x}",
)
