"""The ids the Python service's tokenizer gives what it was given: every voice's transcript in
VOICES_DIR, each followed by a game message as the service sent them, and the messages alone.
The tokenizer is VoxCPM2's own, which VOXCPM2_DIR names, run as transformers 4 runs it, under
upstream's mask_multichar_chinese_tokens.

The messages are made here, from the characters the game lets through its filter, and cleaned
up as the service cleaned them. The transcripts are kept out of the golden, which holds a
checksum of each and of its ids.

transformers 5 builds the tokenizer anew rather than run the file, and gives other ids for text
that starts with a space; with a transcript in front, what the service tokenized never does, and
this refuses to make a golden on which the two disagree there.

Without both directories this leaves the golden as it is.
"""

import hashlib
import json
import os
import random
import struct
import sys
import zlib
from pathlib import Path

import torch
from transformers import LlamaTokenizerFast, PreTrainedTokenizerFast

import common

# tokenizer.json of openbmb/VoxCPM2 at 32279effe8c19989596f05d353d1447f51d9e915, which prod
# serves.
TOKENIZER_SHA256 = "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20"
# What tts_speech_filter in the game's code/__HELPERS/tts.dm lets through.
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,?.!'&-"
LINES = [
    "Security, report to the bridge.",
    "AI, open the door!",
    "I'm not going to tell you again",
    "Help! Help! Maint, by the bar!!",
    "Rock & roll - all night",
    "is anyone there?",
    "The supermatter is delaminating, evacuate NOW",
    "  waves  ",
    "Station time is 12-45, crew 34 of 40.",
    "...",
    "-",
    "a",
    "HONK HONK",
    "it's   the  clown's  fault",
    "Yes. No? Maybe!",
]
WORDS = (
    "the captain is dead clown help medbay I you we're don't AI borg door open "
    "please now rock & - 42 1984 syndicate nuke ops space station QM cargo why what ? !"
).split()


def message(rng):
    """A message the game could send: words or bare characters from its alphabet."""
    if rng.random() < 0.3:
        return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 60)))
    words = [rng.choice(WORDS) for _ in range(rng.randint(1, 16))]
    return rng.choice([" ", "  ", ", "]).join(words) + rng.choice(
        ["", ".", "!", "?", "!!"]
    )


def cleaned(text):
    """What tg_tts_service.py did to a message before generating."""
    if text and text[-1] not in [".", "?", "!"]:
        text += ". "
    if text and text[0].isalpha() and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text


def transcript(path):
    header = path.read_bytes()
    (size,) = struct.unpack("<Q", header[:8])
    return json.loads(header[8 : 8 + size])["__metadata__"]["text"]


def ids_crc32(ids):
    return zlib.crc32(struct.pack(f"<{len(ids)}q", *ids))


checkpoint, voices = os.environ.get("VOXCPM2_DIR"), os.environ.get("VOICES_DIR")
if not checkpoint or not voices:
    print(
        "voxcpm_tokenizer_checkpoint: kept the golden; making it needs VOXCPM2_DIR and "
        "VOICES_DIR",
        file=sys.stderr,
    )
    sys.exit()
checkpoint, voices = Path(checkpoint), Path(voices)
path = checkpoint / "tokenizer.json"
with path.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
if digest != TOKENIZER_SHA256:
    sys.exit(f"{path} has sha256 {digest}, not that of the tokenizer the golden is of")

fixture = common.Fixture(seed=0)
utils = common.upstream("model.utils")
tokenizer = utils.mask_multichar_chinese_tokens(
    PreTrainedTokenizerFast(tokenizer_file=str(path), unk_token="<unk>")
)
transformers5 = utils.mask_multichar_chinese_tokens(
    LlamaTokenizerFast.from_pretrained(checkpoint)
)

rng = random.Random(0)
messages = [cleaned(text) for text in LINES + [message(rng) for _ in range(185)]]
stems = sorted(
    latent.name.removesuffix(".safetensors") for latent in voices.glob("latents/*")
)
tensors = {}
transcripts, prompts = [], []
for i, stem in enumerate(stems):
    text = transcript(voices / "latents" / f"{stem}.safetensors")
    prompt = text + messages[i % len(messages)]
    for sample, rows in [(text, transcripts), (prompt, prompts)]:
        ids = tokenizer(sample)
        assert ids == transformers5(sample), stem
        rows.append([zlib.crc32(sample.encode()), len(ids), ids_crc32(ids)])
tensors["transcripts"] = torch.tensor(transcripts, dtype=torch.int64)
tensors["prompts"] = torch.tensor(prompts, dtype=torch.int64)
differ = 0
for i, text in enumerate(messages):
    ids = tokenizer(text)
    differ += ids != transformers5(text)
    tensors[f"messages.{i}"] = torch.tensor(ids, dtype=torch.int64)
print(
    f"transformers 5 gives other ids for {differ} of {len(messages)} messages alone",
    file=sys.stderr,
)

fixture.save(
    "voxcpm",
    tensors,
    stems=json.dumps(stems),
    messages=json.dumps(messages),
    tokenizer=f"tokenizer.json sha256:{digest}",
    generator_crc32=f"{zlib.crc32(Path(__file__).read_bytes()):08x}",
)
