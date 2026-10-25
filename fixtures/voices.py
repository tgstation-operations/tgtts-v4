"""Voice files as the Python service has them, written by torch.save, for the voices crate.

latent.speaker_latent and blips.blips hold what the service's files hold, at a size a test can
read: every byte value in the latents and a transcript beyond ASCII, and blips at every base,
number and key the service has, with a few samples each and enough of them that the pickle's
memo outgrows BINPUT, as the service's does. The others each differ from those in one way the
converter refuses.
"""

import collections
import io
import struct
import wave
import zipfile

import torch

import common

FIXTURES = common.ROOT / "crates" / "voices" / "tests" / "fixtures"
RATE = 48000
BASES = ["male", "female"]
NUMBERS = ["1", "2", "3", "4"]
KEYS = sorted(
    [*"0123456789abcdefghijklmnopqrstuvwxyz", "10", "11", "OK", "Gwah", "Deska"]
)
# -0.0, infinity, -infinity, the least subnormal and a NaN with a payload.
SPECIAL_BITS = [0x80000000, 0x7F800000, 0xFF800000, 0x00000001, 0x7FC00001]


def wav(samples, channels=1, rate=RATE):
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return out.getvalue()


def blip(base, number, key):
    """Up to three samples, the first saying where the blip belongs."""
    where = 1000 * BASES.index(base) + 100 * int(number) + KEYS.index(key)
    return [where, -32768, 32767][: KEYS.index(key) % 4]


def silence_but(odd):
    """Blips at two bases, numbers and keys, silent but for male/1/b, which is `odd`.

    Two of each, since the pickler writes a dict of one item with SETITEM, which the service's
    files never have.
    """
    return {
        base: {
            number: {
                key: io.BytesIO(
                    odd if (base, number, key) == ("male", "1", "b") else wav([0])
                )
                for key in ["a", "b"]
            }
            for number in ["1", "2"]
        }
        for base in BASES
    }


def save(name, obj, **kwargs):
    torch.save(obj, FIXTURES / name, **kwargs)
    written(name)


def written(name):
    path = FIXTURES / name
    print(f"{path.relative_to(common.ROOT)}: {path.stat().st_size} bytes")


specials = struct.pack("<64I", *SPECIAL_BITS, *[0] * (64 - len(SPECIAL_BITS)))
latent = {
    "latents": bytes(range(256)) * 3 + specials,
    "text": f'Said "caf{chr(0xE9)}", {chr(0x2014)} then {chr(0x79C1)}{chr(0x306F)}.\nA second line.',
}
blips = {
    base: {
        number: {key: io.BytesIO(wav(blip(base, number, key))) for key in KEYS}
        for number in NUMBERS
    }
    for base in BASES
}

FIXTURES.mkdir(parents=True, exist_ok=True)
save("latent.speaker_latent", latent)
save("blips.blips", blips)
save("ordered.speaker_latent", collections.OrderedDict(latent))
save("protocol4.speaker_latent", latent, pickle_protocol=4)
save("tensor.speaker_latent", {**latent, "latents": torch.zeros(4, 64)})
save("stereo.blips", silence_but(wav([0, 0], channels=2)))
save("rates.blips", silence_but(wav([0], rate=44100)))

with (
    zipfile.ZipFile(FIXTURES / "latent.speaker_latent") as source,
    zipfile.ZipFile(FIXTURES / "deflated.speaker_latent", "w") as deflated,
):
    for record in source.infolist():
        name = "deflated" + record.filename.removeprefix("latent")
        deflated.writestr(
            zipfile.ZipInfo(name), source.read(record), zipfile.ZIP_DEFLATED
        )
written("deflated.speaker_latent")
