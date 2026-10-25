"""The old radio service's effect with its real clicks and static, which ASSETS_DIR names, by the
restatement of the service in dsp_radio, on made-up speech at 48 kHz and at the 44.1 kHz of the
silicon voices.

Without them this leaves the golden as it is, and the dsp tests refuse it once it was made by
other libraries than the fixtures beside it, or by another version of this file or of dsp_radio.
"""

import hashlib
import os
import sys
import zlib
from pathlib import Path

import numpy as np

import common
import dsp_radio

NAMES = ["mic_click_on.wav", "mic_click_off.wav", "diffstatic.wav"]

if not os.environ.get("ASSETS_DIR"):
    print("ASSETS_DIR is not set: the golden is left as it is", file=sys.stderr)
    sys.exit()
assets = Path(os.environ["ASSETS_DIR"])
files = {name: (assets / name).read_bytes() for name in NAMES}


def voice(seconds, rate):
    """Speech-like harmonics at 16 bits."""
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(180 * (1 + 0.1 * np.sin(2 * np.pi * 4 * t))) / rate
    x = sum(np.sin(k * phase) * np.exp(-k / 8) for k in range(1, 40))
    return np.round(14000 * x / np.max(np.abs(x))).astype(np.int16)


generators = [Path(__file__), Path(dsp_radio.__file__)]
common.Fixture(seed=0).save(
    "dsp",
    dsp_radio.run(
        files, [(voice(0.1, 48000), 48000, 1), (voice(0.1, 44100), 44100, 1)]
    ),
    assets=" ".join(
        f"{name} sha256:{hashlib.sha256(data).hexdigest()}"
        for name, data in files.items()
    ),
    generator_crc32=f"{zlib.crc32(b''.join(p.read_bytes() for p in generators)):08x}",
    **dsp_radio.provenance(),
)
