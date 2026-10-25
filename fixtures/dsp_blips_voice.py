"""The old blips service's blips in a real voice, whose converted voices directory VOICES_DIR
names, by the restatement of the service in dsp_blips.

Without it this leaves the golden as it is, and the dsp tests refuse it once it was made by other
libraries than the fixtures beside it, or by another version of this file or of dsp_blips.
"""

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

# Before dsp_blips imports numpy, for the same reason it pins it.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"

from safetensors import safe_open  # noqa: E402

import common  # noqa: E402
import dsp_blips  # noqa: E402

VOICE = "Man (Ace Detective)"

if not os.environ.get("VOICES_DIR"):
    print("VOICES_DIR is not set: the golden is left as it is", file=sys.stderr)
    sys.exit()
voices = Path(os.environ["VOICES_DIR"])
stem = json.loads((voices / "voices.json").read_text())[VOICE]
path = voices / "blips" / f"{stem}.safetensors"
with path.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
prefix = f"{dsp_blips.BASE}/{dsp_blips.NUMBER}/"
with safe_open(path, "np") as file:
    rate = int(file.metadata()["sample_rate"])
    voice = {
        name.removeprefix(prefix): file.get_tensor(name)
        for name in file.keys()
        if name.startswith(prefix)
    }

cases = [
    ("hello 12 there?", "0", None),
    ("'Hi", "0", None),
    ("Hi", "0", 1),
    ("ok 1", "4", 2),
    ("hello there", "0", None),
]
messages = [("go 1. ok 2?", "0", None)]
generators = [Path(__file__), Path(dsp_blips.__file__)]
common.Fixture(seed=0).save(
    "dsp",
    dsp_blips.run(voice, rate, cases, messages, checksum_exact=True),
    voice=VOICE,
    blips=f"{path.name} sha256:{digest}",
    generator_crc32=f"{zlib.crc32(b''.join(p.read_bytes() for p in generators)):08x}",
    **dsp_blips.provenance(cases, messages),
)
