"""librosa's resample at its default soxr_hq, for the dsp crate's resample module.

The rates are those the old service resamples at: speech to 8 kHz and back for the radio, its
assets from theirs to the speech's, and the stretched rates of a pitch shift. Signals are short
to keep the file small; one long enough that python-soxr feeds it to libsoxr in several chunks
is saved as the CRC-32 of its output.
"""

import importlib.metadata
import zlib

import librosa
import numpy as np
import soxr
import torch

import common

fixture = common.Fixture(seed=0)
rng = np.random.default_rng(0)
tensors = {}


def stored(x):
    """float32 as F32, float64 as the I64 of its bits."""
    if x.dtype == np.float64:
        x = x.view(np.int64)
    # torch.from_numpy gives an empty array a stride of 0, which safetensors cannot view as bytes.
    if not x.size:
        return torch.zeros(0, dtype=torch.from_numpy(x).dtype)
    return torch.from_numpy(x.copy())


def signal(n, dtype):
    """Harmonics of a wavering pitch under noise, with a gap, at about speech level."""
    t = np.arange(n)
    phase = np.cumsum(0.02 + 0.01 * (t % 97) / 97)
    x = sum(np.cos(k * phase) / k for k in range(1, 30)) * (t % 1500 > 300)
    x = x + 0.05 * rng.standard_normal(n)
    return (0.3 * x / np.max(np.abs(x), initial=1e-9)).astype(dtype)


cases = [
    (np.float32, 1200, 48000, 8000),
    (np.float32, 400, 8000, 48000),
    (np.float32, 1500, 44100, 8000),
    (np.float64, 1000, 44100, 48000),
    (np.float64, 500, 11025, 48000),
    (np.float64, 600, 22050, 48000),
    (np.float64, 250, 8000, 48000),
    (np.float64, 500, 11025, 44100),
    (np.float32, 7, 48000, 8000),
    (np.float32, 1, 8000, 48000),
    (np.float32, 0, 8000, 48000),
    (np.float32, 100, 48000, 48000),
]
for steps, bins in [(3, 12), (-5, 12), (7, 24), (-1, 24), (1.2345, 12), (-0.0042, 12)]:
    rate = 2.0 ** (-float(steps) / bins)
    cases.append((np.float32, 1500, float(48000) / rate, 48000))

tensors["cases"] = torch.tensor(len(cases))
for case, (dtype, n, orig_sr, target_sr) in enumerate(cases):
    x = signal(n, dtype)
    y = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)
    assert y.dtype == dtype
    tensors[f"{case}/f64"] = torch.tensor(int(dtype == np.float64))
    tensors[f"{case}/rates"] = stored(np.array([orig_sr, target_sr], dtype=np.float64))
    tensors[f"{case}/input"] = stored(x)
    tensors[f"{case}/output"] = stored(y)

# python-soxr feeds libsoxr 8 kHz to 48 kHz in chunks of 8000 samples.
long = signal(20000, np.float32)
tensors["long/input"] = stored(long)
y = librosa.resample(long, orig_sr=8000, target_sr=48000)
tensors["long/output_len"] = torch.tensor(len(y))
tensors["long/output_crc32"] = torch.tensor(zlib.crc32(y.tobytes()))

fixture.save(
    "dsp",
    tensors,
    librosa=librosa.__version__,
    numpy=np.__version__,
    soxr=soxr.__version__,
    libsoxr=soxr.__libsoxr_version__,
)
