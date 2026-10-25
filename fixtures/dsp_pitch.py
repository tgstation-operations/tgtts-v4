"""librosa's pitch shift, and the old service's conversions around it, for the dsp crate.

The signals are voiced, decaying like a blip, at the 48 kHz the service runs at; the shifts are
those the blips draw, in semitones, and the pitch the game sends, in quarter tones. One is clipped
at full scale, and its shift rings past it, where the conversion back to 16 bits saturates.
"""

import librosa
import numpy as np
import scipy
import torch

import common

RATE = 48000

fixture = common.Fixture(seed=0)
rng = np.random.default_rng(0)
tensors = {}


def audiosegment_to_numpy(samples):
    """The old service's, on mono 16-bit samples."""
    return samples.astype(np.float32) / (1 << 15)


def numpy_to_audiosegment(samples):
    """The old service's, to the 16-bit samples it makes a segment of, but held at full scale
    where its cast wraps around, and a NaN made 0, as the port has it."""
    scaled = np.nan_to_num(samples * 32767, nan=0, posinf=32767, neginf=-32768)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def blip(n, f0, peak):
    """16-bit samples of a voiced sound that starts sharply and decays, over a little noise."""
    t = np.arange(n) / RATE
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 5 * t))) / RATE
    x = sum(np.sin(k * phase) * np.exp(-k / 8) for k in range(1, 40))
    x = x * np.minimum(1, t / 0.01) * np.exp(-t / 0.15) + 0.01 * rng.standard_normal(n)
    return np.round(peak * x / np.max(np.abs(x))).astype(np.int16)


def clipped(n, f0):
    """A sine clipped at full scale, whose shift rings past it."""
    t = np.arange(n) / RATE
    x = np.clip(3 * np.sin(2 * np.pi * f0 * t), -1, 1) * np.minimum(1, t / 0.005)
    return np.round(32767 * x).astype(np.int16)


cases = [
    (11776, 180, 20000, 1.6 + 0.0037, 12),
    (3000, 220, 12000, -1.1, 12),
    (3000, 140, 12000, 0.4, 12),
    (3000, 300, 9000, 5, 24),
    (3000, 120, 9000, -12, 24),
    (1500, 200, 15000, 0.8, 12),
    (3000, 250, None, 2.3, 12),
]
tensors["cases"] = torch.tensor(len(cases))
for case, (n, f0, peak, steps, bins) in enumerate(cases):
    samples = blip(n, f0, peak) if peak else clipped(n, f0)
    y = librosa.effects.pitch_shift(
        audiosegment_to_numpy(samples), sr=RATE, n_steps=steps, bins_per_octave=bins
    )
    tensors[f"{case}/input"] = torch.from_numpy(samples.astype(np.float32))
    tensors[f"{case}/steps"] = torch.from_numpy(
        np.array([steps], dtype=np.float64).view(np.int64)
    )
    tensors[f"{case}/bins_per_octave"] = torch.tensor(bins)
    tensors[f"{case}/output"] = torch.from_numpy(y)

# The steps before the inverse transform, which agree to the bit, for the shortest case.
samples = blip(1500, 200, 15000)
spectrum = librosa.stft(audiosegment_to_numpy(samples))
rate = 2.0 ** (-0.8 / 12)
stretched = librosa.phase_vocoder(spectrum, rate=rate)
tensors["steps/input"] = torch.from_numpy(samples.astype(np.float32))
tensors["steps/rate"] = torch.from_numpy(np.array([rate]).view(np.int64))
tensors["steps/stft"] = torch.from_numpy(spectrum.T.copy().view(np.float32))
tensors["steps/phase_vocoder"] = torch.from_numpy(stretched.T.copy().view(np.float32))

special = [
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.00002,
    1.4,
    -1.00004,
    -2.0,
    1e4,
    1e5,
    -1e5,
    1e6,
    np.inf,
    -np.inf,
    np.nan,
]
samples = np.array([*special, *rng.uniform(-1.5, 1.5, 200)])
for dtype in [np.float32, np.float64]:
    x = samples.astype(dtype)
    name = np.dtype(dtype).name
    tensors[f"convert/{name}/input"] = torch.from_numpy(
        x.view(np.int32 if dtype == np.float32 else np.int64).astype(np.int64)
    )
    tensors[f"convert/{name}/pcm"] = torch.from_numpy(
        numpy_to_audiosegment(x).astype(np.int64)
    )
pcm = np.arange(-32768, 32768, 61, dtype=np.int16)
tensors["convert/pcm"] = torch.from_numpy(pcm.astype(np.int64))
tensors["convert/float"] = torch.from_numpy(audiosegment_to_numpy(pcm))

fixture.save(
    "dsp",
    tensors,
    librosa=librosa.__version__,
    numpy=np.__version__,
    scipy=scipy.__version__,
)
