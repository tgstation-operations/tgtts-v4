"""EBU Tech 3341's test signals for loudness meters, in one channel, with what ffmpeg's ebur128
filter, an implementation of ITU-R BS.1770-4 apart from the dsp crate's, measures of them.

Cases 1 to 5 are tones for the integrated loudness, long enough that only their levels and
durations are saved, for the tests to make them again; cases 15 to 23 are short signals for the
true peak, saved whole, 20 to 23 made at four times the rate and filtered down to it as the
document has them. Its signals are stereo, both channels alike, so a channel of each is 10 log10 2
dB less loud than it expects of the pair; the true peak is a channel's own.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
import torch

import common

RATE = 48000

tensors = {}


def f64(*values):
    return torch.from_numpy(np.array(values, dtype=np.float64).view(np.int64))


def ebur128(samples, subtype):
    """The integrated loudness and the true peak, in dBTP, that ffmpeg reads of `samples`."""
    with tempfile.TemporaryDirectory() as tmp:
        wav, log = Path(tmp) / "in.wav", Path(tmp) / "meta.txt"
        sf.write(wav, samples, RATE, subtype=subtype)
        metadata = f"ebur128=peak=true:metadata=1,ametadata=mode=print:file={log}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", wav, "-af", metadata]
            + ["-f", "null", "-"],
            check=True,
        )
        last = {}
        for line in log.read_text().splitlines():
            key, _, value = line.partition("=")
            last[key] = value
    true_peak = float(last["lavfi.r128.true_peak"])
    return float(last["lavfi.r128.I"]), 20 * math.log10(true_peak)


def tone(levels, seconds):
    """A 1 kHz sine at each level, in dBFS, for each duration, its phase running on."""
    amplitude = np.concatenate(
        [
            np.full(round(s * RATE), 10 ** (level / 20))
            for level, s in zip(levels, seconds)
        ]
    )
    n = np.arange(len(amplitude))
    return amplitude * np.sin(2 * np.pi * 1000 * n / RATE)


def tapered(x, ms, rate):
    ramp = np.linspace(0, 1, rate * ms // 1000)
    x = x.copy()
    x[: len(ramp)] *= ramp
    x[len(x) - len(ramp) :] *= ramp[::-1]
    return x


def sine(period, amplitude, degrees):
    """0.1 s of a sine of `period` samples, tapered over 10 ms at each end."""
    n = np.arange(RATE // 10)
    x = amplitude * np.sin(2 * np.pi * n / period + math.radians(degrees))
    return tapered(x, 10, RATE)


def burst(offset):
    """A sine of RATE / 6 at 0.5 with one period of RATE / 4 at 1.0 in it, continuous in phase,
    made at four times the rate and filtered down to it from `offset`."""
    fast = 4 * RATE
    slow, single = 4 * 6, 4 * 4
    start = 400 * slow
    n = np.arange(2 * start + single)
    x = 0.5 * np.sin(2 * np.pi * np.where(n < start, n, n - single) / slow)
    peak = (n >= start) & (n < start + single)
    x[peak] = np.sin(2 * np.pi * (n[peak] - start) / single)
    return scipy.signal.resample_poly(tapered(x, 2, fast)[offset:], 1, 4)


stereo = 10 * math.log10(2)
tones = {
    1: ([-23.0], [20.0], -23.0),
    2: ([-33.0], [20.0], -33.0),
    3: ([-36.0, -23.0, -36.0], [10.0, 60.0, 10.0], -23.0),
    4: ([-72.0, -36.0, -23.0, -36.0, -72.0], [10.0, 10.0, 60.0, 10.0, 10.0], -23.0),
    5: ([-26.0, -20.0, -26.0], [20.0, 20.1, 20.0], -23.0),
}
for case, (levels, seconds, expected) in tones.items():
    loudness, _ = ebur128(tone(levels, seconds), "DOUBLE")
    tensors[f"tone/{case}/levels"] = f64(*levels)
    tensors[f"tone/{case}/seconds"] = f64(*seconds)
    tensors[f"tone/{case}/expected"] = f64(expected - stereo)
    tensors[f"tone/{case}/ffmpeg"] = f64(loudness)

peaks = {
    15: (sine(4, 0.5, 0.0), -6.0),
    16: (sine(4, 0.5, 45.0), -6.0),
    17: (sine(6, 0.5, 60.0), -6.0),
    18: (sine(8, 0.5, 67.5), -6.0),
    19: (sine(4, 1.41, 45.0), 3.0),
    20: (burst(0), 0.0),
    21: (burst(1), 0.0),
    22: (burst(2), 0.0),
    23: (burst(3), 0.0),
}
for case, (samples, expected) in peaks.items():
    samples = samples.astype(np.float32)
    _, true_peak = ebur128(samples, "FLOAT")
    tensors[f"peak/{case}/samples"] = torch.from_numpy(samples)
    tensors[f"peak/{case}/expected"] = f64(expected)
    tensors[f"peak/{case}/ffmpeg"] = f64(true_peak)

ffmpeg = subprocess.run(
    ["ffmpeg", "-version"], capture_output=True, text=True, check=True
)
common.Fixture(seed=0).save(
    "dsp",
    tensors,
    tones=",".join(map(str, tones)),
    peaks=",".join(map(str, peaks)),
    ffmpeg=ffmpeg.stdout.split()[2],
    numpy=np.__version__,
    scipy=scipy.__version__,
)
