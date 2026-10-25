"""pydub's operations on 16-bit mono PCM, for the dsp crate's segment module.

Each case is a segment at a rate where milliseconds and frames disagree, mostly low ones to keep
the file small (below 1000 Hz, pydub's overlay of a short crossfade drops frames), with loud
stretches that clip and quiet ones about the -50 dBFS that silence detection looks for. The operations run on it with parameters drawn per case; an operation pydub
refuses is saved as the error it raised.
"""

import importlib.metadata
import math
import warnings

import numpy as np
import torch

import common

with warnings.catch_warnings():
    # pydub warns at import that it has no ffmpeg, which none of this needs.
    warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment, effects
    from pydub.exceptions import TooManyMissingFrames
    from pydub.silence import detect_leading_silence

# The old service's own helpers.
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]  # noqa: E731
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()  # noqa: E731
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))  # noqa: E731

RATES = [1000, 1500, 2205, 3001, 4410, 8000, 1000, 2205, 3001, 1500, 700, 700]
# 120.5 ms, which pydub's round() takes to 120.
TIE = (2000, 241)
# Where a crossfade of 3 ms fades out frames that the overlay then leaves out.
DROPPED = (900, 70)
CROSSFADE_TOO_LONG = 1
TOO_MANY_MISSING_FRAMES = 2
EMPTY_FADE = 3

fixture = common.Fixture(seed=0)
rng = np.random.default_rng(0)
tensors = {}


def pcm(segment):
    samples = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32)
    # torch.from_numpy gives an empty array a stride of 0, which safetensors cannot view as bytes.
    return torch.from_numpy(samples) if samples.size else torch.zeros(0)


def f64(*values):
    return torch.from_numpy(np.array(values, dtype=np.float64).view(np.int64))


def i64(*values):
    return torch.tensor(values, dtype=torch.int64)


def save(name, run):
    """Save what `run` gives under `name`, and under `name/error` 0, or what pydub raised."""
    try:
        result = run()
    except ValueError:
        tensors[f"{name}/error"] = i64(CROSSFADE_TOO_LONG)
        return
    except TooManyMissingFrames:
        tensors[f"{name}/error"] = i64(TOO_MANY_MISSING_FRAMES)
        return
    except ZeroDivisionError:
        tensors[f"{name}/error"] = i64(EMPTY_FADE)
        return
    tensors[f"{name}/error"] = i64(0)
    tensors[name] = pcm(result)


def samples(rate, frames=None):
    """Noise in stretches of random level, some silent, some about the threshold, some clipped."""
    if frames is None:
        frames = int(rate * rng.uniform(0.02, 0.3 if rate < 4000 else 0.1))
    out = np.zeros(frames)
    start = 0
    while start < frames:
        end = min(frames, start + int(rng.integers(1, max(2, frames // 3))))
        level = rng.choice([0, 0, 60, 100, 110, 150, 3000, 20000, 40000])
        out[start:end] = rng.normal(0, level, end - start) if level else 0
        start = end
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


segments = [
    AudioSegment(
        samples(rate, frames).tobytes(), frame_rate=rate, sample_width=2, channels=1
    )
    for rate, frames in [*((rate, None) for rate in RATES), TIE, DROPPED]
]
tensors["cases"] = i64(len(segments))
for case, seg in enumerate(segments):
    name = str(case)
    tensors[f"{name}/rate"] = i64(seg.frame_rate)
    tensors[f"{name}/input"] = pcm(seg)
    tensors[f"{name}/len_ms"] = i64(len(seg))
    tensors[f"{name}/rms"] = i64(seg.rms)
    tensors[f"{name}/max"] = i64(seg.max)
    tensors[f"{name}/dbfs"] = f64(seg.dBFS)
    tensors[f"{name}/max_dbfs"] = f64(seg.max_dBFS)
    tensors[f"{name}/leading_silence"] = i64(detect_leading_silence(seg))
    save(f"{name}/strip_silence", lambda: strip_silence(seg))

    gains = [
        rng.uniform(-30, 30),
        rng.uniform(-3, 3),
        20 * math.log10(rng.choice([0.5, 0.6, 0.7])),
        float("inf"),
        -float("inf"),
    ]
    tensors[f"{name}/gain"] = f64(*gains)
    for k, db in enumerate(gains):
        save(f"{name}/gain/{k}", lambda: seg.apply_gain(db))

    headrooms = [0.1, rng.uniform(-1, 10)]
    tensors[f"{name}/normalize"] = f64(*headrooms)
    for k, headroom in enumerate(headrooms):
        save(f"{name}/normalize/{k}", lambda: effects.normalize(seg, headroom))

    fades = [3, 8, int(rng.integers(1, 100)), int(rng.integers(100, 300))]
    tensors[f"{name}/fade"] = i64(*fades)
    for k, duration in enumerate(fades):
        save(f"{name}/fade_in/{k}", lambda: seg.fade_in(duration))
        save(f"{name}/fade_out/{k}", lambda: seg.fade_out(duration))

    slices = []
    for _ in range(4):
        start = int(rng.integers(-len(seg) - 5, len(seg) + 5))
        end = int(rng.integers(-len(seg) - 5, len(seg) + 5))
        has_start, has_end = rng.random() < 0.8, rng.random() < 0.8
        slices.append([has_start, start, has_end, end])
    tensors[f"{name}/slice"] = torch.tensor(slices, dtype=torch.int64).reshape(-1, 4)
    for k, (has_start, start, has_end, end) in enumerate(slices):
        save(
            f"{name}/slice/{k}",
            lambda: seg[(start if has_start else None) : (end if has_end else None)],
        )

    appends = [[case, 3]] if (seg.frame_rate, int(seg.frame_count())) == DROPPED else []
    for _ in range(3):
        others = [i for i, s in enumerate(segments) if s.frame_rate == seg.frame_rate]
        other = int(rng.choice(others))
        crossfade = int(
            rng.choice([0, 1, 3, 50, 100, 101, 150, int(rng.integers(1, 300))])
        )
        appends.append([other, crossfade])
    tensors[f"{name}/append"] = torch.tensor(appends, dtype=torch.int64).reshape(-1, 2)
    for k, (other, crossfade) in enumerate(appends):
        save(
            f"{name}/append/{k}",
            lambda: seg.append(segments[other], crossfade=crossfade),
        )

for duration in [250, 1, 33]:
    for rate in [48000, 44100, 11025, 3001]:
        silent = AudioSegment.silent(duration, rate)
        assert not any(silent.raw_data)
        tensors[f"silent/{duration}/{rate}"] = i64(int(silent.frame_count()))

fixture.save(
    "dsp",
    tensors,
    pydub=importlib.metadata.version("pydub"),
    audioop=importlib.metadata.version("audioop-lts"),
)
