"""The old radio-gibberish service's masking and ducking, for the dsp crate's gibberish module.

The service's code is restated below as it was, from the letters' times its alignment gave on,
with the helpers it shares with the radio service taken from dsp_radio and its calls to Python's
random recorded. The letters' times are made up, as are the speech and the clicks and static.

Where the port fixes the old service, the restatement takes the fix too: the speech is filtered
and ducked at the 48 kHz it was resampled to, and the static loops under speech longer than it.
"""

import json
import os
import random

# numpy's AVX-512 kernels round differently from its AVX2 ones: pinned to AVX2, as fixtures.nix
# pins torch, any x86-64 host since Haswell writes the same bytes.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pydub import AudioSegment  # noqa: E402

import common  # noqa: E402
import dsp_radio  # noqa: E402
from dsp_radio import (  # noqa: E402
    OUTPUT_SR,
    audiosegment_to_numpy,
    bandpass,
    load_and_match,
    loop,
    normalize,
    numpy_to_audiosegment,
    saturate,
)

SPAN_PAST_END = 1


# The old service's code, as it had it.
def find_corrupted_spans(clean_text, corrupted_text, char_timestamps):
    clean_chars   = [(i, c) for i, c in enumerate(clean_text)   if c.isalpha() or c == " "]  # fmt: skip
    corrupt_chars = [(i, c) for i, c in enumerate(corrupted_text) if c.isalpha() or c == " " or not c.isspace()]  # fmt: skip

    # First pass: find which alpha indices are corrupted
    corrupted_alpha_indices = set()
    alpha_idx = 0
    for idx, (_, clean_char) in enumerate(clean_chars):
        if clean_char == " ":
            continue
        if idx < len(corrupt_chars) and not corrupt_chars[idx][1].isalpha():
            corrupted_alpha_indices.add(alpha_idx)
        alpha_idx += 1

    # Second pass: build spans
    corrupted_spans = []
    ts_idx = 0

    for idx, (_, clean_char) in enumerate(clean_chars):
        if idx >= len(corrupt_chars):
            break
        corrupt_char = corrupt_chars[idx][1]

        if clean_char == " ":
            # Check if the alpha char before or after this space is corrupted
            prev_alpha = ts_idx - 1
            next_alpha = ts_idx
            adjacent_corrupted = (
                prev_alpha in corrupted_alpha_indices
                or next_alpha in corrupted_alpha_indices
            )
            if not corrupt_char.isspace() and adjacent_corrupted:
                if ts_idx > 0 and ts_idx < len(char_timestamps):
                    corrupted_spans.append(
                        (
                            char_timestamps[ts_idx - 1]["end"],
                            char_timestamps[ts_idx]["start"],
                        )
                    )
        else:
            if not corrupt_char.isalpha() and ts_idx < len(char_timestamps):
                corrupted_spans.append(
                    (
                        char_timestamps[ts_idx]["start"],
                        char_timestamps[ts_idx]["end"],
                    )
                )
            ts_idx += 1

    return merge_spans(corrupted_spans, gap_threshold=0.05)


def merge_spans(spans, gap_threshold=0.05):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= gap_threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def apply_corruption_static(
    speech_audio,
    corrupted_spans,
    static,
    sr,
    duck_gain=0,
    static_gain=0.85,
    pre_mask_ms=60,  # bleed into preceding audio
    post_mask_ms=80,  # tail after character ends
    fade_ms=25,
):
    total = len(speech_audio)
    duck_envelope = np.ones(total)

    pre = int(pre_mask_ms * sr / 1000)
    post = int(post_mask_ms * sr / 1000)
    fade = int(fade_ms * sr / 1000)

    for start_sec, end_sec in corrupted_spans:
        s = max(0, int(start_sec * sr) - pre)
        e = min(total, int(end_sec * sr) + post)

        duck_envelope[s:e] = duck_gain

        # Fade in (speech -> ducked)
        fade_in_len = min(fade, (e - s) // 2)
        duck_envelope[s : s + fade_in_len] = np.linspace(1.0, duck_gain, fade_in_len)

        # Fade out (ducked -> speech), longer so word endings stay buried
        fade_out_len = min(fade * 2, (e - s) // 2)
        duck_envelope[e - fade_out_len : e] = np.linspace(duck_gain, 1.0, fade_out_len)

    ducked_speech = speech_audio * duck_envelope
    static_mask = build_static_mask(
        corrupted_spans,
        static,
        total,
        sr,
        pre_ms=pre_mask_ms,
        post_ms=post_mask_ms,
        fade_ms=fade_ms,
        static_gain=static_gain,
    )
    return ducked_speech + static_mask


def build_static_mask(
    corrupted_spans,
    static,
    total_samples,
    sr,
    pre_ms=40,
    post_ms=40,
    fade_ms=25,
    static_gain=0.8,
):
    pre = int(pre_ms * sr / 1000)
    post = int(post_ms * sr / 1000)
    fade = int(fade_ms * sr / 1000)

    mask_envelope = np.zeros(total_samples)

    for start_sec, end_sec in corrupted_spans:
        s = max(0, int(start_sec * sr) - pre)
        e = min(total_samples, int(end_sec * sr) + post)

        mask_envelope[s:e] = 1.0

        fade_in_len = min(fade, (e - s) // 2)
        fade_out_len = min(fade * 2, (e - s) // 2)
        mask_envelope[s : s + fade_in_len] *= np.linspace(0, 1, fade_in_len)
        mask_envelope[e - fade_out_len : e] *= np.linspace(1, 0, fade_out_len)

    static_full = np.tile(static, int(np.ceil(total_samples / len(static))))[
        :total_samples
    ]
    return static_full * mask_envelope * static_gain


def radio_effect(base_audio, corrupted_spans):
    """The service's radio_effect, from the spans it found on."""
    click_on = load_and_match("mic_click_on.wav", OUTPUT_SR)
    click_off = load_and_match("mic_click_off.wav", OUTPUT_SR)
    static = load_and_match("diffstatic.wav", OUTPUT_SR)
    samples, numpy_sr = audiosegment_to_numpy(base_audio)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    speech = librosa.resample(samples, orig_sr=numpy_sr, target_sr=8000)
    speech = librosa.resample(speech, orig_sr=8000, target_sr=OUTPUT_SR)
    speech = normalize(speech, 0.5)
    speech = bandpass(speech, OUTPUT_SR)
    speech = saturate(speech, 2)
    speech = normalize(speech, 0.7)

    # Duck speech + inject static bursts
    speech = apply_corruption_static(
        speech, corrupted_spans, static, OUTPUT_SR, duck_gain=0, static_gain=0.85
    )

    static = loop(static, len(speech))
    speech = speech + static * 0.2
    output = np.concatenate(
        [click_on, speech, static[: int(OUTPUT_SR * 0.1)], click_off]
    )

    output = normalize(output, 0.9)

    return output


def mask_string(text):
    words = text.split()
    length = len(words)
    percent = random.uniform(0.35, 0.65)
    num_to_replace = int(length * percent)
    indices = random.sample(range(length), num_to_replace)
    for i in indices:
        words[i] = "#" * len(words[i])
    return " ".join(words)


class Recorded(random.Random):
    """Python's generator, with each draw kept as (kind, bound, value): 0 for random(), its
    value as the bits of the double, and 1 for _randbelow(bound)."""

    def __init__(self, seed):
        super().__init__(seed)
        self.draws = []

    def random(self):
        value = super().random()
        self.draws.append((0, 0, int(np.float64(value).view(np.int64))))
        return value

    def _randbelow(self, n):
        value = super()._randbelow(n)
        self.draws.append((1, n, value))
        return value


def letters(text, start=0.1, step=0.07, length=0.05):
    """Made-up times for the characters the alignment times: letters and apostrophes."""
    timed = [c for c in text.upper() if c.isalpha() or c == "'"]
    return [
        {"start": start + i * step, "end": start + i * step + length}
        for i in range(len(timed))
    ]


texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Hi, Bob. It's 5 o'clock, AI.",
    "help",
    "Security to the bridge, the clown is loose and armed with a banana.",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "  spaced   out  ",
]
tensors = {}
cases = []
for case, text in enumerate(texts):
    rng = Recorded(case)
    random.uniform, random.sample = rng.uniform, rng.sample
    masked = mask_string(text)
    timestamps = letters(text)
    spans = find_corrupted_spans(text, masked, timestamps)
    cases.append({"text": text, "masked": masked})
    tensors[f"mask/{case}/draws"] = torch.tensor(rng.draws, dtype=torch.int64).reshape(
        -1, 3
    )
    times = [(t["start"], t["end"]) for t in timestamps]
    tensors[f"mask/{case}/letters"] = dsp_radio.f64(times).reshape(-1, 2)
    tensors[f"mask/{case}/spans"] = dsp_radio.f64(spans).reshape(-1, 2)

rng = np.random.default_rng(0)


def noise(seconds, rate, channels, level):
    n = int(seconds * rate)
    x = rng.normal(0, level, (n, channels)) * np.linspace(1, 0.3, n)[:, None]
    return np.clip(np.round(x), -32768, 32767).astype(np.int16).reshape(-1)


def voice(seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(160 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) / rate
    x = sum(np.sin(k * phase) * np.exp(-k / 10) for k in range(1, 40))
    return np.round(12000 * x / np.max(np.abs(x))).astype(np.int16)


clips = {
    "mic_click_on.wav": (noise(0.03, 44100, 2, 9000), 44100, 2),
    "mic_click_off.wav": (noise(0.04, 11025, 1, 7000), 11025, 1),
    "diffstatic.wav": (noise(0.5, 8000, 2, 2500), 8000, 2),
}
dsp_radio.assets = {}
for name, (samples, rate, channels) in clips.items():
    dsp_radio.assets[name] = dsp_radio.wav(samples, rate, channels)
    stem = name.removesuffix(".wav")
    tensors[f"{stem}/samples"] = dsp_radio.pcm(samples)
    tensors[f"{stem}/rate"] = torch.tensor(rate)
    tensors[f"{stem}/channels"] = torch.tensor(channels)

# A span runs past the speech's end, one starts past it, and the last speech outlasts the static,
# which loops under it.
radios = [
    (voice(0.2, 48000), 48000, [(0.02, 0.05), (0.07, 0.09), (0.15, 0.3)]),
    (voice(0.08, 44100), 44100, [(0.0, 0.01)]),
    (voice(0.08, 48000), 48000, [(0.02, 0.03), (0.2, 0.25)]),
    (voice(0.6, 8000), 8000, [(0.1, 0.2)]),
]
for case, (samples, rate, spans) in enumerate(radios):
    name = f"radio/{case}"
    tensors[f"{name}/rate"] = torch.tensor(rate)
    tensors[f"{name}/input"] = dsp_radio.pcm(samples)
    tensors[f"{name}/spans"] = dsp_radio.f64(spans)
    seg = AudioSegment(
        samples.astype("<i2").tobytes(), frame_rate=rate, sample_width=2, channels=1
    )
    try:
        out = numpy_to_audiosegment(radio_effect(seg, spans), OUTPUT_SR)
    except ValueError:
        tensors[f"{name}/error"] = torch.tensor(SPAN_PAST_END)
        continue
    tensors[f"{name}/error"] = torch.tensor(0)
    tensors[f"{name}/output"] = dsp_radio.pcm(
        np.frombuffer(out.raw_data, dtype=np.int16)
    )

common.Fixture(seed=0).save(
    "dsp",
    tensors,
    cases=json.dumps(cases),
    radios=str(len(radios)),
    **dsp_radio.provenance(),
)
