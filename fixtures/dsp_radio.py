"""The old radio service's effect on made-up speech, for the dsp crate's radio module.

The service's code is restated below as it was, around librosa, scipy and soundfile, from the
segment it decoded the speech into on. The clicks and static are made up, at the rates and
channels of the real ones (stereo 44.1 kHz, mono 11025 Hz, stereo 8 kHz), and read from WAV
files as the service read them, and the CRC-32 of each as it matched them to 48 kHz is saved;
dsp_radio_assets runs it on the real ones. The speech is mono, stereo and in three channels at
48 kHz, mono at the 44.1 kHz of the silicon voices, mono at 8 kHz, and none at all.

Where the port fixes the old service, the restatement takes the fix too: the speech is filtered
at the 48 kHz it was resampled to, the clicks and static are matched to 48 kHz, whatever the
speech's rate was, speech of more than two channels is averaged over them as stereo is, the
static loops under speech longer than it, the static after the speech goes on from where the
static under it left off, and speech of no samples is put between the clicks as any other, where
numpy could not take its peak.
"""

import io
import wave
import zlib

import librosa
import numpy as np
import scipy
import soundfile as sf
import torch
from pydub import AudioSegment
from scipy.signal import butter, lfilter

import common

OUTPUT_SR = 48000

# The WAV files the service read its clicks and static from, by name, which run() sets.
assets = None


# The old service's code, as it had it, but for the files it opens.
def audiosegment_to_numpy(seg):
    samples = np.array(seg.get_array_of_samples())

    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels))

    samples = samples.astype(np.float32) / (1 << (8 * seg.sample_width - 1))

    return samples, seg.frame_rate


def numpy_to_audiosegment(samples, sr, sample_width=2, channels=1):
    samples_int16 = (samples * 32767).astype(np.int16)

    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )


def bandpass(x, sr, low=300, high=3000, order=4):
    nyq = sr * 0.5
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, x)


def saturate(x, drive=2.5):
    return np.tanh(drive * x)


def normalize(x, peak=0.8):
    m = np.max(np.abs(x), initial=0)
    if m > 0:
        x = x / m * peak
    return x


def ensure_mono(x):
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def load_and_match(path, target_sr):
    audio, sr = sf.read(io.BytesIO(assets[path]))
    audio = ensure_mono(audio)

    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def radio_effect(base_audio):
    samples, numpy_sr = audiosegment_to_numpy(base_audio)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    click_on = load_and_match("mic_click_on.wav", OUTPUT_SR)
    click_off = load_and_match("mic_click_off.wav", OUTPUT_SR)
    static = load_and_match("diffstatic.wav", OUTPUT_SR)
    speech = librosa.resample(samples, orig_sr=numpy_sr, target_sr=8000)
    speech = librosa.resample(speech, orig_sr=8000, target_sr=OUTPUT_SR)
    speech = normalize(speech, 0.5)
    speech = bandpass(speech, OUTPUT_SR)
    speech = saturate(speech, 2)
    speech = normalize(speech, 0.7)
    static = loop(static, len(speech) + int(OUTPUT_SR * 0.1))
    speech = speech + static[: len(speech)] * 0.2
    output = np.concatenate([click_on, speech, static[len(speech) :], click_off])

    output = normalize(output, 0.9)

    return output


def loop(static, n):
    """The first `n` samples of `static` looped, as the port loops it under speech."""
    return np.tile(static, int(np.ceil(n / len(static))))[:n]


def wav(samples, rate, channels):
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


def pcm(x):
    # torch.from_numpy gives an empty array a stride of 0, which safetensors cannot view as bytes.
    return torch.from_numpy(x.astype(np.float32)) if x.size else torch.zeros(0)


def f64(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float64).view(np.int64))


def run(files, speeches):
    """The radio's output for each speech, (16-bit samples, rate, channels), with the clicks and
    static `files`, WAV bytes by name."""
    global assets
    assets = files
    tensors = {}
    tensors["cases"] = torch.tensor(len(speeches))
    for case, (samples, rate, channels) in enumerate(speeches):
        name = str(case)
        tensors[f"{name}/rate"] = torch.tensor(rate)
        tensors[f"{name}/channels"] = torch.tensor(channels)
        tensors[f"{name}/input"] = pcm(samples)
        seg = AudioSegment(
            samples.astype("<i2").tobytes(),
            frame_rate=rate,
            sample_width=2,
            channels=channels,
        )
        out = numpy_to_audiosegment(radio_effect(seg), OUTPUT_SR)
        tensors[f"{name}/output"] = pcm(np.frombuffer(out.raw_data, dtype=np.int16))
    nyq = OUTPUT_SR * 0.5
    b, a = butter(4, [300 / nyq, 3000 / nyq], btype="band")
    tensors["butter/b"] = f64(b)
    tensors["butter/a"] = f64(a)
    for name in files:
        matched = load_and_match(name, OUTPUT_SR).astype("<f8").tobytes()
        stem = name.removesuffix(".wav")
        tensors[f"matched/{stem}/len"] = torch.tensor(len(matched) // 8)
        tensors[f"matched/{stem}/crc32"] = torch.tensor(zlib.crc32(matched))
    return tensors


def provenance():
    return {
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def noise(seconds, rate, channels, level):
        n = int(seconds * rate)
        x = rng.normal(0, level, (n, channels)) * np.linspace(1, 0.3, n)[:, None]
        return np.clip(np.round(x), -32768, 32767).astype(np.int16).reshape(-1)

    def voice(seconds, rate, channels):
        n = int(seconds * rate)
        t = np.arange(n) / rate
        phase = (
            2 * np.pi * np.cumsum(140 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) / rate
        )
        x = sum(np.sin(k * phase) * np.exp(-k / 10) for k in range(1, 40))
        x = 12000 * x / np.max(np.abs(x)) * np.minimum(1, t / 0.01)
        x = np.stack([x * (0.6 + 0.4 * c) for c in range(channels)], axis=1)
        return np.round(x).astype(np.int16).reshape(-1)

    clips = {
        "mic_click_on.wav": (noise(0.03, 44100, 2, 9000), 44100, 2),
        "mic_click_off.wav": (noise(0.04, 11025, 1, 7000), 11025, 1),
        "diffstatic.wav": (noise(0.5, 8000, 2, 2500), 8000, 2),
    }
    tensors = {}
    files = {}
    for name, (samples, rate, channels) in clips.items():
        files[name] = wav(samples, rate, channels)
        stem = name.removesuffix(".wav")
        tensors[f"{stem}/samples"] = pcm(samples)
        tensors[f"{stem}/rate"] = torch.tensor(rate)
        tensors[f"{stem}/channels"] = torch.tensor(channels)
    speeches = [
        (voice(0.1, 48000, 1), 48000, 1),
        (voice(0.1, 48000, 2), 48000, 2),
        (voice(0.1, 44100, 1), 44100, 1),
        (voice(0.1, 8000, 1), 8000, 1),
        (voice(0.05, 48000, 3), 48000, 3),
        (np.zeros(0, dtype=np.int16), 48000, 1),
    ]
    tensors |= run(files, speeches)
    common.Fixture(seed=0).save("dsp", tensors, **provenance())
