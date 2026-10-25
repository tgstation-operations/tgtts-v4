"""The old frontend's finishing of the model's speech, for the dsp crate's speech module.

The service's code is restated below as it was, from the WAV the model sent on, around pydub and
librosa. The speech is made up, at the model's 48 kHz.
"""

import io
import os
import wave
import zlib

# numpy's AVX-512 kernels round differently from its AVX2 ones: pinned to AVX2, as fixtures.nix
# pins torch, any x86-64 host since Haswell writes the same bytes.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pydub  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402

tts_sample_rate = 48000


# The old service's code, as it had it.
def audiosegment_to_numpy(seg):
    samples = np.array(seg.get_array_of_samples())

    if seg.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32) / (1 << (8 * seg.sample_width - 1))

    return samples, seg.frame_rate


def numpy_to_audiosegment(samples, sr, sample_width=2, channels=1):
    samples_int16 = (samples * 32767).astype(np.int16)

    return pydub.AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )


def finish(response, pitch):
    """text_to_speech_handler for /tts, past the model's response and before the filters."""
    final_audio = pydub.AudioSegment.empty()
    sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response), "wav")
    sentence_silence = pydub.AudioSegment.silent(250, tts_sample_rate)
    sentence_audio += sentence_silence
    final_audio += sentence_audio

    if pitch != 0:
        numpy_audio, sr = audiosegment_to_numpy(final_audio)
        numpy_audio = librosa.effects.pitch_shift(
            numpy_audio, sr=sr, n_steps=pitch, bins_per_octave=24
        )
        final_audio = numpy_to_audiosegment(numpy_audio, sr)
    return final_audio


def voice(seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(150 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) / rate
    x = sum(np.sin(k * phase) * np.exp(-k / 9) for k in range(1, 40))
    return np.round(30000 * x / np.max(np.abs(x))).astype(np.int16)


speech = voice(0.15, tts_sample_rate)
out = io.BytesIO()
with wave.open(out, "wb") as file:
    file.setnchannels(1)
    file.setsampwidth(2)
    file.setframerate(tts_sample_rate)
    file.writeframes(speech.astype("<i2").tobytes())

tensors = {"input": torch.from_numpy(speech.astype(np.float32))}
pitches = [0, 5, -12]
tensors["pitches"] = torch.tensor(pitches)
for pitch in pitches:
    finished = finish(out.getvalue(), pitch)
    samples = np.frombuffer(finished.raw_data, dtype=np.int16)
    if pitch:
        tensors[f"{pitch}/output"] = torch.from_numpy(samples.astype(np.float32))
    else:
        tensors[f"{pitch}/output_len"] = torch.tensor(len(samples))
        tensors[f"{pitch}/output_crc32"] = torch.tensor(zlib.crc32(samples.tobytes()))

common.Fixture(seed=0).save("dsp", tensors, librosa=librosa.__version__)
