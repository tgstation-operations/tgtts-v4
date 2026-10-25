"""The old blips service's blips of a few texts, for the dsp crate's blips module.

The service's code is restated below as it was, around pydub and librosa, with its samples read
from WAV files as it read them and its calls to random.random() recorded; dsp_blips_voice runs it
on a real voice. The voice here is made up: voiced blips of a few letters, digits, Deska and Gwah,
one a click, at 16 kHz to keep the file small, the Gwah so padded with silence that what is left
once it is stripped is shorter than a crossfade. The frontend's messages are made in the real
voice only: its silence is at 48 kHz, which the port has a voice be at too. Texts without capitals, drawn at 0.5, are never
detuned, and their blips are pydub's integer arithmetic alone; the rest are shifted, and how
often is saved beside them.

Where the port fixes the old service, the restatement takes the fix too: text with a letter or
digit of either case makes a sound, a crossfade is no longer than either side, so a sentence can
start past its first character and take blips shorter than a crossfade, the blips of ? and !
are faded as the letters' are, and the conversion back to 16 bits holds samples at full scale
where it wrapped around. The frontend passes the pitch on, and puts nothing, not even its
silence, for a sentence without a sound.
"""

import importlib.metadata
import io
import math
import os
import random
import wave
import zlib

# numpy's AVX-512 kernels round differently from its AVX2 ones: pinned to AVX2, as fixtures.nix
# pins torch, any x86-64 host since Haswell writes the same bytes.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pysbd  # noqa: E402
import torch  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from pydub.silence import detect_leading_silence  # noqa: E402

import common  # noqa: E402

BASE, NUMBER = "male", "1"
CROSSFADE_TOO_LONG = 1

tts_sample_rate = 48000

# The voice's blips as the service kept them, which run() sets.
blips_cache = None


def wav(samples, rate):
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


# The old service's helpers, as it had them.
def audiosegment_to_numpy(seg):
    samples = np.array(seg.get_array_of_samples())
    if seg.channels == 2:
        samples = samples.reshape((-1, 2))
    samples = samples.astype(np.float32) / (1 << (8 * seg.sample_width - 1))
    return samples, seg.frame_rate


def numpy_to_audiosegment(samples, sr, sample_width=2, channels=1):
    samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=channels,
    )


trim_leading_silence = lambda x: x[detect_leading_silence(x) :]  # noqa: E731
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()  # noqa: E731
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))  # noqa: E731


def change_volume(seg, multiplier):
    return seg.apply_gain(20 * math.log10(multiplier))


def normalize_to_target(seg, target_dbfs=-20.0):
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def cap_loudness(seg, max_dbfs=-1.0):
    if seg.max_dBFS > max_dbfs:
        change = max_dbfs - seg.max_dBFS
        return seg.apply_gain(change)
    return seg


def append(result_sound, blip):
    """`blip` crossfaded onto `result_sound` by 150 ms, or as much of that as both have."""
    return result_sound.append(blip, crossfade=min(150, len(result_sound), len(blip)))


def fade(blip):
    """`blip` faded in over 3 ms and out over 8, or over all of it where it is shorter."""
    if len(blip) == 0:
        return blip
    return blip.fade_in(min(3, len(blip))).fade_out(min(8, len(blip)))


def text_to_speech_blips(text, voice, blip_base, blip_number, pitch):
    """The service's /generate-tts-blips, from the text on, returning the WAV it sent."""
    skip_these = " ,:;'\""
    actual_text_found = False
    with io.BytesIO() as data_bytes:
        for i, letter in enumerate(text):
            if letter.isalpha() or letter.isnumeric():
                actual_text_found = True
                break
        if not actual_text_found:
            stub_file = AudioSegment.empty()
            stub_file.set_frame_rate(48000)
            stub_file.export(data_bytes, format="wav")
            return data_bytes.getvalue()
        result_sound = AudioSegment.empty()
        for i, letter in enumerate(text):
            if not letter.isalpha() and not letter.isnumeric():
                if letter in skip_these:
                    continue
                else:
                    if letter == " ":
                        continue
                    if letter == "?" or letter == "!":
                        letter_sound = AudioSegment.from_file(
                            io.BytesIO(
                                blips_cache[voice][blip_base][str(blip_number)][
                                    "Deska" if letter == "?" else "Gwah"
                                ].getvalue()
                            ),
                            format="wav",
                        )
                        samples, sr = audiosegment_to_numpy(letter_sound)
                        detune = 0
                        base_pitch = 0
                        random_pitch = 0.2
                        base_var = 0.2
                        detune = ((0 + base_pitch) * 100) + (
                            (random.random() * (300 + 300) - 300)
                            * (base_var + random_pitch)
                        )
                        semitones = detune / 100
                        if semitones != 0:
                            samples = librosa.effects.pitch_shift(
                                samples, sr=sr, n_steps=semitones
                            )
                        speech_audio = numpy_to_audiosegment(
                            samples,
                            sr,
                            sample_width=letter_sound.sample_width,
                            channels=letter_sound.channels,
                        )
                        speech_audio = change_volume(speech_audio, 0.6)
                        stripped_sound = strip_silence(speech_audio)
                        result_sound = append(result_sound, fade(stripped_sound))
            else:
                if not i % 2 == 0:
                    continue
                letter_sound = AudioSegment.from_file(
                    io.BytesIO(
                        blips_cache[voice][blip_base][str(blip_number)][
                            letter.lower()
                        ].getvalue()
                    ),
                    format="wav",
                )
                samples, sr = audiosegment_to_numpy(letter_sound)
                detune = 0
                base_pitch = 1.6 if letter.isupper() else 0
                random_pitch = 0.15 if letter.isupper() else 0
                base_var = 0.2
                detune = ((0 + base_pitch) * 100) + (
                    (random.random() * (300 + 300) - 300) * (base_var + random_pitch)
                )
                semitones = detune / 100
                if semitones != 0:
                    samples = librosa.effects.pitch_shift(
                        samples, sr=sr, n_steps=semitones
                    )
                if pitch != "0":
                    samples = librosa.effects.pitch_shift(
                        samples, sr=sr, n_steps=int(pitch), bins_per_octave=24
                    )
                new_audio = numpy_to_audiosegment(
                    samples,
                    sr,
                    sample_width=letter_sound.sample_width,
                    channels=letter_sound.channels,
                )
                new_audio = change_volume(new_audio, 0.7 if letter.isupper() else 0.5)
                stripped_sound = strip_silence(new_audio)  # noqa: F841
                result_sound = append(result_sound, fade(new_audio))
        result_sound.export(data_bytes, format="wav")
        rawsound = AudioSegment.from_file(io.BytesIO(data_bytes.getvalue()), "wav")
        normalizedsound = normalize_to_target(rawsound, -25)
        normalizedsound = cap_loudness(normalizedsound, max_dbfs=-5)
        normalizedsound.export(data_bytes, format="wav")
        return data_bytes.getvalue()


segmenter = pysbd.Segmenter(language="en", clean=True)


def frontend(text, voice, blip_base, blip_number, pitch):
    """The frontend's /tts-blips up to its ffmpeg: a request per sentence, with the pitch."""
    final_audio = AudioSegment.empty()
    for sentence in segmenter.segment(text):
        response = text_to_speech_blips(sentence, voice, blip_base, blip_number, pitch)
        sentence_audio = AudioSegment.from_file(io.BytesIO(response), "wav")
        if not sentence_audio.frame_count():
            continue
        sentence_silence = AudioSegment.silent(250, tts_sample_rate)
        sentence_audio += sentence_silence
        final_audio += sentence_audio
    return final_audio


class Counted:
    """librosa.effects.pitch_shift, counting its calls."""

    def __init__(self, pitch_shift):
        self.pitch_shift = pitch_shift
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.pitch_shift(*args, **kwargs)


class Recorded:
    """random.random(), or 0.5, with every draw kept."""

    def __init__(self, seed):
        self.source = random.Random(seed) if seed is not None else None
        self.draws = []

    def __call__(self):
        draw = self.source.random() if self.source else 0.5
        self.draws.append(draw)
        return draw


PITCH_SHIFT = librosa.effects.pitch_shift


def run(voice, rate, cases, messages, checksum_exact=False):
    """The service's blips of each case, (text, pitch, seed), and the frontend's of each message,
    alike, in `voice`, a dict of 16-bit samples by key; a seed of None draws 0.5. Where
    nothing was shifted, `checksum_exact` saves the CRC-32 of the samples rather than them."""
    global blips_cache
    blips_cache = {
        "voice": {
            BASE: {NUMBER: {k: io.BytesIO(wav(v, rate)) for k, v in voice.items()}}
        }
    }
    tensors = {}

    def save(name, draws, run):
        random.random = draws
        shifts = librosa.effects.pitch_shift = Counted(PITCH_SHIFT)
        try:
            seg = run()
        except ValueError:
            tensors[f"{name}/error"] = torch.tensor(CROSSFADE_TOO_LONG)
            seg = None
        else:
            tensors[f"{name}/error"] = torch.tensor(0)
        # torch.from_numpy gives an empty array a stride of 0, which safetensors cannot view as
        # bytes.
        bits = np.array(draws.draws, dtype=np.float64).view(np.int64)
        tensors[f"{name}/draws"] = (
            torch.from_numpy(bits) if bits.size else torch.zeros(0, dtype=torch.int64)
        )
        tensors[f"{name}/shifts"] = torch.tensor(shifts.calls)
        if seg is not None:
            if seg.frame_count():
                assert (seg.frame_rate, seg.sample_width, seg.channels) == (rate, 2, 1)
            if checksum_exact and not shifts.calls:
                tensors[f"{name}/output_len"] = torch.tensor(int(seg.frame_count()))
                tensors[f"{name}/output_crc32"] = torch.tensor(zlib.crc32(seg.raw_data))
            else:
                pcm = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
                tensors[f"{name}/output"] = (
                    torch.from_numpy(pcm) if pcm.size else torch.zeros(0)
                )

    tensors["cases"] = torch.tensor(len(cases))
    for case, (text, pitch, seed) in enumerate(cases):
        tensors[f"{case}/pitch"] = torch.tensor(int(pitch))
        save(
            str(case),
            Recorded(seed),
            lambda: AudioSegment.from_file(
                io.BytesIO(text_to_speech_blips(text, "voice", BASE, NUMBER, pitch)),
                "wav",
            ),
        )
    tensors["messages"] = torch.tensor(len(messages))
    for case, (text, pitch, seed) in enumerate(messages):
        tensors[f"message/{case}/pitch"] = torch.tensor(int(pitch))
        save(
            f"message/{case}",
            Recorded(seed),
            lambda: frontend(text, "voice", BASE, NUMBER, pitch),
        )
    return tensors


def provenance(cases, messages):
    return {
        "texts": "\n".join(text for text, _, _ in cases),
        "messages": "\n".join(text for text, _, _ in messages),
        "librosa": librosa.__version__,
        "pydub": importlib.metadata.version("pydub"),
        "pysbd": importlib.metadata.version("pysbd"),
    }


if __name__ == "__main__":
    rate = 16000
    rng = np.random.default_rng(0)

    def blip(f0, ms, lead_ms=0, tail_ms=0, decay=0.08):
        """A voiced syllable that starts sharply and decays, between silences."""
        n = rate * ms // 1000
        t = np.arange(n) / rate
        phase = 2 * np.pi * np.cumsum(f0 * (1 - 0.2 * t)) / rate
        x = sum(np.sin(k * phase) * np.exp(-k / 6) for k in range(1, 30))
        x = x * np.minimum(1, t / 0.008) * np.exp(-t / decay)
        x = np.round(14000 * x / np.max(np.abs(x)) + rng.normal(0, 20, n))
        silence = lambda ms: np.zeros(rate * ms // 1000)  # noqa: E731
        return np.concatenate([silence(lead_ms), x, silence(tail_ms)]).astype(np.int16)

    voice = {
        "a": blip(220, 180),
        "b": blip(180, 170),
        # A click, whose peak the blips' -25 dBFS takes past -5 dBFS.
        "c": blip(400, 170, decay=0.004),
        "h": blip(260, 175),
        "l": blip(200, 170),
        "o": blip(240, 185),
        "1": blip(300, 180),
        "2": blip(150, 170),
        "Deska": blip(330, 190, lead_ms=40, tail_ms=60),
        "Gwah": blip(170, 120, lead_ms=50, tail_ms=70),
    }
    cases = [
        ("hob 12 lab", "0", None),
        ("c1", "0", None),
        ("ab? 1b?", "0", None),
        ("oh, hob a lab.", "0", None),
        ("'Hob", "0", None),
        ("!A", "0", None),
        ("Ha!", "0", None),
        ("Hob lab? AH 12", "0", 1),
        ("hob 1", "5", 2),
        ("Lab 2 oh", "-12", 3),
        ("!.a", "0", None),
    ]
    tensors = {
        f"voice/{key}": torch.from_numpy(samples.astype(np.float32))
        for key, samples in voice.items()
    }
    tensors |= run(voice, rate, cases, [])
    common.Fixture(seed=0).save("dsp", tensors, **provenance(cases, []))
