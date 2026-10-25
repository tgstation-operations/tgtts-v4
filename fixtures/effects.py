"""The game's ffmpeg voice filters on made-up speech, for the effects crate to compare its own
graphs with.

Each case is a filter graph as the game sends it, its placeholders filled as the game and the old
service filled them, beside the effects the game sends in its place. ffmpeg runs the graph on a
made-up voice at 48 kHz, its harmonics and formants moving through a few syllables and a fricative,
and its output is saved as a log-mel spectrum, which is what the comparison reads: a perceptual
likeness is the bar, not the samples. Three cases are not what the game sends. The fishing spot
capsule asks acompressor for a ratio of 40, which ffmpeg refuses, failing the request, so it is run
at 20, ffmpeg's most, while the effects keep 40. The fish tongue's afftfilt is run as its comment
intends, its cutoff swinging in seconds rather than in samples and its lowpass clamped at zero,
since the effects follow the intent. The silicon effect, which the service runs on its own, takes
made-up impulses at the rates, channels and lengths of the real ones, the synthetic one with the
real one's correlation between its channels and its ratio of energy to magnitude, which sets the
level afir gives it.

Three more cases are single effects whose part in a graph is too small for the comparison to
see: the compressor, the hailer's phaser and the cult's vibrato, whose pitch track is saved
beside its spectrum, since a vibrato moves the harmonics less than a mel band is wide. The
made-up voice is brought to the loudness the service gives speech before its effects.

The noise source is seeded so that a rerun writes the same bytes.
"""

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

import common

RATE = 48000
# The loudness the service brings speech to before its effects, in LUFS, and its true-peak
# ceiling, in dBTP.
TARGET = -16.0
CEILING = -1.0
# The log-mel spectrum the comparison reads: frames of 1024 every 10 ms, 40 HTK mel bands from 60
# Hz to 16 kHz, in dB, kept in f16 as the impulse is, to keep the file small; the effects tests
# compute it the same way.
FRAME = 1024
HOP = 480
BANDS = 40
LOW = 60.0
HIGH = 16000.0
FLOOR = 1e-10
# The pitch the comparison reads where log-mel bands are too coarse for the effect: frames of 40
# ms every 10 ms, voiced where their normalized autocorrelation peaks over VOICED.
PITCH_FRAME = 1920
VOICED = 0.6
NEAR = 0.9
PITCHED = ["vibrato", "cult_slurring"]

# SynthImpulse.wav's: 9570 stereo frames at 44.1 kHz, its channels correlated by 0.55, the mean of
# its channels with an L2 norm 0.017123 times the mean of their L1 norms.
SYNTH = (44100, 9570, 0.017123)
CORRELATION = 0.55
ROOM = (44100, 62976)

SYNTHETIC = (
    "alimiter=0.9,acompressor=threshold=0.2:ratio=20:attack=10:release=50:makeup=2,"
    "highpass=f=1000"
)
SYNTHETIC_EFFECTS = [
    {"effect": "gain", "gain_db": -0.915},
    {"effect": "limiter", "ceiling_db": 0},
    {
        "effect": "compressor",
        "threshold_db": -13.979,
        "ratio": 20,
        "attack_ms": 10,
        "release_ms": 50,
        "makeup_db": 6.021,
        "knee_db": 9.031,
    },
    {"effect": "highpass", "frequency_hz": 1000},
]


def pitched(ratio):
    return {"effect": "pitch", "semitones": round(12 * math.log2(ratio), 3)}


def three_tongues(low, high, volume):
    weight = round(volume / 3, 4)
    return {
        "nodes": {
            "low": pitched(low),
            "high": pitched(high),
            "mix": {
                "effect": "mix",
                "weights": {"low": weight, "input": 0.3333, "high": weight},
            },
        },
        "edges": [
            ["input", "low"],
            ["input", "high"],
            ["input", "mix"],
            ["low", "mix"],
            ["high", "mix"],
            ["mix", "output"],
        ],
    }


def tongue(low, high, volume):
    shift = lambda ratio: f"asetrate=48000*{ratio},aresample=48000,atempo=1/{ratio}"  # noqa: E731
    gain = f",volume={volume}" if volume != 1 else ""
    return (
        f"[0:a] asplit [out0][out2]; [out0] {shift(low)},aformat=channel_layouts=mono{gain} [p0]; "
        f"[out2] {shift(high)},aformat=channel_layouts=mono{gain}[p2]; [p0][0][p2] amix=inputs=3"
    )


FISH = (
    "rubberband=pitch='ifnot(0,2-(0+if(0,4))/16,1)':formant=preserved,"
    "highpass=f=1000:t=s:w=24,equalizer=f=1200:g=15,equalizer=f=4350:g=-15,highshelf=f=870:g=1,"
    "afftfilt=real='{gain}':imag='{gain_imag}':win_size=1024"
)
FISH_GAIN = (
    "st(0,(b+0.5)/nb*sr);st(1,3000+1500*sin(9.3*2*PI*pts/sr));st(2,ld(0)/ld(1));"
    "{part}*(max(0,1-ld(2)^2)+2*gauss(log(ld(2)+1)))"
)

SILICON = (
    "[0] aresample=44100 [re_1]; [re_1] apad=pad_dur=2 [in_1]; "
    "[in_1] asplit=2 [in_1_1] [in_1_2]; [in_1_1] [1] afir=dry=10:wet=10 [reverb_1]; "
    "[in_1_2] [reverb_1] amix=inputs=2:weights=8 1 [mix_1]; [mix_1] asplit=2 [mix_1_1] [mix_1_2]; "
    "[mix_1_1] [2] afir=dry=1:wet=1 [reverb_2]; [mix_1_2] [reverb_2] amix=inputs=2:weights=10 1 [mix_2]; "
    "[mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [out]; "
    "[out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled"
)

CASES = [
    ("synthetic", SYNTHETIC, [SYNTHETIC_EFFECTS]),
    (
        "tram",
        "highpass=f=300,lowpass=f=3500,aecho=0.8:0.9:70|140:0.3|0.15," + SYNTHETIC,
        [
            [
                {"effect": "highpass", "frequency_hz": 300},
                {"effect": "lowpass", "frequency_hz": 3500},
                {
                    "effect": "echo",
                    "taps": [
                        {"delay_ms": 0, "weight": 0.72},
                        {"delay_ms": 70, "weight": 0.27},
                        {"delay_ms": 140, "weight": 0.135},
                    ],
                },
            ],
            SYNTHETIC_EFFECTS,
        ],
    ),
    (
        "fishing",
        "alimiter=0.9,acompressor=threshold=0.3:ratio=20:attack=15:release=350:makeup=1.5,"
        "highpass=f=1000,rubberband=pitch=1.5",
        [
            [
                {"effect": "gain", "gain_db": -0.915},
                {"effect": "limiter", "ceiling_db": 0},
                {
                    "effect": "compressor",
                    "threshold_db": -10.458,
                    "ratio": 40,
                    "attack_ms": 15,
                    "release_ms": 350,
                    "makeup_db": 3.522,
                    "knee_db": 9.031,
                },
                {"effect": "highpass", "frequency_hz": 1000},
                {"effect": "pitch", "semitones": 7.02},
            ]
        ],
    ),
    (
        "gas_mask",
        "lowpass=f=750,volume=2",
        [
            [
                {"effect": "lowpass", "frequency_hz": 750},
                {"effect": "gain", "gain_db": 6.021},
            ]
        ],
    ),
    (
        "megaphone",
        "acrusher=samples=2:level_out=6,aecho=delays=90:decays=0.3,aemphasis=type=cd,"
        "acontrast=30,crystalizer=i=5",
        [
            [
                {"effect": "crush", "rate_hz": 24000, "mix": 0.5},
                {"effect": "gain", "gain_db": 16.0},
                {
                    "effect": "echo",
                    "taps": [
                        {"delay_ms": 0, "weight": 0.18},
                        {"delay_ms": 90, "weight": 0.09},
                    ],
                },
                {
                    "effect": "highshelf",
                    "frequency_hz": 5784,
                    "gain_db": -10.93,
                    "q": 0.456,
                },
                {"effect": "saturate", "drive_db": 5.11},
                {
                    "effect": "highshelf",
                    "frequency_hz": 4474,
                    "gain_db": 20.83,
                    "q": 0.422,
                },
            ]
        ],
    ),
    (
        "megaphone_emagged",
        "acrusher=samples=9:level_out=7,aecho=delays=100:decays=0.4,aemphasis=type=emi,"
        "crystalizer=i=6,acontrast=60,rubberband=pitch=0.9",
        [
            [
                {"effect": "crush", "rate_hz": 5333.3, "mix": 0.5},
                {"effect": "gain", "gain_db": 16.902},
                {
                    "effect": "echo",
                    "taps": [
                        {"delay_ms": 0, "weight": 0.18},
                        {"delay_ms": 100, "weight": 0.12},
                    ],
                },
                {
                    "effect": "lowshelf",
                    "frequency_hz": 202.2,
                    "gain_db": 15.74,
                    "q": 0.487,
                },
                {
                    "effect": "highshelf",
                    "frequency_hz": 20000,
                    "gain_db": -56.13,
                    "q": 0.205,
                },
                {
                    "effect": "highshelf",
                    "frequency_hz": 4134,
                    "gain_db": 22.28,
                    "q": 0.412,
                },
                {"effect": "saturate", "drive_db": 5.57},
                {"effect": "pitch", "semitones": -1.824},
            ]
        ],
    ),
    (
        "hailer",
        "[0:a] asetrate=48000*0.7,aresample=16000,atempo=1/0.7,lowshelf=g=-20:f=500,highpass=f=500,"
        "aphaser=in_gain=1:out_gain=1:delay=3.0:decay=0.4:speed=0.5:type=t [out]; "
        "[out]atempo=1.2,volume=15dB [final]; anoisesrc=a=0.01:d=60:seed=1 [noise]; "
        "[final][noise] amix=duration=shortest",
        [
            [
                {"effect": "pitch", "semitones": -6.175},
                {"effect": "lowshelf", "frequency_hz": 500, "gain_db": -20, "q": 0.5},
                {"effect": "highpass", "frequency_hz": 500},
                {
                    "effect": "flanger",
                    "delay_ms": 3,
                    "rate_hz": 0.5,
                    "feedback": 0.4,
                    "waveform": "triangle",
                },
                {"effect": "tempo", "ratio": 1.2},
                {"effect": "gain", "gain_db": 8.979},
                {"effect": "noise", "level_db": -50.79},
                {"effect": "lowpass", "frequency_hz": 7600, "q": 0.5098},
                {"effect": "lowpass", "frequency_hz": 7600, "q": 0.6013},
                {"effect": "lowpass", "frequency_hz": 7600, "q": 0.9},
                {"effect": "lowpass", "frequency_hz": 7600, "q": 2.5629},
            ]
        ],
    ),
    ("lizard_tongue", tongue(0.9, 1.1, 0.2), [three_tongues(0.9, 1.1, 0.2)]),
    ("alien_tongue", tongue(0.8, 1.2, 1), [three_tongues(0.8, 1.2, 1)]),
    (
        "ethereal_tongue",
        "[0:a] asplit [out0][out2]; [out0] asetrate=48000*0.99,aresample=48000,volume=0.3 [p0]; "
        "[p0][out2] amix=inputs=2",
        [
            {
                "nodes": {
                    "slow": {"effect": "speed", "ratio": 0.99},
                    "mix": {"effect": "mix", "weights": {"slow": 0.15, "input": 0.5}},
                },
                "edges": [
                    ["input", "slow"],
                    ["input", "mix"],
                    ["slow", "mix"],
                    ["mix", "output"],
                ],
            }
        ],
    ),
    ("snail_tongue", "atempo=0.5", [[{"effect": "tempo", "ratio": 0.5}]]),
    (
        "stutter",
        "tremolo=f=10:d=0.8,rubberband=tempo=0.5",
        [
            [
                {"effect": "tremolo", "rate_hz": 10, "depth": 0.8},
                {"effect": "tempo", "ratio": 0.5},
            ]
        ],
    ),
    (
        "cult_slurring",
        "rubberband=pitch=0.5,vibrato=5",
        [
            [
                {"effect": "pitch", "semitones": -12},
                {"effect": "vibrato", "rate_hz": 5, "depth_semitones": 0.74},
            ]
        ],
    ),
    ("poly", "rubberband=pitch=1.5", [[{"effect": "pitch", "semitones": 7.02}]]),
    (
        "fish_tongue",
        FISH.format(
            gain=FISH_GAIN.format(part="re"),
            gain_imag=FISH_GAIN.format(part="im"),
        ),
        [
            [
                {"effect": "pitch", "semitones": 12, "keep_formants": True},
                {"effect": "highpass", "frequency_hz": 1000, "q": 3.464},
                {"effect": "peak", "frequency_hz": 1200, "gain_db": 15, "q": 1},
                {"effect": "peak", "frequency_hz": 4350, "gain_db": -15, "q": 1},
                {"effect": "highshelf", "frequency_hz": 870, "gain_db": 1, "q": 0.5},
                {
                    "effect": "wobble",
                    "center_hz": 3000,
                    "swing_hz": 1500,
                    "rate_hz": 9.3,
                    "q": 2,
                },
            ]
        ],
    ),
    (
        "compressor",
        "acompressor=threshold=0.2:ratio=20:attack=10:release=50:makeup=2",
        [[SYNTHETIC_EFFECTS[2]]],
    ),
    (
        "flanger",
        "aphaser=in_gain=1:out_gain=1:delay=3.0:decay=0.4:speed=0.5:type=t",
        [
            [
                {
                    "effect": "flanger",
                    "delay_ms": 3,
                    "rate_hz": 0.5,
                    "feedback": 0.4,
                    "waveform": "triangle",
                }
            ]
        ],
    ),
    (
        "vibrato",
        "vibrato=5",
        [[{"effect": "vibrato", "rate_hz": 5, "depth_semitones": 0.74}]],
    ),
    (
        "silicon",
        SILICON,
        [
            {
                "nodes": {
                    "synth": {"effect": "convolve", "impulse": "synth"},
                    "mix": {
                        "effect": "mix",
                        "weights": {"input": 0.5714, "synth": 0.1223},
                    },
                    "air": {
                        "effect": "peak",
                        "frequency_hz": 7710,
                        "gain_db": -6,
                        "q": 0.6,
                    },
                    "rumble": {
                        "effect": "peak",
                        "frequency_hz": 33,
                        "gain_db": -10,
                        "q": 0.44,
                    },
                    "limit": {"effect": "limiter", "ceiling_db": -6.021},
                },
                "edges": [
                    ["input", "synth"],
                    ["input", "mix"],
                    ["synth", "mix"],
                    ["mix", "air"],
                    ["air", "rumble"],
                    ["rumble", "limit"],
                    ["limit", "output"],
                ],
            }
        ],
    ),
]


def voice(rng):
    """1.6 s of made-up speech: five voiced syllables, each a vowel's formants over band-limited
    harmonics of a gliding pitch, from a low voice's to a high one's, and a fricative."""
    t = np.arange(int(1.6 * RATE)) / RATE
    out = np.zeros_like(t)
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240)]
    syllables = [(0.05, 0.25, 105, 120), (0.33, 0.25, 130, 115), (0.66, 0.22, 175, 210)]
    syllables += [(0.95, 0.2, 230, 200), (1.3, 0.25, 150, 110)]
    for i, (start, length, f0, f1) in enumerate(syllables):
        n = int(length * RATE)
        at = int(start * RATE)
        u = np.arange(n) / n
        f = f0 + (f1 - f0) * u + 3 * np.sin(2 * np.pi * 5.5 * u * length)
        phase = 2 * np.pi * np.cumsum(f) / RATE
        source = sum(
            np.sin(k * phase) / k * (k * f < 10000)
            for k in range(1, int(10000 / min(f0, f1)) + 1)
        )
        shaped = source
        for formant, bandwidth in zip(vowels[i % len(vowels)], (80, 100, 140)):
            r = math.exp(-math.pi * bandwidth / RATE)
            w = 2 * math.pi * formant / RATE
            shaped = scipy.signal.lfilter(
                [1 - r], [1, -2 * r * math.cos(w), r * r], shaped
            )
        envelope = np.minimum(1, np.minimum(u * length / 0.02, (1 - u) * length / 0.04))
        out[at : at + n] += shaped * envelope
    n = int(0.12 * RATE)
    at = int(0.56 * RATE)
    hiss = scipy.signal.lfilter(
        *scipy.signal.butter(4, 3500, "high", fs=RATE), rng.standard_normal(n)
    )
    u = np.arange(n) / n
    out[at : at + n] += 0.15 * hiss * np.minimum(1, np.minimum(u / 0.2, (1 - u) / 0.3))
    return out / np.abs(out).max()


def loudness(y):
    """BS.1770-4's integrated loudness of `y` at 48 kHz, in LUFS: K-weighted, in 400 ms blocks
    every 100 ms, gated at -70 LUFS and 10 LU under the mean of the blocks over it."""
    y = scipy.signal.lfilter(
        [1.53512485958697, -2.69169618940638, 1.19839281085285],
        [1, -1.69065929318241, 0.73248077421585],
        y,
    )
    y = scipy.signal.lfilter([1, -2, 1], [1, -1.99004745483398, 0.99007225036621], y)
    block, step = int(0.4 * RATE), int(0.1 * RATE)
    powers = np.array(
        [np.mean(y[i : i + block] ** 2) for i in range(0, len(y) - block + 1, step)]
    )
    lufs = lambda p: -0.691 + 10 * np.log10(p)  # noqa: E731
    powers = powers[lufs(powers) > -70]
    return lufs(np.mean(powers[lufs(powers) > lufs(np.mean(powers)) - 10]))


def impulse(rng, rate, frames, ratio=None):
    """Exponentially decaying stereo noise, its channels correlated by CORRELATION, its decay set
    so that the mean of its channels has `ratio` of L2 norm to the mean of their L1 norms, which
    afir divides each by."""
    noise = rng.standard_normal((frames, 2))
    noise[:, 1] = (
        CORRELATION * noise[:, 0] + math.sqrt(1 - CORRELATION**2) * noise[:, 1]
    )

    def shaped(decay):
        envelope = np.exp(-np.arange(frames) / (decay * rate))[:, None]
        return noise * envelope

    def measure(x):
        return np.linalg.norm(x.mean(1)) / np.abs(x).sum(0).mean()

    decay = 0.3
    if ratio is not None:
        low, high = 1e-4, 1.0
        for _ in range(60):
            decay = math.sqrt(low * high)
            if measure(shaped(decay)) > ratio:
                low = decay
            else:
                high = decay
    x = shaped(decay)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def ffmpeg(graph, signal, inputs):
    """`signal` through `graph`, with the WAV files `inputs` as its further inputs: its output, as
    the mean of its channels at 48 kHz."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.wav"
        sf.write(source, signal, RATE, subtype="FLOAT")
        args = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source)]
        for path in inputs:
            args += ["-i", str(path)]
        out = Path(tmp) / "out.wav"
        args += ["-filter_complex", graph, "-c:a", "pcm_f32le", str(out)]
        subprocess.run(args, check=True)
        y, rate = sf.read(out, dtype="float64", always_2d=True)
    y = y.mean(1)
    if rate != RATE:
        g = math.gcd(rate, RATE)
        y = scipy.signal.resample_poly(y, RATE // g, rate // g)
    return y


def mel_filters():
    """HTK triangles, peaking at 1, over the bins of a FRAME-point transform."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    edges = hz(np.linspace(mel(LOW), mel(HIGH), BANDS + 2))
    bins = np.arange(FRAME // 2 + 1) * RATE / FRAME
    filters = np.zeros((BANDS, bins.size))
    for b in range(BANDS):
        low, mid, high = edges[b : b + 3]
        rising = (bins - low) / (mid - low)
        falling = (high - bins) / (high - mid)
        filters[b] = np.maximum(0, np.minimum(rising, falling))
    return filters


def log_mel(y):
    """The log-mel spectrum of `y`, frames by bands, in dB."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)
    frames = 1 + max(0, len(y) - FRAME) // HOP
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(window * y[t * HOP : t * HOP + FRAME], FRAME)) ** 2
            for t in range(frames)
        ]
    )
    return 10 * np.log10(np.maximum(spectra @ mel_filters().T, FLOOR))


def pitch(y):
    """The pitch of each frame of `y`, in Hz, 0 where it is unvoiced: the shortest lag between 50
    and 500 Hz where the frame's normalized autocorrelation peaks within NEAR of its highest,
    refined by a parabola, where that highest is over VOICED; the effects tests track it the
    same way."""
    frames = 1 + max(0, len(y) - PITCH_FRAME) // HOP
    shortest, longest = RATE // 500, RATE // 50
    lags = np.arange(PITCH_FRAME)
    track = np.zeros(frames)
    for t in range(frames):
        frame = np.zeros(2 * PITCH_FRAME)
        part = y[t * HOP : t * HOP + PITCH_FRAME]
        frame[: len(part)] = part
        r = np.fft.ifft(np.abs(np.fft.fft(frame)) ** 2).real[:PITCH_FRAME]
        if r[0] <= 0:
            continue
        normalized = r / r[0] * PITCH_FRAME / (PITCH_FRAME - lags)
        best = normalized[shortest : longest + 1].max()
        if best < VOICED:
            continue
        # The shortest lag near the best, whose multiples would also be near it.
        lag = shortest + np.argmax(normalized[shortest : longest + 1] >= NEAR * best)
        while lag < longest and normalized[lag + 1] > normalized[lag]:
            lag += 1
        if lag in (shortest, longest):
            continue
        a, b, c = normalized[lag - 1 : lag + 2]
        track[t] = RATE / (lag + 0.5 * (a - c) / (a - 2 * b + c))
    return track


def version():
    out = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    )
    return out.stdout.split("\n")[0]


fixture = common.Fixture(seed=0)
rng = np.random.default_rng(0)
signal = voice(rng)
# Brought to the service's loudness, as speech is before its effects, under its true-peak
# ceiling without a limiter.
signal = signal * 10 ** ((TARGET - loudness(signal)) / 20)
assert np.abs(scipy.signal.resample_poly(signal, 4, 1)).max() < 10 ** (CEILING / 20)
signal = signal.astype(np.float32)
# Rounded to the f16 it is kept in, so that ffmpeg convolves with what the tests read.
synth = impulse(rng, SYNTH[0], SYNTH[1], SYNTH[2]).astype(np.float16).astype(np.float32)
room = impulse(rng, *ROOM)
tensors = {
    "input": torch.from_numpy(signal),
    "synth": torch.from_numpy(synth).half(),
}
with tempfile.TemporaryDirectory() as tmp:
    impulses = [Path(tmp) / "synth.wav", Path(tmp) / "room.wav"]
    sf.write(impulses[0], synth, SYNTH[0], subtype="FLOAT")
    sf.write(impulses[1], room, ROOM[0], subtype="FLOAT")
    for name, graph, _ in CASES:
        y = ffmpeg(graph, signal, impulses if name == "silicon" else [])
        tensors[f"{name}/mel"] = torch.from_numpy(log_mel(y)).half()
        if name in PITCHED:
            tensors[f"{name}/pitch"] = torch.from_numpy(pitch(y).astype(np.float32))
cases = [
    {"name": name, "ffmpeg": graph, "effects": effects}
    for name, graph, effects in CASES
]
fixture.save(
    "effects",
    tensors,
    cases=json.dumps(cases, separators=(",", ":")),
    ffmpeg=version(),
    synth_rate=str(SYNTH[0]),
)
