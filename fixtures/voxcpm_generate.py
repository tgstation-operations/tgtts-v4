"""VoxCPM2 generating at the tiny config in crates/voxcpm/tests/fixtures, from given token ids
and a random voice, prompted as the Python service prompted it: with the voice's latents as both
the reference audio and the audio to continue, after its transcript.

Upstream's generate runs twice on the same noise: once held from stopping for eight patches,
recording each patch, the noise it was sampled from, the stop head's logits and the audio it
decoded, and once stopping as nanovllm-voxcpm stops, after the first patch the stop head says
to.
"""

import torch

import common

STEPS = 8
SEED = 1234
VOICE_FRAMES = 20

fixture = common.Fixture(seed=0)
model = common.voxcpm2(
    (common.ROOT / "crates/voxcpm/tests/fixtures/voxcpm.json").read_text()
)
tensors = {
    name: parameter.data
    for name, parameter in model.named_parameters()
    if not name.startswith("audio_vae.")
}
decoder = model.audio_vae.decoder.state_dict()
# common writes no I32.
decoder["sr_bin_boundaries"] = decoder["sr_bin_boundaries"].long()
tensors.update({f"decoder.{name}": value for name, value in decoder.items()})

# Token ids given for the transcript and the text, which upstream tokenizes together and apart.
transcript, text = "transcript", "text"
transcript_ids, text_ids = [5, 17, 42, 9], [77, 3, 64]
model.text_tokenizer = {
    transcript + text: transcript_ids + text_ids,
    text: text_ids,
}.__getitem__
voice = torch.randn(VOICE_FRAMES, model.feat_dim)
model._encode_wav = lambda *args, **kwargs: voice.view(
    -1, model.patch_size, model.feat_dim
)

prompt = {}
inference = model._inference


def record_prompt(text, text_mask, feat, feat_mask, **kwargs):
    prompt.update(tokens=text, text_mask=text_mask, patches=feat, audio=feat_mask)
    return inference(text, text_mask, feat, feat_mask, **kwargs)


model._inference = record_prompt

steps = common.record_steps(model)


def generate(min_len):
    for recorded in steps.values():
        recorded.clear()
    audio = model.generate(
        text,
        prompt_text=transcript,
        prompt_wav_path="voice",
        reference_wav_path="voice",
        min_len=min_len,
        max_len=STEPS,
        inference_timesteps=10,
        cfg_value=2.0,
        seed=SEED,
    )
    return {"audio": audio[0]} | {
        name: torch.cat(recorded) for name, recorded in steps.items()
    }


with torch.no_grad():
    held = generate(min_len=STEPS)
    # A random stop head says to go on throughout. Its rows are made to differ by a direction
    # across the mean of what it is given, so that it says to stop at some patches and not at
    # others, as a trained one does over an utterance, and not at the first.
    seen = held["stop_inputs"]
    mean = seen.mean(dim=0)
    across = seen[STEPS // 2] - mean
    across -= (across @ mean) / (mean @ mean) * mean
    across /= (seen @ across).abs().max()
    if seen[0] @ across > 0:
        across = -across
    model.stop_head.weight[1] = model.stop_head.weight[0] + across
    tensors["stop_head.weight"] = model.stop_head.weight.data
    held = generate(min_len=STEPS)
    stopping = generate(min_len=-1)

stops = held["stop_logits"].argmax(dim=-1).tolist()
assert 0 < stops.index(1) < STEPS - 1, f"the stop head says {stops}"
assert torch.equal(stopping["patches"], held["patches"][: stops.index(1) + 1])

tensors.update(
    {
        "input.text_tokens": torch.tensor(transcript_ids + text_ids),
        "input.voice": voice,
        "input.prompt.tokens": prompt["tokens"][0].long(),
        "input.prompt.patches": prompt["patches"][0],
        "input.prompt.audio": prompt["audio"][0].long(),
        "input.noise": held["noise"],
        "output.patches": held["patches"],
        "output.stop_logits": held["stop_logits"],
        "output.audio": held["audio"],
    }
)
fixture.save("voxcpm", tensors)
