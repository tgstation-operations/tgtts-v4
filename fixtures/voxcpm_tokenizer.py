"""A tokenizer.json of VoxCPM2's form with a vocabulary of a few dozen tokens, and the ids that
the Python service's tokenizer gives texts chosen to exercise it: Hugging Face's tokenizers
running the file, as transformers 4 runs it, under upstream's mask_multichar_chinese_tokens.

"b c" merges before "a b", so that in "abc" the queued "a b" goes stale, and the vocabulary
leaves out characters to fall back to bytes for. The file is written beside the fixture, as
voxcpm_tokenizer.json.
"""

import json

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import common

SPACE = "\u2581"
NI, HAO, SHI, JIE, MA = "\u4f60", "\u597d", "\u4e16", "\u754c", "\u5417"

fixture = common.Fixture(seed=0)
utils = common.upstream("model.utils")

merges = [
    ("b", "c"),
    ("a", "b"),
    ("ab", "c"),
    ("a", "bc"),
    (SPACE, "a"),
    ("a", "a"),
    ("aa", "a"),
    ("l", "l"),
    ("h", "e"),
    ("he", "ll"),
    ("hell", "o"),
    (SPACE, "hello"),
    ("w", "o"),
    ("wo", "r"),
    ("wor", "l"),
    ("worl", "d"),
    (SPACE, "world"),
    (SPACE, SPACE),
    (NI, HAO),
    (SHI, JIE),
    (SPACE, NI + HAO),
    (NI + HAO, SHI + JIE),
    (SPACE + NI + HAO, MA),
    (SPACE, SHI + JIE),
]
tokens = ["<unk>", "<s>", "</s>"]
tokens += [f"<0x{byte:02X}>" for byte in range(256)]
tokens += [SPACE, *"abcdehlorwst.,!?'&- ", NI, HAO, SHI, JIE, MA]
for a, b in merges:
    if a + b not in tokens:
        tokens.append(a + b)
vocab = {token: id for id, token in enumerate(tokens)}
added = ["<unk>", "<s>", "</s>", "<|im_end|>", "<|im_start|>", "<|fim|>", "<|fim|>!"]
ids = dict(vocab)
for content in added:
    ids.setdefault(content, len(ids))
added_tokens = [
    {
        "id": ids[content],
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    for content in added
]
spec = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": added_tokens,
    "normalizer": {
        "type": "Sequence",
        "normalizers": [
            {"type": "Prepend", "prepend": SPACE},
            {"type": "Replace", "pattern": {"String": " "}, "content": SPACE},
        ],
    },
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": True,
        "byte_fallback": True,
        "ignore_merges": False,
        "vocab": vocab,
        "merges": [f"{a} {b}" for a, b in merges],
    },
}
text = json.dumps(spec, indent=2) + "\n"
path = common.ROOT / "crates/voxcpm/tests/fixtures/voxcpm_tokenizer.json"
path.write_text(text)

texts = [
    "",
    " ",
    "  ",
    "a",
    "abc",
    "abcabc",
    "a bc",
    "aaaa",
    "aaaaa",
    "hello world",
    "hello  world",
    " hello",
    "hello ",
    "Hello World",
    "hello, world! it's a-b & c?",
    "caf\u00e9",
    "\U0001f600",
    NI + HAO,
    NI + HAO + SHI + JIE,
    " " + NI + HAO + MA,
    NI + HAO + MA,
    SHI + JIE + " " + SHI + JIE,
    "a<s>b",
    "<s> a",
    "</s>",
    "<|im_end|>",
    "x<|im_end|>y",
    "<|fim|>!",
    "<|fim|>?",
    "<<s>>",
    "\n\t",
    SPACE + "a",
]
raw = Tokenizer.from_str(text)
fast = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
tokenizer = utils.mask_multichar_chinese_tokens(fast)
tensors = {}
for i, sample in enumerate(texts):
    assert fast.tokenize(sample) == raw.encode(sample, add_special_tokens=False).tokens
    tensors[f"ids.{i}"] = torch.tensor(tokenizer(sample), dtype=torch.int64)

fixture.save("voxcpm", tensors, texts=json.dumps(texts))
