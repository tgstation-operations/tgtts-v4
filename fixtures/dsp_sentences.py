"""pysbd's sentences of chat text in the game's alphabet, for the dsp crate's segmenter.

The game filters what it sends to letters, digits and ` ,?.!'&-`. The corpus is cases written for
pysbd's rules (abbreviations, numbers and lists, ellipses, runs of ! and ?, quotes made of
apostrophes, dashes), chat made up of station words with those in it, and random strings of the
alphabet, some dense in the characters the rules look at. The texts and their sentences go in
the fixture's metadata as JSON.
"""

import importlib.metadata
import json
import random

import pysbd

import common

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,?.!'&-"

CASES = [
    "",
    " ",
    "  ",
    ".",
    "!",
    "?",
    "...",
    "....",
    ".....",
    "......",
    "?!",
    "!?",
    "??",
    "!!",
    "!!!",
    "a",
    "A",
    "ok",
    "Ok.",
    "hi",
    "Hi!",
    "Hello there.",
    "Hello there",
    "hello there. how are you",
    "Hi.  Bye.",
    " Hi. Bye. ",
    "Hi.Bye.",
    "Hi. bye.",
    "HELP MAINT",
    "HELP! MAINT!",
    "AI, open the door.",
    "Dr. Smith is here.",
    "Mr. Jones went to St. Louis.",
    "Capt. Picard and Lt. Col. Data.",
    "I saw Mrs. Robinson. She waved.",
    "Ask Prof. Oak. He knows.",
    "Sgt. Pepper's band.",
    "e.g. this one",
    "Use tools, e.g. a wrench.",
    "i.e. that one",
    "Things, i.e. stuff. More.",
    "Stuff etc. and more.",
    "Stuff etc. More stuff.",
    "It is No. 5.",
    "Page no. 5 please.",
    "Them vs. us.",
    "The U.S. army. The end.",
    "The U.S.A. The best.",
    "U.S. The end.",
    "Co. KG is a firm.",
    "At 5 a.m. The sun rose.",
    "At 5 p.m. we left.",
    "At 5 P.M. They left.",
    "A Ph.D. student.",
    "He is a Ph.D. He knows.",
    "On Jan. 5 we met.",
    "See Fig. 3 now.",
    "Approx. 5 units.",
    "Version 2.0.1 is out.",
    "It costs 3.14 credits.",
    "Pi is 3.14. Tau is 6.28.",
    "Go to deck 5. Then left.",
    "Call 911. Now!",
    "1. eat 2. sleep 3. code",
    "1. Eat. 2. Sleep.",
    "Steps 1. 2. 3. done",
    "a. foo b. bar c. baz",
    "a. one",
    "i. one v. five x. ten",
    "-1. and -2. too",
    "it is 10. Go home.",
    "end.12 The next",
    "we have 12. 13. 14. items",
    "for 1. a thing 2. b",
    "wait...",
    "wait... What?",
    "wait... what?",
    "wait.... Ok",
    "wait..... ok",
    "wait . . . ok",
    "wait . . . . ",
    "So. . . . ",
    "Hmm...I see",
    "...what",
    "....5 minutes",
    "Wait....5 minutes",
    "really?!",
    "really!? no",
    "what?? no",
    "stop!! now",
    "Hello!!! Bye",
    "what??? Who",
    "!hello",
    "Yahoo! Yum! Y!J",
    "!Kung people",
    "Yummy! tasty",
    "Yum!",
    "don't stop",
    "'hello' he said.",
    "'Hello.' He said.",
    "He said 'hi. there' Bob.",
    "''quoted'' text",
    "He said ''Hi. There'' Bob.",
    "''Hi!'' She said.",
    "'twas fun.",
    "rock 'n' roll. Yes.",
    "the dogs' bones. Yes.",
    "it's 5 o'clock.",
    "I'm here. You're there.",
    "well -- I think. So -- yes",
    "-- hi --",
    "--. --",
    "co-op. e-mail. -5 degrees.",
    "R&D is fun. You & me.",
    "&&& ok",
    "Hi, Bob. Ok, fine.",
    "1,000 credits. 2,000 more.",
    ", leading comma",
    "trailing,",
    "end.Start",
    "5.Five",
    "planet.Earth",
    "www.foo.Com",
    "a.b.c.d",
    "A.B.C.",
    "x.Y z.W",
    "I. The end",
    "Hello I. The end",
    "He is No. 1. The best.",
    "Mt. Everest. Tall.",
    "Ave. Maria",
    "Inc. And Ltd.",
    "St. 5 is here",
    "Rev. 5 now",
    "art. 5",
    "pp. 10",
    "p. 5 here",
    "ext. 42",
    "Gen. Kenobi! Hello.",
    "Col. Mustard did it.",
    "NT is bad. Syndicate is good!",
    "Captain? Captain! CAPTAIN.",
    "ok ok ok. ok? ok!",
    "HoP, give me AA. Please?",
    "Sec, help! Clown is - honking - me.",
    "Is the AI rogue? Yes.",
    # Each changes if one of pysbd's quirks is left out: an abbreviation's period matching any
    # character, an empty pair of double quotes, an empty pair of dashes, the item before the first
    # of a list being the last, and two rules no other case reaches.
    "The u.s. navy and the u s. army.",
    "Tools, e.g. a wrench, e g. a crowbar.",
    "''''Hi. there'' Bob.",
    "Wait ---- a. b -- ok.",
    "b. go x. now a. stop",
    "12. Go home.",
    "Run the .exe file. Now.",
]

WORDS = (
    "the a an I you he she it we they captain clown janitor sec security officer engineer AI "
    "borg cargo medbay doctor chemist botanist chaplain mime station shuttle airlock door maint "
    "plasma singularity tesla nuke disk syndicate traitor changeling cult heretic wizard ninja "
    "is are was were go come help stop run fix open close kill save get give take need want "
    "please now here there not no yes ok okay really very quite just only also too and or but "
    "so because if then when where why how what who which this that these those my your our"
).split()
ABBREVIATIONS = (
    "Dr Mr Mrs Ms St Capt Lt Col Sgt Gen Prof Rev Mt No vs etc e.g i.e U.S U.S.A a.m p.m Ph.D "
    "Jan Feb Fig approx Inc Ltd Co Jr Sr art pp p ext Sec Ok"
).split()
ENDINGS = [".", ".", ".", "!", "?", "...", "?!", "!!", "", " .", "....", "!!!", "??"]


def word(rng):
    roll = rng.random()
    if roll < 0.08:
        return str(
            rng.choice([rng.randrange(10), rng.randrange(100), rng.randrange(10000)])
        )
    if roll < 0.11:
        return f"{rng.randrange(100)}.{rng.randrange(100)}"
    if roll < 0.18:
        return rng.choice(ABBREVIATIONS) + "."
    if roll < 0.2:
        return rng.choice(["don't", "it's", "I'm", "'em", "'cause", "o'clock", "dogs'"])
    if roll < 0.22:
        return rng.choice(["--", "-", "&", "R&D", "co-op"])
    w = rng.choice(WORDS)
    if rng.random() < 0.1:
        w = w.upper()
    return w


def sentence(rng):
    words = [word(rng) for _ in range(rng.randrange(1, 12))]
    if rng.random() < 0.8:
        words[0] = words[0][:1].upper() + words[0][1:]
    for i in range(len(words) - 1):
        if rng.random() < 0.1:
            words[i] += ","
    if rng.random() < 0.05:
        quoted = rng.randrange(len(words))
        q = rng.choice(["'", "''"])
        words[quoted] = q + words[quoted] + q
    return " ".join(words) + rng.choice(ENDINGS)


def chat(rng):
    sentences = [sentence(rng) for _ in range(rng.randrange(1, 5))]
    text = ""
    for s in sentences:
        text += s + rng.choice([" ", " ", " ", "  ", "", "   "])
    if rng.random() < 0.5:
        text = text.rstrip(" ")
    return text


def fuzz(rng):
    weights = [8] * 52 + [2] * 10 + [12, 2, 2, 4, 2, 1, 1, 1]
    return "".join(rng.choices(ALPHABET, weights=weights, k=rng.randrange(1, 80)))


def punctuation_fuzz(rng):
    """Short strings dense in what pysbd's rules look at."""
    pieces = [
        "a",
        "I",
        "A",
        "Dr",
        "no",
        "5",
        "12",
        "x",
        "i",
        "v",
        "U",
        "S",
        " ",
        " ",
        ".",
        ".",
        "!",
        "?",
        "'",
        "''",
        "-",
        "--",
        ",",
        "&",
        "The",
        "He",
        "e",
        "g",
    ]
    return "".join(rng.choice(pieces) for _ in range(rng.randrange(1, 16)))


def corpus(seed, chats, fuzzes):
    rng = random.Random(seed)
    texts = list(CASES)
    texts += [chat(rng) for _ in range(chats)]
    texts += [fuzz(rng) for _ in range(fuzzes)]
    texts += [punctuation_fuzz(rng) for _ in range(fuzzes)]
    for text in texts:
        assert all(c in ALPHABET for c in text), text
    return texts


texts = corpus(seed=0, chats=600, fuzzes=600)
segmenter = pysbd.Segmenter(language="en", clean=True)
sentences = [list(segmenter.segment(text)) for text in texts]
common.Fixture(seed=0).save(
    "dsp",
    {},
    pysbd=importlib.metadata.version("pysbd"),
    texts=json.dumps(texts),
    sentences=json.dumps(sentences),
)
