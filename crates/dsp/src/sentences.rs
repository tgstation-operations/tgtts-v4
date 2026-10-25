//! Sentence segmentation as `pysbd.Segmenter(language="en", clean=True)` does it, which the old
//! service split the blips' text with.
//!
//! pysbd marks what is not a sentence boundary with placeholder characters, splits, and puts the
//! characters back, all with regular expressions; its rules are transliterated here in the order
//! it applies them, with its placeholders and the full-width punctuation it knows written as
//! escapes (`\u{222f}` is the period that does not end a sentence).
//!
//! The text the game sends has been filtered to letters, digits and ` ,?.!'&-`, and the rules that
//! can only fire on other characters (newlines, brackets, quotes other than `'` and the `"` that
//! pysbd makes of `''`, non-ASCII punctuation) are left out: on such text this splits as pysbd
//! does, and on other text it may not.

use std::sync::LazyLock;

use fancy_regex::{Captures, Regex};

/// The sentences of `text` as `pysbd.Segmenter(language="en", clean=True).segment(text)` gives
/// them, for text of the game's alphabet.
pub fn segment(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    process(&clean(text))
}

macro_rules! regex {
    ($name:ident = $pattern:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($pattern).unwrap());
    };
}

/// `re.sub(pattern, replacement, text)` with a `$`-style replacement. Text fancy-regex gives up
/// on, past its backtracking limit, is left as it is.
fn sub(re: &Regex, text: &str, replacement: &str) -> String {
    re.try_replacen(text, 0, replacement)
        .map_or_else(|_| text.to_owned(), |s| s.into_owned())
}

/// `re.sub(pattern, function, text)`.
fn sub_with(re: &Regex, text: &str, f: impl Fn(&Captures<str>) -> String) -> String {
    re.try_replacen(text, 0, |caps: &Captures<str>| f(caps))
        .map_or_else(|_| text.to_owned(), |s| s.into_owned())
}

/// `re.search(pattern, text)`.
fn search(re: &Regex, text: &str) -> bool {
    re.is_match(text).unwrap_or(false)
}

/// `re.findall(pattern, text)` for a pattern without groups.
fn findall<'t>(re: &Regex, text: &'t str) -> Vec<&'t str> {
    re.find_iter(text)
        .map_while(Result::ok)
        .map(|m| m.as_str())
        .collect()
}

/// Python's `str.isspace` for one character, which `\s` and `strip` go by.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

regex!(TABLE_OF_CONTENTS = r"\.{4,}\s*\d+-*\d*");
regex!(CONSECUTIVE_PERIODS = r"\.{5,}");
regex!(CONSECUTIVE_FORWARD_SLASH = r"/{3}");
regex!(NO_SPACE_BETWEEN_SENTENCES = r"(?<=[a-z])\.(?=[A-Z])");
regex!(NO_SPACE_BETWEEN_SENTENCES_DIGIT = r"(?<=\d)\.(?=[A-Z])");
const URL_EMAIL_KEYWORDS: [&str; 6] = ["@", "http", ".com", "net", "www", "//"];

/// pysbd's `Cleaner.clean`.
fn clean(text: &str) -> String {
    let mut text = text
        .replace('`', "'")
        .replace("''", "\"")
        .replace("``", "\"");
    text = sub(&TABLE_OF_CONTENTS, &text, "\r");
    text = sub(&CONSECUTIVE_PERIODS, &text, " ");
    text = sub(&CONSECUTIVE_FORWARD_SLASH, &text, "");
    // check_for_no_space_in_between_sentences: the words are those of the text as it was, and a
    // word's replacement goes wherever it occurs.
    let words: Vec<String> = text.split(' ').map(str::to_owned).collect();
    for word in &words {
        for rule in [
            &NO_SPACE_BETWEEN_SENTENCES,
            &NO_SPACE_BETWEEN_SENTENCES_DIGIT,
        ] {
            if search(rule, word) && !URL_EMAIL_KEYWORDS.iter().any(|k| word.contains(k)) {
                text = text.replace(word.as_str(), &sub(rule, word, ". "));
            }
        }
    }
    text = sub(&CONSECUTIVE_PERIODS, &text, " ");
    sub(&CONSECUTIVE_FORWARD_SLASH, &text, "")
}

/// pysbd's `Processor.process`.
fn process(text: &str) -> Vec<String> {
    let mut text = text.replace('\n', "\r");
    text = lists::add_line_breaks(&text);
    text = abbreviations::replace(&text);
    text = replace_numbers(&text);
    text = sub_with(&CONTINUOUS_PUNCTUATION, &text, |caps| {
        caps[0]
            .replace('!', "&\u{14f4}&")
            .replace('?', "&\u{14f7}&")
    });
    text = sub_with(&NUMBERED_REFERENCE, &text, |caps| {
        format!("\u{222f}{}\r{}", &caps[2], &caps[7])
    });
    text = sub(&WITH_MULTIPLE_PERIODS_AND_EMAIL, &text, "${1}\u{222e}${3}");
    text = sub(&FILE_FORMAT, &text, "\u{222f}");
    split_into_segments(&text)
}

regex!(PERIOD_BEFORE_NUMBER = r"\.(?=\d)");
regex!(NUMBER_AFTER_PERIOD_BEFORE_LETTER = r"(?<=\d)\.(?=\S)");
regex!(NEW_LINE_NUMBER_PERIOD_SPACE_LETTER = r"(?<=\r\d)\.(?=(\s\S)|\))");
regex!(START_LINE_NUMBER_PERIOD = r"(?<=^\d)\.(?=(\s\S)|\))");
regex!(START_LINE_TWO_DIGIT_NUMBER_PERIOD = r"(?<=^\d\d)\.(?=(\s\S)|\))");

fn replace_numbers(text: &str) -> String {
    [
        &PERIOD_BEFORE_NUMBER,
        &NUMBER_AFTER_PERIOD_BEFORE_LETTER,
        &NEW_LINE_NUMBER_PERIOD_SPACE_LETTER,
        &START_LINE_NUMBER_PERIOD,
        &START_LINE_TWO_DIGIT_NUMBER_PERIOD,
    ]
    .into_iter()
    .fold(text.to_owned(), |text, rule| sub(rule, &text, "\u{222f}"))
}

regex!(CONTINUOUS_PUNCTUATION = r"(?<=\S)(!|\?){3,}(?=(\s|\z|$))");
regex!(
    NUMBERED_REFERENCE = r"(?<=[^\d\s])(\.|\x{222F})((\[(\d{1,3},?\s?-?\s?)*\b\d{1,3}\])+|((\d{1,3}\s?)?\d{1,3}))(\s)(?=[A-Z])"
);
regex!(WITH_MULTIPLE_PERIODS_AND_EMAIL = r"([a-zA-Z0-9_])(\.)([a-zA-Z0-9_])");
regex!(
    FILE_FORMAT = r"(?<=\s)\.(?=(jpe?g|png|gif|tiff?|pdf|ps|docx?|xlsx?|svg|bmp|tga|exif|odt|html?|txt|rtf|bat|sxw|xml|zip|exe|msi|blend|wmv|mp[34]|pptx?|flac|rb|cpp|cs|js)\s)"
);

const PUNCTUATIONS: [char; 7] = [
    '\u{3002}', '\u{ff0e}', '.', '\u{ff01}', '!', '?', '\u{ff1f}',
];

regex!(THREE_SPACE = r"(\s\.){3}\s");
regex!(FOUR_SPACE = r"(?<=[a-z])(\.\s){3}\.($|\\n)");
regex!(FOUR_CONSECUTIVE = r"(?<=\S)\.{3}(?=\.\s[A-Z])");
regex!(THREE_CONSECUTIVE = r"\.\.\.(?=\s+[A-Z])");
regex!(OTHER_THREE_PERIOD = r"\.\.\.");

fn split_into_segments(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for part in text.split('\r').filter(|s| !s.is_empty()) {
        let mut part = sub(
            &THREE_SPACE,
            part,
            "\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}",
        );
        part = sub(
            &FOUR_SPACE,
            &part,
            "\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}",
        );
        part = sub(&FOUR_CONSECUTIVE, &part, "\u{1aa}\u{1aa}\u{1aa}");
        part = sub(&THREE_CONSECUTIVE, &part, "\u{260f}\u{260f}.");
        part = sub(&OTHER_THREE_PERIOD, &part, "\u{1aa}\u{1aa}\u{1aa}");
        let parts = if part.contains(PUNCTUATIONS) {
            process_text(&part)
        } else {
            vec![part]
        };
        for sentence in parts.into_iter().filter(|s| !s.is_empty()) {
            sentences.extend(post_process_segment(&sub_symbols(&sentence)));
        }
    }
    sentences
        .into_iter()
        .map(|s| s.replace("&\u{238b}&", "'"))
        .collect()
}

/// pysbd's `SubSymbolsRules`.
fn sub_symbols(text: &str) -> String {
    [
        ("\u{222f}", "."),
        ("\u{266c}", "\u{60c}"),
        ("\u{266d}", ":"),
        ("&\u{14f0}&", "\u{3002}"),
        ("&\u{14f1}&", "\u{ff0e}"),
        ("&\u{14f3}&", "\u{ff01}"),
        ("&\u{14f4}&", "!"),
        ("&\u{14f7}&", "?"),
        ("&\u{14f8}&", "\u{ff1f}"),
        ("\u{2609}", "?!"),
        ("\u{2607}", "??"),
        ("\u{2608}", "!?"),
        ("\u{2604}", "!!"),
        ("&\u{2702}&", "("),
        ("&\u{232c}&", ")"),
        ("\u{238}", ""),
        ("\u{239}", "\n"),
    ]
    .into_iter()
    .fold(text.to_owned(), |text, (from, to)| text.replace(from, to))
}

regex!(QUOTATION_AT_END_OF_SENTENCE = r#"[!?\.-]["'\x{201C}\x{201D}]\s{1}[A-Z]"#);
regex!(
    SPLIT_SPACE_QUOTATION_AT_END_OF_SENTENCE = r#"(?<=[!?\.-]["'\x{201C}\x{201D}])\s{1}(?=[A-Z])"#
);

/// pysbd's `post_process_segments`, which gives no sentence for one that strips to nothing.
fn post_process_segment(text: &str) -> Vec<String> {
    if text.chars().count() > 2 && text.chars().all(|c| c.is_ascii_alphabetic()) {
        return vec![text.to_owned()];
    }
    let text = [
        ("\u{1aa}\u{1aa}\u{1aa}", "..."),
        (
            "\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}\u{265f}",
            " . . . ",
        ),
        (
            "\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}\u{265d}",
            ". . . .",
        ),
        ("\u{260f}\u{260f}", ".."),
        ("\u{222e}", "."),
    ]
    .into_iter()
    .fold(text.to_owned(), |text, (from, to)| text.replace(from, to));
    if search(&QUOTATION_AT_END_OF_SENTENCE, &text) {
        // re.split, whose pieces pysbd keeps unstripped.
        let mut pieces = Vec::new();
        let mut start = 0;
        for m in SPLIT_SPACE_QUOTATION_AT_END_OF_SENTENCE
            .find_iter(&text)
            .map_while(Result::ok)
        {
            pieces.push(text[start..m.start()].to_owned());
            start = m.end();
        }
        pieces.push(text[start..].to_owned());
        return pieces;
    }
    let text = strip(&text.replace('\n', "")).to_owned();
    if text.is_empty() { vec![] } else { vec![text] }
}

regex!(
    EXCLAMATION_WORDS = r"!X\x{169}|!Kung|\x{1C3}\x{2BC}O\x{1C3}Kung|!Xuun|!Kung\-Ekoka|\x{1C3}Hu|\x{1C3}Khung|\x{1C3}Ku|\x{1C3}ung|\x{1C3}Xo|\x{1C3}X\x{FB}|\x{1C3}Xung|\x{1C3}X\x{169}|!Xun|Yahoo!|Y!J|Yum!"
);
regex!(DOUBLE_PUNCTUATION = r"^(\?!|!\?|\?\?|!!)");
regex!(QUESTION_MARK_IN_QUOTATION = r#"\?(?=('|"))"#);
regex!(EXCLAMATION_IN_QUOTATION = r#"!(?=('|"))"#);
regex!(EXCLAMATION_BEFORE_COMMA_MID_SENTENCE = r"!(?=,\s[a-z])");
regex!(EXCLAMATION_MID_SENTENCE = r"!(?=\s[a-z])");
regex!(EXCLAMATION_AT_END = r"&\x{14F4}&$");
regex!(
    SENTENCE_BOUNDARY = r#"\x{FF08}(?:[^\x{FF09}])*\x{FF09}(?=\s?[A-Z])|\x{300C}(?:[^\x{300D}])*\x{300D}(?=\s[A-Z])|\((?:[^\)]){2,}\)(?=\s[A-Z])|'(?:[^'])*[^,]'(?=\s[A-Z])|"(?:[^"])*[^,]"(?=\s[A-Z])|\x{201C}(?:[^\x{201D}])*[^,]\x{201D}(?=\s[A-Z])|[\x{3002}\x{FF0E}.\x{FF01}!?\x{FF1F} ]{2,}|\S.*?[\x{3002}\x{FF0E}.\x{FF01}!?\x{FF1F}\x{238}\x{239}\x{2609}\x{2608}\x{2607}\x{2604}]|[\x{3002}\x{FF0E}.\x{FF01}!?\x{FF1F}]"#
);

/// pysbd's `process_text`, for a part with punctuation in it.
fn process_text(text: &str) -> Vec<String> {
    let mut text = text.to_owned();
    if !text.ends_with(PUNCTUATIONS) {
        text.push('\u{238}');
    }
    text = sub_with(&EXCLAMATION_WORDS, &text, |caps| {
        replace_punctuation(&caps[0], true)
    });
    text = between_punctuation(&text);
    if !search(&DOUBLE_PUNCTUATION, &text) {
        text = text
            .replace("?!", "\u{2609}")
            .replace("!?", "\u{2608}")
            .replace("??", "\u{2607}")
            .replace("!!", "\u{2604}");
    }
    text = sub(&QUESTION_MARK_IN_QUOTATION, &text, "&\u{14f7}&");
    for rule in [
        &EXCLAMATION_IN_QUOTATION,
        &EXCLAMATION_BEFORE_COMMA_MID_SENTENCE,
        &EXCLAMATION_MID_SENTENCE,
    ] {
        text = sub(rule, &text, "&\u{14f4}&");
    }
    text = sub(&EXCLAMATION_AT_END, &text, "!");
    findall(&SENTENCE_BOUNDARY, &text)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// pysbd's `replace_punctuation`, of what sits between quotes or dashes: `'` too unless the
/// quotes are single ones.
fn replace_punctuation(text: &str, quotes: bool) -> String {
    let text = text
        .replace('.', "\u{222f}")
        .replace('\u{3002}', "&\u{14f0}&")
        .replace('\u{ff0e}', "&\u{14f1}&")
        .replace('\u{ff01}', "&\u{14f3}&")
        .replace('!', "&\u{14f4}&")
        .replace('?', "&\u{14f7}&")
        .replace('\u{ff1f}', "&\u{14f8}&");
    if quotes {
        text.replace('\'', "&\u{238b}&")
    } else {
        text
    }
}

regex!(WORD_WITH_LEADING_APOSTROPHE = r"(?<=\s)'(?:[^']|'[a-zA-Z])*'\S");
regex!(APOSTROPHE_BEFORE_SPACE = r"'\s");
regex!(BETWEEN_SINGLE_QUOTES = r"(?<=\s)'(?:[^']|'[a-zA-Z])*'");

/// pysbd's `BetweenPunctuation.replace`, for the quotes and dashes the game's text can have.
fn between_punctuation(text: &str) -> String {
    let mut text = text.to_owned();
    if !search(&WORD_WITH_LEADING_APOSTROPHE, &text) || search(&APOSTROPHE_BEFORE_SPACE, &text) {
        text = sub_with(&BETWEEN_SINGLE_QUOTES, &text, |caps| {
            replace_punctuation(&caps[0], false)
        });
    }
    // pysbd's patterns for these emulate atomic groups with a backreference into a lookahead;
    // without backslashes in the text, they come to a quoted run that is not empty and a dashed
    // one that may be.
    text = replace_between(&text, "\"", |c| c != '"', false);
    replace_between(&text, "--", |c| c != '-', true)
}

/// Replaces the punctuation in each `delimiter`, run of `inner` characters, `delimiter`, left
/// to right, where the run must not be empty unless `empty`.
fn replace_between(text: &str, delimiter: &str, inner: fn(char) -> bool, empty: bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(delimiter) {
        let after = &rest[open + delimiter.len()..];
        let run = after.find(|c| !inner(c)).unwrap_or(after.len());
        if (run > 0 || empty) && after[run..].starts_with(delimiter) {
            let end = open + 2 * delimiter.len() + run;
            out.push_str(&rest[..open]);
            out.push_str(&replace_punctuation(&rest[open..end], true));
            rest = &rest[end..];
        } else {
            // No match here: the next try starts a character on.
            let next = open + rest[open..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&rest[..next]);
            rest = &rest[next..];
        }
    }
    out.push_str(rest);
    out
}

mod lists {
    //! pysbd's `ListItemReplacer.add_line_break`, which marks the periods of list items and
    //! breaks the text before them.

    use super::*;

    const ROMAN_NUMERALS: [&str; 24] = [
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv",
        "x", "xi", "xii", "xiii", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    ];

    regex!(
        ALPHABETICAL_LIST_WITH_PERIODS = r"(?<=^)[a-z](?=\.)|(?<=\A)[a-z](?=\.)|(?<=\s)[a-z](?=\.)"
    );
    regex!(
        ALPHABETICAL_LIST_LETTERS_AND_PERIODS = r"(?i)(?<=^)[a-z]\.|(?<=\A)[a-z]\.|(?<=\s)[a-z]\."
    );
    regex!(
        NUMBERED_LIST_1 =
            r"\s\d{1,2}(?=\.\s)|^\d{1,2}(?=\.\s)|(?<=\s-)\d{1,2}(?=\.\s)|(?<=^-)\d{1,2}(?=\.\s)"
    );
    regex!(
        NUMBERED_LIST_2 = r"(?<=\s)\d{1,2}\.(?=\s)|^\d{1,2}\.(?=\s)|(?<=\s-)\d{1,2}\.(?=\s)|(?<=^-)\d{1,2}\.(?=\s)"
    );
    regex!(LIST_ITEMS_ACROSS_LINES = r"\x{2668}.+(\n|\r).+\x{2668}");
    regex!(FOR_LIST_ITEM = r"for\s\d{1,2}\x{2668}\s[a-z]");
    regex!(SPACE_BETWEEN_LIST_ITEMS_FIRST = r"(?<=\S\S)\s(?=\S\s*\d+\x{2668})");
    regex!(SPACE_BETWEEN_LIST_ITEMS_SECOND = r"(?<=\S\S)\s(?=\d{1,2}\x{2668})");

    pub(super) fn add_line_breaks(text: &str) -> String {
        let latin: Vec<String> = ('a'..='z').map(String::from).collect();
        let latin: Vec<&str> = latin.iter().map(String::as_str).collect();
        let text = alphabetical_list(text, &latin);
        let text = alphabetical_list(&text, &ROMAN_NUMERALS);
        numbered_list_with_periods(&text)
    }

    /// pysbd's `iterate_alphabet_array` for items lettered from `alphabet` with periods.
    fn alphabetical_list(text: &str, alphabet: &[&str]) -> String {
        let index = |item: &str| alphabet.iter().position(|a| *a == item);
        let items: Vec<&str> = findall(&ALPHABETICAL_LIST_WITH_PERIODS, text)
            .into_iter()
            .filter(|item| index(item).is_some())
            .collect();
        let mut text = text.to_owned();
        let distance = |a: &str, b: &str| index(a).unwrap() as i64 - index(b).unwrap() as i64;
        for (i, &item) in items.iter().enumerate() {
            // Python's items[i - 1] is the last item when i is 0.
            let previous = items[(i + items.len() - 1) % items.len()];
            let replace = if i == items.len() - 1 {
                distance(previous, item).abs() == 1
            } else {
                distance(items[i + 1], item) == 1 || distance(previous, item).abs() == 1
            };
            if replace {
                text = sub_with(&ALPHABETICAL_LIST_LETTERS_AND_PERIODS, &text, |caps| {
                    let found = &caps[0];
                    if found.trim_matches('.') == item {
                        format!("\r{item}\u{222f}")
                    } else {
                        found.to_owned()
                    }
                });
            }
        }
        text
    }

    fn numbered_list_with_periods(text: &str) -> String {
        let items: Vec<i64> = findall(&NUMBERED_LIST_1, text)
            .into_iter()
            .map(|item| strip(item).parse().unwrap())
            .collect();
        let mut text = text.to_owned();
        for (i, &item) in items.iter().enumerate() {
            let next = items.get(i + 1);
            let previous = i.checked_sub(1).map(|p| items[p]);
            let listed = if i + 1 < items.len() && next == Some(&(item + 1)) {
                true
            } else if let Some(previous) = previous {
                item - 1 == previous || (item == 0 && previous == 9) || (item == 9 && previous == 0)
            } else {
                false
            };
            if listed {
                let item = item.to_string();
                text = sub_with(&NUMBERED_LIST_2, &text, |caps| {
                    let found = strip(&caps[0]);
                    let chomped = if found.chars().count() == 1 {
                        found
                    } else {
                        found.trim_matches(['.', ']', ')'])
                    };
                    if chomped == item {
                        format!("{item}\u{2668}")
                    } else {
                        found.to_owned()
                    }
                });
            }
        }
        if text.contains('\u{2668}')
            && !search(&LIST_ITEMS_ACROSS_LINES, &text)
            && !search(&FOR_LIST_ITEM, &text)
        {
            text = sub(&SPACE_BETWEEN_LIST_ITEMS_FIRST, &text, "\r");
            text = sub(&SPACE_BETWEEN_LIST_ITEMS_SECOND, &text, "\r");
        }
        text.replace('\u{2668}', "\u{222f}")
    }
}

mod abbreviations {
    //! pysbd's `AbbreviationReplacer`, which marks the periods of abbreviations.

    use super::*;

    const ABBREVIATIONS: [&str; 188] = [
        "adj",
        "adm",
        "adv",
        "al",
        "ala",
        "alta",
        "apr",
        "arc",
        "ariz",
        "ark",
        "art",
        "assn",
        "asst",
        "attys",
        "aug",
        "ave",
        "bart",
        "bld",
        "bldg",
        "blvd",
        "brig",
        "bros",
        "btw",
        "cal",
        "calif",
        "capt",
        "cl",
        "cmdr",
        "co",
        "col",
        "colo",
        "comdr",
        "con",
        "conn",
        "corp",
        "cpl",
        "cres",
        "ct",
        "d.phil",
        "dak",
        "dec",
        "del",
        "dept",
        "det",
        "dist",
        "dr",
        "dr.phil",
        "dr.philos",
        "drs",
        "e.g",
        "ens",
        "esp",
        "esq",
        "etc",
        "exp",
        "expy",
        "ext",
        "feb",
        "fed",
        "fla",
        "ft",
        "fwy",
        "fy",
        "ga",
        "gen",
        "gov",
        "hon",
        "hosp",
        "hr",
        "hway",
        "hwy",
        "i.e",
        "ia",
        "id",
        "ida",
        "ill",
        "inc",
        "ind",
        "ing",
        "insp",
        "is",
        "jan",
        "jr",
        "jul",
        "jun",
        "kan",
        "kans",
        "ken",
        "ky",
        "la",
        "lt",
        "ltd",
        "maj",
        "man",
        "mar",
        "mass",
        "may",
        "md",
        "me",
        "med",
        "messrs",
        "mex",
        "mfg",
        "mich",
        "min",
        "minn",
        "miss",
        "mlle",
        "mm",
        "mme",
        "mo",
        "mont",
        "mr",
        "mrs",
        "ms",
        "msgr",
        "mssrs",
        "mt",
        "mtn",
        "neb",
        "nebr",
        "nev",
        "no",
        "nos",
        "nov",
        "nr",
        "oct",
        "ok",
        "okla",
        "ont",
        "op",
        "ord",
        "ore",
        "p",
        "pa",
        "pd",
        "pde",
        "penn",
        "penna",
        "pfc",
        "ph",
        "ph.d",
        "pl",
        "plz",
        "pp",
        "prof",
        "pvt",
        "que",
        "rd",
        "rs",
        "ref",
        "rep",
        "reps",
        "res",
        "rev",
        "rt",
        "sask",
        "sec",
        "sen",
        "sens",
        "sep",
        "sept",
        "sfc",
        "sgt",
        "sr",
        "st",
        "supt",
        "surg",
        "tce",
        "tenn",
        "tex",
        "univ",
        "usafa",
        "u.s",
        "ut",
        "va",
        "v",
        "ver",
        "viz",
        "vs",
        "vt",
        "wash",
        "wis",
        "wisc",
        "wy",
        "wyo",
        "yuk",
        "fig",
    ];
    const PREPOSITIVE: [&str; 33] = [
        "adm", "attys", "brig", "capt", "cmdr", "col", "cpl", "det", "dr", "gen", "gov", "ing",
        "lt", "maj", "mr", "mrs", "ms", "mt", "messrs", "mssrs", "prof", "ph", "rep", "reps",
        "rev", "sen", "sens", "sgt", "st", "supt", "v", "vs", "fig",
    ];
    const NUMBER: [&str; 6] = ["art", "ext", "no", "nos", "p", "pp"];
    const SENTENCE_STARTERS: [&str; 23] = [
        "A", "Being", "Did", "For", "He", "How", "However", "I", "In", "It", "Millions", "More",
        "She", "That", "The", "There", "They", "We", "What", "When", "Where", "Who", "Why",
    ];

    regex!(POSSESSIVE_ABBREVIATION = r"\.(?='s\s)|\.(?='s$)|\.(?='s\z)");
    regex!(KOMMANDITGESELLSCHAFT = r"(?<=Co)\.(?=\sKG)");
    regex!(SINGLE_UPPER_CASE_LETTER_AT_START_OF_LINE = r"(?<=^[A-Z])\.(?=\s)");
    regex!(SINGLE_UPPER_CASE_LETTER = r"(?<=\s[A-Z])\.(?=,?\s)");
    regex!(MULTI_PERIOD_ABBREVIATION = r"(?i)\b[a-z](?:\.[a-z])+[.]");
    regex!(UPPER_CASE_PM = r"(?<= P\x{222F}M)\x{222F}(?=\s[A-Z])");
    regex!(UPPER_CASE_AM = r"(?<=A\x{222F}M)\x{222F}(?=\s[A-Z])");
    regex!(LOWER_CASE_PM = r"(?<=p\x{222F}m)\x{222F}(?=\s[A-Z])");
    regex!(LOWER_CASE_AM = r"(?<=a\x{222F}m)\x{222F}(?=\s[A-Z])");
    static ABBREVIATION_AS_SENTENCE_BOUNDARY: LazyLock<Regex> = LazyLock::new(|| {
        let starters: Vec<String> = SENTENCE_STARTERS
            .iter()
            .map(|word| format!(r"(?=\s{word}\s)"))
            .collect();
        let pattern = String::from(
            r"(U\x{222F}S|U\.S|U\x{222F}K|E\x{222F}U|E\.U|U\x{222F}S\x{222F}A|U\.S\.A|I|i.v|I.V)\x{222F}(",
        ) + &starters.join("|")
            + ")";
        Regex::new(&pattern).unwrap()
    });

    pub(super) fn replace(text: &str) -> String {
        let mut text = sub(&POSSESSIVE_ABBREVIATION, text, "\u{222f}");
        text = sub(&KOMMANDITGESELLSCHAFT, &text, "\u{222f}");
        text = sub(
            &SINGLE_UPPER_CASE_LETTER_AT_START_OF_LINE,
            &text,
            "\u{222f}",
        );
        text = sub(&SINGLE_UPPER_CASE_LETTER, &text, "\u{222f}");
        text = split_lines(&text)
            .into_iter()
            .map(search_for_abbreviations)
            .collect();
        text = sub_with(&MULTI_PERIOD_ABBREVIATION, &text, |caps| {
            caps[0].replace('.', "\u{222f}")
        });
        for rule in [
            &UPPER_CASE_PM,
            &UPPER_CASE_AM,
            &LOWER_CASE_PM,
            &LOWER_CASE_AM,
        ] {
            text = sub(rule, &text, ".");
        }
        sub(&ABBREVIATION_AS_SENTENCE_BOUNDARY, &text, "${1}.")
    }

    /// Python's `str.splitlines(True)`.
    fn split_lines(text: &str) -> Vec<&str> {
        let breaks = |c: char| {
            matches!(
                c,
                '\n' | '\r' | '\x0b' | '\x0c' | '\x1c'
                    ..='\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
            )
        };
        let mut lines = Vec::new();
        let mut start = 0;
        let mut chars = text.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if breaks(c) {
                let mut end = i + c.len_utf8();
                if c == '\r' && chars.peek().is_some_and(|&(_, n)| n == '\n') {
                    chars.next();
                    end += 1;
                }
                lines.push(&text[start..end]);
                start = end;
            }
        }
        if start < text.len() {
            lines.push(&text[start..]);
        }
        lines
    }

    fn search_for_abbreviations(line: &str) -> String {
        let lowered = line.to_lowercase();
        let mut text = line.to_owned();
        for abbreviation in ABBREVIATIONS {
            if !lowered.contains(abbreviation) {
                continue;
            }
            // pysbd puts the abbreviation into this pattern unescaped, its periods matching any
            // character.
            for found in find_abbreviation(&text, abbreviation) {
                let found = strip(&found).to_owned();
                let lower = found.to_lowercase();
                text = if PREPOSITIVE.contains(&lower.as_str()) {
                    replace_period(&text, &found, |after| {
                        after.starts_with(is_space) || colon_digits(after)
                    })
                } else if NUMBER.contains(&lower.as_str()) {
                    replace_period(&text, &found, |after| {
                        let mut chars = after.chars();
                        chars.next().is_some_and(is_space)
                            && (chars.next().is_some_and(|c| c.is_ascii_digit())
                                || after.trim_start_matches(is_space).starts_with('('))
                    })
                } else {
                    replace_period(&text, &found, period_of_abbreviation_follows)
                };
            }
        }
        text
    }

    /// `re.findall(r"(?:^|\s|\r|\n)" + abbreviation, text, flags=re.IGNORECASE)`, the
    /// abbreviation's periods matching any character but a newline.
    fn find_abbreviation(text: &str, abbreviation: &str) -> Vec<String> {
        let pattern: Vec<char> = abbreviation.chars().collect();
        let matches_at = |rest: &str| -> Option<usize> {
            let mut chars = rest.char_indices();
            for &p in &pattern {
                let (_, c) = chars.next()?;
                let ok = if p == '.' {
                    c != '\n'
                } else {
                    c.to_lowercase().eq(p.to_lowercase())
                };
                if !ok {
                    return None;
                }
            }
            Some(chars.next().map_or(rest.len(), |(i, _)| i))
        };
        let mut found = Vec::new();
        let mut i = 0;
        while i <= text.len() {
            let rest = &text[i..];
            let at_start = if i == 0 { matches_at(rest) } else { None };
            let after_space = rest
                .chars()
                .next()
                .filter(|&c| is_space(c))
                .and_then(|c| matches_at(&rest[c.len_utf8()..]).map(|n| n + c.len_utf8()));
            if let Some(len) = at_start.or(after_space) {
                found.push(rest[..len].to_owned());
                i += len.max(1);
            } else {
                i += rest.chars().next().map_or(1, char::len_utf8);
            }
        }
        found
    }

    /// `(?<=\s{abbreviation})\.(?=...)` replaced with pysbd's period placeholder in `" " + text`, the lookahead a test
    /// of what follows the period.
    fn replace_period(text: &str, abbreviation: &str, follows: impl Fn(&str) -> bool) -> String {
        let spaced = format!(" {text}");
        let mut out = String::with_capacity(spaced.len());
        for (i, c) in spaced.char_indices() {
            let before = &spaced[..i];
            let marks = c == '.'
                && before.ends_with(abbreviation)
                && before[..before.len() - abbreviation.len()]
                    .chars()
                    .next_back()
                    .is_some_and(is_space)
                && follows(&spaced[i + 1..]);
            if marks {
                out.push('\u{222f}');
            } else {
                out.push(c);
            }
        }
        out[1..].to_owned()
    }

    /// `:\d+`.
    fn colon_digits(after: &str) -> bool {
        after
            .strip_prefix(':')
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    }

    /// `((\.|\:|-|\?|,)|(\s([a-z]|I\s|I'm|I'll|\d|\()))`.
    fn period_of_abbreviation_follows(after: &str) -> bool {
        if after.starts_with(['.', ':', '-', '?', ',']) {
            return true;
        }
        let mut chars = after.chars();
        if !chars.next().is_some_and(is_space) {
            return false;
        }
        let rest = chars.as_str();
        rest.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '(')
            || rest.starts_with("I'm")
            || rest.starts_with("I'll")
            || rest
                .strip_prefix('I')
                .is_some_and(|r| r.starts_with(is_space))
    }
}
