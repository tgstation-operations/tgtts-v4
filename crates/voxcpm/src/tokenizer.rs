use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use serde::Deserialize;
use serde_json::json;

use crate::Error;

/// SentencePiece's mark for a space, which the normalizer puts in place of each.
const SPACE: char = '\u{2581}';

/// The text tokenizer the Python service ran: transformers' `LlamaTokenizerFast` over the
/// checkpoint's `tokenizer.json`, wrapped by upstream's `mask_multichar_chinese_tokens`, which
/// splits each token of several CJK ideographs into a token per ideograph.
///
/// It implements as much of Hugging Face's tokenizers as VoxCPM2's `tokenizer.json` uses: its
/// added tokens are split out of the text first, and each piece between them has a space mark
/// put before it and in place of each space, and is split by a BPE model that falls back to
/// bytes. A file that asks for more is refused.
///
/// This is the file as transformers 4 runs it. transformers 5 builds the tokenizer anew, marking
/// spaces in a way that differs for a piece that starts with one.
#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), Merged>,
    bytes: Vec<u32>,
    added: Vec<(String, u32)>,
    splits: HashMap<u32, Vec<u32>>,
}

#[derive(Clone, Copy, Debug)]
struct Merged {
    rank: u32,
    id: u32,
}

#[derive(Deserialize)]
struct File {
    added_tokens: Vec<AddedToken>,
    normalizer: serde_json::Value,
    pre_tokenizer: serde_json::Value,
    model: Bpe,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

#[derive(Deserialize)]
struct Bpe {
    #[serde(rename = "type")]
    kind: String,
    dropout: Option<f64>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
}

/// A merge as tokenizers writes it, joined by a space in older files.
#[derive(Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Joined(String),
    Pair(String, String),
}

impl Tokenizer {
    /// Reads a `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// If it is not one, or asks for what this does not implement.
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let unsupported = |what: String| Error::Tokenizer(what);
        let file: File = serde_json::from_str(json).map_err(|e| unsupported(e.to_string()))?;
        let normalizer = json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": SPACE.to_string()},
                {"type": "Replace", "pattern": {"String": " "}, "content": SPACE.to_string()},
            ],
        });
        if file.normalizer != normalizer {
            return Err(unsupported(format!("normalizer {}", file.normalizer)));
        }
        if !file.pre_tokenizer.is_null() {
            return Err(unsupported(format!("pre_tokenizer {}", file.pre_tokenizer)));
        }
        let model = file.model;
        let plain = model.kind == "BPE"
            && model.dropout.is_none()
            && model.continuing_subword_prefix.is_none()
            && model.end_of_word_suffix.is_none()
            && model.byte_fallback
            && !model.ignore_merges;
        if !plain {
            return Err(unsupported(
                "a model other than BPE falling back to bytes, without dropout, affixes or \
                 ignore_merges"
                    .into(),
            ));
        }
        let vocab = model.vocab;
        let id = |token: &str| {
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| unsupported(format!("{token:?} is not in the vocabulary")))
        };
        let bytes = (0..=u8::MAX)
            .map(|byte| id(&format!("<{byte:#04X}>")))
            .collect::<Result<_, _>>()?;
        let mut merges = HashMap::with_capacity(model.merges.len());
        for (rank, entry) in model.merges.into_iter().enumerate() {
            let (a, b) = match entry {
                MergeEntry::Pair(a, b) => (a, b),
                MergeEntry::Joined(joined) => match joined.split(' ').collect::<Vec<_>>()[..] {
                    [a, b] => (a.to_owned(), b.to_owned()),
                    _ => return Err(unsupported(format!("merge {joined:?}"))),
                },
            };
            let merged = Merged {
                rank: rank as u32,
                id: id(&format!("{a}{b}"))?,
            };
            merges.insert((id(&a)?, id(&b)?), merged);
        }
        let mut added = Vec::with_capacity(file.added_tokens.len());
        for token in file.added_tokens {
            if token.normalized || token.single_word || token.lstrip || token.rstrip {
                let content = &token.content;
                return Err(unsupported(format!(
                    "added token {content:?} normalized, single_word, lstrip or rstrip"
                )));
            }
            added.push((token.content, token.id));
        }
        let unk = model.unk_token.as_deref().map(id).transpose()?;
        let splits = multichar_chinese_splits(&vocab, unk);
        Ok(Self {
            vocab,
            merges,
            bytes,
            added,
            splits,
        })
    }

    /// The ids of `text`'s tokens, with no special token added.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![];
        let mut piece = 0;
        let mut at = 0;
        while at < text.len() {
            let longest = self
                .added
                .iter()
                .filter(|(content, _)| text[at..].starts_with(content.as_str()))
                .max_by_key(|(content, _)| content.len());
            match longest {
                Some((content, added)) => {
                    self.encode_piece(&text[piece..at], &mut ids);
                    ids.push(*added);
                    at += content.len();
                    piece = at;
                }
                None => at += text[at..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.encode_piece(&text[piece..], &mut ids);
        ids.into_iter()
            .flat_map(|id| match self.splits.get(&id) {
                Some(split) => split.clone(),
                None => vec![id],
            })
            .collect()
    }

    fn encode_piece(&self, piece: &str, ids: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let normalized: String = std::iter::once(SPACE)
            .chain(piece.chars().map(|c| if c == ' ' { SPACE } else { c }))
            .collect();
        let mut symbols = vec![];
        for c in normalized.chars() {
            match self.vocab.get(c.encode_utf8(&mut [0; 4]) as &str) {
                Some(&id) => symbols.push(id),
                None => {
                    let mut utf8 = [0; 4];
                    let bytes = c.encode_utf8(&mut utf8).bytes();
                    symbols.extend(bytes.map(|byte| self.bytes[usize::from(byte)]));
                }
            }
        }
        ids.extend(self.merge(symbols));
    }

    /// Merges `symbols` as tokenizers' `Word::merge_all` does: always the pair of the lowest
    /// rank, leftmost among equals, judging a queued pair by the token it would make.
    fn merge(&self, ids: Vec<u32>) -> Vec<u32> {
        let len = ids.len();
        let mut symbols: Vec<Symbol> = ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| Symbol {
                id,
                prev: i.checked_sub(1),
                next: (i + 1 < len).then_some(i + 1),
                removed: false,
            })
            .collect();
        let queued = |pos: usize, left: u32, right: u32| {
            let merged = self.merges.get(&(left, right))?;
            Some(Candidate {
                pos,
                rank: merged.rank,
                id: merged.id,
            })
        };
        let mut queue: BinaryHeap<Candidate> = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(pos, pair)| queued(pos, pair[0].id, pair[1].id))
            .collect();
        while let Some(top) = queue.pop() {
            let symbol = symbols[top.pos];
            let Some(next) = symbol.next.filter(|_| !symbol.removed) else {
                continue;
            };
            let right = symbols[next];
            let current = self.merges.get(&(symbol.id, right.id));
            if current.is_none_or(|merged| merged.id != top.id) {
                continue;
            }
            symbols[top.pos].id = top.id;
            symbols[top.pos].next = right.next;
            symbols[next].removed = true;
            if let Some(after) = right.next {
                symbols[after].prev = Some(top.pos);
            }
            let symbol = symbols[top.pos];
            if let Some(prev) = symbol.prev {
                queue.extend(queued(prev, symbols[prev].id, symbol.id));
            }
            if let Some(next) = symbol.next {
                queue.extend(queued(top.pos, symbol.id, symbols[next].id));
            }
        }
        symbols
            .into_iter()
            .filter(|symbol| !symbol.removed)
            .map(|symbol| symbol.id)
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    removed: bool,
}

/// A merge waiting in the queue, popped lowest rank first, then leftmost.
#[derive(Debug, PartialEq, Eq)]
struct Candidate {
    pos: usize,
    rank: u32,
    id: u32,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.rank, other.pos).cmp(&(self.rank, self.pos))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// What upstream's `mask_multichar_chinese_tokens` splits: each token that, but for its space
/// marks, is a token of two or more ideographs of the CJK Unified Ideographs block, into the
/// ids of those ideographs, the unknown token's for one that is not a token.
fn multichar_chinese_splits(
    vocab: &HashMap<String, u32>,
    unk: Option<u32>,
) -> HashMap<u32, Vec<u32>> {
    let chinese = |token: &str| {
        token.chars().count() >= 2
            && token
                .chars()
                .all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
    };
    vocab
        .iter()
        .filter_map(|(token, &id)| {
            let clean = token.replace(SPACE, "");
            if !chinese(&clean) || !vocab.contains_key(&clean) {
                return None;
            }
            let chars = clean.chars().map(|c| {
                let id = vocab.get(c.encode_utf8(&mut [0; 4]) as &str);
                id.copied().or(unk)
            });
            Some((id, chars.collect::<Option<Vec<_>>>()?))
        })
        .collect()
}
