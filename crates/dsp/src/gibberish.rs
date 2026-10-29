//! The gibberish radio: a message over the radio with some of its words lost in static, as the
//! old radio-gibberish service made it.
//!
//! The service masked some words of the message, aligned the message to the speech with wav2vec2
//! to time each letter, and ducked the speech under static where the masked words fell. The
//! alignment is not here: [`corrupted_spans`] takes the letters' times it gives.

use crate::{
    RATE, Random,
    radio::{self, Assets, Pcm, narrow},
    segment::Segment,
};

/// The words of `text` joined with single spaces, those of them that `gibberish`, the game's
/// garbled copy of `text`, changed replaced with as many `#`. Where the copy is the same text, as
/// the game sends today, from 35 to 65% of the words, drawn by `random`, are replaced instead, as
/// the old service's `mask_string` replaced them, ignoring the copy.
pub fn mask(text: &str, gibberish: &str, random: &mut impl Random) -> String {
    let plain: Vec<&str> = words(text).collect();
    let garbled: Vec<&str> = words(gibberish).collect();
    let masked = if garbled == plain {
        let percent = 0.35 + (0.65 - 0.35) * random.random();
        let count = (plain.len() as f64 * percent) as usize;
        sample(plain.len(), count, random)
    } else {
        changed(&plain, &garbled)
    };
    let mut words: Vec<String> = plain.into_iter().map(str::to_owned).collect();
    for i in masked {
        words[i] = "#".repeat(words[i].chars().count());
    }
    words.join(" ")
}

/// The indices of the words of `plain` left out of a longest alignment of them, in order, with
/// the same words of `garbled`, which aligns each word to the earliest it can.
fn changed(plain: &[&str], garbled: &[&str]) -> Vec<usize> {
    let (n, m) = (plain.len(), garbled.len());
    // The most words of plain[i..] that align with words of garbled[j..].
    let mut aligned = vec![vec![0; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            aligned[i][j] = if plain[i] == garbled[j] {
                aligned[i + 1][j + 1] + 1
            } else {
                aligned[i + 1][j].max(aligned[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut changed = Vec::new();
    while i < n {
        if j < m && plain[i] == garbled[j] {
            (i, j) = (i + 1, j + 1);
        } else if j == m || aligned[i + 1][j] >= aligned[i][j + 1] {
            changed.push(i);
            i += 1;
        } else {
            j += 1;
        }
    }
    changed
}

/// Python's `random.sample(range(n), k)`, taking the same draws: from a pool of what is left. It
/// draws by rejecting repeats instead once `n` exceeds `21 + 4^ceil(log4(3k))`, which with `k` at
/// least 35% of `n` it never does.
fn sample(n: usize, k: usize, random: &mut impl Random) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    (0..k)
        .map(|i| {
            let j = random.below(n - i);
            let chosen = pool[j];
            pool[j] = pool[n - i - 1];
            chosen
        })
        .collect()
}

/// The words of `text` as Python's `str.split` gives them.
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_space).filter(|w| !w.is_empty())
}

/// Python's `str.isspace` for one character.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// The times, in seconds, of the letters of the words of `text` that `masked` masks, merged
/// where less than 50 ms apart, as the old service's `find_corrupted_spans` meant them. `masked`
/// is `text` as [`mask`] gives it, word for word, and `letters` are the start and end of each
/// character of `text` the alignment times, its letters and apostrophes, in order.
pub fn corrupted_spans(text: &str, masked: &str, letters: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let masked: Vec<&str> = words(masked).collect();
    let mut letters = letters.iter();
    let mut spans = Vec::new();
    for (i, word) in words(text).enumerate() {
        let timed = word
            .chars()
            .filter(|&c| c.is_ascii_alphabetic() || c == '\'');
        let times: Vec<(f64, f64)> = letters.by_ref().take(timed.count()).copied().collect();
        if masked.get(i).is_some_and(|m| *m != word) {
            spans.extend(times);
        }
    }
    merge(spans, 0.05)
}

/// The old service's `merge_spans`: sorted, and merged into the last where they start no more
/// than `gap` after it ends.
fn merge(mut spans: Vec<(f64, f64)>, gap: f64) -> Vec<(f64, f64)> {
    spans.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start - last.1 <= gap => {
                if end > last.1 {
                    last.1 = end;
                }
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// `speech` over the radio as [`radio::radio`] makes it, but ducked under static over `spans`,
/// in seconds, as the old radio-gibberish service made it: the speech faded out over 25 ms from
/// 60 ms before each span and back in over 50 ms to 80 ms after it, with the static, at 0.85,
/// faded in and out alike. Spans are cut to the speech, and those whose margins overlap are
/// ducked as one.
pub fn gibberish(speech: &Pcm, assets: &Assets, spans: &[(f64, f64)]) -> Segment {
    let ducked = duck(&narrow(speech), spans, assets.noise());
    radio::finish(ducked, assets)
}

/// The old service's `apply_corruption_static` with `build_static_mask`, at its settings, but
/// over the spans as one where their margins overlap: the old service faded each span in turn,
/// so a span's fade in brought the speech back over the end of the last.
fn duck(speech: &[f64], spans: &[(f64, f64)], noise: &[f64]) -> Vec<f64> {
    let total = speech.len();
    let rate = f64::from(RATE);
    let ms = |ms: u32| (f64::from(ms * RATE.value()) / 1000.0) as i64;
    let (pre, post, fade) = (ms(60), ms(80), ms(25));
    let mut windows: Vec<(i64, i64)> = spans
        .iter()
        .map(|&(start, end)| {
            let e = ((end * rate) as i64 + post).clamp(0, total as i64);
            (((start * rate) as i64 - pre).clamp(0, e), e)
        })
        .collect();
    windows.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (s, e) in windows {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    let mut envelope = vec![1.0; total];
    let mut mask = vec![0.0; total];
    for (s, e) in merged {
        let half = (e - s) / 2;
        let (s, e) = (s as usize, e as usize);
        let (fade_in, fade_out) = (fade.min(half) as usize, (fade * 2).min(half) as usize);
        let range = s..e;
        envelope[range.clone()].fill(0.0);
        envelope[s..s + fade_in].copy_from_slice(&linspace(1.0, 0.0, fade_in));
        envelope[e - fade_out..e].copy_from_slice(&linspace(0.0, 1.0, fade_out));
        mask[range].fill(1.0);
        for (m, f) in mask[s..s + fade_in]
            .iter_mut()
            .zip(linspace(0.0, 1.0, fade_in))
        {
            *m *= f;
        }
        for (m, f) in mask[e - fade_out..e]
            .iter_mut()
            .zip(linspace(1.0, 0.0, fade_out))
        {
            *m *= f;
        }
    }
    (speech.iter().zip(&envelope))
        .zip(radio::looped(noise).zip(&mask))
        .map(|((s, d), (n, m))| s * d + n * m * 0.85)
        .collect()
}

/// numpy's `linspace(start, stop, n)`: `i * step + start`, but the last exactly `stop`.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n as f64 - 1.0);
    (0..n)
        .map(|i| match i {
            0 => start,
            i if i + 1 == n => stop,
            i => i as f64 * step + start,
        })
        .collect()
}
