//! The gibberish radio: a message over the radio with some of its words lost in static, as the
//! old radio-gibberish service made it.
//!
//! The service masked words of the message at random, aligned the message to the speech with
//! wav2vec2 to time each letter, and ducked the speech under static where the masked words fell.
//! The alignment is not here: [`corrupted_spans`] takes the letters' times it gives.

use crate::{
    RATE, Random,
    radio::{self, Assets, Error, Pcm, narrow},
    segment::Segment,
};

/// The old service's `mask_string`: from 35 to 65% of the words of `text`, drawn by `random`,
/// replaced with as many `#`, and the words joined with single spaces. The service ignored the
/// masked text the game sent and made its own this way.
pub fn mask(text: &str, random: &mut impl Random) -> String {
    let mut words: Vec<String> = text
        .split(is_space)
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    let percent = 0.35 + (0.65 - 0.35) * random.random();
    let count = (words.len() as f64 * percent) as usize;
    for i in sample(words.len(), count, random) {
        words[i] = "#".repeat(words[i].chars().count());
    }
    words.join(" ")
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

/// Python's `str.isspace` for one character.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// The old service's `find_corrupted_spans`: the times, in seconds, of the letters of `clean`
/// that `corrupted` masks, and of the spaces beside them, merged where less than 50 ms apart.
/// `letters` are the start and end of each letter of the alignment, in order.
///
/// Its quirks are kept: it pairs the letters and spaces of `clean` with the characters of
/// `corrupted` by position, so a digit or punctuation mark shifts the pairing, and it counts
/// through `letters` by the letters of `clean`, though the alignment also times apostrophes.
pub fn corrupted_spans(clean: &str, corrupted: &str, letters: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let clean: Vec<char> = clean
        .chars()
        .filter(|&c| c.is_alphabetic() || c == ' ')
        .collect();
    let corrupt: Vec<char> = corrupted
        .chars()
        .filter(|&c| c == ' ' || !is_space(c))
        .collect();
    let mut masked = Vec::new();
    let mut letter = 0;
    for (i, &c) in clean.iter().enumerate() {
        if c == ' ' {
            continue;
        }
        if corrupt.get(i).is_some_and(|c| !c.is_alphabetic()) {
            masked.push(letter);
        }
        letter += 1;
    }
    let mut spans = Vec::new();
    let mut at = 0;
    for (&c, &corrupt) in clean.iter().zip(&corrupt) {
        if c == ' ' {
            // Python's at - 1 is -1 at the first letter, which no letter is.
            let adjacent = (at > 0 && masked.contains(&(at - 1))) || masked.contains(&at);
            if !is_space(corrupt) && adjacent && at > 0 && at < letters.len() {
                spans.push((letters[at - 1].1, letters[at].0));
            }
        } else {
            if !corrupt.is_alphabetic() && at < letters.len() {
                spans.push(letters[at]);
            }
            at += 1;
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
/// faded in and out alike.
pub fn gibberish(speech: &Pcm, assets: &Assets, spans: &[(f64, f64)]) -> Result<Segment, Error> {
    let narrowed = narrow(speech)?;
    let ducked = duck(&narrowed, spans, assets.noise())?;
    radio::finish(ducked, assets)
}

/// The old service's `apply_corruption_static` with `build_static_mask`, at its settings.
fn duck(speech: &[f64], spans: &[(f64, f64)], noise: &[f64]) -> Result<Vec<f64>, Error> {
    let total = speech.len();
    let rate = f64::from(RATE);
    let ms = |ms: u32| (f64::from(ms * RATE) / 1000.0) as i64;
    let (pre, post, fade) = (ms(60), ms(80), ms(25));
    let mut envelope = vec![1.0; total];
    let mut mask = vec![0.0; total];
    for &(start, end) in spans {
        let s = ((start * rate) as i64 - pre).max(0);
        let e = ((end * rate) as i64 + post).min(total as i64);
        // numpy's linspace of a negative length, where the span starts past the speech's end.
        let half = (e - s).div_euclid(2);
        if half < 0 {
            return Err(Error::SpanPastEnd(start));
        }
        let (s, e) = (s as usize, e as usize);
        let (fade_in, fade_out) = (fade.min(half) as usize, (fade * 2).min(half) as usize);
        let range = s..e.max(s);
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
    let looped = noise.iter().cycle();
    Ok((speech.iter().zip(&envelope))
        .zip(looped.zip(&mask))
        .map(|((s, d), (n, m))| s * d + n * m * 0.85)
        .collect())
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
