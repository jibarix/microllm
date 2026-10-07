//! Byte-pair encoding (BPE) on characters: a tokenizer that is itself trained.
//!
//! It starts from single characters (the same ids the character-level models use) and
//! repeatedly merges the most frequent adjacent pair of tokens in the training text into a
//! new token. With no merges it *is* the character tokenizer.
//!
//! Pre-tokenization (version `PRETOK`): the text is first cut into chunks, and merges never
//! cross a chunk boundary, so a token never spans two words. Scanning left to right, a chunk is
//!   - one newline, or
//!   - an optional single space followed by a word: a letter, then letters, or an apostrophe
//!     that is followed by a letter ("tis", " we'll"), or
//!   - a run of digits, or
//!   - any other single character (punctuation, a space not followed by a letter).
//!
//! Training (on the train split only): count every adjacent token pair inside every chunk,
//! weighted by how often the chunk occurs; merge the pair with the highest count, ties broken
//! by the smaller left id, then the smaller right id. Merging replaces the pair left to right
//! without overlap ("a a a" with pair (a, a) → "aa a"). Repeat until the vocabulary reaches the
//! target size or no pair occurs twice. Encoding any text applies the merges in that same
//! order, so it reproduces exactly what training did.

use std::collections::HashMap;

pub const PRETOK: &str = "v1: newline | space? letter (letter | 'letter)* | digit+ | other char";

#[derive(Clone, Debug)]
pub struct Merge {
    pub left: usize,
    pub right: usize,
    pub id: usize,
    /// Weighted count of the pair in the training text when it was chosen.
    pub count: usize,
}

/// One point of the compression curve: after `merges` merges, the training and validation
/// text are this many tokens long.
#[derive(Clone, Debug)]
pub struct CurvePoint {
    pub merges: usize,
    pub train_tokens: usize,
    pub val_tokens: usize,
}

pub struct Bpe {
    /// Token strings: the base characters (sorted), then one per merge.
    pub vocab: Vec<String>,
    pub base: Vec<char>,
    pub merges: Vec<Merge>,
    pub curve: Vec<CurvePoint>,
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

/// Cut `text` into pre-tokenization chunks (see the module notes).
pub fn chunks(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let at = |k: usize| cs.get(k).map(|&(_, c)| c);
    let end_of = |k: usize| cs.get(k).map(|&(b, _)| b).unwrap_or(text.len());
    while i < cs.len() {
        let c = cs[i].1;
        let start = i;
        if c == ' ' && at(i + 1).is_some_and(is_letter) || is_letter(c) {
            i += if c == ' ' { 2 } else { 1 };
            loop {
                match at(i) {
                    Some(d) if is_letter(d) => i += 1,
                    Some('\'') if at(i + 1).is_some_and(is_letter) => i += 2,
                    _ => break,
                }
            }
        } else if c.is_ascii_digit() {
            while at(i).is_some_and(|d| d.is_ascii_digit()) {
                i += 1;
            }
        } else {
            i += 1;
        }
        out.push(&text[end_of(start)..end_of(i)]);
    }
    out
}

/// Replace every non-overlapping (left, right) pair in `w`, left to right, by `id`.
/// Returns how many replacements were made.
fn apply(w: &mut Vec<usize>, left: usize, right: usize, id: usize) -> usize {
    let mut n = 0;
    let mut out = Vec::with_capacity(w.len());
    let mut i = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == left && w[i + 1] == right {
            out.push(id);
            i += 2;
            n += 1;
        } else {
            out.push(w[i]);
            i += 1;
        }
    }
    *w = out;
    n
}

/// Distinct chunks of `text` as base-token ids with their counts, in a fixed (sorted) order.
fn chunk_counts(text: &str, base: &[char]) -> Vec<(Vec<usize>, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for ch in chunks(text) {
        *counts.entry(ch).or_insert(0) += 1;
    }
    let mut list: Vec<(&str, usize)> = counts.into_iter().collect();
    list.sort_unstable();
    list.into_iter().map(|(s, n)| (s.chars().map(|c| base.binary_search(&c).expect("character outside the base alphabet")).collect(), n)).collect()
}

impl Bpe {
    /// Train on `train` up to `vocab_size` tokens. `base` is every character the tokenizer
    /// must be able to encode (the whole file's alphabet, as for the character models);
    /// `val` is only measured, never learned from, to record the compression curve.
    pub fn train(train: &str, val: &str, base: Vec<char>, vocab_size: usize, curve_every: usize) -> Bpe {
        let mut vocab: Vec<String> = base.iter().map(|c| c.to_string()).collect();
        let mut words = chunk_counts(train, &base);
        let mut val_words = chunk_counts(val, &base);
        let tokens = |ws: &[(Vec<usize>, usize)]| ws.iter().map(|(w, n)| w.len() * n).sum::<usize>();
        let (mut train_tokens, mut val_tokens) = (tokens(&words), tokens(&val_words));
        let mut curve = vec![CurvePoint { merges: 0, train_tokens, val_tokens }];
        let mut merges = Vec::new();
        while vocab.len() < vocab_size {
            let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();
            for (w, n) in &words {
                for p in w.windows(2) {
                    *pairs.entry((p[0], p[1])).or_insert(0) += n;
                }
            }
            // highest count; ties → smaller left id, then smaller right id (independent of map order)
            let Some((&(left, right), &count)) = pairs.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))) else { break };
            if count < 2 {
                break;
            }
            let id = vocab.len();
            vocab.push(format!("{}{}", vocab[left], vocab[right]));
            for (w, n) in &mut words {
                train_tokens -= apply(w, left, right, id) * *n;
            }
            for (w, n) in &mut val_words {
                val_tokens -= apply(w, left, right, id) * *n;
            }
            merges.push(Merge { left, right, id, count });
            if merges.len() % curve_every.max(1) == 0 {
                curve.push(CurvePoint { merges: merges.len(), train_tokens, val_tokens });
            }
        }
        if curve.last().is_some_and(|p| p.merges != merges.len()) {
            curve.push(CurvePoint { merges: merges.len(), train_tokens, val_tokens });
        }
        Bpe { vocab, base, merges, curve }
    }

    /// Token ids for `text`: chunk it, then apply every merge in order to each distinct chunk.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut cache: HashMap<&str, Vec<usize>> = HashMap::new();
        // merges by pair, so each chunk only tries merges whose pair it contains
        let rank: HashMap<(usize, usize), usize> = self.merges.iter().enumerate().map(|(r, m)| ((m.left, m.right), r)).collect();
        let mut out = Vec::with_capacity(text.len());
        for ch in chunks(text) {
            let ids = cache.entry(ch).or_insert_with(|| {
                let mut w: Vec<usize> = ch.chars().map(|c| self.base.binary_search(&c).expect("character outside the base alphabet")).collect();
                // the lowest-ranked pair present is the next merge training would have applied
                while let Some(r) = w.windows(2).filter_map(|p| rank.get(&(p[0], p[1])).copied()).min() {
                    let m = &self.merges[r];
                    apply(&mut w, m.left, m.right, m.id);
                }
                w
            });
            out.extend_from_slice(ids);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alphabet(s: &str) -> Vec<char> {
        let mut v: Vec<char> = s.chars().collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    #[test]
    fn chunks_follow_the_grammar() {
        assert_eq!(chunks("First Citizen:\nWe'll go, 'tis  12 o'clock."), vec![
            "First", " Citizen", ":", "\n", "We'll", " go", ",", " ", "'", "tis", " ", " ", "12", " o'clock", "."
        ]);
        assert_eq!(chunks("").len(), 0);
        assert_eq!(chunks(" a'").concat(), " a'");
    }

    #[test]
    fn a_hand_worked_example() {
        // chunks: "aaab" ×1 → a a a b ; " ab" ×2 → ' ' a b
        // pairs: (a,a)=2 (a,b)=1+2=3 (' ',a)=2  → merge (a,b) first
        let text = "aaab ab ab";
        let base = alphabet(text); // ' ', a, b → ids 0, 1, 2
        let bpe = Bpe::train(text, "", base, 5, 1);
        assert_eq!(bpe.vocab, vec![" ", "a", "b", "ab", " ab"]);
        assert_eq!(bpe.merges.iter().map(|m| (m.left, m.right, m.count)).collect::<Vec<_>>(), vec![(1, 2, 3), (0, 3, 2)]);
        assert_eq!(bpe.encode(text), vec![1, 1, 3, 4, 4]);
        // train tokens: 10 characters, then 7 after (a,b), then 5 after (' ',ab)
        assert_eq!(bpe.curve.iter().map(|p| p.train_tokens).collect::<Vec<_>>(), vec![10, 7, 5]);
        // overlapping pair: (a,a) in "aaa" merges once, left first
        let bpe = Bpe::train("aaa aaa", "", alphabet("aaa aaa"), 3, 1);
        assert_eq!(bpe.encode("aaa"), vec![2, 1]);
    }

    #[test]
    fn encoding_reproduces_training_and_decodes_to_the_text() {
        let text = std::fs::read_to_string("data/tinyshakespeare.txt").unwrap_or_else(|_| "to be or not to be, that is the question\n".repeat(50));
        let cut = text.char_indices().nth(text.chars().count() * 9 / 10).map(|(b, _)| b).unwrap();
        let (train, val) = text.split_at(cut);
        let bpe = Bpe::train(train, val, alphabet(&text), 300, 50);
        let again = Bpe::train(train, val, alphabet(&text), 300, 50);
        assert_eq!(bpe.vocab, again.vocab, "training is deterministic");
        let last = bpe.curve.last().unwrap();
        for (part, tokens) in [(train, last.train_tokens), (val, last.val_tokens)] {
            let ids = bpe.encode(part);
            assert_eq!(ids.iter().map(|&i| bpe.vocab[i].as_str()).collect::<String>(), part);
            assert_eq!(ids.len(), tokens, "encoding gives the token count training tracked");
        }
    }
}
