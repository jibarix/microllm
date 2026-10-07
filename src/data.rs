//! The dataset: text -> tokens -> training batches.
//!
//! Tokens come from a BPE tokenizer (see bpe.rs). With no merges every token is one
//! character, which is the character-level dataset the bigram, MLP and first transformer
//! used, with the same ids. Each token knows how many characters it covers, so losses can be
//! compared per character across tokenizers.

use crate::bpe::Bpe;
use crate::rng::Rng;

pub struct Dataset {
    pub tok: Bpe,
    /// Characters each token covers (1 for every base character).
    pub chars: Vec<usize>,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    /// The two splits as text (the val text is what the final benchmark scores).
    pub train_chars: usize,
    pub val_chars: usize,
}

#[derive(Clone, Copy)]
pub enum Split {
    Train,
    Val,
}

impl Dataset {
    /// The alphabet is every distinct character of the whole text (so validation characters
    /// are always encodable); the last `val_fraction` of the characters is held out. With
    /// `vocab` above the alphabet size, a BPE tokenizer is trained on the train split only.
    pub fn from_text(text: &str, val_fraction: f64, vocab: usize) -> Self {
        let mut base: Vec<char> = text.chars().collect();
        base.sort_unstable();
        base.dedup();
        let n = text.chars().count();
        let n_train = ((1.0 - val_fraction) * n as f64) as usize;
        let cut = text.char_indices().nth(n_train).map(|(b, _)| b).unwrap_or(text.len());
        let (train, val) = text.split_at(cut);
        let tok = Bpe::train(train, val, base, vocab, 16);
        let chars = tok.vocab.iter().map(|t| t.chars().count()).collect();
        Dataset { train: tok.encode(train), val: tok.encode(val), tok, chars, train_chars: n_train, val_chars: n - n_train }
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.vocab.len()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.tok.vocab[i].as_str()).collect()
    }

    pub fn encode_char(&self, c: char) -> Option<usize> {
        self.tok.base.binary_search(&c).ok()
    }

    /// Characters covered by these tokens.
    pub fn chars_of(&self, ids: &[usize]) -> usize {
        ids.iter().map(|&i| self.chars[i]).sum()
    }

    /// `n` random examples. Each is `ctx` consecutive tokens (x, flattened) and the
    /// token right after them (y). Bigram uses ctx = 1; the MLP uses a longer context.
    pub fn contexts(&self, split: Split, n: usize, ctx: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let data = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };
        assert!(data.len() > ctx, "split has {} tokens; it needs more than the context length {ctx}", data.len());
        let mut x = Vec::with_capacity(n * ctx);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            // Valid starts are 0..len-ctx (the target data[start+ctx] must exist), so the exclusive bound is len-ctx.
            let start = rng.below(data.len() - ctx);
            x.extend_from_slice(&data[start..start + ctx]);
            y.push(data[start + ctx]);
        }
        (x, y)
    }

    /// `n` random windows of `t` consecutive tokens (x, flattened) and, for each position,
    /// the token right after it (y): y is x shifted left by one. The transformer predicts
    /// all t of them, so a batch holds n·t predictions.
    pub fn windows(&self, split: Split, n: usize, t: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let data = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };
        assert!(data.len() > t, "split has {} tokens; it needs more than the window length {t}", data.len());
        let mut x = Vec::with_capacity(n * t);
        let mut y = Vec::with_capacity(n * t);
        for _ in 0..n {
            // The last target is data[start + t], so start < len − t, as in `contexts`.
            let start = rng.below(data.len() - t);
            x.extend_from_slice(&data[start..start + t]);
            y.extend_from_slice(&data[start + 1..start + t + 1]);
        }
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_targets_are_inputs_shifted_by_one() {
        let ds = Dataset::from_text("abcdefghijklmnopqrstuvwxyz", 0.1, 0);
        let (x, y) = ds.windows(Split::Train, 50, 5, &mut Rng::new(3));
        assert_eq!((x.len(), y.len()), (250, 250));
        for w in 0..50 {
            for i in 0..5 {
                // in this text every character's successor is the next letter
                assert_eq!(y[w * 5 + i], x[w * 5 + i] + 1);
            }
            assert!(x[w * 5] + 5 < ds.train.len(), "the last target must still be in the train split");
        }
    }
}
