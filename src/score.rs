//! The final benchmark: the validation split scored token by token, the same text for every
//! model and tokenizer, in bits per character.
//!
//! The text scored is the validation split, preceded by the last `context` tokens of the
//! training split, so even the first validation token has a full context. Every validation
//! token is predicted exactly once:
//!   - bigram and MLP: from the `context` tokens right before it;
//!   - transformer: windows of T tokens that move forward T/2 tokens at a time; each window
//!     scores only its last T/2 predictions, so every scored token sees at least T/2 + 1
//!     tokens of context (the first window's, from the train tail, too).
//! Bits per character = Σ −ln p(token) / Σ characters of the scored tokens / ln 2.

use crate::autograd::{per_row_cross_entropy, Op};
use crate::model::logits_eval;
use crate::data::Dataset;
use crate::metrics::{head_patterns, PATTERNS};
use crate::model::Model;
use crate::tensor::F;

pub struct Score {
    /// Σ −ln p over the scored tokens.
    pub nats: F,
    pub tokens: usize,
    pub chars: usize,
    /// Σ over scored tokens of the characters in the context the model saw for it.
    pub context_chars: usize,
    /// Tokens a transformer window moves forward (1 for the bigram and MLP).
    pub stride: usize,
}

impl Score {
    pub fn bpc(&self) -> F {
        self.nats / self.chars as F / std::f64::consts::LN_2 as F
    }
    pub fn nats_per_token(&self) -> F {
        self.nats / self.tokens as F
    }
    pub fn mean_context_chars(&self) -> F {
        self.context_chars as F / self.tokens as F
    }
}

/// Score the first `limit` validation tokens (all of them when `limit` is larger), `batch`
/// windows per forward pass.
pub fn score_val(model: &dyn Model, ds: &Dataset, batch: usize, limit: usize) -> Score {
    let t = model.context();
    let per = model.targets_per_example();
    let seq: Vec<usize> = ds.train[ds.train.len() - t..].iter().chain(&ds.val).copied().collect();
    let end = t + limit.min(ds.val.len());
    // cum[i] = characters in seq[..i]
    let mut cum = vec![0usize; seq.len() + 1];
    for (i, &id) in seq.iter().enumerate() {
        cum[i + 1] = cum[i] + ds.chars[id];
    }
    // Jobs (s, from, to): the model reads seq[s..s + t] and the targets at positions
    // from..to are scored. Target p is predicted from seq[s..p].
    let stride = if per == 1 { 1 } else { (t / 2).max(1) };
    let mut jobs = Vec::new();
    let mut next = t;
    while next < end {
        let to = (next + stride).min(end);
        jobs.push((if per == 1 { next - t } else { to - 1 - t }, next, to));
        next = to;
    }
    let mut score = Score { nats: 0.0, tokens: 0, chars: 0, context_chars: 0, stride };
    for group in jobs.chunks(batch.max(1)) {
        let x: Vec<usize> = group.iter().flat_map(|&(s, _, _)| seq[s..s + t].iter().copied()).collect();
        // the transformer predicts seq[s + 1 + i] at row i; the others only seq[s + t]
        let y: Vec<usize> = group.iter().flat_map(|&(s, _, _)| if per == 1 { &seq[s + t..s + t + 1] } else { &seq[s + 1..s + t + 1] }.iter().copied()).collect();
        let (g, logits) = logits_eval(model, &x, false);
        let losses = per_row_cross_entropy(g.value(logits), &y);
        for (k, &(s, from, to)) in group.iter().enumerate() {
            for pos in from..to {
                let row = if per == 1 { 0 } else { pos - s - 1 };
                score.nats += losses[k * per + row];
                score.tokens += 1;
                score.chars += ds.chars[seq[pos]];
                score.context_chars += cum[pos] - cum[s];
            }
        }
    }
    score
}

/// For each head: the score with only that head switched off (see `Model::set_ablation`),
/// on the first `limit` validation tokens. Returns the baseline on the same tokens first.
pub fn head_ablation(model: &mut dyn Model, ds: &Dataset, batch: usize, limit: usize) -> Option<(Score, Vec<(usize, usize, Score)>)> {
    let (layers, heads) = model.head_grid();
    if layers == 0 {
        return None;
    }
    let base = score_val(model, ds, batch, limit);
    let mut out = Vec::new();
    for l in 0..layers {
        for h in 0..heads {
            model.set_ablation(vec![(l, h)]);
            out.push((l, h, score_val(model, ds, batch, limit)));
        }
    }
    model.set_ablation(Vec::new());
    Some((base, out))
}

pub struct Roles {
    /// Per head (layer, head): the mean attention weight on each of `PATTERNS`, over the
    /// positions where the pattern has somewhere to look.
    pub heads: Vec<(usize, usize, Vec<F>)>,
    /// Per pattern: the weight attention spread evenly would put there (the baseline).
    pub even: Vec<F>,
    pub tokens: usize,
    pub windows: usize,
}

/// What every head attends to, on the first `limit` validation tokens cut into whole windows
/// (a transformer only). See `head_patterns` for the patterns.
pub fn head_roles(model: &dyn Model, ds: &Dataset, batch: usize, limit: usize) -> Option<Roles> {
    let (layers, _) = model.head_grid();
    let t = model.context();
    let windows = limit.min(ds.val.len()) / t;
    if layers == 0 || windows == 0 {
        return None;
    }
    let np = PATTERNS.len();
    // sums[layer][head][pattern] = (weight, rows); even[pattern] = (sum, rows)
    let mut sums: Vec<Vec<Vec<(F, usize)>>> = Vec::new();
    let mut even = vec![(0.0, 0usize); np];
    for first in (0..windows).step_by(batch.max(1)) {
        let nw = batch.max(1).min(windows - first);
        let x = &ds.val[first * t..(first + nw) * t];
        let (g, _) = logits_eval(model, x, true);
        let attn = g.nodes.iter().filter(|nd| matches!(nd.op, Op::CausalSoftmax { .. }));
        for (l, nd) in attn.enumerate() {
            let heads = nd.value.rows / (nw * t);
            let (s, ev) = head_patterns(&nd.value, t, heads, x, &ds.tok.vocab);
            if sums.len() <= l {
                sums.push(vec![vec![(0.0, 0); np]; heads]);
            }
            for (h, per) in s.iter().enumerate() {
                for k in 0..np {
                    sums[l][h][k].0 += per[k].0;
                    sums[l][h][k].1 += per[k].1;
                }
            }
            if l == 0 {
                for k in 0..np {
                    even[k].0 += ev[k];
                    even[k].1 += s[0][k].1;
                }
            }
        }
    }
    let mean = |(s, n): (F, usize)| if n > 0 { s / n as F } else { F::NAN };
    let heads = sums.iter().enumerate().flat_map(|(l, hs)| hs.iter().enumerate().map(move |(h, ks)| (l, h, ks.iter().map(|&v| mean(v)).collect()))).collect();
    Some(Roles { heads, even: even.into_iter().map(mean).collect(), tokens: windows * t, windows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Bigram, Transformer};
    use crate::rng::Rng;

    fn tiny() -> Dataset {
        Dataset::from_text(&"to be or not to be, that is the question.\n".repeat(40), 0.25, 0)
    }

    #[test]
    fn every_val_token_is_scored_once() {
        let ds = tiny();
        let tf = Transformer::new(ds.vocab_size(), 8, 8, 2, 1, &mut Rng::new(1));
        let s = score_val(&tf, &ds, 3, usize::MAX);
        assert_eq!((s.tokens, s.chars, s.stride), (ds.val.len(), ds.val_chars, 4));
        // context is at least T/2 + 1 tokens and at most T
        assert!(s.mean_context_chars() >= 5.0 && s.mean_context_chars() <= 8.0);
        // batching does not change the result
        let one = score_val(&tf, &ds, 1, usize::MAX);
        assert!((one.nats - s.nats).abs() < 1e-9 * s.nats);
        let bg = Bigram::new(ds.vocab_size());
        let b = score_val(&bg, &ds, 16, 10);
        assert_eq!((b.tokens, b.context_chars), (10, 10));
        // an untrained bigram is uniform: ln(vocab) nats per token
        assert!((b.nats_per_token() - (ds.vocab_size() as F).ln()).abs() < 1e-9);
    }

    #[test]
    fn ablating_a_head_changes_the_score_and_is_undone() {
        let ds = tiny();
        let mut tf = Transformer::new(ds.vocab_size(), 8, 8, 2, 2, &mut Rng::new(2));
        let before = score_val(&tf, &ds, 4, 50).nats;
        let (base, heads) = head_ablation(&mut tf, &ds, 4, 50).unwrap();
        assert_eq!(base.nats, before);
        assert_eq!(heads.len(), 4);
        assert!(heads.iter().all(|(_, _, s)| s.nats != before));
        assert_eq!(score_val(&tf, &ds, 4, 50).nats, before);
    }

    #[test]
    fn head_roles_cover_every_head_and_weights_are_shares() {
        let ds = tiny();
        let tf = Transformer::new(ds.vocab_size(), 8, 8, 2, 2, &mut Rng::new(3));
        let r = head_roles(&tf, &ds, 3, 80).unwrap();
        assert_eq!((r.heads.len(), r.windows, r.tokens), (4, 10, 80));
        for (_, _, w) in &r.heads {
            assert_eq!(w.len(), PATTERNS.len());
            // previous token and itself always exist; every weight is a share of one row's attention
            assert!(w[0].is_finite() && w[1].is_finite());
            assert!(w.iter().filter(|v| v.is_finite()).all(|&v| (0.0..=1.0 + 1e-12).contains(&v)));
        }
        // evenly spread attention over p + 1 positions puts 1/(p + 1) on the previous token
        let t = 8.0;
        let want: F = (1..8).map(|p| 1.0 / (p as F + 1.0)).sum::<F>() / (t - 1.0);
        assert!((r.even[0] - want).abs() < 1e-12);
        // a bigram has no heads
        assert!(head_roles(&Bigram::new(ds.vocab_size()), &ds, 3, 80).is_none());
    }
}
