//! Everything recorded over training, stored column by column (one array per
//! quantity) so it stays compact and can be sent in slices to the live viewer.
//!
//! Per step: batch loss and, per parameter, |grad|, |param| (before the update),
//! |update| and |update|/|param|. Per evaluation: fixed-batch train/val loss, a
//! text sample, quantiles of every parameter / gradient / activation, and a
//! fixed-bin histogram of the tanh layer, and the whole character-embedding table.
//! Transformer only: per-head attention entropy and distance, and the residual stream's RMS.

use crate::tensor::{Tensor, F};
use crate::trace::{array, ints, nums, object, packed, string};

/// Quantile levels recorded for each tensor: min, 1%, 5%, 25%, median, 75%, 95%, 99%, max.
/// Quantiles need no bin edges, so the distribution over time can be compared on one axis.
pub const QUANTILES: [F; 9] = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0];

/// tanh histogram: 40 fixed bins over [-1, 1], identical at every step.
pub const TANH_BINS: usize = 40;
/// |h| above this counts as saturated: tanh's slope 1 − h² is below 0.06 there.
pub const SATURATED: F = 0.97;
/// A unit is "dead" if it is saturated on at least this fraction of the evaluation examples.
pub const DEAD_FRACTION: F = 0.99;

/// Linear-interpolated quantiles of `xs` at `QUANTILES`.
pub fn quantiles(xs: &[F]) -> Vec<F> {
    let mut s: Vec<F> = xs.to_vec();
    let n = s.len();
    // Put just the order statistics needed in place (exactly what a full sort would give there):
    // each selection leaves everything above its position at or above it, so the next one only
    // searches what is left.
    let mut need: Vec<usize> = QUANTILES.iter().filter(|_| n > 0).flat_map(|&q| { let lo = (q * (n - 1) as F).floor() as usize; [lo, (lo + 1).min(n - 1)] }).collect();
    need.sort_unstable();
    need.dedup();
    let mut from = 0;
    for i in need {
        if i >= from {
            s[from..].select_nth_unstable_by(i - from, |a, b| a.total_cmp(b));
            from = i + 1;
        }
    }
    QUANTILES
        .iter()
        .map(|&q| {
            if n == 0 {
                return F::NAN;
            }
            let pos = q * (n - 1) as F;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            s[lo] + (s[hi] - s[lo]) * (pos - lo as F)
        })
        .collect()
}

pub struct TanhStats {
    pub counts: Vec<u32>,
    /// Fraction of all activations with |h| > SATURATED.
    pub saturated: F,
    /// Hidden units saturated on ≥ DEAD_FRACTION of the rows (examples).
    pub dead: usize,
    /// Mean of tanh's slope 1 − h² over all activations: the average share of gradient let through.
    pub mean_slope: F,
    /// Per hidden unit, the fraction of examples on which it is saturated; then the median and
    /// 95th percentile of that over units (between "typical unit" and "dead").
    pub unit_sat_p50: F,
    pub unit_sat_p95: F,
}

pub fn tanh_stats(h: &Tensor) -> TanhStats {
    let mut counts = vec![0u32; TANH_BINS];
    let mut sat = 0usize;
    for &v in &h.data {
        let bin = (((v + 1.0) / 2.0 * TANH_BINS as F) as usize).min(TANH_BINS - 1);
        counts[bin] += 1;
        if v.abs() > SATURATED {
            sat += 1;
        }
    }
    let unit_sat: Vec<F> = (0..h.cols).map(|c| (0..h.rows).filter(|&r| h.at(r, c).abs() > SATURATED).count() as F / h.rows.max(1) as F).collect();
    let dead = unit_sat.iter().filter(|&&f| f >= DEAD_FRACTION).count();
    let q = quantiles(&unit_sat);
    let mean_slope = h.data.iter().map(|v| 1.0 - v * v).sum::<F>() / h.len().max(1) as F;
    TanhStats { counts, saturated: sat as F / h.len().max(1) as F, dead, mean_slope, unit_sat_p50: q[4], unit_sat_p95: q[6] }
}

/// Distance buckets for where attention goes: weight on the token itself (0 back), 1, 2, 3,
/// 4-7 and 8 or more tokens back. Each bucket's share is averaged like the entropy.
pub const DIST_BUCKETS: [(usize, usize, &str); 6] = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 7, "4-7"), (8, usize::MAX, "8+")];

fn bucket(back: usize) -> usize {
    DIST_BUCKETS.iter().position(|&(lo, hi, _)| back >= lo && back <= hi).unwrap()
}

/// Per head, the summed attention weight in each of `DIST_BUCKETS` (rows as in `attention_sums`).
pub fn attention_buckets(attn: &Tensor, t: usize, heads: usize) -> Vec<[F; 6]> {
    let mut out = vec![[0.0; 6]; heads];
    for r in 0..attn.rows {
        let (h, p) = ((r / t) % heads, r % t);
        if p == 0 {
            continue;
        }
        for (j, &a) in attn.row(r)[..=p].iter().enumerate() {
            out[h][bucket(p - j)] += a;
        }
    }
    out
}

/// Attention summaries for one causal-softmax output `attn`, (n·H·T) x T, where row
/// (b·H + h)·T + p is position p of example b in head h. Per head, sums over rows of
///   entropy  = −Σ_j a_j·ln a_j / ln(p + 1), 0 when all weight is on one position and
///              1 when it is spread evenly over the p + 1 allowed ones
///   distance = Σ_j a_j·(p − j), how many characters back the head looks on average
/// and the number of rows summed (the same for every head). Position 0 is left out:
/// it can only attend to itself, so it says nothing about the head.
pub fn attention_sums(attn: &Tensor, t: usize, heads: usize) -> (Vec<F>, Vec<F>, usize) {
    assert!(t > 0 && heads > 0 && attn.cols == t && attn.rows % (heads * t) == 0, "attention_sums: {}x{} is not n·{heads}·{t} x {t}", attn.rows, attn.cols);
    let (mut ent, mut dist) = (vec![0.0; heads], vec![0.0; heads]);
    for r in 0..attn.rows {
        let (h, p) = ((r / t) % heads, r % t);
        if p == 0 {
            continue;
        }
        let row = &attn.row(r)[..=p];
        ent[h] -= row.iter().filter(|&&a| a > 0.0).map(|&a| a * a.ln()).sum::<F>() / ((p + 1) as F).ln();
        dist[h] += row.iter().enumerate().map(|(j, &a)| a * (p - j) as F).sum::<F>();
    }
    (ent, dist, attn.rows / (heads * t) * (t - 1))
}

/// What kind of text a predicted token is, to see which kinds the model learns first.
/// Decided by the token and the one before it, so it works for any tokenizer:
///   space or newline: the token is only whitespace;
///   word start:       its first non-space character is a letter that starts a word (the
///                     token starts with a space, or the previous token does not end in a
///                     letter or an apostrophe; "o'er" counts as one word);
///   inside a word:    it starts with a letter right after a letter or an apostrophe;
///   punctuation:      anything else (punctuation and digits).
/// With BPE one token can hold a space and a whole word (" the"): it counts as a word start.
pub const KINDS: [&str; 4] = ["space or newline", "punctuation", "word start", "inside a word"];

pub fn token_kind(tok: &str, prev: &str) -> usize {
    if tok.chars().all(char::is_whitespace) {
        return 0;
    }
    let first = tok.chars().next().unwrap();
    let letter = tok.chars().find(|c| !c.is_whitespace()).is_some_and(char::is_alphabetic);
    if !letter {
        return 1;
    }
    if first.is_whitespace() {
        return 2;
    }
    match prev.chars().last() {
        Some(p) if p.is_alphabetic() || p == '\'' => 3,
        _ => 2,
    }
}

/// Patterns a head's attention can follow, measured by `head_patterns`.
/// prev token / itself: weight on position p − 1 / p;
/// last space: on the most recent earlier token holding a space or newline (a word boundary);
/// last newline: on the most recent earlier token holding a newline (the start of the line);
/// same token: on earlier copies of the current token;
/// induction: on tokens that followed an earlier copy of the current token (what came next
/// last time), the pattern that lets a model continue a repeated phrase.
pub const PATTERNS: [&str; 6] = ["previous token", "itself", "last space", "last newline", "same token", "induction"];

/// Per head (rows as in `attention_sums`) and pattern: the summed attention weight on that
/// pattern's positions and the number of rows where the pattern had any position. Also, per
/// pattern, the sum of what attention spread evenly would give (positions / (p + 1)), the
/// baseline a head must beat to be following the pattern. `x` holds each example's tokens.
pub fn head_patterns(attn: &Tensor, t: usize, heads: usize, x: &[usize], vocab: &[String]) -> (Vec<[(F, usize); 6]>, [F; 6]) {
    let mut out = vec![[(0.0, 0usize); 6]; heads];
    let mut even = [0.0; 6];
    let has = |id: usize, c: char| vocab[id].contains(c);
    for r in 0..attn.rows {
        let (b, h, p) = (r / (t * heads), (r / t) % heads, r % t);
        if p == 0 {
            continue;
        }
        let xs = &x[b * t..(b + 1) * t];
        let a = attn.row(r);
        let last = |c: char| (0..p).rev().find(|&j| has(xs[j], c));
        let mut sets: [Vec<usize>; 6] = [vec![p - 1], vec![p], Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        if let Some(j) = (0..p).rev().find(|&j| has(xs[j], ' ') || has(xs[j], '\n')) {
            sets[2].push(j);
        }
        if let Some(j) = last('\n') {
            sets[3].push(j);
        }
        sets[4] = (0..p).filter(|&j| xs[j] == xs[p]).collect();
        sets[5] = (1..p).filter(|&j| xs[j - 1] == xs[p]).collect();
        for (k, s) in sets.iter().enumerate() {
            if s.is_empty() {
                continue;
            }
            out[h][k].0 += s.iter().map(|&j| a[j]).sum::<F>();
            out[h][k].1 += 1;
            if h == 0 {
                even[k] += s.len() as F / (p + 1) as F;
            }
        }
    }
    (out, even)
}

/// Root mean square of all the numbers: the typical size of one entry.
pub fn rms(xs: &[F]) -> F {
    (xs.iter().map(|v| v * v).sum::<F>() / xs.len().max(1) as F).sqrt()
}

/// Several named numbers recorded at every evaluation (one per head, one per layer, ...).
#[derive(Clone, Default)]
pub struct EvalSeries {
    /// Fixed at the first evaluation.
    pub names: Vec<String>,
    /// values[eval][i] belongs to names[i].
    pub values: Vec<Vec<F>>,
}

impl EvalSeries {
    pub fn push(&mut self, names: Vec<String>, values: Vec<F>) {
        assert_eq!(names.len(), values.len());
        if self.values.is_empty() {
            self.names = names;
        } else {
            assert_eq!(self.names, names, "series must be recorded with the same names every time");
        }
        self.values.push(values);
    }

    fn slice(&self, from_eval: usize) -> EvalSeries {
        EvalSeries { names: self.names.clone(), values: self.values[from_eval.min(self.values.len())..].to_vec() }
    }

    fn json(&self) -> String {
        object(&[
            ("names", array(&self.names.iter().map(|n| string(n)).collect::<Vec<_>>())),
            ("values", array(&self.values.iter().map(|v| nums(v)).collect::<Vec<_>>())),
        ])
    }
}

#[derive(Clone, Default)]
pub struct ParamSeries {
    pub name: String,
    pub grad_norm: Vec<F>,
    pub param_norm: Vec<F>,
    pub update_norm: Vec<F>,
    /// |update| / |param|; NaN (written as null) when |param| is 0, e.g. a zero-initialized bias.
    pub ratio: Vec<F>,
}

#[derive(Clone, Default)]
pub struct Metrics {
    pub loss: Vec<F>,
    /// The learning rate each step used.
    pub lr: Vec<F>,
    pub params: Vec<ParamSeries>,
    pub eval_step: Vec<usize>,
    pub eval_train: Vec<F>,
    pub eval_val: Vec<F>,
    /// The same evaluations in bits per character: Σ token losses / Σ characters of the
    /// target tokens / ln 2. Comparable across tokenizers.
    pub eval_train_bpc: Vec<F>,
    pub eval_val_bpc: Vec<F>,
    /// Training characters consumed before this evaluation (Σ characters of every training target).
    pub eval_chars_seen: Vec<usize>,
    /// Seconds spent training (batches, forward, backward, update) before each evaluation:
    /// no evaluation, snapshot or logging time, so runs on the same machine compare fairly.
    pub eval_train_secs: Vec<F>,
    pub sample: Vec<String>,
    /// Tensor names for `quant`, fixed at the first evaluation.
    pub quant_names: Vec<String>,
    /// quant[tensor][eval] = the QUANTILES of that tensor at that evaluation.
    pub quant: Vec<Vec<Vec<F>>>,
    pub tanh_counts: Vec<Vec<u32>>,
    pub tanh_saturated: Vec<F>,
    pub tanh_dead: Vec<usize>,
    pub tanh_slope: Vec<F>,
    pub tanh_unit_p50: Vec<F>,
    pub tanh_unit_p95: Vec<F>,
    /// embedding[eval] = the embedding table C (vocab × dim, row-major) before that step's update.
    /// Empty for models without one.
    pub embedding: Vec<Vec<F>>,
    /// Per evaluation, per layer and head (empty for models without attention): see `attention_sums`.
    pub attn_entropy: EvalSeries,
    pub attn_distance: EvalSeries,
    /// Per head and distance bucket ("<head>|<bucket>"): share of the attention weight.
    pub attn_buckets: EvalSeries,
    /// Per evaluation, the RMS of each watched point on the residual stream.
    pub resid_rms: EvalSeries,
    /// Per evaluation, the validation bits per character of each of `KINDS`, and the share
    /// of the validation characters that kind covers (the same at every evaluation).
    pub kind_bpc: EvalSeries,
    pub kind_chars: EvalSeries,
}

impl Metrics {
    pub fn new(param_names: &[String]) -> Self {
        Metrics { params: param_names.iter().map(|n| ParamSeries { name: n.clone(), ..Default::default() }).collect(), ..Default::default() }
    }

    pub fn steps(&self) -> usize {
        self.loss.len()
    }

    pub fn evals(&self) -> usize {
        self.eval_step.len()
    }

    /// Record one tensor's quantiles for the current evaluation (call in the same order every time).
    pub fn push_quantiles(&mut self, eval_index: usize, slot: usize, name: &str, q: Vec<F>) {
        if eval_index == 0 {
            assert_eq!(slot, self.quant_names.len());
            self.quant_names.push(name.to_string());
            self.quant.push(Vec::new());
        }
        assert_eq!(self.quant_names[slot], name, "quantile tensors must be recorded in the same order");
        self.quant[slot].push(q);
    }

    /// Every column must have one entry per step (or per evaluation).
    pub fn check_lengths(&self) -> Result<(), String> {
        let (s, e) = (self.steps(), self.evals());
        if self.lr.len() != s {
            return Err(format!("lr has {} entries for {s} steps", self.lr.len()));
        }
        for (n, series) in [("attention entropy", &self.attn_entropy), ("attention distance", &self.attn_distance), ("attention buckets", &self.attn_buckets), ("residual RMS", &self.resid_rms), ("kind bpc", &self.kind_bpc), ("kind chars", &self.kind_chars)] {
            if !series.values.is_empty() && series.values.len() != e {
                return Err(format!("{n} has {} entries for {e} evaluations", series.values.len()));
            }
        }
        for p in &self.params {
            for (n, len) in [("grad_norm", p.grad_norm.len()), ("param_norm", p.param_norm.len()), ("update_norm", p.update_norm.len()), ("ratio", p.ratio.len())] {
                if len != s {
                    return Err(format!("{}.{n} has {len} entries for {s} steps", p.name));
                }
            }
        }
        for (n, len) in [
            ("eval_train", self.eval_train.len()),
            ("eval_val", self.eval_val.len()),
            ("eval_train_bpc", self.eval_train_bpc.len()),
            ("eval_val_bpc", self.eval_val_bpc.len()),
            ("eval_chars_seen", self.eval_chars_seen.len()),
            ("eval_train_secs", self.eval_train_secs.len()),
            ("sample", self.sample.len()),
        ] {
            if len != e {
                return Err(format!("{n} has {len} entries for {e} evaluations"));
            }
        }
        for (name, q) in self.quant_names.iter().zip(&self.quant) {
            if q.len() != e {
                return Err(format!("quantiles of {name} have {} entries for {e} evaluations", q.len()));
            }
        }
        let t = self.tanh_counts.len();
        if t != 0 && [t, self.tanh_saturated.len(), self.tanh_dead.len(), self.tanh_slope.len(), self.tanh_unit_p50.len(), self.tanh_unit_p95.len()].iter().any(|&l| l != e) {
            return Err(format!("tanh stats have {t} entries for {e} evaluations"));
        }
        if !self.embedding.is_empty() && self.embedding.len() != e {
            return Err(format!("embedding has {} entries for {e} evaluations", self.embedding.len()));
        }
        Ok(())
    }

    /// Copy of steps `from_step..` and evaluations `from_eval..` (for live deltas).
    pub fn slice(&self, from_step: usize, from_eval: usize) -> Metrics {
        let (fs, fe) = (from_step.min(self.steps()), from_eval.min(self.evals()));
        let tanh_from = fe.min(self.tanh_counts.len());
        Metrics {
            loss: self.loss[fs..].to_vec(),
            lr: self.lr[fs.min(self.lr.len())..].to_vec(),
            params: self
                .params
                .iter()
                .map(|p| ParamSeries {
                    name: p.name.clone(),
                    grad_norm: p.grad_norm[fs..].to_vec(),
                    param_norm: p.param_norm[fs..].to_vec(),
                    update_norm: p.update_norm[fs..].to_vec(),
                    ratio: p.ratio[fs..].to_vec(),
                })
                .collect(),
            eval_step: self.eval_step[fe..].to_vec(),
            eval_train: self.eval_train[fe..].to_vec(),
            eval_val: self.eval_val[fe..].to_vec(),
            eval_train_bpc: self.eval_train_bpc[fe..].to_vec(),
            eval_val_bpc: self.eval_val_bpc[fe..].to_vec(),
            eval_chars_seen: self.eval_chars_seen[fe..].to_vec(),
            eval_train_secs: self.eval_train_secs[fe..].to_vec(),
            sample: self.sample[fe..].to_vec(),
            quant_names: self.quant_names.clone(),
            quant: self.quant.iter().map(|q| q[fe.min(q.len())..].to_vec()).collect(),
            tanh_counts: self.tanh_counts[tanh_from..].to_vec(),
            tanh_saturated: self.tanh_saturated[tanh_from..].to_vec(),
            tanh_dead: self.tanh_dead[tanh_from..].to_vec(),
            tanh_slope: self.tanh_slope[tanh_from..].to_vec(),
            tanh_unit_p50: self.tanh_unit_p50[tanh_from..].to_vec(),
            tanh_unit_p95: self.tanh_unit_p95[tanh_from..].to_vec(),
            embedding: self.embedding[fe.min(self.embedding.len())..].to_vec(),
            attn_entropy: self.attn_entropy.slice(fe),
            attn_distance: self.attn_distance.slice(fe),
            attn_buckets: self.attn_buckets.slice(fe),
            resid_rms: self.resid_rms.slice(fe),
            kind_bpc: self.kind_bpc.slice(fe),
            kind_chars: self.kind_chars.slice(fe),
        }
    }

    /// JSON for this (possibly sliced) set; `from_step`/`from_eval` say where it starts.
    pub fn json(&self, from_step: usize, from_eval: usize) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                object(&[
                    ("name", string(&p.name)),
                    // one number per step and tensor: packed, or a long run's file is mostly digits
                    ("grad_norm", packed(&p.grad_norm)),
                    ("param_norm", packed(&p.param_norm)),
                    ("update_norm", packed(&p.update_norm)),
                    ("ratio", packed(&p.ratio)),
                ])
            })
            .collect();
        let samples: Vec<String> = self.sample.iter().map(|s| string(s)).collect();
        let quant: Vec<String> = self.quant.iter().map(|per_eval| array(&per_eval.iter().map(|q| nums(q)).collect::<Vec<_>>())).collect();
        let edges: Vec<F> = (0..=TANH_BINS).map(|i| -1.0 + 2.0 * i as F / TANH_BINS as F).collect();
        let counts: Vec<String> = self.tanh_counts.iter().map(|c| format!("[{}]", c.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))).collect();
        object(&[
            ("from_step", from_step.to_string()),
            ("from_eval", from_eval.to_string()),
            ("loss", nums(&self.loss)),
            ("lr", nums(&self.lr)),
            ("params", array(&params)),
            (
                "evals",
                object(&[
                    ("step", ints(&self.eval_step)),
                    ("train", nums(&self.eval_train)),
                    ("val", nums(&self.eval_val)),
                    ("train_bpc", nums(&self.eval_train_bpc)),
                    ("val_bpc", nums(&self.eval_val_bpc)),
                    ("chars_seen", ints(&self.eval_chars_seen)),
                    ("train_secs", nums(&self.eval_train_secs)),
                    ("sample", array(&samples)),
                ]),
            ),
            (
                "quantiles",
                object(&[
                    ("levels", nums(&QUANTILES)),
                    ("names", array(&self.quant_names.iter().map(|n| string(n)).collect::<Vec<_>>())),
                    ("values", array(&quant)),
                ]),
            ),
            (
                "tanh",
                object(&[
                    ("edges", nums(&edges)),
                    ("counts", array(&counts)),
                    ("saturated", nums(&self.tanh_saturated)),
                    ("dead", ints(&self.tanh_dead)),
                    ("mean_slope", nums(&self.tanh_slope)),
                    ("unit_sat_p50", nums(&self.tanh_unit_p50)),
                    ("unit_sat_p95", nums(&self.tanh_unit_p95)),
                ]),
            ),
            ("embedding", array(&self.embedding.iter().map(|t| packed(t)).collect::<Vec<_>>())),
            ("attention", object(&[("entropy", self.attn_entropy.json()), ("distance", self.attn_distance.json()), ("buckets", self.attn_buckets.json())])),
            ("residual_rms", self.resid_rms.json()),
            ("kinds", object(&[("bits_per_char", self.kind_bpc.json()), ("char_share", self.kind_chars.json())])),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_kinds() {
        assert_eq!(token_kind(" ", "a"), 0);
        assert_eq!(token_kind("\n", "."), 0);
        assert_eq!(token_kind(",", "a"), 1);
        assert_eq!(token_kind("12", " "), 1);
        assert_eq!(token_kind("t", " "), 2);
        assert_eq!(token_kind(" the", "a"), 2);
        assert_eq!(token_kind("the", "\n"), 2);
        assert_eq!(token_kind("h", "t"), 3);
        assert_eq!(token_kind("er", "o'"), 3);
        assert_eq!(token_kind("ing", "think"), 3);
    }

    #[test]
    fn head_patterns_find_each_pattern() {
        // vocab: 0 "a", 1 "b", 2 " ", 3 "\n"; one window "a b\na b" (t = 7), one head that
        // always attends to the previous position (the first row only to itself).
        let vocab: Vec<String> = ["a", "b", " ", "\n"].iter().map(|s| s.to_string()).collect();
        let x = [0, 2, 1, 3, 0, 2, 1];
        let t = 7;
        let mut attn = Tensor::from_vec(t, t, vec![0.0; t * t]);
        for p in 0..t {
            attn.data[p * t + p.saturating_sub(1)] = 1.0;
        }
        let (s, even) = head_patterns(&attn, t, 1, &x, &vocab);
        let h = &s[0];
        assert_eq!(h[0], (6.0, 6)); // previous token, every row but the first
        assert_eq!(h[1], (0.0, 6)); // never itself
        // last space or newline: rows 2..6 have one; it is p − 1 at rows 2, 4 (the "\n"), 6
        assert_eq!(h[2], (3.0, 5));
        // last newline exists from row 4 on; p − 1 = 3 only at row 4
        assert_eq!(h[3], (1.0, 3));
        // same token: rows 4 ("a" at 0), 5 (" " at 1), 6 ("b" at 2)
        assert_eq!(h[4], (0.0, 3));
        // induction: row 4 "a" -> position 1 (after the "a" at 0); row 5 " " -> 2; row 6 "b" -> 3
        assert_eq!(h[5], (0.0, 3));
        assert!((even[0] - (1..7).map(|p| 1.0 / (p as F + 1.0)).sum::<F>()).abs() < 1e-12);
    }

    #[test]
    fn quantiles_match_a_full_sort() {
        let mut rng = crate::rng::Rng::new(7);
        for n in [1usize, 2, 3, 10, 101, 1000, 4097] {
            // Rounded so there are many ties.
            let xs: Vec<F> = (0..n).map(|_| (rng.normal() * 4.0).round() as F).collect();
            let mut s = xs.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            let want: Vec<F> = QUANTILES
                .iter()
                .map(|&q| {
                    let pos = q * (n - 1) as F;
                    let lo = pos.floor() as usize;
                    let hi = (lo + 1).min(n - 1);
                    s[lo] + (s[hi] - s[lo]) * (pos - lo as F)
                })
                .collect();
            assert_eq!(quantiles(&xs), want, "n = {n}");
        }
        assert!(quantiles(&[]).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn quantiles_interpolate() {
        let q = quantiles(&[4.0, 0.0, 2.0, 1.0, 3.0]);
        assert_eq!(q[0], 0.0);
        assert_eq!(q[4], 2.0);
        assert_eq!(q[8], 4.0);
        assert!((q[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tanh_stats_count_saturation_and_dead_units() {
        // 2 examples x 3 units. Unit 0 saturated on both rows (dead), unit 1 on one, unit 2 never.
        let h = Tensor::from_vec(2, 3, vec![0.99, -0.98, 0.0, -0.995, 0.5, 0.1]);
        let s = tanh_stats(&h);
        assert_eq!(s.dead, 1);
        assert!((s.saturated - 3.0 / 6.0).abs() < 1e-12);
        assert_eq!(s.counts.iter().sum::<u32>(), 6);
        assert_eq!(s.counts[TANH_BINS - 1], 1); // 0.99 lands in the top bin
        // Per-unit saturated fractions are [1, 0.5, 0]: median 0.5, 95th percentile 0.95.
        assert!((s.unit_sat_p50 - 0.5).abs() < 1e-12 && (s.unit_sat_p95 - 0.95).abs() < 1e-12);
        let slope: F = [0.99f64, -0.98, 0.0, -0.995, 0.5, 0.1].iter().map(|v| 1.0 - v * v).sum::<F>() / 6.0;
        assert!((s.mean_slope - slope).abs() < 1e-12);
    }

    #[test]
    fn slices_line_up_with_the_whole() {
        let mut m = Metrics::new(&["w".to_string()]);
        for s in 0..5 {
            m.loss.push(s as F);
            m.lr.push(0.1);
            let p = &mut m.params[0];
            p.grad_norm.push(1.0);
            p.param_norm.push(if s == 0 { 0.0 } else { 2.0 });
            p.update_norm.push(0.5);
            p.ratio.push(if s == 0 { F::NAN } else { 0.25 });
        }
        for (i, s) in [0usize, 4].iter().enumerate() {
            m.eval_step.push(*s);
            m.eval_train.push(1.0);
            m.eval_val.push(1.0);
            m.eval_train_bpc.push(1.0);
            m.eval_val_bpc.push(1.0);
            m.eval_chars_seen.push(*s * 10);
            m.eval_train_secs.push(*s as F);
            m.sample.push("x".into());
            m.push_quantiles(i, 0, "w", quantiles(&[1.0, 2.0]));
        }
        m.check_lengths().unwrap();
        m.embedding = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        m.check_lengths().unwrap();
        let d = m.slice(3, 1);
        assert_eq!(d.loss, vec![3.0, 4.0]);
        assert_eq!(d.eval_step, vec![4]);
        assert_eq!(d.embedding, vec![vec![3.0, 4.0]]);
        assert!(d.json(3, 1).contains("\"embedding\":[[3,4]]"));
        m.embedding.pop();
        assert!(m.check_lengths().is_err(), "a missing embedding frame must be caught");
        m.embedding.clear();
        assert_eq!(d.quant[0].len(), 1);
        assert!(m.json(0, 0).contains("\"ratio\":[null,0.25"));
        m.resid_rms.push(vec!["input".into()], vec![1.0]);
        assert!(m.check_lengths().is_err(), "one RMS entry for two evaluations must be caught");
        m.resid_rms.push(vec!["input".into()], vec![2.0]);
        m.check_lengths().unwrap();
        assert_eq!(m.slice(3, 1).resid_rms.values, vec![vec![2.0]]);
    }

    #[test]
    fn attention_sums_measure_spread_and_distance() {
        // 1 example, 2 heads, T = 3. Head 0 always looks at position 0; head 1 spreads evenly.
        #[rustfmt::skip]
        let a = Tensor::from_vec(6, 3, vec![
            1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,          // head 0, positions 0..3
            1.0, 0.0, 0.0,   0.5, 0.5, 0.0,   1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, // head 1
        ]);
        let (ent, dist, count) = attention_sums(&a, 3, 2);
        assert_eq!(count, 2); // positions 1 and 2 of one example
        assert!(ent[0].abs() < 1e-12 && (ent[1] - 2.0).abs() < 1e-12, "{ent:?}");
        // head 0: 1 + 2 characters back; head 1: 0.5 + (2 + 1 + 0)/3 = 1.5
        assert!((dist[0] - 3.0).abs() < 1e-12 && (dist[1] - 1.5).abs() < 1e-12, "{dist:?}");
        assert!((rms(&[3.0, -4.0]) - (12.5 as F).sqrt()).abs() < 1e-12);
    }
}
