//! The training loop: batch → forward → backward → optimizer step, with
//! everything recorded for the viewer and the log.
//!
//! Output, all under runs/<run>/:
//!   meta.js         config, vocab, parameters, gradcheck, provenance (written at start and end)
//!   metrics.js      every step's numbers, column by column (written at the end)
//!   snaps/<step>.js one file per recorded step, loaded by the viewer on demand
//!   train.log       exactly what the terminal printed, plus a provenance header

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::autograd::{per_row_cross_entropy, Graph, Op};
use crate::data::{Dataset, Split};
use crate::gradcheck::{self, GradReport};
use crate::bpe::PRETOK;
use crate::metrics::{attention_buckets, attention_sums, quantiles, rms, tanh_stats, token_kind, Metrics, TanhStats, DEAD_FRACTION, DIST_BUCKETS, KINDS, PATTERNS, SATURATED};
use crate::model::{forward, forward_eval, sample, Bigram, Forward, Mlp, Model, Transformer};
use crate::{cuda, gpu};
use crate::optim::{Kind, Optimizer};
use crate::params::Params;
use crate::rng::Rng;
use crate::serve::{self, LiveState, Shared};
use crate::sha256::{hex, sha256};
use crate::tensor::{softmax_rows, Tensor, F};
use crate::score::{head_ablation, head_roles, score_val, Score};
use crate::trace::{array, graph_nodes, ints, num, num64, nums, object, packed, packed64, string, write_atomic, Detail};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Bigram,
    Mlp,
    Transformer,
}

#[derive(Clone)]
pub struct Config {
    pub data: String,
    /// The last this fraction of the text is the validation split.
    pub val_frac: F,
    pub run: String,
    pub model: ModelKind,
    pub steps: usize,
    /// Examples per step. A bigram or MLP example is one prediction (context → next
    /// character); a transformer example is a window of `context` characters and predicts
    /// the next character at every position.
    pub examples: usize,
    pub context: usize,
    /// Embedding size; for the transformer, the width d of the residual stream.
    pub emb: usize,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    /// Transformer changes for the speed experiments (see `Variant`).
    pub variant: crate::model::Variant,
    pub optimizer: Kind,
    /// Peak learning rate (see `lr_at`).
    pub lr: F,
    pub lr_min: F,
    pub warmup: usize,
    /// Above 0: the warm-up–stable–decay schedule with this fraction of the steps in the
    /// decay (`lr_wsd`); 0: the cosine (`lr_at`).
    pub wsd: F,
    /// Muon's peak learning rate (with --optimizer muon).
    pub muon_lr: F,
    pub weight_decay: F,
    pub seed: u64,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub sample_len: usize,
    pub snap_every: usize,
    /// Steps 0 .. snap_first are always recorded (the start, where everything changes fastest).
    pub snap_first: usize,
    /// None = choose from the size estimate.
    pub detail: Option<Detail>,
    pub serve: Option<u16>,
    pub throttle_ms: u64,
    /// Threads for the large matrix products; results are identical for any count.
    pub threads: usize,
    /// Forward and backward on the GPU (f32); the optimizer and the f64 master parameters
    /// stay on the CPU, as do the gradient checks.
    pub gpu: bool,
    /// Tokenizer size: 0 (or at most the alphabet) = one token per character; above it, BPE
    /// trained on the train split up to this many tokens.
    pub vocab: usize,
    /// When above 0, overrides `steps`: train until about this many characters of training
    /// targets have been seen (so tokenizers with more characters per token get fewer steps).
    pub chars: usize,
}

/// With auto detail, a run whose full snapshots would exceed this switches to summary.
pub(crate) const AUTO_FULL_LIMIT_BYTES: f64 = 300e6;
/// Size of one number in a snapshot: an f32 in base64 (see `trace::packed`).
pub(crate) const BYTES_PER_NUMBER: f64 = 16.0 / 3.0;

pub(crate) fn is_snapshot_step(step: usize, cfg: &Config) -> bool {
    step < cfg.snap_first || step % cfg.snap_every == 0 || step + 1 == cfg.steps
}

fn is_eval_step(step: usize, cfg: &Config) -> bool {
    step % cfg.eval_every == 0 || step + 1 == cfg.steps
}

/// The learning rate at `step` of `steps`: a linear warm-up over the first `warmup` steps
/// (lr/warmup, 2·lr/warmup, … up to lr), then a cosine from lr at step `warmup` down to
/// lr_min at the last step. The last step always uses lr_min, so `warmup` must be below
/// `steps` (checked when parsing). With warmup 0 and lr_min = lr (the bigram and MLP
/// defaults) it is constant.
pub fn lr_at(lr: F, lr_min: F, warmup: usize, steps: usize, step: usize) -> F {
    assert!(warmup < steps, "warm-up ({warmup} steps) must end before the last step ({steps} steps)");
    let last = steps - 1;
    if step < warmup {
        return lr * (step + 1) as F / warmup as F;
    }
    if step >= last {
        return lr_min;
    }
    // here warmup <= step < last, so the span is at least 1
    let progress = (step - warmup) as F / (last - warmup) as F;
    lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Warm-up, stable, decay: the linear warm-up of `lr_at`, then lr until the last `frac` of
/// the steps, then a straight line down to lr_min at the last step.
pub fn lr_wsd(lr: F, lr_min: F, warmup: usize, steps: usize, frac: F, step: usize) -> F {
    assert!(warmup < steps && frac > 0.0 && frac <= 1.0, "wsd: warm-up {warmup} of {steps} steps, decay fraction {frac}");
    let last = steps - 1;
    if step < warmup {
        return lr * (step + 1) as F / warmup as F;
    }
    let start = (last as F * (1.0 - frac)).round().max(warmup as F) as usize;
    if step <= start || start == last {
        return if step >= last { lr_min } else { lr };
    }
    lr + (lr_min - lr) * (step - start) as F / (last - start) as F
}

struct Log {
    file: fs::File,
    path: PathBuf,
    /// The first failed write to train.log; reported when training ends so the log is never silently incomplete.
    error: Option<String>,
}

impl Log {
    /// Print a line and append it to train.log.
    fn line(&mut self, s: impl AsRef<str>) {
        let s = s.as_ref();
        println!("{s}");
        self.file_only(s);
    }

    /// Append to train.log without printing (for errors main() already prints).
    fn file_only(&mut self, s: &str) {
        if let Err(e) = writeln!(self.file, "{s}").and_then(|_| self.file.flush()) {
            self.error.get_or_insert_with(|| format!("writing {}: {e}", self.path.display()));
        }
    }

    fn result(&self) -> Result<(), String> {
        self.error.clone().map_or(Ok(()), Err)
    }
}

/// The loss, or the first gradient, update or parameter entry, that is not a finite number.
fn first_non_finite(loss: F, names: &[String], grads: &[Tensor], updates: &[Tensor], params: &Params) -> Option<String> {
    if !loss.is_finite() {
        return Some(format!("batch loss is {loss}"));
    }
    let bad = |t: &Tensor| t.data.iter().position(|v| !v.is_finite()).map(|i| (i / t.cols.max(1), i % t.cols.max(1), t.data[i]));
    for (what, list) in [("gradient of", grads), ("update to", updates)] {
        for (name, t) in names.iter().zip(list) {
            if let Some((r, c, v)) = bad(t) {
                return Some(format!("{what} {name}[{r},{c}] is {v}"));
            }
        }
    }
    for p in &params.list {
        if let Some((r, c, v)) = bad(&p.value) {
            return Some(format!("{}[{r},{c}] is {v} after the update", p.name));
        }
    }
    None
}

pub fn gradcheck_lines(reports: &[GradReport]) -> Vec<String> {
    let mut out = vec![format!("{:<52} {:>8} {:>12} {:>12}  result", "check", "elements", "max abs err", "max rel err")];
    for r in reports {
        let verdict = if r.passed() { "PASS" } else { "FAIL" };
        out.push(format!("{:<52} {:>8} {:>12.3e} {:>12.3e}  {verdict}", r.name, r.checked, r.max_abs_err, r.max_rel_err));
    }
    out
}

struct Eval {
    train: F,
    val: F,
    /// The same losses in bits per character of the predicted tokens.
    train_bpc: F,
    val_bpc: F,
    text: String,
    quant: Vec<(String, Vec<F>)>,
    tanh: Option<TanhStats>,
    /// Per layer and head (empty without attention): names, mean entropy, mean distance.
    attn: (Vec<String>, Vec<F>, Vec<F>),
    /// Per watched residual-stream point: names, RMS.
    resid: (Vec<String>, Vec<F>),
    /// Per head and distance bucket ("<head>|<bucket>"): the share of attention weight.
    buckets: (Vec<String>, Vec<F>),
    /// Validation bits per character of each of `KINDS`, and each kind's share of the characters.
    kind_bpc: Vec<F>,
    kind_share: Vec<F>,
}

type Examples = (Vec<usize>, Vec<usize>);

/// `n` random examples in the shape the model takes: contexts with one target each
/// (bigram, MLP), or whole windows with a target at every position (transformer).
fn batch(model: &dyn Model, ds: &Dataset, split: Split, n: usize, rng: &mut Rng) -> Examples {
    if model.targets_per_example() == 1 {
        ds.contexts(split, n, model.context(), rng)
    } else {
        ds.windows(split, n, model.context(), rng)
    }
}

/// A watched intermediate value over a whole evaluation set (rows of every chunk, stacked).
struct ActValues {
    name: String,
    value: Tensor,
    tanh: bool,
    residual: bool,
}

/// One attention node's `attention_sums`, added up over every chunk of an evaluation set.
struct AttnSums {
    name: String,
    entropy: Vec<F>,
    distance: Vec<F>,
    buckets: Vec<[F; 6]>,
    rows: usize,
}

/// A mean loss over predictions, with what is needed to express it per character.
struct EvalLoss {
    /// Mean nats per predicted token.
    mean: F,
    /// Bits per character: sum of nats / sum of characters of the predicted tokens / ln 2.
    bpc: F,
    /// With `kinds`: per kind of prediction, Σ nats and Σ characters.
    by_kind: Vec<(F, usize)>,
}

/// Mean loss over a fixed evaluation set, computed `chunk` examples at a time so each graph
/// is no bigger than a training batch's. Each chunk's mean loss is weighted by its number of
/// predictions, so the result is the mean over every prediction in the set, as one big
/// forward pass would give. With `keep_acts`, also returns the watched values over the set
/// and the attention summaries of every causal-softmax node. With `kinds` (one per prediction,
/// an index into `KINDS`), also splits the loss by kind.
fn eval_loss(model: &dyn Model, set: &Examples, chunk: usize, tok_chars: &[usize], keep_acts: bool, kinds: Option<&[usize]>) -> (EvalLoss, Vec<ActValues>, Vec<AttnSums>) {
    let (ctx, per) = (model.context(), model.targets_per_example());
    let n = set.1.len() / per;
    let (mut sum, mut count) = (0.0, 0usize);
    let mut by_kind = vec![(0.0, 0usize); if kinds.is_some() { KINDS.len() } else { 0 }];
    let mut acts: Vec<ActValues> = Vec::new();
    let mut attn: Vec<AttnSums> = Vec::new();
    for a in (0..n).step_by(chunk.max(1)) {
        let b = (a + chunk).min(n);
        // The activation and attention statistics come from the first DIAG_EXAMPLES examples
        // (all of them exact for those); the loss is over the whole set.
        let keep_acts = keep_acts && a < DIAG_EXAMPLES;
        let (g, f) = forward_eval(model, &set.0[a * ctx..b * ctx], &set.1[a * per..b * per], keep_acts);
        sum += g.value(f.loss).item() * ((b - a) * per) as F;
        count += (b - a) * per;
        if let Some(k) = kinds {
            let ys = &set.1[a * per..b * per];
            for (i, l) in per_row_cross_entropy(g.value(f.logits), ys).into_iter().enumerate() {
                let slot = &mut by_kind[k[a * per + i]];
                slot.0 += l;
                slot.1 += tok_chars[ys[i]];
            }
        }
        if keep_acts {
            for (i, act) in f.acts.iter().enumerate() {
                let v = g.value(act.var);
                if a == 0 {
                    acts.push(ActValues { name: act.name.clone(), value: Tensor::from_vec(0, v.cols, Vec::new()), tanh: act.tanh, residual: act.residual });
                }
                acts[i].value.rows += v.rows;
                acts[i].value.data.extend_from_slice(&v.data);
            }
            let attn_nodes = g.nodes.iter().filter_map(|nd| match nd.op {
                Op::CausalSoftmax { t, .. } => Some((nd, t)),
                _ => None,
            });
            for (i, (nd, t)) in attn_nodes.enumerate() {
                let heads = nd.value.rows / ((b - a) * t);
                let (e, d, rows) = attention_sums(&nd.value, t, heads);
                if a == 0 {
                    attn.push(AttnSums { name: nd.name.clone(), entropy: vec![0.0; heads], distance: vec![0.0; heads], buckets: vec![[0.0; 6]; heads], rows: 0 });
                }
                let s = &mut attn[i];
                s.entropy.iter_mut().zip(&e).for_each(|(x, v)| *x += v);
                s.distance.iter_mut().zip(&d).for_each(|(x, v)| *x += v);
                for (acc, bk) in s.buckets.iter_mut().zip(attention_buckets(&nd.value, t, heads)) {
                    acc.iter_mut().zip(bk).for_each(|(x, v)| *x += v);
                }
                s.rows += rows;
            }
        }
    }
    let chars: usize = set.1.iter().map(|&y| tok_chars[y]).sum();
    (EvalLoss { mean: sum / count as F, bpc: sum / chars as F / std::f64::consts::LN_2 as F, by_kind }, acts, attn)
}

/// Everything measured at an evaluation step, before that step's update.
#[allow(clippy::too_many_arguments)]
fn evaluate(model: &dyn Model, grads: &[Tensor], tr: &Examples, va: &Examples, va_kinds: &[usize], chunk: usize, ds: &Dataset, start_id: usize, sample_len: usize, rng: &mut Rng) -> Eval {
    let mut quant = Vec::new();
    for (p, g) in model.params().list.iter().zip(grads) {
        quant.push((p.name.clone(), quantiles(&p.value.data)));
        quant.push((format!("|grad {}|", p.name), quantiles(&g.data.iter().map(|v| v.abs()).collect::<Vec<_>>())));
    }
    let (train, acts, attn_sums) = eval_loss(model, tr, chunk, &ds.chars, true, None);
    let mut tanh = None;
    let mut resid = (Vec::new(), Vec::new());
    // The activations are large (every position of every evaluation example), so each one's
    // statistics are computed on a thread of its own; the results are the same either way.
    let stats = crate::par::map(&acts, |a| (quantiles(&a.value.data), a.tanh.then(|| tanh_stats(&a.value)), a.residual.then(|| rms(&a.value.data))));
    for (a, (q, t, r)) in acts.iter().zip(stats) {
        quant.push((a.name.clone(), q));
        if t.is_some() {
            tanh = t;
        }
        if let Some(r) = r {
            resid.0.push(a.name.clone());
            resid.1.push(r);
        }
    }
    let mut attn = (Vec::new(), Vec::new(), Vec::new());
    let mut buckets = (Vec::new(), Vec::new());
    // With a context of 1 every position is the first, which can only attend to itself:
    // no informative rows, so no attention series.
    for s in attn_sums.iter().filter(|s| s.rows > 0) {
        for h in 0..s.entropy.len() {
            attn.0.push(format!("{} head {h}", s.name));
            attn.1.push(s.entropy[h] / s.rows as F);
            attn.2.push(s.distance[h] / s.rows as F);
            for (b, &(_, _, label)) in DIST_BUCKETS.iter().enumerate() {
                buckets.0.push(format!("{} head {h}|{label}", s.name));
                buckets.1.push(s.buckets[h][b] / s.rows as F);
            }
        }
    }
    let (val, _, _) = eval_loss(model, va, chunk, &ds.chars, false, Some(va_kinds));
    let all_chars: usize = val.by_kind.iter().map(|k| k.1).sum();
    // A kind absent from the set has no loss to report (NaN, written as null).
    let kind_bpc = val.by_kind.iter().map(|&(n, c)| if c > 0 { n / c as F / std::f64::consts::LN_2 as F } else { F::NAN }).collect();
    let kind_share = val.by_kind.iter().map(|&(_, c)| c as F / all_chars.max(1) as F).collect();
    let text = ds.decode(&sample(model, start_id, sample_len, rng));
    Eval { train: train.mean, val: val.mean, train_bpc: train.bpc, val_bpc: val.bpc, text, quant, tanh, attn, resid, buckets, kind_bpc, kind_share }
}

#[allow(clippy::too_many_arguments)]
fn snapshot_json(
    step: usize,
    detail: Detail,
    lr: F,
    x: &[usize],
    y: &[usize],
    ctx: usize,
    per_example_rows: &[usize],
    g: &Graph,
    f: &Forward,
    params: &Params,
    updates: &[Tensor],
    moments_before: Option<&[(Tensor, Tensor)]>,
) -> String {
    // n examples, each with `per` predictions (1, or one per position of a window)
    let n = x.len() / ctx;
    let per = y.len() / n;
    let logits = g.value(f.logits);
    let probs = softmax_rows(logits);
    let losses = per_row_cross_entropy(logits, y);
    let (probs_s, losses_s) = match detail {
        Detail::Full => (packed(&probs.data), packed(&losses)),
        Detail::Summary => (packed(&probs.data[..per * probs.cols]), packed(&losses[..per])),
    };
    // The optimizer state: the update and the moments before it in both details (enough to
    // recompute the update); the moments after it only in full (they follow from the rest).
    let optim = array(
        &params
            .list
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut fields = vec![("name", string(&p.name)), ("decay", p.decay.to_string()), ("update", packed(&updates[i].data))];
                if let Some(mb) = moments_before {
                    fields.push(("m_before", packed(&mb[i].0.data)));
                    fields.push(("v_before", packed(&mb[i].1.data)));
                    if detail == Detail::Full {
                        fields.push(("m_after", packed(&p.m.data)));
                        fields.push(("v_after", packed(&p.v.data)));
                    }
                }
                object(&fields)
            })
            .collect::<Vec<_>>(),
    );
    object(&[
        ("step", step.to_string()),
        ("detail", string(detail.name())),
        ("lr", num64(lr)),
        ("n", n.to_string()),
        ("context", ctx.to_string()),
        ("x", ints(x)),
        ("y", ints(y)),
        ("nodes", graph_nodes(g, detail, n, per_example_rows)),
        ("probs", probs_s),
        ("token_losses", losses_s),
        ("optim", optim),
    ])
}

/// Per parameter, ‖a − b‖ / ‖b‖, with ‖b‖ floored at 1e-3 of the whole gradient's norm: some
/// gradients are exactly 0 in exact arithmetic (the key bias: softmax ignores a constant added
/// to a row of scores), and there only rounding is left on either side.
pub fn grad_rel_diffs(a: &[Tensor], b: &[Tensor]) -> Vec<F> {
    let total = b.iter().map(|t| t.norm().powi(2)).sum::<F>().sqrt();
    a.iter()
        .zip(b)
        .map(|(x, y)| x.data.iter().zip(&y.data).map(|(p, q)| (p - q) * (p - q)).sum::<F>().sqrt() / y.norm().max(1e-3 * total).max(1e-30))
        .collect()
}

/// Turns the GPU on and checks it against the CPU on one real batch: the loss and every
/// parameter's gradient (`grad_rel_diffs`; the GPU works in f32, so about 1e-6 is expected).
/// Refuses to train when they disagree. Returns the result for meta.js.
fn gpu_check(model: &dyn Model, ds: &Dataset, cfg: &Config, log: &mut Log) -> Result<String, String> {
    let name = cuda::enable()?;
    let (x, y) = batch(model, ds, Split::Train, cfg.examples, &mut Rng::new(1));
    let mut c = Graph::new();
    let fc = forward(model, &mut c, &x, &y);
    c.backward(fc.loss);
    let mut gg = gpu::Graph::new();
    let fg = forward(model, &mut gg, &x, &y);
    gg.backward(fg.loss);
    let (lc, lg) = (c.value(fc.loss).item(), gg.item(fg.loss));
    let loss_rel = (lg - lc).abs() / lc.abs().max(1e-30);
    let cpu: Vec<Tensor> = fc.params.iter().map(|p| c.grad(*p).clone()).collect();
    let on_gpu: Vec<Tensor> = fg.params.iter().map(|p| gg.grad(*p)).collect();
    let (mut worst, mut worst_name) = (0.0, String::new());
    for (rel, p) in grad_rel_diffs(&on_gpu, &cpu).into_iter().zip(&model.params().list) {
        if rel >= worst {
            (worst, worst_name) = (rel, p.name.clone());
        }
    }
    log.line(format!("
GPU: {name}; kernels from {}; forward and backward in f32, AdamW and the f64 master parameters on the CPU", cuda::ptx_build()));
    log.line(format!("GPU vs CPU on one batch: loss {lg:.6} vs {lc:.6} (relative difference {loss_rel:.1e}); largest gradient difference {worst:.1e} ({worst_name})"));
    if !(loss_rel < 1e-4 && worst < 1e-3) {
        return Err("the GPU disagrees with the CPU; refusing to train".into());
    }
    Ok(object(&[
        ("name", string(name)),
        ("kernels", string(cuda::ptx_build())),
        ("precision", string("f32 forward and backward; f64 optimizer and master parameters on the CPU")),
        ("check_loss_rel_diff", num(loss_rel)),
        ("check_grad_rel_diff", num(worst)),
        ("check_grad_worst_param", string(&worst_name)),
    ]))
}

/// Numbers in one full snapshot, measured on a real batch.
fn snapshot_numbers(model: &dyn Model, ds: &Dataset, cfg: &Config, adam: bool) -> usize {
    let (x, y) = batch(model, ds, Split::Train, cfg.examples, &mut Rng::new(0));
    let mut g = Graph::new();
    let f = forward(model, &mut g, &x, &y);
    g.backward(f.loss);
    graph_numbers(model, &g, &f, y.len(), adam)
}

/// Numbers in one full snapshot of a step whose graph is `g` (after backward).
pub(crate) fn graph_numbers(model: &dyn Model, g: &Graph, f: &crate::model::Forward, targets: usize, adam: bool) -> usize {
    let nodes: usize = g.nodes.iter().map(|nd| nd.value.len() * if nd.requires_grad { 2 } else { 1 }).sum();
    let optim = model.params().count() * if adam { 5 } else { 1 };
    nodes + g.value(f.logits).len() + targets + optim
}

fn provenance() -> Vec<(&'static str, String)> {
    let exe = std::env::current_exe().ok().and_then(|p| fs::read(p).ok()).map(|b| hex(&sha256(&b))).unwrap_or_else(|| "unknown".into());
    vec![
        ("exe_sha256", exe),
        ("rustc", env!("MICROLLM_RUSTC").to_string()),
        ("profile", if cfg!(debug_assertions) { "debug" } else { "release" }.to_string()),
        ("target", env!("MICROLLM_TARGET").to_string()),
        ("version", env!("CARGO_PKG_VERSION").to_string()),
    ]
}

/// Remove the previous run's generated files so nothing stale mixes with this one.
/// A run folder that is a link (symlink or junction) is refused, so deleting its old files can
/// never reach outside runs/.
fn prepare_dir(dir: &Path) -> Result<(), String> {
    if fs::symlink_metadata(dir).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(format!("{} is a link to another folder; choose another run name", dir.display()));
    }
    fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let inside = |p: &Path| fs::canonicalize(p).ok();
    match (inside(dir), dir.parent().and_then(inside)) {
        (Some(d), Some(root)) if d.parent() == Some(root.as_path()) => {}
        _ => return Err(format!("{} does not resolve to a folder inside runs/", dir.display())),
    }
    for f in ["trace.js", "metrics.js", "meta.js", "model.js", "landscape.js"] {
        let p = dir.join(f);
        if p.exists() {
            fs::remove_file(&p).map_err(|e| format!("removing {}: {e}", p.display()))?;
        }
    }
    let snaps = dir.join("snaps");
    if snaps.exists() {
        fs::remove_dir_all(&snaps).map_err(|e| format!("clearing {}: {e}", snaps.display()))?;
    }
    fs::create_dir_all(&snaps).map_err(|e| format!("creating {}: {e}", snaps.display()))
}

pub fn train(cfg: &Config) -> Result<(), String> {
    let dir = Path::new("runs").join(&cfg.run);
    prepare_dir(&dir)?;
    let log_path = dir.join("train.log");
    let file = fs::File::create(&log_path).map_err(|e| format!("creating {}: {e}", log_path.display()))?;
    let mut log = Log { file, path: log_path, error: None };
    // From here on every fatal error also lands in train.log.
    let result = run(cfg, &dir, &mut log);
    if let Err(e) = &result {
        log.file_only(&format!("error: {e}"));
    }
    match (result, log.result()) {
        (Err(e), Err(log_e)) => Err(format!("{e}; and train.log is incomplete: {log_e}")),
        (result, log_result) => result.and(log_result),
    }
}

fn run(cfg: &Config, dir: &Path, log: &mut Log) -> Result<(), String> {
    crate::par::set_threads(cfg.threads);
    let bytes = fs::read(&cfg.data).map_err(|e| format!("reading {}: {e}", cfg.data))?;
    let data_sha = hex(&sha256(&bytes));
    let text = String::from_utf8(bytes).map_err(|e| format!("{} is not UTF-8: {e}", cfg.data))?;
    let t_tok = Instant::now();
    let ds = Dataset::from_text(&text, cfg.val_frac, cfg.vocab);
    let tok_secs = t_tok.elapsed().as_secs_f64();
    let v = ds.vocab_size();
    // --chars: the step count that consumes about that many characters of training targets.
    let owned;
    let cfg = if cfg.chars > 0 {
        let per_step = (cfg.examples * if cfg.model == ModelKind::Transformer { cfg.context } else { 1 }) as F * ds.train_chars as F / ds.train.len() as F;
        let steps = (cfg.chars as F / per_step).ceil() as usize;
        if steps <= cfg.warmup {
            return Err(format!("--chars {} gives {steps} steps, not more than --warmup ({})", cfg.chars, cfg.warmup));
        }
        owned = Config { steps, ..cfg.clone() };
        &owned
    } else {
        cfg
    };

    let mut init_rng = Rng::new(cfg.seed ^ 0x1A17);
    let mut model: Box<dyn Model> = match cfg.model {
        ModelKind::Bigram => Box::new(Bigram::new(v)),
        ModelKind::Mlp => Box::new(Mlp::new(v, cfg.context, cfg.emb, cfg.hidden, &mut init_rng)),
        ModelKind::Transformer => Box::new(Transformer::with_variant(v, cfg.context, cfg.emb, cfg.heads, cfg.layers, cfg.variant, &mut init_rng)),
    };
    let ctx = model.context();
    let per = model.targets_per_example();
    let need = ctx;
    if ds.train.len() <= need || ds.val.len() <= need {
        return Err(format!(
            "{} is too short: the train and validation splits ({} and {} characters) must each be longer than the context ({need})",
            cfg.data,
            ds.train.len(),
            ds.val.len()
        ));
    }

    let mut opt = match cfg.optimizer {
        Kind::Sgd => Optimizer::sgd(cfg.lr),
        Kind::AdamW => Optimizer::adamw(cfg.lr, cfg.weight_decay),
        Kind::Muon => Optimizer::muon(cfg.lr, cfg.muon_lr, cfg.weight_decay, model.params()),
    };
    // whether the optimizer keeps per-number state (m and v)
    let adam = opt.kind != Kind::Sgd;
    let scheduled = cfg.warmup > 0 || cfg.lr_min != cfg.lr;

    let prov = provenance();
    let started = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);

    let n_params = model.params().count();
    let arch: Vec<String> = model.describe().iter().filter(|(k, _)| *k != "context").map(|(k, v)| format!("{k}={v}")).collect();
    log.line(format!("microllm {} · run {} · model {} · started at unix time {started}", env!("CARGO_PKG_VERSION"), cfg.run, model.kind()));
    log.line(format!(
        "config: steps={} examples/step={} context={} {} optimizer={} lr={} weight_decay={} seed={} eval_every={} eval_batches={} snap_every={} throttle_ms={} threads={} device={}",
        cfg.steps, cfg.examples, ctx, arch.join(" "), opt.name(), num64(cfg.lr), num64(if adam { cfg.weight_decay } else { 0.0 }), cfg.seed, cfg.eval_every, cfg.eval_batches, cfg.snap_every, cfg.throttle_ms, cfg.threads,
        if cfg.gpu { "gpu" } else { "cpu" }
    ));
    if cfg.wsd > 0.0 {
        log.line(format!(
            "learning rate: linear warm-up over {} steps to {}, constant, then over the last {}% of the steps a straight line down to {}",
            cfg.warmup,
            num64(cfg.lr),
            num64(100.0 * cfg.wsd),
            num64(cfg.lr_min)
        ));
    } else if scheduled {
        log.line(format!(
            "learning rate: linear warm-up over {} steps to {}, then a cosine down to {} at the last step",
            cfg.warmup,
            num64(cfg.lr),
            num64(cfg.lr_min)
        ));
    }
    log.line(format!("data: {} · {} characters · sha256 {data_sha}", cfg.data, text.chars().count()));
    let cpt = |toks: usize, chars: usize| chars as F / toks.max(1) as F;
    if ds.tok.merges.is_empty() {
        log.line(format!("tokenizer: characters ({v} tokens)"));
    } else {
        log.line(format!(
            "tokenizer: BPE, {v} tokens ({} base characters + {} merges) learned on the train split in {tok_secs:.1}s; characters per token: train {:.3}, val {:.3}",
            ds.tok.base.len(),
            ds.tok.merges.len(),
            cpt(ds.train.len(), ds.train_chars),
            cpt(ds.val.len(), ds.val_chars)
        ));
    }
    if cfg.chars > 0 {
        log.line(format!("--chars {}: {} steps", cfg.chars, cfg.steps));
    }
    log.line(format!("binary: {}", prov.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" · ")));

    log.line("\ngradcheck before training:");
    let checks = gradcheck::run_all();
    for l in gradcheck_lines(&checks) {
        log.line(l);
    }
    if checks.iter().any(|r| !r.passed()) {
        return Err("gradient check failed; refusing to train on wrong gradients".into());
    }
    let gpu_json = if cfg.gpu { Some(gpu_check(model.as_ref(), &ds, cfg, log)?) } else { None };

    // Fixed evaluation sets: the same examples every time, so evaluations are comparable.
    let mut eval_rng = Rng::new(cfg.seed ^ 0xE7A1);
    let n_eval = cfg.eval_batches * cfg.examples;
    let train_eval = batch(model.as_ref(), &ds, Split::Train, n_eval, &mut eval_rng);
    let val_eval = batch(model.as_ref(), &ds, Split::Val, n_eval, &mut eval_rng);
    // The kind of every validation prediction (see `token_kind`), from its token and the one before.
    let val_kinds: Vec<usize> = {
        let (c, p) = (model.context(), model.targets_per_example());
        (0..val_eval.1.len()).map(|i| token_kind(&ds.tok.vocab[val_eval.1[i]], &ds.tok.vocab[val_eval.0[(i / p) * c + if p == 1 { c - 1 } else { i % p }]])).collect()
    };

    // How big will the trace be? Decide full vs summary snapshots before starting.
    let snap_count = (0..cfg.steps).filter(|&s| is_snapshot_step(s, cfg)).count();
    let per_snap = snapshot_numbers(model.as_ref(), &ds, cfg, adam);
    let est_bytes = (per_snap * snap_count) as f64 * BYTES_PER_NUMBER;
    let (detail, detail_reason) = match cfg.detail {
        Some(d) => (d, "chosen with --detail".to_string()),
        None if est_bytes > AUTO_FULL_LIMIT_BYTES => (
            Detail::Summary,
            format!("auto: full snapshots would be about {:.0} MB (limit {:.0} MB); use --detail full to force", est_bytes / 1e6, AUTO_FULL_LIMIT_BYTES / 1e6),
        ),
        None => (Detail::Full, format!("auto: full snapshots estimated at {:.0} MB", est_bytes / 1e6)),
    };
    log.line(format!(
        "\n{}: vocab {v}, {n_params} params, {} train / {} val tokens ({} / {} characters), baseline loss ln({v}) = {:.4}",
        model.kind(),
        ds.train.len(),
        ds.val.len(),
        ds.train_chars,
        ds.val_chars,
        (v as F).ln()
    ));
    log.line(format!("trace: {snap_count} snapshots × {per_snap} numbers; detail = {} ({detail_reason})", detail.name()));

    let start_id = ds.encode_char('\n').unwrap_or(0);
    let names: Vec<String> = model.params().list.iter().map(|p| p.name.clone()).collect();
    // The character embeddings (the MLP's C, the transformer's wte) are recorded at every
    // evaluation for the character map.
    let emb_index = names.iter().position(|n| n == "C" || n == "wte");
    let params_json: Vec<String> = model
        .params()
        .list
        .iter()
        .enumerate()
        .map(|(k, p)| object(&[("name", string(&p.name)), ("shape", ints(&[p.value.rows, p.value.cols])), ("count", p.value.len().to_string()), ("decay", p.decay.to_string()), ("muon", opt.muon.get(k).copied().unwrap_or(false).to_string())]))
        .collect();
    let checks_json: Vec<String> = checks
        .iter()
        .map(|r| {
            object(&[
                ("name", string(&r.name)),
                ("checked", r.checked.to_string()),
                ("max_abs_err", num(r.max_abs_err)),
                ("max_rel_err", num(r.max_rel_err)),
                ("passed", r.passed().to_string()),
            ])
        })
        .collect();
    let vocab: Vec<String> = ds.tok.vocab.iter().map(|t| string(t)).collect();
    let example: String = text.chars().skip(ds.train_chars).take(600).collect();
    let tokenizer_json = object(&[
        ("kind", string(if ds.tok.merges.is_empty() { "char" } else { "bpe" })),
        ("pretokenizer", string(PRETOK)),
        ("base_size", ds.tok.base.len().to_string()),
        ("merges", array(&ds.tok.merges.iter().map(|m| ints(&[m.left, m.right, m.id, m.count])).collect::<Vec<_>>())),
        (
            "curve",
            object(&[
                ("merges", ints(&ds.tok.curve.iter().map(|p| p.merges).collect::<Vec<_>>())),
                ("train_tokens", ints(&ds.tok.curve.iter().map(|p| p.train_tokens).collect::<Vec<_>>())),
                ("val_tokens", ints(&ds.tok.curve.iter().map(|p| p.val_tokens).collect::<Vec<_>>())),
            ]),
        ),
        ("train_chars", ds.train_chars.to_string()),
        ("val_chars", ds.val_chars.to_string()),
        ("train_chars_per_token", num(cpt(ds.train.len(), ds.train_chars))),
        ("val_chars_per_token", num(cpt(ds.val.len(), ds.val_chars))),
        ("train_seconds", num(tok_secs)),
        // a test vector for other implementations of the tokenizer: the start of the val text and its ids
        ("example_text", string(&example)),
        ("example_ids", ints(&ds.tok.encode(&example))),
    ]);
    let mut arch_json: Vec<(&str, String)> = model.describe();
    arch_json.insert(0, ("kind", string(model.kind())));
    let mut prov_json: Vec<(&str, String)> = prov.iter().map(|(k, v)| (*k, string(v))).collect();
    prov_json.insert(0, ("data_sha256", string(&data_sha)));
    prov_json.push(("started_unix", started.to_string()));
    let static_meta: Vec<(&str, String)> = vec![
        ("version", "3".to_string()),
        ("run", string(&cfg.run)),
        ("instance", string(serve::instance())),
        ("val_fraction", num64(cfg.val_frac)),
        ("data", string(&Path::new(&cfg.data).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| cfg.data.clone()))),
        ("model", string(model.kind())),
        ("architecture", object(&arch_json)),
        ("vocab", array(&vocab)),
        ("vocab_size", v.to_string()),
        ("tokenizer", tokenizer_json),
        ("params", array(&params_json)),
        ("param_count", n_params.to_string()),
        ("steps", cfg.steps.to_string()),
        ("examples_per_step", cfg.examples.to_string()),
        ("context_length", ctx.to_string()),
        ("predictions_per_example", per.to_string()),
        ("tokens_per_step", (cfg.examples * per).to_string()),
        (
            "optimizer",
            object(&[
                ("kind", string(opt.name())),
                ("lr", num64(opt.lr)),
                ("lr_min", num64(cfg.lr_min)),
                ("warmup_steps", cfg.warmup.to_string()),
                ("schedule", string(if cfg.wsd > 0.0 { "wsd" } else { "cosine" })),
                ("wsd_decay_fraction", num64(cfg.wsd)),
                ("muon_lr", num64(cfg.muon_lr)),
                ("muon_momentum", num64(crate::optim::MUON_MOMENTUM)),
                ("beta1", num64(opt.beta1)),
                ("beta2", num64(opt.beta2)),
                ("eps", num64(opt.eps)),
                ("weight_decay", num64(opt.weight_decay)),
            ]),
        ),
        ("seed", cfg.seed.to_string()),
        ("eval_every", cfg.eval_every.to_string()),
        ("eval_examples", n_eval.to_string()),
        ("snap_every", cfg.snap_every.to_string()),
        ("snap_first", cfg.snap_first.to_string()),
        ("throttle_ms", cfg.throttle_ms.to_string()),
        ("threads", cfg.threads.to_string()),
        ("device", string(if cfg.gpu { "gpu" } else { "cpu" })),
        ("gpu", gpu_json.clone().unwrap_or_else(|| "null".into())),
        ("detail", string(detail.name())),
        ("detail_reason", string(&detail_reason)),
        ("train_tokens", ds.train.len().to_string()),
        ("val_tokens", ds.val.len().to_string()),
        ("baseline_loss", num((v as F).ln())),
        ("gradcheck", array(&checks_json)),
        ("gradcheck_tolerance", num(gradcheck::TOLERANCE)),
        ("gradcheck_floor", num(gradcheck::FLOOR)),
        (
            "definitions",
            object(&[
                ("saturated", string(&format!("|h| > {SATURATED}"))),
                ("dead", string(&format!("hidden unit saturated on at least {}% of the {n_eval} fixed train-eval examples", DEAD_FRACTION * 100.0))),
                ("precision", string("computed in f64; stored rounded to f32 (about 7 significant digits); long arrays in snapshots are packed as base64 f32")),
                (
                    "attention_entropy",
                    string(&format!("per head: −Σ a·ln a over the allowed positions, divided by ln(number allowed), so 0 = all weight on one character and 1 = spread evenly; averaged over every position except the first (which can only see itself) of the {n_eval} fixed train-eval windows")),
                ),
                ("attention_distance", string("per head: Σ a_j·(i − j), how many tokens back position i looks on average; averaged like the entropy")),
                ("attention_buckets", string("per head: the share of attention weight on tokens 0, 1, 2, 3, 4-7 and 8+ positions back; averaged like the entropy")),
                ("bits_per_char", string("Σ over predicted tokens of −ln p(token) / Σ characters of those tokens / ln 2: the loss per character in bits, comparable across tokenizers")),
                ("chars_seen", string("characters covered by the training targets of every update applied before this evaluation")),
                ("residual_rms", string("√(mean of the squares) of every number at that point of the residual stream, over the fixed train-eval windows")),
            ]),
        ),
        ("provenance", object(&prov_json)),
    ];
    let meta_path = dir.join("meta.js");
    let write_meta = |status: &str, reason: Option<&str>, snaps: &[usize], end: Option<(F, F, F, F, Vec<(&'static str, String)>)>| -> Result<(), String> {
        let mut fields = static_meta.clone();
        fields.push(("status", string(status)));
        if let Some(r) = reason {
            fields.push(("failure", string(r)));
        }
        fields.push(("snapshot_steps", ints(snaps)));
        if let Some((secs, compute, tl, vl, extra)) = end {
            fields.push(("train_seconds", num(secs)));
            fields.push(("compute_seconds", num(compute)));
            fields.push(("final_train_eval", num(tl)));
            fields.push(("final_val_eval", num(vl)));
            fields.extend(extra);
        }
        write_atomic(&meta_path, &format!("window.TRACE_META = {};\n", object(&fields)))
    };
    write_meta("running", None, &[], None)?;
    write_run_index()?;

    let live: Shared = Arc::new(Mutex::new(LiveState { metrics: Metrics::new(&names), snapshots: Vec::new(), status: "running", last_completed_step: None }));
    if let Some(port) = cfg.serve {
        serve::start(port, live.clone(), &cfg.run)?;
        log.line(format!("\nlive view: http://127.0.0.1:{port}/   (serving viewer/ and runs/ on this machine only)"));
    }
    if cfg.throttle_ms > 0 {
        log.line(format!("throttled: sleeping {} ms after every step so it can be watched", cfg.throttle_ms));
    }
    log.line("");

    let mut rng = Rng::new(cfg.seed);
    let mut sample_rng = Rng::new(cfg.seed ^ 0x5A5A);
    let mut slept = Duration::ZERO;
    let t0 = Instant::now();
    let mut failure = None;
    let mut chars_seen = 0usize;
    // Training time alone: batches, forward, backward and the update, without the
    // evaluations, snapshots and logging around them.
    let mut train_secs = 0.0;
    let mut eval_secs = 0.0;

    // On the GPU the parameters and the optimizer state stay there between steps.
    let resident = cfg.gpu.then(|| gpu::Resident::new(model.params(), &opt.muon));

    for step in 0..cfg.steps {
        let t_step = Instant::now();
        let (x, y) = batch(model.as_ref(), &ds, Split::Train, cfg.examples, &mut rng);

        // Forward and backward. On the GPU only the loss comes back, except at snapshot steps,
        // which record the whole graph; the gradients stay there for the optimizer.
        let snap = is_snapshot_step(step, cfg);
        let eval_step = is_eval_step(step, cfg);
        let mut snap_secs = 0.0;
        let (g, f, loss, mut grads) = if let Some(res) = &resident {
            let mut gg = gpu::Graph::with_params(res);
            let f = forward(model.as_ref(), &mut gg, &x, &y);
            gg.backward(f.loss);
            let loss = gg.item(f.loss);
            let t_snap = Instant::now();
            let g = if snap { gg.to_host(|_, _| true, true) } else { Graph::new() };
            snap_secs = t_snap.elapsed().as_secs_f64();
            (g, f, loss, Vec::new())
        } else {
            let mut g = Graph::new();
            let f = forward(model.as_ref(), &mut g, &x, &y);
            g.backward(f.loss);
            let loss = g.value(f.loss).item();
            let grads: Vec<Tensor> = f.params.iter().map(|p| g.grad(*p).clone()).collect();
            (g, f, loss, grads)
        };
        // On the CPU path this includes keeping the graph, which the step needs anyway.
        let mut step_secs = t_step.elapsed().as_secs_f64() - snap_secs;

        // Measure before updating, so everything recorded for this step is pre-update.
        let t_eval = Instant::now();
        if let (Some(res), true) = (&resident, eval_step) {
            // An evaluation reads the host copies: the parameters, and the gradients' quantiles.
            res.download(model.params_mut(), false);
            grads = res.grads();
        }
        let eval = eval_step.then(|| evaluate(model.as_ref(), &grads, &train_eval, &val_eval, &val_kinds, cfg.examples, &ds, start_id, cfg.sample_len, &mut sample_rng));
        // The embedding table as it is at this evaluation, before the update (like everything else recorded here).
        let embedding = eval.as_ref().and(emb_index).map(|i| model.params().list[i].value.data.clone());
        eval_secs += t_eval.elapsed().as_secs_f64();
        let moments_before: Option<Vec<(Tensor, Tensor)>> = (snap && adam).then(|| match &resident {
            Some(res) => res.moments(),
            None => model.params().list.iter().map(|p| (p.m.clone(), p.v.clone())).collect(),
        });

        let lr = if cfg.wsd > 0.0 { lr_wsd(cfg.lr, cfg.lr_min, cfg.warmup, cfg.steps, cfg.wsd, step) } else { lr_at(cfg.lr, cfg.lr_min, cfg.warmup, cfg.steps, step) };
        opt.lr = lr;
        // Muon's rate follows the same schedule
        opt.muon_lr = cfg.muon_lr * lr / cfg.lr;
        let t_update = Instant::now();
        let (updates, norms) = match &resident {
            Some(res) => {
                let norms = res.step(&mut opt);
                step_secs += t_update.elapsed().as_secs_f64();
                // A snapshot shows the parameters and moments after the update, and the update.
                let updates = if snap {
                    res.download(model.params_mut(), true);
                    res.updates()
                } else {
                    Vec::new()
                };
                (updates, norms)
            }
            None => {
                let param = model.params().list.iter().map(|p| p.value.norm()).collect();
                let t_update = Instant::now();
                let updates = opt.step(model.params_mut(), &grads);
                step_secs += t_update.elapsed().as_secs_f64();
                let norms = gpu::StepNorms { grad: grads.iter().map(Tensor::norm).collect(), param, update: updates.iter().map(Tensor::norm).collect() };
                (updates, norms)
            }
        };

        if snap {
            let json = snapshot_json(step, detail, lr, &x, &y, ctx, &model.rows_per_example(), &g, &f, model.params(), &updates, moments_before.as_deref());
            write_atomic(&dir.join("snaps").join(format!("{step}.js")), &format!("window.SNAP_LOADED({json});\n"))?;
        }

        let eval_failure = eval.as_ref().and_then(|e| {
            [("train", e.train), ("validation", e.val)].into_iter().find(|(_, l)| !l.is_finite()).map(|(which, l)| format!("{which} evaluation loss is {l}"))
        });
        let mut line = None;
        {
            let mut st = serve::lock(&live);
            let m = &mut st.metrics;
            m.loss.push(loss);
            m.lr.push(lr);
            for (i, ps) in m.params.iter_mut().enumerate() {
                let (pn, un) = (norms.param[i], norms.update[i]);
                ps.grad_norm.push(norms.grad[i]);
                ps.param_norm.push(pn);
                ps.update_norm.push(un);
                ps.ratio.push(if pn > 0.0 { un / pn } else { F::NAN });
            }
            if let Some(e) = eval {
                let k = m.evals();
                m.eval_step.push(step);
                m.eval_train.push(e.train);
                m.eval_val.push(e.val);
                m.eval_train_bpc.push(e.train_bpc);
                m.eval_val_bpc.push(e.val_bpc);
                m.eval_chars_seen.push(chars_seen);
                m.eval_train_secs.push(train_secs);
                m.sample.push(e.text);
                for (slot, (name, q)) in e.quant.into_iter().enumerate() {
                    m.push_quantiles(k, slot, &name, q);
                }
                let mut sat = String::new();
                if let Some(t) = e.tanh {
                    sat = format!("  tanh saturated {:.1}%  dead {}", t.saturated * 100.0, t.dead);
                    m.tanh_counts.push(t.counts);
                    m.tanh_saturated.push(t.saturated);
                    m.tanh_dead.push(t.dead);
                    m.tanh_slope.push(t.mean_slope);
                    m.tanh_unit_p50.push(t.unit_sat_p50);
                    m.tanh_unit_p95.push(t.unit_sat_p95);
                }
                if let Some(table) = embedding {
                    m.embedding.push(table);
                }
                if !e.attn.0.is_empty() {
                    m.attn_entropy.push(e.attn.0.clone(), e.attn.1);
                    m.attn_distance.push(e.attn.0, e.attn.2);
                    m.attn_buckets.push(e.buckets.0, e.buckets.1);
                }
                if !e.resid.0.is_empty() {
                    m.resid_rms.push(e.resid.0, e.resid.1);
                }
                let kind_names: Vec<String> = KINDS.iter().map(|k| k.to_string()).collect();
                m.kind_bpc.push(kind_names.clone(), e.kind_bpc);
                m.kind_chars.push(kind_names, e.kind_share);
                let gn = norms.grad.iter().map(|n| n.powi(2)).sum::<F>().sqrt();
                let lr_s = if scheduled { format!("  lr {lr:.2e}") } else { String::new() };
                line = Some(format!("step {step:>5}  batch loss {loss:.4}  train {:.4}  val {:.4} ({:.3} bits/char)  |grad| {gn:.4}{lr_s}{sat}", e.train, e.val, e.val_bpc));
            }
            if snap {
                st.snapshots.push(step); // only after its file is completely written
            }
            st.last_completed_step = Some(step);
        }
        chars_seen += ds.chars_of(&y);
        train_secs += step_secs;
        if let Some(l) = line {
            log.line(l);
        }
        // Stop at the first number that is not finite: training past it only spreads NaN.
        let non_finite = match &resident {
            // A sum of squares is finite only if every number in it is: find which one from host copies.
            Some(res) => {
                let all = [&norms.grad, &norms.param, &norms.update];
                (!loss.is_finite() || all.iter().any(|v| v.iter().any(|n| !n.is_finite()))).then(|| {
                    res.download(model.params_mut(), true);
                    first_non_finite(loss, &names, &res.grads(), &res.updates(), model.params()).unwrap_or_else(|| "a parameter was not finite before the update".into())
                })
            }
            None => first_non_finite(loss, &names, &grads, &updates, model.params()),
        };
        if let Some(what) = non_finite.or(eval_failure) {
            failure = Some(format!("step {step}: {what}"));
            break;
        }
        if cfg.throttle_ms > 0 {
            let d = Duration::from_millis(cfg.throttle_ms);
            std::thread::sleep(d);
            slept += d;
        }
    }

    if let Some(res) = &resident {
        res.download(model.params_mut(), true);
    }

    if let Some(reason) = failure {
        log.line(format!("\nTRAINING FAILED at {reason}; the recorded steps are kept for inspection"));
        let (metrics_json, snaps) = {
            let st = serve::lock(&live);
            st.metrics.check_lengths()?;
            (st.metrics.json(0, 0), st.snapshots.clone())
        };
        write_atomic(&dir.join("metrics.js"), &format!("window.TRACE_METRICS = {metrics_json};\n"))?;
        write_meta("failed", Some(&reason), &snaps, None)?;
        write_run_index()?;
        serve::lock(&live).status = "failed";
        if let Some(port) = cfg.serve {
            log.line(format!("still serving http://127.0.0.1:{port}/ so the failure can be inspected; press Ctrl+C to stop"));
        }
        log.result()?;
        if cfg.serve.is_some() {
            loop {
                std::thread::park();
            }
        }
        return Err(format!("training failed at {reason}"));
    }

    let secs = t0.elapsed().as_secs_f64();
    let compute = secs - slept.as_secs_f64();
    // The per-step evaluations are pre-update; this one sees the parameters after the last update.
    let (final_train, _, _) = eval_loss(model.as_ref(), &train_eval, cfg.examples, &ds.chars, false, None);
    let (final_val, _, _) = eval_loss(model.as_ref(), &val_eval, cfg.examples, &ds.chars, false, None);
    let (final_train, final_val) = (final_train.mean, final_val.mean);
    log.line(format!(
        "\ntrained {} steps in {secs:.2}s ({compute:.2}s excluding throttle); after training: train {final_train:.4}  val {final_val:.4}",
        cfg.steps
    ));
    log.line(format!(
        "time: training {train_secs:.2}s ({:.1} ms/step), evaluations {eval_secs:.2}s, everything else {:.2}s",
        train_secs * 1000.0 / cfg.steps as f64,
        compute - train_secs - eval_secs
    ));
    // How fast each level of validation loss was reached: the measure of training speed that
    // runs are compared on (the evaluations are on the same fixed validation batches every time).
    let reached: Vec<(F, Option<(usize, usize, F)>)> = {
        let st = serve::lock(&live);
        let m = &st.metrics;
        TARGET_BPC
            .iter()
            .map(|&t| (t, (0..m.evals()).find(|&i| m.eval_val_bpc[i] <= t).map(|i| (m.eval_step[i], m.eval_chars_seen[i], m.eval_train_secs[i]))))
            .collect()
    };
    log.line(format!("validation targets (first evaluation at or below; every {} steps):", cfg.eval_every));
    for (t, r) in &reached {
        log.line(match r {
            Some((step, chars, secs)) => format!("  val <= {t:.2} bits/char  at step {step:>5}  after {chars:>9} characters  {secs:>8.2}s of training"),
            None => format!("  val <= {t:.2} bits/char  not reached"),
        });
    }
    log.line(format!("sample:\n{}", ds.decode(&sample(model.as_ref(), start_id, 300, &mut sample_rng))));

    let (metrics_json, snaps) = {
        let st = serve::lock(&live);
        st.metrics.check_lengths()?;
        (st.metrics.json(0, 0), st.snapshots.clone())
    };
    write_atomic(&dir.join("metrics.js"), &format!("window.TRACE_METRICS = {metrics_json};\n"))?;
    // The benchmark every run is compared on, then what each head contributes to it.
    let t_bench = Instant::now();
    let bench = score_val(model.as_ref(), &ds, cfg.examples, usize::MAX);
    log.line(format!(
        "\nfinal benchmark (whole val split, {} tokens, {} characters, stride {}, mean context {:.1} characters): {:.4} bits/char ({:.4} nats/token)  [{:.1}s]",
        bench.tokens,
        bench.chars,
        bench.stride,
        bench.mean_context_chars(),
        bench.bpc(),
        bench.nats_per_token(),
        t_bench.elapsed().as_secs_f64()
    ));
    let score_json = |s: &Score| {
        object(&[
            ("bits_per_char", num(s.bpc())),
            ("nats_per_token", num(s.nats_per_token())),
            ("nats", num(s.nats)),
            ("tokens", s.tokens.to_string()),
            ("chars", s.chars.to_string()),
            ("mean_context_chars", num(s.mean_context_chars())),
            ("stride", s.stride.to_string()),
        ])
    };
    let targets: Vec<String> = reached
        .iter()
        .map(|(t, r)| {
            let (step, chars, secs) = match r {
                Some((a, b, c)) => (a.to_string(), b.to_string(), num(*c)),
                None => ("null".into(), "null".into(), "null".into()),
            };
            object(&[("val_bpc", num(*t)), ("step", step), ("chars_seen", chars), ("train_secs", secs)])
        })
        .collect();
    let mut extra = vec![
        ("final_bench", score_json(&bench)),
        ("final_chars_seen", chars_seen.to_string()),
        ("train_secs", num(train_secs)),
        ("eval_secs", num(eval_secs)),
        ("targets", array(&targets)),
    ];
    // The ablation scores the same amount of validation text for every tokenizer.
    let abl_tokens = ds.val.iter().scan(0usize, |c, &id| { *c += ds.chars[id]; Some(*c) }).take_while(|&c| c <= ABLATION_CHARS).count().max(1);
    // What each head attends to, on the same validation text as the ablation.
    let t_roles = Instant::now();
    if let Some(r) = head_roles(model.as_ref(), &ds, cfg.examples, abl_tokens) {
        log.line(format!("head patterns (first {} val tokens in {} windows; weight on each pattern, then what even attention would give):", r.tokens, r.windows));
        log.line(format!("  {:<18}{}", "", PATTERNS.iter().map(|p| format!("{p:>16}")).collect::<String>()));
        log.line(format!("  {:<18}{}", "even attention", r.even.iter().map(|v| format!("{v:>16.3}")).collect::<String>()));
        let rows: Vec<String> = r
            .heads
            .iter()
            .map(|(l, h, mass)| {
                log.line(format!("  {:<18}{}", format!("L{l} head {h}"), mass.iter().map(|v| format!("{v:>16.3}")).collect::<String>()));
                object(&[("layer", l.to_string()), ("head", h.to_string()), ("weight", nums(mass))])
            })
            .collect();
        log.line(format!("  [{:.1}s]", t_roles.elapsed().as_secs_f64()));
        extra.push((
            "head_patterns",
            object(&[
                ("patterns", array(&PATTERNS.iter().map(|p| string(p)).collect::<Vec<_>>())),
                ("tokens", r.tokens.to_string()),
                ("windows", r.windows.to_string()),
                ("even", nums(&r.even)),
                ("heads", array(&rows)),
            ]),
        ));
    }
    let t_abl = Instant::now();
    if let Some((base, heads)) = head_ablation(model.as_mut(), &ds, cfg.examples, abl_tokens) {
        log.line(format!("head ablation (first {} val tokens = {} characters, baseline {:.4} bits/char), change in bits/char with one head off:", base.tokens, base.chars, base.bpc()));
        let rows: Vec<String> = heads
            .iter()
            .map(|(l, h, s)| {
                log.line(format!("  L{l} head {h}: {:+.4}", s.bpc() - base.bpc()));
                object(&[("layer", l.to_string()), ("head", h.to_string()), ("bits_per_char", num(s.bpc())), ("delta", num(s.bpc() - base.bpc()))])
            })
            .collect();
        log.line(format!("  [{:.1}s]", t_abl.elapsed().as_secs_f64()));
        extra.push(("head_ablation", object(&[("baseline", score_json(&base)), ("heads", array(&rows))])));
    }
    write_checkpoint(model.as_ref(), &ds, &dir.join("model.js"))?;
    write_meta("complete", None, &snaps, Some((secs, compute, final_train, final_val, extra)))?;
    write_run_index()?;
    let size: u64 = fs::read_dir(dir.join("snaps")).map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
    log.line(format!("wrote {} (snapshots {:.1} MB)", dir.display(), size as f64 / 1e6));
    serve::lock(&live).status = "complete";
    if let Some(port) = cfg.serve {
        log.line(format!("training finished; still serving http://127.0.0.1:{port}/ — press Ctrl+C to stop"));
    }
    log.result()?;

    if cfg.serve.is_some() {
        loop {
            std::thread::park();
        }
    }
    Ok(())
}

/// Validation characters scored for each head ablation (a prefix: one pass per head).
const ABLATION_CHARS: usize = 16384;
/// Evaluation examples whose activations and attention are summarized at each evaluation
/// (8192 positions at context 64): enough for stable statistics, and copying every activation
/// of all of them back from the GPU would cost more than the training between evaluations.
const DIAG_EXAMPLES: usize = 128;
/// Validation levels (bits per character) whose time to reach is reported for every run.
const TARGET_BPC: [F; 7] = [3.0, 2.8, 2.6, 2.4, 2.3, 2.25, 2.2];

/// model.js: the final parameters at full precision, and a test vector (the first window
/// of the validation split and the logits this code gives for it) so another
/// implementation of the forward pass can prove it computes the same thing.
fn write_checkpoint(model: &dyn Model, ds: &Dataset, path: &Path) -> Result<(), String> {
    let params: Vec<String> = model
        .params()
        .list
        .iter()
        .map(|p| object(&[("name", string(&p.name)), ("shape", ints(&[p.value.rows, p.value.cols])), ("value", packed64(&p.value.data))]))
        .collect();
    let prompt = &ds.val[..model.context()];
    let mut g = Graph::new();
    let leaves = model.params().leaves(&mut g);
    let (logits, _) = model.logits_from(&mut g, &leaves, prompt);
    let lv = g.value(logits);
    let test = object(&[("prompt", ints(prompt)), ("rows", lv.rows.to_string()), ("logits", packed64(&lv.data))]);
    write_atomic(path, &format!("window.MODEL_LOADED({});\n", object(&[("params", array(&params)), ("test", test)])))
}

/// runs/index.js lists every run so the viewer can offer a picker.
pub fn write_run_index() -> Result<(), String> {
    let mut runs: Vec<String> = fs::read_dir("runs")
        .map_err(|e| format!("listing runs: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("meta.js").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    runs.sort();
    let list: Vec<String> = runs.iter().map(|r| string(r)).collect();
    write_atomic(&PathBuf::from("runs/index.js"), &format!("window.RUNS = {};\n", array(&list)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_folder_must_not_be_a_link() {
        let base = std::env::temp_dir().join(format!("microllm-prepare-test-{}", std::process::id()));
        let (runs, outside) = (base.join("runs"), base.join("outside"));
        fs::create_dir_all(&runs).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("meta.js"), "keep").unwrap();
        prepare_dir(&runs.join("plain")).unwrap();
        assert!(runs.join("plain").join("snaps").is_dir());
        // A junction needs no special rights on Windows; elsewhere a symlink does the same.
        let link = runs.join("link");
        #[cfg(windows)]
        let made = std::process::Command::new("cmd").args(["/c", "mklink", "/J"]).arg(&link).arg(&outside).output().is_ok_and(|o| o.status.success());
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&outside, &link).is_ok();
        if made {
            assert!(prepare_dir(&link).is_err());
            assert!(outside.join("meta.js").exists());
            fs::remove_dir(&link).unwrap();
        }
        fs::remove_dir_all(&base).unwrap();
    }

    fn schedule(lr_min: F, warmup: usize, steps: usize) -> Vec<F> {
        (0..steps).map(|s| lr_at(1.0, lr_min, warmup, steps, s)).collect()
    }

    #[test]
    fn learning_rate_warms_up_then_ends_at_lr_min() {
        let close = |a: &[F], b: &[F]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        // constant when warmup = 0 and lr_min = lr
        assert!(close(&schedule(1.0, 0, 4), &[1.0; 4]));
        // warmup 0: cosine from lr at step 0 to lr_min at the last step
        assert!(close(&schedule(0.0, 0, 3), &[1.0, 0.5, 0.0]));
        // warmup = steps − 2: one warm-up ramp, lr at step `warmup`, lr_min at the last
        assert!(close(&schedule(0.1, 2, 4), &[0.5, 1.0, 1.0, 0.1]));
        // warmup = steps − 1: the ramp, then the last step is already lr_min
        assert!(close(&schedule(1.0 / 3.0, 3, 4), &[1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0 / 3.0]));
        // a single step uses lr_min
        assert!(close(&schedule(0.25, 0, 1), &[0.25]));
        // the decay never goes below lr_min or above lr
        let s = schedule(0.1, 100, 5000);
        assert!(s.iter().all(|&v| v > 0.0 && v <= 1.0) && s[4999] == 0.1 && s[99] == 1.0 && s[100] == 1.0);
    }

    #[test]
    fn wsd_holds_then_decays_linearly() {
        let s: Vec<F> = (0..11).map(|t| lr_wsd(1.0, 0.1, 2, 11, 0.2, t)).collect();
        let want = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55, 0.1];
        assert!(s.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{s:?}");
    }

    #[test]
    #[should_panic]
    fn warmup_must_end_before_the_last_step() {
        lr_at(1.0, 0.1, 4, 4, 0);
    }
}
