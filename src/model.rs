//! The models. Each owns a parameter registry (built once) and knows how to turn
//! a batch of contexts into next-character scores (logits) on a graph.
//!
//! - Bigram: one V x V table. Row `a` holds the scores for what follows `a`.
//! - MLP (Bengio et al. 2003): embed each of the last k characters, concatenate,
//!   one tanh hidden layer, then a linear layer to V scores.
//! - Transformer (a GPT in miniature): token + position embeddings, L pre-layernorm
//!   blocks of causal multi-head attention and a GELU MLP, a final layernorm, then a
//!   linear layer to V scores at every one of the T positions.

use crate::autograd::{Graph, Op, Tape, Var};
use crate::{cuda, gpu};
use crate::params::{ParamId, Params};
use crate::rng::Rng;
use crate::tensor::{softmax_rows, Tensor, F};

/// Intermediate values worth watching over training (histograms, saturation).
pub struct Act {
    pub name: String,
    pub var: Var,
    pub tanh: bool,
    /// A point on the transformer's residual stream (its size is tracked as an RMS).
    pub residual: bool,
}

pub trait Model {
    fn kind(&self) -> &'static str;
    /// Characters of context per example.
    fn context(&self) -> usize;
    /// Predictions per example: 1 (the character after the context) for the bigram and
    /// MLP; one per position (T) for the transformer, whose examples are whole windows.
    fn targets_per_example(&self) -> usize {
        1
    }
    /// Row counts a graph node can have per example (one row per example, one per context
    /// character, ...). In summary snapshots a node with n·k rows keeps example 0's first k.
    fn rows_per_example(&self) -> Vec<usize> {
        vec![1, self.context()]
    }
    fn params(&self) -> &Params;
    fn params_mut(&mut self) -> &mut Params;
    /// Build logits for `x` (n examples x context ids, flattened) from parameter
    /// leaves `p`, where `p[id.0]` is parameter `id`. Gradcheck calls this directly
    /// with its own leaves, so the real model code is what gets checked.
    fn logits_from(&self, g: &mut dyn Tape, p: &[Var], x: &[usize]) -> (Var, Vec<Act>);
    /// Architecture settings for the trace (name, JSON value).
    fn describe(&self) -> Vec<(&'static str, String)>;
    /// Heads (layer, head) whose output is zeroed in every later forward pass, to measure
    /// what each contributes. Only the transformer has heads.
    fn set_ablation(&mut self, _heads: Vec<(usize, usize)>) {}
    /// (layers, heads per layer); (0, 0) without attention.
    fn head_grid(&self) -> (usize, usize) {
        (0, 0)
    }
}

pub struct Forward {
    /// Graph leaf for every parameter, indexed by `ParamId.0`.
    pub params: Vec<Var>,
    pub logits: Var,
    pub loss: Var,
    pub acts: Vec<Act>,
}

pub fn forward(m: &dyn Model, g: &mut dyn Tape, x: &[usize], y: &[usize]) -> Forward {
    let params = m.params().leaves(g);
    let (logits, acts) = m.logits_from(g, &params, x);
    let loss = g.cross_entropy("loss", logits, y);
    Forward { params, logits, loss, acts }
}

/// A forward pass for measuring (no backward): on the GPU when it is enabled, returned as a
/// host graph either way. On the GPU only the logits and the loss are copied back, plus, with
/// `acts`, the watched values and every attention node; the other nodes are empty.
pub fn forward_eval(m: &dyn Model, x: &[usize], y: &[usize], acts: bool) -> (Graph, Forward) {
    if !cuda::enabled() {
        let mut g = Graph::new();
        let f = forward(m, &mut g, x, y);
        return (g, f);
    }
    let mut g = gpu::Graph::new();
    let f = forward(m, &mut g, x, y);
    let watched: Vec<usize> = f.acts.iter().map(|a| a.var.0).collect();
    let h = g.to_host(|k, op| k == f.logits.0 || k == f.loss.0 || (acts && (watched.contains(&k) || matches!(op, Op::CausalSoftmax { .. }))), false);
    (h, f)
}

/// The logits for `x` (and with `attn`, every attention node) like `forward_eval`, without a loss.
pub fn logits_eval(m: &dyn Model, x: &[usize], attn: bool) -> (Graph, Var) {
    if !cuda::enabled() {
        let mut g = Graph::new();
        let p = m.params().leaves(&mut g);
        let (logits, _) = m.logits_from(&mut g, &p, x);
        return (g, logits);
    }
    let mut g = gpu::Graph::new();
    let p = m.params().leaves(&mut g);
    let (logits, _) = m.logits_from(&mut g, &p, x);
    let h = g.to_host(|k, op| k == logits.0 || (attn && matches!(op, Op::CausalSoftmax { .. })), false);
    (h, logits)
}

/// Generate `n` token ids, one at a time, starting from a context filled with `start`.
pub fn sample(m: &dyn Model, start: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut ctx = vec![start; m.context()];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (g, logits) = logits_eval(m, &ctx, false);
        // The last row predicts what follows the whole context (the transformer gives one row
        // per position; the bigram and MLP give just that one).
        let probs = softmax_rows(g.value(logits));
        let next = rng.categorical(probs.row(probs.rows - 1));
        out.push(next);
        ctx.remove(0);
        ctx.push(next);
    }
    out
}

fn randn(rng: &mut Rng, rows: usize, cols: usize, scale: F) -> Tensor {
    Tensor::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal() * scale).collect())
}

pub struct Bigram {
    params: Params,
    table: ParamId,
}

impl Bigram {
    /// All-zero start: every next character is equally likely, so the first loss is exactly ln(V).
    pub fn new(vocab: usize) -> Self {
        let mut params = Params::default();
        let table = params.add("table", Tensor::zeros(vocab, vocab), false);
        Bigram { params, table }
    }
}

impl Model for Bigram {
    fn kind(&self) -> &'static str {
        "bigram"
    }
    fn context(&self) -> usize {
        1
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    fn logits_from(&self, g: &mut dyn Tape, p: &[Var], x: &[usize]) -> (Var, Vec<Act>) {
        (g.embedding("logits", p[self.table.0], x), vec![])
    }
    fn describe(&self) -> Vec<(&'static str, String)> {
        vec![]
    }
}

pub struct Mlp {
    params: Params,
    ctx: usize,
    emb: usize,
    hidden: usize,
    c: ParamId,
    w1: ParamId,
    b1: ParamId,
    w2: ParamId,
    b2: ParamId,
}

/// Gain for layers feeding tanh: keeps pre-activations from shrinking layer to layer (PyTorch's calculate_gain).
pub const TANH_GAIN: F = 5.0 / 3.0;
/// Extra shrink on the output layer so the first predictions are near uniform (loss ≈ ln V).
pub const OUTPUT_SCALE: F = 0.1;

impl Mlp {
    /// Initialization:
    ///   C  ~ N(0, 1)                         each input to W1 has variance 1
    ///   W1 ~ N(0, 1) · (5/3) / √(k·d)       tanh gain over fan-in
    ///   W2 ~ N(0, 1) · 0.1 / √hidden        linear gain, shrunk ×0.1
    ///   b1 = b2 = 0
    pub fn new(vocab: usize, ctx: usize, emb: usize, hidden: usize, rng: &mut Rng) -> Self {
        let mut params = Params::default();
        let fan_in = (ctx * emb) as F;
        let c = params.add("C", randn(rng, vocab, emb, 1.0), false);
        let w1 = params.add("W1", randn(rng, ctx * emb, hidden, TANH_GAIN / fan_in.sqrt()), true);
        let b1 = params.add("b1", Tensor::zeros(1, hidden), false);
        let w2 = params.add("W2", randn(rng, hidden, vocab, OUTPUT_SCALE / (hidden as F).sqrt()), true);
        let b2 = params.add("b2", Tensor::zeros(1, vocab), false);
        Mlp { params, ctx, emb, hidden, c, w1, b1, w2, b2 }
    }
}

impl Model for Mlp {
    fn kind(&self) -> &'static str {
        "mlp"
    }
    fn context(&self) -> usize {
        self.ctx
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    fn logits_from(&self, g: &mut dyn Tape, p: &[Var], x: &[usize]) -> (Var, Vec<Act>) {
        let n = x.len() / self.ctx;
        assert_eq!(n * self.ctx, x.len(), "mlp: {} ids is not a whole number of {}-character contexts", x.len(), self.ctx);
        let e = g.embedding("emb", p[self.c.0], x); // (n·k) x d: one row per context character
        let concat = g.reshape("concat", e, n, self.ctx * self.emb); // n x (k·d): a context's rows side by side
        let xw = g.matmul("concat·W1", concat, p[self.w1.0]);
        let pre = g.add_row("pre", xw, p[self.b1.0]);
        let h = g.tanh("h", pre);
        let hw = g.matmul("h·W2", h, p[self.w2.0]);
        let logits = g.add_row("logits", hw, p[self.b2.0]);
        (logits, vec![Act { name: "pre".into(), var: pre, tanh: false, residual: false }, Act { name: "h".into(), var: h, tanh: true, residual: false }])
    }
    fn describe(&self) -> Vec<(&'static str, String)> {
        vec![
            ("context", self.ctx.to_string()),
            ("embedding_dim", self.emb.to_string()),
            ("hidden", self.hidden.to_string()),
            ("tanh_gain", format!("{}", TANH_GAIN)),
            ("output_scale", format!("{}", OUTPUT_SCALE)),
        ]
    }
}

/// Layernorm's ε, as in GPT-2.
pub const LN_EPS: F = 1e-5;
/// Standard deviation of every weight matrix and embedding at the start (GPT-2).
pub const INIT_STD: F = 0.02;

/// Changes to the GPT-2 transformer that the speed experiments switch on one at a time
/// (all off by default).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Variant {
    /// The MLP uses relu(x)² instead of GELU.
    pub relu2: bool,
    /// Queries and keys are normalized per head (no mean or variance left, then a learned
    /// gain per head dimension) before their dot products, which bounds the attention scores.
    pub qk_norm: bool,
    /// Wo and W_down (the matrices that write into the residual stream) start at zero, so
    /// every block starts as the identity.
    pub zero_init: bool,
}

impl Variant {
    /// "none" or a comma-separated list of relu2, qk-norm, zero-init.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut v = Variant::default();
        for part in s.split(',').filter(|p| *p != "none") {
            match part {
                "relu2" => v.relu2 = true,
                "qk-norm" => v.qk_norm = true,
                "zero-init" => v.zero_init = true,
                other => return Err(format!("unknown variant {other}; use none or a list of relu2, qk-norm, zero-init")),
            }
        }
        Ok(v)
    }
}

/// One block's parameters.
struct Block {
    ln1_g: ParamId,
    ln1_b: ParamId,
    wq: ParamId,
    bq: ParamId,
    wk: ParamId,
    bk: ParamId,
    wv: ParamId,
    bv: ParamId,
    wo: ParamId,
    bo: ParamId,
    ln2_g: ParamId,
    ln2_b: ParamId,
    w_up: ParamId,
    b_up: ParamId,
    w_down: ParamId,
    b_down: ParamId,
    /// With qk-norm: gains (1 × head size) for the normalized queries and keys.
    q_gain: Option<ParamId>,
    k_gain: Option<ParamId>,
}

pub struct Transformer {
    params: Params,
    t: usize,
    d: usize,
    heads: usize,
    layers: usize,
    wte: ParamId,
    wpe: ParamId,
    blocks: Vec<Block>,
    lnf_g: ParamId,
    lnf_b: ParamId,
    w_out: ParamId,
    b_out: ParamId,
    /// Heads (layer, head) switched off: see `set_ablation`.
    ablate: Vec<(usize, usize)>,
    variant: Variant,
}

impl Transformer {
    /// Initialization (GPT-2):
    ///   token and position embeddings, every weight matrix ~ N(0, 0.02²)
    ///   the two matrices that write into the residual stream (Wo, W_down) ~ N(0, 0.02²) / √(2L),
    ///   so the stream's variance does not grow with depth
    ///   biases 0, layernorm gains 1 and biases 0
    /// Weight decay applies to the weight matrices only, not to embeddings, biases or layernorm.
    pub fn new(vocab: usize, t: usize, d: usize, heads: usize, layers: usize, rng: &mut Rng) -> Self {
        Self::with_variant(vocab, t, d, heads, layers, Variant::default(), rng)
    }

    /// As `new`, with `variant`'s changes. The random draws are the same either way (a
    /// zero-initialized matrix is drawn and then cleared), so variants start from the same
    /// numbers everywhere else.
    pub fn with_variant(vocab: usize, t: usize, d: usize, heads: usize, layers: usize, variant: Variant, rng: &mut Rng) -> Self {
        assert!(vocab > 0 && t > 0 && d > 0 && layers > 0, "transformer: vocab, context, width and layers must be positive");
        assert!(heads > 0 && d % heads == 0, "transformer: width {d} must split evenly into {heads} heads");
        let mut params = Params::default();
        let ones = |n| Tensor::from_vec(1, n, vec![1.0; n]);
        let resid = INIT_STD / ((2 * layers) as F).sqrt();
        let out_init = |w: Tensor| if variant.zero_init { Tensor::zeros(w.rows, w.cols) } else { w };
        let wte = params.add("wte", randn(rng, vocab, d, INIT_STD), false);
        let wpe = params.add("wpe", randn(rng, t, d, INIT_STD), false);
        let mut blocks = Vec::new();
        for l in 0..layers {
            let n = |s: &str| format!("L{l}.{s}");
            blocks.push(Block {
                ln1_g: params.add(&n("ln1.gain"), ones(d), false),
                ln1_b: params.add(&n("ln1.bias"), Tensor::zeros(1, d), false),
                wq: params.add(&n("Wq"), randn(rng, d, d, INIT_STD), true),
                bq: params.add(&n("bq"), Tensor::zeros(1, d), false),
                wk: params.add(&n("Wk"), randn(rng, d, d, INIT_STD), true),
                bk: params.add(&n("bk"), Tensor::zeros(1, d), false),
                wv: params.add(&n("Wv"), randn(rng, d, d, INIT_STD), true),
                bv: params.add(&n("bv"), Tensor::zeros(1, d), false),
                wo: params.add(&n("Wo"), out_init(randn(rng, d, d, resid)), true),
                bo: params.add(&n("bo"), Tensor::zeros(1, d), false),
                ln2_g: params.add(&n("ln2.gain"), ones(d), false),
                ln2_b: params.add(&n("ln2.bias"), Tensor::zeros(1, d), false),
                w_up: params.add(&n("W_up"), randn(rng, d, 4 * d, INIT_STD), true),
                b_up: params.add(&n("b_up"), Tensor::zeros(1, 4 * d), false),
                w_down: params.add(&n("W_down"), out_init(randn(rng, 4 * d, d, resid)), true),
                b_down: params.add(&n("b_down"), Tensor::zeros(1, d), false),
                q_gain: variant.qk_norm.then(|| params.add(&n("q.gain"), ones(d / heads), false)),
                k_gain: variant.qk_norm.then(|| params.add(&n("k.gain"), ones(d / heads), false)),
            });
        }
        let lnf_g = params.add("lnf.gain", ones(d), false);
        let lnf_b = params.add("lnf.bias", Tensor::zeros(1, d), false);
        let w_out = params.add("W_out", randn(rng, d, vocab, INIT_STD), true);
        let b_out = params.add("b_out", Tensor::zeros(1, vocab), false);
        Transformer { params, t, d, heads, layers, wte, wpe, blocks, lnf_g, lnf_b, w_out, b_out, ablate: Vec::new(), variant }
    }

    /// layernorm(x) = gain · (x − mean)/√(var + ε) + bias, as three readable nodes.
    fn layernorm(g: &mut dyn Tape, name: &str, x: Var, gain: Var, bias: Var) -> Var {
        let xhat = g.normalize(&format!("{name}.xhat"), x, LN_EPS);
        let scaled = g.mul_row(&format!("{name}.scaled"), xhat, gain);
        g.add_row(name, scaled, bias)
    }

    fn linear(g: &mut dyn Tape, name: &str, x: Var, w: Var, b: Var) -> Var {
        let xw = g.matmul(&format!("{name}.xW"), x, w);
        g.add_row(name, xw, b)
    }
}

impl Model for Transformer {
    fn kind(&self) -> &'static str {
        "transformer"
    }
    fn context(&self) -> usize {
        self.t
    }
    fn targets_per_example(&self) -> usize {
        self.t
    }
    /// Per example: 1 row (the loss), T rows (one per position), H·T rows (head-split
    /// nodes and attention: rows (b·H + h)·T + p, so example 0 is the first H·T).
    fn rows_per_example(&self) -> Vec<usize> {
        vec![1, self.t, self.heads * self.t]
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    /// `x` holds n windows of T ids; the logits have one row per position (n·T rows),
    /// row b·T + i scoring the character after position i of window b.
    fn logits_from(&self, g: &mut dyn Tape, p: &[Var], x: &[usize]) -> (Var, Vec<Act>) {
        let (t, h) = (self.t, self.heads);
        let n = x.len() / t;
        assert_eq!(n * t, x.len(), "transformer: {} ids is not a whole number of {t}-character windows", x.len());
        let positions: Vec<usize> = (0..n).flat_map(|_| 0..t).collect();
        let tok = g.embedding("tok", p[self.wte.0], x);
        let pos = g.embedding("pos", p[self.wpe.0], &positions);
        let mut res = g.add("input", tok, pos);
        let watch = |name: String, var: Var, residual: bool| Act { name, var, tanh: false, residual };
        let mut acts = vec![watch("input".into(), res, true)];
        let scale = 1.0 / ((self.d / h) as F).sqrt();
        for (l, b) in self.blocks.iter().enumerate() {
            let n_ = |s: &str| format!("L{l}.{s}");
            // attention
            let a = Self::layernorm(g, &n_("ln1"), res, p[b.ln1_g.0], p[b.ln1_b.0]);
            acts.push(watch(n_("ln1"), a, false));
            let q = Self::linear(g, &n_("q"), a, p[b.wq.0], p[b.bq.0]);
            let k = Self::linear(g, &n_("k"), a, p[b.wk.0], p[b.bk.0]);
            let v = Self::linear(g, &n_("v"), a, p[b.wv.0], p[b.bv.0]);
            let mut qh = g.split_heads(&n_("q.heads"), q, n, t, h);
            let mut kh = g.split_heads(&n_("k.heads"), k, n, t, h);
            if let (Some(qg), Some(kg)) = (b.q_gain, b.k_gain) {
                let qn = g.normalize(&n_("q.norm"), qh, LN_EPS);
                qh = g.mul_row(&n_("q.normed"), qn, p[qg.0]);
                let kn = g.normalize(&n_("k.norm"), kh, LN_EPS);
                kh = g.mul_row(&n_("k.normed"), kn, p[kg.0]);
            }
            let vh = g.split_heads(&n_("v.heads"), v, n, t, h);
            let qk = g.batch_matmul(&n_("q·kᵀ"), qh, kh, n * h, true);
            let scores = g.scale(&n_("scores"), qk, scale);
            let attn = g.causal_softmax(&n_("attn"), scores, t);
            let ctx = g.batch_matmul(&n_("attn·v"), attn, vh, n * h, false);
            let mut merged = g.merge_heads(&n_("context"), ctx, n, t, h);
            // An ablated head's slice of the context is multiplied by 0 before Wo, so it adds
            // nothing to the residual stream (bo still does).
            if self.ablate.iter().any(|&(al, _)| al == l) {
                let hd = self.d / h;
                let mask: Vec<F> = (0..self.d).map(|c| if self.ablate.contains(&(l, c / hd)) { 0.0 } else { 1.0 }).collect();
                let mask = g.input(&n_("ablation mask"), Tensor::from_vec(1, self.d, mask));
                merged = g.mul_row(&n_("context.ablated"), merged, mask);
            }
            let out = Self::linear(g, &n_("attn_out"), merged, p[b.wo.0], p[b.bo.0]);
            res = g.add(&n_("resid1"), res, out);
            acts.push(watch(n_("resid1"), res, true));
            // MLP
            let m = Self::layernorm(g, &n_("ln2"), res, p[b.ln2_g.0], p[b.ln2_b.0]);
            let up = Self::linear(g, &n_("up"), m, p[b.w_up.0], p[b.b_up.0]);
            let act = if self.variant.relu2 { g.relu2(&n_("relu²"), up) } else { g.gelu(&n_("gelu"), up) };
            let down = Self::linear(g, &n_("mlp_out"), act, p[b.w_down.0], p[b.b_down.0]);
            res = g.add(&n_("resid2"), res, down);
            acts.push(watch(n_("up"), up, false));
            acts.push(watch(n_("resid2"), res, true));
        }
        let f = Self::layernorm(g, "lnf", res, p[self.lnf_g.0], p[self.lnf_b.0]);
        acts.push(watch("lnf".into(), f, false));
        let logits = Self::linear(g, "logits", f, p[self.w_out.0], p[self.b_out.0]);
        (logits, acts)
    }
    fn describe(&self) -> Vec<(&'static str, String)> {
        vec![
            ("context", self.t.to_string()),
            ("embedding_dim", self.d.to_string()),
            ("heads", self.heads.to_string()),
            ("layers", self.layers.to_string()),
            ("mlp_hidden", (4 * self.d).to_string()),
            ("ln_eps", format!("{LN_EPS}")),
            ("init_std", format!("{INIT_STD}")),
            ("mlp_relu2", self.variant.relu2.to_string()),
            ("qk_norm", self.variant.qk_norm.to_string()),
            ("zero_init_out", self.variant.zero_init.to_string()),
        ]
    }
    fn set_ablation(&mut self, heads: Vec<(usize, usize)>) {
        self.ablate = heads;
    }
    fn head_grid(&self) -> (usize, usize) {
        (self.layers, self.heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_has_expected_parameter_count() {
        let m = Mlp::new(65, 8, 16, 128, &mut Rng::new(1));
        assert_eq!(m.params().count(), 65 * 16 + 128 * 128 + 128 + 128 * 65 + 65);
        assert_eq!(m.params().count(), 25_937);
    }

    #[test]
    fn mlp_starts_near_uniform() {
        let m = Mlp::new(65, 8, 16, 128, &mut Rng::new(1));
        let x: Vec<usize> = (0..32 * 8).map(|i| (i * 7) % 65).collect();
        let y: Vec<usize> = (0..32).map(|i| (i * 3) % 65).collect();
        let mut g = Graph::new();
        let f = forward(&m, &mut g, &x, &y);
        let loss = g.value(f.loss).item();
        assert!((loss - (65.0 as F).ln()).abs() < 0.1, "initial loss {loss} should be near ln 65");
    }

    #[test]
    fn transformer_has_expected_parameter_count() {
        let (v, t, d, l) = (65, 32, 48, 2);
        let m = Transformer::new(v, t, d, 4, l, &mut Rng::new(1));
        let block = 2 * d + 4 * (d * d + d) + 2 * d + (d * 4 * d + 4 * d) + (4 * d * d + d);
        assert_eq!(m.params().count(), v * d + t * d + l * block + 2 * d + d * v + v);
        assert_eq!(m.params().count(), 64_481);
        // weight decay on the 6 matrices per block and the output matrix, nothing else
        assert_eq!(m.params().list.iter().filter(|p| p.decay).count(), 6 * l + 1);
    }

    #[test]
    fn transformer_starts_near_uniform() {
        let m = Transformer::new(65, 8, 16, 4, 2, &mut Rng::new(1));
        let x: Vec<usize> = (0..4 * 8).map(|i| (i * 7) % 65).collect();
        let y: Vec<usize> = (0..4 * 8).map(|i| (i * 3) % 65).collect();
        let mut g = Graph::new();
        let f = forward(&m, &mut g, &x, &y);
        assert_eq!(g.value(f.logits).rows, 4 * 8, "one row of logits per position");
        let loss = g.value(f.loss).item();
        assert!((loss - (65.0 as F).ln()).abs() < 0.05, "initial loss {loss} should be near ln 65");
    }

    /// Attention recomputed from its definition, straight from the q, k, v nodes before the
    /// head split (head h = columns h·dh .. (h+1)·dh), so a missing 1/√dh, mixed-up heads or a
    /// wrong merge order cannot hide behind split and merge agreeing with each other.
    #[test]
    fn transformer_attention_matches_definition() {
        let (n, t, d, h) = (2, 5, 8, 2);
        let dh = d / h;
        let mut rng = Rng::new(6);
        let mut m = Transformer::new(65, t, d, h, 1, &mut rng);
        for p in &mut m.params_mut().list {
            for v in &mut p.value.data {
                *v += 0.3 * rng.normal();
            }
        }
        let x: Vec<usize> = (0..n * t).map(|i| (i * 13 + 1) % 65).collect();
        let mut g = Graph::new();
        let p = m.params().leaves(&mut g);
        m.logits_from(&mut g, &p, &x);
        let node = |name: &str| g.nodes.iter().find(|nd| nd.name == name).unwrap_or_else(|| panic!("no node {name}")).value.clone();
        let (tok, pos, input) = (node("tok"), node("pos"), node("input"));
        let wpe = &m.params().list[m.wpe.0].value;
        for r in 0..n * t {
            assert_eq!(pos.row(r), wpe.row(r % t), "row {r} gets position {}'s embedding", r % t);
            for c in 0..d {
                assert_eq!(input.at(r, c), tok.at(r, c) + pos.at(r, c), "input = token + position");
            }
        }
        let (q, k, v, attn, context) = (node("L0.q"), node("L0.k"), node("L0.v"), node("L0.attn"), node("L0.context"));
        let scale = 1.0 / (dh as F).sqrt();
        for b in 0..n {
            for hd in 0..h {
                for i in 0..t {
                    let row = (b * h + hd) * t + i;
                    let qi = &q.row(b * t + i)[hd * dh..(hd + 1) * dh];
                    let scores: Vec<F> = (0..=i).map(|j| qi.iter().zip(&k.row(b * t + j)[hd * dh..(hd + 1) * dh]).map(|(a, c)| a * c).sum::<F>() * scale).collect();
                    let mx = scores.iter().cloned().fold(F::NEG_INFINITY, F::max);
                    let z: F = scores.iter().map(|s| (s - mx).exp()).sum();
                    for j in 0..t {
                        let want = if j <= i { (scores[j] - mx).exp() / z } else { 0.0 };
                        assert!((attn.at(row, j) - want).abs() < 1e-12, "attn window {b} head {hd} row {i} col {j}: {} vs {want}", attn.at(row, j));
                    }
                    for c in 0..dh {
                        let want: F = (0..=i).map(|j| attn.at(row, j) * v.at(b * t + j, hd * dh + c)).sum();
                        assert!((context.at(b * t + i, hd * dh + c) - want).abs() < 1e-12, "context window {b} position {i} head {hd} column {c}");
                    }
                }
            }
        }
    }

    /// Changing characters after position i must not change position i's logits at all:
    /// any leak through the mask, the head reshapes or the residual stream would show here.
    #[test]
    fn transformer_is_causal() {
        let mut rng = Rng::new(5);
        let m = Transformer::new(65, 8, 16, 4, 2, &mut rng);
        // Make biases and layernorm parameters non-trivial so every path carries signal.
        let mut m = m;
        for p in &mut m.params_mut().list {
            for v in &mut p.value.data {
                *v += 0.3 * rng.normal();
            }
        }
        let logits = |x: &[usize]| {
            let mut g = Graph::new();
            let p = m.params().leaves(&mut g);
            let (l, _) = m.logits_from(&mut g, &p, x);
            g.value(l).clone()
        };
        let base: Vec<usize> = (0..2 * 8).map(|i| (i * 11 + 3) % 65).collect();
        let lb = logits(&base);
        for cut in 0..8 {
            let mut x = base.clone();
            for w in 0..2 {
                for i in cut + 1..8 {
                    x[w * 8 + i] = (x[w * 8 + i] + 17) % 65;
                }
            }
            let lx = logits(&x);
            for w in 0..2 {
                for i in 0..8 {
                    let same = lb.row(w * 8 + i) == lx.row(w * 8 + i);
                    assert_eq!(same, i <= cut, "window {w}, position {i}, future changed after {cut}: logits {}", if same { "unchanged" } else { "changed" });
                }
            }
        }
    }
}
