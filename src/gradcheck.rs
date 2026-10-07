//! Gradient checking: compare backprop's gradients with finite differences.
//!
//! For every parameter element p we nudge it by ±h and ±2h, rebuild the graph each
//! time, and estimate d(loss)/dp with the 4-point central difference
//!   (−L(p+2h) + 8·L(p+h) − 8·L(p−h) + L(p−2h)) / (12h),
//! whose error shrinks like h⁴. That allows a fairly large h, which keeps rounding
//! noise (≈ 1e-16·|L| / h) far below the gradients being checked, even the tiny ones
//! in a deep graph. If backward is right, the two numbers agree to many decimal places.

use crate::autograd::{Graph, Var};
use crate::model::{Mlp, Model, Transformer};
use crate::rng::Rng;
use crate::tensor::{Tensor, F};

const EPS: F = 1e-3;
/// Step for the normalize checks (see `check_with`).
const NORM_EPS: F = 1e-4;
/// Max relative error to count as a pass. The 4-point differences in f64 land near 1e-10.
pub const TOLERANCE: F = 1e-6;
/// Relative error is |analytic − numeric| / max(|analytic| + |numeric|, FLOOR). Gradients
/// smaller than this (some are exactly 0, e.g. a key bias, which softmax ignores) are so judged
/// by absolute error, which must stay below FLOOR · TOLERANCE = 1e-11. Without a floor, rounding
/// noise of ~1e-13 on an exactly-zero gradient would read as a large "relative" error.
pub const FLOOR: F = 1e-5;

#[derive(Clone, Debug)]
pub struct GradReport {
    pub name: String,
    pub checked: usize,
    pub max_abs_err: F,
    pub max_rel_err: F,
}

impl GradReport {
    pub fn passed(&self) -> bool {
        self.max_rel_err < TOLERANCE
    }
}

/// Build the graph from `params` with `build`, then compare analytic and numeric gradients.
pub fn check<B>(name: &str, params: Vec<Tensor>, build: B) -> GradReport
where
    B: Fn(&mut Graph, &[Var]) -> Var,
{
    check_with(name, EPS, params, build)
}

/// `check` with step `h` instead of EPS. Normalize needs a smaller one: when a row's spread
/// is close to √eps, a nudge of 1e-3 changes the row's variance a lot, and the h⁴ error of
/// the difference formula is no longer small.
pub fn check_with<B>(name: &str, h: F, params: Vec<Tensor>, build: B) -> GradReport
where
    B: Fn(&mut Graph, &[Var]) -> Var,
{
    let run = |ps: &[Tensor]| -> (Graph, Vec<Var>, Var) {
        let mut g = Graph::new();
        let vars: Vec<Var> = ps.iter().enumerate().map(|(i, p)| g.param(&format!("p{i}"), p.clone())).collect();
        let loss = build(&mut g, &vars);
        (g, vars, loss)
    };

    let (mut g, vars, loss) = run(&params);
    g.backward(loss);
    let analytic: Vec<Tensor> = vars.iter().map(|v| g.grad(*v).clone()).collect();

    let mut report = GradReport { name: name.to_string(), checked: 0, max_abs_err: 0.0, max_rel_err: 0.0 };
    let mut ps = params.clone();
    for (pi, grad) in analytic.iter().enumerate() {
        for e in 0..ps[pi].len() {
            let orig = ps[pi].data[e];
            let mut loss_at = |delta: F| {
                ps[pi].data[e] = orig + delta;
                let (g, _, l) = run(&ps);
                g.value(l).item()
            };
            let (p2, p1, m1, m2) = (loss_at(2.0 * h), loss_at(h), loss_at(-h), loss_at(-2.0 * h));
            ps[pi].data[e] = orig;

            let numeric = (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * h);
            let a = grad.data[e];
            let abs = (a - numeric).abs();
            let mut rel = abs / (a.abs() + numeric.abs()).max(FLOOR);
            // f64::max silently drops NaN, so a NaN gradient would otherwise score a perfect 0.
            if !rel.is_finite() {
                rel = F::INFINITY;
            }
            report.max_abs_err = report.max_abs_err.max(if abs.is_finite() { abs } else { F::INFINITY });
            report.max_rel_err = report.max_rel_err.max(rel);
            report.checked += 1;
        }
    }
    report
}

fn randn(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
    Tensor::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal()).collect())
}

/// Reduce a non-scalar output to a scalar with fixed random weights, so every
/// output element gets a different, non-trivial upstream gradient.
fn project(g: &mut Graph, out: Var, rng_seed: u64) -> Var {
    let (r, c) = (g.value(out).rows, g.value(out).cols);
    let w = g.input("projection", randn(&mut Rng::new(rng_seed), r, c));
    let weighted = g.mul("weighted", out, w);
    g.mean("loss", weighted)
}

/// The standard suite: every op on its own, then graphs with branching and reuse.
pub fn run_all() -> Vec<GradReport> {
    let mut rng = Rng::new(1234);
    let mut reports = Vec::new();

    let (a, b) = (randn(&mut rng, 3, 4), randn(&mut rng, 3, 4));
    reports.push(check("add", vec![a.clone(), b.clone()], |g, p| {
        let o = g.add("out", p[0], p[1]);
        project(g, o, 1)
    }));
    reports.push(check("mul", vec![a.clone(), b.clone()], |g, p| {
        let o = g.mul("out", p[0], p[1]);
        project(g, o, 2)
    }));

    let (m1, m2) = (randn(&mut rng, 3, 5), randn(&mut rng, 5, 2));
    reports.push(check("matmul", vec![m1, m2], |g, p| {
        let o = g.matmul("out", p[0], p[1]);
        project(g, o, 3)
    }));

    let bias = randn(&mut rng, 1, 4);
    reports.push(check("add_row", vec![a.clone(), bias], |g, p| {
        let o = g.add_row("out", p[0], p[1]);
        project(g, o, 4)
    }));

    reports.push(check("tanh", vec![a.clone()], |g, p| {
        let o = g.tanh("out", p[0]);
        project(g, o, 5)
    }));

    let table = randn(&mut rng, 6, 3);
    reports.push(check("embedding (repeated ids)", vec![table.clone()], |g, p| {
        let o = g.embedding("out", p[0], &[2, 0, 2, 5, 2]);
        project(g, o, 6)
    }));

    let logits = randn(&mut rng, 4, 5);
    // Projected, not terminal: as the final node their upstream gradient would be exactly 1,
    // so a backward that ignored it would still pass.
    reports.push(check("cross_entropy", vec![logits], |g, p| {
        let o = g.cross_entropy("ce", p[0], &[1, 4, 0, 1]);
        project(g, o, 9)
    }));

    reports.push(check("mean", vec![a.clone()], |g, p| {
        let o = g.mean("m", p[0]);
        project(g, o, 10)
    }));

    let r = randn(&mut rng, 4, 6);
    reports.push(check("reshape (4x6 → 3x8)", vec![r], |g, p| {
        let o = g.reshape("out", p[0], 3, 8);
        project(g, o, 11)
    }));

    // Reuse: the same value feeds several ops, including both sides of one op.
    reports.push(check("reuse: x*x + x", vec![a.clone()], |g, p| {
        let sq = g.mul("x*x", p[0], p[0]);
        let o = g.add("x*x+x", sq, p[0]);
        project(g, o, 7)
    }));
    reports.push(check("reuse: x + x", vec![a.clone()], |g, p| {
        let o = g.add("x+x", p[0], p[0]);
        project(g, o, 8)
    }));

    // Branching: a small MLP language model step, end to end.
    let (emb, w, bvec) = (randn(&mut rng, 6, 3), randn(&mut rng, 3, 6), randn(&mut rng, 1, 6));
    reports.push(check("mlp: embedding→matmul→add_row→tanh→cross_entropy", vec![emb, w, bvec], |g, p| {
        let x = g.embedding("emb", p[0], &[1, 3, 3, 0]);
        let h = g.matmul("h", x, p[1]);
        let hb = g.add_row("h+b", h, p[2]);
        let t = g.tanh("tanh", hb);
        g.cross_entropy("loss", t, &[2, 2, 5, 0])
    }));

    // The real MLP model code (model.rs), shrunk: vocab 7, context 3, embedding 2, hidden 5.
    // Biases get random values too, so their gradients are checked away from zero.
    let tiny = Mlp::new(7, 3, 2, 5, &mut Rng::new(99));
    let tiny_params: Vec<Tensor> = tiny.params().list.iter().map(|p| randn(&mut rng, p.value.rows, p.value.cols)).collect();
    let (tx, ty) = ([1, 4, 4, 0, 6, 2, 3, 3, 5, 2, 0, 1], [6, 0, 4, 4]);
    reports.push(check("mlp model (real forward code, tiny)", tiny_params, |g, p| {
        let (logits, _) = tiny.logits_from(g, p, &tx);
        let ce = g.cross_entropy("ce", logits, &ty);
        project(g, ce, 12)
    }));

    // ---- transformer ops ----
    let w = randn(&mut rng, 3, 4);
    reports.push(check("scale", vec![w.clone()], |g, p| {
        let o = g.scale("out", p[0], 0.37);
        project(g, o, 13)
    }));
    let gain = randn(&mut rng, 1, 4);
    reports.push(check("mul_row", vec![w.clone(), gain], |g, p| {
        let o = g.mul_row("out", p[0], p[1]);
        project(g, o, 14)
    }));
    let wide = Tensor::from_vec(2, 5, vec![-4.0, -1.3, -0.2, 0.0, 0.1, 0.6, 1.1, 2.5, 3.7, -2.2]);
    reports.push(check("gelu (−4 … 3.7, including 0)", vec![wide], |g, p| {
        let o = g.gelu("out", p[0]);
        project(g, o, 15)
    }));
    let wide = Tensor::from_vec(2, 5, vec![-4.0, -1.3, -0.2, 0.05, 0.1, 0.6, 1.1, 2.5, 3.7, -2.2]);
    reports.push(check("relu² (−4 … 3.7)", vec![wide], |g, p| {
        let o = g.relu2("out", p[0]);
        project(g, o, 25)
    }));
    let rows = randn(&mut rng, 3, 5);
    reports.push(check_with("normalize (layernorm core)", NORM_EPS, vec![rows], |g, p| {
        let o = g.normalize("out", p[0], 1e-5);
        project(g, o, 16)
    }));
    // Rows whose spread is near √eps, where normalize's gradient bends hardest: a small
    // spread, and an almost constant row.
    let flat = Tensor::from_vec(2, 4, vec![0.01, -0.01, 0.02, 0.0, 1.0, 1.001, 0.999, 1.0005]);
    reports.push(check_with("normalize (rows nearly constant)", NORM_EPS, vec![flat], |g, p| {
        let o = g.normalize("out", p[0], 1e-5);
        project(g, o, 18)
    }));
    let (x, lg, lb) = (randn(&mut rng, 3, 4), randn(&mut rng, 1, 4), randn(&mut rng, 1, 4));
    reports.push(check_with("layernorm: normalize→mul_row→add_row (x, gain, bias)", NORM_EPS, vec![x, lg, lb], |g, p| {
        let xh = g.normalize("xhat", p[0], 1e-5);
        let s = g.mul_row("scaled", xh, p[1]);
        let o = g.add_row("ln", s, p[2]);
        project(g, o, 17)
    }));
    // n = 2 windows, T = 3 positions, H = 2 heads of width 2.
    let hx = randn(&mut rng, 6, 4);
    reports.push(check("split_heads (2·3 x 2·2 → 2·2·3 x 2)", vec![hx.clone()], |g, p| {
        let o = g.split_heads("out", p[0], 2, 3, 2);
        project(g, o, 18)
    }));
    let hy = randn(&mut rng, 12, 2);
    reports.push(check("merge_heads (2·2·3 x 2 → 2·3 x 2·2)", vec![hy], |g, p| {
        let o = g.merge_heads("out", p[0], 2, 3, 2);
        project(g, o, 19)
    }));
    let (ba, bb, bt) = (randn(&mut rng, 6, 4), randn(&mut rng, 8, 3), randn(&mut rng, 6, 4));
    reports.push(check("batch_matmul (2 blocks of 3x4 · 4x3)", vec![ba.clone(), bb], |g, p| {
        let o = g.batch_matmul("out", p[0], p[1], 2, false);
        project(g, o, 20)
    }));
    reports.push(check("batch_matmul, transposed (2 blocks of 3x4 · (3x4)ᵀ)", vec![ba, bt], |g, p| {
        let o = g.batch_matmul("out", p[0], p[1], 2, true);
        project(g, o, 21)
    }));
    let sc = randn(&mut rng, 6, 3);
    reports.push(check("causal_softmax (2 blocks, T = 3)", vec![sc], |g, p| {
        let o = g.causal_softmax("out", p[0], 3);
        project(g, o, 22)
    }));
    // One attention layer from the primitives: 2 windows, T = 3, 2 heads of width 2.
    let (qx, kx, vx) = (randn(&mut rng, 6, 4), randn(&mut rng, 6, 4), randn(&mut rng, 6, 4));
    reports.push(check("attention: split→q·kᵀ→scale→causal_softmax→·v→merge", vec![qx, kx, vx], |g, p| {
        let (q, k, v) = (g.split_heads("q", p[0], 2, 3, 2), g.split_heads("k", p[1], 2, 3, 2), g.split_heads("v", p[2], 2, 3, 2));
        let s = g.batch_matmul("qk", q, k, 4, true);
        let s = g.scale("scores", s, 1.0 / (2.0 as F).sqrt());
        let a = g.causal_softmax("attn", s, 3);
        let c = g.batch_matmul("ctx", a, v, 4, false);
        let o = g.merge_heads("context", c, 2, 3, 2);
        project(g, o, 23)
    }));

    // The real transformer model code (model.rs), shrunk: vocab 7, T 3, width 4, 2 heads, 2 layers.
    // All parameters random (gains away from 1, biases away from 0), 2 windows, n·T targets.
    let tiny = Transformer::new(7, 3, 4, 2, 2, &mut Rng::new(98));
    let tiny_params: Vec<Tensor> = tiny.params().list.iter().map(|p| randn(&mut rng, p.value.rows, p.value.cols)).collect();
    let (tx, ty) = ([1, 4, 4, 0, 6, 2], [4, 4, 0, 6, 2, 5]);
    reports.push(check("transformer model (real forward code, tiny)", tiny_params, |g, p| {
        let (logits, _) = tiny.logits_from(g, p, &tx);
        let ce = g.cross_entropy("ce", logits, &ty);
        project(g, ce, 24)
    }));

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_gradients_match_finite_differences() {
        for r in run_all() {
            assert!(r.passed(), "{}: max relative error {:.3e} (max abs {:.3e})", r.name, r.max_rel_err, r.max_abs_err);
        }
    }

    #[test]
    fn split_and_merge_heads_are_exact_inverses() {
        let mut rng = Rng::new(3);
        let x = randn(&mut rng, 2 * 5, 3 * 4); // n 2, T 5, H 3, dh 4
        let mut g = Graph::new();
        let a = g.input("x", x.clone());
        let s = g.split_heads("s", a, 2, 5, 3);
        // Window 1, head 2, position 3 is row (1·3 + 2)·5 + 3 = 28 and holds input row 1·5 + 3, columns 8..12.
        assert_eq!(g.value(s).row(28), &x.row(8)[8..12]);
        let m = g.merge_heads("m", s, 2, 5, 3);
        assert_eq!(g.value(m), &x, "merge(split(x)) must give x back bit for bit");
        let y = randn(&mut rng, 2 * 3 * 5, 4);
        let b = g.input("y", y.clone());
        let m2 = g.merge_heads("m2", b, 2, 5, 3);
        let s2 = g.split_heads("s2", m2, 2, 5, 3);
        assert_eq!(g.value(s2), &y, "split(merge(y)) must give y back bit for bit");
    }

    #[test]
    fn causal_softmax_masks_exactly() {
        let mut rng = Rng::new(4);
        let mut g = Graph::new();
        let x = g.param("scores", randn(&mut rng, 2 * 4, 4));
        let p = g.causal_softmax("p", x, 4);
        let w = g.input("w", randn(&mut rng, 8, 4));
        let pw = g.mul("pw", p, w);
        let loss = g.mean("loss", pw);
        g.backward(loss);
        for r in 0..8 {
            let row = g.value(p).row(r);
            assert!((row.iter().sum::<F>() - 1.0).abs() < 1e-12, "row {r} sums to 1");
            for c in 0..4 {
                let masked = c > r % 4;
                assert_eq!(row[c] == 0.0, masked, "row {r} col {c}: probability zero exactly when masked");
                if masked {
                    assert_eq!(g.grad(x).row(r)[c], 0.0, "masked entries get exactly zero gradient");
                }
            }
        }
    }

    #[test]
    fn normalize_gives_zero_mean_and_shrunk_unit_variance() {
        let mut g = Graph::new();
        let x = g.input("x", Tensor::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 0.01, -0.01, 0.02, 0.0]));
        let eps = 1e-5;
        let y = g.normalize("y", x, eps);
        for r in 0..2 {
            let xr = g.value(x).row(r);
            let m = xr.iter().sum::<F>() / 4.0;
            let var = xr.iter().map(|v| (v - m) * (v - m)).sum::<F>() / 4.0;
            let yr = g.value(y).row(r);
            let ym = yr.iter().sum::<F>() / 4.0;
            let yv = yr.iter().map(|v| v * v).sum::<F>() / 4.0;
            assert!(ym.abs() < 1e-12);
            // variance is var/(var+eps), not 1: visibly below 1 for the tiny second row
            assert!((yv - var / (var + eps)).abs() < 1e-12, "row {r}: {yv} vs {}", var / (var + eps));
        }
    }

    #[test]
    fn zero_logits_give_uniform_loss() {
        let mut g = Graph::new();
        let x = g.param("logits", Tensor::zeros(3, 65));
        let loss = g.cross_entropy("loss", x, &[0, 10, 64]);
        assert!((g.value(loss).item() - (65.0 as F).ln()).abs() < 1e-12);
    }}
