//! Optimizers. A step reads each parameter's gradient, updates its state, and
//! returns the exact change it applied, so the trace can show it.

use crate::params::{Param, Params};
use crate::tensor::{Tensor, F};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// p ← p − lr·g
    Sgd,
    /// Adam with bias correction and decoupled weight decay (Loshchilov & Hutter).
    AdamW,
    /// Muon on the hidden weight matrices (see `muon_eligible`), AdamW on everything else.
    Muon,
}

/// Newton–Schulz coefficients (Keller Jordan's quintic): five iterations of
/// X ← a·X + (b·A + c·A²)·X with A = X·Xᵀ take X's singular values close to 1.
pub const NS: (F, F, F) = (3.4445, -4.7750, 2.0315);
pub const NS_STEPS: usize = 5;
/// Muon's (Nesterov) momentum.
pub const MUON_MOMENTUM: F = 0.95;

/// Whether Muon updates this parameter: the weight matrices inside the blocks (decayed, and
/// named L<layer>.…), not the embeddings, the output matrix, biases or gains.
pub fn muon_eligible(p: &Param) -> bool {
    p.decay && p.value.rows > 1 && p.value.cols > 1 && p.name.starts_with('L') && p.name.contains('.')
}

/// Muon's update scale for a rows × cols matrix (Moonlight): the orthogonalized update has
/// RMS 1/√max(rows, cols), so this gives it RMS 0.2, about AdamW's, and AdamW's learning
/// rate and weight decay carry over.
pub fn muon_scale(rows: usize, cols: usize) -> F {
    0.2 * (rows.max(cols) as F).sqrt()
}

/// The orthogonalized version of the rows × cols matrix `x` (row-major): x/‖x‖ through
/// `NS_STEPS` Newton–Schulz iterations, done on the wide orientation (transposed when
/// rows > cols, so A = X·Xᵀ is the smaller square).
pub fn newton_schulz(x: &[F], rows: usize, cols: usize) -> Vec<F> {
    let tall = rows > cols;
    let (r, c) = if tall { (cols, rows) } else { (rows, cols) };
    let mut a_x: Vec<F> = if tall { (0..r * c).map(|k| x[(k % c) * cols + k / c]).collect() } else { x.to_vec() };
    let norm = a_x.iter().map(|v| v * v).sum::<F>().sqrt() + 1e-7;
    a_x.iter_mut().for_each(|v| *v /= norm);
    let matmul = |p: &[F], q: &[F], n: usize, k: usize, m: usize, q_t: bool| -> Vec<F> {
        // (n × k) · (k × m), or · qᵀ when q is m × k
        let mut out = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                out[i * m + j] = (0..k).map(|t| p[i * k + t] * if q_t { q[j * k + t] } else { q[t * m + j] }).sum();
            }
        }
        out
    };
    let (ca, cb, cc) = NS;
    for _ in 0..NS_STEPS {
        let a = matmul(&a_x, &a_x, r, c, r, true);
        let aa = matmul(&a, &a, r, r, r, false);
        let b: Vec<F> = a.iter().zip(&aa).map(|(a, aa)| cb * a + cc * aa).collect();
        let bx = matmul(&b, &a_x, r, r, c, false);
        a_x = a_x.iter().zip(&bx).map(|(x, bx)| ca * x + bx).collect();
    }
    if tall { (0..rows * cols).map(|k| a_x[(k % cols) * rows + k / cols]).collect() } else { a_x }
}

pub struct Optimizer {
    pub kind: Kind,
    pub lr: F,
    pub beta1: F,
    pub beta2: F,
    pub eps: F,
    pub weight_decay: F,
    /// Optimizer steps taken so far. Incremented once per step, before the update (Adam's t).
    pub t: u64,
    /// Muon only: its learning rate (AdamW's is `lr`; both follow the same schedule) and,
    /// per parameter, whether Muon updates it (see `muon_eligible`).
    pub muon_lr: F,
    pub muon: Vec<bool>,
}

impl Optimizer {
    pub fn sgd(lr: F) -> Self {
        Optimizer { kind: Kind::Sgd, lr, beta1: 0.0, beta2: 0.0, eps: 0.0, weight_decay: 0.0, t: 0, muon_lr: 0.0, muon: Vec::new() }
    }

    pub fn adamw(lr: F, weight_decay: F) -> Self {
        Optimizer { kind: Kind::AdamW, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay, t: 0, muon_lr: 0.0, muon: Vec::new() }
    }

    /// Muon (learning rate `muon_lr`) for the parameters of `params` that `muon_eligible`
    /// picks, AdamW (`lr`) for the rest; weight decay `weight_decay` for both.
    pub fn muon(lr: F, muon_lr: F, weight_decay: F, params: &Params) -> Self {
        let muon = params.list.iter().map(muon_eligible).collect();
        Optimizer { kind: Kind::Muon, muon_lr, muon, ..Self::adamw(lr, weight_decay) }
    }

    /// Whether parameter i gets Muon's update.
    pub fn is_muon(&self, i: usize) -> bool {
        self.kind == Kind::Muon && self.muon[i]
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            Kind::Sgd => "sgd",
            Kind::AdamW => "adamw",
            Kind::Muon => "muon",
        }
    }

    /// Counts a step (Adam's t) and returns its bias corrections (1−β1^t, 1−β2^t).
    pub fn begin_step(&mut self) -> (F, F) {
        self.t += 1;
        let t = i32::try_from(self.t).expect("step count overflow");
        (1.0 - self.beta1.powi(t), 1.0 - self.beta2.powi(t))
    }

    /// One step over every parameter. `grads[i]` belongs to `params.list[i]`.
    /// Returns the applied change per parameter: after = before + update.
    ///
    /// AdamW, per element, with p the value before this step:
    ///   m ← β1·m + (1−β1)·g          v ← β2·v + (1−β2)·g²
    ///   m̂ = m / (1−β1^t)              v̂ = v / (1−β2^t)
    ///   update = −lr·( m̂ / (√v̂ + ε)  +  wd·p )     (wd only where `decay`; it never enters m or v)
    ///
    /// Muon, on its matrices (m is the momentum buffer, v unused), with p the value before:
    ///   m ← μ·m + g     O = newton_schulz(g + μ·m)
    ///   update = −muon_lr·( muon_scale·O + wd·p )
    pub fn step(&mut self, params: &mut Params, grads: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(grads.len(), params.list.len(), "optimizer: {} gradients for {} parameters", grads.len(), params.list.len());
        let (bc1, bc2) = self.begin_step();
        let muon: Vec<bool> = (0..grads.len()).map(|i| self.is_muon(i)).collect();
        params
            .list
            .iter_mut()
            .zip(grads)
            .zip(muon)
            .map(|((p, g), muon)| {
                if muon {
                    let x: Vec<F> = (0..g.len())
                        .map(|i| {
                            p.m.data[i] = MUON_MOMENTUM * p.m.data[i] + g.data[i];
                            g.data[i] + MUON_MOMENTUM * p.m.data[i]
                        })
                        .collect();
                    let o = newton_schulz(&x, g.rows, g.cols);
                    let s = muon_scale(g.rows, g.cols);
                    let mut update = Tensor::zeros(g.rows, g.cols);
                    for i in 0..g.len() {
                        let before = p.value.data[i];
                        let u = -self.muon_lr * (s * o[i] + self.weight_decay * before);
                        update.data[i] = u;
                        p.value.data[i] = before + u;
                    }
                    return update;
                }
                assert!(p.value.same_shape(g), "optimizer: {} is {}x{} but its gradient is {}x{}", p.name, p.value.rows, p.value.cols, g.rows, g.cols);
                let mut update = Tensor::zeros(g.rows, g.cols);
                for i in 0..g.len() {
                    let (gi, before) = (g.data[i], p.value.data[i]);
                    let u = match self.kind {
                        Kind::Sgd => -self.lr * gi,
                        Kind::AdamW | Kind::Muon => {
                            let m = self.beta1 * p.m.data[i] + (1.0 - self.beta1) * gi;
                            let v = self.beta2 * p.v.data[i] + (1.0 - self.beta2) * gi * gi;
                            p.m.data[i] = m;
                            p.v.data[i] = v;
                            let (m_hat, v_hat) = (m / bc1, v / bc2);
                            let decay = if p.decay { self.weight_decay * before } else { 0.0 };
                            -self.lr * (m_hat / (v_hat.sqrt() + self.eps) + decay)
                        }
                    };
                    update.data[i] = u;
                    p.value.data[i] = before + u;
                }
                update
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: F, b: F) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    /// Expected values computed independently (plain Python, same formulas) for two
    /// steps, so bias correction, β, t and decoupled decay all show up.
    #[test]
    fn adamw_two_steps_match_reference() {
        let mut ps = Params::default();
        ps.add("A", Tensor::from_vec(1, 2, vec![1.0, -2.0]), true); // decay, negative grad, zero grad on step 2
        ps.add("B", Tensor::scalar(3.0), false); // no decay, gradient flips sign
        ps.add("Z", Tensor::scalar(2.0), true); // decay with zero gradient: only decay moves it
        let mut opt = Optimizer::adamw(0.1, 0.1);
        let g1 = [Tensor::from_vec(1, 2, vec![0.5, -0.25]), Tensor::scalar(1.0), Tensor::scalar(0.0)];
        let g2 = [Tensor::from_vec(1, 2, vec![-0.1, 0.0]), Tensor::scalar(-2.0), Tensor::scalar(0.0)];

        let before: Vec<Vec<F>> = ps.list.iter().map(|p| p.value.data.clone()).collect();
        let upd = opt.step(&mut ps, &g1);
        let after1 = [vec![0.890000002, -1.8800000039999998], vec![2.900000001], vec![1.98]];
        for (i, p) in ps.list.iter().enumerate() {
            for (e, &want) in after1[i].iter().enumerate() {
                assert!(close(p.value.data[e], want), "step 1 {}[{e}] = {} want {want}", p.name, p.value.data[e]);
                // The returned update is the change that was actually applied.
                assert!(close(before[i][e] + upd[i].data[e], p.value.data[e]));
            }
        }

        opt.step(&mut ps, &g2);
        assert_eq!(opt.t, 2, "one t increment per step, not per parameter");
        let after2 = [vec![0.8299973956793238, -1.794194182337716], vec![2.9366103534720747], vec![1.9602]];
        let m2 = [vec![0.03499999999999999, -0.022499999999999996], vec![-0.10999999999999997], vec![0.0]];
        let v2 = [vec![0.00025975000000000027, 6.243750000000006e-05], vec![0.004999000000000004], vec![0.0]];
        for (i, p) in ps.list.iter().enumerate() {
            for e in 0..p.value.len() {
                assert!(close(p.value.data[e], after2[i][e]), "step 2 {}[{e}] = {} want {}", p.name, p.value.data[e], after2[i][e]);
                assert!(close(p.m.data[e], m2[i][e]), "m {}[{e}]", p.name);
                assert!(close(p.v.data[e], v2[i][e]), "v {}[{e}]", p.name);
            }
        }
    }

    /// Newton–Schulz leaves singular values near 1 (between about 0.6 and 1.2 for the
    /// quintic), in either orientation.
    #[test]
    fn newton_schulz_orthogonalizes() {
        let mut rng = crate::rng::Rng::new(4);
        for (r, c) in [(6, 10), (10, 6)] {
            let x: Vec<F> = (0..r * c).map(|_| rng.normal()).collect();
            let o = newton_schulz(&x, r, c);
            // Gram matrix on the small side: its eigenvalues are the squared singular values,
            // so its diagonal sits in that range too and its trace is near min(r, c).
            let k = r.min(c);
            let gram = |i: usize, j: usize| -> F {
                if r <= c { (0..c).map(|t| o[i * c + t] * o[j * c + t]).sum() } else { (0..r).map(|t| o[t * c + i] * o[t * c + j]).sum() }
            };
            for i in 0..k {
                let d = gram(i, i);
                assert!((0.3..1.5).contains(&d), "{r}x{c}: diagonal {d}");
            }
            let tr: F = (0..k).map(|i| gram(i, i)).sum();
            assert!((tr / k as F - 0.8).abs() < 0.35, "{r}x{c}: mean squared singular value {}", tr / k as F);
        }
    }

    #[test]
    fn sgd_moves_against_gradient() {
        let mut ps = Params::default();
        ps.add("w", Tensor::from_vec(1, 2, vec![1.0, 1.0]), true);
        let upd = Optimizer::sgd(0.5).step(&mut ps, &[Tensor::from_vec(1, 2, vec![2.0, -4.0])]);
        assert_eq!(ps.list[0].value.data, vec![0.0, 3.0]);
        assert_eq!(upd[0].data, vec![-1.0, 2.0]);
    }
}
