//! Reverse-mode automatic differentiation on a tape.
//!
//! Every operation appends a node to `Graph::nodes`, so nodes are already in
//! topological order: a node's inputs always have smaller indices. `backward`
//! walks the tape from the loss back to index 0 and *accumulates* (`+=`) each
//! node's gradient into its inputs, which handles values used more than once.

use crate::par;
use crate::tensor::{softmax_rows, Tensor, F};

/// Handle to a node in the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

#[derive(Clone, Debug)]
pub enum Op {
    /// Input data or a parameter; nothing to differentiate through.
    Leaf,
    /// Elementwise a + b (same shape).
    Add(Var, Var),
    /// Elementwise a * b (same shape).
    Mul(Var, Var),
    /// Matrix product a (n x k) @ b (k x m).
    MatMul(Var, Var),
    /// a (n x m) plus a row vector b (1 x m) added to every row.
    AddRow(Var, Var),
    /// Elementwise tanh.
    Tanh(Var),
    /// Pick rows `ids` out of `table`: out[r] = table[ids[r]].
    Embedding { table: Var, ids: Vec<usize> },
    /// Mean over rows of -log softmax(logits)[target]. Output is 1x1.
    CrossEntropy { logits: Var, targets: Vec<usize> },
    /// Mean of all elements. Output is 1x1.
    Mean(Var),
    /// Same numbers, new shape (row-major order is unchanged). Used to concatenate
    /// a context's embeddings into one row.
    Reshape(Var),
    /// Elementwise a · by, for a fixed number `by` (e.g. 1/√d_head).
    Scale { a: Var, by: F },
    /// a (n x m) times a row vector g (1 x m), elementwise in every row (a layernorm gain).
    MulRow(Var, Var),
    /// Elementwise GELU, tanh approximation: x/2 · (1 + tanh(√(2/π)·(x + 0.044715·x³))).
    Gelu(Var),
    Relu2(Var),
    /// Row-wise (x − mean) · rstd with rstd = 1/√(var + eps), var the population variance.
    /// The core of layernorm; each row's mean and rstd are kept for the trace and backward.
    Normalize { a: Var, eps: F, mean: Vec<F>, rstd: Vec<F> },
    /// (n·T) x (H·dh) → (n·H·T) x dh: rows of example b, head h are the T positions'
    /// columns h·dh .. (h+1)·dh. A pure reordering of the same numbers.
    SplitHeads { a: Var, n: usize, t: usize, heads: usize },
    /// The exact inverse of SplitHeads: (n·H·T) x dh → (n·T) x (H·dh).
    MergeHeads { a: Var, n: usize, t: usize, heads: usize },
    /// `batches` independent products stacked by rows: block i of the output is
    /// A_i · B_i, or A_i · B_iᵀ when `trans_b` (used for Q·Kᵀ).
    BatchMatMul { a: Var, b: Var, batches: usize, trans_b: bool },
    /// Row-wise softmax with a causal mask: row r is position r % t, which may only see
    /// columns 0 ..= r % t. Masked entries are never set to −∞; their probability is
    /// exactly 0 and they get exactly 0 gradient.
    CausalSoftmax { a: Var, t: usize },
}

impl Op {
    pub fn kind(&self) -> &'static str {
        match self {
            Op::Leaf => "leaf",
            Op::Add(..) => "add",
            Op::Mul(..) => "mul",
            Op::MatMul(..) => "matmul",
            Op::AddRow(..) => "add_row",
            Op::Tanh(..) => "tanh",
            Op::Embedding { .. } => "embedding",
            Op::CrossEntropy { .. } => "cross_entropy",
            Op::Mean(..) => "mean",
            Op::Reshape(..) => "reshape",
            Op::Scale { .. } => "scale",
            Op::MulRow(..) => "mul_row",
            Op::Gelu(..) => "gelu",
            Op::Relu2(..) => "relu²",
            Op::Normalize { .. } => "normalize",
            Op::SplitHeads { .. } => "split_heads",
            Op::MergeHeads { .. } => "merge_heads",
            Op::BatchMatMul { .. } => "batch_matmul",
            Op::CausalSoftmax { .. } => "causal_softmax",
        }
    }

    pub fn inputs(&self) -> Vec<Var> {
        match self {
            Op::Leaf => vec![],
            Op::Add(a, b) | Op::Mul(a, b) | Op::MatMul(a, b) | Op::AddRow(a, b) | Op::MulRow(a, b) => vec![*a, *b],
            Op::BatchMatMul { a, b, .. } => vec![*a, *b],
            Op::Tanh(a) | Op::Mean(a) | Op::Reshape(a) | Op::Gelu(a) | Op::Relu2(a) => vec![*a],
            Op::Scale { a, .. } | Op::Normalize { a, .. } | Op::SplitHeads { a, .. } | Op::MergeHeads { a, .. } | Op::CausalSoftmax { a, .. } => vec![*a],
            Op::Embedding { table, .. } => vec![*table],
            Op::CrossEntropy { logits, .. } => vec![*logits],
        }
    }
}

pub struct Node {
    pub name: String,
    pub op: Op,
    pub value: Tensor,
    pub grad: Tensor,
    /// True for parameters and anything computed from them.
    pub requires_grad: bool,
    /// True for trainable parameters (leaves the optimizer updates).
    pub is_param: bool,
}

#[derive(Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// When set (by `microllm profile`), seconds spent per node: forward (time since the
    /// previous node was added) and backward.
    pub profile: Option<(std::time::Instant, Vec<f64>, Vec<f64>)>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new(), profile: None }
    }

    fn push(&mut self, name: &str, op: Op, value: Tensor) -> Var {
        if let Some((t, fwd, bwd)) = &mut self.profile {
            fwd.push(t.elapsed().as_secs_f64());
            bwd.push(0.0);
            *t = std::time::Instant::now();
        }
        let requires_grad = op.inputs().iter().any(|v| self.nodes[v.0].requires_grad);
        let grad = Tensor::zeros(value.rows, value.cols);
        self.nodes.push(Node { name: name.to_string(), op, value, grad, requires_grad, is_param: false });
        Var(self.nodes.len() - 1)
    }

    /// Fixed data (inputs, targets, constants). No gradient flows into it.
    pub fn input(&mut self, name: &str, value: Tensor) -> Var {
        self.push(name, Op::Leaf, value)
    }

    /// A trainable parameter.
    pub fn param(&mut self, name: &str, value: Tensor) -> Var {
        let v = self.push(name, Op::Leaf, value);
        self.nodes[v.0].requires_grad = true;
        self.nodes[v.0].is_param = true;
        v
    }

    pub fn value(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].value
    }

    pub fn grad(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].grad
    }

    pub fn add(&mut self, name: &str, a: Var, b: Var) -> Var {
        let (x, y) = (self.value(a), self.value(b));
        assert!(x.same_shape(y), "add: shape mismatch {}x{} vs {}x{}", x.rows, x.cols, y.rows, y.cols);
        let data = x.data.iter().zip(&y.data).map(|(p, q)| p + q).collect();
        let out = Tensor::from_vec(x.rows, x.cols, data);
        self.push(name, Op::Add(a, b), out)
    }

    pub fn mul(&mut self, name: &str, a: Var, b: Var) -> Var {
        let (x, y) = (self.value(a), self.value(b));
        assert!(x.same_shape(y), "mul: shape mismatch {}x{} vs {}x{}", x.rows, x.cols, y.rows, y.cols);
        let data = x.data.iter().zip(&y.data).map(|(p, q)| p * q).collect();
        let out = Tensor::from_vec(x.rows, x.cols, data);
        self.push(name, Op::Mul(a, b), out)
    }

    pub fn matmul(&mut self, name: &str, a: Var, b: Var) -> Var {
        let out = matmul(self.value(a), self.value(b));
        self.push(name, Op::MatMul(a, b), out)
    }

    pub fn add_row(&mut self, name: &str, a: Var, bias: Var) -> Var {
        let (x, b) = (self.value(a), self.value(bias));
        assert!(b.rows == 1 && b.cols == x.cols, "add_row: bias must be 1x{}, got {}x{}", x.cols, b.rows, b.cols);
        let mut out = x.clone();
        for r in 0..out.rows {
            for (o, bv) in out.row_mut(r).iter_mut().zip(&b.data) {
                *o += bv;
            }
        }
        self.push(name, Op::AddRow(a, bias), out)
    }

    pub fn tanh(&mut self, name: &str, a: Var) -> Var {
        let x = self.value(a);
        let out = Tensor::from_vec(x.rows, x.cols, x.data.iter().map(|v| v.tanh()).collect());
        self.push(name, Op::Tanh(a), out)
    }

    pub fn embedding(&mut self, name: &str, table: Var, ids: &[usize]) -> Var {
        let t = self.value(table);
        let mut out = Tensor::zeros(ids.len(), t.cols);
        for (r, &id) in ids.iter().enumerate() {
            assert!(id < t.rows, "embedding: id {id} out of range for table with {} rows", t.rows);
            out.row_mut(r).copy_from_slice(t.row(id));
        }
        self.push(name, Op::Embedding { table, ids: ids.to_vec() }, out)
    }

    pub fn cross_entropy(&mut self, name: &str, logits: Var, targets: &[usize]) -> Var {
        let x = self.value(logits);
        assert_eq!(x.rows, targets.len(), "cross_entropy: {} rows but {} targets", x.rows, targets.len());
        let loss = per_row_cross_entropy(x, targets).iter().sum::<F>() / x.rows as F;
        self.push(name, Op::CrossEntropy { logits, targets: targets.to_vec() }, Tensor::scalar(loss))
    }

    pub fn mean(&mut self, name: &str, a: Var) -> Var {
        let x = self.value(a);
        let m = x.data.iter().sum::<F>() / x.len() as F;
        self.push(name, Op::Mean(a), Tensor::scalar(m))
    }

    pub fn reshape(&mut self, name: &str, a: Var, rows: usize, cols: usize) -> Var {
        let x = self.value(a);
        assert_eq!(x.len(), rows * cols, "reshape: {}x{} has {} elements, cannot become {rows}x{cols}", x.rows, x.cols, x.len());
        let out = Tensor::from_vec(rows, cols, x.data.clone());
        self.push(name, Op::Reshape(a), out)
    }

    pub fn scale(&mut self, name: &str, a: Var, by: F) -> Var {
        let x = self.value(a);
        let out = Tensor::from_vec(x.rows, x.cols, x.data.iter().map(|v| v * by).collect());
        self.push(name, Op::Scale { a, by }, out)
    }

    pub fn mul_row(&mut self, name: &str, a: Var, gain: Var) -> Var {
        let (x, gv) = (self.value(a), self.value(gain));
        assert!(gv.rows == 1 && gv.cols == x.cols, "mul_row: gain must be 1x{}, got {}x{}", x.cols, gv.rows, gv.cols);
        let mut out = x.clone();
        for r in 0..out.rows {
            for (o, g) in out.row_mut(r).iter_mut().zip(&gv.data) {
                *o *= g;
            }
        }
        self.push(name, Op::MulRow(a, gain), out)
    }

    pub fn gelu(&mut self, name: &str, a: Var) -> Var {
        let x = self.value(a);
        let mut out = Tensor::zeros(x.rows, x.cols);
        par::chunks(&mut out.data, x.rows, x.len() * 8, |r, o| o.iter_mut().zip(x.row(r)).for_each(|(o, &v)| *o = gelu(v).0));
        self.push(name, Op::Gelu(a), out)
    }

    /// relu(x)² elementwise.
    pub fn relu2(&mut self, name: &str, a: Var) -> Var {
        let x = self.value(a);
        let out = Tensor::from_vec(x.rows, x.cols, x.data.iter().map(|&v| relu2(v).0).collect());
        self.push(name, Op::Relu2(a), out)
    }

    pub fn normalize(&mut self, name: &str, a: Var, eps: F) -> Var {
        let x = self.value(a);
        let (mut out, mut mean, mut rstd) = (Tensor::zeros(x.rows, x.cols), Vec::with_capacity(x.rows), Vec::with_capacity(x.rows));
        for r in 0..x.rows {
            let row = x.row(r);
            let m = row.iter().sum::<F>() / x.cols as F;
            let var = row.iter().map(|v| (v - m) * (v - m)).sum::<F>() / x.cols as F;
            let s = 1.0 / (var + eps).sqrt();
            for (o, v) in out.row_mut(r).iter_mut().zip(row) {
                *o = (v - m) * s;
            }
            mean.push(m);
            rstd.push(s);
        }
        self.push(name, Op::Normalize { a, eps, mean, rstd }, out)
    }

    pub fn split_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        let x = self.value(a);
        assert!(x.rows == n * t && heads > 0 && x.cols % heads == 0, "split_heads: {}x{} is not ({n}·{t}) x ({heads}·dh)", x.rows, x.cols);
        let map = head_map(n, t, heads, x.cols / heads);
        let out = Tensor::from_vec(n * heads * t, x.cols / heads, map.iter().map(|&i| x.data[i]).collect());
        self.push(name, Op::SplitHeads { a, n, t, heads }, out)
    }

    pub fn merge_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        let x = self.value(a);
        assert_eq!(x.rows, n * heads * t, "merge_heads: {} rows is not {n}·{heads}·{t}", x.rows);
        let map = head_map(n, t, heads, x.cols);
        let mut out = Tensor::zeros(n * t, heads * x.cols);
        for (k, &i) in map.iter().enumerate() {
            out.data[i] = x.data[k];
        }
        self.push(name, Op::MergeHeads { a, n, t, heads }, out)
    }

    pub fn batch_matmul(&mut self, name: &str, a: Var, b: Var, batches: usize, trans_b: bool) -> Var {
        let (x, y) = (self.value(a), self.value(b));
        assert!(batches > 0 && x.rows % batches == 0 && y.rows % batches == 0, "batch_matmul: {} and {} rows do not split into {batches} blocks", x.rows, y.rows);
        let (r, k) = (x.rows / batches, x.cols);
        let (inner, c) = if trans_b { (y.cols, y.rows / batches) } else { (y.rows / batches, y.cols) };
        assert_eq!(k, inner, "batch_matmul: blocks {r}x{k} and {}x{}{} do not multiply", y.rows / batches, y.cols, if trans_b { "ᵀ" } else { "" });
        let mut out = Tensor::zeros(x.rows, c);
        let yb = y.rows / batches * y.cols;
        // one block per call: blocks are independent, so threads cannot change any number
        par::chunks(&mut out.data, batches, batches * r * k * c, |i, oi| {
            let (ai, bi) = (&x.data[i * r * k..(i + 1) * r * k], &y.data[i * yb..(i + 1) * yb]);
            if trans_b { mm_bt(ai, bi, oi, r, k, c) } else { mm(ai, bi, oi, r, k, c) }
        });
        self.push(name, Op::BatchMatMul { a, b, batches, trans_b }, out)
    }

    pub fn causal_softmax(&mut self, name: &str, a: Var, t: usize) -> Var {
        let x = self.value(a);
        assert!(t > 0, "causal_softmax: block size must be positive");
        assert!(x.cols == t && x.rows % t == 0, "causal_softmax: {}x{} is not a stack of {t}x{t} blocks", x.rows, x.cols);
        let mut out = Tensor::zeros(x.rows, x.cols);
        for r in 0..x.rows {
            let seen = &x.row(r)[..=r % t];
            let max = seen.iter().cloned().fold(F::NEG_INFINITY, F::max);
            let sum: F = seen.iter().map(|v| (v - max).exp()).sum();
            for (o, v) in out.row_mut(r).iter_mut().zip(seen) {
                *o = (v - max).exp() / sum;
            }
        }
        self.push(name, Op::CausalSoftmax { a, t }, out)
    }

    /// Fill every node's `grad` with d(loss)/d(node).
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(self.value(loss).len(), 1, "backward needs a scalar (1x1) loss");
        for node in &mut self.nodes {
            node.grad.data.iter_mut().for_each(|g| *g = 0.0);
        }
        self.nodes[loss.0].grad.data[0] = 1.0;

        for i in (0..=loss.0).rev() {
            let t0 = self.profile.is_some().then(std::time::Instant::now);
            self.backward_node(i);
            if let (Some(t0), Some((_, _, bwd))) = (t0, &mut self.profile) {
                bwd[i] += t0.elapsed().as_secs_f64();
            }
        }
    }

    fn backward_node(&mut self, i: usize) {
        {
            // Inputs always come before node i, so split the tape to borrow both.
            let (before, rest) = self.nodes.split_at_mut(i);
            let node = &rest[0];
            if !node.requires_grad {
                return;
            }
            let g = &node.grad;
            match &node.op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    for x in [a, b] {
                        if before[x.0].requires_grad {
                            accumulate(&mut before[x.0].grad, &g.data);
                        }
                    }
                }
                Op::Mul(a, b) => {
                    // d(a*b)/da = b, d(a*b)/db = a. Compute both before writing, in case a == b.
                    let da: Vec<F> = g.data.iter().zip(&before[b.0].value.data).map(|(g, y)| g * y).collect();
                    let db: Vec<F> = g.data.iter().zip(&before[a.0].value.data).map(|(g, x)| g * x).collect();
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &da);
                    }
                    if before[b.0].requires_grad {
                        accumulate(&mut before[b.0].grad, &db);
                    }
                }
                Op::MatMul(a, b) => {
                    // out = A @ B  =>  dA = G @ B^T,  dB = A^T @ G
                    let (av, bv) = (&before[a.0].value, &before[b.0].value);
                    let da = matmul(g, &transpose(bv));
                    let db = matmul_at(av, g);
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &da.data);
                    }
                    if before[b.0].requires_grad {
                        accumulate(&mut before[b.0].grad, &db.data);
                    }
                }
                Op::AddRow(a, bias) => {
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &g.data);
                    }
                    if before[bias.0].requires_grad {
                        // The bias was added to every row, so its gradient sums over rows.
                        let bg = &mut before[bias.0].grad;
                        for r in 0..g.rows {
                            for (acc, gv) in bg.data.iter_mut().zip(g.row(r)) {
                                *acc += gv;
                            }
                        }
                    }
                }
                Op::Tanh(a) => {
                    if before[a.0].requires_grad {
                        // d tanh(x)/dx = 1 - tanh(x)^2, and node.value is tanh(x).
                        let da: Vec<F> = g.data.iter().zip(&node.value.data).map(|(g, y)| g * (1.0 - y * y)).collect();
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
                Op::Embedding { table, ids } => {
                    if before[table.0].requires_grad {
                        // Scatter-add: the same id can appear several times in a batch.
                        let tg = &mut before[table.0].grad;
                        for (r, &id) in ids.iter().enumerate() {
                            for (acc, gv) in tg.row_mut(id).iter_mut().zip(g.row(r)) {
                                *acc += gv;
                            }
                        }
                    }
                }
                Op::CrossEntropy { logits, targets } => {
                    if before[logits.0].requires_grad {
                        // d loss / d logits[r][j] = (softmax[r][j] - [j == target_r]) / rows
                        let lv = &before[logits.0].value;
                        let mut dl = softmax_rows(lv);
                        let scale = g.item() / lv.rows as F;
                        for (r, &t) in targets.iter().enumerate() {
                            dl.row_mut(r)[t] -= 1.0;
                            dl.row_mut(r).iter_mut().for_each(|v| *v *= scale);
                        }
                        accumulate(&mut before[logits.0].grad, &dl.data);
                    }
                }
                Op::Mean(a) => {
                    if before[a.0].requires_grad {
                        let n = before[a.0].value.len() as F;
                        let share = g.item() / n;
                        before[a.0].grad.data.iter_mut().for_each(|acc| *acc += share);
                    }
                }
                Op::Reshape(a) => {
                    // Element i of the output is element i of the input, so gradients pass straight through.
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &g.data);
                    }
                }
                Op::Scale { a, by } => {
                    if before[a.0].requires_grad {
                        let da: Vec<F> = g.data.iter().map(|g| g * by).collect();
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
                Op::MulRow(a, gain) => {
                    let (x, gv) = (&before[a.0].value, &before[gain.0].value);
                    let cols = x.cols;
                    // out[r][c] = x[r][c]·gain[c]: dx = g·gain, dgain[c] = Σ_r g[r][c]·x[r][c].
                    let da: Vec<F> = g.data.iter().enumerate().map(|(i, g)| g * gv.data[i % cols]).collect();
                    let mut dg = vec![0.0; cols];
                    for (i, (g, xv)) in g.data.iter().zip(&x.data).enumerate() {
                        dg[i % cols] += g * xv;
                    }
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &da);
                    }
                    if before[gain.0].requires_grad {
                        accumulate(&mut before[gain.0].grad, &dg);
                    }
                }
                Op::Gelu(a) => {
                    if before[a.0].requires_grad {
                        let x = &before[a.0].value;
                        let mut da = vec![0.0; x.len()];
                        let cols = x.cols;
                        par::chunks(&mut da, x.rows, x.len() * 8, |r, o| {
                            o.iter_mut().zip(&g.data[r * cols..(r + 1) * cols]).zip(x.row(r)).for_each(|((o, g), &v)| *o = g * gelu(v).1)
                        });
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
                Op::Relu2(a) => {
                    if before[a.0].requires_grad {
                        let x = &before[a.0].value;
                        let da: Vec<F> = g.data.iter().zip(&x.data).map(|(g, &v)| g * relu2(v).1).collect();
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
                Op::Normalize { a, rstd, .. } => {
                    if before[a.0].requires_grad {
                        // y = (x − mean)·rstd. Per row, with m = cols:
                        //   dx = rstd · (g − mean(g) − y · mean(g·y))
                        let y = &node.value;
                        let m = y.cols as F;
                        let mut da = vec![0.0; y.len()];
                        for r in 0..y.rows {
                            let (gr, yr) = (g.row(r), y.row(r));
                            let gm = gr.iter().sum::<F>() / m;
                            let gym = gr.iter().zip(yr).map(|(g, y)| g * y).sum::<F>() / m;
                            for c in 0..y.cols {
                                da[r * y.cols + c] = rstd[r] * (gr[c] - gm - yr[c] * gym);
                            }
                        }
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
                Op::SplitHeads { a, n, t, heads } => {
                    if before[a.0].requires_grad {
                        // out[k] = x[map[k]], so dx[map[k]] += g[k].
                        let ag = &mut before[a.0].grad;
                        for (k, &i) in head_map(*n, *t, *heads, g.cols).iter().enumerate() {
                            ag.data[i] += g.data[k];
                        }
                    }
                }
                Op::MergeHeads { a, n, t, heads } => {
                    if before[a.0].requires_grad {
                        // out[map[k]] = x[k], so dx[k] += g[map[k]].
                        let ag = &mut before[a.0].grad;
                        for (k, &i) in head_map(*n, *t, *heads, g.cols / heads).iter().enumerate() {
                            ag.data[k] += g.data[i];
                        }
                    }
                }
                Op::BatchMatMul { a, b, batches, trans_b } => {
                    // Per block: out = A·B  =>  dA = G·Bᵀ, dB = Aᵀ·G
                    //            out = A·Bᵀ =>  dA = G·B,  dB = Gᵀ·A
                    let (x, y) = (&before[a.0].value, &before[b.0].value);
                    let (r, k, c) = (x.rows / batches, x.cols, g.cols);
                    let (mut da, mut db) = (vec![0.0; x.len()], vec![0.0; y.len()]);
                    let yb = y.len() / batches;
                    // block i's dA and dB are owned by one call, which sums in the one-thread order
                    par::chunks2(&mut da, &mut db, *batches, 2 * batches * r * k * c, |i, dai, dbi| {
                        let (ai, bi, gi) = (&x.data[i * r * k..(i + 1) * r * k], &y.data[i * yb..(i + 1) * yb], &g.data[i * r * c..(i + 1) * r * c]);
                        if *trans_b {
                            mm(gi, bi, dai, r, c, k);
                            mm_at(gi, ai, dbi, r, c, k);
                        } else {
                            mm_bt(gi, bi, dai, r, c, k);
                            mm_at(ai, gi, dbi, r, k, c);
                        }
                    });
                    if before[a.0].requires_grad {
                        accumulate(&mut before[a.0].grad, &da);
                    }
                    if before[b.0].requires_grad {
                        accumulate(&mut before[b.0].grad, &db);
                    }
                }
                Op::CausalSoftmax { a, t } => {
                    if before[a.0].requires_grad {
                        // dx_j = p_j · (g_j − Σ_k p_k·g_k). Masked p_j are exactly 0, so their dx_j is exactly 0.
                        let p = &node.value;
                        let mut da = vec![0.0; p.len()];
                        for r in 0..p.rows {
                            let (t, seen) = (*t, r % *t + 1);
                            let (pr, gr) = (&p.row(r)[..seen], &g.row(r)[..seen]);
                            let dot: F = pr.iter().zip(gr).map(|(p, g)| p * g).sum();
                            for (j, (p, g)) in pr.iter().zip(gr).enumerate() {
                                da[r * t + j] = p * (g - dot);
                            }
                        }
                        accumulate(&mut before[a.0].grad, &da);
                    }
                }
            }
        }
    }
}

/// GELU (tanh approximation) and its derivative at x.
fn gelu(x: F) -> (F, F) {
    const C: F = 0.797_884_560_802_865_4; // √(2/π)
    let u = C * (x + 0.044715 * x * x * x);
    let th = u.tanh();
    let du = C * (1.0 + 3.0 * 0.044715 * x * x);
    (0.5 * x * (1.0 + th), 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du)
}

/// relu(x)² and its derivative at x.
fn relu2(x: F) -> (F, F) {
    if x > 0.0 { (x * x, 2.0 * x) } else { (0.0, 0.0) }
}

/// What a model needs to build its forward pass. The CPU tape (`Graph`, f64) and the GPU tape
/// (`gpu::Graph`, f32) both implement it, so one model definition runs on either, and node i
/// means the same thing on both (`Var`s are interchangeable).
pub trait Tape {
    fn input(&mut self, name: &str, value: Tensor) -> Var;
    fn param(&mut self, name: &str, value: Tensor) -> Var;
    fn add(&mut self, name: &str, a: Var, b: Var) -> Var;
    fn matmul(&mut self, name: &str, a: Var, b: Var) -> Var;
    fn add_row(&mut self, name: &str, a: Var, bias: Var) -> Var;
    fn tanh(&mut self, name: &str, a: Var) -> Var;
    fn embedding(&mut self, name: &str, table: Var, ids: &[usize]) -> Var;
    fn cross_entropy(&mut self, name: &str, logits: Var, targets: &[usize]) -> Var;
    fn reshape(&mut self, name: &str, a: Var, rows: usize, cols: usize) -> Var;
    fn scale(&mut self, name: &str, a: Var, by: F) -> Var;
    fn mul_row(&mut self, name: &str, a: Var, gain: Var) -> Var;
    fn gelu(&mut self, name: &str, a: Var) -> Var;
    fn relu2(&mut self, name: &str, a: Var) -> Var;
    fn normalize(&mut self, name: &str, a: Var, eps: F) -> Var;
    fn split_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var;
    fn merge_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var;
    fn batch_matmul(&mut self, name: &str, a: Var, b: Var, batches: usize, trans_b: bool) -> Var;
    fn causal_softmax(&mut self, name: &str, a: Var, t: usize) -> Var;
}

impl Tape for Graph {
    fn input(&mut self, name: &str, value: Tensor) -> Var {
        Graph::input(self, name, value)
    }
    fn param(&mut self, name: &str, value: Tensor) -> Var {
        Graph::param(self, name, value)
    }
    fn add(&mut self, name: &str, a: Var, b: Var) -> Var {
        Graph::add(self, name, a, b)
    }
    fn matmul(&mut self, name: &str, a: Var, b: Var) -> Var {
        Graph::matmul(self, name, a, b)
    }
    fn add_row(&mut self, name: &str, a: Var, bias: Var) -> Var {
        Graph::add_row(self, name, a, bias)
    }
    fn tanh(&mut self, name: &str, a: Var) -> Var {
        Graph::tanh(self, name, a)
    }
    fn embedding(&mut self, name: &str, table: Var, ids: &[usize]) -> Var {
        Graph::embedding(self, name, table, ids)
    }
    fn cross_entropy(&mut self, name: &str, logits: Var, targets: &[usize]) -> Var {
        Graph::cross_entropy(self, name, logits, targets)
    }
    fn reshape(&mut self, name: &str, a: Var, rows: usize, cols: usize) -> Var {
        Graph::reshape(self, name, a, rows, cols)
    }
    fn scale(&mut self, name: &str, a: Var, by: F) -> Var {
        Graph::scale(self, name, a, by)
    }
    fn mul_row(&mut self, name: &str, a: Var, gain: Var) -> Var {
        Graph::mul_row(self, name, a, gain)
    }
    fn gelu(&mut self, name: &str, a: Var) -> Var {
        Graph::gelu(self, name, a)
    }
    fn relu2(&mut self, name: &str, a: Var) -> Var {
        Graph::relu2(self, name, a)
    }
    fn normalize(&mut self, name: &str, a: Var, eps: F) -> Var {
        Graph::normalize(self, name, a, eps)
    }
    fn split_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        Graph::split_heads(self, name, a, n, t, heads)
    }
    fn merge_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        Graph::merge_heads(self, name, a, n, t, heads)
    }
    fn batch_matmul(&mut self, name: &str, a: Var, b: Var, batches: usize, trans_b: bool) -> Var {
        Graph::batch_matmul(self, name, a, b, batches, trans_b)
    }
    fn causal_softmax(&mut self, name: &str, a: Var, t: usize) -> Var {
        Graph::causal_softmax(self, name, a, t)
    }
}

/// For split_heads: element k of the (n·H·T) x dh output comes from element map[k] of the
/// (n·T) x (H·dh) input. Output row (b·H + h)·T + p, column j  ←  input row b·T + p, column h·dh + j.
pub fn head_map(n: usize, t: usize, heads: usize, dh: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(n * heads * t * dh);
    for b in 0..n {
        for h in 0..heads {
            for p in 0..t {
                for j in 0..dh {
                    map.push((b * t + p) * heads * dh + h * dh + j);
                }
            }
        }
    }
    map
}

/// out (r x c) += a (r x k) · b (k x c), on row-major slices.
fn mm(a: &[F], b: &[F], out: &mut [F], r: usize, k: usize, c: usize) {
    for i in 0..r {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for (o, bv) in out[i * c..(i + 1) * c].iter_mut().zip(&b[kk * c..(kk + 1) * c]) {
                *o += aik * bv;
            }
        }
    }
}

/// out (r x c) += a (r x k) · bᵀ, with b stored as c x k.
fn mm_bt(a: &[F], b: &[F], out: &mut [F], r: usize, k: usize, c: usize) {
    for i in 0..r {
        let ar = &a[i * k..(i + 1) * k];
        for j in 0..c {
            out[i * c + j] += ar.iter().zip(&b[j * k..(j + 1) * k]).map(|(x, y)| x * y).sum::<F>();
        }
    }
}

/// out (k x c) += aᵀ · b, with a stored as r x k and b as r x c.
fn mm_at(a: &[F], b: &[F], out: &mut [F], r: usize, k: usize, c: usize) {
    for i in 0..r {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for (o, bv) in out[kk * c..(kk + 1) * c].iter_mut().zip(&b[i * c..(i + 1) * c]) {
                *o += aik * bv;
            }
        }
    }
}

fn accumulate(grad: &mut Tensor, delta: &[F]) {
    debug_assert_eq!(grad.len(), delta.len());
    for (acc, d) in grad.data.iter_mut().zip(delta) {
        *acc += d;
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.cols, b.rows, "matmul: {}x{} @ {}x{}", a.rows, a.cols, b.rows, b.cols);
    let mut out = Tensor::zeros(a.rows, b.cols);
    // Groups of up to 4 output rows per call, each element summed over k in order: the same
    // numbers as the plain loop, on any thread count. A group reads each row of b once for
    // all its rows, which is most of the speed-up.
    let (k, c) = (a.cols, b.cols);
    let group = (1..=4).rev().find(|d| a.rows % d == 0).unwrap_or(1);
    par::chunks(&mut out.data, a.rows / group.max(1), a.rows * k * c, |gi, o| {
        let i0 = gi * group;
        if group == 4 {
            let (o0, rest) = o.split_at_mut(c);
            let (o1, rest) = rest.split_at_mut(c);
            let (o2, o3) = rest.split_at_mut(c);
            let (r0, r1, r2, r3) = (a.row(i0), a.row(i0 + 1), a.row(i0 + 2), a.row(i0 + 3));
            for kk in 0..k {
                let br = &b.data[kk * c..(kk + 1) * c];
                let (x0, x1, x2, x3) = (r0[kk], r1[kk], r2[kk], r3[kk]);
                for j in 0..c {
                    let bv = br[j];
                    o0[j] += x0 * bv;
                    o1[j] += x1 * bv;
                    o2[j] += x2 * bv;
                    o3[j] += x3 * bv;
                }
            }
        } else {
            mm(&a.data[i0 * k..(i0 + group) * k], &b.data, o, group, k, c);
        }
    });
    out
}

/// aᵀ · b without forming aᵀ: the same sums, in the same order (over the rows of a and
/// b, first to last), as matmul(&transpose(a), b), so the numbers are identical. Reading
/// a and b once per thread instead of b once per output row keeps it out of main memory.
pub fn matmul_at(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.rows, b.rows, "matmul_at: {}x{}ᵀ @ {}x{}", a.rows, a.cols, b.rows, b.cols);
    let (r, k, c) = (a.rows, a.cols, b.cols);
    let mut out = Tensor::zeros(k, c);
    // Each call owns a band of output rows (columns of a): the largest band of at most 16
    // rows that divides k, so every band is the same size.
    let band = (1..=16).rev().find(|d| k % d == 0).unwrap_or(1);
    par::chunks(&mut out.data, k / band.max(1), r * k * c, |bi, o| {
        let k0 = bi * band;
        for i in 0..r {
            let br = &b.data[i * c..(i + 1) * c];
            for kk in 0..band {
                let aik = a.data[i * k + k0 + kk];
                for (ov, bv) in o[kk * c..(kk + 1) * c].iter_mut().zip(br) {
                    *ov += aik * bv;
                }
            }
        }
    });
    out
}

pub fn transpose(a: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(a.cols, a.rows);
    for r in 0..a.rows {
        for c in 0..a.cols {
            out.data[c * a.rows + r] = a.at(r, c);
        }
    }
    out
}

/// -log softmax(row)[target] for each row, using the log-sum-exp trick.
pub fn per_row_cross_entropy(logits: &Tensor, targets: &[usize]) -> Vec<F> {
    (0..logits.rows)
        .map(|r| {
            let row = logits.row(r);
            let max = row.iter().cloned().fold(F::NEG_INFINITY, F::max);
            let lse = max + row.iter().map(|v| (v - max).exp()).sum::<F>().ln();
            lse - row[targets[r]]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rand(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(rows, cols, (0..rows * cols).map(|_| rng.uniform() - 0.5).collect())
    }

    /// Forward and backward of matmul and both batch_matmuls on odd shapes, all gradients.
    fn run(threads: usize) -> Vec<Vec<F>> {
        crate::par::set_threads(threads);
        let mut rng = Rng::new(9);
        let mut g = Graph::new();
        let a = g.param("a", rand(&mut rng, 7 * 37, 53));
        let w = g.param("w", rand(&mut rng, 53, 41));
        let k = g.param("k", rand(&mut rng, 7 * 37, 41));
        let h = g.matmul("h", a, w); // (7·37) x 41, about 750k multiply-adds
        let s = g.batch_matmul("s", h, k, 7, true); // 7 blocks of 37x37
        let o = g.batch_matmul("o", s, h, 7, false); // 7 blocks of 37x41
        let m = g.mul("sq", o, o);
        let loss = g.mean("loss", m);
        g.backward(loss);
        g.nodes.iter().flat_map(|n| [n.value.data.clone(), n.grad.data.clone()]).collect()
    }

    #[test]
    fn threads_do_not_change_any_number() {
        let one = run(1);
        for t in [2, 3, 8, 20] {
            assert!(run(t) == one, "{t} threads gave different numbers than 1");
        }
        crate::par::set_threads(1);
    }

    #[test]
    fn matmul_at_equals_transpose_then_matmul_exactly() {
        let mut rng = Rng::new(4);
        for (r, k, c) in [(300, 48, 70), (301, 53, 7), (5, 3, 2), (2048, 128, 65)] {
            let (a, b) = (rand(&mut rng, r, k), rand(&mut rng, r, c));
            for t in [1, 8] {
                crate::par::set_threads(t);
                assert!(matmul_at(&a, &b) == matmul(&transpose(&a), &b), "{r}x{k}, {r}x{c}, {t} threads");
            }
        }
        crate::par::set_threads(1);
    }
}
