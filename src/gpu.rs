//! The tape on the GPU: the same ops as autograd.rs with the same meaning, but values and
//! gradients live in GPU memory as f32 and every op is a kernel from kernels.cu.
//!
//! Node i of a GPU graph is node i of the CPU graph the same model code would build (both
//! implement `Tape`), so `to_host` turns it into an ordinary `autograd::Graph` that the
//! trace, the evaluations and the snapshots read exactly as before. Only what is asked for
//! is copied back; a training step copies back just the loss and the parameter gradients.

use crate::autograd::{self, Op, Tape, Var};
use crate::cuda::{dev, Arg, Buf};
use crate::optim::{Kind, Optimizer};
use crate::params::Params;
use crate::tensor::{Tensor, F};

struct Node {
    name: String,
    op: Op,
    rows: usize,
    cols: usize,
    value: Buf,
    grad: Option<Buf>,
    requires_grad: bool,
    is_param: bool,
    /// Integer data and saved statistics an op needs on the GPU: an embedding's ids and its
    /// rows grouped by id; cross-entropy's targets and per-row losses; normalize's mean and rstd.
    aux: Vec<Buf>,
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
    /// Views of resident parameter values and gradients, used by the next `param` calls in order
    /// (last first) instead of uploading the host values; see `Graph::with_params`.
    preset: Vec<(Buf, Buf)>,
    /// The one zeroed buffer every other gradient is a view of (see `backward`).
    grads: Option<Buf>,
}

fn upload(t: &Tensor) -> Buf {
    Buf::from_f32(&t.data.iter().map(|&v| v as f32).collect::<Vec<_>>())
}

fn download(b: &Buf, rows: usize, cols: usize) -> Tensor {
    Tensor::from_vec(rows, cols, b.to_f32().into_iter().map(F::from).collect())
}

fn ints(v: &[usize]) -> Buf {
    Buf::from_i32(&v.iter().map(|&i| i32::try_from(i).expect("id too large for the GPU")).collect::<Vec<_>>())
}

const NULL: Arg = Arg::P(0);
fn i(n: usize) -> Arg {
    Arg::I(i32::try_from(n).expect("size too large for the GPU kernels"))
}

/// C (+)= op(A)·op(B), batched; see k_gemm. Strides are in elements. When one product has few
/// output tiles but a long inner dimension (a weight gradient: K = every row of the batch), the
/// inner dimension is split across blocks and the partial products are added in a fixed order.
#[allow(clippy::too_many_arguments)]
fn gemm(c: &Buf, a: &Buf, b: &Buf, m: usize, n: usize, k: usize, ar_ac: (usize, usize), br_bc: (usize, usize), ldc: usize, batches: usize, strides: (usize, usize, usize), beta: f32) {
    gemm_res(c, a, b, m, n, k, ar_ac, br_bc, ldc, batches, strides, beta, None);
}

/// `gemm`, or with `res = Some((alpha, R, rbeta))`: C = alpha·op(A)·op(B) + rbeta·R, R laid
/// out like C (one launch instead of a product, a copy and an axpby).
#[allow(clippy::too_many_arguments)]
fn gemm_res(c: &Buf, a: &Buf, b: &Buf, m: usize, n: usize, k: usize, (a_r, a_c): (usize, usize), (b_r, b_c): (usize, usize), ldc: usize, batches: usize, (sa, sb, sc): (usize, usize, usize), beta: f32, res: Option<(f32, &Buf, f32)>) {
    let d = dev();
    let tiles = m.div_ceil(64) * n.div_ceil(64);
    let mut splits = 1;
    if batches == 1 && tiles < 48 && k >= 512 && res.is_none() {
        splits = k.div_ceil(256).min(96usize.div_ceil(tiles));
    }
    let kchunk = k.div_ceil(splits).div_ceil(16) * 16;
    splits = k.div_ceil(kchunk).max(1);
    let grid = (n.div_ceil(64) as u32, m.div_ceil(64) as u32, (batches * splits) as u32);
    let common = |out: &Buf, ldc: usize, sc: usize, beta: f32| {
        d.launch(
            "k_gemm",
            grid,
            (256, 1, 1),
            &[out.arg(), a.arg(), b.arg(), i(m), i(n), i(k), i(a_r), i(a_c), i(b_r), i(b_c), i(ldc), Arg::L(sa as i64), Arg::L(sb as i64), Arg::L(sc as i64), i(splits), i(kchunk), Arg::F(beta), Arg::F(res.map_or(1.0, |r| r.0)), res.map_or(NULL, |r| r.1.arg()), Arg::F(res.map_or(0.0, |r| r.2))],
        )
    };
    if splits == 1 {
        common(c, ldc, sc, beta);
    } else {
        assert_eq!(ldc, n, "split products need a contiguous output");
        let w = Buf::new(splits * m * n);
        common(&w, n, m * n, 0.0);
        d.launch1("k_sum_splits", m * n, &[c.arg(), w.arg(), i(splits), i(m * n), Arg::F(beta)]);
    }
}

/// Column sums y[c] += Σ_r g[r][c] (· x[r][c]).
fn colsum(y: &Buf, g: &Buf, x: Option<&Buf>, rows: usize, cols: usize) {
    dev().launch("k_colsum", (cols.div_ceil(32) as u32, 1, 1), (32, 8, 1), &[y.arg(), g.arg(), x.map_or(NULL, Buf::arg), i(rows), i(cols)]);
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    fn push(&mut self, name: &str, op: Op, rows: usize, cols: usize, value: Buf, aux: Vec<Buf>) -> Var {
        let requires_grad = op.inputs().iter().any(|v| self.nodes[v.0].requires_grad);
        self.nodes.push(Node { name: name.to_string(), op, rows, cols, value, grad: None, requires_grad, is_param: false, aux });
        Var(self.nodes.len() - 1)
    }

    fn shape(&self, v: Var) -> (usize, usize) {
        (self.nodes[v.0].rows, self.nodes[v.0].cols)
    }

    fn val(&self, v: Var) -> &Buf {
        &self.nodes[v.0].value
    }

    /// Elementwise op with one input of the same shape.
    fn unary(&mut self, name: &str, op: Op, a: Var, kernel: &str, extra: &[Arg]) -> Var {
        let (r, c) = self.shape(a);
        let out = Buf::new(r * c);
        let mut args = vec![out.arg(), self.val(a).arg()];
        args.extend_from_slice(extra);
        args.push(i(r * c));
        dev().launch1(kernel, r * c, &args);
        self.push(name, op, r, c, out, Vec::new())
    }

    /// A row vector (1 x cols) applied to every row of a.
    fn row_op(&mut self, name: &str, op: Op, a: Var, row: Var, kernel: &str) -> Var {
        let ((r, c), (rr, rc)) = (self.shape(a), self.shape(row));
        assert!(rr == 1 && rc == c, "{}: row vector must be 1x{c}, got {rr}x{rc}", op.kind());
        let out = Buf::new(r * c);
        dev().launch1(kernel, r * c, &[out.arg(), self.val(a).arg(), self.val(row).arg(), i(r * c), i(c)]);
        self.push(name, op, r, c, out, Vec::new())
    }

    /// A graph whose parameter leaves are copies of the resident parameters (in the order the
    /// model makes them, which is `Params` order) rather than uploads of the host values.
    /// Their gradients go straight into the resident gradient array, which this zeroes.
    /// The graph must not outlive `r`.
    pub fn with_params(r: &Resident) -> Self {
        r.grad.zero();
        let preset = r.shapes.iter().rev().map(|&(rows, cols, off)| (r.p32.view(off, rows * cols), r.grad.view(off, rows * cols))).collect();
        Graph { nodes: Vec::new(), preset, grads: None }
    }

    pub fn item(&self, v: Var) -> F {
        F::from(self.val(v).to_f32()[0])
    }

    /// The gradient of a node after `backward` (zeros if nothing reached it).
    pub fn grad(&self, v: Var) -> Tensor {
        let nd = &self.nodes[v.0];
        nd.grad.as_ref().map_or_else(|| Tensor::zeros(nd.rows, nd.cols), |g| download(g, nd.rows, nd.cols))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn value(&self, v: Var) -> Tensor {
        let nd = &self.nodes[v.0];
        download(&nd.value, nd.rows, nd.cols)
    }

    /// The same graph on the host. Values are copied for the nodes `keep` selects (the
    /// others are empty, 0 rows) and gradients only with `grads`.
    pub fn to_host(&self, keep: impl Fn(usize, &Op) -> bool, grads: bool) -> autograd::Graph {
        let nodes = self
            .nodes
            .iter()
            .enumerate()
            .map(|(k, nd)| {
                let kept = keep(k, &nd.op);
                let value = if kept { download(&nd.value, nd.rows, nd.cols) } else { Tensor::from_vec(0, nd.cols, Vec::new()) };
                let mut op = nd.op.clone();
                if let (Op::Normalize { mean, rstd, .. }, true) = (&mut op, kept) {
                    *mean = nd.aux[0].to_f32().into_iter().map(F::from).collect();
                    *rstd = nd.aux[1].to_f32().into_iter().map(F::from).collect();
                }
                let grad = match (&nd.grad, grads) {
                    (Some(g), true) => download(g, nd.rows, nd.cols),
                    (None, true) => Tensor::zeros(nd.rows, nd.cols),
                    (_, false) => Tensor::from_vec(0, nd.cols, Vec::new()),
                };
                autograd::Node { name: nd.name.clone(), op, value, grad, requires_grad: nd.requires_grad, is_param: nd.is_param }
            })
            .collect();
        autograd::Graph { nodes, profile: None }
    }

    /// Fill every gradient with d(loss)/d(node), walking the tape backwards like the CPU.
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(self.shape(loss), (1, 1), "backward needs a scalar (1x1) loss");
        // Every gradient is a view of one zeroed buffer (one allocation, one memset), except the
        // loss's (1) and those of resident parameters (given already, in the resident array).
        let resident = |nd: &Node| nd.is_param && nd.grad.is_some();
        let needs = |k: usize, nd: &Node| nd.requires_grad && k < loss.0 && !resident(nd);
        let slot = |nd: &Node| (nd.rows * nd.cols).div_ceil(ALIGN) * ALIGN;
        let total: usize = self.nodes.iter().enumerate().filter(|(k, nd)| needs(*k, nd)).map(|(_, nd)| slot(nd)).sum();
        let arena = Buf::new(total);
        arena.zero();
        let mut off = 0;
        for k in 0..self.nodes.len() {
            let nd = &self.nodes[k];
            if resident(nd) {
                continue;
            }
            let g = if k == loss.0 && nd.requires_grad {
                Some(Buf::from_f32(&[1.0]))
            } else if needs(k, nd) {
                let v = arena.view(off, nd.rows * nd.cols);
                off += slot(nd);
                Some(v)
            } else {
                None
            };
            self.nodes[k].grad = g;
        }
        self.grads = Some(arena);
        for k in (0..=loss.0).rev() {
            self.backward_node(k);
        }
    }

    fn backward_node(&self, k: usize) {
        let d = dev();
        let node = &self.nodes[k];
        let Some(g) = node.grad.as_ref().filter(|_| node.requires_grad) else { return };
        let (rows, cols) = (node.rows, node.cols);
        let len = rows * cols;
        // the gradient buffer of an input, if it needs one
        let gr = |v: &Var| self.nodes[v.0].grad.as_ref();
        let acc = |dst: &Buf, src: &Buf, s: f32, n: usize| d.launch1("k_acc", n, &[dst.arg(), src.arg(), Arg::F(s), i(n)]);
        match &node.op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                for x in [a, b] {
                    if let Some(dx) = gr(x) {
                        acc(dx, g, 1.0, len);
                    }
                }
            }
            Op::MatMul(a, b) => {
                // out = A (n x k) · B (k x m): dA = G·Bᵀ, dB = Aᵀ·G
                let (n, kk) = self.shape(*a);
                let m = cols;
                if let Some(da) = gr(a) {
                    gemm(da, g, self.val(*b), n, kk, m, (m, 1), (1, m), kk, 1, (0, 0, 0), 1.0);
                }
                if let Some(db) = gr(b) {
                    gemm(db, self.val(*a), g, kk, m, n, (1, kk), (m, 1), m, 1, (0, 0, 0), 1.0);
                }
            }
            Op::AddRow(a, bias) => {
                if let Some(da) = gr(a) {
                    acc(da, g, 1.0, len);
                }
                if let Some(db) = gr(bias) {
                    colsum(db, g, None, rows, cols);
                }
            }
            Op::Tanh(a) => {
                if let Some(da) = gr(a) {
                    d.launch1("k_tanh_bwd", len, &[da.arg(), g.arg(), node.value.arg(), i(len)]);
                }
            }
            Op::Embedding { table, .. } => {
                if let Some(dt) = gr(table) {
                    let vocab = self.nodes[table.0].rows;
                    d.launch1("k_scatter_rows", vocab * cols, &[dt.arg(), g.arg(), node.aux[1].arg(), node.aux[2].arg(), i(vocab), i(cols)]);
                }
            }
            Op::CrossEntropy { logits, .. } => {
                if let Some(dl) = gr(logits) {
                    let (r, c) = self.shape(*logits);
                    d.launch_rows("k_ce_bwd", r, &[dl.arg(), self.val(*logits).arg(), node.aux[0].arg(), g.arg(), i(r), i(c)]);
                }
            }
            Op::Reshape(a) => {
                if let Some(da) = gr(a) {
                    acc(da, g, 1.0, len);
                }
            }
            Op::Scale { a, by } => {
                if let Some(da) = gr(a) {
                    acc(da, g, *by as f32, len);
                }
            }
            Op::MulRow(a, gain) => {
                if let Some(da) = gr(a) {
                    d.launch1("k_acc_mul_row", len, &[da.arg(), g.arg(), self.val(*gain).arg(), i(len), i(cols)]);
                }
                if let Some(dg) = gr(gain) {
                    colsum(dg, g, Some(self.val(*a)), rows, cols);
                }
            }
            Op::Gelu(a) => {
                if let Some(da) = gr(a) {
                    d.launch1("k_gelu_bwd", len, &[da.arg(), g.arg(), self.val(*a).arg(), i(len)]);
                }
            }
            Op::Relu2(a) => {
                if let Some(da) = gr(a) {
                    d.launch1("k_relu2_bwd", len, &[da.arg(), g.arg(), self.val(*a).arg(), i(len)]);
                }
            }
            Op::Normalize { a, .. } => {
                if let Some(da) = gr(a) {
                    d.launch_rows("k_normalize_bwd", rows, &[da.arg(), g.arg(), node.value.arg(), node.aux[1].arg(), i(rows), i(cols)]);
                }
            }
            Op::SplitHeads { a, t, heads, .. } => {
                // out[k] = x[map k], so dx[map k] += g[k]
                if let Some(da) = gr(a) {
                    d.launch1("k_heads", len, &[da.arg(), g.arg(), i(*t), i(*heads), i(cols), i(0), i(1), i(len)]);
                }
            }
            Op::MergeHeads { a, t, heads, .. } => {
                // out[map k] = x[k], so dx[k] += g[map k]
                if let Some(da) = gr(a) {
                    d.launch1("k_heads", len, &[da.arg(), g.arg(), i(*t), i(*heads), i(cols / heads), i(1), i(1), i(len)]);
                }
            }
            Op::BatchMatMul { a, b, batches, trans_b } => {
                let ((xr, k_), (yr, yc)) = (self.shape(*a), self.shape(*b));
                let (r, c, nb) = (xr / batches, cols, *batches);
                let (x, y) = (self.val(*a), self.val(*b));
                if *trans_b {
                    // out = A·Bᵀ (B block c x k): dA = G·B, dB = Gᵀ·A
                    if let Some(da) = gr(a) {
                        gemm(da, g, y, r, k_, c, (c, 1), (yc, 1), k_, nb, (r * c, yr / nb * yc, r * k_), 1.0);
                    }
                    if let Some(db) = gr(b) {
                        gemm(db, g, x, c, k_, r, (1, c), (k_, 1), k_, nb, (r * c, r * k_, c * k_), 1.0);
                    }
                } else {
                    // out = A·B (B block k x c): dA = G·Bᵀ, dB = Aᵀ·G
                    if let Some(da) = gr(a) {
                        gemm(da, g, y, r, k_, c, (c, 1), (1, c), k_, nb, (r * c, k_ * c, r * k_), 1.0);
                    }
                    if let Some(db) = gr(b) {
                        gemm(db, x, g, k_, c, r, (1, k_), (c, 1), c, nb, (r * k_, r * c, k_ * c), 1.0);
                    }
                }
            }
            Op::CausalSoftmax { a, t } => {
                if let Some(da) = gr(a) {
                    d.launch_rows("k_causal_softmax_bwd", rows, &[da.arg(), g.arg(), node.value.arg(), i(rows), i(*t)]);
                }
            }
            Op::Mul(..) | Op::Mean(_) => unreachable!("{} is not built on the GPU", node.op.kind()),
        }
    }
}

impl Tape for Graph {
    fn input(&mut self, name: &str, value: Tensor) -> Var {
        let b = upload(&value);
        self.push(name, Op::Leaf, value.rows, value.cols, b, Vec::new())
    }

    fn param(&mut self, name: &str, value: Tensor) -> Var {
        let (v, grad) = match self.preset.pop() {
            Some((b, g)) => {
                assert_eq!(b.len, value.len(), "{name}: the GPU copy has another size");
                (self.push(name, Op::Leaf, value.rows, value.cols, b, Vec::new()), Some(g))
            }
            None => (self.input(name, value), None),
        };
        let nd = &mut self.nodes[v.0];
        nd.grad = grad;
        nd.requires_grad = true;
        nd.is_param = true;
        v
    }

    fn add(&mut self, name: &str, a: Var, b: Var) -> Var {
        let ((r, c), sb) = (self.shape(a), self.shape(b));
        assert_eq!((r, c), sb, "add: shape mismatch");
        let out = Buf::new(r * c);
        dev().launch1("k_add", r * c, &[out.arg(), self.val(a).arg(), self.val(b).arg(), i(r * c)]);
        self.push(name, Op::Add(a, b), r, c, out, Vec::new())
    }

    fn matmul(&mut self, name: &str, a: Var, b: Var) -> Var {
        let ((n, k), (k2, m)) = (self.shape(a), self.shape(b));
        assert_eq!(k, k2, "matmul: {n}x{k} @ {k2}x{m}");
        let out = Buf::new(n * m);
        gemm(&out, self.val(a), self.val(b), n, m, k, (k, 1), (m, 1), m, 1, (0, 0, 0), 0.0);
        self.push(name, Op::MatMul(a, b), n, m, out, Vec::new())
    }

    fn add_row(&mut self, name: &str, a: Var, bias: Var) -> Var {
        self.row_op(name, Op::AddRow(a, bias), a, bias, "k_add_row")
    }

    fn tanh(&mut self, name: &str, a: Var) -> Var {
        self.unary(name, Op::Tanh(a), a, "k_tanh", &[])
    }

    fn embedding(&mut self, name: &str, table: Var, ids: &[usize]) -> Var {
        let (vocab, cols) = self.shape(table);
        assert!(ids.iter().all(|&id| id < vocab), "embedding: id out of range for table with {vocab} rows");
        // rows grouped by id (counting sort, rows in increasing order within an id) for the backward
        let mut offsets = vec![0usize; vocab + 1];
        for &id in ids {
            offsets[id + 1] += 1;
        }
        for v in 0..vocab {
            offsets[v + 1] += offsets[v];
        }
        let mut next = offsets.clone();
        let mut order = vec![0usize; ids.len()];
        for (r, &id) in ids.iter().enumerate() {
            order[next[id]] = r;
            next[id] += 1;
        }
        let ids_b = ints(ids);
        let out = Buf::new(ids.len() * cols);
        dev().launch1("k_gather", ids.len() * cols, &[out.arg(), self.val(table).arg(), ids_b.arg(), i(ids.len() * cols), i(cols)]);
        self.push(name, Op::Embedding { table, ids: ids.to_vec() }, ids.len(), cols, out, vec![ids_b, ints(&offsets), ints(&order)])
    }

    fn cross_entropy(&mut self, name: &str, logits: Var, targets: &[usize]) -> Var {
        let (r, c) = self.shape(logits);
        assert_eq!(r, targets.len(), "cross_entropy: {r} rows but {} targets", targets.len());
        let (t, rowloss, out) = (ints(targets), Buf::new(r), Buf::new(1));
        let d = dev();
        d.launch_rows("k_ce_rows", r, &[rowloss.arg(), self.val(logits).arg(), t.arg(), i(r), i(c)]);
        d.launch("k_sum", (1, 1, 1), (256, 1, 1), &[out.arg(), rowloss.arg(), i(r), Arg::F(1.0 / r as f32)]);
        self.push(name, Op::CrossEntropy { logits, targets: targets.to_vec() }, 1, 1, out, vec![t, rowloss])
    }

    fn reshape(&mut self, name: &str, a: Var, rows: usize, cols: usize) -> Var {
        let (r, c) = self.shape(a);
        assert_eq!(r * c, rows * cols, "reshape: {r}x{c} cannot become {rows}x{cols}");
        let v = self.unary(name, Op::Reshape(a), a, "k_copy", &[]);
        let nd = &mut self.nodes[v.0];
        (nd.rows, nd.cols) = (rows, cols);
        v
    }

    fn scale(&mut self, name: &str, a: Var, by: F) -> Var {
        self.unary(name, Op::Scale { a, by }, a, "k_scale", &[Arg::F(by as f32)])
    }

    fn mul_row(&mut self, name: &str, a: Var, gain: Var) -> Var {
        self.row_op(name, Op::MulRow(a, gain), a, gain, "k_mul_row")
    }

    fn gelu(&mut self, name: &str, a: Var) -> Var {
        self.unary(name, Op::Gelu(a), a, "k_gelu", &[])
    }

    fn relu2(&mut self, name: &str, a: Var) -> Var {
        self.unary(name, Op::Relu2(a), a, "k_relu2", &[])
    }

    fn normalize(&mut self, name: &str, a: Var, eps: F) -> Var {
        let (r, c) = self.shape(a);
        let (out, mean, rstd) = (Buf::new(r * c), Buf::new(r), Buf::new(r));
        dev().launch_rows("k_normalize", r, &[out.arg(), mean.arg(), rstd.arg(), self.val(a).arg(), i(r), i(c), Arg::F(eps as f32)]);
        self.push(name, Op::Normalize { a, eps, mean: Vec::new(), rstd: Vec::new() }, r, c, out, vec![mean, rstd])
    }

    fn split_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        let (r, c) = self.shape(a);
        assert!(r == n * t && heads > 0 && c % heads == 0, "split_heads: {r}x{c} is not ({n}·{t}) x ({heads}·dh)");
        let out = Buf::new(r * c);
        dev().launch1("k_heads", r * c, &[out.arg(), self.val(a).arg(), i(t), i(heads), i(c / heads), i(1), i(0), i(r * c)]);
        self.push(name, Op::SplitHeads { a, n, t, heads }, n * heads * t, c / heads, out, Vec::new())
    }

    fn merge_heads(&mut self, name: &str, a: Var, n: usize, t: usize, heads: usize) -> Var {
        let (r, c) = self.shape(a);
        assert_eq!(r, n * heads * t, "merge_heads: {r} rows is not {n}·{heads}·{t}");
        let out = Buf::new(r * c);
        dev().launch1("k_heads", r * c, &[out.arg(), self.val(a).arg(), i(t), i(heads), i(c), i(0), i(0), i(r * c)]);
        self.push(name, Op::MergeHeads { a, n, t, heads }, n * t, heads * c, out, Vec::new())
    }

    fn batch_matmul(&mut self, name: &str, a: Var, b: Var, batches: usize, trans_b: bool) -> Var {
        let ((xr, k), (yr, yc)) = (self.shape(a), self.shape(b));
        assert!(batches > 0 && xr % batches == 0 && yr % batches == 0, "batch_matmul: rows do not split into {batches} blocks");
        let r = xr / batches;
        let (inner, c) = if trans_b { (yc, yr / batches) } else { (yr / batches, yc) };
        assert_eq!(k, inner, "batch_matmul: blocks do not multiply");
        let out = Buf::new(xr * c);
        let bstride = if trans_b { (1, yc) } else { (yc, 1) };
        gemm(&out, self.val(a), self.val(b), r, c, k, (k, 1), bstride, c, batches, (r * k, yr / batches * yc, r * c), 0.0);
        self.push(name, Op::BatchMatMul { a, b, batches, trans_b }, xr, c, out, Vec::new())
    }

    fn causal_softmax(&mut self, name: &str, a: Var, t: usize) -> Var {
        let (r, c) = self.shape(a);
        assert!(t > 0 && c == t && r % t == 0, "causal_softmax: {r}x{c} is not a stack of {t}x{t} blocks");
        let out = Buf::new(r * c);
        dev().launch_rows("k_causal_softmax", r, &[out.arg(), self.val(a).arg(), i(r), i(t)]);
        self.push(name, Op::CausalSoftmax { a, t }, r, c, out, Vec::new())
    }
}

/// Training state kept on the GPU between steps: each parameter's f64 master value and Adam
/// moments (as on the CPU, so the optimizer's numbers are the same), the f32 copy the forward
/// pass reads, its gradient, and the change the last step applied. Every array holds all the
/// parameters one after another (each starting at a multiple of `ALIGN`), so one kernel launch
/// updates them all; graphs made by `Graph::with_params` read and write views into them. The
/// host copies in `Params` are only brought up to date when asked (`download`), for
/// evaluations, snapshots and the end of training.
pub struct Resident {
    /// (rows, cols, offset) per parameter.
    shapes: Vec<(usize, usize, usize)>,
    total: usize,
    p: Buf,
    m: Buf,
    v: Buf,
    u: Buf,
    p32: Buf,
    grad: Buf,
    /// 1 where weight decay applies, per element.
    decay: Buf,
    /// Offsets and lengths per parameter, for the norm kernel.
    offsets: Buf,
    lens: Buf,
    /// Sums of squares: gradients, values before the step, updates (one double per parameter each).
    norms: Buf,
    muon: Option<MuonState>,
}

/// Muon's workspace: every Muon matrix in its wide orientation (rows ≤ cols), grouped by
/// shape so each Newton–Schulz product is one batched launch per group.
struct MuonState {
    /// Per element of the flat arrays: its index in the workspace, or −1 (not Muon's).
    widx: Buf,
    /// Per element: `muon_scale` of its matrix.
    scale: Buf,
    /// (rows, cols, matrices, workspace offset) per shape.
    groups: Vec<(usize, usize, usize, usize)>,
    /// Workspace offset and size of each matrix, for the norms.
    offsets: Buf,
    lens: Buf,
    sq: Buf,
    x: Buf,
    xn: Buf,
    a: Buf,
    b: Buf,
}

/// Parameters (and gradients in a graph's arena) start at multiples of this many elements.
const ALIGN: usize = 64;
/// After an odd number of Newton–Schulz steps the result is in the second buffer and is
/// copied back so `k_muon_apply` always reads the first.
const NS_RESULT_IN_XN: bool = crate::optim::NS_STEPS % 2 == 1;

/// Per-parameter norms from one optimizer step.
pub struct StepNorms {
    pub grad: Vec<F>,
    /// Before the update, like everything else recorded for a step.
    pub param: Vec<F>,
    pub update: Vec<F>,
}

impl Resident {
    /// `muon[i]`: parameter i gets Muon's update (empty: none do).
    pub fn new(params: &Params, muon: &[bool]) -> Self {
        let mut shapes = Vec::new();
        let mut total = 0;
        for p in &params.list {
            shapes.push((p.value.rows, p.value.cols, total));
            total += p.value.len().div_ceil(ALIGN) * ALIGN;
        }
        let flat = |f: &dyn Fn(&crate::params::Param) -> &Tensor| {
            let mut all = vec![0.0; total];
            for (p, &(_, _, off)) in params.list.iter().zip(&shapes) {
                all[off..off + p.value.len()].copy_from_slice(&f(p).data);
            }
            Buf::from_f64(&all)
        };
        let (p, m, v) = (flat(&|p| &p.value), flat(&|p| &p.m), flat(&|p| &p.v));
        // bit 0: weight decay (AdamW); bit 1: Muon updates it (k_adamw skips it)
        let is_muon = |k: usize| muon.get(k).copied().unwrap_or(false);
        let mut decay = vec![0i32; total];
        for (k, (q, &(_, _, off))) in params.list.iter().zip(&shapes).enumerate() {
            decay[off..off + q.value.len()].fill(q.decay as i32 | (is_muon(k) as i32) << 1);
        }
        let muon = (0..shapes.len()).any(is_muon).then(|| {
            // the wide shapes, in order of first appearance
            let wide = |k: usize| (shapes[k].0.min(shapes[k].1), shapes[k].0.max(shapes[k].1));
            let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new();
            for k in (0..shapes.len()).filter(|&k| is_muon(k)) {
                let (r, c) = wide(k);
                match groups.iter_mut().find(|g| (g.0, g.1) == (r, c)) {
                    Some(g) => g.2 += 1,
                    None => groups.push((r, c, 1, 0)),
                }
            }
            let mut off = 0;
            for g in &mut groups {
                g.3 = off;
                off += g.0 * g.1 * g.2;
            }
            let ws = off;
            let (mut widx, mut scale) = (vec![-1i32; total], vec![0f32; total]);
            let mut next: Vec<usize> = groups.iter().map(|g| g.3).collect();
            let mut moff = Vec::new();
            for k in (0..shapes.len()).filter(|&k| is_muon(k)) {
                let (rows, cols, off) = shapes[k];
                let gi = groups.iter().position(|g| (g.0, g.1) == wide(k)).unwrap();
                let base = next[gi];
                next[gi] += rows * cols;
                moff.push(base);
                for e in 0..rows * cols {
                    let (r, c) = (e / cols, e % cols);
                    // tall matrices are stored transposed: element (r, c) at (c, r)
                    let at = if rows > cols { c * rows + r } else { e };
                    widx[off + e] = (base + at) as i32;
                    scale[off + e] = crate::optim::muon_scale(rows, cols) as f32;
                }
            }
            // the norms are read per group (k_scale_mats), so list the matrices in workspace order
            moff.sort_unstable();
            let sq_max = groups.iter().map(|g| g.0 * g.0 * g.2).max().unwrap_or(0);
            MuonState {
                widx: Buf::from_i32(&widx),
                scale: Buf::from_f32(&scale),
                offsets: ints(&moff),
                lens: ints(&moff.iter().map(|&o| groups.iter().find(|g| o >= g.3 && o < g.3 + g.0 * g.1 * g.2).map_or(0, |g| g.0 * g.1)).collect::<Vec<_>>()),
                sq: Buf::zeros(2 * moff.len()),
                x: Buf::new(ws),
                xn: Buf::new(ws),
                a: Buf::new(sq_max),
                b: Buf::new(sq_max),
                groups,
            }
        });
        let p32 = Buf::new(total);
        dev().launch1("k_to_f32", total, &[p32.arg(), p.arg(), i(total)]);
        let offsets = ints(&shapes.iter().map(|s| s.2).collect::<Vec<_>>());
        let lens = ints(&shapes.iter().map(|s| s.0 * s.1).collect::<Vec<_>>());
        let norms = Buf::zeros(2 * 3 * shapes.len());
        Resident { total, p, m, v, u: Buf::zeros(2 * total), p32, grad: Buf::zeros(total), decay: Buf::from_i32(&decay), offsets, lens, norms, shapes, muon }
    }

    /// One optimizer step on the gradients the last graph from `with_params` left in `grad`.
    /// Same formulas and rounding as `Optimizer::step`, which also counts the step (`opt.t`).
    pub fn step(&self, opt: &mut Optimizer) -> StepNorms {
        let (bc1, bc2) = opt.begin_step();
        let adam = opt.kind != Kind::Sgd;
        let d = dev();
        let np = self.shapes.len();
        let sumsq = |slot: usize, x: &Buf, f64: bool| {
            d.launch("k_sumsq_segs", (np as u32, 1, 1), (256, 1, 1), &[Arg::P(self.norms.ptr + 8 * slot as u64), x.arg(), self.offsets.arg(), self.lens.arg(), Arg::I(f64 as i32)]);
        };
        sumsq(0, &self.grad, false);
        sumsq(np, &self.p, true);
        d.launch1(
            "k_adamw",
            self.total,
            &[self.p.arg(), self.m.arg(), self.v.arg(), self.u.arg(), self.p32.arg(), self.grad.arg(), self.decay.arg(), i(self.total), Arg::D(opt.lr), Arg::D(opt.beta1), Arg::D(opt.beta2), Arg::D(opt.eps), Arg::D(opt.weight_decay), Arg::D(bc1), Arg::D(bc2), Arg::I(adam as i32)],
        );
        if let Some(ms) = &self.muon {
            self.muon_step(ms, opt);
        }
        sumsq(2 * np, &self.u, true);
        let s: Vec<F> = self.norms.to_f64().into_iter().map(F::sqrt).collect();
        StepNorms { grad: s[..np].to_vec(), param: s[np..2 * np].to_vec(), update: s[2 * np..].to_vec() }
    }

    /// Muon's part of a step, as in `Optimizer::step` but with Newton–Schulz in f32.
    fn muon_step(&self, ms: &MuonState, opt: &Optimizer) {
        let d = dev();
        let n = self.total;
        d.launch1("k_muon_mom", n, &[self.m.arg(), self.grad.arg(), ms.x.arg(), ms.widx.arg(), i(n), Arg::D(crate::optim::MUON_MOMENTUM)]);
        let mats = ms.groups.iter().map(|g| g.2).sum::<usize>();
        d.launch("k_sumsq_segs", (mats as u32, 1, 1), (256, 1, 1), &[ms.sq.arg(), ms.x.arg(), ms.offsets.arg(), ms.lens.arg(), Arg::I(0)]);
        let (ca, cb, cc) = crate::optim::NS;
        let mut first = 0;
        for &(r, c, count, off) in &ms.groups {
            let (rc, rr) = (r * c, r * r);
            let x = ms.x.view(off, rc * count);
            let xn = ms.xn.view(off, rc * count);
            let sq = Buf::view(&ms.sq, 2 * first, 2 * count);
            d.launch1("k_scale_mats", rc * count, &[x.arg(), sq.arg(), i(rc), i(rc * count)]);
            let (a, b) = (ms.a.view(0, rr * count), ms.b.view(0, rr * count));
            let (mut cur, mut nxt) = (x, xn);
            for _ in 0..crate::optim::NS_STEPS {
                // A = X·Xᵀ, B = c·A·A + b·A, X' = B·X + a·X
                gemm(&a, &cur, &cur, r, r, c, (c, 1), (1, c), r, count, (rc, rc, rr), 0.0);
                gemm_res(&b, &a, &a, r, r, r, (r, 1), (r, 1), r, count, (rr, rr, rr), 0.0, Some((cc as f32, &a, cb as f32)));
                gemm_res(&nxt, &b, &cur, r, c, r, (r, 1), (c, 1), c, count, (rr, rc, rc), 0.0, Some((1.0, &cur, ca as f32)));
                std::mem::swap(&mut cur, &mut nxt);
            }
            if NS_RESULT_IN_XN {
                d.launch1("k_copy", rc * count, &[ms.x.view(off, rc * count).arg(), cur.arg(), i(rc * count)]);
            }
            first += count;
        }
        d.launch1(
            "k_muon_apply",
            n,
            &[self.p.arg(), self.u.arg(), self.p32.arg(), ms.x.arg(), ms.widx.arg(), ms.scale.arg(), i(n), Arg::D(opt.muon_lr), Arg::D(opt.weight_decay)],
        );
    }

    fn split(&self, b: &Buf) -> Vec<Tensor> {
        let all = b.to_f64();
        self.shapes.iter().map(|&(r, c, off)| Tensor::from_vec(r, c, all[off..off + r * c].to_vec())).collect()
    }

    /// The parameter values (and, with `moments`, Adam's m and v) into the host copies.
    pub fn download(&self, params: &mut Params, moments: bool) {
        for (t, p) in self.split(&self.p).into_iter().zip(params.list.iter_mut()) {
            p.value = t;
        }
        if moments {
            for ((m, v), p) in self.moments().into_iter().zip(params.list.iter_mut()) {
                (p.m, p.v) = (m, v);
            }
        }
    }

    /// Adam's m and v as they are now.
    pub fn moments(&self) -> Vec<(Tensor, Tensor)> {
        self.split(&self.m).into_iter().zip(self.split(&self.v)).collect()
    }

    /// The change the last step applied to each parameter.
    pub fn updates(&self) -> Vec<Tensor> {
        self.split(&self.u)
    }

    /// The gradients the last graph left, widened to f64 as the host sees them.
    pub fn grads(&self) -> Vec<Tensor> {
        let all = self.grad.to_f32();
        self.shapes.iter().map(|&(r, c, off)| Tensor::from_vec(r, c, all[off..off + r * c].iter().map(|&g| F::from(g)).collect())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Graph as Cpu;
    use crate::cuda;
    use crate::model::{forward, Mlp, Model, Transformer};
    use crate::rng::Rng;

    /// ‖a − b‖ / ‖b‖
    fn rel(a: &Tensor, b: &Tensor) -> F {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        let d = a.data.iter().zip(&b.data).map(|(x, y)| (x - y) * (x - y)).sum::<F>().sqrt();
        d / b.norm().max(1e-30)
    }

    /// The same model, batch and parameters on both tapes: every node's value and every
    /// parameter's gradient agree to f32 precision. Returns the GPU gradients.
    fn agree(m: &dyn Model, x: &[usize], y: &[usize]) -> Vec<Tensor> {
        let mut c = Cpu::new();
        let fc = forward(m, &mut c, x, y);
        c.backward(fc.loss);
        let mut g = Graph::new();
        let fg = forward(m, &mut g, x, y);
        g.backward(fg.loss);
        let h = g.to_host(|_, _| true, true);
        assert_eq!(h.nodes.len(), c.nodes.len());
        for (a, b) in h.nodes.iter().zip(&c.nodes) {
            assert_eq!(a.name, b.name);
            let r = rel(&a.value, &b.value);
            assert!(r < 1e-5, "{} ({}): value differs by {r:.1e}", a.name, a.op.kind());
            if let (Op::Normalize { rstd: ra, .. }, Op::Normalize { rstd: rb, .. }) = (&a.op, &b.op) {
                assert!(ra.iter().zip(rb).all(|(p, q)| (p - q).abs() <= 1e-4 * q.abs()), "{}: rstd differs", a.name);
            }
        }
        let grads: Vec<Tensor> = fg.params.iter().map(|p| g.grad(*p)).collect();
        let cpu: Vec<Tensor> = fc.params.iter().map(|p| c.grad(*p).clone()).collect();
        for (k, r) in crate::train::grad_rel_diffs(&grads, &cpu).into_iter().enumerate() {
            assert!(r < 1e-4, "{}: gradient differs by {r:.1e}", m.params().list[k].name);
        }
        grads
    }

    fn jiggle(m: &mut dyn Model, rng: &mut Rng) {
        for p in &mut m.params_mut().list {
            for v in &mut p.value.data {
                *v += 0.3 * rng.normal();
            }
        }
    }

    fn gpu() -> bool {
        match cuda::enable() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                false
            }
        }
    }

    #[test]
    fn transformer_matches_cpu_and_repeats_exactly() {
        if !gpu() {
            return;
        }
        let mut rng = Rng::new(3);
        let (v, t, n) = (23, 7, 3);
        let mut m = Transformer::new(v, t, 16, 4, 2, &mut rng);
        jiggle(&mut m, &mut rng);
        let x: Vec<usize> = (0..n * t).map(|i| (i * 7 + 2) % v).collect();
        let y: Vec<usize> = (0..n * t).map(|i| (i * 5 + 1) % v).collect();
        let first = agree(&m, &x, &y);
        let again = agree(&m, &x, &y);
        assert!(first.iter().zip(&again).all(|(a, b)| a.data == b.data), "two GPU runs gave different gradients");
        // an ablated head (the mask input and mul_row by a constant)
        m.set_ablation(vec![(1, 2)]);
        agree(&m, &x, &y);
        // the speed-experiment variants (relu², normalized queries and keys); zero-init is
        // left out because jiggle moves the matrices off zero anyway
        let variant = crate::model::Variant { relu2: true, qk_norm: true, zero_init: false };
        let mut m = Transformer::with_variant(v, t, 16, 4, 2, variant, &mut rng);
        jiggle(&mut m, &mut rng);
        agree(&m, &x, &y);
    }

    /// Training with the parameters and optimizer on the GPU gives bit for bit what uploading the
    /// host parameters and running the CPU optimizer gives: gradients, updates, values, moments.
    #[test]
    fn resident_training_matches_host_optimizer_exactly() {
        if !gpu() {
            return;
        }
        let (v, t, n) = (23, 7, 3);
        let x: Vec<usize> = (0..n * t).map(|i| (i * 7 + 2) % v).collect();
        let y: Vec<usize> = (0..n * t).map(|i| (i * 5 + 1) % v).collect();
        for adam in [true, false] {
            let mut rng = Rng::new(5);
            let mut m = Transformer::new(v, t, 16, 4, 2, &mut rng);
            jiggle(&mut m, &mut rng);
            let mut host = crate::params::Params::default();
            for p in &m.params().list {
                host.add(&p.name, p.value.clone(), p.decay);
            }
            assert!(host.list.iter().any(|p| p.decay) && host.list.iter().any(|p| !p.decay));
            let new_opt = || if adam { Optimizer::adamw(0.01, 0.1) } else { Optimizer::sgd(0.05) };
            let (mut opt_gpu, mut opt_cpu) = (new_opt(), new_opt());
            let res = Resident::new(m.params(), &[]);
            for step in 0..3 {
                let mut g = Graph::with_params(&res);
                let f = forward(&m, &mut g, &x, &y);
                g.backward(f.loss);
                let grads = res.grads();
                // The same step from the host values, the way training ran before.
                let mut plain = Graph::new();
                let fp = forward(&m, &mut plain, &x, &y);
                plain.backward(fp.loss);
                for (k, p) in fp.params.iter().enumerate() {
                    assert_eq!(plain.grad(*p).data, grads[k].data, "step {step}: gradient of {}", host.list[k].name);
                }
                let opt_lr = 0.01 * (step + 1) as F;
                opt_gpu.lr = opt_lr;
                opt_cpu.lr = opt_lr;
                let norms = res.step(&mut opt_gpu);
                let param_before: Vec<F> = host.list.iter().map(|p| p.value.norm()).collect();
                let updates = opt_cpu.step(&mut host, &grads);
                let gpu_updates = res.updates();
                res.download(m.params_mut(), true);
                for (k, p) in host.list.iter().enumerate() {
                    let q = &m.params().list[k];
                    assert_eq!(gpu_updates[k].data, updates[k].data, "step {step}: update to {}", p.name);
                    assert_eq!(q.value.data, p.value.data, "step {step}: {}", p.name);
                    assert_eq!((&q.m.data, &q.v.data), (&p.m.data, &p.v.data), "step {step}: moments of {}", p.name);
                    for (a, b) in [(norms.grad[k], grads[k].norm()), (norms.param[k], param_before[k]), (norms.update[k], updates[k].norm())] {
                        assert!((a - b).abs() <= 1e-12 * b.max(1e-30), "step {step}: a norm of {}: {a} vs {b}", p.name);
                    }
                }
            }
        }
    }

    /// Muon on the GPU (Newton–Schulz in f32, both orientations, batched by shape) gives the
    /// CPU optimizer's updates to f32 precision; AdamW's part stays bit-identical.
    #[test]
    fn resident_muon_matches_host_optimizer() {
        if !gpu() {
            return;
        }
        let (v, t, n) = (23, 7, 3);
        let x: Vec<usize> = (0..n * t).map(|i| (i * 7 + 2) % v).collect();
        let y: Vec<usize> = (0..n * t).map(|i| (i * 5 + 1) % v).collect();
        let mut rng = Rng::new(6);
        let mut m = Transformer::new(v, t, 16, 4, 2, &mut rng);
        jiggle(&mut m, &mut rng);
        let mut host = crate::params::Params::default();
        for p in &m.params().list {
            host.add(&p.name, p.value.clone(), p.decay);
        }
        let new_opt = || Optimizer::muon(0.01, 0.02, 0.1, m.params());
        let (mut opt_gpu, mut opt_cpu) = (new_opt(), new_opt());
        assert_eq!(opt_cpu.muon.iter().filter(|&&b| b).count(), 6 * 2, "the six block matrices of each layer");
        let res = Resident::new(m.params(), &opt_gpu.muon);
        for step in 0..3 {
            let mut g = Graph::with_params(&res);
            let f = forward(&m, &mut g, &x, &y);
            g.backward(f.loss);
            let grads = res.grads();
            res.step(&mut opt_gpu);
            let updates = opt_cpu.step(&mut host, &grads);
            let gpu_updates = res.updates();
            for (k, p) in host.list.iter().enumerate() {
                let r = rel(&gpu_updates[k], &updates[k]);
                let tol = if opt_cpu.muon[k] { 1e-4 } else { 1e-9 };
                assert!(r < tol, "step {step}: update to {} differs by {r:.1e}", p.name);
            }
        }
        res.download(m.params_mut(), true);
        for (q, p) in m.params().list.iter().zip(&host.list) {
            assert!(rel(&q.value, &p.value) < 1e-6, "{}", p.name);
        }
    }

    #[test]
    fn mlp_matches_cpu() {
        if !gpu() {
            return;
        }
        let mut rng = Rng::new(4);
        let mut m = Mlp::new(19, 5, 6, 40, &mut rng);
        jiggle(&mut m, &mut rng);
        let n = 9;
        let x: Vec<usize> = (0..n * 5).map(|i| (i * 3 + 1) % 19).collect();
        let y: Vec<usize> = (0..n).map(|i| (i * 7) % 19).collect();
        agree(&m, &x, &y);
    }

    /// Long inner dimensions take the split path (partial products added in order): a weight
    /// gradient over 1000 rows, and sizes that are not multiples of the 64x64 tiles.
    #[test]
    fn long_products_match_cpu() {
        if !gpu() {
            return;
        }
        let mut rng = Rng::new(5);
        let r = |rows, cols, rng: &mut Rng| Tensor::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal()).collect());
        let (a, w) = (r(1000, 70, &mut rng), r(70, 33, &mut rng));
        let (mut c, mut g) = (Cpu::new(), Graph::new());
        let ids: Vec<usize> = (0..1000).map(|i| i % 33).collect();
        let (ac, wc) = (c.input("a", a.clone()), c.param("w", w.clone()));
        let (ag, wg) = (Tape::input(&mut g, "a", a), Tape::param(&mut g, "w", w));
        let (oc, og) = (c.matmul("o", ac, wc), Tape::matmul(&mut g, "o", ag, wg));
        let (lc, lg) = (c.cross_entropy("l", oc, &ids), Tape::cross_entropy(&mut g, "l", og, &ids));
        c.backward(lc);
        g.backward(lg);
        dev().sync();
        assert!(rel(&g.value(og), c.value(oc)) < 1e-5);
        assert!(rel(&g.grad(wg), c.grad(wc)) < 1e-4, "split weight gradient differs by {:.1e}", rel(&g.grad(wg), c.grad(wc)));
    }
}
