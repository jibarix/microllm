//! A speed check before building the transformer: one training step's worth of work at a given size.
//!
//! It runs the step's matrix products (forward, and backward the way Op::MatMul does it:
//! dA = G·Bᵀ, dB = Aᵀ·G, with real transposes) through the project's own `matmul`, plus
//! loops standing in for the elementwise parts (layernorm, softmax, GELU, cross-entropy).
//! There is no autograd tape and no optimizer here, so a real step will be somewhat slower.

use std::time::Instant;

use crate::autograd::{matmul, transpose};
use crate::rng::Rng;
use crate::tensor::{F, Tensor};

const VOCAB: usize = 65;

#[derive(Clone, Copy)]
pub struct Size {
    pub n: usize,
    pub t: usize,
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
}

fn rand(rng: &mut Rng, rows: usize, cols: usize) -> Tensor {
    Tensor::from_vec(rows, cols, (0..rows * cols).map(|_| rng.uniform() - 0.5).collect())
}

/// A product A·B and, if `backward`, its two gradient products. Returns the number of
/// multiply-adds done ×2 (FLOPs) and a checksum so the work cannot be optimized away.
fn product(a: &Tensor, b: &Tensor, backward: bool, flops: &mut f64) -> F {
    let out = matmul(a, b);
    *flops += 2.0 * (a.rows * a.cols * b.cols) as f64;
    let mut sum = out.data[0];
    if backward {
        let g = &out; // any matrix of the output's shape does the same work
        let da = matmul(g, &transpose(b));
        let db = matmul(&transpose(a), g);
        *flops += 4.0 * (a.rows * a.cols * b.cols) as f64;
        sum += da.data[0] + db.data[0];
    }
    sum
}

/// Elementwise work over `len` values with a transcendental each, done `times` times.
fn elementwise(x: &mut [F], times: usize) -> F {
    for _ in 0..times {
        for v in x.iter_mut() {
            *v = (*v * 0.5).tanh() + 1e-3 * v.exp().min(10.0);
        }
    }
    x[0]
}

/// Inputs for one step, made before timing so random-number generation is not measured.
struct Inputs {
    x: Tensor,
    wqkv: Tensor,
    wo: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    p: Tensor,
    w1: Tensor,
    w2: Tensor,
    h: Tensor,
    wout: Tensor,
    logits: Tensor,
}

fn inputs(s: Size, rng: &mut Rng) -> Inputs {
    let (rows, d, dh) = (s.n * s.t, s.d, s.d / s.heads);
    Inputs {
        x: rand(rng, rows, d),
        wqkv: rand(rng, d, 3 * d),
        wo: rand(rng, d, d),
        q: rand(rng, s.t, dh),
        k: rand(rng, dh, s.t),
        v: rand(rng, s.t, dh),
        p: rand(rng, s.t, s.t),
        w1: rand(rng, d, 4 * d),
        w2: rand(rng, 4 * d, d),
        h: rand(rng, rows, 4 * d),
        wout: rand(rng, d, VOCAB),
        logits: rand(rng, rows, VOCAB),
    }
}

/// One step (forward, plus backward when `backward`) over n × T positions.
fn step(s: Size, i: &Inputs, backward: bool, flops: &mut f64) -> F {
    let bw = if backward { 2 } else { 1 }; // backward elementwise work ~ forward
    let mut sum = 0.0;
    let mut x = i.x.clone();
    for _ in 0..s.layers {
        sum += elementwise(&mut x.data, bw); // LN1
        sum += product(&x, &i.wqkv, backward, flops);
        // attention, per example and head: scores T×T = Q·Kᵀ, then context = P·V
        for _ in 0..s.n * s.heads {
            sum += product(&i.q, &i.k, backward, flops);
            let mut p = i.p.clone();
            sum += elementwise(&mut p.data, bw); // causal softmax
            sum += product(&p, &i.v, backward, flops);
        }
        sum += product(&x, &i.wo, backward, flops);
        sum += elementwise(&mut x.data, bw); // LN2
        sum += product(&x, &i.w1, backward, flops);
        let mut h = i.h.clone();
        sum += elementwise(&mut h.data, bw); // GELU
        sum += product(&h, &i.w2, backward, flops);
    }
    sum += elementwise(&mut x.data, bw); // final LN
    sum += product(&x, &i.wout, backward, flops);
    let mut logits = i.logits.clone();
    sum + elementwise(&mut logits.data, bw) // softmax + cross-entropy
}

fn params(s: Size) -> usize {
    let d = s.d;
    VOCAB * d + s.t * d + s.layers * (4 * d * d + 4 * d + 8 * d * d + 5 * d + 4 * d) + 2 * d + d * VOCAB + VOCAB
}

/// Median seconds and FLOPs per call of `f`, after one warm-up call.
fn time(reps: usize, mut f: impl FnMut(&mut f64) -> F) -> (f64, f64, F) {
    let mut sink = 0.0;
    let mut fl = 0.0;
    sink += f(&mut fl);
    let mut times = Vec::new();
    let mut flops = 0.0;
    for _ in 0..reps {
        flops = 0.0;
        let t0 = Instant::now();
        sink += f(&mut flops);
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], flops, sink)
}

pub fn run(sizes: &[Size]) -> Result<(), String> {
    let mut rng = Rng::new(1);
    let mut sink = 0.0;
    println!("matmul alone (the project's own, f64, one thread):");
    for (r, k, c) in [(1024, 48, 144), (1024, 48, 192), (1024, 192, 48), (32, 12, 32), (1024, 48, 65)] {
        let (a, b) = (rand(&mut rng, r, k), rand(&mut rng, k, c));
        let (t, fl, s) = time(9, |fl| product(&a, &b, false, fl));
        sink += s;
        println!("  {r:>5}×{k:<4}· {k:>4}×{c:<4} {:>8.3} ms  {:>6.2} GFLOP/s", t * 1e3, fl / t / 1e9);
    }
    println!("\none training step (forward + backward; no tape, no optimizer) and one forward-only eval batch:");
    println!("  {:>3} {:>3} {:>3} {:>2} {:>2} {:>8}  {:>10} {:>9} {:>9}  {:>12}", "n", "T", "d", "H", "L", "params", "step ms", "GFLOP", "GFLOP/s", "eval ms");
    for &s in sizes {
        if s.heads == 0 || s.d % s.heads != 0 {
            return Err(format!("d = {} is not divisible by H = {}", s.d, s.heads));
        }
        let inp = inputs(s, &mut rng);
        let (t, fl, a) = time(5, |fl| step(s, &inp, true, fl));
        let (te, _, b) = time(5, |fl| step(s, &inp, false, fl));
        sink += a + b;
        println!("  {:>3} {:>3} {:>3} {:>2} {:>2} {:>8}  {:>10.1} {:>9.3} {:>9.2}  {:>12.1}", s.n, s.t, s.d, s.heads, s.layers, params(s), t * 1e3, fl / 1e9, fl / t / 1e9, te * 1e3);
    }
    if sink.is_nan() {
        println!("(checksum {sink})");
    }
    Ok(())
}

/// Time the real model's training step (tape, backward, AdamW), split by op kind, at the
/// current thread count. Forward time of a node is the time since the previous node.
pub fn profile(n: usize, s: Size, steps: usize) -> Result<(), String> {
    use crate::autograd::Graph;
    use crate::model::{forward, Model, Transformer};
    use crate::optim::Optimizer;
    use std::collections::BTreeMap;
    let mut rng = Rng::new(1);
    let mut model = Transformer::new(VOCAB, s.t, s.d, s.heads, s.layers, &mut rng);
    let mut opt = Optimizer::adamw(1e-3, 0.1);
    let x: Vec<usize> = (0..n * s.t).map(|_| rng.below(VOCAB)).collect();
    let y: Vec<usize> = (0..n * s.t).map(|_| rng.below(VOCAB)).collect();
    let mut by_kind: BTreeMap<&'static str, (f64, f64)> = BTreeMap::new();
    let (mut tf, mut tb, mut to) = (0.0, 0.0, 0.0);
    for it in 0..=steps {
        let mut g = Graph::new();
        g.profile = Some((Instant::now(), Vec::new(), Vec::new()));
        let t0 = Instant::now();
        let f = forward(&model, &mut g, &x, &y);
        let t1 = Instant::now();
        g.backward(f.loss);
        let t2 = Instant::now();
        let grads: Vec<Tensor> = f.params.iter().map(|p| g.grad(*p).clone()).collect();
        opt.step(model.params_mut(), &grads);
        let t3 = Instant::now();
        if it == 0 {
            continue; // warm-up
        }
        tf += (t1 - t0).as_secs_f64();
        tb += (t2 - t1).as_secs_f64();
        to += (t3 - t2).as_secs_f64();
        let (_, fwd, bwd) = g.profile.take().unwrap();
        for (i, nd) in g.nodes.iter().enumerate() {
            let e = by_kind.entry(nd.op.kind()).or_insert((0.0, 0.0));
            e.0 += fwd[i];
            e.1 += bwd[i];
        }
    }
    let k = steps as f64;
    println!("threads {}: n={n} T={} d={} H={} L={}  per step: forward {:.1} ms  backward {:.1} ms  optimizer {:.1} ms", crate::par::threads(), s.t, s.d, s.heads, s.layers, tf / k * 1e3, tb / k * 1e3, to / k * 1e3);
    let mut rows: Vec<_> = by_kind.into_iter().collect();
    rows.sort_by(|a, b| (b.1 .0 + b.1 .1).partial_cmp(&(a.1 .0 + a.1 .1)).unwrap());
    for (kind, (f, b)) in rows {
        println!("  {kind:<16} forward {:>8.1} ms  backward {:>8.1} ms", f / k * 1e3, b / k * 1e3);
    }
    Ok(())
}
