//! The loss landscape around a finished run's training path (Li et al. 2018, "Visualizing the
//! Loss Landscape of Neural Nets", section 7): the plane through the final weights spanned by
//! the two principal directions of the path, the loss on a grid over that plane, and the path
//! itself projected onto it.
//!
//!   microllm landscape --run NAME [--grid N] [--tokens N] [--threads N]
//!
//! Reads runs/NAME/meta.js (the architecture), snaps/*.js (the weights at each snapshot step and
//! the batches seen there) and model.js (the final weights); writes runs/NAME/landscape.js.
//! The plane is a 2-D slice of a space with one dimension per parameter, so it shows the loss
//! along two chosen directions only; other directions can look different.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::json::{parse_js, Json};
use crate::model::{forward_eval, Bigram, Mlp, Model, Transformer, Variant};
use crate::rng::Rng;
use crate::tensor::F;
use crate::trace::{array, ints, num, num64, object, packed, string, write_atomic};

pub struct Options {
    pub run: String,
    /// Grid points per side.
    pub grid: usize,
    /// Predictions to measure the loss on (snapshot batches, in step order, until at least this many).
    pub tokens: usize,
}

fn read_js(path: &Path) -> Result<Json, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_js(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn field<'a>(v: &'a Json, key: &str, file: &str) -> Result<&'a Json, String> {
    v.get(key).ok_or_else(|| format!("{file}: no {key:?}"))
}

fn count(v: &Json, key: &str, file: &str) -> Result<usize, String> {
    field(v, key, file)?.as_usize().ok_or_else(|| format!("{file}: {key:?} is not a count"))
}

/// The run's model with its initial weights (overwritten before every measurement).
fn build(meta: &Json) -> Result<Box<dyn Model>, String> {
    let arch = field(meta, "architecture", "meta.js")?;
    let v = count(meta, "vocab_size", "meta.js")?;
    let flag = |k: &str| arch.get(k).and_then(Json::as_bool).unwrap_or(false);
    let mut rng = Rng::new(0);
    Ok(match field(arch, "kind", "meta.js")?.as_str() {
        Some("bigram") => Box::new(Bigram::new(v)),
        Some("mlp") => Box::new(Mlp::new(v, count(arch, "context", "meta.js")?, count(arch, "embedding_dim", "meta.js")?, count(arch, "hidden", "meta.js")?, &mut rng)),
        Some("transformer") => {
            let variant = Variant { relu2: flag("mlp_relu2"), qk_norm: flag("qk_norm"), zero_init: flag("zero_init_out") };
            let (t, d) = (count(arch, "context", "meta.js")?, count(arch, "embedding_dim", "meta.js")?);
            Box::new(Transformer::with_variant(v, t, d, count(arch, "heads", "meta.js")?, count(arch, "layers", "meta.js")?, variant, &mut rng))
        }
        other => return Err(format!("meta.js: unknown model kind {other:?}")),
    })
}

/// All weights as one flat vector, in the model's parameter order, from (name, shape, values) entries.
fn flatten(model: &dyn Model, found: HashMap<String, (Vec<usize>, Vec<F>)>, file: &str) -> Result<Vec<F>, String> {
    let mut out = Vec::with_capacity(model.params().count());
    for p in &model.params().list {
        let (shape, values) = found.get(&p.name).ok_or_else(|| format!("{file}: no parameter {:?}", p.name))?;
        if shape[..] != [p.value.rows, p.value.cols] || values.len() != p.value.len() {
            return Err(format!("{file}: parameter {:?} has shape {shape:?}, the model expects {}x{}", p.name, p.value.rows, p.value.cols));
        }
        out.extend_from_slice(values);
    }
    Ok(out)
}

fn set_weights(model: &mut dyn Model, w: &[F]) {
    let mut at = 0;
    for p in &mut model.params_mut().list {
        let n = p.value.len();
        p.value.data.copy_from_slice(&w[at..at + n]);
        at += n;
    }
}

/// Mean loss per prediction (nats) over the evaluation batches.
fn loss(model: &mut dyn Model, w: &[F], batches: &[(Vec<usize>, Vec<usize>)]) -> F {
    set_weights(model, w);
    let (mut sum, mut n) = (0.0, 0);
    for (x, y) in batches {
        let (g, f) = forward_eval(model, x, y, false);
        sum += g.value(f.loss).item() * y.len() as F;
        n += y.len();
    }
    sum / n as F
}

fn dot(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Eigenvalues and unit eigenvectors (columns of the returned matrix, row-major k×k) of a
/// symmetric k×k matrix, by cyclic Jacobi rotations; sorted by eigenvalue, largest first.
fn eigen(mut a: Vec<F>, k: usize) -> (Vec<F>, Vec<F>) {
    let mut v = vec![0.0; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: F = (0..k).flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * k + j] * a[i * k + j]).sum();
        let diag: F = (0..k).map(|i| a[i * k + i] * a[i * k + i]).sum();
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                let apq = a[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for r in 0..k {
                    let (arp, arq) = (a[r * k + p], a[r * k + q]);
                    a[r * k + p] = c * arp - s * arq;
                    a[r * k + q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let (apr, aqr) = (a[p * k + r], a[q * k + r]);
                    a[p * k + r] = c * apr - s * aqr;
                    a[q * k + r] = s * apr + c * aqr;
                }
                for r in 0..k {
                    let (vrp, vrq) = (v[r * k + p], v[r * k + q]);
                    v[r * k + p] = c * vrp - s * vrq;
                    v[r * k + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| a[j * k + j].total_cmp(&a[i * k + i]));
    let vals = order.iter().map(|&i| a[i * k + i]).collect();
    let mut vecs = vec![0.0; k * k];
    for (c, &i) in order.iter().enumerate() {
        for r in 0..k {
            vecs[r * k + c] = v[r * k + i];
        }
    }
    (vals, vecs)
}

/// Evenly spaced points covering `xs` and 0 with a 20% margin on each side.
fn axis(xs: &[F], n: usize) -> Vec<F> {
    let lo = xs.iter().copied().fold(0.0, F::min);
    let hi = xs.iter().copied().fold(0.0, F::max);
    let pad = 0.2 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);
    (0..n).map(|i| lo + (hi - lo) * i as F / (n - 1) as F).collect()
}

pub fn run(opts: &Options) -> Result<(), String> {
    let start = Instant::now();
    let dir = Path::new("runs").join(&opts.run);
    let meta = read_js(&dir.join("meta.js"))?;
    let mut model = build(&meta)?;
    let names: Vec<String> = model.params().list.iter().map(|p| p.name.clone()).collect();
    let n_params = model.params().count();

    // the final weights
    let ckpt = read_js(&dir.join("model.js"))?;
    let mut found = HashMap::new();
    for p in field(&ckpt, "params", "model.js")?.as_arr().ok_or("model.js: params is not a list")? {
        let name = field(p, "name", "model.js")?.as_str().ok_or("model.js: bad name")?.to_string();
        let shape = field(p, "shape", "model.js")?.usizes().ok_or("model.js: bad shape")?;
        let value = field(p, "value", "model.js")?.numbers().ok_or("model.js: bad value")?;
        found.insert(name, (shape, value));
    }
    let fin = flatten(&*model, found, "model.js")?;
    drop(ckpt);

    // the weights at each snapshot step, and the batches seen there
    let steps = field(&meta, "snapshot_steps", "meta.js")?.usizes().ok_or("meta.js: bad snapshot_steps")?;
    let mut path: Vec<(usize, Vec<F>)> = Vec::new();
    let mut batches: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut from_steps = Vec::new();
    let mut have = 0;
    for &step in &steps {
        let file = format!("snaps/{step}.js");
        eprintln!("landscape: reading {file} ({:.0}s)", start.elapsed().as_secs_f64());
        let snap = read_js(&dir.join(&file))?;
        let mut found = HashMap::new();
        for node in field(&snap, "nodes", &file)?.as_arr().ok_or_else(|| format!("{file}: nodes is not a list"))? {
            if node.get("param").and_then(Json::as_bool) != Some(true) {
                continue;
            }
            let name = field(node, "name", &file)?.as_str().unwrap_or_default().to_string();
            if !names.contains(&name) {
                continue;
            }
            let shape = field(node, "shape", &file)?.usizes().ok_or_else(|| format!("{file}: bad shape"))?;
            let value = field(node, "value", &file)?.numbers().ok_or_else(|| format!("{file}: {name}: bad value"))?;
            found.insert(name, (shape, value));
        }
        path.push((step, flatten(&*model, found, &file)?));
        if have < opts.tokens {
            let x = field(&snap, "x", &file)?.usizes().ok_or_else(|| format!("{file}: bad x"))?;
            let y = field(&snap, "y", &file)?.usizes().ok_or_else(|| format!("{file}: bad y"))?;
            // whole examples only, and no more than needed
            let n = count(&snap, "n", &file)?.max(1);
            let (xe, ye) = (x.len() / n, y.len() / n);
            let take = n.min((opts.tokens - have).div_ceil(ye.max(1)));
            have += take * ye;
            batches.push((x[..take * xe].to_vec(), y[..take * ye].to_vec()));
            from_steps.push(step);
        }
    }
    if path.len() < 2 {
        return Err("the run needs at least two snapshots for a plane".into());
    }

    // principal directions of the path relative to the final weights (Li et al. 2018, sec. 7):
    // u_i = θ_i − θ_final, G = U Uᵀ; d_j = Σ_i e_j[i] u_i / √λ_j, so u_i's coordinate is √λ_j e_j[i]
    let u: Vec<Vec<F>> = path.iter().map(|(_, w)| w.iter().zip(&fin).map(|(a, b)| a - b).collect()).collect();
    let k = u.len();
    let mut gram = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let g = dot(&u[i], &u[j]);
            gram[i * k + j] = g;
            gram[j * k + i] = g;
        }
    }
    let (vals, vecs) = eigen(gram, k);
    let total: F = vals.iter().map(|v| v.max(0.0)).sum();
    if !(vals[1] > 1e-12 * vals[0]) {
        return Err("the path has no second direction (all snapshots on one line)".into());
    }
    let mut dirs = Vec::new();
    let mut coords = Vec::new();
    for j in 0..2 {
        let s = vals[j].sqrt();
        // sign: the start of the path on the negative side
        let sign = if vecs[j] > 0.0 { -1.0 } else { 1.0 };
        let mut d = vec![0.0; n_params];
        for (i, ui) in u.iter().enumerate() {
            let c = sign * vecs[i * k + j] / s;
            for (dv, uv) in d.iter_mut().zip(ui) {
                *dv += c * uv;
            }
        }
        coords.push((0..k).map(|i| sign * s * vecs[i * k + j]).collect::<Vec<F>>());
        dirs.push(d);
    }
    let at = |a: F, b: F| -> Vec<F> { fin.iter().zip(&dirs[0]).zip(&dirs[1]).map(|((f, x), y)| f + a * x + b * y).collect() };

    let preds: usize = batches.iter().map(|(_, y)| y.len()).sum();
    eprintln!("landscape: {} params, {k} snapshots, plane explains {:.1}% + {:.1}% of the path; loss on {preds} predictions", n_params, 100.0 * vals[0] / total, 100.0 * vals[1] / total);

    // the path: true loss, and the loss at its shadow on the plane
    let mut path_loss = Vec::new();
    let mut plane_loss = Vec::new();
    let mut off_plane = Vec::new();
    let reach = u.iter().map(|ui| dot(ui, ui)).fold(0.0, F::max).max(1e-300);
    for (i, (_, w)) in path.iter().enumerate() {
        path_loss.push(loss(&mut *model, w, &batches));
        let shadow = at(coords[0][i], coords[1][i]);
        plane_loss.push(loss(&mut *model, &shadow, &batches));
        let gone: F = shadow.iter().zip(w).map(|(s, w)| (s - w) * (s - w)).sum();
        off_plane.push((gone / reach).sqrt());
    }
    let final_loss = loss(&mut *model, &fin, &batches);

    let a = axis(&coords[0], opts.grid);
    let b = axis(&coords[1], opts.grid);
    let mut grid = Vec::with_capacity(opts.grid * opts.grid);
    let t0 = Instant::now();
    for (r, &bv) in b.iter().enumerate() {
        for &av in &a {
            grid.push(loss(&mut *model, &at(av, bv), &batches));
        }
        let done = (r + 1) as F / b.len() as F;
        eprintln!("landscape: grid row {}/{} ({:.0}s, about {:.0}s left)", r + 1, b.len(), t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() * (1.0 / done - 1.0));
    }

    let out = object(&[
        ("version", "1".into()),
        ("run", string(&opts.run)),
        ("method", string("pca")),
        ("units", string("nats per prediction")),
        ("params", n_params.to_string()),
        ("a", packed(&a)),
        ("b", packed(&b)),
        // row-major: loss[r * a.length + c] is at (a[c], b[r])
        ("loss", packed(&grid)),
        (
            "path",
            object(&[
                ("step", ints(&path.iter().map(|(s, _)| *s).collect::<Vec<_>>())),
                ("a", packed(&coords[0])),
                ("b", packed(&coords[1])),
                ("loss", packed(&path_loss)),
                ("plane_loss", packed(&plane_loss)),
                // distance from each point to its shadow on the plane, as a fraction of the farthest point's distance to the final weights
                ("off_plane", packed(&off_plane)),
            ]),
        ),
        ("final_loss", num(final_loss)),
        ("explained", array(&vals.iter().map(|v| num64(v.max(0.0) / total)).collect::<Vec<_>>())),
        ("eval", object(&[("batches", batches.len().to_string()), ("predictions", preds.to_string()), ("from_steps", ints(&from_steps))])),
        ("seconds", num(start.elapsed().as_secs_f64())),
    ]);
    write_atomic(&dir.join("landscape.js"), &format!("window.TRACE_LANDSCAPE = {out};\n"))?;
    eprintln!("landscape: wrote {} ({:.0}s)", dir.join("landscape.js").display(), start.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_diagonalizes() {
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let (vals, vecs) = eigen(a.clone(), 3);
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2]);
        for j in 0..3 {
            for r in 0..3 {
                let av: F = (0..3).map(|c| a[r * 3 + c] * vecs[c * 3 + j]).sum();
                assert!((av - vals[j] * vecs[r * 3 + j]).abs() < 1e-10);
            }
        }
        assert!((vals.iter().sum::<F>() - 12.0).abs() < 1e-10);
    }
}
