//! Minimal JSON writing (no serde) for the training trace the viewer reads.

use crate::autograd::{Graph, Op};
use crate::tensor::F;

/// A number, rounded to f32 precision to keep traces small. Non-finite -> null,
/// including values that are finite in f64 but overflow f32.
pub fn num(x: F) -> String {
    let y = x as f32;
    if y.is_finite() { format!("{y}") } else { "null".to_string() }
}

/// A number at full f64 precision, for settings like the learning rate. Non-finite -> null.
pub fn num64(x: F) -> String {
    if !x.is_finite() {
        "null".to_string()
    } else if x == 0.0 || (1e-6..1e16).contains(&x.abs()) {
        format!("{x}")
    } else {
        format!("{x:e}")
    }
}

pub fn nums(xs: &[F]) -> String {
    format!("[{}]", xs.iter().map(|&x| num(x)).collect::<Vec<_>>().join(","))
}

/// Arrays of at least this many numbers are written packed (see `packed`).
const PACK_FROM: usize = 32;

/// Like `nums`, but a long array is written as {"f32": base64}: the same f32 values as
/// little-endian bytes, about 5.3 characters per number instead of about 11-16, and much
/// faster to write and to read. The viewer unpacks it on load (non-finite values become
/// null, as in `nums`).
pub fn packed(xs: &[F]) -> String {
    if xs.len() < PACK_FROM {
        return nums(xs);
    }
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&(x as f32).to_le_bytes());
    }
    format!("{{\"f32\":\"{}\"}}", base64(&bytes))
}

/// Every number at full f64 precision, as `{"f64":"<base64 of little-endian f64s>"}`
/// (for the checkpoint, which the viewer's own forward pass must reproduce exactly).
pub fn packed64(xs: &[F]) -> String {
    let bytes: Vec<u8> = xs.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect();
    format!("{{\"f64\":\"{}\"}}", base64(&bytes))
}

/// Standard base64 with padding.
pub fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for ch in bytes.chunks(3) {
        let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            if k <= ch.len() {
                out.push(A[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn ints(xs: &[usize]) -> String {
    format!("[{}]", xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `{"k": v, ...}` from already-encoded values.
pub fn object(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("{}:{}", string(k), v)).collect();
    format!("{{{}}}", body.join(","))
}

pub fn array(items: &[String]) -> String {
    format!("[{}]", items.join(","))
}

/// How much of each snapshot to write.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Detail {
    /// Every node's full value and gradient, plus optimizer state.
    Full,
    /// Only the probe example's rows of each per-example node; parameters and optimizer
    /// state still in full (they are small next to the batch's intermediate values).
    Summary,
}

impl Detail {
    pub fn name(self) -> &'static str {
        match self {
            Detail::Full => "full",
            Detail::Summary => "summary",
        }
    }
}

/// The settings an op was built with, so the viewer can recompute it: the block size of
/// the causal softmax, the head layout, 1/√d_head, and normalize's per-row mean and
/// 1/√(var + eps) (only the kept rows). null for ops without settings.
fn op_attrs(op: &Op, kept: Option<(usize, usize)>) -> String {
    match op {
        Op::Normalize { eps, mean, rstd, .. } => {
            let (a, b) = kept.unwrap_or((0, 0));
            object(&[("eps", num64(*eps)), ("mean", packed(&mean[a..b])), ("rstd", packed(&rstd[a..b]))])
        }
        Op::CausalSoftmax { t, .. } => object(&[("t", t.to_string())]),
        Op::SplitHeads { n, t, heads, .. } | Op::MergeHeads { n, t, heads, .. } => object(&[("n", n.to_string()), ("t", t.to_string()), ("heads", heads.to_string())]),
        Op::BatchMatMul { batches, trans_b, .. } => object(&[("batches", batches.to_string()), ("trans_b", trans_b.to_string())]),
        Op::Scale { by, .. } => object(&[("by", num64(*by))]),
        _ => "null".to_string(),
    }
}

/// Every node on the tape: what op made it, from which inputs, its value and its gradient.
///
/// In `Summary` detail, a node with n·k rows for some k in `per_example` (1 row per
/// example, one per context character, one per head and position, ...) keeps only the
/// probe's rows (example 0, the first k), and `kept_rows` says which. Parameters and
/// scalars are kept whole; other nodes keep nothing (null). The kept gradient rows are that
/// example's slice of the batch gradient (already divided by the batch's prediction
/// count), not a gradient of the example on its own.
pub fn graph_nodes(g: &Graph, detail: Detail, n: usize, per_example: &[usize]) -> String {
    let nodes: Vec<String> = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| {
            let (r, c) = (nd.value.rows, nd.value.cols);
            let kept: Option<(usize, usize)> = match detail {
                Detail::Full => Some((0, r)),
                Detail::Summary if nd.is_param => Some((0, r)),
                Detail::Summary if nd.value.len() == 1 => Some((0, 1)),
                Detail::Summary => per_example.iter().find(|&&k| r == n * k).map(|&k| (0, k)),
            };
            let slice = |t: &crate::tensor::Tensor| match kept {
                Some((a, b)) => packed(&t.data[a * c..b * c]),
                None => "null".to_string(),
            };
            object(&[
                ("id", i.to_string()),
                ("name", string(&nd.name)),
                ("op", string(nd.op.kind())),
                ("inputs", ints(&nd.op.inputs().iter().map(|v| v.0).collect::<Vec<_>>())),
                ("shape", ints(&[r, c])),
                ("param", nd.is_param.to_string()),
                ("kept_rows", match kept {
                    Some((a, b)) => ints(&[a, b]),
                    None => "null".to_string(),
                }),
                ("value", slice(&nd.value)),
                ("grad", if nd.requires_grad { slice(&nd.grad) } else { "null".to_string() }),
                ("attrs", op_attrs(&nd.op, kept)),
            ])
        })
        .collect();
    array(&nodes)
}

/// Write via a temporary file in the same directory, then rename, so a reader
/// (the live server or the browser) never sees a half-written file.
pub fn write_atomic(path: &std::path::Path, contents: &str) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, contents).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("renaming {} -> {}: {e}", tmp.display(), path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_always_valid_json() {
        assert_eq!(num(0.5), "0.5");
        assert_eq!(num(1e300), "null"); // finite in f64, infinite as f32
        assert_eq!(num(F::NAN), "null");
        assert_eq!(num64(3e-3), "0.003");
        assert_eq!(num64(1e300), "1e300");
        assert_eq!(num64(F::INFINITY), "null");
    }

    #[test]
    fn base64_matches_the_standard_vectors() {
        for (input, out) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(input.as_bytes()), out);
        }
        assert_eq!(packed(&[1.0; 3]), "[1,1,1]");
        assert!(packed(&[1.0; 40]).starts_with("{\"f32\":\"AACAPwAAgD8"));
    }
}
