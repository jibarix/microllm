//! The parameter registry: every trainable tensor, created once, with a stable id
//! and its optimizer state. Forward passes only make graph leaves from it; they
//! never create or re-register parameters.

use crate::autograd::{Tape, Var};
use crate::tensor::Tensor;

/// Stable handle to a parameter: its index in `Params::list`, fixed at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamId(pub usize);

pub struct Param {
    pub name: String,
    pub value: Tensor,
    /// Whether AdamW's weight decay applies (weight matrices yes; biases and embeddings no).
    pub decay: bool,
    /// Adam's running averages of the gradient (m) and squared gradient (v). Same shape as `value`.
    pub m: Tensor,
    pub v: Tensor,
}

#[derive(Default)]
pub struct Params {
    pub list: Vec<Param>,
}

impl Params {
    pub fn add(&mut self, name: &str, value: Tensor, decay: bool) -> ParamId {
        assert!(self.list.iter().all(|p| p.name != name), "duplicate parameter name {name:?}");
        let (r, c) = (value.rows, value.cols);
        self.list.push(Param { name: name.to_string(), value, decay, m: Tensor::zeros(r, c), v: Tensor::zeros(r, c) });
        ParamId(self.list.len() - 1)
    }

    /// Total number of trainable numbers.
    pub fn count(&self) -> usize {
        self.list.iter().map(|p| p.value.len()).sum()
    }

    /// One graph leaf per parameter, in id order, so `vars[id.0]` is parameter `id`.
    pub fn leaves(&self, g: &mut dyn Tape) -> Vec<Var> {
        self.list.iter().map(|p| g.param(&p.name, p.value.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Graph;

    #[test]
    #[should_panic(expected = "duplicate parameter name")]
    fn duplicate_names_are_rejected() {
        let mut p = Params::default();
        p.add("W", Tensor::zeros(2, 2), true);
        p.add("W", Tensor::zeros(2, 2), true);
    }

    #[test]
    fn leaves_follow_id_order() {
        let mut p = Params::default();
        let a = p.add("a", Tensor::scalar(1.0), false);
        let b = p.add("b", Tensor::scalar(2.0), false);
        let mut g = Graph::new();
        let vars = p.leaves(&mut g);
        assert_eq!(g.value(vars[a.0]).item(), 1.0);
        assert_eq!(g.value(vars[b.0]).item(), 2.0);
        assert_eq!(p.list[b.0].name, "b");
    }
}
