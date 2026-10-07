//! A 2-D tensor (matrix) stored row-major. Every value in the model lives in one of these.

/// Scalar type for all math. f64 keeps gradient checks exact enough to trust;
/// we can drop to f32 when we scale up.
pub type F = f64;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<F>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(data.len(), rows * cols, "tensor data length does not match shape {rows}x{cols}");
        Tensor { rows, cols, data }
    }

    pub fn scalar(x: F) -> Self {
        Tensor::from_vec(1, 1, vec![x])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn at(&self, r: usize, c: usize) -> F {
        self.data[r * self.cols + c]
    }

    pub fn row(&self, r: usize) -> &[F] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [F] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.rows == other.rows && self.cols == other.cols
    }

    /// Euclidean (L2) norm over all elements.
    pub fn norm(&self) -> F {
        self.data.iter().map(|x| x * x).sum::<F>().sqrt()
    }

    /// Scalar value of a 1x1 tensor.
    pub fn item(&self) -> F {
        assert_eq!(self.len(), 1, "item() needs a 1x1 tensor, got {}x{}", self.rows, self.cols);
        self.data[0]
    }
}

/// Row-wise softmax, shifted by each row's max for numerical stability.
pub fn softmax_rows(x: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        let row = x.row(r);
        let max = row.iter().cloned().fold(F::NEG_INFINITY, F::max);
        let exps: Vec<F> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: F = exps.iter().sum();
        for (o, e) in out.row_mut(r).iter_mut().zip(exps) {
            *o = e / sum;
        }
    }
    out
}
