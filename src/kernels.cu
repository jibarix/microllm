// GPU kernels for the f32 tape in gpu.rs. Every op the CPU tape (autograd.rs) has for the
// models, forward and backward, with the same meaning. Compiled to PTX by build.rs.
//
// Determinism: no atomics. Every output number is written by exactly one thread, and every
// sum is taken in a fixed order, so the same inputs always give the same bits.
//
// Backward kernels accumulate (y += ...) into gradients, like `accumulate` on the CPU.

#define WARP 32

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
#define ELEMENTWISE int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
// one warp per row: blockDim.x = 256 = 8 rows per block
#define ROW_PER_WARP int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP, lane = threadIdx.x % WARP; if (r >= rows) return;

extern "C" {

// ---------- elementwise ----------
__global__ void k_add(float* out, const float* a, const float* b, int n) { ELEMENTWISE out[i] = a[i] + b[i]; }
__global__ void k_scale(float* out, const float* a, float s, int n) { ELEMENTWISE out[i] = a[i] * s; }
__global__ void k_copy(float* out, const float* a, int n) { ELEMENTWISE out[i] = a[i]; }
// y += s·x
__global__ void k_acc(float* y, const float* x, float s, int n) { ELEMENTWISE y[i] += s * x[i]; }
__global__ void k_tanh(float* out, const float* a, int n) { ELEMENTWISE out[i] = tanhf(a[i]); }
// y += g·(1 − out²)
__global__ void k_tanh_bwd(float* y, const float* g, const float* out, int n) { ELEMENTWISE y[i] += g[i] * (1.0f - out[i] * out[i]); }

// GELU, tanh approximation, and its derivative (as `gelu` in autograd.rs).
__device__ __forceinline__ void gelu(float x, float* v, float* d) {
    const float C = 0.7978845608028654f;
    float u = C * (x + 0.044715f * x * x * x), th = tanhf(u), du = C * (1.0f + 3.0f * 0.044715f * x * x);
    *v = 0.5f * x * (1.0f + th);
    *d = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * du;
}
__global__ void k_gelu(float* out, const float* a, int n) { ELEMENTWISE float v, d; gelu(a[i], &v, &d); out[i] = v; }
__global__ void k_gelu_bwd(float* y, const float* g, const float* x, int n) { ELEMENTWISE float v, d; gelu(x[i], &v, &d); y[i] += g[i] * d; }
// relu(x)² and its derivative 2·relu(x).
__global__ void k_relu2(float* out, const float* a, int n) { ELEMENTWISE float x = fmaxf(a[i], 0.0f); out[i] = x * x; }
__global__ void k_relu2_bwd(float* y, const float* g, const float* x, int n) { ELEMENTWISE y[i] += g[i] * (2.0f * fmaxf(x[i], 0.0f)); }

// ---------- row vectors ----------
__global__ void k_add_row(float* out, const float* a, const float* b, int n, int cols) { ELEMENTWISE out[i] = a[i] + b[i % cols]; }
__global__ void k_mul_row(float* out, const float* a, const float* g, int n, int cols) { ELEMENTWISE out[i] = a[i] * g[i % cols]; }
// y += g·gain (the row vector)
__global__ void k_acc_mul_row(float* y, const float* g, const float* gain, int n, int cols) { ELEMENTWISE y[i] += g[i] * gain[i % cols]; }
// y[c] += Σ_r g[r][c] (· x[r][c] when x is given), rows summed in a fixed order.
// Block (32, 8): column blockIdx.x·32 + tx; the 8 ty stripes are added in order at the end.
__global__ void k_colsum(float* y, const float* g, const float* x, int rows, int cols) {
    __shared__ float part[8][32];
    int c = blockIdx.x * 32 + threadIdx.x, ty = threadIdx.y;
    float s = 0.0f;
    if (c < cols)
        for (int r = ty; r < rows; r += 8) s += x ? g[r * cols + c] * x[r * cols + c] : g[r * cols + c];
    part[ty][threadIdx.x] = s;
    __syncthreads();
    if (ty == 0 && c < cols) {
        float t = 0.0f;
        for (int k = 0; k < 8; k++) t += part[k][threadIdx.x];
        y[c] += t;
    }
}

// ---------- embedding ----------
__global__ void k_gather(float* out, const float* table, const int* ids, int n, int cols) { ELEMENTWISE out[i] = table[ids[i / cols] * cols + (i % cols)]; }
// Table gradient: row v gets the gradient rows order[offsets[v] .. offsets[v+1]] (every row
// whose id is v, in increasing row order), so each number has one writer and a fixed order.
__global__ void k_scatter_rows(float* tg, const float* g, const int* offsets, const int* order, int vocab, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab * cols) return;
    int v = i / cols, c = i % cols;
    float s = 0.0f;
    for (int k = offsets[v]; k < offsets[v + 1]; k++) s += g[order[k] * cols + c];
    tg[i] += s;
}

// ---------- heads ----------
// Element k of the (n·H·T) x dh split layout ↔ element map(k) of the (n·T) x (H·dh) layout
// (`head_map` in autograd.rs). gather: dst[k] (+)= src[map(k)]; otherwise dst[map(k)] (+)= src[k].
// Either way each destination number has exactly one source.
__global__ void k_heads(float* dst, const float* src, int t, int heads, int dh, int gather, int acc, int n) {
    ELEMENTWISE
    int j = i % dh, row = i / dh, p = row % t, bh = row / t, h = bh % heads, b = bh / heads;
    int m = (b * t + p) * (heads * dh) + h * dh + j;
    int d = gather ? i : m, s = gather ? m : i;
    dst[d] = acc ? dst[d] + src[s] : src[s];
}

// ---------- layernorm core ----------
// out = (x − mean)·rstd per row, rstd = 1/√(var + eps), population variance.
__global__ void k_normalize(float* out, float* mean, float* rstd, const float* x, int rows, int cols, float eps) {
    ROW_PER_WARP
    const float* xr = x + (size_t)r * cols;
    float s = 0.0f;
    for (int c = lane; c < cols; c += WARP) s += xr[c];
    float m = warp_sum(s) / cols;
    float q = 0.0f;
    for (int c = lane; c < cols; c += WARP) q += (xr[c] - m) * (xr[c] - m);
    float rs = rsqrtf(warp_sum(q) / cols + eps);
    for (int c = lane; c < cols; c += WARP) out[(size_t)r * cols + c] = (xr[c] - m) * rs;
    if (lane == 0) { mean[r] = m; rstd[r] = rs; }
}
// dx += rstd·(g − mean(g) − y·mean(g·y)), y the normalized output.
__global__ void k_normalize_bwd(float* dx, const float* g, const float* y, const float* rstd, int rows, int cols) {
    ROW_PER_WARP
    const float *gr = g + (size_t)r * cols, *yr = y + (size_t)r * cols;
    float a = 0.0f, b = 0.0f;
    for (int c = lane; c < cols; c += WARP) { a += gr[c]; b += gr[c] * yr[c]; }
    float gm = warp_sum(a) / cols, gym = warp_sum(b) / cols;
    for (int c = lane; c < cols; c += WARP) dx[(size_t)r * cols + c] += rstd[r] * (gr[c] - gm - yr[c] * gym);
}

// ---------- attention ----------
// Row r is position r % t and sees columns 0 ..= r % t; the rest are exactly 0.
__global__ void k_causal_softmax(float* out, const float* x, int rows, int t) {
    ROW_PER_WARP
    int seen = r % t + 1;
    const float* xr = x + (size_t)r * t;
    float mx = -__int_as_float(0x7f800000);
    for (int c = lane; c < seen; c += WARP) mx = fmaxf(mx, xr[c]);
    mx = warp_max(mx);
    float s = 0.0f;
    for (int c = lane; c < seen; c += WARP) s += expf(xr[c] - mx);
    s = warp_sum(s);
    for (int c = lane; c < t; c += WARP) out[(size_t)r * t + c] = c < seen ? expf(xr[c] - mx) / s : 0.0f;
}
// dx_j += p_j·(g_j − Σ_k p_k·g_k) on the allowed columns.
__global__ void k_causal_softmax_bwd(float* dx, const float* g, const float* p, int rows, int t) {
    ROW_PER_WARP
    int seen = r % t + 1;
    const float *gr = g + (size_t)r * t, *pr = p + (size_t)r * t;
    float d = 0.0f;
    for (int c = lane; c < seen; c += WARP) d += pr[c] * gr[c];
    d = warp_sum(d);
    for (int c = lane; c < seen; c += WARP) dx[(size_t)r * t + c] += pr[c] * (gr[c] - d);
}

// ---------- cross-entropy ----------
// loss[r] = logsumexp(row) − row[target]
__global__ void k_ce_rows(float* loss, const float* logits, const int* targets, int rows, int cols) {
    ROW_PER_WARP
    const float* lr = logits + (size_t)r * cols;
    float mx = -__int_as_float(0x7f800000);
    for (int c = lane; c < cols; c += WARP) mx = fmaxf(mx, lr[c]);
    mx = warp_max(mx);
    float s = 0.0f;
    for (int c = lane; c < cols; c += WARP) s += expf(lr[c] - mx);
    s = warp_sum(s);
    if (lane == 0) loss[r] = logf(s) + mx - lr[targets[r]];
}
// dlogits += (softmax − onehot(target)) · gloss[0] / rows
__global__ void k_ce_bwd(float* dl, const float* logits, const int* targets, const float* gloss, int rows, int cols) {
    ROW_PER_WARP
    const float* lr = logits + (size_t)r * cols;
    float mx = -__int_as_float(0x7f800000);
    for (int c = lane; c < cols; c += WARP) mx = fmaxf(mx, lr[c]);
    mx = warp_max(mx);
    float s = 0.0f;
    for (int c = lane; c < cols; c += WARP) s += expf(lr[c] - mx);
    s = warp_sum(s);
    float k = gloss[0] / rows;
    for (int c = lane; c < cols; c += WARP) dl[(size_t)r * cols + c] += (expf(lr[c] - mx) / s - (c == targets[r] ? 1.0f : 0.0f)) * k;
}
// out[0] = scale · Σ x, one block of 256 threads, summed in double in a fixed order.
__global__ void k_sum(float* out, const float* x, int n, float scale) {
    __shared__ double part[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += 256) s += x[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = (float)(part[0] * scale);
}

// ---------- matrix products ----------
// For z = blockIdx.z: batch b = z / splits, split s = z % splits.
//   C_z (M x N) = beta·C_z + op(A_b)·op(B_b) over k in [s·kchunk, (s+1)·kchunk) ∩ [0, K)
// A_b(i, k) = A[b·sa + i·a_r + k·a_c], B_b(k, j) = B[b·sb + k·b_r + j·b_c], C_z(i, j) = C[z·sc + i·ldc + j],
// so any transpose is just a choice of strides. With R non-null (no splits) the result is
// alpha·op(A)·op(B) + rbeta·R_z instead, R_z laid out like C_z (Muon's Newton–Schulz). With splits > 1, C is a workspace of partial
// products that k_sum_splits adds up in order. 64x64 tile per block, 16 deep; thread (tx, ty)
// owns rows ty + 16i and columns tx + 16j, i, j < 4.
__global__ void k_gemm(float* C, const float* A, const float* B, int M, int N, int K,
                       int a_r, int a_c, int b_r, int b_c, int ldc,
                       long long sa, long long sb, long long sc, int splits, int kchunk, float beta,
                       float alpha, const float* R, float rbeta) {
    __shared__ float As[16][64 + 4];
    __shared__ float Bs[16][64 + 4];
    int z = blockIdx.z, b = z / splits, s = z % splits;
    A += b * sa; B += b * sb; C += z * sc;
    if (R) R += z * sc;
    int k0 = s * kchunk, k1 = min(K, k0 + kchunk);
    int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    float acc[4][4] = {};
    for (int kt = k0; kt < k1; kt += 16) {
        for (int q = 0; q < 4; q++) {
            int idx = tid + q * 256;
            // A tile: 64 rows x 16 k; walk k fastest when A is row-major in k (a_c == 1)
            int ar, ak;
            if (a_c == 1) { ar = idx / 16; ak = idx % 16; } else { ak = idx / 64; ar = idx % 64; }
            int gi = m0 + ar, gk = kt + ak;
            As[ak][ar] = (gi < M && gk < k1) ? A[(long long)gi * a_r + (long long)gk * a_c] : 0.0f;
            // B tile: 16 k x 64 cols; walk cols fastest when B is row-major in j (b_c == 1)
            int bk, bj;
            if (b_c == 1) { bk = idx / 64; bj = idx % 64; } else { bj = idx / 16; bk = idx % 16; }
            int gk2 = kt + bk, gj = n0 + bj;
            Bs[bk][bj] = (gk2 < k1 && gj < N) ? B[(long long)gk2 * b_r + (long long)gj * b_c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; k++) {
            float a[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[k][ty + 16 * i]; bv[i] = Bs[k][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * bv[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        int gi = m0 + ty + 16 * i;
        if (gi >= M) continue;
        for (int j = 0; j < 4; j++) {
            int gj = n0 + tx + 16 * j;
            if (gj >= N) continue;
            float* c = C + (long long)gi * ldc + gj;
            if (R) *c = alpha * acc[i][j] + rbeta * R[(long long)gi * ldc + gj];
            else *c = beta == 0.0f ? acc[i][j] : beta * *c + acc[i][j];
        }
    }
}
// C[i] = beta·C[i] + Σ_s W[s·n + i], splits added in order.
__global__ void k_sum_splits(float* C, const float* W, int splits, int n, float beta) {
    ELEMENTWISE
    float s = 0.0f;
    for (int k = 0; k < splits; k++) s += W[(long long)k * n + i];
    C[i] = beta == 0.0f ? s : beta * C[i] + s;
}

// ---------- optimizer ----------
// One optimizer step on one parameter, in double, with every operation rounded on its own
// (the _rn intrinsics are never fused into multiply-adds), in the same order as the CPU
// optimizer, so every number is bit-identical to it. p, m, v: master value and Adam moments;
// u: the change applied; p32: the value the next forward pass reads; g: the gradient.
//   AdamW: m ← β1·m + (1−β1)·g,  v ← β2·v + (1−β2)·g·g,
//          u = −lr·( (m/bc1) / (√(v/bc2) + ε) + [decay] wd·p )
//   SGD (adam = 0): u = −lr·g
__global__ void k_adamw(double* p, double* m, double* v, double* u, float* p32, const float* g, const int* decay, int n,
                        double lr, double b1, double b2, double eps, double wd, double bc1, double bc2, int adam) {
    ELEMENTWISE
    if (decay[i] & 2) return;  // Muon's (k_muon_apply)
    double gi = (double)g[i], before = p[i], du;
    if (adam) {
        double mi = __dadd_rn(__dmul_rn(b1, m[i]), __dmul_rn(__dsub_rn(1.0, b1), gi));
        double vi = __dadd_rn(__dmul_rn(b2, v[i]), __dmul_rn(__dmul_rn(__dsub_rn(1.0, b2), gi), gi));
        m[i] = mi;
        v[i] = vi;
        double step = __ddiv_rn(__ddiv_rn(mi, bc1), __dadd_rn(__dsqrt_rn(__ddiv_rn(vi, bc2)), eps));
        double dec = (decay[i] & 1) ? __dmul_rn(wd, before) : 0.0;
        du = __dmul_rn(-lr, __dadd_rn(step, dec));
    } else {
        du = __dmul_rn(-lr, gi);
    }
    u[i] = du;
    double after = __dadd_rn(before, du);
    p[i] = after;
    p32[i] = __double2float_rn(after);
}

// out[b] = Σ x[i]² over segment b = blockIdx.x (offset off[b], length len[b]), summed in double
// in a fixed order by one block; x is double if f64 != 0, else float.
__global__ void k_sumsq_segs(double* out, const void* x, const int* off, const int* len, int f64) {
    __shared__ double part[256];
    double s = 0.0;
    int n = len[blockIdx.x];
    for (int k = threadIdx.x; k < n; k += 256) {
        long long i = (long long)off[blockIdx.x] + k;
        double a = f64 ? ((const double*)x)[i] : (double)((const float*)x)[i];
        s += a * a;
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}

// The f32 copy of a double array.
__global__ void k_to_f32(float* out, const double* a, int n) { ELEMENTWISE out[i] = __double2float_rn(a[i]); }

// ---------- Muon ----------
// For each element with w[i] >= 0 (a Muon matrix): momentum m ← μ·m + g, and X[w[i]] = g + μ·m,
// the Nesterov input to Newton–Schulz (w puts each matrix in its wide orientation).
__global__ void k_muon_mom(double* m, const float* g, float* X, const int* w, int n, double mu) {
    ELEMENTWISE
    if (w[i] < 0) return;
    double gi = (double)g[i], mi = __dadd_rn(__dmul_rn(mu, m[i]), gi);
    m[i] = mi;
    X[w[i]] = (float)__dadd_rn(gi, __dmul_rn(mu, mi));
}
// X[i] /= (‖matrix‖ + 1e-7), matrix = i / per; sq holds each matrix's sum of squares.
__global__ void k_scale_mats(float* X, const double* sq, int per, int n) {
    ELEMENTWISE
    X[i] = (float)((double)X[i] / (sqrt(sq[i / per]) + 1e-7));
}
// The Muon update for each element with w[i] >= 0: u = −lr·(s[i]·O[w[i]] + wd·p).
__global__ void k_muon_apply(double* p, double* u, float* p32, const float* O, const int* w, const float* s, int n, double lr, double wd) {
    ELEMENTWISE
    if (w[i] < 0) return;
    double before = p[i];
    double du = -lr * ((double)s[i] * (double)O[w[i]] + wd * before);
    u[i] = du;
    p[i] = before + du;
    p32[i] = (float)p[i];
}

}
