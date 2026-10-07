//! Splitting work over threads without changing a single result.
//!
//! Each output chunk is computed by exactly one thread, with the same loop order as the
//! one-thread code, so the numbers are bit-identical for any thread count. Reductions are
//! never split across threads (that would change the order of additions).

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::F;

static THREADS: AtomicUsize = AtomicUsize::new(1);

/// Below this many multiply-adds an op runs on one thread: starting threads costs more.
const MIN_WORK: usize = 1 << 17;

pub fn set_threads(n: usize) {
    THREADS.store(n.max(1), Ordering::Relaxed);
}

pub fn threads() -> usize {
    THREADS.load(Ordering::Relaxed)
}

/// The machine's core count (the default for --threads).
pub fn available() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn threads_for(work: usize, items: usize) -> usize {
    if work < MIN_WORK { 1 } else { threads().min(items).max(1) }
}

/// Call `f(i, chunk_i)` for each of the `n` equal chunks of `out`, spread over threads.
/// `work` is the total multiply-adds, used only to decide whether threads are worth it.
pub fn chunks(out: &mut [F], n: usize, work: usize, f: impl Fn(usize, &mut [F]) + Sync) {
    if n == 0 {
        return;
    }
    let len = out.len() / n;
    let t = threads_for(work, n);
    if t <= 1 || len == 0 {
        for (i, c) in out.chunks_mut(len.max(1)).enumerate().take(n) {
            f(i, c);
        }
        return;
    }
    let per = n.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        for (ti, block) in out.chunks_mut(per * len).enumerate() {
            s.spawn(move || {
                for (j, c) in block.chunks_mut(len).enumerate() {
                    f(ti * per + j, c);
                }
            });
        }
    });
}

/// Like `chunks`, for two outputs split into the same number of chunks (chunk i of each
/// goes to the same call).
pub fn chunks2(a: &mut [F], b: &mut [F], n: usize, work: usize, f: impl Fn(usize, &mut [F], &mut [F]) + Sync) {
    if n == 0 {
        return;
    }
    let (la, lb) = (a.len() / n, b.len() / n);
    let t = threads_for(work, n);
    if t <= 1 || la == 0 || lb == 0 {
        for (i, (ca, cb)) in a.chunks_mut(la.max(1)).zip(b.chunks_mut(lb.max(1))).enumerate().take(n) {
            f(i, ca, cb);
        }
        return;
    }
    let per = n.div_ceil(t);
    let f = &f;
    std::thread::scope(|s| {
        for (ti, (ba, bb)) in a.chunks_mut(per * la).zip(b.chunks_mut(per * lb)).enumerate() {
            s.spawn(move || {
                for (j, (ca, cb)) in ba.chunks_mut(la).zip(bb.chunks_mut(lb)).enumerate() {
                    f(ti * per + j, ca, cb);
                }
            });
        }
    });
}

/// `items.iter().map(f).collect()`, with the items handed out to threads one at a time.
/// Each result is computed by one thread exactly as on its own, and they come back in order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = threads().min(items.len());
    if n <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut out: Vec<(usize, R)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..n)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            return done;
                        }
                        done.push((i, f(&items[i])));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    out.sort_unstable_by_key(|r| r.0);
    out.into_iter().map(|r| r.1).collect()
}
