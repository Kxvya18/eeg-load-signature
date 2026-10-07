"""
Null models for weighted complete networks with weights in [0, 1].

strength_null: Markov chain whose moves add +delta to w_ab, w_cd and -delta to
    w_ac, w_bd for four distinct vertices. Every vertex strength is preserved
    exactly, weights stay in [0, 1], and the moves are symmetric, so the chain
    targets the uniform distribution on the polytope of weighted networks with
    the observed node strengths. Clique placement is destroyed; strengths are not.
perm_null: random permutation of the off-diagonal weights (preserves the
    weight multiset, not strengths).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _strength_moves(w, n_moves, seed):
    np.random.seed(seed)
    n = w.shape[0]
    for _ in range(n_moves):
        a = np.random.randint(n)
        b = np.random.randint(n)
        c = np.random.randint(n)
        d = np.random.randint(n)
        if a == b or a == c or a == d or b == c or b == d or c == d:
            continue
        lo = max(-w[a, b], -w[c, d], w[a, c] - 1.0, w[b, d] - 1.0)
        hi = min(1.0 - w[a, b], 1.0 - w[c, d], w[a, c], w[b, d])
        if hi <= lo:
            continue
        delta = lo + (hi - lo) * np.random.random()
        w[a, b] += delta; w[b, a] = w[a, b]
        w[c, d] += delta; w[d, c] = w[c, d]
        w[a, c] -= delta; w[c, a] = w[a, c]
        w[b, d] -= delta; w[d, b] = w[b, d]
    return w


def strength_null(w: np.ndarray, rng: np.random.Generator, moves_per_edge: int = 30) -> np.ndarray:
    w = np.array(w, dtype=np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    n = w.shape[0]
    n_moves = moves_per_edge * n * (n - 1) // 2
    out = _strength_moves(w, n_moves, int(rng.integers(0, 2**31 - 1)))
    np.fill_diagonal(out, 1.0)
    return out


def perm_null(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.array(w, dtype=float, copy=True)
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = rng.permutation(w[iu])
    out = np.zeros_like(w)
    out[iu] = vals
    out = out + out.T
    np.fill_diagonal(out, 1.0)
    return out


# --------------------------------------------------------------------------- #
# v3.1: weight-multiset-exact, strength-matched null (Rubinov & Sporns 2011 style)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _swap_anneal(w, target, n_iter, temp0, seed):
    """Swap weight values between random edge pairs; accept moves that reduce the
    squared strength error (Metropolis with decreasing temperature). The weight
    multiset is preserved exactly by construction."""
    np.random.seed(seed)
    n = w.shape[0]
    s = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                s[i] += w[i, j]
    for it in range(n_iter):
        temp = temp0 * (1.0 - it / n_iter) + 1e-12
        a = np.random.randint(n); b = np.random.randint(n)
        c = np.random.randint(n); d = np.random.randint(n)
        if a == b or c == d or (a == c and b == d) or (a == d and b == c):
            continue
        x = w[a, b]; y = w[c, d]
        if x == y:
            continue
        delta = y - x                      # edge (a,b) gains delta, (c,d) loses delta
        ds = np.zeros(n)
        ds[a] += delta; ds[b] += delta; ds[c] -= delta; ds[d] -= delta
        seen = np.zeros(n, dtype=np.bool_)
        old = 0.0; new = 0.0
        for v in (a, b, c, d):
            if not seen[v]:
                seen[v] = True
                old += (s[v] - target[v]) ** 2
                new += (s[v] + ds[v] - target[v]) ** 2
        if new <= old or np.random.random() < np.exp(-(new - old) / temp):
            w[a, b] = y; w[b, a] = y; w[c, d] = x; w[d, c] = x
            for v in (a, b, c, d):
                if seen[v]:
                    s[v] += ds[v]
                    seen[v] = False
    return w


def dist_strength_null(w: np.ndarray, rng: np.random.Generator, iters_per_edge: int = 400,
                       temp0: float = 1e-3) -> np.ndarray:
    """Random network with exactly the observed off-diagonal weight multiset and node
    strengths matched by annealed weight swaps. Start: random permutation of weights."""
    w = np.array(w, dtype=np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    target = w.sum(1)
    start = perm_null(w, rng)
    np.fill_diagonal(start, 0.0)
    n = w.shape[0]
    out = _swap_anneal(start, target, iters_per_edge * n * (n - 1) // 2, temp0,
                       int(rng.integers(0, 2**31 - 1)))
    np.fill_diagonal(out, 1.0)
    return out
