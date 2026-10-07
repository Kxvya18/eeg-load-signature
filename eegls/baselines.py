"""Baseline descriptors: graph metrics, clique topology, band power, Riemannian tangent space."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch
from scipy.sparse.csgraph import shortest_path

from .nwcf import _edge_index, rips_diagrams, _finite


# ---------------------------- graph metrics -------------------------------- #
def onnela_clustering(w: np.ndarray) -> float:
    w = np.array(w, dtype=float)
    np.fill_diagonal(w, 0.0)
    mx = w.max()
    if mx <= 0:
        return 0.0
    c = np.cbrt(w / mx)
    n = w.shape[0]
    return float(np.mean(np.diag(c @ c @ c)) / ((n - 1) * (n - 2)))


def global_efficiency(d: np.ndarray) -> float:
    d = np.array(d, dtype=float)
    d = np.where(d > 1e-9, d, 1e-9)
    np.fill_diagonal(d, 0.0)
    sp = shortest_path(d, directed=False)
    iu, _ = _edge_index(d.shape[0])
    return float(np.mean(1.0 / sp[iu]))


def graph_metrics(w: np.ndarray, d: np.ndarray) -> Dict[str, float]:
    n = w.shape[0]
    iu, _ = _edge_index(n)
    wz = np.array(w, dtype=float); np.fill_diagonal(wz, 0.0)
    return {
        "graph_mean_fc": float(w[iu].mean()),
        "graph_strength_sd": float(wz.sum(1).std()),
        "graph_clustering": onnela_clustering(w),
        "graph_efficiency": global_efficiency(d),
    }


# ---------------------------- clique topology ------------------------------ #
def clique_topology(d: np.ndarray) -> Dict[str, float]:
    """Order complex (Giusti et al. 2015): Rips on rank-normalized edge order."""
    n = d.shape[0]
    iu, _ = _edge_index(n)
    ranks = np.argsort(np.argsort(d[iu])) + 1
    r = np.zeros_like(d)
    r[iu] = ranks / len(ranks)
    r = r + r.T
    h1 = _finite(rips_diagrams(r)[1])
    if len(h1) == 0:
        return {"ct_h1_total": 0.0, "ct_betti1_max": 0.0}
    grid = np.linspace(0, 1, 201)
    betti = ((h1[:, [0]] <= grid[None]) & (h1[:, [1]] > grid[None])).sum(0)
    return {"ct_h1_total": float((h1[:, 1] - h1[:, 0]).sum()),
            "ct_betti1_max": float(betti.max())}


# ---------------------------- band power ----------------------------------- #
def log_rel_bandpower(x: np.ndarray, fs: int, bands: Dict[str, tuple]) -> Dict[str, np.ndarray]:
    """x: [T, N] broadband window. Returns {band: log relative power [N]}."""
    f, pxx = welch(x, fs=fs, nperseg=min(256, x.shape[0]), axis=0)
    tot = pxx[(f >= 1) & (f <= 30)].sum(0)
    return {b: np.log(pxx[(f >= lo) & (f < hi)].sum(0) / tot) for b, (lo, hi) in bands.items()}


# ---------------------------- Riemannian ----------------------------------- #
def _eig_fn(c, fn):
    vals, vecs = np.linalg.eigh(c)
    return (vecs * fn(vals)) @ vecs.T


def riemann_mean(covs: np.ndarray, n_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    m = np.mean(covs, axis=0)
    for _ in range(n_iter):
        m_isqrt = _eig_fn(m, lambda v: 1 / np.sqrt(v))
        m_sqrt = _eig_fn(m, np.sqrt)
        t = np.mean([_eig_fn(m_isqrt @ c @ m_isqrt, np.log) for c in covs], axis=0)
        m = m_sqrt @ _eig_fn(t, np.exp) @ m_sqrt
        if np.linalg.norm(t) < tol:
            break
    return m


def tangent_space(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n = ref.shape[0]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    r_isqrt = _eig_fn(ref, lambda v: 1 / np.sqrt(v))
    return np.array([_eig_fn(r_isqrt @ c @ r_isqrt, np.log)[iu] * coef for c in covs])


class TangentSpace:
    """Fit the reference on training covariances only (no leakage)."""

    def fit(self, covs):
        self.ref_ = riemann_mean(covs)
        return self

    def transform(self, covs):
        return tangent_space(covs, self.ref_)
