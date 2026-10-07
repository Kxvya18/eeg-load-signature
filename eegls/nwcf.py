"""
NWCF v3 — Neurophysiological Weighted Clique Filtration.

    gamma^(q)_ij = min over q-cliques C containing {i,j} of M_p({D_ab : a<b in C})
    D_hat_ij     = min(D_ij, min_{q in Q} gamma^(q)_ij)
    NWCF_eps(D)  = VR_eps(D_hat)

M_p is the power mean over the C(q,2) clique edges. For p >= 1 the operator
D -> D_hat is 1-Lipschitz in the sup norm (Theorem 4), so NWCF diagrams are
bottleneck-stable. p = 0 (geometric mean) reproduces the legacy v2 rule and is
only Hoelder-continuous near zero; it is kept for ablation only.

All clique enumeration is precomputed per N and fully vectorized.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np
from ripser import ripser

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Clique index structures
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _edge_index(n: int):
    iu = np.triu_indices(n, k=1)
    eid = -np.ones((n, n), dtype=int)
    eid[iu] = np.arange(len(iu[0]))
    eid[(iu[1], iu[0])] = np.arange(len(iu[0]))
    return iu, eid


@lru_cache(maxsize=None)
def _clique_tables(n: int, q: int):
    """Return (cliques[C,q], clique_edges[C,m], edge_cliques[E, c_per_edge])."""
    iu, eid = _edge_index(n)
    n_edges = len(iu[0])
    cliques = np.array(list(combinations(range(n), q)), dtype=int)
    if len(cliques) == 0:
        return cliques, np.empty((0, 0), int), np.empty((n_edges, 0), int)
    pairs = list(combinations(range(q), 2))
    clique_edges = np.stack([eid[cliques[:, a], cliques[:, b]] for a, b in pairs], axis=1)
    members = [[] for _ in range(n_edges)]
    for c, edges in enumerate(clique_edges):
        for e in edges:
            members[e].append(c)
    edge_cliques = np.array(members, dtype=int)
    return cliques, clique_edges, edge_cliques


def power_mean(x: np.ndarray, p: float, axis: int = -1) -> np.ndarray:
    if np.isinf(p):
        return np.max(x, axis=axis)
    if p == 0:
        return np.exp(np.mean(np.log(np.clip(x, EPS, None)), axis=axis))
    if p == 1:
        return np.mean(x, axis=axis)
    return np.mean(np.power(x, p), axis=axis) ** (1.0 / p)


def to_dissimilarity(w: np.ndarray) -> np.ndarray:
    d = 1.0 - np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


# --------------------------------------------------------------------------- #
# The operator
# --------------------------------------------------------------------------- #
def nwcf_operator(d: np.ndarray, p: float = 1.0, orders: Sequence[int] = (3, 4),
                  return_details: bool = False):
    """Compute D_hat. With return_details, also return per-edge supports and
    the clique that realises each edge's entry value (for mediator maps)."""
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    iu, _ = _edge_index(n)
    raw = d[iu]
    best = raw.copy()
    best_order = np.zeros(len(raw), dtype=int)      # 0 = direct edge kept
    best_clique = -np.ones(len(raw), dtype=int)
    supports = {}
    for q in orders:
        if n < q:
            supports[q] = np.full(len(raw), np.inf)
            continue
        cliques, cedges, ecl = _clique_tables(n, q)
        vals = power_mean(raw[cedges], p, axis=1)          # [C]
        per_edge = vals[ecl]                               # [E, c_per_edge]
        arg = np.argmin(per_edge, axis=1)
        gam = per_edge[np.arange(len(raw)), arg]
        supports[q] = gam
        better = gam < best - 1e-12
        best = np.where(better, gam, best)
        best_order = np.where(better, q, best_order)
        best_clique = np.where(better, ecl[np.arange(len(raw)), arg], best_clique)
    d_hat = np.zeros_like(d)
    d_hat[iu] = best
    d_hat = d_hat + d_hat.T
    if not return_details:
        return d_hat
    return d_hat, {"raw": raw, "hat": best, "supports": supports,
                   "best_order": best_order, "best_clique": best_clique}


def mediator_map(n: int, det: Dict) -> np.ndarray:
    """Share of total shrinkage attributed to each vertex acting as a mediator
    (clique vertices other than the advanced edge's endpoints)."""
    iu, _ = _edge_index(n)
    shrink = det["raw"] - det["hat"]
    m = np.zeros(n)
    for e in np.nonzero(shrink > 1e-12)[0]:
        q = det["best_order"][e]
        cl = _clique_tables(n, q)[0][det["best_clique"][e]]
        med = [v for v in cl if v != iu[0][e] and v != iu[1][e]]
        m[med] += shrink[e] / len(med)
    tot = m.sum()
    return m / tot if tot > 0 else m


# --------------------------------------------------------------------------- #
# Persistence summaries
# --------------------------------------------------------------------------- #
def rips_diagrams(d: np.ndarray, maxdim: int = 1):
    d = np.asarray(d, dtype=float)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return ripser(np.clip(d, 0, None), maxdim=maxdim, distance_matrix=True)["dgms"]


def _finite(dgm):
    dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
    dgm = dgm[np.isfinite(dgm[:, 1])]
    return dgm[(dgm[:, 1] - dgm[:, 0]) > 1e-12]


def ph_summary(d: np.ndarray, prefix: str) -> Dict[str, float]:
    dg = rips_diagrams(d)
    h0, h1 = _finite(dg[0]), _finite(dg[1])
    l1 = h1[:, 1] - h1[:, 0]
    if len(l1):
        pr = l1 / l1.sum()
        ent = float(-(pr * np.log(pr)).sum())
    else:
        ent = 0.0
    return {
        f"{prefix}_h0_total": float((h0[:, 1] - h0[:, 0]).sum()),
        f"{prefix}_h1_total": float(l1.sum()),
        f"{prefix}_h1_max": float(l1.max()) if len(l1) else 0.0,
        f"{prefix}_h1_count": float(len(l1)),
        f"{prefix}_h1_entropy": ent,
    }


def h1_total(d: np.ndarray) -> float:
    h1 = _finite(rips_diagrams(d)[1])
    return float((h1[:, 1] - h1[:, 0]).sum())


# --------------------------------------------------------------------------- #
# Window-level descriptors
# --------------------------------------------------------------------------- #
def nwcf_energy(d: np.ndarray, p: float = 1.0, orders=(3, 4)) -> float:
    n = d.shape[0]
    iu, _ = _edge_index(n)
    return float(np.mean(d[iu] - nwcf_operator(d, p, orders)[iu]))


def nwcf_descriptors(d: np.ndarray, p: float = 1.0, orders=(3, 4),
                     prefix: str = "nwcf", with_ph: bool = True,
                     with_mediators: bool = True) -> Dict[str, float]:
    n = d.shape[0]
    d_hat, det = nwcf_operator(d, p, orders, return_details=True)
    raw, hat = det["raw"], det["hat"]
    shrink = raw - hat
    adv = shrink > 1e-12
    out = {
        f"{prefix}_energy": float(shrink.mean()),
        f"{prefix}_frac": float(adv.mean()),
        f"{prefix}_relshrink": float(np.mean(shrink[adv] / np.clip(raw[adv], EPS, None))) if adv.any() else 0.0,
    }
    if 3 in orders and 4 in orders:
        out[f"{prefix}_frac4win"] = float(np.mean(det["supports"][4] < det["supports"][3] - 1e-12))
    if with_ph:
        out.update(ph_summary(d_hat, prefix))
    if with_mediators:
        m = mediator_map(n, det)
        nz = m[m > 0]
        out[f"{prefix}_mediator_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(n)) if len(nz) else 0.0
        for k in range(n):
            out[f"{prefix}_mediator_{k:02d}"] = float(m[k])
    return out
