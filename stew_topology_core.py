"""
stew_topology_core.py
======================
Shared, method-agnostic low-level helpers used by both NWCF (stew_nwcf.py)
and PDRH (stew_pdrh.py). Extracted unchanged from the original
stew_advanced_topology.py so neither method-specific module needs to
import the other, and so stew_advanced_topology.py can be kept as a thin
backward-compatible re-export shim (see that file's docstring).

Nothing in this file was modified in content; it is a verbatim relocation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from ripser import ripser as compute_rips


def _safe_rips_dm(dist_matrix: np.ndarray, maxdim: int = 1) -> List[np.ndarray]:
    """Run Rips on a pre-supplied distance matrix; return list of diagram arrays."""
    dm = np.asarray(dist_matrix, dtype=float)
    dm = (dm + dm.T) / 2.0
    np.fill_diagonal(dm, 0.0)
    dm = np.clip(dm, 0.0, None)
    if dm.shape[0] < 2:
        return [np.empty((0, 2), dtype=float)] * (maxdim + 1)
    try:
        return compute_rips(dm, maxdim=maxdim, distance_matrix=True)["dgms"]
    except Exception:
        return [np.empty((0, 2), dtype=float)] * (maxdim + 1)


def _finite_bars(diag: np.ndarray) -> np.ndarray:
    """Return finite bars with strictly positive persistence."""
    d = np.asarray(diag, dtype=float)
    if d.size == 0 or d.ndim != 2:
        return np.empty((0, 2), dtype=float)
    finite = d[np.isfinite(d[:, 1])]
    return finite[finite[:, 1] > finite[:, 0]]


def _persistence_scalars(diag: np.ndarray, suffix: str) -> Dict:
    """Standard scalar summaries for one homology diagram."""
    fb = _finite_bars(diag)
    if len(fb) == 0:
        return {
            f"max_persistence_{suffix}":   0.0,
            f"total_persistence_{suffix}": 0.0,
            f"n_bars_{suffix}":            0,
            f"mean_lifetime_{suffix}":     0.0,
            f"entropy_{suffix}":           0.0,
        }
    lts    = fb[:, 1] - fb[:, 0]
    normed = lts / (lts.sum() + 1e-10)
    return {
        f"max_persistence_{suffix}":   float(lts.max()),
        f"total_persistence_{suffix}": float(lts.sum()),
        f"n_bars_{suffix}":            int(len(fb)),
        f"mean_lifetime_{suffix}":     float(lts.mean()),
        f"entropy_{suffix}":           float(-np.sum(normed * np.log(normed + 1e-10))),
    }


def _fc_to_base_distance(similarity: np.ndarray) -> np.ndarray:
    """d(i,j) = 1 - clip(sim, 0, 1), symmetrised, diagonal = 0."""
    sim  = np.asarray(similarity, dtype=float)
    dist = 1.0 - np.clip(sim, 0.0, 1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist