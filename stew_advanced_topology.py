"""
stew_advanced_topology.py  —  compatibility shim (post-split)
================================================================
NWCF and PDRH have been separated into two independent, self-contained
modules — stew_nwcf.py and stew_pdrh.py — as two different papers, and
NWCF was additionally rebuilt as a genuine filtration (see stew_nwcf.py's
module docstring for the full rationale; the short version: v1's
"post-hoc death-time correction" was an approximation of a filtration,
not an actual one, and could not be fixed without changing which object
the geometric mean is applied to. v2 reweights edges via clique-cohesion
and runs ordinary Vietoris-Rips on the reweighted matrix, which is a
provably valid filtration and is provably more clique-sensitive than
plain Rips — see stew_nwcf.py, Proposition 1).

THIS FILE NOW EXISTS PURELY FOR BACKWARD COMPATIBILITY.
----------------------------------------------------------
Every name that used to be importable from `stew_advanced_topology` is
still importable from it — this file just re-exports them from their new
homes. Concretely:

    stew_topology_core.py   -> shared low-level helpers (unchanged)
    stew_nwcf.py            -> everything NWCF-specific (rebuilt, v2)
    stew_pdrh.py            -> everything PDRH-specific (unchanged, v1)
    stew_advanced_topology.py (this file) -> re-exports both, plus the
                                              cohort-level batch loader
                                              that combines them.

stew_advanced_analysis.py and stew_methodological_justification.py do
NOT need to be modified: their existing

    from stew_advanced_topology import (...)

import lines continue to work exactly as before. stew_pipeline.py and
stew_analysis.py are untouched and were never part of this split.

If you are starting a fresh script, prefer importing directly from
stew_nwcf / stew_pdrh / stew_topology_core instead of through this shim
— it is kept only so existing files do not need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

# ── Shared low-level helpers (unchanged; re-exported for compatibility) ──
from stew_topology_core import (   # noqa: F401
    _fc_to_base_distance,
    _safe_rips_dm,
    _finite_bars,
    _persistence_scalars,
)

# ── NWCF (v2 — genuine filtration; see stew_nwcf.py) ─────────────────────
from stew_nwcf import (   # noqa: F401
    _build_triangle_table,
    _best_supporting_triangle_geomean,
    compute_clique_cohesion_distance,
    nwcf_persistence,
    nwcf_window_features,
    benchmark_rips_vs_nwcf_timing,
    load_nwcf_subject_condition,
)

# ── PDRH (unchanged; see stew_pdrh.py) ───────────────────────────────────
from stew_pdrh import (   # noqa: F401
    _surrogate_h1_diagram,
    _sliced_wasserstein_1_surrogate,
    _build_diagram_sequence,
    _wasserstein_distance_matrix,
    pdrh_features,
    load_pdrh_subject_condition,
)


# ═════════════════════════════════════════════════════════════════════════════
# Cohort-level batch loader (combines both methods; unchanged from v1
# except that it now calls the v2 NWCF loader transparently)
# ═════════════════════════════════════════════════════════════════════════════

def build_advanced_topology_features(
    per_subject_dir: Path,
    subject_ids: List[int],
    region_names: List[str],
    landscape_layers: int = 3,
    landscape_grid:   int = 64,
    image_shape: "tuple" = (20, 20),
    image_sigma: float = 0.04,
    pdrh_min_windows: int = 6,
    pdrh_n_sw_dirs: int = 10,
    use_full_rep: bool = True,
    compute_nwcf: bool = True,
    compute_pdrh: bool = True,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Compute NWCF and/or PDRH features for all subject-IDs x conditions x
    regions. Returns a DataFrame mergeable with subject_level_features.csv
    on (subject_id, condition).

    compute_nwcf / compute_pdrh
    ----------------------------
    Both default to True, which reproduces the EXACT original v1 behavior
    (compute both, always) — so any existing caller that doesn't pass
    these new arguments is completely unaffected.

    Set compute_pdrh=False while only working on the NWCF paper: PDRH's
    pairwise sliced-Wasserstein step is O(n_windows^2) per subject x
    condition x region and is by far the dominant cost of this function
    when sessions have more than ~100 windows (see profiling discussion).
    Skipping it here means that cost is never paid at all, rather than
    paid and then ignored.
    """
    import pandas as pd

    rows = []
    for sid in subject_ids:
        for condition in ("lo", "hi"):
            row: Dict = {"subject_id": int(sid), "condition": condition}
            nwcf_res = pdrh_res = None

            for region in region_names:
                prefix = f"ADVTOPO_{region}"

                if compute_nwcf:
                    nwcf_res = load_nwcf_subject_condition(
                        per_subject_dir, sid, condition, region,
                        landscape_layers=landscape_layers,
                        landscape_grid=landscape_grid,
                        image_shape=image_shape,
                        image_sigma=image_sigma,
                    )
                    if nwcf_res is not None:
                        for k, v in nwcf_res.items():
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row[f"{prefix}_NWCF_{k}"] = float(v)
                    else:
                        row[f"{prefix}_NWCF_available"] = 0.0

                if compute_pdrh:
                    pdrh_res = load_pdrh_subject_condition(
                        per_subject_dir, sid, condition, region,
                        landscape_layers=landscape_layers,
                        landscape_grid=landscape_grid,
                        image_shape=image_shape,
                        image_sigma=image_sigma,
                        n_sw_dirs=pdrh_n_sw_dirs,
                        min_windows=pdrh_min_windows,
                        use_full_rep=use_full_rep,
                    )
                    if pdrh_res is not None:
                        for k, v in pdrh_res.items():
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row[f"{prefix}_PDRH_{k}"] = float(v)
                    else:
                        row[f"{prefix}_PDRH_available"] = 0.0

            rows.append(row)
            if verbose:
                status = []
                if compute_nwcf: status.append(f"NWCF={'ok' if nwcf_res else 'none'}")
                if compute_pdrh: status.append(f"PDRH={'ok' if pdrh_res else 'none'}")
                print(f"  sub{sid:02d} {condition}: " + ", ".join(status))

    return pd.DataFrame(rows)