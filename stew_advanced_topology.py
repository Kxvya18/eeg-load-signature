from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
from stew_topology_core import _fc_to_base_distance, _safe_rips_dm, _finite_bars, _persistence_scalars
from stew_nwcf import _build_triangle_table, _best_supporting_triangle_geomean, compute_clique_cohesion_distance, nwcf_persistence, nwcf_window_features, benchmark_rips_vs_nwcf_timing, load_nwcf_subject_condition
from stew_pdrh import _surrogate_h1_diagram, _sliced_wasserstein_1_surrogate, _build_diagram_sequence, _wasserstein_distance_matrix, pdrh_features, load_pdrh_subject_condition

def build_advanced_topology_features(per_subject_dir: Path, subject_ids: List[int], region_names: List[str], landscape_layers: int=3, landscape_grid: int=64, image_shape: 'tuple'=(20, 20), image_sigma: float=0.04, pdrh_min_windows: int=6, pdrh_n_sw_dirs: int=10, use_full_rep: bool=True, compute_nwcf: bool=True, compute_pdrh: bool=True, verbose: bool=True) -> 'pd.DataFrame':
    import pandas as pd
    rows = []
    for sid in subject_ids:
        for condition in ('lo', 'hi'):
            row: Dict = {'subject_id': int(sid), 'condition': condition}
            nwcf_res = pdrh_res = None
            for region in region_names:
                prefix = f'ADVTOPO_{region}'
                if compute_nwcf:
                    nwcf_res = load_nwcf_subject_condition(per_subject_dir, sid, condition, region, landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma)
                    if nwcf_res is not None:
                        for (k, v) in nwcf_res.items():
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row[f'{prefix}_NWCF_{k}'] = float(v)
                    else:
                        row[f'{prefix}_NWCF_available'] = 0.0
                if compute_pdrh:
                    pdrh_res = load_pdrh_subject_condition(per_subject_dir, sid, condition, region, landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma, n_sw_dirs=pdrh_n_sw_dirs, min_windows=pdrh_min_windows, use_full_rep=use_full_rep)
                    if pdrh_res is not None:
                        for (k, v) in pdrh_res.items():
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row[f'{prefix}_PDRH_{k}'] = float(v)
                    else:
                        row[f'{prefix}_PDRH_available'] = 0.0
            rows.append(row)
            if verbose:
                status = []
                if compute_nwcf:
                    status.append(f"NWCF={('ok' if nwcf_res else 'none')}")
                if compute_pdrh:
                    status.append(f"PDRH={('ok' if pdrh_res else 'none')}")
                print(f'  sub{sid:02d} {condition}: ' + ', '.join(status))
    return pd.DataFrame(rows)
