from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from stew_topology_core import _safe_rips_dm, _finite_bars, _persistence_scalars
from stew_topology_extensions import summarize_h1_representations

def _surrogate_h1_diagram(max_pers: float, n_bars: int, entropy: float) -> np.ndarray:
    if n_bars <= 0 or max_pers <= 0.0:
        return np.empty((0, 2), dtype=float)
    bars = [[0.0, float(max_pers)]]
    if n_bars > 1:
        n1 = n_bars - 1
        H_max = float(np.log(float(n_bars)))
        H_target = float(entropy)
        if H_target <= 0.0 or H_target >= H_max:
            best_p1 = 1.0 / n_bars
        else:
            best_p1 = 1.0 / n_bars
            best_err = float('inf')
            for p1_cand in np.linspace(1.0 / n_bars, 1.0 - 1e-06, 400):
                p2_cand = (1.0 - p1_cand) / n1
                if p2_cand <= 0:
                    continue
                H_cand = -(p1_cand * np.log(p1_cand + 1e-12) + n1 * p2_cand * np.log(p2_cand + 1e-12))
                err = abs(H_cand - H_target)
                if err < best_err:
                    best_err = err
                    best_p1 = p1_cand
        l2 = float(max_pers) * (1.0 - best_p1) / (n1 * best_p1 + 1e-10)
        l2 = max(l2, 0.0)
        if l2 > 0:
            bars.append([0.0, l2])
    return np.asarray(bars, dtype=float)

def _sliced_wasserstein_1_surrogate(diag_a: np.ndarray, diag_b: np.ndarray, n_dirs: int=10) -> float:
    fa = _finite_bars(diag_a)
    fb = _finite_bars(diag_b)
    (na, nb) = (len(fa), len(fb))
    if na == 0 and nb == 0:
        return 0.0
    thetas = np.linspace(0.0, np.pi, n_dirs, endpoint=False)
    slices = []
    for theta in thetas:
        (ct, st) = (np.cos(theta), np.sin(theta))
        pa = fa[:, 0] * ct + fa[:, 1] * st if na > 0 else np.array([], dtype=float)
        pb = fb[:, 0] * ct + fb[:, 1] * st if nb > 0 else np.array([], dtype=float)
        da = (fa[:, 0] + fa[:, 1]) / 2.0 * (ct + st) if na > 0 else np.array([], dtype=float)
        db = (fb[:, 0] + fb[:, 1]) / 2.0 * (ct + st) if nb > 0 else np.array([], dtype=float)
        full_a = np.sort(np.concatenate([pa, db]) if nb > 0 else pa)
        full_b = np.sort(np.concatenate([pb, da]) if na > 0 else pb)
        m = min(len(full_a), len(full_b))
        if m == 0:
            slices.append(0.0)
            continue
        slices.append(float(np.mean(np.abs(full_a[:m] - full_b[:m]))))
    return float(np.mean(slices)) if slices else 0.0

def _build_diagram_sequence(per_window: list, representative_window: Optional[dict]=None, use_full_rep: bool=True) -> Tuple[List[np.ndarray], List[float], List[bool]]:
    rep_time: float = np.nan
    rep_exact: Optional[np.ndarray] = None
    if use_full_rep and representative_window is not None:
        raw = representative_window.get('diagrams', {}).get('H1', [])
        if raw:
            arr = np.asarray(raw, dtype=float)
            fb = _finite_bars(arr)
            if len(fb) > 0:
                rep_exact = fb
        rep_time = float(representative_window.get('window_time_sec', np.nan))
    diagrams: List[np.ndarray] = []
    times: List[float] = []
    is_exact: List[bool] = []
    for entry in per_window:
        if entry is None:
            continue
        try:
            mp = float(entry['fc_max_persistence_H1'])
            nb = int(float(entry['fc_n_bars_H1']))
            ent = float(entry['fc_persistence_entropy_H1'])
            t = float(entry.get('window_time_sec', np.nan))
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(mp) and np.isfinite(ent) and (nb >= 0)):
            continue
        exact_entry = None
        raw_h1 = entry.get('diagrams', {}).get('H1', []) if isinstance(entry, dict) else []
        if raw_h1:
            arr = np.asarray(raw_h1, dtype=float)
            fb = _finite_bars(arr)
            if len(fb) > 0:
                exact_entry = fb
        if exact_entry is not None:
            diagrams.append(exact_entry)
            is_exact.append(True)
        elif rep_exact is not None and np.isfinite(rep_time) and np.isfinite(t) and (abs(t - rep_time) < 0.5):
            diagrams.append(rep_exact)
            is_exact.append(True)
        else:
            diagrams.append(_surrogate_h1_diagram(mp, nb, ent))
            is_exact.append(False)
        times.append(t)
    return (diagrams, times, is_exact)

def _wasserstein_distance_matrix(diagrams: List[np.ndarray], n_dirs: int=10) -> np.ndarray:
    N = len(diagrams)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            w = _sliced_wasserstein_1_surrogate(diagrams[i], diagrams[j], n_dirs=n_dirs)
            D[i, j] = w
            D[j, i] = w
    return D

def pdrh_features(per_window: list, representative_window: Optional[dict]=None, landscape_layers: int=3, landscape_grid: int=64, image_shape: Tuple[int, int]=(20, 20), image_sigma: float=0.04, n_sw_dirs: int=10, min_windows: int=6, use_full_rep: bool=True) -> Optional[Dict]:
    (diagrams, times, is_exact) = _build_diagram_sequence(per_window, representative_window=representative_window, use_full_rep=use_full_rep)
    N = len(diagrams)
    if N < min_windows:
        return None
    W = _wasserstein_distance_matrix(diagrams, n_dirs=n_sw_dirs)
    step_dists = np.array([W[t, t + 1] for t in range(N - 1)])
    eps_rec = float(np.median(step_dists)) if len(step_dists) > 0 else 0.0
    recurrence_mat = (W <= eps_rec).astype(float)
    np.fill_diagonal(recurrence_mat, 0.0)
    n_recurrent_pairs = int(recurrence_mat.sum()) // 2
    recurrence_rate = float(n_recurrent_pairs) / max(N * (N - 1) / 2, 1)
    dgms = _safe_rips_dm(W, maxdim=1)
    h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h0_scalars = _persistence_scalars(h0, 'H0')
    h1_scalars = _persistence_scalars(h1, 'H1')
    repr_dict = summarize_h1_representations(h1, normalization_mode='unit_diameter', landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma)
    feats: Dict = {'pdrh_n_windows': N, 'pdrh_n_exact_diagrams': int(sum(is_exact)), 'pdrh_eps_rec': eps_rec, 'pdrh_recurrence_rate': recurrence_rate, 'pdrh_n_recurrent_pairs': n_recurrent_pairs, 'pdrh_mean_step_dist': float(np.mean(step_dists)), 'pdrh_std_step_dist': float(np.std(step_dists)), 'pdrh_max_step_dist': float(np.max(step_dists)) if len(step_dists) > 0 else 0.0, 'pdrh_volatility_step_dist': float(np.mean(np.abs(np.diff(step_dists)))) if len(step_dists) > 1 else 0.0, 'pdrh_total_diagram_path': float(np.sum(step_dists))}
    for (k, v) in h0_scalars.items():
        feats[f'pdrh_{k}'] = v
    for (k, v) in h1_scalars.items():
        feats[f'pdrh_{k}'] = v
    for (k, v) in repr_dict['summary'].items():
        feats[f'pdrh_{k}'] = v
    return feats

def load_pdrh_subject_condition(per_subject_dir: Path, subject_id: int, condition: str, region_name: str, landscape_layers: int=3, landscape_grid: int=64, image_shape: Tuple[int, int]=(20, 20), image_sigma: float=0.04, n_sw_dirs: int=10, min_windows: int=6, use_full_rep: bool=True) -> Optional[Dict]:
    path = per_subject_dir / f'sub{int(subject_id):02d}_{condition}.json'
    if not path.exists():
        return None
    with open(path) as fh:
        data = json.load(fh)
    region_data = data.get('fc_regions', {}).get(region_name)
    if region_data is None:
        return None
    return pdrh_features(region_data.get('per_window', []), representative_window=region_data.get('representative_window'), landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma, n_sw_dirs=n_sw_dirs, min_windows=min_windows, use_full_rep=use_full_rep)
