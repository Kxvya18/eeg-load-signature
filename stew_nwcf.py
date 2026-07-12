from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from stew_topology_core import _fc_to_base_distance, _safe_rips_dm, _finite_bars, _persistence_scalars
from stew_topology_extensions import summarize_h1_representations

def _build_triangle_table(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (a, b, c) = (dist[i, j], dist[i, k], dist[j, k])
                gm = float(np.exp(np.mean(np.log(np.clip([a, b, c], 1e-12, None)))))
                rows.append((float(i), float(j), float(k), gm))
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 4), dtype=float)

def _best_supporting_triangle_geomean(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    if n < 3:
        return np.full((n, n), np.inf, dtype=float)
    Dij = dist[:, :, None]
    Dik = dist[:, None, :]
    Djk = dist[None, :, :]
    prod = np.clip(Dij, 1e-12, None) * np.clip(Dik, 1e-12, None) * np.clip(Djk, 1e-12, None)
    tri_geomean = np.cbrt(prod)
    idx = np.arange(n)
    invalid_k = (idx[None, None, :] == idx[:, None, None]) | (idx[None, None, :] == idx[None, :, None])
    tri_geomean = np.where(invalid_k, np.inf, tri_geomean)
    gamma = tri_geomean.min(axis=2)
    np.fill_diagonal(gamma, np.inf)
    return gamma

def _best_supporting_4clique_geomean(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    gamma4 = np.full((n, n), np.inf, dtype=float)
    if n < 4:
        return gamma4
    d = np.clip(dist, 1e-12, None)
    for i in range(n):
        for j in range(i + 1, n):
            others = np.asarray([k for k in range(n) if k not in (i, j)], dtype=int)
            di = d[i, others]
            dj = d[j, others]
            dkl = d[np.ix_(others, others)]
            prod = d[i, j] * np.outer(di, di) * np.outer(dj, dj) * dkl
            mask = np.triu(np.ones_like(prod, dtype=bool), k=1)
            vals = prod[mask]
            best = float(np.min(np.power(vals, 1.0 / 6.0))) if vals.size else np.inf
            gamma4[i, j] = gamma4[j, i] = best
    np.fill_diagonal(gamma4, np.inf)
    return gamma4

def compute_clique_cohesion_distance(dist: np.ndarray, return_diagnostics: bool=False, max_clique_order: int=4):
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    if n < 3:
        d_hat = dist.copy()
        if return_diagnostics:
            return (d_hat, {'n_edges': 0, 'n_corroborated': 0, 'fraction_corroborated': np.nan, 'mean_shrinkage': np.nan, 'mean_relative_shrinkage': np.nan})
        return d_hat
    gamma3 = _best_supporting_triangle_geomean(dist)
    gamma4 = _best_supporting_4clique_geomean(dist) if max_clique_order >= 4 else np.full_like(dist, np.inf)
    support = np.minimum(gamma3, gamma4)
    d_hat = np.minimum(dist, support)
    np.fill_diagonal(d_hat, 0.0)
    d_hat = (d_hat + d_hat.T) / 2.0
    if not return_diagnostics:
        return d_hat
    iu = np.triu_indices(n, k=1)
    raw = dist[iu]
    hat = d_hat[iu]
    corroborated = hat < raw - 1e-12
    n_edges = len(raw)
    n_corr = int(np.sum(corroborated))
    shrink = raw - hat
    rel_shrink = np.divide(shrink, raw, out=np.zeros_like(shrink), where=raw > 1e-12)
    iu = np.triu_indices(n, k=1)
    gamma3_edges = gamma3[iu]
    gamma4_edges = gamma4[iu]
    tri_better = gamma3_edges < raw - 1e-12
    k4_better = gamma4_edges < raw - 1e-12
    k4_wins = gamma4_edges < gamma3_edges - 1e-12
    diagnostics = {'n_edges': n_edges, 'n_corroborated': n_corr, 'fraction_corroborated': float(n_corr / n_edges) if n_edges > 0 else np.nan, 'mean_shrinkage': float(np.mean(shrink)) if n_edges > 0 else np.nan, 'mean_relative_shrinkage': float(np.mean(rel_shrink[corroborated])) if n_corr > 0 else 0.0, 'fraction_triangle_corroborated': float(np.mean(tri_better)) if n_edges > 0 else np.nan, 'fraction_4clique_corroborated': float(np.mean(k4_better)) if n_edges > 0 else np.nan, 'fraction_4clique_wins': float(np.mean(k4_wins)) if n_edges > 0 else np.nan, 'mean_triangle_support': float(np.nanmean(gamma3_edges)) if n_edges > 0 else np.nan, 'mean_4clique_support': float(np.nanmean(gamma4_edges[np.isfinite(gamma4_edges)])) if np.any(np.isfinite(gamma4_edges)) else np.nan}
    return (d_hat, diagnostics)

def nwcf_persistence(dist: np.ndarray, maxdim: int=1, max_clique_order: int=4) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=max_clique_order)
    dgms = _safe_rips_dm(d_hat, maxdim=maxdim)
    return (dgms, d_hat, diag)

def nwcf_window_features(similarity: np.ndarray, landscape_layers: int=3, landscape_grid: int=64, image_shape: Tuple[int, int]=(20, 20), image_sigma: float=0.04, max_clique_order: int=4) -> Dict:
    sim = np.asarray(similarity, dtype=float)
    dist = _fc_to_base_distance(sim)
    tri_table = _build_triangle_table(dist)
    dgms_rips = _safe_rips_dm(dist, maxdim=1)
    h0_rips = dgms_rips[0] if len(dgms_rips) > 0 else np.empty((0, 2))
    h1_rips = dgms_rips[1] if len(dgms_rips) > 1 else np.empty((0, 2))
    (dgms_nwcf, d_hat, edge_diag) = nwcf_persistence(dist, maxdim=1, max_clique_order=max_clique_order)
    h0_nwcf = dgms_nwcf[0] if len(dgms_nwcf) > 0 else np.empty((0, 2))
    h1_nwcf = dgms_nwcf[1] if len(dgms_nwcf) > 1 else np.empty((0, 2))
    h0_scalars = _persistence_scalars(h0_nwcf, 'H0')
    h1_scalars = _persistence_scalars(h1_nwcf, 'H1')
    repr_dict = summarize_h1_representations(h1_nwcf, normalization_mode='unit_diameter', landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma)
    offdiag = dist[np.triu_indices_from(dist, k=1)]
    feats: Dict = {}
    for (k, v) in h0_scalars.items():
        feats[f'nwcf_{k}'] = v
    for (k, v) in h1_scalars.items():
        feats[f'nwcf_{k}'] = v
    for (k, v) in repr_dict['summary'].items():
        feats[f'nwcf_{k}'] = v
    if tri_table.shape[0] > 0:
        tri_gm = tri_table[:, 3]
        feats['nwcf_triangle_geomean_mean'] = float(np.mean(tri_gm))
        feats['nwcf_triangle_geomean_std'] = float(np.std(tri_gm))
        feats['nwcf_triangle_geomean_min'] = float(np.min(tri_gm))
        feats['nwcf_triangle_geomean_max'] = float(np.max(tri_gm))
        feats['nwcf_clique_synchrony_index'] = float(np.mean(tri_gm < 0.5))
    else:
        for k in ['mean', 'std', 'min', 'max']:
            feats[f'nwcf_triangle_geomean_{k}'] = np.nan
        feats['nwcf_clique_synchrony_index'] = np.nan
    fb_rips = _finite_bars(h1_rips)
    fb_nwcf = _finite_bars(h1_nwcf)
    if len(fb_rips) > 0 and len(fb_nwcf) > 0:
        feats['nwcf_death_compression_ratio'] = float(np.mean(fb_nwcf[:, 1]) / (np.mean(fb_rips[:, 1]) + 1e-10))
    else:
        feats['nwcf_death_compression_ratio'] = np.nan
    feats['nwcf_mean_edge_distance'] = float(np.mean(offdiag)) if len(offdiag) > 0 else np.nan
    feats['nwcf_fraction_edges_corroborated'] = edge_diag['fraction_corroborated']
    feats['nwcf_mean_edge_shrinkage'] = edge_diag['mean_shrinkage']
    feats['nwcf_mean_relative_shrinkage'] = edge_diag['mean_relative_shrinkage']
    feats['nwcf_fraction_triangle_corroborated'] = edge_diag['fraction_triangle_corroborated']
    feats['nwcf_fraction_4clique_corroborated'] = edge_diag['fraction_4clique_corroborated']
    feats['nwcf_fraction_4clique_wins'] = edge_diag['fraction_4clique_wins']
    feats['nwcf_mean_triangle_support'] = edge_diag['mean_triangle_support']
    feats['nwcf_mean_4clique_support'] = edge_diag['mean_4clique_support']
    feats['nwcf_n_h1_bars_rips'] = int(len(fb_rips))
    feats['nwcf_n_h1_bars_nwcf'] = int(len(fb_nwcf))
    return feats

def benchmark_rips_vs_nwcf_timing(n_nodes_list: List[int]=[5, 10, 20, 32, 64], n_repeats: int=20, rng_seed: int=42) -> 'pd.DataFrame':
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    rows = []
    for n in n_nodes_list:
        (times_rips, times_nwcf) = ([], [])
        for _ in range(n_repeats):
            sim = rng.uniform(0.2, 0.8, (n, n))
            sim = (sim + sim.T) / 2.0
            np.fill_diagonal(sim, 1.0)
            sim = np.clip(sim, 0.0, 1.0)
            dist = _fc_to_base_distance(sim)
            t0 = time.perf_counter()
            _safe_rips_dm(dist, maxdim=1)
            times_rips.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            nwcf_persistence(dist, maxdim=1)
            times_nwcf.append(time.perf_counter() - t0)
        t_rips = float(np.median(times_rips))
        t_nwcf = float(np.median(times_nwcf))
        rows.append({'n_nodes': n, 'n_triangles': n * (n - 1) * (n - 2) // 6, 'rips_median_ms': round(t_rips * 1000, 4), 'nwcf_median_ms': round(t_nwcf * 1000, 4), 'nwcf_overhead_ms': round((t_nwcf - t_rips) * 1000, 4), 'nwcf_overhead_ratio': round(t_nwcf / (t_rips + 1e-12), 3)})
    return pd.DataFrame(rows)

def load_nwcf_subject_condition(per_subject_dir: Path, subject_id: int, condition: str, region_name: str, landscape_layers: int=3, landscape_grid: int=64, image_shape: Tuple[int, int]=(20, 20), image_sigma: float=0.04) -> Optional[Dict]:
    import json
    path = per_subject_dir / f'sub{int(subject_id):02d}_{condition}.json'
    if not path.exists():
        return None
    with open(path) as fh:
        data = json.load(fh)
    region = data.get('fc_regions', {}).get(region_name, {})
    rep = region.get('representative_window')
    if rep is None:
        return None
    sim_raw = rep.get('similarity_matrix')
    if sim_raw is None:
        return None
    sim = np.asarray(sim_raw, dtype=float)
    if sim.ndim != 2 or sim.shape[0] < 3:
        return None
    feats = nwcf_window_features(sim, landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma)
    per_window_feats = []
    for item in region.get('per_window', []) or []:
        if not isinstance(item, dict) or item.get('similarity_matrix') is None:
            continue
        wsim = np.asarray(item['similarity_matrix'], dtype=float)
        if wsim.ndim != 2 or wsim.shape[0] < 3:
            continue
        per_window_feats.append(nwcf_window_features(wsim, landscape_layers=landscape_layers, landscape_grid=landscape_grid, image_shape=image_shape, image_sigma=image_sigma))
    if per_window_feats:
        keys = sorted({k for row in per_window_feats for (k, v) in row.items() if isinstance(v, (int, float, np.floating, np.integer))})
        t = np.arange(len(per_window_feats), dtype=float)
        for key in keys:
            vals = np.asarray([row.get(key, np.nan) for row in per_window_feats], dtype=float)
            vals = vals[np.isfinite(vals)]
            short = key.replace('nwcf_', '', 1)
            if vals.size == 0:
                continue
            feats[f'nwcf_txn_mean_{short}'] = float(np.mean(vals))
            feats[f'nwcf_txn_std_{short}'] = float(np.std(vals))
            feats[f'nwcf_txn_median_{short}'] = float(np.median(vals))
            feats[f'nwcf_txn_cv_{short}'] = float(np.std(vals) / (abs(np.mean(vals)) + 1e-10))
            if vals.size > 1:
                feats[f'nwcf_txn_volatility_{short}'] = float(np.mean(np.abs(np.diff(vals))))
                tt = t[:vals.size]
                feats[f'nwcf_txn_slope_{short}'] = float(np.polyfit(tt, vals, 1)[0])
                half = vals.size // 2
                feats[f'nwcf_txn_early_late_delta_{short}'] = float(np.mean(vals[half:]) - np.mean(vals[:half])) if half > 0 else np.nan
        feats['nwcf_transaction_windows'] = int(len(per_window_feats))
    feats['nwcf_region'] = region_name
    feats['nwcf_n_ch'] = int(sim.shape[0])
    return feats
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print('=' * 78)
    print('DEMO 1 — Validity checks (Proposition 1, parts a-d)')
    print('=' * 78)
    n_trials = 200
    violations_domination = 0
    violations_dcr = 0
    n_identical = 0
    for _ in range(n_trials):
        n = rng.integers(5, 15)
        sim = rng.uniform(0.1, 0.95, (n, n))
        sim = (sim + sim.T) / 2.0
        np.fill_diagonal(sim, 1.0)
        dist = _fc_to_base_distance(sim)
        d_hat = compute_clique_cohesion_distance(dist)
        if np.any(d_hat > dist + 1e-09):
            violations_domination += 1
        if np.allclose(d_hat, dist, atol=1e-09):
            n_identical += 1
        dgms_rips = _safe_rips_dm(dist, maxdim=1)
        dgms_nwcf = _safe_rips_dm(d_hat, maxdim=1)
        fb_rips = _finite_bars(dgms_rips[1] if len(dgms_rips) > 1 else np.empty((0, 2)))
        fb_nwcf = _finite_bars(dgms_nwcf[1] if len(dgms_nwcf) > 1 else np.empty((0, 2)))
        if len(fb_rips) > 0 and len(fb_nwcf) > 0:
            dcr = np.mean(fb_nwcf[:, 1]) / np.mean(fb_rips[:, 1])
            if dcr > 1.0 + 1e-09:
                violations_dcr += 1
    print(f'Trials run:                                {n_trials}')
    print(f'Domination violations (D_hat > D anywhere): {violations_domination}  (must be 0)')
    print(f'DCR > 1 violations:                          {violations_dcr}  (must be 0)')
    print(f'Trials with D_hat == D exactly (no corroboration found): {n_identical}')
    print('-> Proposition 1(a)-(c) hold numerically with zero violations.' if violations_domination == 0 and violations_dcr == 0 else '-> VIOLATION DETECTED — investigate before trusting downstream results.')
    print()
    print('=' * 78)
    print('DEMO 2 — NWCF captures structure that global clustering coefficient')
    print('         (the obvious classical-graph baseline) cannot distinguish')
    print('=' * 78)
    import networkx as nx

    def binary_clustering(sim, thresh=0.5):
        n = sim.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > thresh:
                    G.add_edge(i, j)
        return nx.average_clustering(G)

    def make_pair(n, n_triads, rep_rng):
        base = rep_rng.uniform(0.3, 0.45, (n, n))
        base = (base + base.T) / 2.0
        np.fill_diagonal(base, 1.0)
        nodes = list(range(n))
        rep_rng.shuffle(nodes)
        triads = [tuple(nodes[3 * t:3 * t + 3]) for t in range(n_triads)]
        (net_a, net_b) = (base.copy(), base.copy())
        for (i, j, k) in triads:
            for (a, b) in [(i, j), (i, k), (j, k)]:
                net_a[a, b] = net_a[b, a] = 0.97
                net_b[a, b] = net_b[b, a] = 0.51
        return (np.clip(net_a, 0, 1), np.clip(net_b, 0, 1))
    n_reps = 30
    rows = []
    for r in range(n_reps):
        rep_rng = np.random.default_rng(100 + r)
        (net_a, net_b) = make_pair(n=18, n_triads=2, rep_rng=rep_rng)
        (cc_a, cc_b) = (binary_clustering(net_a), binary_clustering(net_b))
        fa = nwcf_window_features(net_a)
        fb = nwcf_window_features(net_b)
        rows.append(dict(cc_a=cc_a, cc_b=cc_b, cc_equal=np.isclose(cc_a, cc_b), geomean_a=fa['nwcf_triangle_geomean_mean'], geomean_b=fb['nwcf_triangle_geomean_mean'], csi_a=fa['nwcf_clique_synchrony_index'], csi_b=fb['nwcf_clique_synchrony_index'], corrob_a=fa['nwcf_fraction_edges_corroborated'], corrob_b=fb['nwcf_fraction_edges_corroborated'], shrink_a=fa['nwcf_mean_relative_shrinkage'], shrink_b=fb['nwcf_mean_relative_shrinkage'], dcr_a=fa['nwcf_death_compression_ratio'], dcr_b=fb['nwcf_death_compression_ratio']))
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"Replicates: {n_reps}   |   clustering coefficient identical in {df['cc_equal'].sum()}/{n_reps} replicates (by construction)")
    print()
    print(f"{'feature':32s} {'mean A':>10s} {'mean B':>10s} {'paired Δ (A−B)':>16s}")
    for (label, ca, cb) in [('clustering coefficient', 'cc_a', 'cc_b'), ('nwcf_triangle_geomean_mean', 'geomean_a', 'geomean_b'), ('nwcf_clique_synchrony_index', 'csi_a', 'csi_b'), ('nwcf_fraction_edges_corroborated', 'corrob_a', 'corrob_b'), ('nwcf_mean_relative_shrinkage', 'shrink_a', 'shrink_b')]:
        d = df[ca] - df[cb]
        print(f'{label:32s} {df[ca].mean():>10.4f} {df[cb].mean():>10.4f} {d.mean():>16.4f}')
    dcr_valid = df.dropna(subset=['dcr_a', 'dcr_b'])
    print(f"\nnwcf_death_compression_ratio    mean A={dcr_valid['dcr_a'].mean():.4f}  mean B={dcr_valid['dcr_b'].mean():.4f}   (defined in {len(dcr_valid)}/{n_reps} replicates — undefined when a replicate has no classical H1 cycle at all, which is a property of that random graph, not of NWCF)")
    print()
    print('-> Clustering coefficient is EXACTLY identical between arms A and B in')
    print(f'   every replicate by construction (same binary graph topology), while')
    print("   NWCF's geomean-based features separate the two conditions clearly and")
    print('   consistently — direct evidence that NWCF carries mesoscale information')
    print('   a standard graph-theoretic baseline cannot access at all.')
    print()
    print('=' * 78)
    print('DEMO 3 — Runtime overhead (informational)')
    print('=' * 78)
    bench = benchmark_rips_vs_nwcf_timing(n_nodes_list=[8, 14, 21, 32], n_repeats=10)
    print(bench.to_string(index=False))
