from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Polygon, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
from stew_nwcf import compute_clique_cohesion_distance, nwcf_window_features
from stew_topology_core import _fc_to_base_distance
PROJECT_ROOT = Path(__file__).resolve().parent
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SCALP_POS = {'AF3': (-0.38, 0.92), 'F7': (-0.82, 0.58), 'F3': (-0.42, 0.55), 'FC5': (-0.62, 0.2), 'T7': (-0.95, 0.0), 'P7': (-0.72, -0.48), 'O1': (-0.3, -0.86), 'O2': (0.3, -0.86), 'P8': (0.72, -0.48), 'T8': (0.95, 0.0), 'FC6': (0.62, 0.2), 'F4': (0.42, 0.55), 'F8': (0.82, 0.58), 'AF4': (0.38, 0.92)}

def set_style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.titlesize': 8.8, 'axes.labelsize': 8.2, 'axes.linewidth': 0.9, 'figure.dpi': 160, 'savefig.dpi': 450, 'savefig.bbox': 'tight', 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})

def panel(ax, label: str) -> None:
    ax.text(-0.075, 1.04, label, transform=ax.transAxes, fontweight='bold', fontsize=11, va='bottom')

def clean_axis_arrows(ax, xlabel: str, ylabel: str) -> None:
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
    ax.annotate('', xy=(0, 1.03), xytext=(0, 0), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
    ax.text(0.5, -0.08, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.text(-0.08, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center', rotation=90, fontsize=8)

def header(fig, title: str, subtitle: str | None=None) -> None:
    fig.text(0.055, 0.965, title, ha='left', va='top', fontsize=10.8, fontweight='bold')
    if subtitle:
        fig.text(0.055, 0.925, subtitle, ha='left', va='top', fontsize=7.8, color='#486471')

def phase_bar(ax, x0: float, x1: float, y: float, text: str, color: str) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=1.3, clip_on=False)
    ax.plot([x0, x0], [y - 0.018, y + 0.018], color=color, lw=1.0, clip_on=False)
    ax.plot([x1, x1], [y - 0.018, y + 0.018], color=color, lw=1.0, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.035, text, color=color, ha='center', va='bottom', fontsize=7.2)

def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{name}.png')
    plt.close(fig)

def data_dir(config: str) -> Path:
    p = PROJECT_ROOT / f'stew_tda_results_final_{config}'
    if not p.exists():
        raise FileNotFoundError(f'Missing original topology output folder: {p}')
    return p

def nwcf_dir(config: str) -> Path:
    p = PROJECT_ROOT / f'stew_output_{config}'
    if not p.exists():
        raise FileNotFoundError(f'Missing NWCF output folder: {p}')
    return p

def subject_features(config: str) -> pd.DataFrame:
    p = data_dir(config) / 'subject_level_features.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def paired(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = df.pivot_table(index='subject_id', columns='condition', values=col, aggfunc='mean')
    if 'lo' not in p.columns or 'hi' not in p.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = p[['lo', 'hi']].dropna()
    p['delta'] = p['hi'] - p['lo']
    return p.reset_index()

def dz(delta: pd.Series) -> float:
    x = pd.to_numeric(delta, errors='coerce').dropna()
    if len(x) < 3 or x.std(ddof=1) <= 1e-12:
        return np.nan
    return float(x.mean() / x.std(ddof=1))

def draw_scalp(ax, values: dict[str, float], title: str, cmap='coolwarm', vlim=1.0):
    xy = np.array([SCALP_POS[ch] for ch in CHANNELS])
    z = np.array([values.get(ch, np.nan) for ch in CHANNELS])
    ax.add_patch(Circle((0, 0), 1.04, fill=False, color='black', lw=1.1))
    ax.plot([0, -0.08, 0.08, 0], [1.04, 1.16, 1.04, 1.04], color='black', lw=1.0)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, s=150, cmap=cmap, vmin=-vlim, vmax=vlim, edgecolor='black', lw=0.55)
    for (ch, (x, y)) in SCALP_POS.items():
        ax.text(x, y - 0.14, ch, ha='center', va='center', fontsize=6.2)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.08, 1.22)
    ax.axis('off')
    return sc

def raw_subject_pair(config: str, subject: int | None=None) -> tuple[Path, Path]:
    raw = data_dir(config) / 'per_subject_raw'
    lo_files = sorted(raw.glob('*_lo.json'))
    if subject is not None:
        lo_files = [raw / f'sub{subject:02d}_lo.json']
    for lo in lo_files:
        hi = raw / lo.name.replace('_lo.json', '_hi.json')
        if lo.exists() and hi.exists():
            return (lo, hi)
    raise FileNotFoundError('No paired raw JSON files found')

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def representative_distance(config: str, condition: str, region: str) -> tuple[np.ndarray, list[str], str]:
    (lo, hi) = raw_subject_pair(config)
    path = hi if condition == 'hi' else lo
    data = load_json(path)
    reg = data['fc_regions'][region]
    sim = np.asarray(reg['representative_window']['similarity_matrix'], dtype=float)
    return (_fc_to_base_distance(sim), reg['channels'], path.stem)

def mean_similarity_by_condition(config: str, region: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = data_dir(config) / 'per_subject_raw'
    (lo_mats, hi_mats, labels) = ([], [], None)
    for lo in sorted(raw.glob('*_lo.json')):
        hi = raw / lo.name.replace('_lo.json', '_hi.json')
        if not hi.exists():
            continue
        for (path, bucket) in [(lo, lo_mats), (hi, hi_mats)]:
            data = load_json(path)
            reg = data['fc_regions'][region]
            labels = reg['channels']
            bucket.append(np.asarray(reg['representative_window']['similarity_matrix'], dtype=float))
    return (np.nanmean(lo_mats, axis=0), np.nanmean(hi_mats, axis=0), labels or [])

def graph_pos(labels: list[str]) -> dict[int, tuple[float, float]]:
    if all((ch in SCALP_POS for ch in labels)):
        return {i: SCALP_POS[ch] for (i, ch) in enumerate(labels)}
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    return {i: (math.cos(t), math.sin(t)) for (i, t) in enumerate(theta)}

def fig1_workload_structure(config: str, outdir: Path) -> None:
    df = subject_features(config)
    rows = []
    for ch in CHANNELS:
        for (suffix, label) in [('mean_max_persistence_H1', 'H1 max persistence'), ('mean_landscape_total_area_H1', 'landscape area')]:
            pv = paired(df, f'{ch}_{suffix}')
            if not pv.empty:
                rows.append({'channel': ch, 'metric': label, 'dz': dz(pv['delta']), 'median_delta': pv['delta'].median()})
    eff = pd.DataFrame(rows)
    h1 = eff[eff['metric'].eq('H1 max persistence')]
    land = eff[eff['metric'].eq('landscape area')]
    top_ch = h1.iloc[h1['dz'].abs().argmax()]['channel']
    pv = paired(df, f'{top_ch}_mean_max_persistence_H1')
    fig = plt.figure(figsize=(8.6, 5.7))
    header(fig, 'Original topology establishes workload structure', 'Channel-wise persistent-homology summaries reveal task-rest organization before the custom filtration is introduced.')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.95, top=0.84, bottom=0.1, wspace=0.36, hspace=0.34, height_ratios=[1.0, 0.82])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    sc = draw_scalp(ax1, dict(zip(h1['channel'], h1['dz'])), 'H1 topology effect', vlim=1.0)
    draw_scalp(ax2, dict(zip(land['channel'], land['dz'])), 'Landscape effect', vlim=1.0)
    cbar = fig.colorbar(sc, ax=[ax1, ax2], fraction=0.038, pad=0.025)
    cbar.set_label('paired dz')
    for (_, r) in pv.iterrows():
        ax3.plot([0, 1], [r['lo'], r['hi']], color='#d7d7d7', lw=0.55, zorder=1)
    med = pv[['lo', 'hi']].median()
    ax3.plot([0, 1], med, color='#d64b3a', marker='o', lw=2.0, label='median', zorder=3)
    ax3.set_xticks([0, 1], ['Rest', 'Task'])
    ax3.set_ylabel(f'{top_ch} H1 max persistence')
    ax3.set_title(f'Paired shift ({top_ch})')
    ax3.grid(True, ls='-', lw=0.45, color='#eeeeee')
    ax3.annotate('topological shift\nunder demand', xy=(1, med['hi']), xytext=(0.22, 0.86), textcoords='axes fraction', arrowprops=dict(arrowstyle='->', color='#d64b3a', lw=1.0), fontsize=7.2, color='#9f2f24')
    ax3.text(0.06, 0.08, f"N={len(pv)}\ndz={dz(pv['delta']):.2f}", transform=ax3.transAxes, va='bottom', fontsize=7, bbox=dict(fc='white', ec='#888888', lw=0.7))
    clean_axis_arrows(ax4, 'analysis stage', 'evidence for workload structure')
    xs = np.array([0.08, 0.32, 0.56, 0.8])
    ys = np.array([0.14, 0.42, 0.68, 0.88])
    ax4.plot(xs, ys, color='#2ab7c9', lw=2.0)
    ax4.scatter(xs, ys, s=42, color=['#9aa600', '#2aa198', '#2ab7c9', '#0072b2'], edgecolor='white', zorder=4)
    for (x, label, color) in [(0.08, 'raw EEG\nwindows', '#9aa600'), (0.32, 'delay-cloud\nPH', '#2aa198'), (0.56, 'channel-wise\ntopological shifts', '#2ab7c9'), (0.8, 'workload\nstructure', '#0072b2')]:
        ax4.text(x, 0.05 if x < 0.6 else 0.73, label, transform=ax4.transAxes, ha='center', fontsize=7.2, color=color)
    phase_bar(ax4, 0.06, 0.34, 0.98, 'signal formation', '#9aa600')
    phase_bar(ax4, 0.38, 0.62, 0.98, 'topological encoding', '#2aa198')
    phase_bar(ax4, 0.66, 0.9, 0.98, 'brain-state inference', '#0072b2')
    panel(ax1, 'a')
    panel(ax2, 'b')
    panel(ax3, 'c')
    panel(ax4, 'd')
    save(fig, outdir, 'Original_01_WorkloadStructure')

def family_of(col: str) -> str:
    low = col.lower()
    if col in {'subject_id', 'condition', 'condition_binary'}:
        return 'metadata'
    if 'surrogate' in low:
        return 'surrogate topology'
    if 'fcregion' in low or low.startswith('fc') or '_fc_' in low:
        return 'FC topology'
    if 'landscape' in low:
        return 'persistence landscapes'
    if 'persistence_image' in low:
        return 'persistence images'
    if 'max_persistence' in low or 'total_persistence' in low or 'entropy_h' in low or ('n_significant_cycles' in low):
        return 'channel PH'
    if 'alpha_power' in low or 'alpha_rms' in low or 'alpha_envelope' in low:
        return 'alpha amplitude'
    if 'pre_' in low or 'post_' in low or 'asr_' in low:
        return 'QC/preprocessing'
    return 'other'

def fig2_feature_universe(config: str, outdir: Path) -> None:
    df = subject_features(config)
    numeric = df.select_dtypes(include=[np.number]).copy()
    rows = []
    for col in numeric.columns:
        if col in {'subject_id', 'condition_binary'}:
            continue
        fam = family_of(col)
        pv = paired(df, col)
        effect = abs(dz(pv['delta'])) if not pv.empty else np.nan
        rows.append({'feature': col, 'family': fam, 'effect': effect})
    info = pd.DataFrame(rows).dropna()
    fam_stats = info.groupby('family').agg(n_features=('feature', 'size'), median_abs_dz=('effect', 'median'), max_abs_dz=('effect', 'max'))
    fam_stats = fam_stats.sort_values('n_features', ascending=False)
    order = fam_stats.index.tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(order)))
    cmap = dict(zip(order, colors))
    fig = plt.figure(figsize=(8.7, 5.4))
    header(fig, 'Original topology is informative but high-dimensional', 'The discovery model spans many topological representations; NWCF is introduced later as a compact custom filtration.')
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, top=0.82, bottom=0.1, height_ratios=[1.0, 0.72], width_ratios=[1.18, 1.0], hspace=0.38, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    y = np.arange(len(order))
    ax1.barh(y, fam_stats.loc[order, 'n_features'], color=[cmap[o] for o in order], edgecolor='white', lw=0.7)
    ax1.set_yticks(y, order)
    ax1.invert_yaxis()
    ax1.set_xlabel('Number of original features')
    ax1.set_title('Feature universe')
    ax1.grid(True, axis='x', ls='-', lw=0.45, color='#eeeeee')
    ax1.annotate('broad discovery space', xy=(fam_stats['n_features'].max() * 0.72, 1), xytext=(fam_stats['n_features'].max() * 0.4, len(order) - 1.0), arrowprops=dict(arrowstyle='->', lw=1.0, color='#444444'), fontsize=7.3, bbox=dict(fc='white', ec='#888888', lw=0.7, alpha=0.95))
    ax2.scatter(fam_stats['n_features'], fam_stats['max_abs_dz'], s=95, c=[cmap[o] for o in fam_stats.index], edgecolor='white', lw=0.7)
    for (fam, r) in fam_stats.iterrows():
        ax2.text(r['n_features'] * 1.05, r['max_abs_dz'], fam.split()[0], fontsize=6.6, va='center', color='#333333')
    ax2.set_xscale('log')
    ax2.set_xlabel('Family size (log scale)')
    ax2.set_ylabel('Max |paired dz|')
    ax2.set_title('Signal exists, but is distributed')
    ax2.grid(True, ls='-', lw=0.45, color='#eeeeee')
    clean_axis_arrows(ax3, 'modeling path', 'interpretability pressure')
    x = np.linspace(0.05, 0.92, 160)
    broad = 0.22 + 0.54 / (1 + np.exp(-10 * (x - 0.34)))
    compact = 0.18 + 0.62 / (1 + np.exp(-16 * (x - 0.68)))
    ax3.plot(x, broad, color='#2ab7c9', lw=2.0)
    ax3.plot(x, compact, color='#0072b2', lw=1.8, ls='--')
    ax3.text(0.12, 0.76, 'Original topology:\nhigh coverage,\nhigh dimensionality', color='#2a9daf', transform=ax3.transAxes, fontsize=7.3)
    ax3.text(0.6, 0.28, 'Need next step:\ncompact filtration\nwith mechanistic support', color='#0072b2', transform=ax3.transAxes, fontsize=7.3)
    ax3.annotate('', xy=(0.58, 0.54), xytext=(0.43, 0.54), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0, color='#444444'))
    ax3.text(0.5, 0.58, 'motivates', transform=ax3.transAxes, ha='center', fontsize=7.2, color='#444444')
    panel(ax1, 'a')
    panel(ax2, 'b')
    panel(ax3, 'c')
    save(fig, outdir, 'Original_02_FeatureUniverse')

def fig3_threshold_limitation(config: str, outdir: Path) -> None:
    (dist, labels, tag) = representative_distance(config, 'hi', 'whole_head')
    raw = dist[np.triu_indices_from(dist, k=1)]
    pct = np.arange(3, 76, 3)
    rows = []
    for p in pct:
        theta = np.percentile(raw, p)
        g = nx.Graph()
        g.add_nodes_from(range(len(labels)))
        edges = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if dist[i, j] <= theta]
        g.add_edges_from(edges)
        comps = nx.number_connected_components(g)
        cycle_rank = max(0, g.number_of_edges() - len(labels) + comps)
        rows.append({'pct': p, 'theta': theta, 'edges': len(edges), 'components': comps, 'cycle_rank': cycle_rank})
    cur = pd.DataFrame(rows)
    fig = plt.figure(figsize=(8.7, 6.2))
    header(fig, 'Classical topology is sensitive to pairwise threshold trajectory', 'A Rips filtration is governed by edge-entry scales; small threshold changes can reorganize components and cycles.')
    gs = fig.add_gridspec(3, 2, left=0.08, right=0.95, top=0.84, bottom=0.1, height_ratios=[0.82, 0.82, 0.72], hspace=0.45, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])
    ax0.plot(cur['pct'], cur['edges'], color='#2ab7c9', lw=2.0, label='edges')
    ax0.plot(cur['pct'], cur['cycle_rank'] * (cur['edges'].max() / max(1, cur['cycle_rank'].max())), color='#0072b2', lw=1.6, ls='--', label='scaled cycle rank')
    ax0.axvline(24, color='#9aa600', lw=1.1, ls=':')
    ax0.axvline(39, color='#2aa198', lw=1.1, ls=':')
    ax0.set_ylabel('Filtration growth')
    ax0.set_xticks([])
    ax0.set_title('Edge entry creates abrupt topological regime changes')
    ax0.grid(True, ls='-', lw=0.45, color='#eeeeee')
    phase_bar(ax0, 6, 23, ax0.get_ylim()[1] * 1.02, 'sparse graph', '#9aa600')
    phase_bar(ax0, 25, 40, ax0.get_ylim()[1] * 1.02, 'rapid merger', '#2aa198')
    phase_bar(ax0, 42, 70, ax0.get_ylim()[1] * 1.02, 'cycle-rich', '#0072b2')
    ax0.legend(fontsize=7, loc='lower right')
    panel(ax0, 'a')
    ax1.plot(cur['pct'], cur['components'], color='#d55e00', lw=2.0)
    ax1.scatter(cur['pct'][::3], cur['components'][::3], s=18, color='#d55e00', edgecolor='white', zorder=3)
    ax1.set_xlabel('Distance percentile threshold')
    ax1.set_ylabel('Connected components')
    ax1.set_title('H0 compression')
    ax1.grid(True, ls='-', lw=0.45, color='#eeeeee')
    ax1.text(0.08, 0.18, 'components collapse as\nedges enter the graph', transform=ax1.transAxes, color='#8f3b00', fontsize=7.2)
    panel(ax1, 'b')
    ax2.plot(cur['pct'], cur['cycle_rank'], color='#009e73', lw=2.0)
    ax2.scatter(cur['pct'][::3], cur['cycle_rank'][::3], s=18, color='#009e73', edgecolor='white', zorder=3)
    ax2.set_xlabel('Distance percentile threshold')
    ax2.set_ylabel('Cycle rank proxy')
    ax2.set_title('H1 opportunity')
    ax2.grid(True, ls='-', lw=0.45, color='#eeeeee')
    ax2.text(0.08, 0.78, 'cycles depend on precise\nedge combinations', transform=ax2.transAxes, color='#006b50', fontsize=7.2)
    panel(ax2, 'c')
    clean_axis_arrows(ax3, 'classical edge threshold', 'interpretation risk')
    x = np.linspace(0.08, 0.9, 120)
    y = 0.15 + 0.65 / (1 + np.exp(-18 * (x - 0.42)))
    ax3.plot(x, y, color='#2ab7c9', lw=2.2)
    ax3.plot(x, 0.2 + 0.45 / (1 + np.exp(-10 * (x - 0.58))), color='#0072b2', lw=1.5, ls='--')
    ax3.axvline(0.42, color='#2aa198', ls=':', lw=1.3)
    ax3.text(0.13, 0.66, 'pairwise distances\ncontrol entry', transform=ax3.transAxes, color='#2aa198', fontsize=7.3)
    ax3.text(0.6, 0.2, 'custom filtration should use\nmesoscale support, not only\nisolated edge strength', transform=ax3.transAxes, color='#0072b2', fontsize=7.3)
    ax3.annotate('motivation for NWCF', xy=(0.5, 0.6), xytext=(0.66, 0.82), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=7.4)
    panel(ax3, 'd')
    fig.text(0.56, 0.055, f'Exemplar: {config}, {tag}, whole-head representative FC window', ha='center', color='#666666', fontsize=6.8)
    save(fig, outdir, 'Original_03_ClassicalThresholdLimitation')

def fig4_frontal_alpha_compression(config: str, outdir: Path) -> None:
    df = subject_features(config)
    region = 'frontal'
    (rest_sim, task_sim, labels) = mean_similarity_by_condition(config, region)
    if len(labels) == 0:
        return
    (rest_dist, task_dist) = (_fc_to_base_distance(rest_sim), _fc_to_base_distance(task_sim))
    delta = task_sim - rest_sim
    tri = np.triu_indices_from(rest_sim, k=1)
    top_edges = sorted([(i, j, delta[i, j]) for (i, j) in zip(*tri)], key=lambda e: abs(e[2]), reverse=True)[:8]
    comp_cols = ['FCREGION_frontal_mean_fc_mean_similarity', 'FCREGION_frontal_mean_fc_total_persistence_H0', 'FCREGION_frontal_mean_fc_max_persistence_H1']
    existing = [c for c in comp_cols if c in df.columns]
    effects = []
    for c in existing:
        pv = paired(df, c)
        if not pv.empty:
            effects.append((c, dz(pv['delta']), pv))
    fig = plt.figure(figsize=(8.8, 6.0))
    header(fig, 'Frontal alpha FC undergoes topological compression under cognitive demand', 'Mean frontal networks reveal candidate edge-level drivers; paired topology summaries quantify compression.')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.95, top=0.84, bottom=0.1, hspace=0.4, wspace=0.3, height_ratios=[1.0, 0.78])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_story = fig.add_subplot(gs[1, 1:])
    pos = graph_pos(labels)
    for (ax, sim, ttl, color, pl) in [(axes[0], rest_sim, 'Rest frontal FC', '#0072b2', 'a'), (axes[1], task_sim, 'Task frontal FC', '#d62728', 'b')]:
        raw = sim[np.triu_indices_from(sim, k=1)]
        thr = np.percentile(raw, 72)
        edges = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if sim[i, j] >= thr]
        widths = [0.7 + 3.2 * max(0, sim[i, j] - thr) / max(1e-09, raw.max() - thr) for (i, j) in edges]
        g = nx.Graph()
        g.add_nodes_from(range(len(labels)))
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edges, width=widths, edge_color=color, alpha=0.62)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=225, node_color='white', edgecolors='black', linewidths=0.8)
        nx.draw_networkx_labels(g, pos, ax=ax, labels={i: labels[i] for i in range(len(labels))}, font_size=6.7)
        ax.set_title(ttl)
        ax.axis('off')
        panel(ax, pl)
    ax = axes[2]
    g = nx.Graph()
    g.add_nodes_from(range(len(labels)))
    inc = [(i, j) for (i, j, d) in top_edges if d > 0]
    dec = [(i, j) for (i, j, d) in top_edges if d <= 0]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=inc, edge_color='#d64b3a', width=2.0, alpha=0.86)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=dec, edge_color='#0072b2', width=2.0, alpha=0.82, style='dashed')
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=225, node_color='white', edgecolors='black', linewidths=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={i: labels[i] for i in range(len(labels))}, font_size=6.7)
    ax.set_title('Candidate edge-level drivers')
    ax.text(0.02, -0.02, 'red: stronger in task     blue dashed: weaker in task', transform=ax.transAxes, fontsize=6.7, va='top')
    ax.axis('off')
    panel(ax, 'c')
    ax = ax_bar
    if effects:
        names = [e[0].replace('FCREGION_frontal_mean_fc_', '').replace('_', ' ') for e in effects]
        vals = [e[1] for e in effects]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=['#d64b3a' if v > 0 else '#0072b2' for v in vals], edgecolor='white', lw=0.7)
        ax.axvline(0, color='black', lw=0.9)
        ax.set_yticks(y, names, fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlabel('paired dz (task-rest)')
        ax.set_title('Compression markers')
        ax.grid(True, axis='x', ls='-', lw=0.45, color='#eeeeee')
    panel(ax, 'd')
    clean_axis_arrows(ax_story, 'inference step', 'interpretability')
    x = np.array([0.08, 0.33, 0.58, 0.83])
    y = np.array([0.22, 0.46, 0.64, 0.82])
    ax_story.plot(x, y, color='#2ab7c9', lw=2.0)
    ax_story.scatter(x, y, s=42, color=['#0072b2', '#2aa198', '#d64b3a', '#9aa600'], edgecolor='white', zorder=4)
    for (xx, yy, txt, col) in [(0.08, 0.1, 'frontal\nalpha FC', '#0072b2'), (0.33, 0.56, 'network\ngeometry', '#2aa198'), (0.58, 0.74, 'topological\ncompression', '#d64b3a'), (0.83, 0.53, 'brain-state\nmarker', '#8a9500')]:
        ax_story.text(xx, yy, txt, transform=ax_story.transAxes, ha='center', fontsize=7.1, color=col)
    ax_story.text(0.18, 0.88, 'beyond conventional\nalpha-power summaries', transform=ax_story.transAxes, fontsize=7.2, color='#486471')
    ax_story.annotate('', xy=(0.48, 0.62), xytext=(0.32, 0.82), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.9, color='#486471'))
    panel(ax_story, 'e')
    save(fig, outdir, 'Original_04_FrontalAlphaFCCompression')

def nwcf_table17(config: str) -> pd.DataFrame:
    p = nwcf_dir(config) / 'Table17_NWCF_BiomarkerPanels.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def nwcf_representative(config: str, condition: str='hi', region: str='whole_head') -> tuple[np.ndarray, list[str], str]:
    return representative_distance(config, condition, region)

def fig5_nwcf_pipeline_schematic(original_config: str, nwcf_config: str, outdir: Path) -> None:
    t17 = nwcf_table17(nwcf_config).set_index('Feature_set')
    sel_ba = float(t17.loc['Selected_NWCF_Biomarker', 'Balanced_accuracy']) if 'Selected_NWCF_Biomarker' in t17.index else np.nan
    graph_ba = float(t17.loc['Graph_Metrics', 'Balanced_accuracy']) if 'Graph_Metrics' in t17.index else np.nan
    ph_ba = float(t17.loc['Classical_PH', 'Balanced_accuracy']) if 'Classical_PH' in t17.index else np.nan
    fig = plt.figure(figsize=(9.2, 6.1))
    header(fig, 'NWCF working on the STEW pipeline', 'The custom filtration is introduced after classical topology exposes pairwise-threshold dependence.')
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.8])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = [(0.05, 0.76, 'Raw STEW EEG', '14-channel workload\nrecordings', '#9aa600'), (0.25, 0.76, 'Preprocessing', 'ASR/QC + alpha\nband extraction', '#2aa198'), (0.45, 0.76, 'FC windows', 'transaction-level\nfunctional graphs', '#2ab7c9'), (0.65, 0.76, 'Classical topology', 'Rips PH on\npairwise distances', '#0072b2'), (0.28, 0.34, 'Limitation', 'unsupported edges can\nenter by threshold', '#d55e00'), (0.5, 0.34, 'NWCF', 'entry reweighted\nby clique support', '#d64b3a'), (0.72, 0.34, 'Biomarker panel', 'compact mesoscale\nworkload signature', '#8f4ea3'), (0.89, 0.34, 'Inference', 'task/rest workload\nclassification', '#7a9a01')]
    (box_w, box_h) = (0.145, 0.13)
    for (x, y, title, desc, color) in stages:
        ax.add_patch(Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, fc='white', ec=color, lw=1.4))
        ax.text(x, y + 0.025, title, ha='center', va='center', fontsize=8.0, fontweight='bold', color=color)
        ax.text(x, y - 0.028, desc, ha='center', va='center', fontsize=6.7, color='#333333')
    for (x0, y0, x1, y1) in [(0.125, 0.76, 0.175, 0.76), (0.325, 0.76, 0.375, 0.76), (0.525, 0.76, 0.575, 0.76), (0.65, 0.68, 0.36, 0.42), (0.36, 0.34, 0.425, 0.34), (0.575, 0.34, 0.645, 0.34), (0.795, 0.34, 0.835, 0.34)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='->', mutation_scale=10, lw=1.0, color='#486471'))
    x = np.linspace(0.08, 0.92, 200)
    y = 0.05 + 0.075 / (1 + np.exp(-12 * (x - 0.52)))
    ax.plot(x, y, color='#2ab7c9', lw=2.0)
    ax.text(0.09, 0.16, 'discovery', color='#9aa600', fontsize=7.2)
    ax.text(0.43, 0.16, 'custom filtration', color='#d64b3a', fontsize=7.2)
    ax.text(0.77, 0.16, 'biomarker', color='#7a9a01', fontsize=7.2)
    ax.text(0.07, 0.55, f'Result anchor ({nwcf_config}):\nSelected NWCF BA={sel_ba:.3f}\nGraph BA={graph_ba:.3f}\nClassical PH BA={ph_ba:.3f}', fontsize=7.1, bbox=dict(fc='white', ec='#888888', lw=0.7))
    ax.text(0.72, 0.55, 'Claim: NWCF is not a post-hoc feature;\nit changes the filtration space\nbefore PH summaries are extracted.', fontsize=7.4, color='#9f2f24')
    save(fig, outdir, 'NWCFWorking_01_PipelineSchematic')

def fig6_nwcf_brain_network_exemplar(nwcf_config: str, outdir: Path) -> None:
    (dist, labels, tag) = nwcf_representative(nwcf_config, 'hi', 'whole_head')
    (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
    sim = 1.0 - dist
    raw = dist[np.triu_indices_from(dist, k=1)]
    theta = float(np.percentile(raw, 18))
    nwcf_only = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if d_hat[i, j] <= theta and dist[i, j] > theta]
    nwcf_only.sort(key=lambda e: dist[e[0], e[1]] - d_hat[e[0], e[1]], reverse=True)
    thr_mat = (dist <= theta).astype(float)
    np.fill_diagonal(thr_mat, np.nan)
    shrink = dist - d_hat
    shrink[np.eye(len(shrink), dtype=bool)] = np.nan
    fig = plt.figure(figsize=(8.8, 6.9))
    header(fig, 'NWCF changes the actual STEW brain-network filtration', 'A real STEW transaction is mapped from cortical regions to pairwise FC and then to a clique-corroborated filtration.')
    ax_brain = fig.add_axes([0.055, 0.54, 0.46, 0.31])
    ax_thr = fig.add_axes([0.63, 0.57, 0.24, 0.25])
    ax_fc = fig.add_axes([0.2, 0.25, 0.27, 0.25])
    ax_nwcf = fig.add_axes([0.62, 0.25, 0.27, 0.25])
    ax_story = fig.add_axes([0.08, 0.07, 0.84, 0.12])
    ax_brain.set_title('STEW cortical EEG regions')
    ax_brain.set_xlim(0, 10)
    ax_brain.set_ylim(0, 4.4)
    ax_brain.axis('off')
    draw_cortical_surface(ax_brain, 1.35, 3.05, 0.64, '#d64b3a', 'frontal')
    draw_cortical_surface(ax_brain, 3.7, 3.05, 0.64, '#2aa198', 'central', medial=True)
    draw_cortical_surface(ax_brain, 6.05, 3.05, 0.64, '#8f4ea3', 'temporal', flip=True)
    draw_cortical_surface(ax_brain, 8.35, 3.05, 0.64, '#7a9a01', 'posterior', flip=True, medial=True)
    draw_cortical_surface(ax_brain, 2.3, 1.25, 0.78, '#0072b2', 'left network')
    draw_cortical_surface(ax_brain, 5.1, 1.25, 0.78, '#2ab7c9', 'clique support', medial=True)
    draw_cortical_surface(ax_brain, 7.9, 1.25, 0.78, '#d64b3a', 'right network', flip=True)
    ax_brain.text(0.18, 4.05, 'regional surfaces summarize the 14 EEG sensors', fontsize=7.0, color='#486471')
    ax_brain.text(0.18, 0.04, f'{nwcf_config}, {tag}', fontsize=6.8, color='#486471')
    panel(ax_brain, 'a')
    ax_thr.imshow(thr_mat, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax_thr.set_title('Pairwise threshold matrix', pad=10)
    ax_thr.set_xticks([])
    ax_thr.set_yticks([])
    ax_thr.text(0.02, -0.1, 'edges enter by distance only', transform=ax_thr.transAxes, fontsize=6.4, color='#555555')
    ax_thr.text(-0.16, 1.1, 'b', transform=ax_thr.transAxes, fontweight='bold', fontsize=11, va='bottom')
    im = ax_fc.imshow(sim, cmap='turbo', vmin=np.nanmin(sim), vmax=np.nanmax(sim), interpolation='nearest')
    ax_fc.set_title('Classical FC heatmap')
    ax_fc.set_xticks([])
    ax_fc.set_yticks([])
    cbar1 = fig.colorbar(im, ax=ax_fc, fraction=0.032, pad=0.015)
    cbar1.ax.tick_params(labelsize=6)
    ax_fc.text(0.02, -0.13, 'what graph metrics / classical PH see: pairwise intensity geometry', transform=ax_fc.transAxes, fontsize=6.8, color='#0072b2')
    panel(ax_fc, 'c')
    nwcf_display = 1.0 - d_hat
    im2 = ax_nwcf.imshow(nwcf_display, cmap='turbo', vmin=np.nanmin(nwcf_display), vmax=np.nanmax(nwcf_display), interpolation='nearest')
    ax_nwcf.set_title('NWCF clique-corroborated heatmap')
    ax_nwcf.set_xticks([])
    ax_nwcf.set_yticks([])
    cbar2 = fig.colorbar(im2, ax=ax_nwcf, fraction=0.032, pad=0.015)
    cbar2.ax.tick_params(labelsize=6)
    for (i, j) in nwcf_only[:10]:
        ax_nwcf.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='none', ec='#d64b3a', lw=0.75))
        ax_nwcf.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc='none', ec='#d64b3a', lw=0.75))
    ax_nwcf.text(0.02, -0.13, 'what NWCF adds: clique-supported edges move earlier in filtration', transform=ax_nwcf.transAxes, fontsize=6.8, color='#9f2f24')
    panel(ax_nwcf, 'd')
    for (start, end, txt) in [(ax_brain, ax_thr, 'functional\ncoupling'), (ax_thr, ax_fc, ''), (ax_thr, ax_nwcf, '')]:
        (p0, p1) = (start.get_position(), end.get_position())
        if start is ax_brain:
            xy0 = (p0.x1 + 0.006, (p0.y0 + p0.y1) / 2)
            xy1 = (p1.x0 - 0.01, (p1.y0 + p1.y1) / 2)
            text_xy = ((xy0[0] + xy1[0]) / 2, xy0[1] + 0.018)
        else:
            x0 = (p0.x0 + p0.x1) / 2
            xy0 = (x0 - 0.06 if end is ax_fc else x0 + 0.06, p0.y0 - 0.01)
            xy1 = ((p1.x0 + p1.x1) / 2, p1.y1 + 0.014)
            text_xy = ((xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2 + 0.018)
        fig.add_artist(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle='->', mutation_scale=10, lw=0.9, color='#486471'))
        if txt:
            fig.text(text_xy[0], text_xy[1], txt, ha='center', va='center', fontsize=6.5, color='#486471')
    clean_axis_arrows(ax_story, 'filtration scale', 'mesoscale evidence')
    x = np.array([0.08, 0.32, 0.58, 0.84])
    y = np.array([0.14, 0.32, 0.6, 0.84])
    ax_story.plot(x, y, color='#2ab7c9', lw=2.0)
    ax_story.scatter(x, y, s=46, c=['#808080', '#2aa198', '#8f4ea3', '#d64b3a'], edgecolor='white', zorder=3)
    ax_story.text(0.08, 0.04, 'pairwise\nedge', transform=ax_story.transAxes, ha='center', fontsize=7.1)
    ax_story.text(0.32, 0.4, 'triangle\nsupport', transform=ax_story.transAxes, ha='center', fontsize=7.1, color='#2aa198')
    ax_story.text(0.58, 0.68, '4-clique\nsupport', transform=ax_story.transAxes, ha='center', fontsize=7.1, color='#8f4ea3')
    ax_story.text(0.84, 0.58, 'earlier NWCF\nentry', transform=ax_story.transAxes, ha='center', fontsize=7.1, color='#d64b3a')
    ax_story.text(0.05, 0.88, f"corroborated={diag['fraction_corroborated'] * 100:.1f}%\n4-clique wins={diag.get('fraction_4clique_wins', 0) * 100:.1f}%\nred boxes mark NWCF-advanced entries", transform=ax_story.transAxes, fontsize=7.0, color='#486471')
    panel(ax_story, 'e')
    save(fig, outdir, f'NWCFWorking_02_BrainNetworkExemplar_{nwcf_config}')

def fig7_nwcf_filtration_transfer(nwcf_config: str, outdir: Path) -> None:
    rows = []
    diag_rows = []
    for (condition, label, color) in [('lo', 'Rest', '#0072b2'), ('hi', 'Task', '#d64b3a')]:
        (dist, _, _) = nwcf_representative(nwcf_config, condition, 'whole_head')
        (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
        diag_rows.append({'condition': label, 'corroborated': diag['fraction_corroborated'], 'four_clique': diag.get('fraction_4clique_wins', np.nan), 'shrinkage': diag['mean_relative_shrinkage']})
        raw = dist[np.triu_indices_from(dist, k=1)]
        mod = d_hat[np.triu_indices_from(d_hat, k=1)]
        for p in np.arange(5, 61, 5):
            theta = float(np.percentile(raw, p))
            classical = int((raw <= theta).sum())
            nwcf = int((mod <= theta).sum())
            rows.append({'condition': label, 'pct': p, 'classical': classical, 'nwcf': nwcf, 'nwcf_only': max(0, nwcf - classical), 'gain': (nwcf - classical) / max(1, classical), 'color': color})
    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    fig = plt.figure(figsize=(8.8, 6.4))
    header(fig, 'NWCF changes the filtration trajectory on STEW FC', 'The edge-growth curve itself changes before topology is summarized.')
    gs = fig.add_gridspec(3, 3, left=0.08, right=0.95, top=0.86, bottom=0.09, hspace=0.5, wspace=0.32, height_ratios=[1, 0.78, 0.68])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for (cond, color, marker) in [('Rest', '#0072b2', 'o'), ('Task', '#d64b3a', '^')]:
        sub = df[df['condition'].eq(cond)]
        axes[0].plot(sub['pct'], sub['classical'], color=color, ls='--', lw=1.5, marker=marker, ms=3, label=f'{cond} classical')
        axes[0].plot(sub['pct'], sub['nwcf'], color=color, lw=1.8, marker=marker, ms=3, label=f'{cond} NWCF')
        axes[1].plot(sub['pct'], sub['nwcf_only'], color=color, lw=1.8, marker=marker, ms=3, label=cond)
        axes[2].plot(sub['pct'], sub['gain'], color=color, lw=1.8, marker=marker, ms=3, label=cond)
    for (ax, title, ylabel, lab) in zip(axes, ['Filtration edge growth', 'NWCF-only entries', 'Clique gain'], ['edges', 'new edges', 'gain ratio'], ['a', 'b', 'c']):
        ax.set_title(title)
        ax.set_xlabel('distance percentile threshold')
        ax.set_ylabel(ylabel)
        ax.grid(True, color='#eeeeee', lw=0.45)
        ax.legend(fontsize=6.5, loc='best')
        panel(ax, lab)
    axd = fig.add_subplot(gs[1, 0])
    xloc = np.arange(len(diag_df))
    w = 0.26
    axd.bar(xloc - w, diag_df['corroborated'] * 100, width=w, color='#2aa198', edgecolor='white', label='corroborated')
    axd.bar(xloc, diag_df['four_clique'] * 100, width=w, color='#8f4ea3', edgecolor='white', label='4-clique wins')
    axd.bar(xloc + w, diag_df['shrinkage'] * 100, width=w, color='#d64b3a', edgecolor='white', label='relative shrinkage')
    axd.set_xticks(xloc, diag_df['condition'])
    axd.set_ylabel('% / scaled %')
    axd.set_title('edge-level diagnostics')
    axd.grid(True, axis='y', color='#eeeeee', lw=0.45)
    axd.legend(fontsize=6.3, loc='upper right')
    panel(axd, 'd')
    axm = fig.add_subplot(gs[1, 1:])
    pivot = df.pivot(index='pct', columns='condition', values='nwcf_only')
    axm.plot(pivot.index, pivot.get('Rest', pd.Series(index=pivot.index, dtype=float)), color='#0072b2', lw=1.8, marker='o', ms=3, label='Rest')
    axm.plot(pivot.index, pivot.get('Task', pd.Series(index=pivot.index, dtype=float)), color='#d64b3a', lw=1.8, marker='^', ms=3, label='Task')
    axm.fill_between(pivot.index, pivot.min(axis=1), pivot.max(axis=1), color='#2ab7c9', alpha=0.12)
    axm.set_title('where NWCF adds most edges')
    axm.set_xlabel('distance percentile threshold')
    axm.set_ylabel('NWCF-only entries')
    axm.grid(True, color='#eeeeee', lw=0.45)
    axm.legend(fontsize=6.5)
    panel(axm, 'e')
    ax = fig.add_subplot(gs[2, :])
    clean_axis_arrows(ax, 'threshold sweep', 'filtration-level evidence')
    x = np.linspace(0.08, 0.92, 160)
    ax.plot(x, 0.18 + 0.55 / (1 + np.exp(-12 * (x - 0.45))), color='#2ab7c9', lw=2.0)
    ax.plot(x, 0.14 + 0.7 / (1 + np.exp(-13 * (x - 0.35))), color='#d64b3a', lw=1.6, ls='--')
    ax.text(0.16, 0.3, 'classical path', transform=ax.transAxes, color='#0072b2', fontsize=7.2)
    ax.text(0.56, 0.72, 'NWCF path:\nclique-supported edges\nenter earlier', transform=ax.transAxes, color='#9f2f24', fontsize=7.2)
    ax.annotate('mechanistic proof', xy=(0.48, 0.58), xytext=(0.28, 0.82), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=7.3)
    panel(ax, 'f')
    save(fig, outdir, f'NWCFWorking_03_FiltrationTransfer_{nwcf_config}')

def transaction_trace(config: str, condition: str, region: str='whole_head') -> pd.DataFrame:
    (lo, hi) = raw_subject_pair(config)
    path = hi if condition == 'hi' else lo
    data = load_json(path)
    rows = []
    for win in data['fc_regions'][region].get('per_window', []):
        sim = np.asarray(win['similarity_matrix'], dtype=float)
        dist = _fc_to_base_distance(sim)
        feats = nwcf_window_features(dist, max_clique_order=4)
        rows.append({'time': float(win.get('window_time_sec', len(rows))), 'condition': 'Task' if condition == 'hi' else 'Rest', 'frac_4clique': feats.get('nwcf_fraction_4clique_corroborated', np.nan), 'mean_4clique': feats.get('nwcf_mean_4clique_support', np.nan), 'shrinkage': feats.get('nwcf_mean_relative_shrinkage', np.nan), 'h0': feats.get('nwcf_max_persistence_H0', np.nan)})
    return pd.DataFrame(rows)

def draw_cortical_surface(ax, x0: float, y0: float, scale: float, color: str, label: str, flip: bool=False, medial: bool=False) -> None:
    sx = -1 if flip else 1
    verts = [(x0 + sx * -1.35 * scale, y0 - 0.1 * scale), (x0 + sx * -1.16 * scale, y0 + 0.54 * scale), (x0 + sx * -0.35 * scale, y0 + 0.74 * scale), (x0 + sx * 0.58 * scale, y0 + 0.66 * scale), (x0 + sx * 1.27 * scale, y0 + 0.24 * scale), (x0 + sx * 1.34 * scale, y0 - 0.35 * scale), (x0 + sx * 0.64 * scale, y0 - 0.62 * scale), (x0 + sx * -0.25 * scale, y0 - 0.7 * scale), (x0 + sx * -1.1 * scale, y0 - 0.46 * scale), (x0 + sx * -1.35 * scale, y0 - 0.1 * scale)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    brain = PathPatch(MplPath(verts, codes), fc='#8f8f8f', ec='#4f4f4f', lw=0.7, alpha=0.96)
    ax.add_patch(brain)
    ax.add_patch(Ellipse((x0 + sx * 0.1 * scale, y0 - 0.06 * scale), 2.12 * scale, 0.86 * scale, angle=-7 * sx, fc='#bdbdbd', ec='none', alpha=0.35))
    parcel_sets = [[(-0.98, 0.1), (-0.54, 0.45), (-0.1, 0.32), (-0.24, -0.05)], [(-0.18, 0.38), (0.32, 0.5), (0.58, 0.16), (0.08, -0.02)], [(0.28, 0.05), (0.94, 0.16), (0.8, -0.3), (0.2, -0.34)], [(-0.82, -0.24), (-0.22, -0.16), (0.04, -0.5), (-0.54, -0.58)]]
    active = 1 if medial else 2 if flip else 0
    for (k, poly) in enumerate(parcel_sets):
        fc = color if k == active else '#d6d25a'
        alpha = 0.92 if k == active else 0.52
        pts = [(x0 + sx * px * scale, y0 + py * scale) for (px, py) in poly]
        patch = Polygon(pts, closed=True, fc=fc, ec='#555555', lw=0.25, alpha=alpha)
        patch.set_clip_path(brain)
        ax.add_patch(patch)
    sulci = [[(-1.02, -0.06), (-0.56, 0.1), (-0.12, 0.27), (0.48, 0.24), (0.98, 0.05)], [(-0.95, -0.3), (-0.4, -0.1), (0.12, -0.02), (0.72, -0.2)], [(-0.78, 0.28), (-0.18, 0.18), (0.36, 0.4), (0.94, 0.24)], [(-0.4, -0.52), (-0.1, -0.22), (0.34, -0.1), (0.84, -0.38)]]
    for curve in sulci:
        xs = [x0 + sx * px * scale for (px, _) in curve]
        ys = [y0 + py * scale for (_, py) in curve]
        (ln,) = ax.plot(xs, ys, color='#595959', lw=0.45, alpha=0.55)
        ln.set_clip_path(brain)
    if medial:
        ax.add_patch(Ellipse((x0 + sx * 0.18 * scale, y0 - 0.12 * scale), 1.36 * scale, 0.52 * scale, angle=-5 * sx, fc='#666666', ec='#555555', lw=0.35, alpha=0.75))
        ax.add_patch(Ellipse((x0 + sx * 0.02 * scale, y0 + 0.12 * scale), 1.05 * scale, 0.16 * scale, angle=3 * sx, fc='#d6d25a', ec='#555555', lw=0.25, alpha=0.82))
    ax.text(x0, y0 - 0.92 * scale, label, ha='center', va='top', fontsize=6.6, color='#333333')

def fig8_nwcf_transaction_dynamics(nwcf_config: str, outdir: Path) -> None:
    lo = transaction_trace(nwcf_config, 'lo')
    hi = transaction_trace(nwcf_config, 'hi')
    df = pd.concat([lo, hi], ignore_index=True)
    metrics = [('frac_4clique', '4-clique corroboration'), ('mean_4clique', 'mean 4-clique support'), ('shrinkage', 'relative shrinkage'), ('h0', 'H0 max persistence')]
    fig = plt.figure(figsize=(8.8, 6.5))
    header(fig, 'NWCF operates at the transaction/window level', 'Window-wise trajectories show dynamic workload signatures, not only subject-level averages.')
    gs = fig.add_gridspec(3, 3, left=0.08, right=0.95, top=0.86, bottom=0.09, hspace=0.52, wspace=0.32, height_ratios=[1, 0.78, 0.72])
    for (i, (metric, title)) in enumerate(metrics[:3]):
        ax = fig.add_subplot(gs[0, i])
        for (cond, color, marker) in [('Rest', '#0072b2', 'o'), ('Task', '#d64b3a', '^')]:
            sub = df[df['condition'].eq(cond)]
            ax.plot(sub['time'], sub[metric], color=color, lw=1.4, marker=marker, markevery=max(1, len(sub) // 10), ms=3, label=cond)
        ax.set_title(title)
        ax.set_xlabel('window time (s)')
        ax.grid(True, color='#eeeeee', lw=0.45)
        if i == 0:
            ax.set_ylabel('NWCF feature')
            ax.legend(fontsize=6.7)
        panel(ax, chr(ord('a') + i))
    axh = fig.add_subplot(gs[1, 0:2])
    mat = df[df['condition'].eq('Task')][[m[0] for m in metrics]].T.astype(float)
    mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
    im = axh.imshow(mat.values, aspect='auto', cmap='coolwarm', vmin=-2.2, vmax=2.2, interpolation='nearest')
    axh.set_yticks(np.arange(len(metrics)), [m[1] for m in metrics], fontsize=7.0)
    axh.set_xlabel('task transaction window')
    axh.set_title('Task transaction fingerprint')
    cbar = fig.colorbar(im, ax=axh, fraction=0.028, pad=0.012)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.set_title('z', fontsize=6.5, pad=2)
    panel(axh, 'd')
    axb = fig.add_subplot(gs[1, 2])
    summary = []
    for (metric, title) in metrics:
        rest = df[df['condition'].eq('Rest')][metric].astype(float).dropna()
        task = df[df['condition'].eq('Task')][metric].astype(float).dropna()
        summary.append({'metric': title, 'delta': float(task.mean() - rest.mean())})
    sdf = pd.DataFrame(summary)
    yy = np.arange(len(sdf))
    axb.barh(yy, sdf['delta'], color=['#d64b3a' if v > 0 else '#0072b2' for v in sdf['delta']], edgecolor='white')
    axb.axvline(0, color='black', lw=0.8)
    short_metric_labels = ['4-clique', '4-clique support', 'shrinkage', 'H0 persistence']
    axb.set_yticks(yy, short_metric_labels, fontsize=6.4)
    axb.invert_yaxis()
    axb.set_xlabel('Task - Rest')
    axb.set_title('mean transaction shift')
    axb.grid(True, axis='x', color='#eeeeee', lw=0.45)
    panel(axb, 'e')
    ax = fig.add_subplot(gs[2, :])
    clean_axis_arrows(ax, 'time', 'filtration state')
    x = np.array([0.08, 0.3, 0.52, 0.74, 0.9])
    y = np.array([0.2, 0.36, 0.68, 0.62, 0.78])
    ax.plot(x, y, color='#2ab7c9', lw=2.0)
    ax.scatter(x, y, s=45, c=['#0072b2', '#2aa198', '#d64b3a', '#8f4ea3', '#7a9a01'], edgecolor='white', zorder=3)
    ax.text(0.09, 0.08, 'early\nwindows', transform=ax.transAxes, ha='center', fontsize=7.0, color='#0072b2')
    ax.text(0.34, 0.42, 'clique support\nfluctuates', transform=ax.transAxes, ha='center', fontsize=7.0, color='#2aa198')
    ax.text(0.55, 0.75, 'edge shrinkage\nseparates task', transform=ax.transAxes, ha='center', fontsize=7.0, color='#d64b3a')
    ax.text(0.84, 0.62, 'dynamic\nbiomarker state', transform=ax.transAxes, ha='center', fontsize=7.0, color='#7a9a01')
    phase_bar(ax, 0.08, 0.28, 0.98, 'window evidence', '#0072b2')
    phase_bar(ax, 0.34, 0.62, 0.98, 'mesoscale transition', '#2aa198')
    phase_bar(ax, 0.68, 0.92, 0.98, 'state signature', '#7a9a01')
    panel(ax, 'f')
    save(fig, outdir, f'NWCFWorking_04_TransactionDynamics_{nwcf_config}')

def write_summary(config: str, nwcf_config: str, outdir: Path) -> None:
    lines = ['Original topology illustrative figures', '======================================', '', f'Config used: {config}', f'NWCF working config used: {nwcf_config}', '', 'PNG files:', '  Original_01_WorkloadStructure.png', '  Original_02_FeatureUniverse.png', '  Original_03_ClassicalThresholdLimitation.png', '  Original_04_FrontalAlphaFCCompression.png', '  NWCFWorking_01_PipelineSchematic.png', f'  NWCFWorking_02_BrainNetworkExemplar_{nwcf_config}.png', f'  NWCFWorking_03_FiltrationTransfer_{nwcf_config}.png', f'  NWCFWorking_04_TransactionDynamics_{nwcf_config}.png', '', 'Narrative:', '  1. Original topology detects workload structure.', '  2. The original feature universe is rich but high-dimensional.', '  3. Classical Rips/PH has pairwise threshold dependence.', '  4. Frontal alpha FC compression motivates a custom filtration such as NWCF.', '  5. NWCF is shown working directly on STEW windows, networks, and filtrations.']
    (outdir / 'original_topology_figure_summary.txt').write_text('\n'.join(lines) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='84aec', help='Original topology config to visualize. Default uses alpha envelope correlation.')
    parser.add_argument('--nwcf-config', default='84corr', help='NWCF config used for the NWCF-working figures.')
    parser.add_argument('--output-dir', default='stew_original_topology_illustrative_figures')
    args = parser.parse_args()
    set_style()
    outdir = PROJECT_ROOT / args.output_dir
    fig1_workload_structure(args.config, outdir)
    fig2_feature_universe(args.config, outdir)
    fig3_threshold_limitation(args.config, outdir)
    fig4_frontal_alpha_compression(args.config, outdir)
    fig5_nwcf_pipeline_schematic(args.config, args.nwcf_config, outdir)
    fig6_nwcf_brain_network_exemplar(args.nwcf_config, outdir)
    fig7_nwcf_filtration_transfer(args.nwcf_config, outdir)
    fig8_nwcf_transaction_dynamics(args.nwcf_config, outdir)
    write_summary(args.config, args.nwcf_config, outdir)
    print(f'Wrote original topology illustrative figures to {outdir}')
if __name__ == '__main__':
    main()
