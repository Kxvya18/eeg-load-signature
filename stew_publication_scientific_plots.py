from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from stew_nwcf import compute_clique_cohesion_distance, nwcf_window_features
from stew_topology_core import _fc_to_base_distance
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = ['123plv', '83aec', '84aec', '84corr', '84plv']
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SCALP_POS = {'AF3': (-0.38, 0.92), 'F7': (-0.82, 0.58), 'F3': (-0.42, 0.55), 'FC5': (-0.62, 0.2), 'T7': (-0.95, 0.0), 'P7': (-0.72, -0.48), 'O1': (-0.3, -0.86), 'O2': (0.3, -0.86), 'P8': (0.72, -0.48), 'T8': (0.95, 0.0), 'FC6': (0.62, 0.2), 'F4': (0.42, 0.55), 'F8': (0.82, 0.58), 'AF4': (0.38, 0.92)}
METHOD_ORDER = ['Original', 'Original+Selected_NWCF', 'Selected_NWCF_Biomarker', 'NWCF_Clique_Corroboration', 'NWCF_Higher_Order_4Clique', 'NWCF_Transaction_Dynamics', 'Graph_Metrics', 'Classical_PH']
METHOD_LABELS = {'Original': 'Original topology', 'Original+Selected_NWCF': 'Original + NWCF', 'Selected_NWCF_Biomarker': 'Selected NWCF', 'NWCF_Clique_Corroboration': 'Clique NWCF', 'NWCF_Higher_Order_4Clique': '4-clique NWCF', 'NWCF_Transaction_Dynamics': 'Transaction NWCF', 'Graph_Metrics': 'Graph metrics', 'Classical_PH': 'Classical PH'}
METHOD_COLORS = {'Original': '#222222', 'Original+Selected_NWCF': '#d62728', 'Selected_NWCF_Biomarker': '#1b9e77', 'NWCF_Clique_Corroboration': '#0072b2', 'NWCF_Higher_Order_4Clique': '#984ea3', 'NWCF_Transaction_Dynamics': '#e69f00', 'Graph_Metrics': '#666666', 'Classical_PH': '#9e9e9e'}
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

def set_scientific_style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.labelsize': 8.2, 'axes.titlesize': 8.8, 'axes.linewidth': 0.9, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0, 'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': 'black', 'figure.dpi': 160, 'savefig.dpi': 450, 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})

def panel_label(ax, label: str) -> None:
    ax.text(-0.075, 1.04, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')

def header(fig, title: str, subtitle: str | None=None) -> None:
    fig.text(0.055, 0.965, title, ha='left', va='top', fontsize=10.8, fontweight='bold')
    if subtitle:
        fig.text(0.055, 0.925, subtitle, ha='left', va='top', fontsize=7.8, color='#486471')

def clean_axis_arrows(ax, xlabel: str, ylabel: str) -> None:
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
    ax.annotate('', xy=(0, 1.03), xytext=(0, 0), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
    ax.text(0.5, -0.08, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.text(-0.08, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center', rotation=90, fontsize=8)

def phase_bar(ax, x0: float, x1: float, y: float, text: str, color: str) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=1.3, clip_on=False)
    ax.plot([x0, x0], [y - 0.018, y + 0.018], color=color, lw=1.0, clip_on=False)
    ax.plot([x1, x1], [y - 0.018, y + 0.018], color=color, lw=1.0, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.035, text, color=color, ha='center', va='bottom', fontsize=7.2)

def output_dirs(configs: list[str]) -> dict[str, tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for cfg in configs:
        nwcf_dir = PROJECT_ROOT / f'stew_output_{cfg}'
        original_dir = PROJECT_ROOT / f'stew_tda_results_final_{cfg}'
        if (nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv').exists() and original_dir.exists():
            found[cfg] = (nwcf_dir, original_dir)
        else:
            print(f'[skip] {cfg}: missing finished NWCF/original outputs')
    return found

def savefig(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{stem}.png')
    plt.close(fig)

def load_table17(dirs: dict[str, tuple[Path, Path]]) -> pd.DataFrame:
    frames = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        df = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv')
        df['config'] = cfg
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def plot_performance_landscape(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    df = load_table17(dirs)
    if df.empty:
        return
    configs = list(dirs)
    (fig, axes) = plt.subplots(1, 2, figsize=(8.8, 3.25), sharex=True)
    for (ax, metric, ylabel) in [(axes[0], 'Balanced_accuracy', 'Balanced accuracy'), (axes[1], 'ROC_AUC', 'ROC AUC')]:
        for (idx, method) in enumerate(METHOD_ORDER):
            sub = df[df['Feature_set'].eq(method)].set_index('config').reindex(configs)
            if sub[metric].notna().sum() == 0:
                continue
            ax.plot(configs, sub[metric], marker=MARKERS[idx % len(MARKERS)], linewidth=1.4, markersize=5.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(0.5, color='black', linestyle=':', linewidth=0.9)
        ax.set_ylim(0.36, 0.9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Configuration')
        ax.grid(True, which='major', linestyle='--', linewidth=0.55, color='#bdbdbd')
        ax.minorticks_on()
        ax.tick_params(axis='x', rotation=25)
    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')
    axes[0].set_title('Cross-config classifier landscape')
    axes[1].set_title('Discrimination retained in ROC space')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    savefig(fig, outdir, 'FigureS1_PerformanceLandscape')

def paired_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = df.pivot_table(index='subject_id', columns='condition', values=col, aggfunc='mean')
    if 'lo' not in p.columns or 'hi' not in p.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = p[['lo', 'hi']].dropna()
    p['delta'] = p['hi'] - p['lo']
    return p.reset_index()

def dz_effect(delta: pd.Series) -> float:
    delta = pd.to_numeric(delta, errors='coerce').dropna()
    sd = float(delta.std(ddof=1))
    if len(delta) < 3 or sd <= 1e-12:
        return np.nan
    return float(delta.mean() / sd)

def plot_original_topology_evidence(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (_, original_dir) = dirs[exemplar]
    sf = original_dir / 'subject_level_features.csv'
    if not sf.exists():
        return
    df = pd.read_csv(sf)
    metric_sets = [('mean_max_persistence_H1', 'H1 max persistence'), ('mean_landscape_total_area_H1', 'Landscape area')]
    channel_rows = []
    for ch in CHANNELS:
        for (suffix, label) in metric_sets:
            col = f'{ch}_{suffix}'
            pv = paired_values(df, col)
            if not pv.empty:
                channel_rows.append({'channel': ch, 'metric': label, 'dz': dz_effect(pv['delta']), 'median_delta': float(pv['delta'].median())})
    chdf = pd.DataFrame(channel_rows)
    if chdf.empty:
        return
    (fig, axes) = plt.subplots(1, 3, figsize=(10.2, 3.25))
    for (ax, metric, lab) in zip(axes[:2], ['H1 max persistence', 'Landscape area'], ['a', 'b']):
        sub = chdf[chdf['metric'].eq(metric)]
        vals = {r['channel']: r['dz'] for (_, r) in sub.iterrows()}
        xy = np.array([SCALP_POS[ch] for ch in CHANNELS])
        z = np.array([vals.get(ch, np.nan) for ch in CHANNELS])
        head = plt.Circle((0, 0), 1.04, fill=False, color='black', linewidth=1.0)
        ax.add_patch(head)
        ax.plot([0, -0.08, 0.08, 0], [1.04, 1.16, 1.04, 1.04], color='black', linewidth=0.9)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap='coolwarm', vmin=-1.0, vmax=1.0, s=125, edgecolor='black')
        for (ch, (x, y)) in SCALP_POS.items():
            ax.text(x, y - 0.145, ch, ha='center', va='center', fontsize=6.5)
        ax.set_title(f'Original topology: {metric}')
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.12, 1.22)
        ax.set_axis_off()
        panel_label(ax, lab)
    fig.colorbar(sc, ax=axes[:2], fraction=0.035, pad=0.015, label='paired task-rest effect dz')
    top = chdf[chdf['metric'].eq('H1 max persistence')].iloc[chdf[chdf['metric'].eq('H1 max persistence')]['dz'].abs().argmax()]
    col = f"{top['channel']}_mean_max_persistence_H1"
    pv = paired_values(df, col)
    ax = axes[2]
    for (_, row) in pv.iterrows():
        ax.plot([0, 1], [row['lo'], row['hi']], color='#9e9e9e', linewidth=0.65, alpha=0.55)
    med = pv[['lo', 'hi']].median()
    ax.plot([0, 1], med.values, color='#d62728', linewidth=2.2, marker='o', markersize=5)
    ax.set_xticks([0, 1], ['Rest', 'Task'])
    ax.set_ylabel(col.replace('_', ' '))
    ax.set_title(f"Paired original PH shift: {top['channel']}")
    ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
    ax.text(0.03, 0.95, f"dz={top['dz']:.2f}\nN={len(pv)} paired", transform=ax.transAxes, ha='left', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
    panel_label(ax, 'c')
    savefig(fig, outdir, f'OriginalTopology_ScalpAndPairedShift_{exemplar}')

def plot_original_to_nwcf_bridge(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    rows = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv')
        s = t17.set_index('Feature_set')
        if 'Original' in s.index and 'Selected_NWCF_Biomarker' in s.index:
            rows.append({'config': cfg, 'Original_BA': float(s.loc['Original', 'Balanced_accuracy']), 'NWCF_BA': float(s.loc['Selected_NWCF_Biomarker', 'Balanced_accuracy']), 'Graph_BA': float(s.loc['Graph_Metrics', 'Balanced_accuracy']) if 'Graph_Metrics' in s.index else np.nan, 'ClassicalPH_BA': float(s.loc['Classical_PH', 'Balanced_accuracy']) if 'Classical_PH' in s.index else np.nan, 'Original_size': float(s.loc['Original', 'N_features']), 'NWCF_size': float(s.loc['Selected_NWCF_Biomarker', 'N_features'])})
    bridge = pd.DataFrame(rows)
    if bridge.empty:
        return
    (fig, axes) = plt.subplots(1, 2, figsize=(8.2, 3.15))
    ax = axes[0]
    x = np.arange(len(bridge))
    width = 0.18
    for (off, col, color, label) in [(-1.5, 'Original_BA', '#222222', 'Original topology'), (-0.5, 'NWCF_BA', '#1b9e77', 'Selected NWCF'), (0.5, 'Graph_BA', '#777777', 'Graph'), (1.5, 'ClassicalPH_BA', '#bdbdbd', 'Classical PH')]:
        ax.bar(x + off * width, bridge[col], width=width, color=color, edgecolor='black', linewidth=0.5, label=label)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=0.9)
    ax.set_xticks(x, bridge['config'], rotation=25, ha='right')
    ax.set_ylim(0.36, 0.9)
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Why NWCF: compact method against proper baselines')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='#bdbdbd')
    panel_label(ax, 'a')
    ax = axes[1]
    size_ratio = bridge['Original_size'] / bridge['NWCF_size']
    ba_gap = bridge['NWCF_BA'] - bridge['Original_BA']
    sc = ax.scatter(size_ratio, ba_gap, s=80, c=bridge['NWCF_BA'], cmap='viridis', edgecolor='black')
    for (_, r) in bridge.iterrows():
        ax.text(r['Original_size'] / r['NWCF_size'] * 1.01, r['NWCF_BA'] - r['Original_BA'], r['config'], fontsize=7)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('Original feature count / selected NWCF count')
    ax.set_ylabel('Selected NWCF - Original BA')
    ax.set_title('Compactness versus performance')
    ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label='NWCF BA')
    panel_label(ax, 'b')
    savefig(fig, outdir, 'OriginalToNWCF_Bridge_Compactness')

def feature_family(name: str) -> str:
    low = name.lower()
    if '4clique' in low:
        return '4-clique'
    if 'triangle' in low or 'corroborated' in low or 'clique_synchrony' in low:
        return 'clique/triangle'
    if 'shrinkage' in low or 'edge_distance' in low:
        return 'edge reweighting'
    if '_txn_' in low:
        return 'transaction'
    if 'landscape' in low or 'persistence_image' in low:
        return 'vectorized PH'
    if '_h1' in low:
        return 'H1 topology'
    if '_h0' in low:
        return 'H0 topology'
    return 'other NWCF'

def plot_mesoscale_biomarker_map(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    rows = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        path = nwcf_dir / 'Table1_ControlAnalysis.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df['Feature'].str.contains('NWCF', case=False, na=False)].copy()
        if 'Survives_control' in df.columns:
            df = df[df['Survives_control'].astype(bool)]
        if df.empty:
            continue
        df['family'] = df['Feature'].map(feature_family)
        effect_col = 'r_partial_both' if 'r_partial_both' in df.columns else 'r_vs_condition'
        for (fam, sub) in df.groupby('family'):
            rows.append({'config': cfg, 'family': fam, 'max_abs_effect': float(sub[effect_col].abs().max()), 'n_features': int(len(sub))})
    if not rows:
        return
    stats = pd.DataFrame(rows)
    configs = list(dirs)
    families = ['4-clique', 'clique/triangle', 'edge reweighting', 'transaction', 'vectorized PH', 'H0 topology', 'H1 topology', 'other NWCF']
    z = stats.pivot(index='family', columns='config', values='max_abs_effect').reindex(families).reindex(configs, axis=1)
    n = stats.pivot(index='family', columns='config', values='n_features').reindex(families).reindex(configs, axis=1)
    z = z.dropna(how='all')
    n = n.reindex(z.index)
    (fig, ax) = plt.subplots(figsize=(6.8, 4.15))
    im = ax.imshow(z.values, cmap='inferno', vmin=0, vmax=max(0.55, np.nanmax(z.values)))
    ax.set_xticks(np.arange(len(configs)), labels=configs, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(z.index)), labels=z.index)
    ax.set_xlabel('Configuration')
    ax.set_title('NWCF mesoscale families surviving alpha/FC control')
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z.values[i, j]):
                ax.text(j, i, f'{z.values[i, j]:.2f}\n(n={int(n.values[i, j])})', ha='center', va='center', fontsize=6.6, color='white' if z.values[i, j] > 0.32 else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.025)
    cbar.set_label('max |partial r|')
    ax.set_aspect('auto')
    panel_label(ax, 'a')
    savefig(fig, outdir, 'FigureS2_MesoscaleBiomarkerMap')

def plot_nwcf_phase_space(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (nwcf_dir, _) = dirs[exemplar]
    feat_path = nwcf_dir / 'advanced_topology_features.csv'
    if not feat_path.exists():
        return
    df = pd.read_csv(feat_path)
    regions = ['frontal', 'whole_head', 'posterior']
    (fig, axes) = plt.subplots(1, 3, figsize=(10.4, 3.25), sharex=False, sharey=False)
    for (ax, region, lab) in zip(axes, regions, ['a', 'b', 'c']):
        xcol = f'ADVTOPO_{region}_NWCF_nwcf_txn_mean_fraction_4clique_corroborated'
        ycol = f'ADVTOPO_{region}_NWCF_nwcf_txn_mean_mean_relative_shrinkage'
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_axis_off()
            continue
        lo = df[df['condition'].eq('lo')][['subject_id', xcol, ycol]].rename(columns={xcol: 'x_lo', ycol: 'y_lo'})
        hi = df[df['condition'].eq('hi')][['subject_id', xcol, ycol]].rename(columns={xcol: 'x_hi', ycol: 'y_hi'})
        p = lo.merge(hi, on='subject_id').dropna()
        for (_, r) in p.iterrows():
            ax.plot([r['x_lo'], r['x_hi']], [r['y_lo'], r['y_hi']], color='#bdbdbd', linewidth=0.55, alpha=0.58)
        ax.scatter(p['x_lo'], p['y_lo'], s=30, marker='o', color='#0072b2', edgecolor='black', linewidth=0.35, label='Rest')
        ax.scatter(p['x_hi'], p['y_hi'], s=34, marker='^', color='#d62728', edgecolor='black', linewidth=0.35, label='Task')
        dx = p['x_hi'].median() - p['x_lo'].median()
        dy = p['y_hi'].median() - p['y_lo'].median()
        ax.arrow(p['x_lo'].median(), p['y_lo'].median(), dx, dy, color='black', width=0.0006, head_width=0.012, length_includes_head=True)
        ax.set_xlabel('4-clique corroborated fraction')
        ax.set_ylabel('relative edge shrinkage')
        ax.set_title(f"{region.replace('_', ' ')} mesoscale phase space")
        ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
        panel_label(ax, lab)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'NWCF workload trajectory in clique-corroboration space ({exemplar})', y=1.02)
    savefig(fig, outdir, f'NWCF_MesoscalePhaseSpace_{exemplar}')

def plot_nwcf_biomarker_constellation(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    perf_rows = []
    fam_rows = []
    sig_rows = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv').set_index('Feature_set')
        if 'Selected_NWCF_Biomarker' in t17.index:
            for baseline in ['Graph_Metrics', 'Classical_PH', 'Original']:
                if baseline in t17.index:
                    perf_rows.append({'config': cfg, 'baseline': baseline, 'delta': float(t17.loc['Selected_NWCF_Biomarker', 'Balanced_accuracy'] - t17.loc[baseline, 'Balanced_accuracy'])})
            feats = str(t17.loc['Selected_NWCF_Biomarker', 'Included_features']).split('|')
            for feat in feats:
                if feat:
                    fam_rows.append({'config': cfg, 'family': feature_family(feat)})
        sig = nwcf_dir / 'Table10_BootstrapCI_Significant.csv'
        if sig.exists():
            s = pd.read_csv(sig).head(5)
            for (_, r) in s.iterrows():
                sig_rows.append({'config': cfg, 'target': str(r['Target']).replace('ADVTOPO_', '').replace('_NWCF_', ': '), 'rbc': float(r['RBC_signed']), 'lo': float(r['CI_lower_95']), 'hi': float(r['CI_upper_95'])})
    perf = pd.DataFrame(perf_rows)
    fam = pd.DataFrame(fam_rows)
    sig = pd.DataFrame(sig_rows)
    if perf.empty or fam.empty:
        return
    (fig, axes) = plt.subplots(1, 3, figsize=(13.4, 3.6), gridspec_kw={'width_ratios': [1.15, 1.05, 1.35]})
    ax = axes[0]
    piv = perf.pivot(index='config', columns='baseline', values='delta').reindex(list(dirs))
    colors = {'Graph_Metrics': '#777777', 'Classical_PH': '#bdbdbd', 'Original': '#222222'}
    bottom_pos = np.zeros(len(piv))
    bottom_neg = np.zeros(len(piv))
    for col in ['Graph_Metrics', 'Classical_PH', 'Original']:
        if col not in piv:
            continue
        vals = piv[col].fillna(0).values
        bottoms = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(np.arange(len(piv)), vals, bottom=bottoms, color=colors[col], edgecolor='black', linewidth=0.45, label=METHOD_LABELS[col])
        bottom_pos += np.where(vals >= 0, vals, 0)
        bottom_neg += np.where(vals < 0, vals, 0)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set_xticks(np.arange(len(piv)), piv.index, rotation=25, ha='right')
    ax.set_ylabel('Selected NWCF BA advantage')
    ax.set_title('Biomarker gain over baselines')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='#bdbdbd')
    panel_label(ax, 'a')
    ax = axes[1]
    fam_count = fam.groupby(['config', 'family']).size().unstack(fill_value=0).reindex(list(dirs))
    palette = {'4-clique': '#984ea3', 'clique/triangle': '#0072b2', 'edge reweighting': '#1b9e77', 'transaction': '#e69f00', 'vectorized PH': '#d95f02', 'H0 topology': '#66a61e', 'H1 topology': '#7570b3', 'other NWCF': '#999999'}
    bottom = np.zeros(len(fam_count))
    for family in [f for f in palette if f in fam_count.columns]:
        vals = fam_count[family].values
        ax.bar(np.arange(len(fam_count)), vals, bottom=bottom, color=palette[family], edgecolor='black', linewidth=0.35, label=family)
        bottom += vals
    ax.set_xticks(np.arange(len(fam_count)), fam_count.index, rotation=25, ha='right')
    ax.set_ylabel('Selected feature count')
    ax.set_title('What the NWCF biomarker is made of')
    ax.legend(fontsize=6.0, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)
    panel_label(ax, 'b')
    ax = axes[2]
    if not sig.empty:
        show = sig.iloc[:10].copy()
        show['short'] = [f'{cfg} F{i + 1}' for (i, cfg) in enumerate(show['config'].tolist())]
        y = np.arange(len(show))
        ax.errorbar(show['rbc'], y, xerr=[show['rbc'] - show['lo'], show['hi'] - show['rbc']], fmt='o', color='#1b9e77', ecolor='black', elinewidth=0.9, capsize=2.5, markersize=4)
        ax.axvline(0, color='black', linestyle=':', linewidth=0.9)
        ax.set_yticks(y, show['short'], fontsize=7.0)
        ax.invert_yaxis()
        ax.set_xlabel('signed RBC with 95% bootstrap CI')
        key_lines = []
        for (i, (_, r)) in enumerate(show.head(6).iterrows(), start=1):
            cleaned = str(r['target']).split(':')[-1].replace('nwcf_txn_', 'txn_')
            cleaned = cleaned.replace('mean_', 'm_').replace('median_', 'med_')
            key_lines.append(f'F{i}: {cleaned[:32]}')
        ax.text(1.03, 0.98, '\n'.join(key_lines), transform=ax.transAxes, ha='left', va='top', fontsize=6.2, bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
    ax.set_title('Stable NWCF task/rest effects')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='#bdbdbd')
    panel_label(ax, 'c')
    savefig(fig, outdir, 'NWCF_BiomarkerConstellation')

def circular_channel_pos(labels: list[str]) -> dict[int, tuple[float, float]]:
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    return {i: (math.cos(t), math.sin(t)) for (i, t) in enumerate(theta)}

def first_subject_pair(raw_dir: Path) -> tuple[Path, Path] | None:
    lo_files = sorted(raw_dir.glob('*_lo.json'))
    for lo in lo_files:
        hi = raw_dir / lo.name.replace('_lo.json', '_hi.json')
        if hi.exists():
            return (lo, hi)
    return None

def load_rep_distance(original_dir: Path, condition: str='hi', region: str='whole_head'):
    raw_dir = original_dir / 'per_subject_raw'
    pair = first_subject_pair(raw_dir)
    if pair is None:
        return None
    path = pair[1] if condition == 'hi' else pair[0]
    data = json.loads(path.read_text())
    fc = data['fc_regions'][region]['representative_window']
    sim = np.asarray(fc['similarity_matrix'], dtype=float)
    dist = _fc_to_base_distance(sim)
    labels = data['fc_regions'][region]['channels']
    return (dist, labels, path.stem)

def plot_filtration_mechanism(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (_, original_dir) = dirs[exemplar]
    loaded = load_rep_distance(original_dir, 'hi', 'whole_head')
    if loaded is None:
        return
    (dist, labels, subject_tag) = loaded
    (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
    tri = np.triu_indices_from(dist, k=1)
    raw = dist[tri]
    mod = d_hat[tri]
    shrink = raw - mod
    theta = float(np.percentile(raw, 18))
    (fig, axes) = plt.subplots(1, 3, figsize=(10.2, 3.25))
    ax = axes[0]
    sc = ax.scatter(raw, mod, c=shrink, s=22, cmap='viridis', edgecolor='black', linewidth=0.25)
    lim = [min(raw.min(), mod.min()) * 0.98, max(raw.max(), mod.max()) * 1.02]
    ax.plot(lim, lim, color='black', linewidth=1.0, linestyle='--')
    ax.axhline(theta, color='#d62728', linestyle=':', linewidth=1.0)
    ax.axvline(theta, color='#d62728', linestyle=':', linewidth=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Classical edge distance D')
    ax.set_ylabel('NWCF edge distance D-hat')
    ax.set_title('Edge entry-time reweighting')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label='D - D-hat')
    panel_label(ax, 'a')
    ax = axes[1]
    ax.hist(shrink, bins=18, color='#1b9e77', edgecolor='black', linewidth=0.6)
    ax.set_xlabel('Relative filtration advance')
    ax.set_ylabel('Number of edges')
    ax.set_title(f"corroborated={diag['fraction_corroborated'] * 100:.1f}% | 4-clique wins={diag.get('fraction_4clique_wins', 0) * 100:.1f}%")
    ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
    panel_label(ax, 'b')
    ax = axes[2]
    pos = circular_channel_pos(labels)
    n = len(labels)
    base_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= theta]
    nwcf_only = [(i, j) for i in range(n) for j in range(i + 1, n) if d_hat[i, j] <= theta and dist[i, j] > theta]
    nwcf_only.sort(key=lambda e: dist[e[0], e[1]] - d_hat[e[0], e[1]], reverse=True)
    draw_new = nwcf_only[:14]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=base_edges, width=1.0, edge_color='#808080', alpha=0.55)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=draw_new, width=1.7, edge_color='#d62728', alpha=0.9)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=170, node_color='white', edgecolors='black', linewidths=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={i: labels[i] for i in range(n)}, font_size=5.8)
    ax.set_title(f'Filtration edges at theta, {exemplar} {subject_tag}')
    ax.text(0.02, 0.02, f'gray: classical\nred: NWCF-only\nshown {len(draw_new)}/{len(nwcf_only)}', transform=ax.transAxes, fontsize=7, ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
    ax.set_axis_off()
    panel_label(ax, 'c')
    savefig(fig, outdir, f'FigureS3_NWCF_FiltrationMechanism_{exemplar}')

def filtration_curve(dist: np.ndarray, d_hat: np.ndarray, percentiles: np.ndarray) -> pd.DataFrame:
    raw = dist[np.triu_indices_from(dist, k=1)]
    mod = d_hat[np.triu_indices_from(d_hat, k=1)]
    rows = []
    for pct in percentiles:
        theta = float(np.percentile(raw, pct))
        classical = int((raw <= theta).sum())
        nwcf = int((mod <= theta).sum())
        rows.append({'percentile': float(pct), 'theta': theta, 'classical_edges': classical, 'nwcf_edges': nwcf, 'nwcf_only_edges': max(0, nwcf - classical), 'edge_gain_ratio': (nwcf - classical) / max(1, classical)})
    return pd.DataFrame(rows)

def plot_filtration_transfer_curves(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (_, original_dir) = dirs[exemplar]
    loaded_lo = load_rep_distance(original_dir, 'lo', 'whole_head')
    loaded_hi = load_rep_distance(original_dir, 'hi', 'whole_head')
    if loaded_lo is None or loaded_hi is None:
        return
    curves = []
    for (condition, loaded) in [('Rest', loaded_lo), ('Task', loaded_hi)]:
        (dist, _, _) = loaded
        (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
        c = filtration_curve(dist, d_hat, np.arange(5, 61, 5))
        c['condition'] = condition
        c['mean_relative_shrinkage'] = diag.get('mean_relative_shrinkage', np.nan)
        c['fraction_4clique'] = diag.get('fraction_4clique_corroborated', np.nan)
        curves.append(c)
    df = pd.concat(curves, ignore_index=True)
    (fig, axes) = plt.subplots(1, 3, figsize=(10.8, 3.35))
    for (condition, color, marker) in [('Rest', '#0072b2', 'o'), ('Task', '#d62728', '^')]:
        sub = df[df['condition'].eq(condition)]
        axes[0].plot(sub['percentile'], sub['classical_edges'], color=color, linestyle='--', marker=marker, markersize=3.5, label=f'{condition} classical')
        axes[0].plot(sub['percentile'], sub['nwcf_edges'], color=color, linestyle='-', marker=marker, markersize=3.5, label=f'{condition} NWCF')
        axes[1].plot(sub['percentile'], sub['nwcf_only_edges'], color=color, marker=marker, markersize=3.5, label=condition)
        axes[2].plot(sub['percentile'], sub['edge_gain_ratio'], color=color, marker=marker, markersize=3.5, label=condition)
    titles = ['Filtration edge growth', 'NWCF-only mesoscale edges', 'Clique-corroboration gain']
    ylabels = ['Number of edges present', 'Edges entering only under NWCF', 'Gain over classical edge count']
    for (ax, title, ylabel, lab) in zip(axes, titles, ylabels, ['a', 'b', 'c']):
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
        ax.legend(fontsize=7, loc='best')
        panel_label(ax, lab)
    fig.supxlabel('Classical distance percentile threshold', y=-0.02)
    fig.suptitle(f'NWCF changes the filtration trajectory on STEW FC ({exemplar})', y=1.04)
    savefig(fig, outdir, f'Mechanism_FiltrationTransferCurves_{exemplar}')

def plot_transaction_fingerprint(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (_, original_dir) = dirs[exemplar]
    lo = condition_trace(original_dir, 'lo')
    hi = condition_trace(original_dir, 'hi')
    if lo is None or hi is None:
        return
    metrics = ['fraction_4clique_corroborated', 'mean_4clique_support', 'mean_relative_shrinkage', 'max_persistence_H0']
    (fig, axes) = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for (ax, frame, title, lab) in [(axes[0], lo, 'Rest', 'a'), (axes[1], hi, 'Task', 'b')]:
        mat = frame[metrics].T.astype(float)
        mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
        im = ax.imshow(mat.values, aspect='auto', cmap='coolwarm', vmin=-2.2, vmax=2.2, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Transaction window')
        ax.set_yticks(np.arange(len(metrics)), [m.replace('_', ' ') for m in metrics], fontsize=7)
        ax.set_xticks(np.linspace(0, mat.shape[1] - 1, 6).astype(int))
        panel_label(ax, lab)
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label='within-metric z score')
    fig.suptitle(f'NWCF transaction fingerprint: clique support and topology co-evolve ({exemplar})', y=1.02)
    savefig(fig, outdir, f'Mechanism_TransactionFingerprint_{exemplar}')

def condition_trace(original_dir: Path, condition: str, region: str='whole_head') -> pd.DataFrame | None:
    raw_dir = original_dir / 'per_subject_raw'
    pair = first_subject_pair(raw_dir)
    if pair is None:
        return None
    path = pair[1] if condition == 'hi' else pair[0]
    data = json.loads(path.read_text())
    region_data = data['fc_regions'][region]
    rows = []
    for win in region_data.get('per_window', []):
        sim = np.asarray(win['similarity_matrix'], dtype=float)
        dist = _fc_to_base_distance(sim)
        feats = nwcf_window_features(dist, max_clique_order=4)
        rows.append({'window_time_sec': float(win.get('window_time_sec', len(rows))), 'condition': condition, 'fraction_4clique_corroborated': feats.get('nwcf_fraction_4clique_corroborated', np.nan), 'mean_4clique_support': feats.get('nwcf_mean_4clique_support', np.nan), 'mean_relative_shrinkage': feats.get('nwcf_mean_relative_shrinkage', np.nan), 'max_persistence_H0': feats.get('nwcf_max_persistence_H0', np.nan)})
    if not rows:
        return None
    return pd.DataFrame(rows)

def plot_transaction_traces(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (_, original_dir) = dirs[exemplar]
    lo = condition_trace(original_dir, 'lo')
    hi = condition_trace(original_dir, 'hi')
    if lo is None or hi is None:
        return
    df = pd.concat([lo, hi], ignore_index=True)
    metrics = [('fraction_4clique_corroborated', 'Fraction 4-clique corroborated'), ('mean_4clique_support', 'Mean 4-clique support'), ('mean_relative_shrinkage', 'Mean relative shrinkage'), ('max_persistence_H0', 'NWCF H0 max persistence')]
    (fig, axes) = plt.subplots(2, 2, figsize=(8.4, 5.15), sharex=True)
    for (ax, (metric, title), lab) in zip(axes.ravel(), metrics, ['a', 'b', 'c', 'd']):
        for (cond, color, marker, label) in [('lo', '#0072b2', 'o', 'Rest'), ('hi', '#d62728', '^', 'Task')]:
            sub = df[df['condition'].eq(cond)]
            ax.plot(sub['window_time_sec'], sub[metric], color=color, linewidth=1.1, marker=marker, markersize=2.7, markevery=max(1, len(sub) // 12), label=label)
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#bdbdbd')
        ax.set_xlabel('Window time (s)')
        panel_label(ax, lab)
    axes[0, 0].legend(loc='best', fontsize=8)
    fig.suptitle(f'Transaction-level NWCF dynamics in STEW ({exemplar}, exemplar subject)', y=1.02)
    savefig(fig, outdir, f'FigureS4_NWCF_TransactionTraces_{exemplar}')

def compact_feature_name(name: str, max_len: int=32) -> str:
    s = str(name)
    s = s.replace('ADVTOPO_', '').replace('_NWCF_', ': ')
    s = s.replace('nwcf_txn_', 'txn_').replace('nwcf_', '')
    s = s.replace('fraction_', 'frac_').replace('corroborated', 'corrob')
    s = s.replace('mean_', 'm_').replace('median_', 'med_')
    s = s.replace('persistence', 'persist').replace('triangle', 'tri')
    return s[:max_len]

def selected_features_for_config(nwcf_dir: Path) -> list[str]:
    t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv')
    row = t17[t17['Feature_set'].eq('Selected_NWCF_Biomarker')]
    if row.empty:
        return []
    return [f for f in str(row.iloc[0]['Included_features']).split('|') if f]

def family_palette() -> dict[str, str]:
    return {'4-clique': '#8f4ea3', 'clique/triangle': '#0072b2', 'edge reweighting': '#2aa198', 'transaction': '#e6a500', 'vectorized PH': '#d95f02', 'H0 topology': '#7a9a01', 'H1 topology': '#6a69a6', 'other NWCF': '#8a8a8a'}

def nwcf01_biomarker_constellation(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    rows = []
    fam_rows = []
    sig_counts = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv').set_index('Feature_set')
        if 'Selected_NWCF_Biomarker' not in t17.index:
            continue
        sel = t17.loc['Selected_NWCF_Biomarker']
        for method in ['Classical_PH', 'Graph_Metrics', 'Original']:
            if method in t17.index:
                rows.append({'config': cfg, 'baseline': method, 'gain': float(sel['Balanced_accuracy'] - t17.loc[method, 'Balanced_accuracy'])})
        for feat in selected_features_for_config(nwcf_dir):
            fam_rows.append({'config': cfg, 'family': feature_family(feat)})
        sig_path = nwcf_dir / 'Table10_BootstrapCI_Significant.csv'
        sig_counts.append({'config': cfg, 'stable': len(pd.read_csv(sig_path)) if sig_path.exists() else 0})
    gain = pd.DataFrame(rows)
    fam = pd.DataFrame(fam_rows)
    sig = pd.DataFrame(sig_counts).set_index('config').reindex(list(dirs))
    if gain.empty or fam.empty:
        return
    fig = plt.figure(figsize=(8.9, 5.6))
    header(fig, 'NWCF biomarker constellation', 'A compact clique-corroborated panel is compared against graph metrics, classical PH, and the full original feature universe.')
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.95, top=0.84, bottom=0.11, hspace=0.42, wspace=0.38, width_ratios=[1.25, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    ax3 = fig.add_subplot(gs[1, 1:])
    piv = gain.pivot(index='config', columns='baseline', values='gain').reindex(list(dirs))
    colors = {'Classical_PH': '#c8c8c8', 'Graph_Metrics': '#808080', 'Original': '#222222'}
    bottom_pos = np.zeros(len(piv))
    bottom_neg = np.zeros(len(piv))
    for col in ['Classical_PH', 'Graph_Metrics', 'Original']:
        vals = piv.get(col, pd.Series(0, index=piv.index)).fillna(0).values
        base = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax1.bar(np.arange(len(piv)), vals, bottom=base, color=colors[col], edgecolor='white', lw=0.7, label=METHOD_LABELS[col])
        bottom_pos += np.where(vals >= 0, vals, 0)
        bottom_neg += np.where(vals < 0, vals, 0)
    ax1.axhline(0, color='black', lw=0.9)
    ax1.set_xticks(np.arange(len(piv)), piv.index, rotation=28, ha='right')
    ax1.set_ylabel('Selected NWCF BA advantage')
    ax1.set_title('compact biomarker gain')
    ax1.grid(True, axis='y', color='#eeeeee', lw=0.45)
    ax1.legend(fontsize=6.8, loc='upper left')
    ax1.text(0.05, 0.92, 'positive bars mean\nNWCF beats baseline', transform=ax1.transAxes, fontsize=7.0, color='#486471', va='top')
    panel_label(ax1, 'a')
    pal = family_palette()
    fam_count = fam.groupby(['config', 'family']).size().unstack(fill_value=0).reindex(list(dirs))
    bottom = np.zeros(len(fam_count))
    for (family, color) in pal.items():
        if family in fam_count:
            vals = fam_count[family].values
            ax2.bar(np.arange(len(fam_count)), vals, bottom=bottom, color=color, edgecolor='white', lw=0.55, label=family)
            bottom += vals
    ax2.set_xticks(np.arange(len(fam_count)), fam_count.index)
    ax2.set_ylabel('selected features')
    ax2.set_title('what the selected panel contains')
    ax2.legend(fontsize=6.1, ncol=2, loc='upper right', frameon=False)
    panel_label(ax2, 'b')
    ax3.plot(sig.index, sig['stable'], color='#2ab7c9', marker='o', lw=2.0)
    ax3.fill_between(np.arange(len(sig)), sig['stable'].values, color='#2ab7c9', alpha=0.15)
    ax3.set_ylabel('stable FDR effects')
    ax3.set_title('statistical support is not isolated to one run')
    ax3.grid(True, axis='y', color='#eeeeee', lw=0.45)
    ax3.set_xticks(np.arange(len(sig)), sig.index)
    ax3.annotate('stable NWCF effects\nsurvive bootstrap/FDR', xy=(3, sig['stable'].iloc[3]), xytext=(1.3, sig['stable'].max() * 0.83), arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=7.2)
    panel_label(ax3, 'c')
    savefig(fig, outdir, 'NWCF_01_BiomarkerConstellation')

def nwcf02_mesoscale_phase_space(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (nwcf_dir, _) = dirs[exemplar]
    df = pd.read_csv(nwcf_dir / 'advanced_topology_features.csv')
    regions = ['frontal', 'whole_head', 'posterior']
    fig = plt.figure(figsize=(8.9, 5.5))
    header(fig, 'NWCF exposes a mesoscale workload phase shift', 'Rest-to-task paired arrows move in clique corroboration and edge-shrinkage space.')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.95, top=0.84, bottom=0.11, hspace=0.42, wspace=0.32)
    for (idx, region) in enumerate(regions):
        ax = fig.add_subplot(gs[0, idx])
        xcol = f'ADVTOPO_{region}_NWCF_nwcf_txn_mean_fraction_4clique_corroborated'
        ycol = f'ADVTOPO_{region}_NWCF_nwcf_txn_mean_mean_relative_shrinkage'
        if xcol not in df.columns or ycol not in df.columns:
            ax.axis('off')
            continue
        lo = df[df['condition'].eq('lo')][['subject_id', xcol, ycol]].rename(columns={xcol: 'x_lo', ycol: 'y_lo'})
        hi = df[df['condition'].eq('hi')][['subject_id', xcol, ycol]].rename(columns={xcol: 'x_hi', ycol: 'y_hi'})
        p = lo.merge(hi, on='subject_id').dropna()
        for (_, r) in p.iterrows():
            ax.annotate('', xy=(r['x_hi'], r['y_hi']), xytext=(r['x_lo'], r['y_lo']), arrowprops=dict(arrowstyle='->', color='#c7c7c7', lw=0.65, alpha=0.75))
        ax.scatter(p['x_lo'], p['y_lo'], s=22, color='#0072b2', edgecolor='white', lw=0.5, label='Rest')
        ax.scatter(p['x_hi'], p['y_hi'], s=28, color='#d64b3a', marker='^', edgecolor='white', lw=0.5, label='Task')
        ax.set_title(region.replace('_', ' '))
        ax.set_xlabel('4-clique corroboration')
        ax.set_ylabel('relative shrinkage')
        ax.grid(True, color='#eeeeee', lw=0.45)
        panel_label(ax, chr(ord('a') + idx))
    ax = fig.add_subplot(gs[1, :])
    clean_axis_arrows(ax, 'workload transition', 'mesoscale organization')
    x = np.array([0.08, 0.34, 0.62, 0.88])
    y = np.array([0.18, 0.36, 0.62, 0.84])
    ax.plot(x, y, color='#2ab7c9', lw=2.0)
    ax.scatter(x, y, s=46, c=['#0072b2', '#2aa198', '#d64b3a', '#8a9500'], edgecolor='white', zorder=3)
    ax.text(0.08, 0.07, 'pairwise FC\nnetwork', transform=ax.transAxes, ha='center', fontsize=7.2, color='#0072b2')
    ax.text(0.34, 0.43, 'supported\ntriangles', transform=ax.transAxes, ha='center', fontsize=7.2, color='#2aa198')
    ax.text(0.62, 0.7, '4-clique\ncorroboration', transform=ax.transAxes, ha='center', fontsize=7.2, color='#d64b3a')
    ax.text(0.86, 0.58, 'workload\nmesoscale state', transform=ax.transAxes, ha='center', fontsize=7.2, color='#8a9500')
    phase_bar(ax, 0.08, 0.32, 0.98, 'edge evidence', '#0072b2')
    phase_bar(ax, 0.38, 0.65, 0.98, 'clique support', '#2aa198')
    phase_bar(ax, 0.7, 0.92, 0.98, 'biomarker state', '#8a9500')
    panel_label(ax, 'd')
    savefig(fig, outdir, f'NWCF_02_MesoscalePhaseSpace_{exemplar}')

def nwcf03_feature_selection_small_multiples(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (nwcf_dir, _) = dirs[exemplar]
    df = pd.read_csv(nwcf_dir / 'advanced_topology_features.csv')
    feats = selected_features_for_config(nwcf_dir)[:8]
    feats = [f for f in feats if f in df.columns]
    if not feats:
        return
    n = min(8, len(feats))
    fig = plt.figure(figsize=(8.7, 6.2))
    header(fig, 'Selected NWCF biomarkers are compact but interpretable', 'Each panel shows the paired rest/task distribution for one selected clique-corroborated feature.')
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.96, top=0.84, bottom=0.12, hspace=0.48, wspace=0.34)
    for (i, feat) in enumerate(feats[:n]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        p = paired_values(df, feat)
        if p.empty:
            ax.axis('off')
            continue
        for (_, r) in p.iterrows():
            ax.plot([0, 1], [r['lo'], r['hi']], color='#d8d8d8', lw=0.5, zorder=1)
        med = p[['lo', 'hi']].median()
        ax.plot([0, 1], med, color='#2ab7c9', marker='o', lw=1.8, zorder=3)
        ax.scatter(np.zeros(len(p)) - 0.035, p['lo'], s=10, color='#0072b2', alpha=0.65, edgecolor='none')
        ax.scatter(np.ones(len(p)) + 0.035, p['hi'], s=12, marker='^', color='#d64b3a', alpha=0.65, edgecolor='none')
        ax.set_xticks([0, 1], ['Rest', 'Task'])
        ax.set_title(f'F{i + 1}: {compact_feature_name(feat, 24)}', fontsize=7.4)
        ax.grid(True, axis='y', color='#eeeeee', lw=0.45)
        ax.text(0.05, 0.92, f"dz={dz_effect(p['delta']):.2f}", transform=ax.transAxes, fontsize=6.8, va='top', color='#486471')
        if i % 4 == 0:
            ax.set_ylabel('feature value')
        panel_label(ax, chr(ord('a') + i))
    savefig(fig, outdir, f'NWCF_03_SelectedFeatureSmallMultiples_{exemplar}')

def nwcf04_mechanistic_baseline_comparison(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    if exemplar not in dirs:
        exemplar = next(iter(dirs))
    (nwcf_dir, original_dir) = dirs[exemplar]
    loaded = load_rep_distance(original_dir, 'hi', 'whole_head')
    if loaded is None:
        return
    (dist, labels, tag) = loaded
    (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
    raw = dist[np.triu_indices_from(dist, k=1)]
    mod = d_hat[np.triu_indices_from(d_hat, k=1)]
    t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv').set_index('Feature_set')
    fig = plt.figure(figsize=(8.9, 5.8))
    header(fig, 'What NWCF sees beyond classical PH and graph metrics', 'The method changes the filtration object: edges enter earlier only when supported by local clique evidence.')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.96, top=0.84, bottom=0.11, hspace=0.46, wspace=0.38)
    (ax1, ax2, ax3) = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax4 = fig.add_subplot(gs[1, :])
    ax1.hist(raw, bins=18, color='#cfcfcf', edgecolor='white')
    ax1.set_title('Classical PH')
    ax1.set_xlabel('pairwise edge distance')
    ax1.set_ylabel('edges')
    ax1.text(0.06, 0.88, 'filtration driven by\nindividual edge scales', transform=ax1.transAxes, fontsize=7.1, color='#555555')
    panel_label(ax1, 'a')
    graph_vals = []
    for name in ['Graph_Metrics', 'Classical_PH', 'Selected_NWCF_Biomarker']:
        if name in t17.index:
            graph_vals.append((METHOD_LABELS.get(name, name), float(t17.loc[name, 'Balanced_accuracy']), METHOD_COLORS.get(name, '#777777')))
    ax2.barh(np.arange(len(graph_vals)), [v[1] for v in graph_vals], color=[v[2] for v in graph_vals], edgecolor='white')
    ax2.set_yticks(np.arange(len(graph_vals)), [v[0] for v in graph_vals], fontsize=6.8)
    ax2.set_xlim(0.35, 0.85)
    ax2.axvline(0.5, color='black', ls=':', lw=0.8)
    ax2.set_xlabel('balanced accuracy')
    ax2.set_title('Baseline view')
    ax2.grid(True, axis='x', color='#eeeeee', lw=0.45)
    panel_label(ax2, 'b')
    shrink = raw - mod
    sc = ax3.scatter(raw, mod, c=shrink, cmap='viridis', s=20, edgecolor='white', lw=0.25)
    lim = [min(raw.min(), mod.min()) * 0.98, max(raw.max(), mod.max()) * 1.02]
    ax3.plot(lim, lim, color='black', ls='--', lw=0.8)
    ax3.set_xlim(lim)
    ax3.set_ylim(lim)
    ax3.set_xlabel('D')
    ax3.set_ylabel('D-hat')
    ax3.set_title('NWCF edge-entry shift')
    fig.colorbar(sc, ax=ax3, fraction=0.045, pad=0.02, label='advance')
    panel_label(ax3, 'c')
    clean_axis_arrows(ax4, 'representation', 'mesoscale evidence')
    x = np.array([0.1, 0.38, 0.66, 0.9])
    y = np.array([0.2, 0.35, 0.63, 0.84])
    ax4.plot(x, y, color='#2ab7c9', lw=2.2)
    ax4.scatter(x, y, s=48, c=['#c8c8c8', '#808080', '#2aa198', '#d64b3a'], edgecolor='white', zorder=4)
    ax4.text(0.1, 0.08, 'Classical PH\ncomponents/holes', transform=ax4.transAxes, ha='center', fontsize=7.1)
    ax4.text(0.38, 0.43, 'Graph metrics\nscalar summaries', transform=ax4.transAxes, ha='center', fontsize=7.1)
    ax4.text(0.66, 0.7, 'NWCF\nclique support', transform=ax4.transAxes, ha='center', fontsize=7.1)
    ax4.text(0.88, 0.55, 'custom filtration\nbiomarker', transform=ax4.transAxes, ha='center', fontsize=7.1, color='#9f2f24')
    ax4.text(0.08, 0.88, f"fraction corroborated={diag['fraction_corroborated'] * 100:.1f}%\n4-clique wins={diag.get('fraction_4clique_wins', 0) * 100:.1f}%", transform=ax4.transAxes, fontsize=7.0, color='#486471')
    panel_label(ax4, 'd')
    savefig(fig, outdir, f'NWCF_04_MechanisticBaselineComparison_{exemplar}')

def nwcf05_novelty_nonredundancy(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    rows = []
    scatter = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        path = nwcf_dir / 'Table11_NWCF_NoveltyComprehensive.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for (cls, count) in df['redundancy_class'].value_counts().items():
            rows.append({'config': cfg, 'class': cls.split()[0], 'count': int(count)})
        for (_, r) in df.sample(min(160, len(df)), random_state=7).iterrows():
            scatter.append({'config': cfg, 'graph_r': float(r.get('max_r_vs_graph', np.nan)), 'ph_r': float(r.get('max_r_vs_classical_PH', np.nan)), 'uis': float(r.get('unique_info_score', np.nan))})
    counts = pd.DataFrame(rows)
    points = pd.DataFrame(scatter).dropna()
    if counts.empty:
        return
    fig = plt.figure(figsize=(8.8, 5.4))
    header(fig, 'NWCF carries non-redundant information', "Novelty is assessed against classical PH and graph metrics, directly addressing the 'is this already done?' question.")
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, top=0.84, bottom=0.11, width_ratios=[1.05, 1.0], hspace=0.4, wspace=0.34)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    piv = counts.pivot(index='config', columns='class', values='count').fillna(0).reindex(list(dirs))
    cls_order = [c for c in ['Novel', 'Partial', 'Redundant'] if c in piv.columns]
    cls_colors = {'Novel': '#2aa198', 'Partial': '#e6a500', 'Redundant': '#bdbdbd'}
    bottom = np.zeros(len(piv))
    for cls in cls_order:
        vals = piv[cls].values
        ax1.bar(np.arange(len(piv)), vals, bottom=bottom, color=cls_colors[cls], edgecolor='white', label=cls)
        bottom += vals
    ax1.set_xticks(np.arange(len(piv)), piv.index, rotation=28, ha='right')
    ax1.set_ylabel('NWCF features')
    ax1.set_title('Most NWCF descriptors are not graph/PH duplicates')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, axis='y', color='#eeeeee', lw=0.45)
    panel_label(ax1, 'a')
    if not points.empty:
        ax2.scatter(points['graph_r'], points['uis'], s=12, color='#0072b2', alpha=0.35, edgecolor='none')
        ax2.set_xlabel('max |r| vs graph metrics')
        ax2.set_ylabel('unique information score')
        ax2.set_title('relation to graph metrics')
        ax2.grid(True, color='#eeeeee', lw=0.45)
        ax2.axhline(0.7, color='#2aa198', ls=':', lw=1.0)
        panel_label(ax2, 'b')
        ax3.scatter(points['ph_r'], points['uis'], s=12, color='#8f4ea3', alpha=0.35, edgecolor='none')
        ax3.set_xlabel('max |r| vs classical PH')
        ax3.set_ylabel('unique information score')
        ax3.set_title('relation to classical PH')
        ax3.grid(True, color='#eeeeee', lw=0.45)
        ax3.axhline(0.7, color='#2aa198', ls=':', lw=1.0)
        ax3.text(0.04, 0.83, 'above line:\nnovel information', transform=ax3.transAxes, fontsize=7.0, color='#2aa198')
        panel_label(ax3, 'c')
    savefig(fig, outdir, 'NWCF_05_NoveltyNonRedundancy')

def nwcf06_cross_config_stability(dirs: dict[str, tuple[Path, Path]], outdir: Path) -> None:
    rows = []
    perf = []
    for (cfg, (nwcf_dir, _)) in dirs.items():
        sig_path = nwcf_dir / 'Table10_BootstrapCI_Significant.csv'
        if sig_path.exists():
            df = pd.read_csv(sig_path)
            for (_, r) in df.iterrows():
                rows.append({'config': cfg, 'family': feature_family(str(r['Target'])), 'effect': abs(float(r['RBC_signed']))})
        t17 = pd.read_csv(nwcf_dir / 'Table17_NWCF_BiomarkerPanels.csv').set_index('Feature_set')
        for method in ['Selected_NWCF_Biomarker', 'Graph_Metrics', 'Classical_PH']:
            if method in t17.index:
                perf.append({'config': cfg, 'method': method, 'ba': float(t17.loc[method, 'Balanced_accuracy'])})
    fam = pd.DataFrame(rows)
    perf_df = pd.DataFrame(perf)
    if fam.empty:
        return
    families = ['4-clique', 'clique/triangle', 'edge reweighting', 'transaction', 'vectorized PH', 'H0 topology', 'H1 topology', 'other NWCF']
    count = fam.groupby(['family', 'config']).size().unstack(fill_value=0).reindex(families).dropna(how='all')
    eff = fam.groupby(['family', 'config'])['effect'].median().unstack().reindex(count.index).reindex(list(dirs), axis=1)
    fig = plt.figure(figsize=(8.8, 5.6))
    header(fig, 'NWCF effects are stable across parameterizations', 'The strongest evidence is recurrence of clique/transaction families, not a single lucky configuration.')
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, top=0.84, bottom=0.11, hspace=0.4, wspace=0.34)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax1.imshow(count.values, cmap='YlGnBu', aspect='auto')
    ax1.set_xticks(np.arange(len(count.columns)), count.columns, rotation=28, ha='right')
    ax1.set_yticks(np.arange(len(count.index)), count.index, fontsize=7.0)
    ax1.set_title('stable significant NWCF families')
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            val = int(count.values[i, j])
            if val:
                ax1.text(j, i, str(val), ha='center', va='center', fontsize=6.8, color='black')
    fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02, label='stable features')
    panel_label(ax1, 'a')
    for (method, color, marker) in [('Selected_NWCF_Biomarker', '#2aa198', 'o'), ('Graph_Metrics', '#777777', 's'), ('Classical_PH', '#bdbdbd', '^')]:
        sub = perf_df[perf_df['method'].eq(method)].set_index('config').reindex(list(dirs))
        ax2.plot(sub.index, sub['ba'], color=color, marker=marker, lw=1.8, label=METHOD_LABELS[method])
    ax2.axhline(0.5, color='black', ls=':', lw=0.8)
    ax2.set_ylim(0.35, 0.85)
    ax2.set_ylabel('balanced accuracy')
    ax2.set_title('baseline separation')
    ax2.tick_params(axis='x', rotation=28)
    ax2.grid(True, axis='y', color='#eeeeee', lw=0.45)
    ax2.legend(fontsize=6.6, loc='lower left')
    panel_label(ax2, 'b')
    med_eff = eff.median(axis=1).sort_values(ascending=True).tail(6)
    ax3.barh(np.arange(len(med_eff)), med_eff.values, color='#2ab7c9', edgecolor='white')
    ax3.set_yticks(np.arange(len(med_eff)), med_eff.index, fontsize=7.0)
    ax3.set_xlabel('median |RBC|')
    ax3.set_title('recurring effect strength')
    ax3.grid(True, axis='x', color='#eeeeee', lw=0.45)
    ax3.text(0.04, 0.92, 'recurrent families support\na filtration-level claim', transform=ax3.transAxes, fontsize=7.0, color='#486471', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    panel_label(ax3, 'c')
    savefig(fig, outdir, 'NWCF_06_CrossConfigStability')

def write_summary(dirs: dict[str, tuple[Path, Path]], outdir: Path, exemplar: str) -> None:
    df = load_table17(dirs)
    if df.empty:
        return
    lines = ['STEW publication scientific figures', '===================================', '', f"Configs included: {', '.join(dirs)}", f'Exemplar config for filtration/transaction figures: {(exemplar if exemplar in dirs else next(iter(dirs)))}', '', 'Key Table17 balanced-accuracy scoreboard:']
    for cfg in dirs:
        sub = df[df['config'].eq(cfg)].set_index('Feature_set')
        vals = []
        for method in ['Original', 'Original+Selected_NWCF', 'Selected_NWCF_Biomarker', 'Graph_Metrics', 'Classical_PH']:
            if method in sub.index:
                vals.append(f"{METHOD_LABELS[method]}={sub.loc[method, 'Balanced_accuracy']:.3f}")
        lines.append(f'  {cfg}: ' + '; '.join(vals))
    lines += ['', 'Figures written:', '  NWCF_01_BiomarkerConstellation.png', '  NWCF_02_MesoscalePhaseSpace_<exemplar>.png', '  NWCF_03_SelectedFeatureSmallMultiples_<exemplar>.png', '  NWCF_04_MechanisticBaselineComparison_<exemplar>.png', '  NWCF_05_NoveltyNonRedundancy.png', '  NWCF_06_CrossConfigStability.png', '  OriginalTopology_ScalpAndPairedShift_<exemplar>.(png|pdf)', '  OriginalToNWCF_Bridge_Compactness.(png|pdf)', '  FigureS1_PerformanceLandscape.(png|pdf)', '  FigureS2_MesoscaleBiomarkerMap.(png|pdf)', '  NWCF_BiomarkerConstellation.(png|pdf)', '  NWCF_MesoscalePhaseSpace_<exemplar>.(png|pdf)', '  FigureS3_NWCF_FiltrationMechanism_<exemplar>.(png|pdf)', '  Mechanism_FiltrationTransferCurves_<exemplar>.(png|pdf)', '  Mechanism_TransactionFingerprint_<exemplar>.(png|pdf)', '  FigureS4_NWCF_TransactionTraces_<exemplar>.(png|pdf)']
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'publication_figure_summary.txt').write_text('\n'.join(lines) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='*', default=DEFAULT_CONFIGS)
    parser.add_argument('--output-dir', default='stew_publication_figures')
    parser.add_argument('--exemplar', default='84corr', help='Config used for detailed mechanism/trace figures.')
    args = parser.parse_args()
    set_scientific_style()
    dirs = output_dirs(args.configs)
    if not dirs:
        raise SystemExit('No finished config folders found.')
    outdir = PROJECT_ROOT / args.output_dir
    plot_original_topology_evidence(dirs, outdir, args.exemplar)
    plot_original_to_nwcf_bridge(dirs, outdir)
    plot_performance_landscape(dirs, outdir)
    plot_mesoscale_biomarker_map(dirs, outdir)
    plot_nwcf_biomarker_constellation(dirs, outdir)
    plot_nwcf_phase_space(dirs, outdir, args.exemplar)
    nwcf01_biomarker_constellation(dirs, outdir)
    nwcf02_mesoscale_phase_space(dirs, outdir, args.exemplar)
    nwcf03_feature_selection_small_multiples(dirs, outdir, args.exemplar)
    nwcf04_mechanistic_baseline_comparison(dirs, outdir, args.exemplar)
    nwcf05_novelty_nonredundancy(dirs, outdir)
    nwcf06_cross_config_stability(dirs, outdir)
    plot_filtration_mechanism(dirs, outdir, args.exemplar)
    plot_filtration_transfer_curves(dirs, outdir, args.exemplar)
    plot_transaction_fingerprint(dirs, outdir, args.exemplar)
    plot_transaction_traces(dirs, outdir, args.exemplar)
    write_summary(dirs, outdir, args.exemplar)
    print(f'Wrote publication figures to {outdir}')
if __name__ == '__main__':
    main()
