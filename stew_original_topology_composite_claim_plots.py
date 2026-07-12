from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = ['123plv', '254aec', '83aec', '84aec', '84corr', '84plv']
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SCALP_POS = {'AF3': (-0.38, 0.92), 'F7': (-0.82, 0.58), 'F3': (-0.42, 0.55), 'FC5': (-0.62, 0.2), 'T7': (-0.95, 0.0), 'P7': (-0.72, -0.48), 'O1': (-0.3, -0.86), 'O2': (0.3, -0.86), 'P8': (0.72, -0.48), 'T8': (0.95, 0.0), 'FC6': (0.62, 0.2), 'F4': (0.42, 0.55), 'F8': (0.82, 0.58), 'AF4': (0.38, 0.92)}
REGION_ORDER = ['confirmatory_frontal_roi', 'frontal', 'temporal', 'posterior', 'whole_head']
REGION_LABELS = {'confirmatory_frontal_roi': 'frontolateral ROI', 'frontal': 'frontal', 'temporal': 'temporal', 'posterior': 'posterior', 'whole_head': 'whole head'}
COLORS = {'blue': '#0072b2', 'cyan': '#2ab7c9', 'teal': '#2aa198', 'red': '#d64b3a', 'purple': '#8f4ea3', 'green': '#7a9a01', 'gray': '#666666', 'gold': '#e69f00'}

def set_style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.0, 'axes.titlesize': 8.8, 'axes.labelsize': 8.0, 'axes.linewidth': 0.9, 'figure.dpi': 160, 'savefig.dpi': 450, 'savefig.bbox': 'tight', 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

def out_dir(config: str) -> Path:
    return PROJECT_ROOT / f'stew_output_final_{config}'

def tda_dir(config: str) -> Path:
    return PROJECT_ROOT / f'stew_tda_results_final_{config}'

def table(config: str, name: str) -> pd.DataFrame:
    path = out_dir(config) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def features(config: str) -> pd.DataFrame:
    path = tda_dir(config) / 'subject_level_features.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def panel(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, fontweight='bold')

def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.965, title, ha='left', va='top', fontsize=11.2, fontweight='bold')
    fig.text(0.055, 0.925, subtitle, ha='left', va='top', fontsize=7.8, color='#486471')

def save(fig, config: str, stem: str) -> None:
    path = out_dir(config)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'{stem}.png')
    plt.close(fig)

def draw_scalp(ax, values: dict[str, float], title: str, cmap: str='coolwarm', vlim: float | None=None):
    xy = np.array([SCALP_POS[ch] for ch in CHANNELS])
    z = np.array([values.get(ch, np.nan) for ch in CHANNELS], dtype=float)
    if vlim is None:
        vmax = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
        vlim = max(0.1, float(vmax))
    ax.add_patch(Circle((0, 0), 1.04, fill=False, color='black', lw=1.1))
    ax.plot([0, -0.08, 0.08, 0], [1.04, 1.16, 1.04, 1.04], color='black', lw=1.0)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap=cmap, vmin=-vlim, vmax=vlim, s=150, edgecolor='black', lw=0.55)
    for (ch, (x, y)) in SCALP_POS.items():
        ax.text(x, y - 0.14, ch, ha='center', va='center', fontsize=6.1)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.08, 1.22)
    ax.axis('off')
    return sc

def paired(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = df.pivot_table(index='subject_id', columns='condition', values=col, aggfunc='mean')
    if 'lo' not in p.columns or 'hi' not in p.columns:
        return pd.DataFrame(columns=['subject_id', 'lo', 'hi', 'delta'])
    p = p[['lo', 'hi']].dropna()
    p['delta'] = p['hi'] - p['lo']
    return p.reset_index()

def draw_story_axis(ax, labels: list[str], colors: list[str], title: str) -> None:
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    xs = np.linspace(0.08, 0.92, len(labels))
    y = np.linspace(0.18, 0.78, len(labels))
    ax.plot(xs, y, color=COLORS['cyan'], lw=2.0)
    for (x, yy, text, color) in zip(xs, y, labels, colors):
        ax.scatter([x], [yy], s=55, color=color, edgecolor='white', zorder=3)
        ax.text(x, yy + 0.1, text, ha='center', va='bottom', fontsize=7.1, color=color)
    ax.annotate('', xy=(0.98, 0.03), xytext=(0.02, 0.03), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.0))

def fig1_workload_structure(config: str) -> None:
    df = features(config)
    t2 = table(config, 'Table2_WholeHead_Topology.csv')
    t3 = table(config, 'Table3_RegionalSummaries_Topology.csv')
    if t2.empty:
        return
    vals = dict(zip(t2['Target'], t2['RBC_signed']))
    top = t2.assign(abs_rbc=t2['RBC_signed'].abs()).sort_values('abs_rbc', ascending=False).head(1)
    top_ch = str(top['Target'].iloc[0])
    pv = paired(df, f'{top_ch}_median_max_persistence_H1')
    fig = plt.figure(figsize=(9.0, 6.0))
    header(fig, f'{config}: original topology detects workload structure', 'Channel, paired, and regional views point to task-related topological compression.')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.96, top=0.84, bottom=0.1, hspace=0.45, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    sc = draw_scalp(ax0, vals, 'channel-wise effect size', vlim=max(1.0, np.nanmax(np.abs(list(vals.values())))))
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.035, pad=0.015)
    cbar.ax.set_title('RBC', fontsize=6.5, pad=2)
    panel(ax0, 'a')
    ax1 = fig.add_subplot(gs[0, 1])
    if not pv.empty:
        for (_, r) in pv.iterrows():
            ax1.plot([0, 1], [r['lo'], r['hi']], color='#9e9e9e', lw=0.7, alpha=0.45)
        ax1.scatter(np.zeros(len(pv)), pv['lo'], color=COLORS['blue'], s=18, alpha=0.75, label='Rest')
        ax1.scatter(np.ones(len(pv)), pv['hi'], color=COLORS['red'], s=18, alpha=0.75, label='Task')
    ax1.set_xticks([0, 1], ['Rest', 'Task'])
    ax1.set_title(f'paired shift: {top_ch}')
    ax1.set_ylabel('H1 persistence')
    ax1.grid(True, axis='y', color='#eeeeee', lw=0.5)
    ax1.legend(fontsize=6.8)
    panel(ax1, 'b')
    ax2 = fig.add_subplot(gs[0, 2])
    reg = t3.set_index('Target').reindex(REGION_ORDER).dropna(subset=['RBC_signed'], how='all')
    y = np.arange(len(reg))
    ax2.barh(y, reg['RBC_signed'], color=[COLORS['red'] if v > 0 else COLORS['blue'] for v in reg['RBC_signed']], alpha=0.82)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_yticks(y, [REGION_LABELS.get(i, i) for i in reg.index], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel('signed RBC')
    ax2.set_title('regional concentration')
    ax2.grid(True, axis='x', color='#eeeeee', lw=0.5)
    panel(ax2, 'c')
    ax3 = fig.add_subplot(gs[1, :])
    draw_story_axis(ax3, ['rest topology\nmore expanded', 'task shift\ncompresses H1', 'frontolateral\nconcentration', 'candidate\nstate marker'], [COLORS['blue'], COLORS['red'], COLORS['purple'], COLORS['green']], 'claim path: workload structure appears before NWCF is introduced')
    panel(ax3, 'd')
    save(fig, config, 'Composite_Original_01_WorkloadStructure')

def fig2_feature_space_prediction(config: str) -> None:
    t8 = table(config, 'Table8_Classification.csv')
    t26 = table(config, 'Table26_MLFeatureFamilyAblation.csv')
    t29 = table(config, 'Table29_MLTopologyVectorModels.csv')
    if t8.empty and t26.empty:
        return
    fig = plt.figure(figsize=(9.6, 6.3))
    header(fig, f'{config}: original topology is predictive but feature-heavy', 'Predictive signal is distributed across broad feature families and vector encodings.')
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.96, top=0.84, bottom=0.1, hspace=0.5, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    if not t8.empty:
        sub = t8.copy()
        y = np.arange(len(sub))
        ax0.barh(y, sub['Balanced_accuracy'], color=COLORS['teal'], alpha=0.82)
        ax0.axvline(0.5, color='black', ls=':', lw=1.0)
        ax0.set_yticks(y, sub['Model'], fontsize=6.8)
        ax0.invert_yaxis()
        ax0.set_xlim(0.35, max(0.82, sub['Balanced_accuracy'].max() + 0.05))
        ax0.set_xlabel('balanced accuracy')
    ax0.set_title('model-level classification')
    panel(ax0, 'a')
    ax1 = fig.add_subplot(gs[0, 1:])
    if not t26.empty:
        sub = t26.sort_values('Balanced_accuracy', ascending=True)
        colors = [COLORS['red'] if p < 0.05 else '#9e9e9e' for p in sub['Balanced_perm_p'].fillna(1)]
        ax1.barh(np.arange(len(sub)), sub['Balanced_accuracy'], color=colors, alpha=0.86)
        ax1.axvline(0.5, color='black', ls=':', lw=1.0)
        ax1.set_yticks(np.arange(len(sub)), sub['Feature_family'], fontsize=6.2)
        ax1.set_xlim(0.35, max(0.82, sub['Balanced_accuracy'].max() + 0.05))
        ax1.set_xlabel('balanced accuracy')
    ax1.set_title('feature-family ablation')
    panel(ax1, 'b')
    ax2 = fig.add_subplot(gs[1, 0:2])
    if not t29.empty:
        sub = t29.sort_values('N_raw_vector_features')
        ax2.scatter(sub['N_raw_vector_features'], sub['Balanced_accuracy'], s=70, c=sub['ROC_AUC'], cmap='viridis', edgecolor='black', lw=0.5)
        for (_, r) in sub.iterrows():
            ax2.text(r['N_raw_vector_features'], r['Balanced_accuracy'] + 0.012, str(r['Vector_family']).replace('Frontal ', ''), fontsize=6.2, ha='center')
        ax2.axhline(0.5, color='black', ls=':', lw=1.0)
        ax2.set_xscale('log')
        ax2.set_xlabel('raw vector features (log)')
        ax2.set_ylabel('balanced accuracy')
        ax2.grid(True, color='#eeeeee', lw=0.5)
    ax2.set_title('high-dimensional topology vectors')
    panel(ax2, 'c')
    ax3 = fig.add_subplot(gs[1, 2])
    if not t26.empty:
        counts = t26[['Feature_family', 'N_features']].sort_values('N_features', ascending=False).head(7)
        ax3.bar(np.arange(len(counts)), counts['N_features'], color=COLORS['purple'], alpha=0.78)
        ax3.set_xticks(np.arange(len(counts)), [s[:12] for s in counts['Feature_family']], rotation=60, ha='right', fontsize=6.0)
        ax3.set_ylabel('features')
    ax3.set_title('feature universe')
    panel(ax3, 'd')
    save(fig, config, 'Composite_Original_02_FeatureSpacePrediction')

def fig3_alpha_fc_dynamics(config: str) -> None:
    t17 = table(config, 'Table17_RegionAlphaFCTopology.csv')
    t18 = table(config, 'Table18_RegionAlphaFCDynamicLandscapeVolatility.csv')
    t20 = table(config, 'Table20_PrimaryBranchComparison.csv')
    if t17.empty and t18.empty and t20.empty:
        return
    fig = plt.figure(figsize=(9.0, 6.0))
    header(fig, f'{config}: alpha-FC topology links compression and dynamics', 'Static, dynamic, and branch-comparison panels separate topology from simple alpha power.')
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.96, top=0.84, bottom=0.1, hspace=0.48, wspace=0.35)
    for (idx, (tab, title, lab)) in enumerate([(t17, 'static FC topology', 'a'), (t18, 'dynamic FC topology', 'b')]):
        ax = fig.add_subplot(gs[0, idx])
        if not tab.empty:
            sub = tab.set_index('Target').reindex(['confirmatory_frontal_roi', 'frontal', 'posterior', 'whole_head']).dropna(subset=['RBC_signed'], how='all')
            ax.barh(np.arange(len(sub)), sub['RBC_signed'], color=[COLORS['red'] if v > 0 else COLORS['blue'] for v in sub['RBC_signed']], alpha=0.82)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_yticks(np.arange(len(sub)), [REGION_LABELS.get(i, i) for i in sub.index], fontsize=6.8)
            ax.invert_yaxis()
            ax.set_xlabel('signed RBC')
            ax.grid(True, axis='x', color='#eeeeee', lw=0.5)
        ax.set_title(title)
        panel(ax, lab)
    ax2 = fig.add_subplot(gs[0, 2])
    if not t20.empty:
        sub = t20.copy()
        y = np.arange(len(sub))
        ax2.barh(y, sub['RBC_signed'], color=[COLORS['red'] if v > 0 else COLORS['blue'] for v in sub['RBC_signed']], alpha=0.82)
        ax2.axvline(0, color='black', lw=0.8)
        ax2.set_yticks(y, sub['Target'], fontsize=6.5)
        ax2.invert_yaxis()
        ax2.set_xlabel('signed RBC')
    ax2.set_title('topology vs alpha/FC branches')
    panel(ax2, 'c')
    ax3 = fig.add_subplot(gs[1, :])
    draw_story_axis(ax3, ['alpha FC\nnetwork', 'landscape\nsummaries', 'dynamic\nvolatility', 'not reducible\nto alpha alone'], [COLORS['blue'], COLORS['teal'], COLORS['red'], COLORS['green']], 'claim path: original topology reveals a brain-state organization issue that motivates a custom filtration')
    panel(ax3, 'd')
    save(fig, config, 'Composite_Original_03_AlphaFCTopologyDynamics')

def fig4_network_drivers(config: str) -> None:
    edges = table(config, 'Table24_FrontalEdgeTopologyDrivers.csv')
    nodes = table(config, 'Table30_NetworkNodeDriverSummary.csv')
    if edges.empty:
        return
    fig = plt.figure(figsize=(9.0, 6.1))
    header(fig, f'{config}: edge and node drivers explain the topology shift', 'Frontolateral edges and node summaries identify candidate network mechanisms behind compression.')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.96, top=0.84, bottom=0.1, hspace=0.42, wspace=0.32)
    ax0 = fig.add_subplot(gs[:, 0])
    g = nx.Graph()
    frontal = sorted(set(edges['Channel_A']).union(edges['Channel_B']))
    pos = {ch: SCALP_POS.get(ch, (math.cos(i), math.sin(i))) for (i, ch) in enumerate(frontal)}
    for (_, r) in edges.sort_values('Topology_driver_score', ascending=False).head(12).iterrows():
        g.add_edge(r['Channel_A'], r['Channel_B'], weight=float(r['Topology_driver_score']), sig=str(r.get('Sig_edge', '')))
    nx.draw_networkx_nodes(g, pos, ax=ax0, node_color='white', edgecolors='black', node_size=300, linewidths=0.8)
    widths = [1.0 + 4.0 * g[u][v]['weight'] for (u, v) in g.edges]
    colors = [COLORS['red'] if '*' in g[u][v]['sig'] else '#9e9e9e' for (u, v) in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax0, width=widths, edge_color=colors, alpha=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax0, font_size=7)
    ax0.set_title('candidate frontal driver network')
    ax0.axis('off')
    panel(ax0, 'a')
    ax1 = fig.add_subplot(gs[0, 1:])
    top_edges = edges.sort_values('Topology_driver_score', ascending=True).tail(10)
    ax1.barh(np.arange(len(top_edges)), top_edges['Topology_driver_score'], color=COLORS['red'], alpha=0.82)
    ax1.set_yticks(np.arange(len(top_edges)), top_edges['Edge'], fontsize=6.5)
    ax1.set_xlabel('driver score')
    ax1.set_title('top edge-level drivers')
    ax1.grid(True, axis='x', color='#eeeeee', lw=0.5)
    panel(ax1, 'b')
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(edges['Rest_minus_task_median'], edges['RBC_signed_edge'], s=45, c=edges['Topology_driver_score'], cmap='magma', edgecolor='black', lw=0.4)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel('rest-task similarity')
    ax2.set_ylabel('edge RBC')
    ax2.set_title('edge effect geometry')
    ax2.grid(True, color='#eeeeee', lw=0.5)
    panel(ax2, 'c')
    ax3 = fig.add_subplot(gs[1, 2])
    if not nodes.empty:
        sub = nodes.sort_values('Max_driver_score', ascending=True).tail(8)
        ax3.barh(np.arange(len(sub)), sub['Max_driver_score'], color=COLORS['purple'], alpha=0.8)
        ax3.set_yticks(np.arange(len(sub)), sub['Node'], fontsize=6.8)
        ax3.set_xlabel('max node driver score')
    ax3.set_title('node driver concentration')
    ax3.grid(True, axis='x', color='#eeeeee', lw=0.5)
    panel(ax3, 'd')
    save(fig, config, 'Composite_Original_04_NetworkDrivers')

def fig5_evidence_synthesis(config: str) -> None:
    ev = table(config, 'Table25_CrossAnalysisEvidenceMap.csv')
    rank = table(config, 'Table23_DiscoverySignalRanking.csv')
    ctrl = table(config, 'Table21_PrimaryMetric_Controls.csv')
    excl = table(config, 'Table22_PrimaryMetric_ExclusionSensitivity.csv')
    if ev.empty and rank.empty:
        return
    fig = plt.figure(figsize=(9.2, 6.2))
    header(fig, f'{config}: cross-analysis evidence map', 'The original topology result is broad but coherent; weak subjective links remain supplementary.')
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.97, top=0.84, bottom=0.1, hspace=0.45, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0:2])
    if not ev.empty:
        sub = ev.sort_values('Abs_rho', ascending=False).head(18)
        y = np.arange(len(sub))
        ax0.scatter(sub['Abs_rho'], y, s=60 + 160 * sub['Sig'].astype(str).str.contains('\\*').astype(float), c=sub['Spearman_rho_with_primary_delta'], cmap='coolwarm', vmin=-1, vmax=1, edgecolor='black', lw=0.4)
        short = [str(s).replace('confirmatory_frontal_roi', 'frontolateral ROI').replace(' topology', '').replace(' regional', '') for s in sub['Feature_label']]
        ax0.set_yticks(y, short, fontsize=5.8)
        ax0.invert_yaxis()
        ax0.set_xlabel('|rho with primary delta|')
        ax0.set_title('cross-analysis convergence')
        ax0.grid(True, axis='x', color='#eeeeee', lw=0.5)
    panel(ax0, 'a')
    ax1 = fig.add_subplot(gs[0, 2])
    if not rank.empty:
        sub = rank.head(10).iloc[::-1]
        ax1.barh(np.arange(len(sub)), sub['RBC_signed'], color=[COLORS['red'] if v > 0 else COLORS['blue'] for v in sub['RBC_signed']], alpha=0.82)
        ax1.axvline(0, color='black', lw=0.8)
        short_targets = [str(s).replace('confirmatory_frontal_roi', 'frontolateral ROI').replace('Frontal dynamic FC topology', 'frontal dynamic FC') for s in sub['Target']]
        ax1.set_yticks(np.arange(len(sub)), short_targets, fontsize=5.8)
        ax1.set_xlabel('signed RBC')
        ax1.set_title('ranked discovery signals')
    panel(ax1, 'b')
    ax2 = fig.add_subplot(gs[1, 0])
    if not ctrl.empty:
        row = ctrl[ctrl['Analysis'].eq('cross_condition_association')]
        if not row.empty:
            vals = row[['r_primary_vs_condition', 'r_partial_condition_given_frontal_alpha', 'r_partial_condition_given_fc_strength', 'r_partial_condition_given_alpha_plus_fc']].iloc[0]
            labels = ['raw', 'alpha', 'FC', 'alpha+FC']
            ax2.bar(labels, vals.values.astype(float), color=[COLORS['gray'], COLORS['gold'], COLORS['teal'], COLORS['purple']], alpha=0.8)
            ax2.axhline(0, color='black', lw=0.8)
            ax2.set_ylabel('r / partial r')
        ax2.set_title('control-model residual signal')
    panel(ax2, 'c')
    ax3 = fig.add_subplot(gs[1, 1])
    if not excl.empty:
        ax3.barh(np.arange(len(excl)), excl['RBC_signed'], color=[COLORS['red'] if v > 0 else COLORS['blue'] for v in excl['RBC_signed']], alpha=0.82)
        ax3.axvline(0, color='black', lw=0.8)
        ax3.set_yticks(np.arange(len(excl)), excl['Scenario'], fontsize=6.0)
        ax3.set_xlabel('signed RBC')
        ax3.set_title('exclusion sensitivity')
    panel(ax3, 'd')
    ax4 = fig.add_subplot(gs[1, 2])
    draw_story_axis(ax4, ['effect', 'controls', 'drivers', 'NWCF\nmotivation'], [COLORS['red'], COLORS['gold'], COLORS['purple'], COLORS['green']], 'synthesis')
    panel(ax4, 'e')
    save(fig, config, 'Composite_Original_05_EvidenceSynthesis')

def optional_assessment(config: str) -> None:
    corr = table(config, 'Table28_SubjectiveRatingCorrelations.csv')
    clean_corr = False
    if not corr.empty:
        sig = corr['Sig'].astype(str).str.contains('\\*').any()
        clean_corr = bool(sig and corr['p_FDR'].min() < 0.1)
    lines = ['Optional original-topology figure assessment', '===========================================', '', f'Config: {config}', '', 'Main composite figures generated:', '  Composite_Original_01_WorkloadStructure.png', '  Composite_Original_02_FeatureSpacePrediction.png', '  Composite_Original_03_AlphaFCTopologyDynamics.png', '  Composite_Original_04_NetworkDrivers.png', '  Composite_Original_05_EvidenceSynthesis.png', '', f"Figure15_SubjectiveTopologyCorrelation.png: {('possible supplement' if clean_corr else 'do not use in main text; correlations are not clean/FDR-strong enough')}", 'Figure4_RepresentativeGeometry.png: method/QC supplement only, not main text.', 'Figure6_RepresentativeRepresentations.png: method/QC supplement only, not main text.', 'Figure8_RepresentativeCircularCoordinates.png: method/QC supplement only, not main text.']
    (out_dir(config) / 'Composite_Original_keep_drop_notes.txt').write_text('\n'.join(lines) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='*', default=DEFAULT_CONFIGS)
    args = parser.parse_args()
    set_style()
    for cfg in args.configs:
        if not out_dir(cfg).exists():
            print(f'[skip] missing {out_dir(cfg)}')
            continue
        print(f'[plot] {cfg}')
        fig1_workload_structure(cfg)
        fig2_feature_space_prediction(cfg)
        fig3_alpha_fc_dynamics(cfg)
        fig4_network_drivers(cfg)
        fig5_evidence_synthesis(cfg)
        optional_assessment(cfg)
    print('Composite original topology claim figures complete.')
if __name__ == '__main__':
    main()
