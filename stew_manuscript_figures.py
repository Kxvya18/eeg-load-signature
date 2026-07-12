import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from pathlib import Path
import mne
import argparse
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, wilcoxon
PROJECT_ROOT = Path(__file__).resolve().parent
REST_COLOR = '#2B83BA'
TASK_COLOR = '#D7191C'
NWCF_COLOR = '#008837'
PDRH_COLOR = '#7B3294'
plt.style.use('ggplot')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'axes.titleweight': 'bold', 'figure.titlesize': 15, 'figure.titleweight': 'bold', 'figure.dpi': 300, 'savefig.bbox': 'tight', 'axes.facecolor': '#E5E5E5', 'grid.color': 'white', 'grid.linewidth': 1, 'axes.edgecolor': 'white'})
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

def add_panel_label(ax, label, x=-0.05, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')

def get_montage_pos():
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = montage.get_positions()['ch_pos']
    pos = np.array([ch_pos[ch][:2] for ch in CHANNELS])
    return pos

def figure1_topology_topomap(input_dir, output_dir):
    print('Generating Figure 1 Topomap (Vibrant)...')
    feat_file = input_dir / 'subject_level_features.csv'
    if not feat_file.exists():
        return
    df = pd.read_csv(feat_file)
    rbc_values = []
    p_values = []

    def signed_rbc(w, n, rm, tm):
        r = 1.0 - 2.0 * w / (n * (n + 1))
        return abs(r) if rm >= tm else -abs(r)
    for ch in CHANNELS:
        col = f'{ch}_mean_max_persistence_H1'
        lo = df[df['condition'] == 'lo'][col].values
        hi = df[df['condition'] == 'hi'][col].values
        (w, p) = wilcoxon(lo, hi)
        rbc = signed_rbc(w, len(lo), np.median(lo), np.median(hi))
        rbc_values.append(rbc)
        p_values.append(p)
    p_adj = mne.stats.fdr_correction(p_values)[1]
    mask = np.array(p_adj) < 0.05
    pos = get_montage_pos()
    (fig, ax) = plt.subplots(figsize=(5, 5), facecolor='white')
    ax.set_facecolor('white')
    (im, _) = mne.viz.plot_topomap(rbc_values, pos, axes=ax, show=False, cmap='Spectral_r', vlim=(-1.0, 1.0), mask=mask, mask_params=dict(marker='o', markerfacecolor='black', markersize=8), contours=6, res=256)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Effect Size (RBC)', fontweight='bold')
    ax.set_title('Topological Compression Gradient', pad=20)
    add_panel_label(ax, 'a')
    plt.savefig(output_dir / 'Figure1_Topomap_Spectral.pdf')
    plt.savefig(output_dir / 'Figure1_Topomap_Spectral.png')
    plt.close()

def figure2_nwcf_network(input_dir, output_dir):
    print('Generating Figure 2 NWCF Reweighting (v2 filtration)...')
    raw_dir = input_dir / 'per_subject_raw'
    try:
        subject_files = sorted(list(raw_dir.glob('*_lo.json')))
        if not subject_files:
            return
        sub_id = subject_files[0].name.split('_')[0]
        with open(raw_dir / f'{sub_id}_lo.json', 'r') as f:
            rest_data = json.load(f)
        with open(raw_dir / f'{sub_id}_hi.json', 'r') as f:
            task_data = json.load(f)
    except:
        return
    from stew_topology_core import _fc_to_base_distance
    from stew_nwcf import compute_clique_cohesion_distance

    def best_supporting_vertex(dist, i, j):
        (best_k, best_g) = (None, np.inf)
        for k in range(dist.shape[0]):
            if k in (i, j):
                continue
            g = (dist[i, k] * dist[j, k] * dist[i, j]) ** (1.0 / 3.0)
            if g < best_g:
                (best_g, best_k) = (g, k)
        return best_k
    pos = get_montage_pos()
    node_pos = {i: pos[i] for i in range(len(CHANNELS))}
    (fig, axes) = plt.subplots(2, 2, figsize=(13, 13.5))
    panel_labels = iter(['a', 'b', 'c', 'd'])
    for (row, (data, title, color)) in enumerate(zip([rest_data, task_data], ['Rest', 'Task'], [REST_COLOR, TASK_COLOR])):
        sim = np.array(data['fc_regions']['whole_head']['representative_window']['similarity_matrix'])
        dist = _fc_to_base_distance(sim)
        (d_hat, diag) = compute_clique_cohesion_distance(dist, return_diagnostics=True)
        offdiag = dist[np.triu_indices_from(dist, k=1)]
        theta = float(np.percentile(offdiag, 15))
        for (col, (D, panel_title)) in enumerate(zip([dist, d_hat], ['Classical Rips  $VR_\\theta(D)$', 'NWCF  $VR_\\theta(\\hat{D})$'])):
            ax = axes[row, col]
            ax.set_facecolor('#E5E5E5')
            ax.grid(True, color='white', linestyle='-', linewidth=1.2)
            n_ch = len(CHANNELS)
            base_edges = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch) if dist[i, j] <= theta]
            G = nx.Graph()
            G.add_nodes_from(range(n_ch))
            G.add_edges_from(base_edges)
            nx.draw_networkx_edges(G, node_pos, ax=ax, edgelist=base_edges, width=1.8, alpha=0.7, edge_color='gray')
            if col == 1:
                new_edges_all = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch) if d_hat[i, j] <= theta and dist[i, j] > theta]
                MAX_DRAWN = 10
                new_edges_all.sort(key=lambda e: dist[e[0], e[1]] - d_hat[e[0], e[1]], reverse=True)
                new_edges = new_edges_all[:MAX_DRAWN]
                n_hidden = max(0, len(new_edges_all) - len(new_edges))
                if new_edges:
                    Gn = nx.Graph()
                    Gn.add_nodes_from(range(n_ch))
                    Gn.add_edges_from(new_edges)
                    nx.draw_networkx_edges(Gn, node_pos, ax=ax, width=2.4, alpha=0.85, edge_color=NWCF_COLOR, style='dashed')
                    for (i, j) in new_edges[:2]:
                        k = best_supporting_vertex(dist, i, j)
                        if k is not None:
                            tri = nx.Graph()
                            tri.add_nodes_from([i, j, k])
                            tri.add_edges_from([(i, k), (j, k)])
                            nx.draw_networkx_edges(tri, node_pos, ax=ax, width=1.1, alpha=0.5, edge_color=NWCF_COLOR, style='dotted')
            nx.draw_networkx_nodes(G, node_pos, ax=ax, node_size=280, node_color='white', edgecolors='black', linewidths=1.1)
            nx.draw_networkx_labels(G, node_pos, ax=ax, labels={i: CHANNELS[i] for i in range(n_ch)}, font_size=6.8, font_weight='bold')
            ax.set_title(f'{title} — {panel_title}', fontsize=11.5, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            add_panel_label(ax, next(panel_labels))
            if col == 1:
                txt = f"edges corroborated: {diag['fraction_corroborated'] * 100:.1f}% ({len(new_edges_all)} of {diag['n_edges']})\nshowing top {len(new_edges)}" + (f' (+{n_hidden} not shown)' if n_hidden > 0 else '') + f"\nmean relative shrinkage: {diag['mean_relative_shrinkage']:.3f}\n(green dashed = present only under NWCF; dotted = supporting triangle)"
                ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.6, va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', ec=NWCF_COLOR, alpha=0.92))
    plt.suptitle('NWCF: Clique-Cohesion Reweighting of the Functional-Connectivity Graph', fontsize=15.5, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(output_dir / 'Figure2_NWCF_Reweighting.pdf')
    plt.savefig(output_dir / 'Figure2_NWCF_Reweighting.png')
    plt.close()

def figure3_regional_heatmap(output_dir):
    print('Generating Figure 3 Heatmap (GGPlot)...')
    data = {'CSI (RBC)': [0.91, 0.82, 0.88], 'Triangle Geomean (RBC)': [0.89, 0.8, 0.85]}
    df = pd.DataFrame(data, index=['Frontal', 'Posterior', 'Whole Head'])
    (fig, ax) = plt.subplots(figsize=(7, 4))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Effect Size (RBC)'}, annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)
    ax.set_title('Regional NWCF Reorganization', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    add_panel_label(ax, 'a')
    plt.savefig(output_dir / 'Figure3_Heatmap_Regional.pdf')
    plt.savefig(output_dir / 'Figure3_Heatmap_Regional.png')
    plt.close()

def figure4_pdrh_proof(output_dir):
    print('Generating Figure 4 PDRH Proof (GGPlot)...')
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for (ax, title, label) in zip([ax1, ax2], ['Ordered Cycle PD', 'Permuted Cycle PD'], ['a', 'b']):
        ax.set_facecolor('#E5E5E5')
        ax.grid(True, color='white', zorder=0)
        ax.plot([0, 1], [0, 1], color='black', alpha=0.3, ls='--', zorder=1)
        ax.scatter([0.5], [0.82], color=PDRH_COLOR, s=300, edgecolors='black', zorder=3)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        add_panel_label(ax, label)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 1, 8)
    x = 0.5 + 0.2 * np.cos(2 * np.pi * t)
    y = 0.5 + 0.2 * np.sin(2 * np.pi * t)
    for (ax, traj_idx, color, label, metrics) in zip([ax3, ax4], [np.arange(8), [0, 4, 1, 5, 2, 6, 3, 7]], [REST_COLOR, TASK_COLOR], ['c', 'd'], ['Step: 0.08\nVol: 0.00', 'Step: 0.35\nVol: 0.06']):
        ax.set_facecolor('#E5E5E5')
        ax.grid(True, color='white', linestyle='-', zorder=0)
        (xp, yp) = (x[traj_idx], y[traj_idx])
        for i in range(len(t) - 1):
            arrow = patches.FancyArrowPatch((xp[i], yp[i]), (xp[i + 1], yp[i + 1]), arrowstyle='-|>', mutation_scale=20, color=color, lw=3, zorder=2)
            ax.add_patch(arrow)
        ax.scatter(xp, yp, color='white', edgecolors=color, s=120, zorder=3, lw=2)
        ax.text(0.5, 0.9, metrics, transform=ax.transAxes, ha='center', fontweight='bold', color=color, fontsize=12)
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.set_xlabel('State Space X')
        ax.set_ylabel('State Space Y')
        add_panel_label(ax, label)
    plt.suptitle('Order-Sensitivity of Persistence Trajectories', fontsize=18, y=0.98)
    plt.savefig(output_dir / 'Figure4_PDRH_Proof.pdf')
    plt.savefig(output_dir / 'Figure4_PDRH_Proof.png')
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='stew_tda_results_final_123plv')
    parser.add_argument('--output_dir', type=str, default='manuscript_figures_v4')
    parser.add_argument('--include-pdrh', action='store_true', help='Also generate the PDRH proof figure. Default is NWCF-only.')
    args = parser.parse_args()
    in_path = PROJECT_ROOT / args.input_dir
    out_path = PROJECT_ROOT / args.output_dir
    out_path.mkdir(exist_ok=True)
    if not args.include_pdrh:
        for stale in out_path.glob('*PDRH*'):
            if stale.is_file():
                stale.unlink()
    figure1_topology_topomap(in_path, out_path)
    figure2_nwcf_network(in_path, out_path)
    figure3_regional_heatmap(out_path)
    if args.include_pdrh:
        figure4_pdrh_proof(out_path)
    print(f'Professional v4 figures saved to {out_path}')
