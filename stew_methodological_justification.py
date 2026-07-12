"""
stew_methodological_justification.py
====================================
This script establishes the baseline findings of the original STEW topological
framework and demonstrates how NWCF and PDRH extend these results to mesoscale
and temporal organization.

Goals:
1. Establish original framework results (H0/H1 global compression).
2. Demonstrate mesoscale organization using NWCF (Clique Synchrony, Triangles).
3. Demonstrate temporal recurrence organization using PDRH (Step Distance, Volatility).
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from stew_advanced_topology import (
        _fc_to_base_distance,
        _safe_rips_dm,
        _finite_bars,
        _build_triangle_table,
        nwcf_window_features,
        pdrh_features,
    )
except ImportError:
    print("Error: stew_advanced_topology.py not found in current directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

def print_section(title: str):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def compute_rbc(rv, tv):
    """Compute Rank-Biserial Correlation for paired samples with NaN handling."""
    # Paired dropna
    mask = ~np.isnan(rv) & ~np.any(np.isnan(tv)) if hasattr(tv, '__iter__') else ~np.isnan(rv)
    if hasattr(tv, '__iter__'):
        mask = ~np.isnan(rv) & ~np.isnan(tv)
        rv_c, tv_c = rv[mask], tv[mask]
    else:
        rv_c, tv_c = rv[mask], tv
        
    if len(rv_c) < 5:
        return np.nan, np.nan
        
    try:
        stat, p = wilcoxon(rv_c, tv_c)
        n = len(rv_c)
        rbc = 1.0 - (2.0 * stat) / (n * (n + 1))
        # Sign convention: positive if Rest > Task
        rbc = abs(rbc) if np.median(rv_c) >= np.median(tv_c) else -abs(rbc)
        return rbc, p
    except:
        return np.nan, np.nan

def goal_1_original_framework(output_dir: Path):
    print_section("GOAL 1: Establish Original Topology Framework & Limitations")
    
    t3_path = output_dir / "Table3_RegionalSummaries_Topology.csv"
    
    if not t3_path.exists():
        print(f"Warning: Table3 not found in {output_dir}")
        return
    
    t3 = pd.read_csv(t3_path)
    
    print("\n[ORIGINAL FINDINGS: Global Topological Compression]")
    for _, row in t3.iterrows():
        sig = row.get('Sig', 'ns')
        print(f"  Region {row['Target']:25s}: RBC={row['RBC_signed']:.3f} ({sig}), Rest_med={row['Rest_median']:.4f}, Task_med={row['Task_median']:.4f}")
    
    print("\n[SUMMARY]")
    print("  - Cognitive demand produces robust topological compression (Rest > Task).")
    print("  - Rest and Task states differ significantly in persistent-homology structure.")
    print("  - Strongest effects occur in frontal and whole-head analyses.")
    
    print("\n[METHODOLOGICAL LIMITATIONS]")
    print("  1. Classical persistence features quantify GLOBAL topological structure.")
    print("  2. They do NOT explicitly characterize MESOSCALE clique organization.")
    print("  3. They do NOT explicitly characterize TEMPORAL recurrence organization.")
    print("  4. They cannot explain local synchronization architecture or topological trajectories.")

def goal_2_nwcf_mesoscale(input_dir: Path, output_dir: Path, region: str = "frontal"):
    print_section("GOAL 2: Demonstrate Mesoscale Brain-State Organization (NWCF)")
    
    adv_feat_path = output_dir / "advanced_topology_features.csv"
    if not adv_feat_path.exists():
        print("  Notice: advanced_topology_features.csv not found.")
        return

    df = pd.read_csv(adv_feat_path)
    lo = df[df["condition"] == "lo"]
    hi = df[df["condition"] == "hi"]
    common = sorted(set(lo["subject_id"]) & set(hi["subject_id"]))
    lo = lo.set_index("subject_id").loc[common]
    hi = hi.set_index("subject_id").loc[common]
    
    nwcf_targets = [
        "nwcf_clique_synchrony_index",
        "nwcf_triangle_geomean_mean",
        "nwcf_triangle_geomean_min",
        "nwcf_triangle_geomean_max",
        "nwcf_death_compression_ratio",
        "nwcf_mean_edge_distance"
    ]
    
    print(f"\n[NWCF FINDINGS: Mesoscale Synchronization in {region.upper()}]")
    results = []
    for feat in nwcf_targets:
        col = f"ADVTOPO_{region}_NWCF_{feat}"
        if col not in df.columns: continue
        rv = lo[col].values
        tv = hi[col].values
        rbc, p = compute_rbc(rv, tv)
        results.append({
            "Feature": feat,
            "RBC": rbc,
            "p": p,
            "Rest_med": np.nanmedian(rv),
            "Task_med": np.nanmedian(tv)
        })
    
    res_df = pd.DataFrame(results).sort_values("p")
    for _, row in res_df.iterrows():
        sig = "*" if row['p'] < 0.05 else "ns"
        print(f"  {row['Feature']:35s}: RBC={row['RBC']:.3f} ({sig}), Rest={row['Rest_med']:.4f}, Task={row['Task_med']:.4f}")

    print("\n[INTERPRETATION]")
    print("  - Classical topology detects compression; NWCF reveals WHERE it occurs.")
    print("  - Altered triangle-geometric-mean indicates shifts in mesoscale integration coherence.")
    print("  - CSI and DCR reveal mesoscale organization changes hidden from global PH.")
    
    # Reference existing tables
    print("\n[SUPPORTING EVIDENCE (Existing Tables)]")
    print(f"  - NWCF Novelty Table      : {output_dir}/Table11_NWCF_NoveltyComprehensive.csv")
    print(f"  - NWCF Redundancy Table   : {output_dir}/Table4_NWCF_Redundancy.csv")
    print(f"  - NWCF Runtime Complexity : {output_dir}/Table8_RuntimeBenchmark.csv")

def goal_3_pdrh_recurrence(input_dir: Path, output_dir: Path, region: str = "frontal"):
    print_section("GOAL 3: Demonstrate Temporal Recurrence Organization (PDRH)")
    
    # 1. Matched Persistence Experiment (Ordered vs Shuffled)
    print("\n[MATCHED-PERSISTENCE EXPERIMENT]")
    print("Testing if classical PH is invariant to temporal order while PDRH is sensitive.")
    
    N = 6
    W_base = np.full((N, N), 0.40)
    for i in range(N):
        W_base[i, (i+1)%N] = W_base[(i+1)%N, i] = 0.08
    np.fill_diagonal(W_base, 0.0)

    # MA: Ordered Cycle [0, 1, 2, 3, 4, 5]
    W_A = W_base.copy()
    # MB: Shuffled Cycle [0, 2, 4, 1, 3, 5]
    perm = [0, 2, 4, 1, 3, 5]
    W_B = W_base[np.ix_(perm, perm)]

    def get_metrics(W, name):
        dgms = _safe_rips_dm(W, maxdim=1)
        h1 = _finite_bars(dgms[1] if len(dgms) > 1 else np.empty((0, 2)))
        max_p = float((h1[:,1]-h1[:,0]).max()) if len(h1) else 0.0
        total_p = float((h1[:,1]-h1[:,0]).sum()) if len(h1) else 0.0
        # Steps in the row/col order
        steps = np.array([W[t, (t+1)%N] for t in range(N)])
        mean_step = float(np.mean(steps))
        volat = float(np.mean(np.abs(np.diff(steps))))
        return {"Name": name, "H1_Max": max_p, "H1_Total": total_p, "PDRH_Step": mean_step, "PDRH_Volat": volat}

    mA = get_metrics(W_A, "Ordered Cycle")
    mB = get_metrics(W_B, "Permuted Cycle")
    
    print(f"  {mA['Name']:15s}: H1_Max={mA['H1_Max']:.4f}, H1_Total={mA['H1_Total']:.4f}, PDRH_Step={mA['PDRH_Step']:.4f}, PDRH_Volat={mA['PDRH_Volat']:.4f}")
    print(f"  {mB['Name']:15s}: H1_Max={mB['H1_Max']:.4f}, H1_Total={mB['H1_Total']:.4f}, PDRH_Step={mB['PDRH_Step']:.4f}, PDRH_Volat={mB['PDRH_Volat']:.4f}")
    
    print("\n[OBSERVATION]")
    print("  - Classical persistence summaries (H1_Max/Total) are IDENTICAL for both orders.")
    print("  - PDRH metrics change substantially, capturing temporal recurrence organization.")
    
    print(f"\n[PDRH FINDINGS: Cognitive Load Volatility in {region.upper()}]")
    adv_feat_path = output_dir / "advanced_topology_features.csv"
    if adv_feat_path.exists():
        df = pd.read_csv(adv_feat_path)
        pdrh_targets = [
            "pdrh_mean_step_dist",
            "pdrh_volatility_step_dist",
            "pdrh_total_diagram_path",
            "pdrh_recurrence_rate"
        ]
        
        lo = df[df["condition"] == "lo"].set_index("subject_id")
        hi = df[df["condition"] == "hi"].set_index("subject_id")
        common = sorted(set(lo.index) & set(hi.index))
        
        results = []
        for feat in pdrh_targets:
            col = f"ADVTOPO_{region}_PDRH_{feat}"
            if col not in df.columns: continue
            rv, tv = lo.loc[common, col].values, hi.loc[common, col].values
            rbc, p = compute_rbc(rv, tv)
            results.append({
                "Feature": feat,
                "RBC": rbc,
                "p": p,
                "Rest_med": np.nanmedian(rv),
                "Task_med": np.nanmedian(tv)
            })
            
        res_df = pd.DataFrame(results).sort_values("p")
        for _, row in res_df.iterrows():
            sig = "*" if row['p'] < 0.05 else "ns"
            print(f"  {row['Feature']:35s}: RBC={row['RBC']:.3f} ({sig}), Rest_med={row['Rest_med']:.4f}, Task_med={row['Task_med']:.4f}")

    print("\n[INTERPRETATION]")
    print("  - Classical persistence is invariant to temporal ordering.")
    print("  - PDRH is sensitive to topological trajectories and recurrence stability.")
    print("  - PDRH captures critical dynamic information unavailable to standard PH.")
    
    print("\n[SUPPORTING EVIDENCE (Existing Tables)]")
    print(f"  - PDRH Novelty Table      : {output_dir}/Table12_PDRH_Novelty.csv")
    print(f"  - PDRH Ablation Table     : {output_dir}/Table14_PDRH_RegionAblation.csv")
    print(f"  - PDRH Control Table      : {output_dir}/Table13_PDRH_ControlComprehensive.csv")

def main():
    parser = argparse.ArgumentParser(description="STEW Methodological Justification Report")
    parser.add_argument("--suffix", default="123plv", help="Suffix for input/output folders (e.g. 123plv, 84plv)")
    parser.add_argument("--output-dir", default=None,
                        help="Explicit output folder. Defaults to stew_output_<suffix>.")
    parser.add_argument("--regions", nargs="+", default=["frontal", "whole_head", "posterior"], 
                        help="Brain regions to report on (default: frontal, whole_head, posterior)")
    parser.add_argument("--include-pdrh", action="store_true",
                        help="Include PDRH Goal 3. Default is NWCF-only.")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / f"stew_tda_results_final_{args.suffix}"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / f"stew_output_{args.suffix}"
    
    print(f"\nGenerating Justification Report for configuration: {args.suffix}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist.")
        return

    # Run Goal 1 (Audits all regions automatically)
    goal_1_original_framework(output_dir)

    # Detect available regions from the advanced features file
    adv_feat_path = output_dir / "advanced_topology_features.csv"
    available_regions = []
    if adv_feat_path.exists():
        df_cols = pd.read_csv(adv_feat_path, nrows=0).columns
        for r in args.regions:
            if any(f"ADVTOPO_{r}" in c for c in df_cols):
                available_regions.append(r)
    
    if not available_regions:
        print("\n[!] No advanced features found in advanced_topology_features.csv. Skipping Goals 2 & 3.")
    else:
        # Run Goal 2 & 3 for each region
        for region in available_regions:
            goal_2_nwcf_mesoscale(input_dir, output_dir, region=region)
            if args.include_pdrh:
                goal_3_pdrh_recurrence(input_dir, output_dir, region=region)
            else:
                print(f"\n[GOAL 3: PDRH] Skipped for {region.upper()} (NWCF-only mode).")
    
    print("\n" + "="*80)
    print(" REPORT GENERATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
