import argparse
import json
from pathlib import Path
import pandas as pd
PRIMARY_TARGET = 'Frontal dynamic FC topology'

def read_csv_if_exists(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def extract_primary_row(table20):
    if table20.empty or 'Target' not in table20.columns:
        return {}
    match = table20[table20['Target'] == PRIMARY_TARGET]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {'primary_rest_median': row.get('Rest_median'), 'primary_task_median': row.get('Task_median'), 'primary_rbc': row.get('RBC_signed'), 'primary_p_raw': row.get('p_raw'), 'primary_p_fdr': row.get('p_FDR'), 'primary_p_perm': row.get('p_perm'), 'primary_ci_lower': row.get('RBC_CI_lower'), 'primary_ci_upper': row.get('RBC_CI_upper'), 'primary_direction': row.get('Direction')}

def extract_comparator_rows(table20):
    rows = {}
    for (label, prefix) in [('Frontal alpha power', 'alpha'), ('Frontal alpha FC strength', 'fc_strength')]:
        match = table20[table20['Target'] == label] if not table20.empty and 'Target' in table20.columns else pd.DataFrame()
        if match.empty:
            continue
        row = match.iloc[0]
        rows[f'{prefix}_rbc'] = row.get('RBC_signed')
        rows[f'{prefix}_p_fdr'] = row.get('p_FDR')
        rows[f'{prefix}_direction'] = row.get('Direction')
    return rows

def extract_control_rows(table21):
    out = {}
    if table21.empty or 'Analysis' not in table21.columns:
        return out
    cross = table21[table21['Analysis'] == 'cross_condition_association']
    if not cross.empty:
        row = cross.iloc[0]
        out.update({'partial_alpha_r': row.get('r_partial_condition_given_frontal_alpha'), 'partial_alpha_p': row.get('p_partial_condition_given_frontal_alpha'), 'partial_fc_r': row.get('r_partial_condition_given_fc_strength'), 'partial_fc_p': row.get('p_partial_condition_given_fc_strength'), 'partial_alpha_fc_r': row.get('r_partial_condition_given_alpha_plus_fc'), 'partial_alpha_fc_p': row.get('p_partial_condition_given_alpha_plus_fc')})
    delta = table21[table21['Analysis'] == 'within_subject_delta_coupling']
    if not delta.empty:
        row = delta.iloc[0]
        out.update({'delta_alpha_r': row.get('r_delta_primary_vs_delta_alpha'), 'delta_alpha_p': row.get('p_delta_primary_vs_delta_alpha'), 'delta_fc_strength_r': row.get('r_delta_primary_vs_delta_fc_strength'), 'delta_fc_strength_p': row.get('p_delta_primary_vs_delta_fc_strength')})
    return out

def extract_exclusion_rows(table22):
    out = {}
    if table22.empty or 'Scenario' not in table22.columns:
        return out
    for scenario in ['full_qc_exclusions', 'failed_only', 'failed_plus_fallback_asr', 'no_exclusions']:
        match = table22[table22['Scenario'] == scenario]
        if match.empty:
            continue
        row = match.iloc[0]
        prefix = scenario
        out[f'{prefix}_n_pairs'] = row.get('N_pairs')
        out[f'{prefix}_rbc'] = row.get('RBC_signed')
        out[f'{prefix}_p_raw'] = row.get('p_raw')
    return out

def extract_cross_analysis_rows(table25):
    if table25.empty or 'Feature_label' not in table25.columns:
        return {}
    ranked = table25.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False])
    row = ranked.iloc[0]
    return {'top_cross_feature': row.get('Feature_label'), 'top_cross_family': row.get('Feature_family'), 'top_cross_rho': row.get('Spearman_rho_with_primary_delta'), 'top_cross_p_fdr': row.get('p_FDR'), 'top_cross_sig': row.get('Sig')}

def extract_ml_rows(table26):
    if table26.empty or 'Feature_family' not in table26.columns:
        return {}
    ranked = table26.sort_values(['Balanced_accuracy', 'ROC_AUC'], ascending=[False, False])
    row = ranked.iloc[0]
    return {'best_ml_scope': row.get('Scope'), 'best_ml_family': row.get('Feature_family'), 'best_ml_balanced_accuracy': row.get('Balanced_accuracy'), 'best_ml_roc_auc': row.get('ROC_AUC'), 'best_ml_perm_p': row.get('Balanced_perm_p'), 'best_ml_sig': row.get('Sig')}

def extract_subjective_rows(table28):
    if table28.empty or 'Feature_label' not in table28.columns:
        return {}
    ranked = table28.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False])
    row = ranked.iloc[0]
    return {'top_subjective_feature': row.get('Feature_label'), 'top_subjective_family': row.get('Feature_family'), 'top_subjective_rho': row.get('Spearman_rho'), 'top_subjective_p_fdr': row.get('p_FDR'), 'top_subjective_sig': row.get('Sig')}

def extract_vector_ml_rows(table29):
    if table29.empty or 'Vector_family' not in table29.columns:
        return {}
    ranked = table29.sort_values(['Balanced_accuracy', 'ROC_AUC'], ascending=[False, False])
    row = ranked.iloc[0]
    return {'best_vector_ml_family': row.get('Vector_family'), 'best_vector_ml_balanced_accuracy': row.get('Balanced_accuracy'), 'best_vector_ml_roc_auc': row.get('ROC_AUC'), 'best_vector_ml_perm_p': row.get('Balanced_perm_p'), 'best_vector_ml_sig': row.get('Sig')}

def extract_node_rows(table30):
    if table30.empty or 'Node' not in table30.columns:
        return {}
    ranked = table30.sort_values(['N_candidate_edges', 'Max_driver_score'], ascending=[False, False])
    row = ranked.iloc[0]
    return {'top_driver_node': row.get('Node'), 'top_driver_node_candidate_edges': row.get('N_candidate_edges'), 'top_driver_node_max_score': row.get('Max_driver_score')}

def extract_edge_rows(table24):
    if table24.empty or 'Edge' not in table24.columns:
        return {}
    ranked = table24.sort_values('Topology_driver_score', ascending=False)
    row = ranked.iloc[0]
    return {'top_edge': row.get('Edge'), 'top_edge_role': row.get('Discovery_role'), 'top_edge_score': row.get('Topology_driver_score'), 'top_edge_rbc': row.get('RBC_signed_edge'), 'top_edge_rho': row.get('rho_edge_delta_vs_topology_delta'), 'top_edge_p_fdr': row.get('p_FDR_edge')}

def summarize_analysis_dir(analysis_dir):
    analysis_dir = Path(analysis_dir)
    snapshot_path = analysis_dir / 'analysis_config_snapshot.json'
    if not snapshot_path.exists():
        raise FileNotFoundError(f'Missing analysis snapshot: {snapshot_path}')
    with open(snapshot_path) as f:
        snapshot = json.load(f)
    table20 = read_csv_if_exists(analysis_dir / 'Table20_PrimaryBranchComparison.csv')
    table21 = read_csv_if_exists(analysis_dir / 'Table21_PrimaryMetric_Controls.csv')
    table22 = read_csv_if_exists(analysis_dir / 'Table22_PrimaryMetric_ExclusionSensitivity.csv')
    table24 = read_csv_if_exists(analysis_dir / 'Table24_FrontalEdgeTopologyDrivers.csv')
    table25 = read_csv_if_exists(analysis_dir / 'Table25_CrossAnalysisEvidenceMap.csv')
    table26 = read_csv_if_exists(analysis_dir / 'Table26_MLFeatureFamilyAblation.csv')
    table28 = read_csv_if_exists(analysis_dir / 'Table28_SubjectiveRatingCorrelations.csv')
    table29 = read_csv_if_exists(analysis_dir / 'Table29_MLTopologyVectorModels.csv')
    table30 = read_csv_if_exists(analysis_dir / 'Table30_NetworkNodeDriverSummary.csv')
    row = {'analysis_dir': str(analysis_dir), 'paired_n': snapshot.get('paired_n'), 'normalization_mode': snapshot.get('normalization_mode'), 'window_seconds': snapshot.get('window_seconds'), 'surrogate_n_per_method': snapshot.get('surrogate_n_per_method'), 'fc_connectivity_metric': snapshot.get('fc_connectivity_metric'), 'excluded_n': len(snapshot.get('excluded_subjects', []))}
    row.update(extract_primary_row(table20))
    row.update(extract_comparator_rows(table20))
    row.update(extract_control_rows(table21))
    row.update(extract_exclusion_rows(table22))
    row.update(extract_edge_rows(table24))
    row.update(extract_cross_analysis_rows(table25))
    row.update(extract_ml_rows(table26))
    row.update(extract_subjective_rows(table28))
    row.update(extract_vector_ml_rows(table29))
    row.update(extract_node_rows(table30))
    return row

def build_edge_recurrence(analysis_dirs):
    rows = []
    for analysis_dir in analysis_dirs:
        analysis_dir = Path(analysis_dir)
        table24 = read_csv_if_exists(analysis_dir / 'Table24_FrontalEdgeTopologyDrivers.csv')
        if table24.empty or 'Edge' not in table24.columns:
            continue
        for (_, row) in table24.iterrows():
            role = str(row.get('Discovery_role', ''))
            if role == 'low_priority':
                continue
            rows.append({'analysis_dir': str(analysis_dir), 'Edge': row.get('Edge'), 'Discovery_role': role, 'Topology_driver_score': row.get('Topology_driver_score'), 'RBC_signed_edge': row.get('RBC_signed_edge'), 'rho_edge_delta_vs_topology_delta': row.get('rho_edge_delta_vs_topology_delta'), 'p_FDR_edge': row.get('p_FDR_edge')})
    if not rows:
        return pd.DataFrame()
    edge_df = pd.DataFrame(rows)
    edge_df['Topology_driver_score'] = pd.to_numeric(edge_df['Topology_driver_score'], errors='coerce')
    edge_df['p_FDR_edge'] = pd.to_numeric(edge_df['p_FDR_edge'], errors='coerce')
    recurrence = edge_df.groupby('Edge').agg(N_runs=('analysis_dir', 'nunique'), Mean_driver_score=('Topology_driver_score', 'mean'), Best_driver_score=('Topology_driver_score', 'max'), Best_edge_p_FDR=('p_FDR_edge', 'min'), Roles=('Discovery_role', lambda x: ';'.join(sorted(set(map(str, x)))))).reset_index().sort_values(['N_runs', 'Mean_driver_score'], ascending=[False, False])
    return recurrence

def write_summary_text(summary_df, output_path):
    lines = ['PRIMARY ROBUSTNESS SUMMARY']
    if summary_df.empty:
        lines.append('No runs were available.')
    else:
        ranked = summary_df.sort_values(['primary_p_fdr', 'primary_p_raw', 'primary_rbc'], ascending=[True, True, False])
        best = ranked.iloc[0]
        lines.extend([f'Runs summarized: {len(summary_df)}', f"Best primary run: {best['analysis_dir']} | metric={best.get('fc_connectivity_metric')} | window={best.get('window_seconds')} s | surrogates={best.get('surrogate_n_per_method')}", f"Primary effect: RBC={best.get('primary_rbc'):.3f}, p_FDR={best.get('primary_p_fdr'):.4f}, p_perm={best.get('primary_p_perm'):.4f}, direction={best.get('primary_direction')}"])
        if 'best_ml_family' in summary_df.columns:
            ml_ranked = summary_df.dropna(subset=['best_ml_balanced_accuracy']).sort_values('best_ml_balanced_accuracy', ascending=False)
            if not ml_ranked.empty:
                ml = ml_ranked.iloc[0]
                lines.extend(['', 'Best ML feature-family result:', f"{ml.get('analysis_dir')} | {ml.get('best_ml_scope')} / {ml.get('best_ml_family')} | balanced accuracy={float(ml.get('best_ml_balanced_accuracy')):.3f}, ROC-AUC={float(ml.get('best_ml_roc_auc')):.3f}, perm p={float(ml.get('best_ml_perm_p')):.4f}"])
        if 'top_edge' in summary_df.columns:
            edge_counts = summary_df['top_edge'].dropna().value_counts()
            if not edge_counts.empty:
                lines.extend(['', 'Most frequent top edge-driver candidate:', f'{edge_counts.index[0]} ({int(edge_counts.iloc[0])} run(s) as top edge)'])
        if 'best_vector_ml_family' in summary_df.columns:
            vector_ranked = summary_df.dropna(subset=['best_vector_ml_balanced_accuracy']).sort_values('best_vector_ml_balanced_accuracy', ascending=False)
            if not vector_ranked.empty:
                vec = vector_ranked.iloc[0]
                lines.extend(['', 'Best topology-vector ML result:', f"{vec.get('analysis_dir')} | {vec.get('best_vector_ml_family')} | balanced accuracy={float(vec.get('best_vector_ml_balanced_accuracy')):.3f}, ROC-AUC={float(vec.get('best_vector_ml_roc_auc')):.3f}, perm p={float(vec.get('best_vector_ml_perm_p')):.4f}"])
        if 'top_driver_node' in summary_df.columns:
            node_counts = summary_df['top_driver_node'].dropna().value_counts()
            if not node_counts.empty:
                lines.extend(['', 'Most frequent node-level driver candidate:', f'{node_counts.index[0]} ({int(node_counts.iloc[0])} run(s) as top node)'])
    output_path.write_text('\n'.join(lines) + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Summarize robustness runs for the STEW primary FC topology branch.')
    parser.add_argument('analysis_dirs', nargs='+', help='One or more stew analysis output directories.')
    parser.add_argument('--output-dir', default='stew_robustness_summary', help='Directory for summary outputs.')
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_analysis_dir(path) for path in args.analysis_dirs]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / 'robustness_summary.csv', index=False)
    edge_recurrence = build_edge_recurrence(args.analysis_dirs)
    edge_recurrence.to_csv(output_dir / 'edge_recurrence_summary.csv', index=False)
    write_summary_text(summary_df, output_dir / 'robustness_summary.txt')
if __name__ == '__main__':
    main()
