import argparse
import json
import os
import platform
import warnings
from collections import OrderedDict
from pathlib import Path
import matplotlib
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mplconfig'))
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from scipy.stats import norm as norm_dist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
DEFAULT_INPUT_DIR = PROJECT_ROOT / 'stew_tda_results_final_123plv'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'stew_output_final_123plv'
REST_COLOR = '#1F77B4'
TASK_COLOR = '#D62728'
TOPOLOGY_COLOR = '#0B6E4F'
ALPHA_COLOR = '#C44E00'
COMBINED_COLOR = '#6A4C93'
TOPO_FEATURE = 'median_max_persistence_H1'
LANDSCAPE_FEATURE = 'median_landscape_total_area_H1'
IMAGE_FEATURE = 'median_persistence_image_mass_H1'
SURROGATE_FEATURE = 'median_iaaft_surrogate_z_landscape_total_area_H1'
DYNAMIC_LANDSCAPE_FEATURE = 'volatility_landscape_total_area_H1'
FC_TOPO_FEATURE = 'median_fc_landscape_total_area_H1'
FC_DYNAMIC_FEATURE = 'volatility_fc_landscape_total_area_H1'
LATERALIZATION_FEATURE = 'LAT_F_median_landscape_total_area_H1'
ALPHA_FEATURE = 'median_alpha_power'
PRIMARY_REGION = 'frontal'
PRIMARY_FEATURE = f'FCREGION_{PRIMARY_REGION}_{FC_DYNAMIC_FEATURE}'
PRIMARY_ALPHA_FEATURE = f'REGION_{PRIMARY_REGION}_{ALPHA_FEATURE}'
PRIMARY_FC_STRENGTH_FEATURE = f'FCREGION_{PRIMARY_REGION}_mean_fc_mean_similarity'
HEAD_COORDS = {'AF3': (0.28, 0.92), 'F7': (0.08, 0.74), 'F3': (0.28, 0.7), 'FC5': (0.14, 0.52), 'T7': (0.03, 0.32), 'P7': (0.18, 0.12), 'O1': (0.36, 0.03), 'O2': (0.64, 0.03), 'P8': (0.82, 0.12), 'T8': (0.97, 0.32), 'FC6': (0.86, 0.52), 'F4': (0.72, 0.7), 'F8': (0.92, 0.74), 'AF4': (0.72, 0.92)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = pvals[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return np.minimum(adjusted, 1.0)

def signed_rbc(w_stat, n_pairs, rest_med, task_med):
    r = 1.0 - 2.0 * w_stat / (n_pairs * (n_pairs + 1))
    return abs(r) if rest_med >= task_med else -abs(r)

def sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.1:
        return '†'
    return 'ns'

def effect_label(r):
    ar = abs(r)
    if ar >= 0.5:
        return 'Large'
    if ar >= 0.3:
        return 'Medium'
    if ar >= 0.1:
        return 'Small'
    return 'Negligible'

def permutation_wilcoxon(rest_vals, task_vals, n_perm=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    (obs_stat, _) = wilcoxon(rest_vals, task_vals, alternative='two-sided')
    diffs = rest_vals - task_vals
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        shuffled = diffs * signs
        try:
            (stat, _) = wilcoxon(shuffled, alternative='two-sided')
        except Exception:
            stat = 0.0
        if stat <= obs_stat:
            count += 1
    return count / n_perm

def bootstrap_rbc(rest_vals, task_vals, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(rest_vals)
    rbcs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = rest_vals[idx]
        t = task_vals[idx]
        try:
            (w_stat, _) = wilcoxon(r, t, alternative='two-sided')
            rbcs.append(signed_rbc(w_stat, n, np.median(r), np.median(t)))
        except Exception:
            rbcs.append(0.0)
    return np.percentile(rbcs, [2.5, 97.5])

def power_min_rbc(n, alpha=0.05, power=0.8):
    z_power = norm_dist.ppf(power)
    z_alpha = norm_dist.ppf(1 - alpha / 2)
    se = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    return float((z_power + z_alpha) * se)

def partial_spearman(x, y, z):
    from scipy.stats import rankdata

    def resid(a, b):
        a = np.asarray(a, dtype=float) - np.mean(a)
        b = np.asarray(b, dtype=float) - np.mean(b)
        denom = np.dot(b, b)
        beta = np.dot(b, a) / denom if denom > 0 else 0.0
        return a - beta * b
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    return spearmanr(resid(rx, rz), resid(ry, rz))

def run_paired_tests(lo_df, hi_df, targets, feature_name, scope_name, n_perm=1000, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    rows = []
    pvals = []
    for (label, column) in targets:
        if column not in lo_df.columns or column not in hi_df.columns:
            continue
        idx = lo_df[column].dropna().index.intersection(hi_df[column].dropna().index)
        if len(idx) < 5:
            continue
        rest_vals = lo_df.loc[idx, column].values
        task_vals = hi_df.loc[idx, column].values
        (w_stat, p_raw) = wilcoxon(rest_vals, task_vals, alternative='two-sided')
        rbc = signed_rbc(w_stat, len(idx), np.median(rest_vals), np.median(task_vals))
        (ci_lo, ci_hi) = bootstrap_rbc(rest_vals, task_vals, n_boot=n_boot, rng=rng)
        rows.append({'Scope': scope_name, 'Target': label, 'Feature': feature_name, 'N_pairs': len(idx), 'Rest_median': float(np.median(rest_vals)), 'Task_median': float(np.median(task_vals)), 'W': float(w_stat), 'p_raw': float(p_raw), 'RBC_signed': float(rbc), 'Effect_size': effect_label(rbc), 'Direction': 'Rest > Task' if rbc > 0 else 'Task > Rest', 'RBC_CI_lower': float(ci_lo), 'RBC_CI_upper': float(ci_hi), '_rest': rest_vals, '_task': task_vals})
        pvals.append(p_raw)
    if not rows:
        return pd.DataFrame()
    p_adj = fdr_bh(pvals)
    for (row, p_fdr) in zip(rows, p_adj):
        row['p_FDR'] = float(p_fdr)
        row['Sig'] = sig_label(p_fdr)
        row['Sig_raw'] = sig_label(row['p_raw'])
        row['p_perm'] = float(permutation_wilcoxon(row['_rest'], row['_task'], n_perm=n_perm, rng=rng))
        row['Min_detectable_RBC'] = power_min_rbc(row['N_pairs'])
        row['Observed_vs_MDR'] = float(abs(row['RBC_signed']) / row['Min_detectable_RBC']) if row['Min_detectable_RBC'] > 0 else np.inf
    clean_rows = [{k: v for (k, v) in row.items() if not k.startswith('_')} for row in rows]
    return pd.DataFrame(clean_rows)

def add_region_features(features, region_channels):
    region_map = OrderedDict(region_channels)
    for (region_name, channels) in region_map.items():
        for feature_name in [TOPO_FEATURE, LANDSCAPE_FEATURE, IMAGE_FEATURE, SURROGATE_FEATURE, ALPHA_FEATURE, 'mean_total_persistence_H0']:
            cols = [f'{ch}_{feature_name}' for ch in channels if f'{ch}_{feature_name}' in features.columns]
            if cols:
                features[f'REGION_{region_name}_{feature_name}'] = features[cols].median(axis=1)
    return features

def build_stage2_audit(features, preproc, config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    error_rows = preproc[preproc['error'].fillna('').astype(str).str.len() > 0].copy()
    failed_subjects = sorted(error_rows['subject_id'].unique().tolist())
    post_max_by_subject = preproc[preproc['error'].fillna('').astype(str).str.len() == 0].groupby('subject_id')['post_pct_samples_abs_gt_100uv'].max()
    q1 = float(post_max_by_subject.quantile(0.25))
    q3 = float(post_max_by_subject.quantile(0.75))
    iqr = q3 - q1
    iqr_threshold = float(q3 + 3 * iqr)
    residual_outliers = sorted(post_max_by_subject[post_max_by_subject > iqr_threshold].index.tolist())
    excluded_subjects = sorted(set(failed_subjects) | set(residual_outliers))
    features_ok = features[~features['subject_id'].isin(excluded_subjects)].copy()
    lo = features_ok[features_ok['condition'] == 'lo'].set_index('subject_id')
    hi = features_ok[features_ok['condition'] == 'hi'].set_index('subject_id')
    common_subs = sorted(lo.index.intersection(hi.index).tolist())
    missing_pairs = sorted(set(features_ok['subject_id'].unique().tolist()) - set(common_subs))
    audit_lines = ['STAGE 2 AUDIT', f'Project root: {PROJECT_ROOT}', f"Input dir: {config.get('output_dir', str(DEFAULT_INPUT_DIR))}", f'Python: {platform.python_version()}', f"Normalization mode: {config.get('normalization_mode', 'unknown')}", f"Preprocessing sequence: {', '.join(config.get('preprocessing_sequence', []))}", f"Dataset reference: {config.get('references', {}).get('dataset', 'n/a')}", f"Preprocessing reference: {config.get('references', {}).get('preprocessing_sequence_reported', 'n/a')}", '', f'Feature rows read: {len(features)}', f'Preprocessing rows read: {len(preproc)}', f"Unique subjects in features: {features['subject_id'].nunique()}", f'Failed subjects from preprocessing log: {failed_subjects}', f'Residual artefact IQR threshold: {iqr_threshold:.4f}', f'Residual artefact outlier subjects: {residual_outliers}', f'Excluded subjects total: {excluded_subjects}', f'Paired N after exclusions: {len(common_subs)}', f'Subjects present but incomplete after exclusions: {missing_pairs}', '', 'Implemented outputs in this analysis folder are generated from code in the current run.', 'Legacy markdown planning files in the repo are treated as narrative documents, not as evidence of implemented analyses.']
    with open(output_dir / 'stage2_audit.txt', 'w') as f:
        f.write('\n'.join(audit_lines) + '\n')
    with open(output_dir / 'implemented_outputs.txt', 'w') as f:
        f.write('This folder contains implemented analysis outputs generated from stew_pipeline.py and stew_analysis.py.\n')
        f.write('Do not treat legacy Tier-1 planning markdown files as implemented outputs.\n')
    return {'failed_subjects': failed_subjects, 'residual_outliers': residual_outliers, 'excluded_subjects': excluded_subjects, 'paired_subjects': common_subs, 'iqr_threshold': iqr_threshold}

def loso_accuracy(x, y, groups, pipeline, splitter):
    preds = []
    truths = []
    for (train_idx, test_idx) in splitter.split(x, y, groups):
        pipeline.fit(x[train_idx], y[train_idx])
        preds.extend(pipeline.predict(x[test_idx]))
        truths.extend(y[test_idx])
    return accuracy_score(truths, preds)

def perm_classifier(x, y, groups, pipeline, splitter, n_perm=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    obs = loso_accuracy(x, y, groups, pipeline, splitter)
    null_accs = []
    unique_groups = np.unique(groups)
    for _ in range(n_perm):
        y_perm = y.copy()
        for group in unique_groups:
            idx = np.where(groups == group)[0]
            y_perm[idx] = rng.permutation(y_perm[idx])
        null_accs.append(loso_accuracy(x, y_perm, groups, pipeline, splitter))
    null_accs = np.array(null_accs)
    return (obs, null_accs, float(np.mean(null_accs >= obs)))

def loso_predictions(x, y, groups, pipeline, splitter, feature_names):
    preds = []
    scores = []
    truths = []
    coef_rows = []
    for (fold_idx, (train_idx, test_idx)) in enumerate(splitter.split(x, y, groups)):
        pipeline.fit(x[train_idx], y[train_idx])
        fold_pred = pipeline.predict(x[test_idx])
        if hasattr(pipeline.named_steps['svm'], 'decision_function'):
            fold_score = pipeline.decision_function(x[test_idx])
        else:
            fold_score = fold_pred.astype(float)
        preds.extend(fold_pred)
        scores.extend(np.asarray(fold_score, dtype=float).ravel())
        truths.extend(y[test_idx])
        coefs = pipeline.named_steps['svm'].coef_[0]
        for (name, coef) in zip(feature_names, coefs):
            coef_rows.append({'Fold': fold_idx, 'Feature': name, 'Coefficient': float(coef), 'Abs_coefficient': float(abs(coef))})
    return (np.asarray(truths), np.asarray(preds), np.asarray(scores), pd.DataFrame(coef_rows))

def classification_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'Accuracy': float(acc), 'Accuracy_pct': f'{acc * 100:.1f}%', 'Balanced_accuracy': float(bal), 'Balanced_accuracy_pct': f'{bal * 100:.1f}%', 'ROC_AUC': float(auc) if np.isfinite(auc) else np.nan, 'Sensitivity_task': float(sens), 'Specificity_rest': float(spec), 'TN_rest': int(tn), 'FP_rest_as_task': int(fp), 'FN_task_as_rest': int(fn), 'TP_task': int(tp)}

def perm_classifier_balanced(x, y, groups, pipeline, splitter, n_perm=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    (y_true, y_pred, y_score, _) = loso_predictions(x, y, groups, pipeline, splitter, [f'x{i}' for i in range(x.shape[1])])
    obs_metrics = classification_metrics(y_true, y_pred, y_score)
    null_bal = []
    unique_groups = np.unique(groups)
    for _ in range(n_perm):
        y_perm = y.copy()
        for group in unique_groups:
            idx = np.where(groups == group)[0]
            y_perm[idx] = rng.permutation(y_perm[idx])
        (yt, yp, ys, _) = loso_predictions(x, y_perm, groups, pipeline, splitter, [f'x{i}' for i in range(x.shape[1])])
        null_bal.append(classification_metrics(yt, yp, ys)['Balanced_accuracy'])
    null_bal = np.asarray(null_bal, dtype=float)
    return (obs_metrics, null_bal, float(np.mean(null_bal >= obs_metrics['Balanced_accuracy'])))

def build_classifier_table(lo, hi, scopes, rng, n_perm=500):
    logo = LeaveOneGroupOut()
    model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='linear', C=1.0, random_state=42))])
    rows = []
    nulls = {}
    for (scope_name, channels) in scopes.items():
        topo_cols = [f'{ch}_{TOPO_FEATURE}' for ch in channels if f'{ch}_{TOPO_FEATURE}' in lo.columns]
        alpha_cols = [f'{ch}_{ALPHA_FEATURE}' for ch in channels if f'{ch}_{ALPHA_FEATURE}' in lo.columns]
        if not topo_cols or not alpha_cols:
            continue
        x_topo = []
        x_alpha = []
        x_combined = []
        y = []
        groups = []
        for sid in sorted(lo.index.intersection(hi.index)):
            for (cond_df, label) in [(lo, 0), (hi, 1)]:
                topo_row = cond_df.loc[sid, topo_cols].values.astype(float)
                alpha_row = cond_df.loc[sid, alpha_cols].values.astype(float)
                x_topo.append(topo_row)
                x_alpha.append(alpha_row)
                x_combined.append(np.concatenate([topo_row, alpha_row]))
                y.append(label)
                groups.append(sid)
        x_topo = np.asarray(x_topo)
        x_alpha = np.asarray(x_alpha)
        x_combined = np.asarray(x_combined)
        y = np.asarray(y)
        groups = np.asarray(groups)
        mask = ~np.any(np.isnan(x_topo) | np.isnan(x_alpha), axis=1)
        x_topo = x_topo[mask]
        x_alpha = x_alpha[mask]
        x_combined = x_combined[mask]
        y = y[mask]
        groups = groups[mask]
        for (label, x) in [('Topology only', x_topo), ('Alpha only', x_alpha), ('Combined', x_combined)]:
            (obs_metrics, null_accs, p_perm) = perm_classifier_balanced(x, y, groups, model, logo, n_perm=n_perm, rng=rng)
            rows.append({'Scope': scope_name, 'Model': label, 'N_subjects': int(len(np.unique(groups))), 'N_samples': int(len(y)), **obs_metrics, 'Perm_p': p_perm, 'Sig': sig_label(p_perm)})
            nulls[scope_name, label] = null_accs
    return (pd.DataFrame(rows), nulls)

def build_ml_dataset(lo, hi, columns):
    x = []
    y = []
    groups = []
    feature_names = [col for col in columns if col in lo.columns and col in hi.columns]
    if not feature_names:
        return (None, None, None, [])
    for sid in sorted(lo.index.intersection(hi.index)):
        for (cond_df, label) in [(lo, 0), (hi, 1)]:
            row = cond_df.loc[sid, feature_names].values.astype(float)
            x.append(row)
            y.append(label)
            groups.append(sid)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    mask = ~np.any(np.isnan(x), axis=1)
    return (x[mask], y[mask], groups[mask], feature_names)

def feature_family_columns(channels):
    classic = [f'{ch}_{TOPO_FEATURE}' for ch in channels]
    landscapes = [f'{ch}_{LANDSCAPE_FEATURE}' for ch in channels]
    surrogate = [f'{ch}_{SURROGATE_FEATURE}' for ch in channels]
    dynamic = [f'{ch}_{DYNAMIC_LANDSCAPE_FEATURE}' for ch in channels]
    alpha = [f'{ch}_{ALPHA_FEATURE}' for ch in channels]
    topo_all = classic + landscapes + surrogate + dynamic
    return OrderedDict([('Classic topology', classic), ('Persistence landscapes', landscapes), ('Surrogate-normalized topology', surrogate), ('Dynamic topology', dynamic), ('Alpha power', alpha), ('All topology families', topo_all), ('All topology + alpha', topo_all + alpha)])

def build_ml_feature_family_ablation(lo, hi, scopes, rng, output_path, importance_path, n_perm=500):
    logo = LeaveOneGroupOut()
    model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='linear', C=1.0, random_state=42))])
    rows = []
    importance_frames = []
    model_specs = []
    for (scope_name, channels) in scopes.items():
        for (family_name, columns) in feature_family_columns(channels).items():
            model_specs.append((scope_name, family_name, columns))
    fc_cols = [PRIMARY_FEATURE]
    fc_alpha_cols = [PRIMARY_FEATURE, PRIMARY_ALPHA_FEATURE, PRIMARY_FC_STRENGTH_FEATURE]
    model_specs.extend([('primary_frontal_fc', 'Primary FC topology', fc_cols), ('primary_frontal_fc', 'Primary FC topology + alpha/FC strength', fc_alpha_cols)])
    for (scope_name, family_name, columns) in model_specs:
        (x, y, groups, feature_names) = build_ml_dataset(lo, hi, columns)
        if x is None or len(np.unique(groups)) < 5 or x.shape[1] == 0:
            continue
        (y_true, y_pred, y_score, coef_df) = loso_predictions(x, y, groups, model, logo, feature_names)
        metrics = classification_metrics(y_true, y_pred, y_score)
        obs_bal = metrics['Balanced_accuracy']
        null_bal = []
        unique_groups = np.unique(groups)
        for _ in range(n_perm):
            y_perm = y.copy()
            for group in unique_groups:
                idx = np.where(groups == group)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            (yt, yp, ys, _) = loso_predictions(x, y_perm, groups, model, logo, feature_names)
            null_bal.append(classification_metrics(yt, yp, ys)['Balanced_accuracy'])
        p_perm = float(np.mean(np.asarray(null_bal) >= obs_bal))
        rows.append({'Scope': scope_name, 'Feature_family': family_name, 'N_subjects': int(len(np.unique(groups))), 'N_samples': int(len(y)), 'N_features': int(x.shape[1]), **metrics, 'Balanced_perm_p': p_perm, 'Sig': sig_label(p_perm)})
        if not coef_df.empty:
            coef_df['Scope'] = scope_name
            coef_df['Feature_family'] = family_name
            importance_frames.append(coef_df)
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(output_path, index=False)
    if importance_frames:
        coef_all = pd.concat(importance_frames, ignore_index=True)
        importance = coef_all.groupby(['Scope', 'Feature_family', 'Feature']).agg(Mean_coefficient=('Coefficient', 'mean'), Mean_abs_coefficient=('Abs_coefficient', 'mean'), SD_abs_coefficient=('Abs_coefficient', 'std'), N_folds=('Fold', 'nunique')).reset_index()
        importance['Stability_score'] = importance['Mean_abs_coefficient'] / importance['SD_abs_coefficient'].replace(0, np.nan)
        importance = importance.sort_values(['Scope', 'Feature_family', 'Mean_abs_coefficient'], ascending=[True, True, False])
    else:
        importance = pd.DataFrame()
    importance.to_csv(importance_path, index=False)
    return (ablation_df, importance)

def plot_confirmatory_raincloud(lo, hi, stats_df, output_path):
    if stats_df.empty or 'Target' not in stats_df.columns:
        return
    (fig, axes) = plt.subplots(1, 2, figsize=(10, 4.8))
    for (ax, channel) in zip(axes, ['F4', 'F3']):
        col = f'{channel}_{TOPO_FEATURE}'
        idx = lo[col].dropna().index.intersection(hi[col].dropna().index)
        rest_vals = lo.loc[idx, col].values
        task_vals = hi.loc[idx, col].values
        row = stats_df[stats_df['Target'] == channel].iloc[0]
        vp = ax.violinplot([rest_vals, task_vals], positions=[1, 2], widths=0.34, showmedians=False, showextrema=False)
        for (body, color) in zip(vp['bodies'], [REST_COLOR, TASK_COLOR]):
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_edgecolor('none')
        rng = np.random.default_rng(99)
        for (xpos, vals, color) in [(1, rest_vals, REST_COLOR), (2, task_vals, TASK_COLOR)]:
            xjit = xpos + rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(xjit, vals, s=20, color=color, alpha=0.6, edgecolors='white', linewidths=0.4)
        for (rv, tv) in zip(rest_vals, task_vals):
            ax.plot([1, 2], [rv, tv], color='gray', alpha=0.18, linewidth=0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Rest', 'Task'])
        ax.set_ylabel('Median H1 max persistence')
        ax.set_title(f"{channel}\nRBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}\nN={row['N_pairs']}", fontsize=10)
    fig.suptitle('Confirmatory frontal topology effect', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_topomap_all14(channel_stats, output_path):
    if channel_stats.empty or 'Target' not in channel_stats.columns:
        return
    (fig, ax) = plt.subplots(figsize=(8, 8))
    head = plt.Circle((0.5, 0.5), 0.48, fill=False, edgecolor='black', linewidth=2.0)
    ax.add_patch(head)
    ax.fill([0.46, 0.54, 0.5], [0.98, 0.98, 1.03], color='black')
    ax.add_patch(plt.Circle((0.02, 0.5), 0.03, color='black'))
    ax.add_patch(plt.Circle((0.98, 0.5), 0.03, color='black'))
    channel_stats = channel_stats.set_index('Target')
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.cm.RdBu_r
    for (ch, (x, y)) in HEAD_COORDS.items():
        rbc = float(channel_stats.loc[ch, 'RBC_signed'])
        sig = channel_stats.loc[ch, 'Sig']
        circle = plt.Circle((x, y), 0.05, color=cmap(norm(rbc)), ec='black', linewidth=1.2)
        ax.add_patch(circle)
        text_color = 'white' if abs(rbc) > 0.45 else 'black'
        ax.text(x, y, ch, ha='center', va='center', fontsize=8.5, fontweight='bold', color=text_color)
        if sig != 'ns':
            ax.text(x + 0.06, y + 0.05, sig, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    cbar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Signed RBC\n(+ Rest > Task, - Task > Rest)')
    fig.suptitle('Exploratory whole-head topology map (14 channels)', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.86, 0.96])
    plt.savefig(output_path)
    plt.close()

def plot_region_bars(region_topo, region_alpha, output_path):
    if region_topo.empty or region_alpha.empty:
        return
    topo = region_topo.set_index('Target')
    alpha = region_alpha.set_index('Target')
    regions = topo.index.tolist()
    x = np.arange(len(regions))
    width = 0.35
    (fig, ax) = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, topo.loc[regions, 'RBC_signed'], width, color=TOPOLOGY_COLOR, label='Topology')
    ax.bar(x + width / 2, alpha.loc[regions, 'RBC_signed'], width, color=ALPHA_COLOR, label='Alpha power')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=15)
    ax.set_ylabel('Signed RBC')
    ax.set_title('Regional summaries: topology vs alpha')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_representative_embeddings(subject_id, channel, input_dir, output_path):
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_lo.json') as f:
        lo = json.load(f)
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_hi.json') as f:
        hi = json.load(f)
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 8))
    for (row_idx, (label, data, color)) in enumerate([('Rest', lo, REST_COLOR), ('Task', hi, TASK_COLOR)]):
        rep = data['channels'][channel]['representative_window']
        if rep is None:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(0.5, 0.5, 'No representative window', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
            continue
        cloud = np.asarray(rep['point_cloud'], dtype=float)
        diag_h1 = np.asarray(rep['diagrams']['H1'], dtype=float)
        axes[row_idx, 0].scatter(cloud[:, 0], cloud[:, 1], s=14, color=color, alpha=0.55)
        axes[row_idx, 0].set_title(f'{label}: {channel} embedded point cloud')
        axes[row_idx, 0].set_xlabel('Dim 1')
        axes[row_idx, 0].set_ylabel('Dim 2')
        if len(diag_h1) == 0:
            axes[row_idx, 1].text(0.5, 0.5, 'No H1 bars', ha='center', va='center')
            axes[row_idx, 1].axis('off')
        else:
            finite = diag_h1[np.isfinite(diag_h1[:, 1])]
            if len(finite) == 0:
                axes[row_idx, 1].text(0.5, 0.5, 'No finite H1 bars', ha='center', va='center')
                axes[row_idx, 1].axis('off')
            else:
                axes[row_idx, 1].scatter(finite[:, 0], finite[:, 1], s=20, color=color, alpha=0.75)
                max_lim = float(np.max(finite))
                axes[row_idx, 1].plot([0, max_lim], [0, max_lim], color='black', linestyle='--', linewidth=1.0)
                axes[row_idx, 1].set_title(f'{label}: {channel} H1 persistence diagram')
                axes[row_idx, 1].set_xlabel('Birth')
                axes[row_idx, 1].set_ylabel('Death')
    fig.suptitle(f'Representative {channel} geometry for subject {subject_id:02d}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_representative_representations(subject_id, channel, input_dir, output_path):
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_lo.json') as f:
        lo = json.load(f)
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_hi.json') as f:
        hi = json.load(f)
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 7.5))
    for (row_idx, (label, data)) in enumerate([('Rest', lo), ('Task', hi)]):
        rep = data['channels'][channel]['representative_window']
        if rep is None:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(0.5, 0.5, 'No representative window', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
            continue
        landscape = np.asarray(rep.get('landscape_values', []), dtype=float)
        landscape_grid = np.asarray(rep.get('landscape_grid', []), dtype=float)
        image = np.asarray(rep.get('persistence_image_values', []), dtype=float)
        birth_grid = np.asarray(rep.get('persistence_image_birth_grid', []), dtype=float)
        pers_grid = np.asarray(rep.get('persistence_image_persistence_grid', []), dtype=float)
        if landscape.size == 0 or landscape_grid.size == 0:
            axes[row_idx, 0].text(0.5, 0.5, 'No landscape saved', ha='center', va='center')
            axes[row_idx, 0].axis('off')
        else:
            for (layer_idx, layer) in enumerate(landscape):
                axes[row_idx, 0].plot(landscape_grid, layer, linewidth=1.4, label=f'Layer {layer_idx + 1}')
            axes[row_idx, 0].set_title(f'{label}: {channel} persistence landscape')
            axes[row_idx, 0].set_xlabel('Filtration value')
            axes[row_idx, 0].set_ylabel('Landscape value')
            axes[row_idx, 0].legend(frameon=False, fontsize=8)
        if image.size == 0 or birth_grid.size == 0 or pers_grid.size == 0:
            axes[row_idx, 1].text(0.5, 0.5, 'No persistence image saved', ha='center', va='center')
            axes[row_idx, 1].axis('off')
        else:
            extent = [birth_grid.min(), birth_grid.max(), pers_grid.min(), pers_grid.max()]
            im = axes[row_idx, 1].imshow(image, origin='lower', aspect='auto', extent=extent, cmap='magma')
            axes[row_idx, 1].set_title(f'{label}: {channel} persistence image')
            axes[row_idx, 1].set_xlabel('Birth')
            axes[row_idx, 1].set_ylabel('Persistence')
            fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
    fig.suptitle(f'Representative {channel} topological representations for subject {subject_id:02d}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_fc_region_representative(subject_id, region_name, input_dir, output_path):
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_lo.json') as f:
        lo = json.load(f)
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_hi.json') as f:
        hi = json.load(f)
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 7.5))
    for (row_idx, (label, data)) in enumerate([('Rest', lo), ('Task', hi)]):
        rep = data.get('fc_regions', {}).get(region_name, {}).get('representative_window')
        if rep is None:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(0.5, 0.5, 'No representative window', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
            continue
        sim = np.asarray(rep.get('similarity_matrix', []), dtype=float)
        landscape = np.asarray(rep.get('landscape_values', []), dtype=float)
        grid = np.asarray(rep.get('landscape_grid', []), dtype=float)
        if sim.size == 0:
            axes[row_idx, 0].text(0.5, 0.5, 'No FC matrix saved', ha='center', va='center')
            axes[row_idx, 0].axis('off')
        else:
            im = axes[row_idx, 0].imshow(sim, vmin=0.0, vmax=1.0, cmap='viridis')
            axes[row_idx, 0].set_title(f'{label}: {region_name} alpha FC')
            axes[row_idx, 0].set_xlabel('Channel')
            axes[row_idx, 0].set_ylabel('Channel')
            fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
        if landscape.size == 0 or grid.size == 0:
            axes[row_idx, 1].text(0.5, 0.5, 'No FC landscape saved', ha='center', va='center')
            axes[row_idx, 1].axis('off')
        else:
            for (layer_idx, layer) in enumerate(landscape):
                axes[row_idx, 1].plot(grid, layer, linewidth=1.4, label=f'Layer {layer_idx + 1}')
            axes[row_idx, 1].set_title(f'{label}: {region_name} FC landscape')
            axes[row_idx, 1].set_xlabel('Filtration value')
            axes[row_idx, 1].set_ylabel('Landscape value')
            axes[row_idx, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Representative alpha FC topology for subject {subject_id:02d}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_representative_circular_coordinates(subject_id, channel, input_dir, output_path):
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_lo.json') as f:
        lo = json.load(f)
    with open(input_dir / 'per_subject_raw' / f'sub{subject_id:02d}_hi.json') as f:
        hi = json.load(f)
    (fig, axes) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for (ax, (label, data, color)) in zip(axes, [('Rest', lo, REST_COLOR), ('Task', hi, TASK_COLOR)]):
        rep = data['channels'][channel]['representative_window']
        theta = np.asarray(rep.get('circular_coordinates_unwrapped', []), dtype=float)
        if theta.size == 0:
            ax.text(0.5, 0.5, 'No circular coordinates saved', ha='center', va='center')
            ax.axis('off')
            continue
        ax.plot(np.arange(len(theta)), theta, color=color, linewidth=1.4)
        ax.set_ylabel('Unwrapped angle')
        ax.set_title(f'{label}: {channel} persistent circular coordinate')
    axes[-1].set_xlabel('Embedded point index')
    fig.suptitle(f'Representative circular-coordinate trajectory for subject {subject_id:02d}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_classification(results_df, output_path):
    if results_df.empty:
        return
    (fig, ax) = plt.subplots(figsize=(9, 4.8))
    plot_df = results_df.copy()
    plot_df['Label'] = plot_df['Scope'] + '\n' + plot_df['Model']
    colors = [TOPOLOGY_COLOR if 'Topology' in model else ALPHA_COLOR if 'Alpha' in model else COMBINED_COLOR for model in plot_df['Model']]
    ax.bar(plot_df['Label'], plot_df['Accuracy'] * 100.0, color=colors, edgecolor='white', linewidth=0.6)
    ax.axhline(50.0, color='black', linestyle='--', linewidth=1.2)
    for (i, row) in plot_df.reset_index(drop=True).iterrows():
        ax.text(i, row['Accuracy'] * 100.0 + 1.2, f"{row['Sig']}\n{row['Accuracy_pct']}", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('LOSO accuracy (%)')
    ax.set_ylim(40, 100)
    ax.set_title('Classification as secondary validation')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_ml_feature_family_ablation(ablation_df, output_path):
    if ablation_df is None or ablation_df.empty:
        return
    plot_df = ablation_df.copy()
    plot_df['Label'] = plot_df['Scope'] + '\n' + plot_df['Feature_family']
    plot_df = plot_df.sort_values('Balanced_accuracy', ascending=False)
    (fig, ax) = plt.subplots(figsize=(12, max(5, 0.35 * len(plot_df))))
    colors = []
    for family in plot_df['Feature_family']:
        if 'Alpha' in family:
            colors.append(ALPHA_COLOR)
        elif 'topology' in family.lower() or 'landscape' in family.lower():
            colors.append(TOPOLOGY_COLOR)
        else:
            colors.append(COMBINED_COLOR)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df['Balanced_accuracy'] * 100.0, color=colors, alpha=0.88)
    ax.axvline(50.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Label'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('LOSO balanced accuracy (%)')
    ax.set_title('Machine-learning feature family ablation')
    for (i, row) in enumerate(plot_df.itertuples(index=False)):
        ax.text(row.Balanced_accuracy * 100.0 + 0.7, i, f'{row.Balanced_accuracy * 100.0:.1f}% {row.Sig}', va='center', fontsize=8)
    ax.set_xlim(45, min(100, max(60, float(plot_df['Balanced_accuracy'].max() * 100.0 + 8))))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_subjective_rating_correlations(lo, hi, config, output_path):
    common = lo.index.intersection(hi.index)
    if 'subjective_rating' not in lo.columns or 'subjective_rating' not in hi.columns or len(common) < 5:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    rating_delta = hi.loc[common, 'subjective_rating'] - lo.loc[common, 'subjective_rating']
    feature_specs = [('Primary frontal FC topology compression', PRIMARY_FEATURE, 'Primary topology'), ('Frontal alpha power increase', PRIMARY_ALPHA_FEATURE, 'Conventional EEG'), ('Frontal FC strength change', PRIMARY_FC_STRENGTH_FEATURE, 'Conventional FC')]
    for region_name in config.get('fc_region_channels', {}):
        feature_specs.append((f'{region_name} dynamic FC topology compression', f'FCREGION_{region_name}_{FC_DYNAMIC_FEATURE}', 'FC topology dynamics'))
        feature_specs.append((f'{region_name} static FC topology change', f'FCREGION_{region_name}_{FC_TOPO_FEATURE}', 'FC topology'))
    for ch in config.get('confirmatory_frontal_channels', []):
        feature_specs.extend([(f'{ch} classic topology change', f'{ch}_{TOPO_FEATURE}', 'Single-channel topology'), (f'{ch} landscape change', f'{ch}_{LANDSCAPE_FEATURE}', 'Persistence landscape'), (f'{ch} surrogate-z topology change', f'{ch}_{SURROGATE_FEATURE}', 'Surrogate topology'), (f'{ch} dynamic topology change', f'{ch}_{DYNAMIC_LANDSCAPE_FEATURE}', 'Dynamic topology'), (f'{ch} alpha power change', f'{ch}_{ALPHA_FEATURE}', 'Conventional EEG')])
    rows = []
    for (label, column, family) in feature_specs:
        if column not in lo.columns or column not in hi.columns:
            continue
        feature_delta = lo.loc[common, column] - hi.loc[common, column]
        valid = pd.DataFrame({'rating_delta': rating_delta, 'feature_delta': feature_delta}).dropna()
        if len(valid) < 5 or float(np.ptp(valid['feature_delta'].values)) == 0:
            continue
        (rho, p) = spearmanr(valid['rating_delta'], valid['feature_delta'])
        rows.append({'Feature_family': family, 'Feature_label': label, 'Feature_column': column, 'N': int(len(valid)), 'Rating_delta_definition': 'high-demand subjective rating minus low-demand subjective rating', 'Feature_delta_definition': 'low/rest metric minus high/task metric', 'Spearman_rho': float(rho), 'p_raw': float(p), 'Abs_rho': float(abs(rho)), 'Interpretation': 'larger topology compression with larger subjective increase' if rho > 0 else 'smaller topology compression with larger subjective increase'})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['p_FDR'] = fdr_bh(df['p_raw'].fillna(1.0).values)
        df['Sig'] = df['p_FDR'].apply(sig_label)
        df = df.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False])
    df.to_csv(output_path, index=False)
    return df

def plot_subjective_correlation(lo, hi, output_path):
    common = lo.index.intersection(hi.index)
    if 'subjective_rating' not in lo.columns or 'subjective_rating' not in hi.columns or PRIMARY_FEATURE not in lo.columns or (PRIMARY_FEATURE not in hi.columns) or (len(common) < 5):
        return
    df = pd.DataFrame({'rating_delta': hi.loc[common, 'subjective_rating'] - lo.loc[common, 'subjective_rating'], 'primary_delta': lo.loc[common, PRIMARY_FEATURE] - hi.loc[common, PRIMARY_FEATURE]}).dropna()
    if len(df) < 5:
        return
    (rho, p) = spearmanr(df['rating_delta'], df['primary_delta'])
    (fig, ax) = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(df['rating_delta'], df['primary_delta'], s=34, color=TOPOLOGY_COLOR, alpha=0.75, edgecolors='white', linewidths=0.5)
    if float(np.ptp(df['rating_delta'].values)) > 0:
        coeffs = np.polyfit(df['rating_delta'], df['primary_delta'], deg=1)
        xfit = np.linspace(float(df['rating_delta'].min()), float(df['rating_delta'].max()), 100)
        ax.plot(xfit, coeffs[0] * xfit + coeffs[1], color='black', linestyle='--', linewidth=1.1)
    ax.axhline(0.0, color='gray', linewidth=0.8)
    ax.axvline(0.0, color='gray', linewidth=0.8)
    ax.set_xlabel('Subjective rating increase (high - low)')
    ax.set_ylabel('Primary topology compression (low - high)')
    ax.set_title(f'Subjective relevance of primary topology change\nSpearman rho={rho:.3f}, p={p:.4f}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_vector_from_json(data, vector_family, channels):
    values = []
    if vector_family == 'Frontal channel landscapes':
        for ch in channels:
            rep = data.get('channels', {}).get(ch, {}).get('representative_window')
            if rep is None:
                return None
            arr = np.asarray(rep.get('landscape_values', []), dtype=float)
            if arr.size == 0:
                return None
            values.extend(arr.ravel().tolist())
    elif vector_family == 'Frontal channel persistence images':
        for ch in channels:
            rep = data.get('channels', {}).get(ch, {}).get('representative_window')
            if rep is None:
                return None
            arr = np.asarray(rep.get('persistence_image_values', []), dtype=float)
            if arr.size == 0:
                return None
            values.extend(arr.ravel().tolist())
    elif vector_family == 'Frontal FC landscape':
        rep = data.get('fc_regions', {}).get(PRIMARY_REGION, {}).get('representative_window')
        if rep is None:
            return None
        arr = np.asarray(rep.get('landscape_values', []), dtype=float)
        if arr.size == 0:
            return None
        values.extend(arr.ravel().tolist())
    elif vector_family == 'Frontal FC persistence image':
        rep = data.get('fc_regions', {}).get(PRIMARY_REGION, {}).get('representative_window')
        if rep is None:
            return None
        arr = np.asarray(rep.get('persistence_image_values', []), dtype=float)
        if arr.size == 0:
            return None
        values.extend(arr.ravel().tolist())
    elif vector_family == 'Combined frontal topology vectors':
        for name in ['Frontal channel landscapes', 'Frontal FC landscape']:
            sub = get_vector_from_json(data, name, channels)
            if sub is None:
                return None
            values.extend(sub)
    else:
        raise ValueError(f'Unknown vector family: {vector_family}')
    return np.asarray(values, dtype=float)

def build_vector_ml_dataset(input_dir, subject_ids, vector_family, channels):
    x = []
    y = []
    groups = []
    for sid in sorted(subject_ids):
        for (condition, label) in [('lo', 0), ('hi', 1)]:
            path = input_dir / 'per_subject_raw' / f'sub{int(sid):02d}_{condition}.json'
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            vec = get_vector_from_json(data, vector_family, channels)
            if vec is None or np.any(~np.isfinite(vec)):
                continue
            x.append(vec)
            y.append(label)
            groups.append(sid)
    if not x:
        return (None, None, None)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    return (x, y, groups)

def build_topology_vector_ml(input_dir, lo, hi, config, rng, output_path, n_perm=200):
    channels = config.get('confirmatory_frontal_channels', [])
    common = sorted(lo.index.intersection(hi.index).tolist())
    families = ['Frontal channel landscapes', 'Frontal channel persistence images', 'Frontal FC landscape', 'Frontal FC persistence image', 'Combined frontal topology vectors']
    logo = LeaveOneGroupOut()
    rows = []
    for family in families:
        (x, y, groups) = build_vector_ml_dataset(input_dir, common, family, channels)
        if x is None or len(np.unique(groups)) < 5:
            continue
        n_components = int(min(10, x.shape[1], max(2, len(np.unique(groups)) - 1)))
        model = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components, random_state=42)), ('svm', SVC(kernel='linear', C=1.0, random_state=42))])
        feature_names = [f'vector_{i:04d}' for i in range(x.shape[1])]
        (y_true, y_pred, y_score, _) = loso_predictions(x, y, groups, model, logo, feature_names)
        metrics = classification_metrics(y_true, y_pred, y_score)
        null_bal = []
        unique_groups = np.unique(groups)
        for _ in range(n_perm):
            y_perm = y.copy()
            for group in unique_groups:
                idx = np.where(groups == group)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            (yt, yp, ys, _) = loso_predictions(x, y_perm, groups, model, logo, feature_names)
            null_bal.append(classification_metrics(yt, yp, ys)['Balanced_accuracy'])
        p_perm = float(np.mean(np.asarray(null_bal) >= metrics['Balanced_accuracy']))
        rows.append({'Vector_family': family, 'N_subjects': int(len(np.unique(groups))), 'N_samples': int(len(y)), 'N_raw_vector_features': int(x.shape[1]), 'PCA_components_inside_LOSO': n_components, **metrics, 'Balanced_perm_p': p_perm, 'Sig': sig_label(p_perm)})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

def plot_topology_vector_ml(vector_ml_df, output_path):
    if vector_ml_df is None or vector_ml_df.empty:
        return
    plot_df = vector_ml_df.sort_values('Balanced_accuracy', ascending=True)
    (fig, ax) = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df['Balanced_accuracy'] * 100.0, color=TOPOLOGY_COLOR, alpha=0.88)
    ax.axvline(50.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Vector_family'], fontsize=8)
    ax.set_xlabel('LOSO balanced accuracy (%)')
    ax.set_title('Vectorized topology representations for ML')
    for (i, row) in enumerate(plot_df.itertuples(index=False)):
        ax.text(row.Balanced_accuracy * 100.0 + 0.7, i, f'{row.Balanced_accuracy * 100.0:.1f}% {row.Sig}', va='center', fontsize=8)
    ax.set_xlim(45, min(100, max(60, float(plot_df['Balanced_accuracy'].max() * 100.0 + 8))))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_network_node_driver_summary(edge_df, output_path):
    if edge_df is None or edge_df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    rows = []
    for (_, edge) in edge_df.iterrows():
        for node in [edge['Channel_A'], edge['Channel_B']]:
            rows.append({'Node': node, 'Edge': edge['Edge'], 'Discovery_role': edge['Discovery_role'], 'Topology_driver_score': edge['Topology_driver_score'], 'RBC_signed_edge': edge['RBC_signed_edge'], 'rho_edge_delta_vs_topology_delta': edge['rho_edge_delta_vs_topology_delta'], 'p_FDR_edge': edge['p_FDR_edge']})
    node_long = pd.DataFrame(rows)
    node_long['Candidate_weight'] = node_long['Discovery_role'].apply(lambda x: 0.0 if x == 'low_priority' else 1.0)
    node_summary = node_long.groupby('Node').agg(N_edges=('Edge', 'nunique'), N_candidate_edges=('Candidate_weight', 'sum'), Mean_driver_score=('Topology_driver_score', 'mean'), Max_driver_score=('Topology_driver_score', 'max'), Mean_abs_edge_RBC=('RBC_signed_edge', lambda x: float(np.mean(np.abs(x)))), Best_edge_p_FDR=('p_FDR_edge', 'min')).reset_index().sort_values(['N_candidate_edges', 'Max_driver_score'], ascending=[False, False])
    node_summary.to_csv(output_path, index=False)
    return node_summary

def plot_network_node_driver_summary(node_summary, output_path):
    if node_summary is None or node_summary.empty:
        return
    (fig, ax) = plt.subplots(figsize=(7, 6))
    head = plt.Circle((0.5, 0.5), 0.48, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(head)
    ax.fill([0.46, 0.54, 0.5], [0.98, 0.98, 1.03], color='black')
    ax.add_patch(plt.Circle((0.02, 0.5), 0.025, color='black'))
    ax.add_patch(plt.Circle((0.98, 0.5), 0.025, color='black'))
    max_score = max(float(node_summary['Max_driver_score'].max()), 1e-09)
    for (_, row) in node_summary.iterrows():
        if row['Node'] not in HEAD_COORDS:
            continue
        (x, y) = HEAD_COORDS[row['Node']]
        size = 450 + 1600 * float(row['Max_driver_score']) / max_score
        color_val = float(row['N_candidate_edges']) / max(float(node_summary['N_candidate_edges'].max()), 1.0)
        ax.scatter([x], [y], s=size, color=plt.cm.Greens(0.25 + 0.65 * color_val), edgecolors='black', linewidths=1.1, zorder=3)
        ax.text(x, y, row['Node'], ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)
    ax.set_title('Node-level summary of topology-driving frontal edges')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_cross_analysis_evidence_map(lo, hi, config, output_path):
    common = lo.index.intersection(hi.index)
    if PRIMARY_FEATURE not in lo.columns or PRIMARY_FEATURE not in hi.columns or len(common) < 5:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    primary_delta = lo.loc[common, PRIMARY_FEATURE] - hi.loc[common, PRIMARY_FEATURE]
    candidate_specs = []
    for (label, column, family) in [('Frontal alpha power', PRIMARY_ALPHA_FEATURE, 'Conventional EEG'), ('Frontal mean FC strength', PRIMARY_FC_STRENGTH_FEATURE, 'Conventional FC'), ('Frontal topology lateralization', LATERALIZATION_FEATURE, 'Topology lateralization')]:
        candidate_specs.append((label, column, family))
    for ch in config.get('confirmatory_frontal_channels', []):
        candidate_specs.extend([(f'{ch} classic H1', f'{ch}_{TOPO_FEATURE}', 'Single-channel topology'), (f'{ch} landscape', f'{ch}_{LANDSCAPE_FEATURE}', 'Persistence landscape'), (f'{ch} surrogate z', f'{ch}_{SURROGATE_FEATURE}', 'Surrogate topology'), (f'{ch} dynamic landscape', f'{ch}_{DYNAMIC_LANDSCAPE_FEATURE}', 'Dynamic single-channel topology'), (f'{ch} alpha power', f'{ch}_{ALPHA_FEATURE}', 'Conventional EEG')])
    for region_name in config.get('region_channels', {}):
        candidate_specs.extend([(f'{region_name} regional classic H1', f'REGION_{region_name}_{TOPO_FEATURE}', 'Regional topology'), (f'{region_name} regional landscape', f'REGION_{region_name}_{LANDSCAPE_FEATURE}', 'Regional landscape'), (f'{region_name} regional alpha', f'REGION_{region_name}_{ALPHA_FEATURE}', 'Regional alpha')])
    for region_name in config.get('fc_region_channels', {}):
        candidate_specs.extend([(f'{region_name} static FC topology', f'FCREGION_{region_name}_{FC_TOPO_FEATURE}', 'FC topology'), (f'{region_name} dynamic FC topology', f'FCREGION_{region_name}_{FC_DYNAMIC_FEATURE}', 'FC topology dynamics'), (f'{region_name} FC strength', f'FCREGION_{region_name}_mean_fc_mean_similarity', 'FC strength')])
    rows = []
    for (label, column, family) in candidate_specs:
        if column not in lo.columns or column not in hi.columns:
            continue
        feature_delta = lo.loc[common, column] - hi.loc[common, column]
        valid = pd.DataFrame({'primary_delta': primary_delta, 'feature_delta': feature_delta}).dropna()
        if len(valid) < 5 or float(np.ptp(valid['feature_delta'].values)) == 0:
            continue
        (rho, p) = spearmanr(valid['primary_delta'], valid['feature_delta'])
        rows.append({'Feature_family': family, 'Feature_label': label, 'Feature_column': column, 'N': int(len(valid)), 'Spearman_rho_with_primary_delta': float(rho), 'p_raw': float(p), 'Abs_rho': float(abs(rho)), 'Direction_with_primary': 'same_direction' if rho > 0 else 'opposite_direction'})
    evidence = pd.DataFrame(rows)
    if not evidence.empty:
        evidence['p_FDR'] = fdr_bh(evidence['p_raw'].fillna(1.0).values)
        evidence['Sig'] = evidence['p_FDR'].apply(sig_label)
        evidence = evidence.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False])
    evidence.to_csv(output_path, index=False)
    return evidence

def plot_cross_analysis_evidence_map(evidence_df, output_path, top_n=30):
    if evidence_df is None or evidence_df.empty:
        return
    plot_df = evidence_df.sort_values('Abs_rho', ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values('Spearman_rho_with_primary_delta')
    (fig, ax) = plt.subplots(figsize=(10, max(5, 0.3 * len(plot_df))))
    colors = [TOPOLOGY_COLOR if 'topology' in fam.lower() or 'landscape' in fam.lower() else ALPHA_COLOR for fam in plot_df['Feature_family']]
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df['Spearman_rho_with_primary_delta'], color=colors, alpha=0.88)
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['Feature_label'], fontsize=8)
    ax.set_xlabel('Spearman rho with primary topology delta')
    ax.set_title('Cross-analysis evidence map')
    for (i, row) in enumerate(plot_df.itertuples(index=False)):
        ax.text(row.Spearman_rho_with_primary_delta + (0.02 if row.Spearman_rho_with_primary_delta >= 0 else -0.02), i, row.Sig, va='center', ha='left' if row.Spearman_rho_with_primary_delta >= 0 else 'right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_primary_metric(lo, hi, column, output_path, title):
    idx = lo[column].dropna().index.intersection(hi[column].dropna().index)
    if len(idx) == 0:
        return
    rest_vals = lo.loc[idx, column].values
    task_vals = hi.loc[idx, column].values
    (fig, ax) = plt.subplots(figsize=(6.8, 4.8))
    vp = ax.violinplot([rest_vals, task_vals], positions=[1, 2], widths=0.34, showmedians=False, showextrema=False)
    for (body, color) in zip(vp['bodies'], [REST_COLOR, TASK_COLOR]):
        body.set_facecolor(color)
        body.set_alpha(0.25)
        body.set_edgecolor('none')
    rng = np.random.default_rng(123)
    for (xpos, vals, color) in [(1, rest_vals, REST_COLOR), (2, task_vals, TASK_COLOR)]:
        xjit = xpos + rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(xjit, vals, s=20, color=color, alpha=0.6, edgecolors='white', linewidths=0.4)
    for (rv, tv) in zip(rest_vals, task_vals):
        ax.plot([1, 2], [rv, tv], color='gray', alpha=0.18, linewidth=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Rest', 'Task'])
    ax.set_ylabel('Metric value')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_primary_delta_coupling(lo, hi, output_path):
    common = lo.index.intersection(hi.index)
    if len(common) == 0:
        return
    delta_df = pd.DataFrame({'delta_primary': hi.loc[common, PRIMARY_FEATURE] - lo.loc[common, PRIMARY_FEATURE], 'delta_alpha': hi.loc[common, PRIMARY_ALPHA_FEATURE] - lo.loc[common, PRIMARY_ALPHA_FEATURE], 'delta_fc_strength': hi.loc[common, PRIMARY_FC_STRENGTH_FEATURE] - lo.loc[common, PRIMARY_FC_STRENGTH_FEATURE]}).dropna()
    if len(delta_df) < 5:
        return
    (fig, axes) = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [('delta_alpha', 'Delta frontal alpha power'), ('delta_fc_strength', 'Delta frontal alpha FC strength')]
    for (ax, (xcol, xlabel)) in zip(axes, panels):
        x = delta_df[xcol].values
        y = delta_df['delta_primary'].values
        (r, p) = spearmanr(x, y)
        ax.scatter(x, y, s=32, color=TOPOLOGY_COLOR, alpha=0.7, edgecolors='white', linewidths=0.5)
        if len(x) >= 2 and float(np.ptp(x)) > 0:
            coeffs = np.polyfit(x, y, deg=1)
            xfit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yfit = coeffs[0] * xfit + coeffs[1]
            ax.plot(xfit, yfit, color='black', linewidth=1.1, linestyle='--')
        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.axvline(0.0, color='gray', linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Delta frontal FC topology volatility')
        ax.set_title(f'Spearman r={r:.3f}, p={p:.4f}')
    fig.suptitle('Primary branch delta coupling', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_group_frontal_fc_network(input_dir, lo, hi, region_channels, output_path, region_name=PRIMARY_REGION):
    if region_name not in region_channels:
        return
    channels = list(region_channels[region_name])
    matrices = {'lo': [], 'hi': []}
    for sid in sorted(lo.index.intersection(hi.index)):
        for condition in ['lo', 'hi']:
            path = input_dir / 'per_subject_raw' / f'sub{int(sid):02d}_{condition}.json'
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            rep = data.get('fc_regions', {}).get(region_name, {}).get('representative_window')
            if rep is None:
                continue
            sim = np.asarray(rep.get('similarity_matrix', []), dtype=float)
            if sim.shape == (len(channels), len(channels)):
                matrices[condition].append(sim)
    if len(matrices['lo']) < 5 or len(matrices['hi']) < 5:
        return
    rest = np.nanmedian(np.stack(matrices['lo'], axis=0), axis=0)
    task = np.nanmedian(np.stack(matrices['hi'], axis=0), axis=0)
    diff = rest - task
    np.fill_diagonal(rest, np.nan)
    np.fill_diagonal(task, np.nan)
    np.fill_diagonal(diff, 0.0)
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])]
    matrix_specs = [(rest, 'Rest / low-demand frontal alpha FC', 'viridis', 0.0, 1.0), (task, 'Task / high-demand frontal alpha FC', 'viridis', 0.0, 1.0), (diff, 'Rest - task FC difference', 'RdBu_r', -np.nanmax(np.abs(diff)), np.nanmax(np.abs(diff)))]
    for (ax, (mat, title, cmap, vmin, vmax)) in zip(axes[:3], matrix_specs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(channels)))
        ax.set_yticks(np.arange(len(channels)))
        ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(channels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[3]
    head = plt.Circle((0.5, 0.5), 0.48, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(head)
    ax.fill([0.46, 0.54, 0.5], [0.98, 0.98, 1.03], color='black')
    ax.add_patch(plt.Circle((0.02, 0.5), 0.025, color='black'))
    ax.add_patch(plt.Circle((0.98, 0.5), 0.025, color='black'))
    edge_values = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            edge_values.append((abs(diff[i, j]), diff[i, j], channels[i], channels[j]))
    edge_values = sorted(edge_values, reverse=True)[:12]
    max_abs = max([v[0] for v in edge_values] + [1e-09])
    for (abs_val, signed_val, ch_a, ch_b) in edge_values:
        (x1, y1) = HEAD_COORDS[ch_a]
        (x2, y2) = HEAD_COORDS[ch_b]
        color = REST_COLOR if signed_val > 0 else TASK_COLOR
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.35 + 0.55 * abs_val / max_abs, linewidth=0.8 + 3.0 * abs_val / max_abs)
    for ch in channels:
        (x, y) = HEAD_COORDS[ch]
        ax.add_patch(plt.Circle((x, y), 0.04, color='white', ec='black', linewidth=1.1, zorder=3))
        ax.text(x, y, ch, ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)
    ax.text(0.05, -0.02, 'Blue edges: stronger in rest/low-demand', color=REST_COLOR, fontsize=9)
    ax.text(0.55, -0.02, 'Red edges: stronger in task/high-demand', color=TASK_COLOR, fontsize=9)
    ax.set_title('Group-level frontal alpha-FC network difference', fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle('Brain-level visualization of the primary topology branch', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_discovery_signal_table(output_path, primary_branch_df, fc_region_dynamic, confirmatory_topo, confirmatory_landscape, confirmatory_surrogate, confirmatory_dynamic):
    rows = []

    def add_rows(df, priority, analysis_family, note, max_rows=None):
        if df is None or df.empty or 'Target' not in df.columns:
            return
        required = ['Target', 'RBC_signed', 'p_FDR', 'Sig', 'Direction', 'Rest_median', 'Task_median']
        if not all((col in df.columns for col in required)):
            return
        tmp = df.copy()
        tmp['_abs_rbc'] = tmp['RBC_signed'].astype(float).abs()
        tmp = tmp.sort_values(['p_FDR', '_abs_rbc'], ascending=[True, False])
        if max_rows is not None:
            tmp = tmp.head(max_rows)
        for (_, row) in tmp.iterrows():
            rows.append({'Priority': priority, 'Analysis_family': analysis_family, 'Target': row['Target'], 'Direction': row['Direction'], 'RBC_signed': row['RBC_signed'], 'p_FDR': row['p_FDR'], 'Sig': row['Sig'], 'Rest_median': row['Rest_median'], 'Task_median': row['Task_median'], 'Interpretation_note': note})
    add_rows(primary_branch_df[primary_branch_df['Target'] == 'Frontal dynamic FC topology'] if not primary_branch_df.empty else primary_branch_df, 'Primary', 'Frontal dynamic alpha-FC topology', 'Main finding: dynamic topology of frontal alpha connectivity changes across cognitive state.')
    add_rows(fc_region_dynamic, 'Primary-supporting', 'Regional FC topology dynamics', 'Checks whether the dynamic FC topology effect is frontal, posterior, or whole-head.')
    add_rows(confirmatory_topo, 'Supporting', 'Single-channel persistent homology', 'Frontal electrode-level topology; useful as converging evidence, not the headline.', max_rows=6)
    add_rows(confirmatory_landscape, 'Supporting', 'Persistence landscapes', 'Richer topological summaries from H1 diagrams.', max_rows=6)
    add_rows(confirmatory_surrogate, 'Supporting-control', 'IAAFT surrogate-normalized topology', 'Tests whether frontal topology remains after spectrum/amplitude-preserving surrogate controls.', max_rows=6)
    add_rows(confirmatory_dynamic, 'Exploratory', 'Single-channel topology dynamics', 'Secondary dynamic single-channel evidence.', max_rows=6)
    discovery = pd.DataFrame(rows)
    discovery.to_csv(output_path, index=False)
    return discovery

def load_representative_fc_matrix(input_dir, subject_id, condition, region_name, n_channels):
    path = input_dir / 'per_subject_raw' / f'sub{int(subject_id):02d}_{condition}.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    rep = data.get('fc_regions', {}).get(region_name, {}).get('representative_window')
    if rep is None:
        return None
    sim = np.asarray(rep.get('similarity_matrix', []), dtype=float)
    if sim.shape != (n_channels, n_channels):
        return None
    return sim

def build_frontal_edge_driver_table(input_dir, lo, hi, region_channels, output_path, region_name=PRIMARY_REGION):
    if region_name not in region_channels or PRIMARY_FEATURE not in lo.columns or PRIMARY_FEATURE not in hi.columns:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    channels = list(region_channels[region_name])
    common = sorted(lo.index.intersection(hi.index).tolist())
    rows = []
    pvals = []
    topology_delta = (lo.loc[common, PRIMARY_FEATURE] - hi.loc[common, PRIMARY_FEATURE]).dropna()
    for (i, ch_a) in enumerate(channels):
        for j in range(i + 1, len(channels)):
            ch_b = channels[j]
            rest_vals = []
            task_vals = []
            valid_subjects = []
            for sid in common:
                rest = load_representative_fc_matrix(input_dir, sid, 'lo', region_name, len(channels))
                task = load_representative_fc_matrix(input_dir, sid, 'hi', region_name, len(channels))
                if rest is None or task is None:
                    continue
                rest_vals.append(rest[i, j])
                task_vals.append(task[i, j])
                valid_subjects.append(sid)
            if len(valid_subjects) < 5:
                continue
            rest_vals = np.asarray(rest_vals, dtype=float)
            task_vals = np.asarray(task_vals, dtype=float)
            try:
                (w_stat, p_raw) = wilcoxon(rest_vals, task_vals, alternative='two-sided')
                rbc = signed_rbc(w_stat, len(valid_subjects), np.median(rest_vals), np.median(task_vals))
            except Exception:
                w_stat = np.nan
                p_raw = np.nan
                rbc = 0.0
            edge_delta = pd.Series(rest_vals - task_vals, index=valid_subjects)
            coupling_idx = topology_delta.index.intersection(edge_delta.index)
            if len(coupling_idx) >= 5 and float(np.ptp(edge_delta.loc[coupling_idx].values)) > 0:
                (rho, rho_p) = spearmanr(topology_delta.loc[coupling_idx].values, edge_delta.loc[coupling_idx].values)
            else:
                (rho, rho_p) = (np.nan, np.nan)
            row = {'Region': region_name, 'Edge': f'{ch_a}-{ch_b}', 'Channel_A': ch_a, 'Channel_B': ch_b, 'N_pairs': len(valid_subjects), 'Rest_median_similarity': float(np.median(rest_vals)), 'Task_median_similarity': float(np.median(task_vals)), 'Rest_minus_task_median': float(np.median(rest_vals - task_vals)), 'W': float(w_stat) if np.isfinite(w_stat) else np.nan, 'p_raw_edge': float(p_raw) if np.isfinite(p_raw) else np.nan, 'RBC_signed_edge': float(rbc), 'Edge_direction': 'Rest > Task' if rbc > 0 else 'Task > Rest', 'rho_edge_delta_vs_topology_delta': float(rho) if np.isfinite(rho) else np.nan, 'p_edge_delta_vs_topology_delta': float(rho_p) if np.isfinite(rho_p) else np.nan}
            rows.append(row)
            pvals.append(p_raw if np.isfinite(p_raw) else 1.0)
    if not rows:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    p_fdr = fdr_bh(pvals)
    for (row, adj) in zip(rows, p_fdr):
        row['p_FDR_edge'] = float(adj)
        row['Sig_edge'] = sig_label(adj)
        edge_strength = abs(row['RBC_signed_edge'])
        coupling_strength = abs(row['rho_edge_delta_vs_topology_delta']) if np.isfinite(row['rho_edge_delta_vs_topology_delta']) else 0.0
        row['Topology_driver_score'] = float(edge_strength * (0.5 + coupling_strength))
        if row['p_FDR_edge'] < 0.1 and coupling_strength >= 0.25:
            row['Discovery_role'] = 'candidate_topology_coupled_edge'
        elif row['p_FDR_edge'] < 0.1:
            row['Discovery_role'] = 'candidate_state_sensitive_edge'
        elif coupling_strength >= 0.3:
            row['Discovery_role'] = 'candidate_topology_coupled_without_edge_significance'
        else:
            row['Discovery_role'] = 'low_priority'
    edge_df = pd.DataFrame(rows).sort_values(['Discovery_role', 'Topology_driver_score', 'p_FDR_edge'], ascending=[True, False, True])
    edge_df.to_csv(output_path, index=False)
    return edge_df

def plot_frontal_edge_drivers(edge_df, output_path, top_n=10):
    if edge_df is None or edge_df.empty:
        return
    plot_df = edge_df.copy()
    plot_df = plot_df.sort_values('Topology_driver_score', ascending=False).head(top_n)
    if plot_df.empty:
        return
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 5))
    labels = plot_df['Edge'].tolist()
    y = np.arange(len(labels))
    colors = [REST_COLOR if v > 0 else TASK_COLOR for v in plot_df['RBC_signed_edge']]
    axes[0].barh(y, plot_df['RBC_signed_edge'].astype(float), color=colors, alpha=0.85)
    axes[0].axvline(0.0, color='black', linewidth=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Signed RBC for edge similarity')
    axes[0].set_title('State-sensitive frontal edges')
    head = plt.Circle((0.5, 0.5), 0.48, fill=False, edgecolor='black', linewidth=1.5)
    axes[1].add_patch(head)
    axes[1].fill([0.46, 0.54, 0.5], [0.98, 0.98, 1.03], color='black')
    axes[1].add_patch(plt.Circle((0.02, 0.5), 0.025, color='black'))
    axes[1].add_patch(plt.Circle((0.98, 0.5), 0.025, color='black'))
    max_score = max(float(plot_df['Topology_driver_score'].max()), 1e-09)
    for (_, row) in plot_df.iterrows():
        ch_a = row['Channel_A']
        ch_b = row['Channel_B']
        (x1, y1) = HEAD_COORDS[ch_a]
        (x2, y2) = HEAD_COORDS[ch_b]
        color = REST_COLOR if row['RBC_signed_edge'] > 0 else TASK_COLOR
        width = 0.8 + 4.0 * float(row['Topology_driver_score']) / max_score
        axes[1].plot([x1, x2], [y1, y2], color=color, alpha=0.55, linewidth=width)
    for ch in sorted(set(plot_df['Channel_A']).union(set(plot_df['Channel_B']))):
        (x, y0) = HEAD_COORDS[ch]
        axes[1].add_patch(plt.Circle((x, y0), 0.04, color='white', ec='black', linewidth=1.1, zorder=3))
        axes[1].text(x, y0, ch, ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)
    axes[1].text(0.05, -0.02, 'Blue: stronger in rest/low-demand', color=REST_COLOR, fontsize=9)
    axes[1].text(0.55, -0.02, 'Red: stronger in task/high-demand', color=TASK_COLOR, fontsize=9)
    axes[1].set_title('Candidate topology-driving frontal edges')
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.08, 1.05)
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    fig.suptitle('Discovery analysis: frontal FC edges linked to topology change', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def build_primary_branch_comparison(lo, hi, output_path, n_perm=1000, n_boot=2000, rng=None):
    targets = [('Frontal dynamic FC topology', PRIMARY_FEATURE), ('Frontal alpha power', PRIMARY_ALPHA_FEATURE), ('Frontal alpha FC strength', PRIMARY_FC_STRENGTH_FEATURE)]
    df = run_paired_tests(lo, hi, targets, 'primary_branch_comparison', 'primary_branch', n_perm=n_perm, n_boot=n_boot, rng=rng)
    df.to_csv(output_path, index=False)
    return df

def build_primary_control_table(features_ok, lo, hi, output_path):
    rows = []
    all_data = features_ok.copy()
    all_data['condition_binary'] = (all_data['condition'] == 'hi').astype(float)
    needed = [PRIMARY_FEATURE, PRIMARY_ALPHA_FEATURE, PRIMARY_FC_STRENGTH_FEATURE]
    if not all((col in all_data.columns for col in needed)):
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    valid = all_data.dropna(subset=needed)
    if len(valid) >= 10:
        (r_raw, p_raw) = spearmanr(valid[PRIMARY_FEATURE], valid['condition_binary'])
        (r_partial_alpha, p_partial_alpha) = partial_spearman(valid[PRIMARY_FEATURE], valid['condition_binary'], valid[PRIMARY_ALPHA_FEATURE])
        (r_partial_fc, p_partial_fc) = partial_spearman(valid[PRIMARY_FEATURE], valid['condition_binary'], valid[PRIMARY_FC_STRENGTH_FEATURE])
        (r_partial_both, p_partial_both) = partial_spearman(valid[PRIMARY_FEATURE], valid['condition_binary'], 0.5 * (valid[PRIMARY_ALPHA_FEATURE].rank() + valid[PRIMARY_FC_STRENGTH_FEATURE].rank()))
        rows.append({'Analysis': 'cross_condition_association', 'N': len(valid), 'r_primary_vs_condition': round(r_raw, 3), 'p_primary_vs_condition': round(p_raw, 4), 'r_partial_condition_given_frontal_alpha': round(r_partial_alpha, 3), 'p_partial_condition_given_frontal_alpha': round(p_partial_alpha, 4), 'r_partial_condition_given_fc_strength': round(r_partial_fc, 3), 'p_partial_condition_given_fc_strength': round(p_partial_fc, 4), 'r_partial_condition_given_alpha_plus_fc': round(r_partial_both, 3), 'p_partial_condition_given_alpha_plus_fc': round(p_partial_both, 4)})
    common = lo.index.intersection(hi.index)
    if len(common) >= 5:
        delta_primary = hi.loc[common, PRIMARY_FEATURE] - lo.loc[common, PRIMARY_FEATURE]
        delta_alpha = hi.loc[common, PRIMARY_ALPHA_FEATURE] - lo.loc[common, PRIMARY_ALPHA_FEATURE]
        delta_fc_strength = hi.loc[common, PRIMARY_FC_STRENGTH_FEATURE] - lo.loc[common, PRIMARY_FC_STRENGTH_FEATURE]
        valid_delta = pd.DataFrame({'delta_primary': delta_primary, 'delta_alpha': delta_alpha, 'delta_fc_strength': delta_fc_strength}).dropna()
        if len(valid_delta) >= 5:
            (r_da, p_da) = spearmanr(valid_delta['delta_primary'], valid_delta['delta_alpha'])
            (r_dfc, p_dfc) = spearmanr(valid_delta['delta_primary'], valid_delta['delta_fc_strength'])
            rows.append({'Analysis': 'within_subject_delta_coupling', 'N': len(valid_delta), 'r_delta_primary_vs_delta_alpha': round(r_da, 3), 'p_delta_primary_vs_delta_alpha': round(p_da, 4), 'r_delta_primary_vs_delta_fc_strength': round(r_dfc, 3), 'p_delta_primary_vs_delta_fc_strength': round(p_dfc, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

def build_primary_exclusion_sensitivity(features, preproc, output_path):
    scenarios = []
    preproc_ok = preproc.copy()
    failed_subjects = sorted(preproc_ok[preproc_ok['error'].fillna('').astype(str).str.len() > 0]['subject_id'].unique().tolist())
    fallback_subjects = sorted(preproc_ok[preproc_ok['asr_status'].fillna('').astype(str).str.contains('trimmed', case=False)]['subject_id'].unique().tolist())
    post_max_by_subject = preproc_ok[preproc_ok['error'].fillna('').astype(str).str.len() == 0].groupby('subject_id')['post_pct_samples_abs_gt_100uv'].max()
    q1 = float(post_max_by_subject.quantile(0.25))
    q3 = float(post_max_by_subject.quantile(0.75))
    iqr = q3 - q1
    residual_outliers = sorted(post_max_by_subject[post_max_by_subject > float(q3 + 3 * iqr)].index.tolist())
    scenario_defs = [('full_qc_exclusions', sorted(set(failed_subjects) | set(residual_outliers))), ('failed_only', failed_subjects), ('failed_plus_fallback_asr', sorted(set(failed_subjects) | set(fallback_subjects))), ('no_exclusions', [])]
    for (label, excluded) in scenario_defs:
        feat_ok = features[~features['subject_id'].isin(excluded)].copy()
        lo = feat_ok[feat_ok['condition'] == 'lo'].set_index('subject_id')
        hi = feat_ok[feat_ok['condition'] == 'hi'].set_index('subject_id')
        common = lo.index.intersection(hi.index)
        lo = lo.loc[common]
        hi = hi.loc[common]
        if PRIMARY_FEATURE not in lo.columns or PRIMARY_FEATURE not in hi.columns or len(common) < 5:
            continue
        rest_vals = lo[PRIMARY_FEATURE].dropna()
        task_vals = hi[PRIMARY_FEATURE].dropna()
        common2 = rest_vals.index.intersection(task_vals.index)
        if len(common2) < 5:
            continue
        rest_vals = rest_vals.loc[common2].values
        task_vals = task_vals.loc[common2].values
        (w_stat, p_raw) = wilcoxon(rest_vals, task_vals, alternative='two-sided')
        rbc = signed_rbc(w_stat, len(common2), np.median(rest_vals), np.median(task_vals))
        scenarios.append({'Scenario': label, 'Excluded_subjects': ','.join((str(x) for x in excluded)), 'N_pairs': len(common2), 'Rest_median': float(np.median(rest_vals)), 'Task_median': float(np.median(task_vals)), 'RBC_signed': float(rbc), 'p_raw': float(p_raw)})
    df = pd.DataFrame(scenarios)
    df.to_csv(output_path, index=False)
    return df

def write_analysis_snapshot(config, args, input_dir, output_dir, n_paired, excluded_subjects):
    snapshot = {'input_dir': str(input_dir), 'output_dir': str(output_dir), 'paired_n': int(n_paired), 'excluded_subjects': [int(x) for x in excluded_subjects], 'normalization_mode': config.get('normalization_mode', 'unknown'), 'window_seconds': config.get('window_seconds'), 'window_samples': config.get('window_samples'), 'surrogate_n_per_method': config.get('surrogate_n_per_method'), 'surrogate_iaaft_iterations': config.get('surrogate_iaaft_iterations'), 'fc_connectivity_metric': config.get('fc_connectivity_metric'), 'primary_region': PRIMARY_REGION, 'primary_feature': PRIMARY_FEATURE, 'primary_alpha_feature': PRIMARY_ALPHA_FEATURE, 'primary_fc_strength_feature': PRIMARY_FC_STRENGTH_FEATURE, 'wilcoxon_perm': int(args.wilcoxon_perm), 'bootstrap': int(args.bootstrap), 'clf_perm': int(args.clf_perm), 'ml_perm': int(args.ml_perm)}
    with open(output_dir / 'analysis_config_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description='STEW topology analysis v3')
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--wilcoxon-perm', type=int, default=5000)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--clf-perm', type=int, default=2000)
    parser.add_argument('--ml-perm', type=int, default=500)
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(input_dir / 'subject_level_features.csv')
    preproc = pd.read_csv(input_dir / 'preprocessing_log.csv')
    embed = pd.read_csv(input_dir / 'embedding_params.csv')
    with open(input_dir / 'pipeline_config.json') as f:
        config = json.load(f)
    region_channels = config['region_channels']
    features = add_region_features(features, region_channels)
    stage2 = build_stage2_audit(features, preproc, config, output_dir)
    excluded_subjects = stage2['excluded_subjects']
    features_ok = features[~features['subject_id'].isin(excluded_subjects)].copy()
    lo = features_ok[features_ok['condition'] == 'lo'].set_index('subject_id')
    hi = features_ok[features_ok['condition'] == 'hi'].set_index('subject_id')
    common_subs = lo.index.intersection(hi.index)
    lo = lo.loc[common_subs].copy()
    hi = hi.loc[common_subs].copy()
    n_paired = len(common_subs)
    write_analysis_snapshot(config, args, input_dir, output_dir, n_paired, excluded_subjects)
    rng = np.random.default_rng(42)
    confirmatory_targets = [(ch, f'{ch}_{TOPO_FEATURE}') for ch in config['confirmatory_frontal_channels']]
    whole_head_targets = [(ch, f'{ch}_{TOPO_FEATURE}') for ch in config['channels']]
    region_targets = [(name, f'REGION_{name}_{TOPO_FEATURE}') for name in region_channels]
    confirmatory_landscape_targets = [(ch, f'{ch}_{LANDSCAPE_FEATURE}') for ch in config['confirmatory_frontal_channels']]
    whole_head_landscape_targets = [(ch, f'{ch}_{LANDSCAPE_FEATURE}') for ch in config['channels']]
    region_landscape_targets = [(name, f'REGION_{name}_{LANDSCAPE_FEATURE}') for name in region_channels]
    confirmatory_image_targets = [(ch, f'{ch}_{IMAGE_FEATURE}') for ch in config['confirmatory_frontal_channels']]
    region_image_targets = [(name, f'REGION_{name}_{IMAGE_FEATURE}') for name in region_channels]
    confirmatory_surrogate_targets = [(ch, f'{ch}_{SURROGATE_FEATURE}') for ch in config['confirmatory_frontal_channels']]
    region_surrogate_targets = [(name, f'REGION_{name}_{SURROGATE_FEATURE}') for name in region_channels]
    confirmatory_dynamic_targets = [(ch, f'{ch}_{DYNAMIC_LANDSCAPE_FEATURE}') for ch in config['confirmatory_frontal_channels']]
    fc_region_channels = config.get('fc_region_channels', {})
    fc_region_targets = [(name, f'FCREGION_{name}_{FC_TOPO_FEATURE}') for name in fc_region_channels]
    fc_dynamic_targets = [(name, f'FCREGION_{name}_{FC_DYNAMIC_FEATURE}') for name in fc_region_channels]
    lateralization_targets = [('F_right_minus_left', LATERALIZATION_FEATURE)]
    confirmatory_topo = run_paired_tests(lo, hi, confirmatory_targets, TOPO_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    whole_head_topo = run_paired_tests(lo, hi, whole_head_targets, TOPO_FEATURE, 'whole_head', args.wilcoxon_perm, args.bootstrap, rng)
    region_topo = run_paired_tests(lo, hi, region_targets, TOPO_FEATURE, 'regions', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_landscape = run_paired_tests(lo, hi, confirmatory_landscape_targets, LANDSCAPE_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    whole_head_landscape = run_paired_tests(lo, hi, whole_head_landscape_targets, LANDSCAPE_FEATURE, 'whole_head', args.wilcoxon_perm, args.bootstrap, rng)
    region_landscape = run_paired_tests(lo, hi, region_landscape_targets, LANDSCAPE_FEATURE, 'regions', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_image = run_paired_tests(lo, hi, confirmatory_image_targets, IMAGE_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    region_image = run_paired_tests(lo, hi, region_image_targets, IMAGE_FEATURE, 'regions', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_surrogate = run_paired_tests(lo, hi, confirmatory_surrogate_targets, SURROGATE_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    region_surrogate = run_paired_tests(lo, hi, region_surrogate_targets, SURROGATE_FEATURE, 'regions', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_dynamic = run_paired_tests(lo, hi, confirmatory_dynamic_targets, DYNAMIC_LANDSCAPE_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    fc_region_topology = run_paired_tests(lo, hi, fc_region_targets, FC_TOPO_FEATURE, 'fc_regions', args.wilcoxon_perm, args.bootstrap, rng)
    fc_region_dynamic = run_paired_tests(lo, hi, fc_dynamic_targets, FC_DYNAMIC_FEATURE, 'fc_regions', args.wilcoxon_perm, args.bootstrap, rng)
    lateralization_df = run_paired_tests(lo, hi, lateralization_targets, LATERALIZATION_FEATURE, 'lateralization', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_alpha = run_paired_tests(lo, hi, [(ch, f'{ch}_{ALPHA_FEATURE}') for ch in config['confirmatory_frontal_channels']], ALPHA_FEATURE, 'confirmatory_frontal', args.wilcoxon_perm, args.bootstrap, rng)
    region_alpha = run_paired_tests(lo, hi, [(name, f'REGION_{name}_{ALPHA_FEATURE}') for name in region_channels], ALPHA_FEATURE, 'regions', args.wilcoxon_perm, args.bootstrap, rng)
    confirmatory_topo.to_csv(output_dir / 'Table1_ConfirmatoryFrontal_Topology.csv', index=False)
    whole_head_topo.to_csv(output_dir / 'Table2_WholeHead_Topology.csv', index=False)
    region_topo.to_csv(output_dir / 'Table3_RegionalSummaries_Topology.csv', index=False)
    confirmatory_alpha.to_csv(output_dir / 'Table4_ConfirmatoryFrontal_AlphaPower.csv', index=False)
    region_alpha.to_csv(output_dir / 'Table5_RegionalSummaries_AlphaPower.csv', index=False)
    confirmatory_landscape.to_csv(output_dir / 'Table9_ConfirmatoryFrontal_Landscape.csv', index=False)
    whole_head_landscape.to_csv(output_dir / 'Table10_WholeHead_Landscape.csv', index=False)
    region_landscape.to_csv(output_dir / 'Table11_RegionalSummaries_Landscape.csv', index=False)
    confirmatory_image.to_csv(output_dir / 'Table12_ConfirmatoryFrontal_PersistenceImage.csv', index=False)
    region_image.to_csv(output_dir / 'Table13_RegionalSummaries_PersistenceImage.csv', index=False)
    confirmatory_surrogate.to_csv(output_dir / 'Table14_ConfirmatoryFrontal_IAAFTSurrogateLandscapeZ.csv', index=False)
    region_surrogate.to_csv(output_dir / 'Table15_RegionalSummaries_IAAFTSurrogateLandscapeZ.csv', index=False)
    confirmatory_dynamic.to_csv(output_dir / 'Table16_ConfirmatoryFrontal_DynamicLandscapeVolatility.csv', index=False)
    fc_region_topology.to_csv(output_dir / 'Table17_RegionAlphaFCTopology.csv', index=False)
    fc_region_dynamic.to_csv(output_dir / 'Table18_RegionAlphaFCDynamicLandscapeVolatility.csv', index=False)
    lateralization_df.to_csv(output_dir / 'Table19_FrontalLateralization_Landscape.csv', index=False)
    primary_branch_df = build_primary_branch_comparison(lo, hi, output_dir / 'Table20_PrimaryBranchComparison.csv', n_perm=args.wilcoxon_perm, n_boot=args.bootstrap, rng=rng)
    primary_controls_df = build_primary_control_table(features_ok, lo, hi, output_dir / 'Table21_PrimaryMetric_Controls.csv')
    primary_exclusion_df = build_primary_exclusion_sensitivity(features, preproc, output_dir / 'Table22_PrimaryMetric_ExclusionSensitivity.csv')
    discovery_df = build_discovery_signal_table(output_dir / 'Table23_DiscoverySignalRanking.csv', primary_branch_df, fc_region_dynamic, confirmatory_topo, confirmatory_landscape, confirmatory_surrogate, confirmatory_dynamic)
    edge_driver_df = build_frontal_edge_driver_table(input_dir, lo, hi, config.get('fc_region_channels', {}), output_dir / 'Table24_FrontalEdgeTopologyDrivers.csv', region_name=PRIMARY_REGION)
    subjective_corr_df = build_subjective_rating_correlations(lo, hi, config, output_dir / 'Table28_SubjectiveRatingCorrelations.csv')
    node_driver_df = build_network_node_driver_summary(edge_driver_df, output_dir / 'Table30_NetworkNodeDriverSummary.csv')
    embed_ok = embed[~embed['subject_id'].isin(excluded_subjects)].copy()
    embed_summary = pd.DataFrame([{'Condition': condition, 'tau_global_mean': group['tau_global_median'].mean(), 'tau_global_std': group['tau_global_median'].std(), 'd_global_mean': group['d_global_median'].mean(), 'd_global_std': group['d_global_median'].std()} for (condition, group) in embed_ok.groupby('condition')])
    embed_summary.to_csv(output_dir / 'Table6_EmbeddingSummary.csv', index=False)
    partial_rows = []
    all_data = features_ok.copy()
    all_data['condition_binary'] = (all_data['condition'] == 'hi').astype(float)
    for ch in config['channels']:
        topo_col = f'{ch}_{TOPO_FEATURE}'
        alpha_col = f'{ch}_{ALPHA_FEATURE}'
        valid = all_data.dropna(subset=[topo_col, alpha_col])
        if len(valid) < 10:
            continue
        (r_raw, p_raw) = spearmanr(valid[topo_col], valid['condition_binary'])
        (r_partial, p_partial) = partial_spearman(valid[topo_col], valid['condition_binary'], valid[alpha_col])
        (topo_alpha_r, topo_alpha_p) = spearmanr(valid[topo_col], valid[alpha_col])
        partial_rows.append({'Channel': ch, 'N': len(valid), 'r_topology_vs_condition': round(r_raw, 3), 'p_topology_vs_condition': round(p_raw, 4), 'r_partial_condition_given_alpha': round(r_partial, 3), 'p_partial_condition_given_alpha': round(p_partial, 4), 'r_topology_vs_alpha': round(topo_alpha_r, 3), 'p_topology_vs_alpha': round(topo_alpha_p, 4), 'Sig_partial': sig_label(p_partial)})
    partial_df = pd.DataFrame(partial_rows)
    partial_df.to_csv(output_dir / 'Table7_TopologyControllingAlpha.csv', index=False)
    scopes = OrderedDict([('confirmatory_frontal', config['confirmatory_frontal_channels']), ('whole_head', config['channels'])])
    (clf_df, _) = build_classifier_table(lo, hi, scopes, rng, n_perm=args.clf_perm)
    clf_df.to_csv(output_dir / 'Table8_Classification.csv', index=False)
    cross_evidence_df = build_cross_analysis_evidence_map(lo, hi, config, output_dir / 'Table25_CrossAnalysisEvidenceMap.csv')
    (ml_ablation_df, ml_importance_df) = build_ml_feature_family_ablation(lo, hi, scopes, rng, output_dir / 'Table26_MLFeatureFamilyAblation.csv', output_dir / 'Table27_MLFeatureImportanceStability.csv', n_perm=args.ml_perm)
    vector_ml_df = build_topology_vector_ml(input_dir, lo, hi, config, rng, output_dir / 'Table29_MLTopologyVectorModels.csv', n_perm=args.ml_perm)
    if not confirmatory_topo.empty:
        plot_confirmatory_raincloud(lo, hi, confirmatory_topo, output_dir / 'Figure1_ConfirmatoryRaincloud.png')
    if not whole_head_topo.empty:
        plot_topomap_all14(whole_head_topo, output_dir / 'Figure2_All14Topography.png')
    if not region_topo.empty and (not region_alpha.empty):
        plot_region_bars(region_topo, region_alpha, output_dir / 'Figure3_RegionalBars.png')
    if not whole_head_topo.empty and 'Target' in whole_head_topo.columns:
        representative_channel = whole_head_topo.sort_values(['p_FDR', 'p_raw', 'RBC_signed'], ascending=[True, True, False]).iloc[0]['Target']
        rep_col = f'{representative_channel}_{TOPO_FEATURE}'
        deltas = (hi.loc[common_subs, rep_col] - lo.loc[common_subs, rep_col]).dropna()
        representative_subject = int((deltas - deltas.median()).abs().sort_values().index[0])
        plot_representative_embeddings(representative_subject, representative_channel, input_dir, output_dir / 'Figure4_RepresentativeGeometry.png')
        plot_representative_representations(representative_subject, representative_channel, input_dir, output_dir / 'Figure6_RepresentativeRepresentations.png')
        plot_representative_circular_coordinates(representative_subject, representative_channel, input_dir, output_dir / 'Figure8_RepresentativeCircularCoordinates.png')
    if PRIMARY_REGION in fc_region_channels:
        rep_fc_region = PRIMARY_REGION
        rep_fc_col = f'FCREGION_{rep_fc_region}_{FC_DYNAMIC_FEATURE}'
        rep_fc_deltas = (hi.loc[common_subs, rep_fc_col] - lo.loc[common_subs, rep_fc_col]).dropna()
        if len(rep_fc_deltas) > 0:
            rep_fc_subject = int((rep_fc_deltas - rep_fc_deltas.median()).abs().sort_values().index[0])
            plot_fc_region_representative(rep_fc_subject, rep_fc_region, input_dir, output_dir / 'Figure7_RepresentativeAlphaFCTopology.png')
    if not clf_df.empty:
        plot_classification(clf_df, output_dir / 'Figure5_Classification.png')
    plot_cross_analysis_evidence_map(cross_evidence_df, output_dir / 'Figure13_CrossAnalysisEvidenceMap.png')
    plot_ml_feature_family_ablation(ml_ablation_df, output_dir / 'Figure14_MLFeatureFamilyAblation.png')
    plot_subjective_correlation(lo, hi, output_dir / 'Figure15_SubjectiveTopologyCorrelation.png')
    plot_topology_vector_ml(vector_ml_df, output_dir / 'Figure16_MLTopologyVectorModels.png')
    plot_network_node_driver_summary(node_driver_df, output_dir / 'Figure17_NetworkNodeDriverSummary.png')
    if PRIMARY_FEATURE in lo.columns and PRIMARY_FEATURE in hi.columns:
        plot_primary_metric(lo, hi, PRIMARY_FEATURE, output_dir / 'Figure9_PrimaryFrontalFCDynamic.png', 'Primary result: frontal dynamic alpha-FC topology')
    if all((col in lo.columns and col in hi.columns for col in [PRIMARY_FEATURE, PRIMARY_ALPHA_FEATURE, PRIMARY_FC_STRENGTH_FEATURE])):
        plot_primary_delta_coupling(lo, hi, output_dir / 'Figure10_PrimaryDeltaCoupling.png')
    plot_group_frontal_fc_network(input_dir, lo, hi, config.get('fc_region_channels', {}), output_dir / 'Figure11_GroupFrontalFCNetwork.png', region_name=PRIMARY_REGION)
    plot_frontal_edge_drivers(edge_driver_df, output_dir / 'Figure12_FrontalEdgeTopologyDrivers.png')
    primary_row = pd.DataFrame()
    if not primary_branch_df.empty:
        primary_row = primary_branch_df[primary_branch_df['Target'] == 'Frontal dynamic FC topology']
    summary_lines = ['STEW TOPOLOGY ANALYSIS V3', f'Paired N after Stage 2 exclusions: {n_paired}', f'Excluded subjects: {excluded_subjects}', f"Normalization mode: {config.get('normalization_mode', 'unknown')}", f"Window length (s): {config.get('window_seconds', 'unknown')}", f"FC connectivity metric: {config.get('fc_connectivity_metric', 'unknown')}", f"IAAFT / phase surrogate count per method: {config.get('surrogate_n_per_method', 'unknown')}", '', 'PRIMARY CONFIRMATORY RESULT']
    if not primary_row.empty:
        row = primary_row.iloc[0]
        summary_lines.extend(['Frontal dynamic alpha-FC topology is the primary branch.', f"Frontal FC landscape volatility: RBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}, p_perm={row['p_perm']:.4f}", f"Rest median={row['Rest_median']:.4f}, Task median={row['Task_median']:.4f}", f"95% bootstrap RBC CI=[{row['RBC_CI_lower']:.3f}, {row['RBC_CI_upper']:.3f}]"])
    else:
        summary_lines.append('Primary frontal dynamic FC result was not available in this run.')
    if not whole_head_topo.empty and 'Target' in whole_head_topo.columns:
        best = whole_head_topo.sort_values(['p_FDR', 'p_raw']).iloc[0]
        summary_lines.extend(['', 'SUPPORTING SINGLE-CHANNEL TOPOLOGY', f"{best['Target']}: RBC={best['RBC_signed']:.3f}, p_FDR={best['p_FDR']:.4g} {best['Sig']}, p_perm={best['p_perm']:.4f}"])
    summary_lines.extend(['', 'STAGE 3 CHECKS', '- Topology compared directly against conventional alpha power.', '- Partial correlation controlling alpha power computed across all 14 channels.', '- Representative embedded point clouds and persistence diagrams generated for the strongest current channel.', '- Persistence landscapes and persistence images generated from H1 diagrams and summarized at subject level.', '- IAAFT and phase-randomized surrogate controls computed window-by-window to test topology beyond amplitude/spectrum structure.', '- Dynamic volatility features computed from windowwise landscape trajectories.', '- Alpha functional-connectivity topology computed by applying persistent homology to windowed FC distance matrices.', '- Persistent cocycle-derived circular coordinates retained as exploratory phase-geometry outputs only.', '- Classification kept as secondary validation only.'])
    if not confirmatory_landscape.empty:
        row = confirmatory_landscape.sort_values(['p_FDR', 'p_raw', 'RBC_signed'], ascending=[True, True, False]).iloc[0]
        summary_lines.extend(['', 'WORK PACKAGE 1', f"Best confirmatory landscape channel ({row['Target']}): RBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}"])
    if not confirmatory_surrogate.empty:
        row = confirmatory_surrogate.sort_values(['p_FDR', 'p_raw', 'RBC_signed'], ascending=[True, True, False]).iloc[0]
        summary_lines.extend(['', 'WORK PACKAGE 2', f"Best surrogate-resistant channel ({row['Target']}): RBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}"])
    if not confirmatory_dynamic.empty:
        row = confirmatory_dynamic.sort_values(['p_FDR', 'p_raw', 'RBC_signed'], ascending=[True, True, False]).iloc[0]
        summary_lines.extend(['', 'WORK PACKAGE 3', f"Best single-channel dynamic landscape ({row['Target']}): RBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}"])
    if not fc_region_dynamic.empty:
        row = fc_region_dynamic.sort_values(['p_FDR', 'p_raw', 'RBC_signed'], ascending=[True, True, False]).iloc[0]
        summary_lines.extend(['', 'WORK PACKAGE 4', f"Best FC dynamic region ({row['Target']}): RBC={row['RBC_signed']:.3f}, p_FDR={row['p_FDR']:.4f} {row['Sig']}"])
    summary_lines.extend(['', 'WORK PACKAGE 5', 'Persistent cocycle-based circular coordinates saved as exploratory mechanistic illustrations, not as core confirmatory evidence.'])
    if not primary_controls_df.empty:
        summary_lines.extend(['', 'PRIMARY BRANCH CONTROL ANALYSES', 'Table21 reports direct comparisons of frontal FC topology against frontal alpha power and frontal FC strength,', 'including partial-rank controls and within-subject delta coupling checks.'])
    if not primary_exclusion_df.empty:
        summary_lines.extend(['', 'ROBUSTNESS', 'Table22 reports exclusion-sensitivity scenarios, including failed-only, fallback-ASR, and no-exclusion analyses.', 'Use repeated reruns across window length, FC metric, and surrogate count with stew_robustness.py for final sensitivity summaries.'])
    if not discovery_df.empty:
        summary_lines.extend(['', 'DISCOVERY VIEW', 'Table23 ranks primary, supporting, and exploratory topology findings so the manuscript can focus on a coherent brain-level pattern.', 'Figure11 visualizes group-level frontal alpha-FC structure for the primary topology branch.'])
    if not edge_driver_df.empty:
        top_edge = edge_driver_df.sort_values('Topology_driver_score', ascending=False).iloc[0]
        summary_lines.extend(['', 'EDGE-LEVEL TOPOLOGY DISCOVERY', f"Top candidate frontal edge driver: {top_edge['Edge']} (edge RBC={top_edge['RBC_signed_edge']:.3f}, topology-coupling rho={top_edge['rho_edge_delta_vs_topology_delta']:.3f}, role={top_edge['Discovery_role']}).", 'Table24 and Figure12 are designed to identify which frontal alpha connections may drive the topology-level effect.'])
    if not cross_evidence_df.empty:
        top_cross = cross_evidence_df.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False]).iloc[0]
        summary_lines.extend(['', 'CROSS-ANALYSIS EVIDENCE MAP', f"Strongest cross-analysis association with the primary topology delta: {top_cross['Feature_label']} (rho={top_cross['Spearman_rho_with_primary_delta']:.3f}, p_FDR={top_cross['p_FDR']:.4f}, {top_cross['Sig']}).", 'Table25 and Figure13 connect the primary FC-topology effect to channel, regional, alpha, and topology-family changes.'])
    if not ml_ablation_df.empty:
        best_ml = ml_ablation_df.sort_values(['Balanced_accuracy', 'ROC_AUC'], ascending=[False, False]).iloc[0]
        summary_lines.extend(['', 'ENHANCED MACHINE-LEARNING VALIDATION', f"Best feature-family model: {best_ml['Scope']} / {best_ml['Feature_family']} (balanced accuracy={best_ml['Balanced_accuracy']:.3f}, ROC-AUC={best_ml['ROC_AUC']:.3f}, perm p={best_ml['Balanced_perm_p']:.4f}, {best_ml['Sig']}).", 'Table26 reports balanced accuracy, ROC-AUC, sensitivity, specificity, and confusion counts.', 'Table27 reports linear-SVM feature importance stability across LOSO folds.'])
    if not subjective_corr_df.empty:
        top_subjective = subjective_corr_df.sort_values(['p_FDR', 'Abs_rho'], ascending=[True, False]).iloc[0]
        summary_lines.extend(['', 'SUBJECTIVE-RATING RELEVANCE', f"Strongest subjective-rating association: {top_subjective['Feature_label']} (rho={top_subjective['Spearman_rho']:.3f}, p_FDR={top_subjective['p_FDR']:.4f}, {top_subjective['Sig']}).", 'Table28 tests whether topology and EEG changes scale with subject-level rating changes.'])
    if not vector_ml_df.empty:
        best_vector = vector_ml_df.sort_values(['Balanced_accuracy', 'ROC_AUC'], ascending=[False, False]).iloc[0]
        summary_lines.extend(['', 'TOPOLOGY-VECTOR MACHINE LEARNING', f"Best vectorized topology representation: {best_vector['Vector_family']} (balanced accuracy={best_vector['Balanced_accuracy']:.3f}, ROC-AUC={best_vector['ROC_AUC']:.3f}, perm p={best_vector['Balanced_perm_p']:.4f}, {best_vector['Sig']}).", 'Table29 uses persistence landscape/image arrays with PCA inside LOSO cross-validation.'])
    if not node_driver_df.empty:
        top_node = node_driver_df.iloc[0]
        summary_lines.extend(['', 'NETWORK NODE-LEVEL DISCOVERY', f"Top topology-driver node summary: {top_node['Node']} (candidate edges={top_node['N_candidate_edges']:.0f}, max driver score={top_node['Max_driver_score']:.3f}).", 'Table30 and Figure17 summarize whether candidate edge drivers converge on specific frontal nodes.'])
    with open(output_dir / 'results_summary_v3.txt', 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
if __name__ == '__main__':
    main()
