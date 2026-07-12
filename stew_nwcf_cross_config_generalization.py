from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
CONFIGS = ('123plv', '254aec', '83aec', '84aec', '84corr', '84plv')
OUTPUT_PATTERNS = ('stew_output_{config}', 'stew_output_final{config}', 'stew_tda_results_final_{config}', 'stew_tda_results_final{config}')
SUBJECT_COLUMNS = ('subject', 'subject_id', 'sub', 'participant', 'id')
CONDITION_COLUMNS = ('condition', 'state', 'label', 'class', 'task', 'workload')
REST_TOKENS = {'lo', 'low', 'rest', 'resting', 'baseline', '0'}
TASK_TOKENS = {'hi', 'high', 'task', 'active', 'work', 'workload', '1'}
NWCF_NAME_TOKENS = ('nwcf', 'clique', 'corrobor', 'shrink', 'gamma3', 'gamma4', 'triangle', 'four', '4clique', 'death_compression')
EXCLUDE_TOKENS = ('pdrh', 'subjective', 'rating', 'condition', 'label', 'subject', 'file', 'path')

@dataclass
class ConfigData:
    config: str
    folder: Path
    source_csv: Path
    frame: pd.DataFrame
    subject_col: str
    condition_col: str
    feature_cols: list[str]

def _clean_name(value: object) -> str:
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')

def _find_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {_clean_name(c): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in columns:
        lc = _clean_name(c)
        if any((cand in lc for cand in candidates)):
            return c
    return None

def _condition_to_label(value: object) -> int | None:
    token = _clean_name(value)
    if token in TASK_TOKENS or any((t in token for t in ('hi', 'task', 'high'))):
        return 1
    if token in REST_TOKENS or any((t in token for t in ('lo', 'rest', 'low'))):
        return 0
    return None

def _is_nwcf_feature(col: str) -> bool:
    lc = _clean_name(col)
    if any((tok in lc for tok in EXCLUDE_TOKENS)):
        return False
    return any((tok in lc for tok in NWCF_NAME_TOKENS))

def _feature_family(col: str) -> str:
    lc = _clean_name(col)
    if 'shrink' in lc or 'corrobor' in lc or 'gamma' in lc or ('death_compression' in lc):
        return 'reweighting_diagnostics'
    if 'triangle' in lc or 'triad' in lc:
        return 'triangle_support'
    if 'four' in lc or '4clique' in lc or 'clique4' in lc:
        return 'four_clique_support'
    if 'landscape' in lc or 'pi_' in lc or 'persistence_image' in lc or ('image' in lc):
        return 'vectorized_persistence'
    if 'slope' in lc or 'vol' in lc or 'cv' in lc or ('early' in lc) or ('late' in lc):
        return 'transaction_dynamics'
    if 'h0' in lc or 'h1' in lc or 'persistence' in lc or ('entropy' in lc) or ('bars' in lc):
        return 'nwcf_persistence'
    return 'other_nwcf'

def _candidate_score(path: Path, df: pd.DataFrame, subj_col: str, cond_col: str, feats: list[str]) -> tuple:
    n_subjects = df[subj_col].nunique(dropna=True)
    labels = df[cond_col].map(_condition_to_label)
    n_classes = labels.dropna().nunique()
    name_bonus = int('subject' in path.name.lower()) + int('feature' in path.name.lower()) + int('nwcf' in path.name.lower())
    return (n_classes >= 2, n_subjects, len(feats), name_bonus, -len(path.parts))

def find_config_folder(root: Path, config: str) -> Path | None:
    for pattern in OUTPUT_PATTERNS:
        candidate = root / pattern.format(config=config)
        if candidate.exists() and candidate.is_dir():
            return candidate
    matches = sorted((p for p in root.glob(f'*{config}*') if p.is_dir()))
    return matches[0] if matches else None

def load_config_data(root: Path, config: str) -> ConfigData | None:
    folder = find_config_folder(root, config)
    if folder is None:
        return None
    best: tuple | None = None
    best_payload: tuple[Path, pd.DataFrame, str, str, list[str]] | None = None
    for csv_path in sorted(folder.rglob('*.csv')):
        if 'pdrh' in csv_path.name.lower():
            continue
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue
        subj_col = _find_col(header.columns, SUBJECT_COLUMNS)
        cond_col = _find_col(header.columns, CONDITION_COLUMNS)
        if not subj_col or not cond_col:
            continue
        feature_cols = [c for c in header.columns if _is_nwcf_feature(c)]
        if len(feature_cols) < 2:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        labels = df[cond_col].map(_condition_to_label)
        df = df.loc[labels.notna()].copy()
        if df.empty or df[subj_col].nunique(dropna=True) < 10:
            continue
        score = _candidate_score(csv_path, df, subj_col, cond_col, feature_cols)
        if best is None or score > best:
            best = score
            best_payload = (csv_path, df, subj_col, cond_col, feature_cols)
    if best_payload is None:
        return None
    (source_csv, df, subj_col, cond_col, feature_cols) = best_payload
    return ConfigData(config, folder, source_csv, df, subj_col, cond_col, feature_cols)

def paired_effect_table(data: ConfigData) -> pd.DataFrame:
    df = data.frame.copy()
    df['_label'] = df[data.condition_col].map(_condition_to_label)
    rows = []
    for col in data.feature_cols:
        wide = df.pivot_table(index=data.subject_col, columns='_label', values=col, aggfunc='mean')
        if 0 not in wide.columns or 1 not in wide.columns:
            continue
        paired = wide[[0, 1]].dropna()
        if len(paired) < 10:
            continue
        delta = paired[1].to_numpy(float) - paired[0].to_numpy(float)
        delta = delta[np.isfinite(delta)]
        if len(delta) < 10:
            continue
        mean_delta = float(np.mean(delta))
        sd_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        dz = mean_delta / (sd_delta + 1e-12)
        p = np.nan
        if wilcoxon is not None and np.any(np.abs(delta) > 0):
            try:
                p = float(wilcoxon(delta, zero_method='wilcox', alternative='two-sided').pvalue)
            except Exception:
                p = np.nan
        rows.append({'config': data.config, 'feature': col, 'family': _feature_family(col), 'n_pairs': int(len(delta)), 'task_minus_rest_mean': mean_delta, 'cohen_dz': dz, 'abs_dz': abs(dz), 'p_wilcoxon': p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['abs_dz', 'n_pairs'], ascending=[False, False])
    return out

def select_panel(effect_tables: list[pd.DataFrame], top_k: int, min_configs: int) -> pd.DataFrame:
    all_eff = pd.concat(effect_tables, ignore_index=True)
    if all_eff.empty:
        return all_eff
    grouped = all_eff.groupby('feature').agg(family=('family', 'first'), discovery_configs=('config', lambda s: ','.join(sorted(set(map(str, s))))), n_discovery_configs=('config', 'nunique'), mean_abs_dz=('abs_dz', 'mean'), median_abs_dz=('abs_dz', 'median'), direction_consistency=('cohen_dz', lambda x: max(np.mean(np.asarray(x) > 0), np.mean(np.asarray(x) < 0)))).reset_index()
    grouped = grouped[grouped['n_discovery_configs'] >= min_configs]
    if grouped.empty:
        grouped = all_eff.groupby('feature').agg(family=('family', 'first'), discovery_configs=('config', lambda s: ','.join(sorted(set(map(str, s))))), n_discovery_configs=('config', 'nunique'), mean_abs_dz=('abs_dz', 'mean'), median_abs_dz=('abs_dz', 'median'), direction_consistency=('cohen_dz', lambda x: max(np.mean(np.asarray(x) > 0), np.mean(np.asarray(x) < 0)))).reset_index()
    grouped = grouped.sort_values(['n_discovery_configs', 'direction_consistency', 'mean_abs_dz'], ascending=[False, False, False])
    return grouped.head(top_k).copy()

def _available_panel_features(data: ConfigData, panel: pd.DataFrame) -> list[str]:
    available = set(data.feature_cols)
    return [f for f in panel['feature'].tolist() if f in available]

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else np.nan

def _roc_auc_from_scores(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    total = float(len(pos) * len(neg))
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / total)

def _fit_regularized_lda(X: np.ndarray, y: np.ndarray, shrinkage: float=0.2) -> tuple[np.ndarray, float]:
    X0 = X[y == 0]
    X1 = X[y == 1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    centered = np.vstack([X0 - m0, X1 - m1])
    if centered.shape[0] <= 2:
        cov = np.eye(X.shape[1])
    else:
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
    diag = np.diag(np.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diag + 1e-06 * np.eye(X.shape[1])
    w = np.linalg.pinv(cov).dot(m1 - m0)
    prior0 = max(np.mean(y == 0), 1e-06)
    prior1 = max(np.mean(y == 1), 1e-06)
    b = -0.5 * float((m1 + m0).dot(w)) + math.log(prior1 / prior0)
    return (w, b)

def loso_classification(data: ConfigData, features: list[str], model_name: str) -> dict:
    df = data.frame.copy()
    df['_label'] = df[data.condition_col].map(_condition_to_label)
    df = df.dropna(subset=['_label'])
    cols = [c for c in features if c in df.columns]
    if not cols:
        return {'balanced_accuracy': np.nan, 'roc_auc': np.nan, 'n_features': 0, 'n_subjects': 0, 'model': model_name}
    X = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    y = df['_label'].to_numpy(int)
    groups = df[data.subject_col].astype(str).to_numpy()
    if len(np.unique(groups)) < 3 or len(np.unique(y)) < 2:
        return {'balanced_accuracy': np.nan, 'roc_auc': np.nan, 'n_features': len(cols), 'n_subjects': len(np.unique(groups)), 'model': model_name}
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for held_out in np.unique(groups):
        test_idx = np.where(groups == held_out)[0]
        train_idx = np.where(groups != held_out)[0]
        if len(np.unique(y[train_idx])) < 2:
            continue
        X_train = X[train_idx].copy()
        X_test = X[test_idx].copy()
        med = np.nanmedian(X_train, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X_train = np.where(np.isfinite(X_train), X_train, med)
        X_test = np.where(np.isfinite(X_test), X_test, med)
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd
        (w, b) = _fit_regularized_lda(X_train, y[train_idx], shrinkage=0.2)
        score = X_test.dot(w) + b
        pred = (score >= 0).astype(int)
        y_true.extend(y[test_idx].tolist())
        y_pred.extend(pred.tolist())
        y_score.extend(np.asarray(score).reshape(-1).tolist())
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    bal = _balanced_accuracy(y_true_arr, y_pred_arr) if y_true else np.nan
    auc = np.nan
    if y_true and len(set(y_true)) == 2:
        auc = _roc_auc_from_scores(y_true_arr, y_score_arr)
    return {'balanced_accuracy': float(bal), 'roc_auc': float(auc) if not math.isnan(float(auc)) else np.nan, 'n_features': len(cols), 'n_subjects': int(len(np.unique(groups))), 'n_samples': int(len(y)), 'model': model_name}

def family_transfer_score(source_effects: pd.DataFrame, target_effects: pd.DataFrame) -> pd.DataFrame:
    if source_effects.empty or target_effects.empty:
        return pd.DataFrame()
    source = source_effects.groupby('family')['abs_dz'].mean().rename('source_mean_abs_dz')
    target = target_effects.groupby('family')['abs_dz'].mean().rename('target_mean_abs_dz')
    out = pd.concat([source, target], axis=1).dropna().reset_index()
    if out.empty:
        return out
    out['retention_ratio'] = out['target_mean_abs_dz'] / (out['source_mean_abs_dz'] + 1e-12)
    return out.sort_values('target_mean_abs_dz', ascending=False)

def save_figures(out_dir: Path, transfer_df: pd.DataFrame, class_df: pd.DataFrame, family_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (out_dir / 'FIGURES_NOT_GENERATED.txt').write_text('matplotlib is not available in this Python environment. The CSV tables were generated successfully; rerun the script in the STEW analysis environment to create PNG figures.\n')
        return
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160})
    if not transfer_df.empty:
        (fig, ax) = plt.subplots(figsize=(7.2, 4.0))
        pivot = transfer_df.pivot_table(index='target_config', columns='source_panel', values='mean_abs_dz', aggfunc='mean')
        im = ax.imshow(pivot.to_numpy(float), cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title('Cross-configuration transfer of selected NWCF biomarkers')
        ax.set_xlabel('Discovery panel')
        ax.set_ylabel('Test configuration')
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label('mean |paired effect|')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_CC1_NWCF_CrossConfigTransferHeatmap.png', bbox_inches='tight')
        plt.close(fig)
    if not class_df.empty:
        (fig, ax) = plt.subplots(figsize=(7.2, 4.0))
        plot_df = class_df.sort_values(['target_config', 'source_panel'])
        labels = [f'{r.source_panel}->{r.target_config}' for r in plot_df.itertuples()]
        x = np.arange(len(plot_df))
        ax.bar(x, plot_df['balanced_accuracy'], color='#2ca6a4', edgecolor='#222222', linewidth=0.5)
        ax.axhline(0.5, color='#777777', linestyle='--', linewidth=0.9)
        ax.set_xticks(x, labels, rotation=60, ha='right', fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('LOSO balanced accuracy')
        ax.set_title('Same NWCF biomarker panel tested across configurations')
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_CC2_NWCF_CrossConfigClassification.png', bbox_inches='tight')
        plt.close(fig)
    if not family_df.empty:
        top = family_df.groupby('family')['retention_ratio'].median().sort_values(ascending=False)
        (fig, ax) = plt.subplots(figsize=(6.6, 3.6))
        ax.barh(range(len(top)), top.values, color='#6fb6d6', edgecolor='#222222', linewidth=0.5)
        ax.axvline(1.0, color='#777777', linestyle='--', linewidth=0.9)
        ax.set_yticks(range(len(top)), top.index)
        ax.invert_yaxis()
        ax.set_xlabel('median target/source effect retention')
        ax.set_title('Which NWCF feature families generalize?')
        fig.tight_layout()
        fig.savefig(out_dir / 'Figure_CC3_NWCF_FamilyGeneralization.png', bbox_inches='tight')
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description='Test cross-configuration generalization of compact NWCF biomarker panels.')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Folder containing completed STEW output directories.')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), help='Configuration labels to scan.')
    parser.add_argument('--discover', nargs='+', default=['84aec', '84plv', '84corr'], help='Discovery configs used for panel selection.')
    parser.add_argument('--top-k', type=int, default=12, help='Number of selected NWCF features in each compact panel.')
    parser.add_argument('--min-discovery-configs', type=int, default=1, help='Minimum discovery configs in which a feature must appear.')
    parser.add_argument('--model', choices=['regularized_lda'], default='regularized_lda', help='LOSO classifier for compact panels.')
    parser.add_argument('--out-dir', type=Path, default=None, help='Output folder. Defaults to <root>/stew_nwcf_cross_config_generalization.')
    args = parser.parse_args()
    out_dir = args.out_dir or args.root / 'stew_nwcf_cross_config_generalization'
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, ConfigData] = {}
    load_rows = []
    for cfg in args.configs:
        data = load_config_data(args.root, cfg)
        if data is None:
            load_rows.append({'config': cfg, 'status': 'missing_or_no_subject_level_nwcf_table'})
            continue
        loaded[cfg] = data
        load_rows.append({'config': cfg, 'status': 'loaded', 'folder': str(data.folder), 'source_csv': str(data.source_csv), 'n_rows': len(data.frame), 'n_subjects': data.frame[data.subject_col].nunique(), 'n_nwcf_features': len(data.feature_cols)})
    pd.DataFrame(load_rows).to_csv(out_dir / 'Table_CC0_LoadedConfigSources.csv', index=False)
    if not loaded:
        raise SystemExit('No usable NWCF subject-level tables were found.')
    effects = {cfg: paired_effect_table(data) for (cfg, data) in loaded.items()}
    for (cfg, table) in effects.items():
        table.to_csv(out_dir / f'Table_CC1_PairedEffects_{cfg}.csv', index=False)
    discover_configs = [cfg for cfg in args.discover if cfg in effects and (not effects[cfg].empty)]
    if not discover_configs:
        discover_configs = [cfg for (cfg, table) in effects.items() if not table.empty]
    discovery_tables = [effects[cfg] for cfg in discover_configs]
    pooled_panel = select_panel(discovery_tables, args.top_k, args.min_discovery_configs)
    pooled_panel.insert(0, 'panel', 'pooled_discovery')
    pooled_panel.to_csv(out_dir / 'Table_CC2_SelectedNWCFPanel_PooledDiscovery.csv', index=False)
    panel_tables = {'pooled_discovery': pooled_panel}
    for cfg in discover_configs:
        panel = effects[cfg].head(args.top_k).copy()
        panel = panel.rename(columns={'config': 'source_effect_config'})
        panel.insert(0, 'panel', f'{cfg}_panel')
        if 'mean_abs_dz' not in panel.columns:
            panel['mean_abs_dz'] = panel['abs_dz']
        panel_tables[f'{cfg}_panel'] = panel
        panel.to_csv(out_dir / f'Table_CC2_SelectedNWCFPanel_{cfg}.csv', index=False)
    transfer_rows = []
    class_rows = []
    family_rows = []
    for (panel_name, panel) in panel_tables.items():
        features = panel['feature'].tolist()
        source_effects = pd.concat([effects[cfg] for cfg in discover_configs], ignore_index=True)
        for (cfg, data) in loaded.items():
            target_effects = effects[cfg]
            target = target_effects[target_effects['feature'].isin(features)].copy()
            available = _available_panel_features(data, panel)
            perf = loso_classification(data, available, args.model)
            class_rows.append({'source_panel': panel_name, 'target_config': cfg, 'n_panel_features_available': len(available), **perf})
            if not target.empty:
                transfer_rows.append({'source_panel': panel_name, 'target_config': cfg, 'n_panel_features_available': len(available), 'mean_abs_dz': float(target['abs_dz'].mean()), 'median_abs_dz': float(target['abs_dz'].median()), 'direction_consistency': float(max(np.mean(target['cohen_dz'].to_numpy(float) > 0), np.mean(target['cohen_dz'].to_numpy(float) < 0)))})
            fam = family_transfer_score(source_effects[source_effects['feature'].isin(features)], target)
            if not fam.empty:
                fam.insert(0, 'source_panel', panel_name)
                fam.insert(1, 'target_config', cfg)
                family_rows.append(fam)
    transfer_df = pd.DataFrame(transfer_rows)
    class_df = pd.DataFrame(class_rows)
    family_df = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    transfer_df.to_csv(out_dir / 'Table_CC3_CrossConfigEffectTransfer.csv', index=False)
    class_df.to_csv(out_dir / 'Table_CC4_CrossConfigLOSOClassification.csv', index=False)
    family_df.to_csv(out_dir / 'Table_CC5_FeatureFamilyGeneralization.csv', index=False)
    summary = {'loaded_configs': sorted(loaded), 'discovery_configs': discover_configs, 'top_k': args.top_k, 'model': args.model, 'main_outputs': ['Table_CC2_SelectedNWCFPanel_PooledDiscovery.csv', 'Table_CC3_CrossConfigEffectTransfer.csv', 'Table_CC4_CrossConfigLOSOClassification.csv', 'Table_CC5_FeatureFamilyGeneralization.csv', 'Figure_CC1_NWCF_CrossConfigTransferHeatmap.png', 'Figure_CC2_NWCF_CrossConfigClassification.png', 'Figure_CC3_NWCF_FamilyGeneralization.png']}
    (out_dir / 'cross_config_generalization_summary.json').write_text(json.dumps(summary, indent=2))
    save_figures(out_dir, transfer_df, class_df, family_df)
    print(f'Saved cross-configuration NWCF generalization outputs to: {out_dir}')
if __name__ == '__main__':
    main()
