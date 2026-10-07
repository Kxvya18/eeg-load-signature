"""
M03 — hierarchical (shared + task + subject) ordinal load decoder vs the strongest baseline (pooled PCA-LR).

Scenario identical to M01/B10. Shared K-dim PCA of all training windows (all levels) for every variant.
  PCA_LR      pooled L2 logistic regression, levels 0 vs 2                         (bar)
  PCA_LR_ORD  pooled cumulative-link (ordinal) logistic, levels 0<1<2, run-specific thresholds
  HMT         w_{subject,task} = w0 + v_task + u_subject (feature augmentation, Evgeniou & Pontil 2004),
              levels 0 vs 2; prediction on the unseen task uses w0 + u_target only
  HMT_ORD     HMT with the ordinal loss
Ordinal loss via data replication: window x, level L -> two binary rows (L>=1), (L>=2) with one-hot
threshold intercepts per (run, k); thresholds never enter the test score (AUC / rho are threshold-free).
All logistic regressions: L2, C = 0.1. Augmentation scale a (task and subject blocks) fixed at 1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, fit_eval_score  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402

C_REG = 0.1


def prepare(allX, allM, sid, src, tgt, K, levels):
    own = (allM.subject == sid) & (allM.task == src) & allM.session.isin(["S1", "S2"])
    tr = (((allM.subject != sid) | own) & allM.level.isin(levels)).to_numpy()
    te = ((allM.subject == sid) & (allM.task == tgt) & (allM.session == "S3")).to_numpy()
    Xtr = allX[tr]; mu0 = Xtr.mean(0)
    sub = np.random.default_rng(sid).choice(len(Xtr), min(len(Xtr), 6000), replace=False)
    _, _, Vt = randomized_svd(Xtr[sub] - mu0, K, random_state=0)
    Z = (Xtr - mu0) @ Vt.T; sd = Z.std(0)
    return Z / sd, allM[tr].reset_index(drop=True), ((allX[te] - mu0) @ Vt.T) / sd, allM[te].reset_index(drop=True)


def augment(Z, M, subjects, a=1.0):
    """[shared | task blocks | subject blocks] (sparse)."""
    blocks = [sparse.csr_matrix(Z)]
    for t in ("nback", "matb"):
        blocks.append(sparse.csr_matrix(Z * (a * (M.task == t).to_numpy()[:, None])))
    for s in subjects:
        blocks.append(sparse.csr_matrix(Z * (a * (M.subject == s).to_numpy()[:, None])))
    return sparse.hstack(blocks).tocsr()


def ordinal_rows(F, M):
    """Replicate rows for thresholds k=1,2 with run-specific threshold one-hots."""
    run = (M.subject.astype(str) + M.session + M.task).to_numpy()
    runs = np.unique(run); ri = np.searchsorted(runs, run); n = F.shape[0]
    rows, ys = [], []
    for k in (1, 2):
        T = sparse.csr_matrix((np.ones(n), (np.arange(n), ri + (k - 1) * len(runs))), shape=(n, 2 * len(runs)))
        rows.append(sparse.hstack([F, T])); ys.append((M.level.to_numpy() >= k).astype(int))
    return sparse.vstack(rows).tocsr(), np.concatenate(ys), F.shape[1]


def fit(F, M, ordinal):
    if ordinal:
        X, y, p = ordinal_rows(F, M)
        m = LogisticRegression(C=C_REG, max_iter=5000, fit_intercept=False).fit(X, y)
        return m.coef_[0][:p]
    keep = M.level.isin([0, 2]).to_numpy()
    m = LogisticRegression(C=C_REG, max_iter=5000).fit(F[keep], (M.level[keep] == 2).to_numpy())
    return m.coef_[0]


def run_target(sid, allX, allM, K, subjects):
    rows = []
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        Z, M, Zte, Mte = prepare(allX, allM, sid, src, tgt, K, (0, 1, 2))
        k = Z.shape[1]
        scores = {}
        for ordn, tag in ((False, ""), (True, "_ORD")):
            w = fit(sparse.csr_matrix(Z), M, ordn); scores["PCA_LR" + tag] = Zte @ w[:k]
            w = fit(augment(Z, M, subjects), M, ordn)
            si = 3 + subjects.index(sid)
            scores["HMT" + tag] = Zte @ (w[:k] + w[si * k:(si + 1) * k])
            scores["HMT_SHARED" + tag] = Zte @ w[:k]
        for name, s in scores.items():
            rows.append(dict(subject=sid, method=name, train=src, test=tgt, **fit_eval_score(s, Mte)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="RIEM"); ap.add_argument("--K", type=int, default=100)
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), a.family) for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    df = pd.DataFrame([r for rr in Parallel(n_jobs=4)(delayed(run_target)(s, allX, allM, a.K, dev) for s in dev) for r in rr])
    df.to_csv(OUT / f"M03_hierarchical_dev_{a.family}_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("auc", "rho"):
        print(f"\n== {met}"); print(df.groupby(["train", "test", "method"])[met].mean().unstack("method").round(3).to_string())
    from scipy.stats import wilcoxon
    print("\n== paired vs PCA_LR (AUC)")
    for (tr, te), g in df.groupby(["train", "test"]):
        p = g.pivot(index="subject", columns="method", values="auc")
        for m in p.columns.drop("PCA_LR"):
            d = p[m] - p.PCA_LR
            print(f"{tr}->{te} {m:16s} {d.mean():+.3f}  wins {int((d > 0).sum())}/{len(d)}  p={wilcoxon(d).pvalue:.3f}")
