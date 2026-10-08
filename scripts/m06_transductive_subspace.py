"""
M06 — transductive shared subspace: the K-dim PCA is fit on training windows PLUS the target's unlabeled
unseen-task test windows (no labels), so the low-rank space also spans the new task's variability.
Variants: PCA_LR (bar), BST (M04), TPCA_LR, TPCA_BST. RIEM features, C = 0.1, K = 100.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, fit_eval_score  # noqa: E402
from m04_taskcentre_selftrain import bst  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402


def prepare(allX, allM, sid, src, tgt, K, transductive):
    own = (allM.subject == sid) & (allM.task == src) & allM.session.isin(["S1", "S2"])
    tr = (((allM.subject != sid) | own) & allM.level.isin([0, 2])).to_numpy()
    te = ((allM.subject == sid) & (allM.task == tgt) & (allM.session == "S3")).to_numpy()
    Xtr, Xte = allX[tr], allX[te]
    rng = np.random.default_rng(sid)
    sub = Xtr[rng.choice(len(Xtr), min(len(Xtr), 6000), replace=False)]
    if transductive:                                   # unlabeled test windows join the subspace fit
        sub = np.vstack([sub, Xte])
    mu0 = sub.mean(0)
    _, _, Vt = randomized_svd(sub - mu0, K, random_state=0)
    Z = (Xtr - mu0) @ Vt.T; sd = Z.std(0)
    return Z / sd, allM[tr].reset_index(drop=True), ((Xte - mu0) @ Vt.T) / sd, allM[te].reset_index(drop=True)


def run_target(sid, allX, allM, K):
    rows = []
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        for tr_flag, tag in ((False, ""), (True, "T")):
            Z, M, Zte, Mte = prepare(allX, allM, sid, src, tgt, K, tr_flag)
            y = (M.level == 2).to_numpy().astype(int); own = float((M.subject == sid).sum())
            s0 = LogisticRegression(C=0.1, max_iter=5000).fit(Z, y).decision_function(Zte)
            s1 = bst(Z, y, np.ones(len(y)), Zte, own)
            for name, s in ((f"{tag}PCA_LR", s0), (f"{tag}PCA_BST" if tag else "BST", s1)):
                rows.append(dict(subject=sid, method=name, train=src, test=tgt, **fit_eval_score(s, Mte)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=100)
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), "RIEM") for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    df = pd.DataFrame([r for rr in Parallel(n_jobs=2)(delayed(run_target)(s, allX, allM, a.K) for s in dev) for r in rr])
    df.to_csv(OUT / f"M06_transductive_subspace_dev_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["train", "test", "method"])["auc"].mean().unstack("method").round(3).to_string())
    for (tr, te), g in df.groupby(["train", "test"]):
        p = g.pivot(index="subject", columns="method", values="auc")
        for mth in p.columns.drop("PCA_LR"):
            d = p[mth] - p.PCA_LR
            print(f"{tr}->{te} {mth:10s} {d.mean():+.3f}  wins {int((d > 0).sum())}/{len(d)}  p={wilcoxon(d).pvalue:.3f}")
