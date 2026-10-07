"""
M05 — similarity-weighted pooling (SWP), alone and with balanced self-training (SWP+BST), vs the bar.

For target s with source task A: for every population subject j, train an L2 LR on j's task-A data (levels 0/2,
all sessions) in the shared PCA space and score it on s's own task-A data (S1+S2): sim_j = AUC.  Subject weights
  omega_j = softmax(beta * sim_j)  scaled so the population keeps its total weight; own data weight unchanged.
beta fixed a priori (beta = 20: AUC differences of 0.05 change weights by e^1).  Then pooled LR (C = 0.1) with
these window weights; BST as in M04. Hypothesis: people similar on the labelled task are similar on the unseen one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, fit_eval_score  # noqa: E402
from m03_hierarchical import prepare  # noqa: E402
from m04_taskcentre_selftrain import bst  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402

C_REG, BETA = 0.1, 20.0


def run_target(sid, allX, allM, K, allZ_cache=None):
    rows = []
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        Z, M, Zte, Mte = prepare(allX, allM, sid, src, tgt, K, (0, 2))
        y = (M.level == 2).to_numpy().astype(int)
        own = (M.subject == sid).to_numpy()
        pops = sorted(set(M.subject[~own]))
        sim, sim_tgt = {}, {}
        for j in pops:
            jm = ((M.subject == j) & (M.task == src)).to_numpy()
            mj = LogisticRegression(C=C_REG, max_iter=5000).fit(Z[jm], y[jm])
            sim[j] = roc_auc_score(y[own], mj.decision_function(Z[own]))
            jt = ((M.subject == j) & (M.task == tgt)).to_numpy()            # diagnostic only: does j transfer to s's
            mt = LogisticRegression(C=C_REG, max_iter=5000).fit(Z[jt], y[jt])  # unseen task? (uses s's test labels)
            lv = Mte.level.to_numpy(); hl = lv != 1
            sim_tgt[j] = roc_auc_score(lv[hl] == 2, mt.decision_function(Zte)[hl])
        sv = np.array([sim[j] for j in pops]); om = np.exp(BETA * (sv - sv.max())); om = om / om.sum() * len(pops)
        w = np.ones(len(y))
        for j, o in zip(pops, om):
            w[(M.subject == j).to_numpy()] = o
        base = LogisticRegression(C=C_REG, max_iter=5000).fit(Z, y).decision_function(Zte)
        swp = LogisticRegression(C=C_REG, max_iter=5000).fit(Z, y, sample_weight=w).decision_function(Zte)
        swp_bst = bst(Z, y, w, Zte, float(own.sum()))
        bst0 = bst(Z, y, np.ones(len(y)), Zte, float(own.sum()))
        diag = spearmanr([sim[j] for j in pops], [sim_tgt[j] for j in pops])[0]
        for name, s in (("PCA_LR", base), ("BST", bst0), ("SWP", swp), ("SWP+BST", swp_bst)):
            rows.append(dict(subject=sid, method=name, train=src, test=tgt, sim_transfer_rho=diag,
                             eff_n=float(om.sum() ** 2 / (om ** 2).sum()), **fit_eval_score(s, Mte)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=100); ap.add_argument("--family", default="RIEM")
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), a.family) for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    df = pd.DataFrame([r for rr in Parallel(n_jobs=4)(delayed(run_target)(s, allX, allM, a.K) for s in dev) for r in rr])
    df.to_csv(OUT / f"M05_similarity_pooling_dev_{a.family}_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("auc", "rho"):
        print(f"\n== {met}"); print(df.groupby(["train", "test", "method"])[met].mean().unstack("method").round(3).to_string())
    print("\n== diagnostics: does source-task similarity predict unseen-task transfer? (mean Spearman over targets)")
    print(df[df.method == "SWP"].groupby(["train", "test"])[["sim_transfer_rho", "eff_n"]].mean().round(3).to_string())
    print("\n== paired vs PCA_LR (AUC)")
    for (tr, te), g in df.groupby(["train", "test"]):
        p = g.pivot(index="subject", columns="method", values="auc")
        for mth in p.columns.drop("PCA_LR"):
            d = p[mth] - p.PCA_LR
            print(f"{tr}->{te} {mth:10s} {d.mean():+.3f}  wins {int((d > 0).sum())}/{len(d)}  p={wilcoxon(d).pvalue:.3f}")
