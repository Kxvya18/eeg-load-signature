"""
M04 — two label-free levers on top of the strongest baseline (pooled PCA-LR, all three levels kept out of the
fit except 0/2 as in the bar):
  representation : RIEM (session re-centring, as before) vs RIEM_T (re-centring per session x task run set)
  test-time      : balanced self-training (BST). The design guarantees equal time per load level in a session,
                   so the unseen-task test run set is level-balanced. Iterate R=3 times: score the unlabeled test
                   windows, pseudo-label the bottom third as low and the top third as high, refit the pooled LR
                   with these windows added (total pseudo-label weight = weight of the target's own source data).
Variants: PCA_LR[RIEM] (bar), PCA_LR[RIEM_T], BST[RIEM], BST[RIEM_T]. Shared K-dim PCA per variant (fit on
training windows only). L2 logistic, C = 0.1 everywhere.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, fit_eval_score  # noqa: E402
from m03_hierarchical import prepare  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402

C_REG, ROUNDS = 0.1, 3


def bst(Z, y, w, Zte, own_weight):
    m = LogisticRegression(C=C_REG, max_iter=5000).fit(Z, y, sample_weight=w)
    for _ in range(ROUNDS):
        s = m.decision_function(Zte)
        lo, hi = np.quantile(s, [1 / 3, 2 / 3])
        pick = (s <= lo) | (s >= hi)
        Zp, yp = Zte[pick], (s[pick] >= hi).astype(int)
        wp = np.full(len(yp), own_weight / max(len(yp), 1))
        m = LogisticRegression(C=C_REG, max_iter=5000).fit(np.vstack([Z, Zp]), np.concatenate([y, yp]),
                                                           sample_weight=np.concatenate([w, wp]))
    return m.decision_function(Zte)


def run_target(sid, data, K):
    rows = []
    for fam, (allX, allM) in data.items():
        for src, tgt in (("nback", "matb"), ("matb", "nback")):
            Z, M, Zte, Mte = prepare(allX, allM, sid, src, tgt, K, (0, 2))
            y = (M.level == 2).to_numpy().astype(int); w = np.ones(len(y))
            own_w = float(((M.subject == sid)).sum())
            m = LogisticRegression(C=C_REG, max_iter=5000).fit(Z, y)
            for name, s in ((f"PCA_LR[{fam}]", m.decision_function(Zte)), (f"BST[{fam}]", bst(Z, y, w, Zte, own_w))):
                rows.append(dict(subject=sid, method=name, train=src, test=tgt, **fit_eval_score(s, Mte)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--families", default="RIEM,RIEM_T")
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    loaded = {s: load(s) for s in dev}
    data = {}
    for fam in a.families.split(","):
        parts = [build(loaded[s], fam) for s in dev]
        data[fam] = (np.vstack([p[0] for p in parts]), pd.concat([p[1] for p in parts], ignore_index=True))
    df = pd.DataFrame([r for rr in Parallel(n_jobs=3)(delayed(run_target)(s, data, a.K) for s in dev) for r in rr])
    df.to_csv(OUT / f"M04_taskcentre_selftrain_dev_K{a.K}_{a.families.replace(',', '-')}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("auc", "rho", "time_leak"):
        print(f"\n== {met}"); print(df.groupby(["train", "test", "method"])[met].mean().unstack("method").round(3).to_string())
    print("\n== paired vs PCA_LR[RIEM] (AUC)")
    for (tr, te), g in df.groupby(["train", "test"]):
        p = g.pivot(index="subject", columns="method", values="auc")
        for mth in p.columns.drop("PCA_LR[RIEM]"):
            d = p[mth] - p["PCA_LR[RIEM]"]
            print(f"{tr}->{te} {mth:16s} {d.mean():+.3f}  wins {int((d > 0).sum())}/{len(d)}  p={wilcoxon(d).pvalue:.3f}")
