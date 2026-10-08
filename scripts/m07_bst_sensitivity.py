"""
M07 — sensitivity of balanced self-training (BST) to its three settings, around the a-priori default
(rounds R=3, pseudo-label fraction q=1/3 per tail, pseudo weight = own-data weight x1):
  R in {1, 3, 5};  q in {0.2, 1/3, 0.45};  weight multiplier in {0.5, 1, 2}  (one-at-a-time around the default)
Reported: mean AUC gain over PCA_LR and wins/11 per direction. RIEM, K=100, C=0.1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, fit_eval_score  # noqa: E402
from m03_hierarchical import prepare  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402

GRID = [dict(R=3, q=1 / 3, wm=1.0)] + [dict(R=r, q=1 / 3, wm=1.0) for r in (1, 5)] + \
       [dict(R=3, q=q, wm=1.0) for q in (0.2, 0.45)] + [dict(R=3, q=1 / 3, wm=m) for m in (0.5, 2.0)]


def bst(Z, y, Zte, own_w, R, q, wm):
    m = LogisticRegression(C=0.1, max_iter=5000).fit(Z, y)
    for _ in range(R):
        s = m.decision_function(Zte); lo, hi = np.quantile(s, [q, 1 - q])
        pick = (s <= lo) | (s >= hi); yp = (s[pick] >= hi).astype(int)
        w = np.concatenate([np.ones(len(y)), np.full(len(yp), wm * own_w / max(len(yp), 1))])
        m = LogisticRegression(C=0.1, max_iter=5000).fit(np.vstack([Z, Zte[pick]]), np.concatenate([y, yp]), sample_weight=w)
    return m.decision_function(Zte)


def run_target(sid, allX, allM):
    rows = []
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        Z, M, Zte, Mte = prepare(allX, allM, sid, src, tgt, 100, (0, 2))
        y = (M.level == 2).to_numpy().astype(int); own = float((M.subject == sid).sum())
        base = fit_eval_score(LogisticRegression(C=0.1, max_iter=5000).fit(Z, y).decision_function(Zte), Mte)["auc"]
        for g in GRID:
            auc = fit_eval_score(bst(Z, y, Zte, own, **g), Mte)["auc"]
            rows.append(dict(subject=sid, train=src, test=tgt, setting=f"R={g['R']} q={g['q']:.2f} w={g['wm']}", gain=auc - base))
    return rows


if __name__ == "__main__":
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), "RIEM") for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    df = pd.DataFrame([r for rr in Parallel(n_jobs=2)(delayed(run_target)(s, allX, allM) for s in dev) for r in rr])
    df.to_csv(OUT / "M07_bst_sensitivity_dev.csv", index=False)
    out = df.groupby(["setting", "train", "test"]).gain.agg(mean="mean", wins=lambda g: int((g > 0).sum())).unstack(["train", "test"])
    print(out.round(3).to_string())
