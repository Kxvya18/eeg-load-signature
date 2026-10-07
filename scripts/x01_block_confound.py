"""
Feasibility check (development subjects only): how much within-session workload decoding is
explained by recording time / block identity rather than workload?
Per session, tangent-space features (1-40 Hz OAS covariances at the session Riemannian mean) + LR.
  A  N-back 3-class, shuffled 5-fold epoch split (common practice)
  B  N-back 3-class, leave-one-block-out (train on 2 blocks per level, test the 3rd)
  C  block identity WITHIN a level (same workload, different time), shuffled 5-fold; chance 1/3
  D  MATB 3-class, shuffled 5-fold (one block per level -> level and time are inseparable)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b01_baselines import CLEAN, EPOCH, ROOT, covs, riemann_mean, tangent_space  # noqa: E402


def load(sid, ses):
    z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz")
    X, lvl, blk = [], [], []
    for key in sorted(z.files):
        name, b = key.split("|")
        seg = z[key].astype(float)
        for a in range(0, len(seg) - EPOCH + 1, EPOCH):
            X.append(seg[a:a + EPOCH].T); lvl.append(name); blk.append(int(b))
    return np.array(X), np.array(lvl), np.array(blk)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def shuffled(F, y):
    return cross_val_score(clf(), F, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()


def run(sid):
    rows = []
    for ses in ("S1", "S2", "S3"):
        X, lvl, blk = load(sid, ses)
        C = covs(X, "broad"); F = tangent_space(C, riemann_mean(C))
        nb = np.char.startswith(lvl, "nback"); y = np.array([int(l[-1]) for l in lvl[nb]])
        Fn, bn = F[nb], blk[nb]
        r = dict(subject=sid, session=ses, A_nback_shuffled=shuffled(Fn, y))
        r["B_nback_leave_block_out"] = np.mean([clf().fit(Fn[bn != k], y[bn != k]).score(Fn[bn == k], y[bn == k])
                                                for k in np.unique(bn)])
        r["C_block_id_within_level"] = np.mean([shuffled(Fn[y == L], bn[y == L]) for L in np.unique(y)])
        mb = np.char.startswith(lvl, "matb")
        r["D_matb_shuffled"] = shuffled(F[mb], lvl[mb])
        rows.append(r)
    return rows


if __name__ == "__main__":
    dev = json.loads((ROOT / "docs" / "subject_split.json").read_text())["development"]
    df = pd.DataFrame([r for rr in Parallel(n_jobs=7)(delayed(run)(s) for s in dev) for r in rr])
    df.to_csv(ROOT / "outputs" / "X01_block_confound_dev.csv", index=False)
    print(df.drop(columns=["subject", "session"]).describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
