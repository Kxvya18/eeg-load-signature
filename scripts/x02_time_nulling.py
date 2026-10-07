"""
Development probe: does removing label-free TIME directions improve cross-session workload decoding?
Train S1+S2, test S3 (MATB 3-class or N-back 3-class), 2-s epochs, per-session re-centered tangent
spaces (theta/alpha/beta filter bank, as B5).
Drift directions (no workload labels used anywhere):
  slope  : per block, least-squares slope of features on epoch time (all sessions incl. unlabeled S3)
  ses    : session-mean differences are already removed by re-centering; within-session block-mean
           differences are NOT used (they would need level labels)
Directions = top-k right singular vectors of the stacked, normalised slope vectors; features are
projected onto their orthogonal complement before StandardScaler+LR (C by leave-one-session-out).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b01_baselines import CLEAN, EPOCH, ROOT, TASKS, covs, fit_lr, riemann_mean, tangent_space  # noqa: E402

KS = (0, 1, 2, 4, 8, 16, 32)


def load(sid, ses, levels):
    z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz")
    X, y, blk, t = [], [], [], []
    for c, lvl in enumerate(levels):
        for key in sorted(k for k in z.files if k.startswith(lvl + "|")):
            seg = z[key].astype(float)
            for i, a in enumerate(range(0, len(seg) - EPOCH + 1, EPOCH)):
                X.append(seg[a:a + EPOCH].T); y.append(c); blk.append(key); t.append(i)
    return np.array(X), np.array(y), np.array(blk), np.array(t, float)


def slopes(F, blk, t):
    out = []
    for b in np.unique(blk):
        m = blk == b
        tc = t[m] - t[m].mean()
        s = (tc @ (F[m] - F[m].mean(0))) / (tc @ tc)
        out.append(s / (np.linalg.norm(s) + 1e-12))
    return out


def run(sid, levels):
    d = {s: load(sid, s, levels) for s in ("S1", "S2", "S3")}
    F = {}
    for s, (X, *_ ) in d.items():
        F[s] = np.hstack([tangent_space(C, riemann_mean(C)) for C in (covs(X, b) for b in ("theta", "alpha", "beta"))])
    S = np.array([v for s in d for v in slopes(F[s], d[s][2], d[s][3])])
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    ytr = np.concatenate([d["S1"][1], d["S2"][1]]); yte = d["S3"][1]
    sess = np.concatenate([np.zeros(len(d["S1"][1])), np.ones(len(d["S2"][1]))])
    res = {"subject": sid}
    for k in KS:
        P = np.eye(S.shape[1]) - (Vt[:k].T @ Vt[:k] if k else 0)
        Ftr = np.vstack([F["S1"], F["S2"]]) @ P
        res[f"k{k}"] = fit_lr(Ftr, ytr, sess).score(F["S3"] @ P, yte)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="matb", choices=list(TASKS))
    a = ap.parse_args()
    dev = json.loads((ROOT / "docs" / "subject_split.json").read_text())["development"]
    df = pd.DataFrame(Parallel(n_jobs=7)(delayed(run)(s, TASKS[a.task]) for s in dev))
    df.to_csv(ROOT / "outputs" / f"X02_time_nulling_dev_{a.task}.csv", index=False)
    print(df.round(3).to_string(index=False)); print("mean:\n", df.drop(columns="subject").mean().round(3).to_string())
