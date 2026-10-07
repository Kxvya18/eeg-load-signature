"""
Cross-session workload baselines on the DEVELOPMENT half only (docs/subject_split.json).
Protocol: MATB easy/med/diff (3-class) or N-back 0/1/2; train S1+S2, test S3; 2-s epochs.
  B1 log relative band power (θ, α, β per channel) + LR
  B2 Riemannian tangent space (1–40 Hz OAS covariances) at the training mean + LR
  B3 tangent space with per-session re-centering (each session whitened by its own mean;
     the test session uses its own unlabeled mean) + LR
  B4 minimum distance to Riemannian class means after re-centering (MDM-RC)
  B5 filter-bank (θ, α, β) re-centered tangent spaces, concatenated + LR
LR C is chosen by leave-one-training-session-out (S1<->S2) from {0.01, 0.1, 1, 10}.

Usage:  python scripts/b01_baselines.py [--task matb|nback]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eegls.baselines import _eig_fn, riemann_mean, tangent_space  # noqa: E402
from eegls.paths import CLEAN  # noqa: E402

FS, EPOCH = 128, 256
TASKS = {"matb": ("matb_easy", "matb_med", "matb_diff"), "nback": ("nback0", "nback1", "nback2")}
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "broad": (1, 40)}
C_GRID = (0.01, 0.1, 1.0, 10.0)


def epochs(sid, ses, levels):
    z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz")
    X, y = [], []
    for c, lvl in enumerate(levels):
        for key in sorted(k for k in z.files if k.startswith(lvl + "|")):
            seg = z[key].astype(float)
            for a in range(0, len(seg) - EPOCH + 1, EPOCH):
                X.append(seg[a:a + EPOCH].T); y.append(c)
    return np.array(X), np.array(y)


def covs(X, band):
    lo, hi = BANDS[band]
    Xf = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), X, axis=-1)
    return np.array([OAS().fit(x.T).covariance_ for x in Xf])


def bandpower(X):
    f, p = welch(X, fs=FS, nperseg=EPOCH, axis=-1)
    tot = p[..., (f >= 1) & (f <= 40)].sum(-1)
    return np.concatenate([np.log(p[..., (f >= lo) & (f < hi)].sum(-1) / tot)
                           for lo, hi in (BANDS["theta"], BANDS["alpha"], BANDS["beta"])], axis=1)


def airm_dist(A, B):
    ai = _eig_fn(A, lambda v: 1 / np.sqrt(v))
    return np.sqrt(np.sum(np.log(np.linalg.eigvalsh(ai @ B @ ai)) ** 2))


def fit_lr(Xtr, ytr, sess):
    """Choose C by leave-one-training-session-out, then refit on all training data."""
    best, best_acc = C_GRID[0], -1
    for c in C_GRID:
        accs = []
        for s in np.unique(sess):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
            m.fit(Xtr[sess != s], ytr[sess != s]); accs.append(m.score(Xtr[sess == s], ytr[sess == s]))
        if np.mean(accs) > best_acc:
            best, best_acc = c, np.mean(accs)
    return make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(Xtr, ytr)


def run_subject(sid, levels):
    data = {s: epochs(sid, s, levels) for s in ("S1", "S2", "S3")}
    ytr = np.concatenate([data["S1"][1], data["S2"][1]]); yte = data["S3"][1]
    sess = np.concatenate([np.zeros(len(data["S1"][1])), np.ones(len(data["S2"][1]))])
    res = {"subject": sid, "n_test": len(yte)}
    # B1
    bp = {s: bandpower(data[s][0]) for s in data}
    res["B1_bandpower"] = fit_lr(np.vstack([bp["S1"], bp["S2"]]), ytr, sess).score(bp["S3"], yte)
    C = {b: {s: covs(data[s][0], b) for s in data} for b in BANDS}
    # B2: tangent space at the pooled training mean (no alignment)
    ref = riemann_mean(np.vstack([C["broad"]["S1"], C["broad"]["S2"]]))
    T = {s: tangent_space(C["broad"][s], ref) for s in data}
    res["B2_TS"] = fit_lr(np.vstack([T["S1"], T["S2"]]), ytr, sess).score(T["S3"], yte)
    # B3 / B5: per-session re-centering (each session at its own Riemannian mean)
    TR = {b: {s: tangent_space(C[b][s], riemann_mean(C[b][s])) for s in data} for b in BANDS}
    res["B3_TS_RC"] = fit_lr(np.vstack([TR["broad"]["S1"], TR["broad"]["S2"]]), ytr, sess).score(TR["broad"]["S3"], yte)
    fb = {s: np.hstack([TR[b][s] for b in ("theta", "alpha", "beta")]) for s in data}
    res["B5_FB_TS_RC"] = fit_lr(np.vstack([fb["S1"], fb["S2"]]), ytr, sess).score(fb["S3"], yte)
    # B4: MDM after re-centering
    rc = {}
    for s in data:
        mi = _eig_fn(riemann_mean(C["broad"][s]), lambda v: 1 / np.sqrt(v))
        rc[s] = np.array([mi @ c @ mi for c in C["broad"][s]])
    Ctr = np.vstack([rc["S1"], rc["S2"]])
    means = [riemann_mean(Ctr[ytr == k]) for k in range(len(levels))]
    pred = [int(np.argmin([airm_dist(m, c) for m in means])) for c in rc["S3"]]
    res["B4_MDM_RC"] = float(np.mean(np.array(pred) == yte))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="matb", choices=list(TASKS))
    args = ap.parse_args()
    dev = json.loads((ROOT / "docs" / "subject_split.json").read_text())["development"]
    rows = Parallel(n_jobs=7, verbose=5)(delayed(run_subject)(s, TASKS[args.task]) for s in dev)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / f"B01_baselines_dev_{args.task}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nmean accuracy (development, N=%d):" % len(df))
    print(df.drop(columns=["subject", "n_test"]).mean().round(3).to_string())


if __name__ == "__main__":
    main()
