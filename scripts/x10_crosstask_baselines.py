"""
X10 — cross-task load decoding baselines per feature family (development subjects only).

Train on one task's low (level 0) vs high (level 2) windows, test on the other task.
  within : same session (S1, S2, S3 separately; averaged)
  cross  : train S1+S2, test S3
Features are z-scored per (session, task) using all windows of that run set (label-free).
Classifier: StandardScaler + L2 logistic regression, C fixed at 0.1 for every family (no family is tuned).
Metrics on the test task: AUC (low vs high); Spearman rho of decision score with level 0/1/2 (uses the
middle level never seen in training); time leakage = Spearman of score with time-on-task (slot*1e4 +
t_in_run) computed within each level and averaged (0 = no time information in the score).
References: within-task leave-one-session-out AUC for each task.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eegls.paths import CACHE, DOCS, OUT  # noqa: E402

FAMILIES = ("BP", "APER", "OSC", "RIEM", "NWCF", "TOPO")
TASKS = ("nback", "matb")


def load(sid):
    out = {}
    for ses in ("S1", "S2", "S3"):
        d = pickle.load(open(CACHE / f"F01_sub-{sid:02d}_{ses}.pkl", "rb"))
        m = d["meta"]
        feats = {f: d[f].copy() for f in FAMILIES}
        feats["ALL"] = np.hstack([feats[f] for f in FAMILIES])
        for t in TASKS:
            ix = (m.task == t).to_numpy()
            for f in feats:
                v = feats[f][ix]; sd = v.std(0); sd[sd == 0] = 1
                feats[f][ix] = (v - v.mean(0)) / sd
        out[ses] = (m.reset_index(drop=True), feats)
    return out


def fit_eval(Xtr, ytr, Xte, mte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xtr, ytr)
    s = clf.decision_function(Xte)
    lv = mte.level.to_numpy()
    hl = lv != 1
    tot = mte.slot.to_numpy() * 1e4 + mte.t_in_run.to_numpy()
    leak = np.nanmean([spearmanr(s[lv == L], tot[lv == L])[0] for L in (0, 1, 2)])
    return dict(auc=roc_auc_score(lv[hl] == 2, s[hl]), rho=spearmanr(s, lv)[0], time_leak=leak)


def pick(data, sessions, task, fam, levels=(0, 2)):
    X, M = [], []
    for ses in sessions:
        m, F = data[ses]
        ix = ((m.task == task) & m.level.isin(levels)).to_numpy()
        X.append(F[fam][ix]); M.append(m[ix])
    return np.vstack(X), pd.concat(M, ignore_index=True)


def run(sid):
    data = load(sid)
    rows = []
    for fam in FAMILIES + ("ALL",):
        for src, tgt in (("nback", "matb"), ("matb", "nback")):
            for ses in ("S1", "S2", "S3"):
                Xtr, mtr = pick(data, [ses], src, fam); Xte, mte = pick(data, [ses], tgt, fam, (0, 1, 2))
                rows.append(dict(subject=sid, family=fam, setting="within", train=src, test=tgt, session=ses,
                                 **fit_eval(Xtr, (mtr.level == 2).to_numpy(), Xte, mte)))
            Xtr, mtr = pick(data, ["S1", "S2"], src, fam); Xte, mte = pick(data, ["S3"], tgt, fam, (0, 1, 2))
            rows.append(dict(subject=sid, family=fam, setting="cross", train=src, test=tgt, session="S3",
                             **fit_eval(Xtr, (mtr.level == 2).to_numpy(), Xte, mte)))
        for t in TASKS:   # within-task reference, train S1+S2 test S3
            Xtr, mtr = pick(data, ["S1", "S2"], t, fam); Xte, mte = pick(data, ["S3"], t, fam, (0, 1, 2))
            rows.append(dict(subject=sid, family=fam, setting="same_task_cross_session", train=t, test=t, session="S3",
                             **fit_eval(Xtr, (mtr.level == 2).to_numpy(), Xte, mte)))
    return rows


if __name__ == "__main__":
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    df = pd.DataFrame([r for rr in Parallel(n_jobs=7)(delayed(run)(s) for s in dev) for r in rr])
    df.to_csv(OUT / "X10_crosstask_baselines_dev.csv", index=False)
    g = (df.groupby(["setting", "train", "test", "family"])[["auc", "rho", "time_leak"]].mean()
           .round(3).unstack("family"))
    pd.set_option("display.width", 250)
    for met in ("auc", "rho", "time_leak"):
        print(f"\n== {met}"); print(g[met].to_string())
