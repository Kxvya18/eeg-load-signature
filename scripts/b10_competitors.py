"""
B10 — competitor methods on exactly the M01 scenario (target subject, unseen task, S3; training = population
both tasks + own source task S1+S2, levels 0/2), same features and the same K-dim PCA of the training windows.
  PCA_LR  pooled logistic regression in the PCA space (checks the PCA itself is not the reason for M01's gain)
  GDRO    group-DRO logistic regression (Sagawa et al. 2020): groups = subject x task, exponentiated-gradient
          group weights (eta = 1, 30 rounds), worst-group risk minimised
  CORAL   every training run (subject x session x task) re-coloured to the covariance of the unlabeled target
          run (Sun et al. 2016), then pooled logistic regression
  MM      M01 minimax (reference, recomputed here)
All logistic regressions: StandardScaler + L2, C = 0.1 (fixed, as everywhere in this project).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import sqrtm
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, cell_means, directions, fit_eval_score, min_norm_hull  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402


def lr(X, y, w=None):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X), y, sample_weight=w)


def gdro(X, y, groups, rounds=30, eta=1.0):
    g_ids = np.unique(groups); q = np.ones(len(g_ids)) / len(g_ids)
    gi = np.searchsorted(g_ids, groups); cnt = np.bincount(gi)
    for _ in range(rounds):
        sc, m = lr(X, y, w=q[gi] / cnt[gi] * len(y))
        p = np.clip(m.predict_proba(sc.transform(X))[:, 1], 1e-6, 1 - 1e-6)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        gl = np.bincount(gi, weights=loss) / cnt
        q = q * np.exp(eta * gl); q /= q.sum()
    return sc, m


def recolor(Z, Ct):
    Cs = OAS().fit(Z).covariance_
    A = np.real(sqrtm(np.linalg.inv(Cs))) @ np.real(sqrtm(Ct))
    return (Z - Z.mean(0)) @ A


def run_target(sid, allX, allM, K):
    rows = []
    pop = allM.subject != sid
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        own = (allM.subject == sid) & (allM.task == src) & allM.session.isin(["S1", "S2"])
        tr = ((pop | own) & allM.level.isin([0, 2])).to_numpy()
        te = ((allM.subject == sid) & (allM.task == tgt) & (allM.session == "S3")).to_numpy()
        Xtr, Mtr, Mte = allX[tr], allM[tr].reset_index(drop=True), allM[te].reset_index(drop=True)
        mu0 = Xtr.mean(0)
        sub = np.random.default_rng(sid).choice(len(Xtr), min(len(Xtr), 6000), replace=False)
        _, _, Vt = randomized_svd(Xtr[sub] - mu0, K, random_state=0)
        Ztr, Zte = (Xtr - mu0) @ Vt.T, (allX[te] - mu0) @ Vt.T
        y = (Mtr.level == 2).to_numpy().astype(int)
        out = {}
        sc, m = lr(Ztr, y); out["PCA_LR"] = m.decision_function(sc.transform(Zte))
        groups = (Mtr.subject.astype(str) + "_" + Mtr.task).to_numpy()
        sc, m = gdro(Ztr, y, groups); out["GDRO"] = m.decision_function(sc.transform(Zte))
        Ct = OAS().fit(Zte).covariance_
        Zc = np.zeros_like(Ztr)
        for _, ix in Mtr.groupby(["subject", "session", "task"]).indices.items():
            Zc[ix] = recolor(Ztr[ix], Ct)
        sc, m = lr(Zc, y); out["CORAL"] = m.decision_function(sc.transform(Zte - Zte.mean(0)))
        cm = cell_means(Ztr, Mtr, ["subject", "session", "task", "level"])
        Sig = OAS().fit(Ztr - np.array([cm[k] for k in zip(Mtr.subject, Mtr.session, Mtr.task, Mtr.level)])).covariance_
        ev, evec = np.linalg.eigh(Sig); Wh = evec @ np.diag(ev ** -0.5) @ evec.T
        pm, om = (Mtr.subject != sid).to_numpy(), (Mtr.subject == sid).to_numpy()
        dp, do = directions(Ztr[pm], Mtr[pm]), directions(Ztr[om], Mtr[om])
        c, _ = min_norm_hull(np.array([Wh @ v for v in (dp["nback"], dp["matb"], do[src])]))
        out["MM"] = Zte @ (Wh @ c)
        for name, s in out.items():
            rows.append(dict(subject=sid, method=name, train=src, test=tgt, **fit_eval_score(s, Mte)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="RIEM"); ap.add_argument("--K", type=int, default=100)
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), a.family) for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    df = pd.DataFrame([r for rr in Parallel(n_jobs=4)(delayed(run_target)(s, allX, allM, a.K) for s in dev) for r in rr])
    df.to_csv(OUT / f"B10_competitors_dev_{a.family}_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("auc", "rho", "time_leak"):
        print(f"\n== {met}"); print(df.groupby(["train", "test", "method"])[met].mean().unstack("method").round(3).to_string())
