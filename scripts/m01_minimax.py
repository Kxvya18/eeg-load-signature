"""
M01 — Minimax task-general load direction with time-on-task nulling (development subjects only).

Scenario: target subject s, unseen task B, test session S3. Available for training:
  * subject s, source task A, sessions S1+S2, levels 0 vs 2 (labels)
  * population = all OTHER development subjects, both tasks, all sessions, levels 0 vs 2 (labels)
No data of subject s on task B is used for fitting anything except the label-free per-run z-scoring (as X10).

Method (in a K-dim PCA space of the training windows):
  Sigma : pooled within-cell covariance (cell = subject x session x task x level), OAS shrinkage
  mu_t  : load directions (mean level 2 - mean level 0): pop-nback, pop-matb, own-source-task
  tau   : time-on-task directions, identifiable because run order is randomised per session:
          tau_in  = pooled within-run slope of features on time within run
          tau_run = coefficient of run slot in  run_mean ~ level dummies(task) + slot   (population)
  Whitened u_t = Sigma^-1/2 mu_t ; (M_TIME) project u_t onto the complement of span(Sigma^-1/2 tau).
  c* = minimum-norm point of conv{u_t}  ->  w = Sigma^-1/2 c*.
  Guarantee (minimax / convex duality): for every task whose whitened load direction u lies in conv{u_t},
  the standardised separation w'mu / sqrt(w'Sigma w) >= ||c*||; with M_TIME, w'tau = 0 exactly.
Compared on the same features and training data:
  B_SUBJ  logistic regression on own source task only (= X10 'cross')
  B_POOL  logistic regression on population (both tasks) + own source task
  B_MEAN  pooled LDA: w = Sigma^-1 (sum of mu_t)    (ablation: no minimax)
  M_MM    minimax, no time nulling
  M_MMT   minimax + time nulling
Metrics on subject s, task B, S3: AUC (0 vs 2), Spearman rho with level 0/1/2, time leakage, and the
certified margin ||c*|| vs the realised standardised separation on the test task.

Usage: python scripts/m01_minimax.py [--families ALL,RIEM,BP,ALL_noNWCF,NWCF] [--K 100]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from x10_crosstask_baselines import FAMILIES, fit_eval, load  # noqa: E402

FAM_SETS = {"ALL": FAMILIES, "ALL_noNWCF": tuple(f for f in FAMILIES if f != "NWCF")}


def build(data, fam):
    cols = FAM_SETS.get(fam, (fam,))
    X, M = [], []
    for ses, (m, F) in data.items():
        X.append(np.hstack([F[c] for c in cols]).astype(np.float32)); M.append(m)
    return np.vstack(X), pd.concat(M, ignore_index=True)


def min_norm_hull(U):
    """Minimum-norm point of the convex hull of the rows of U (simplex QP)."""
    k = len(U); G = U @ U.T
    res = minimize(lambda a: a @ G @ a, np.full(k, 1 / k), jac=lambda a: 2 * G @ a, method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
    a = np.clip(res.x, 0, None); a /= a.sum()
    return a @ U, a


def cell_means(Z, M, keys):
    g = M.groupby(keys).indices
    return {k: Z[ix].mean(0) for k, ix in g.items()}


def directions(Z, M):
    """Load directions per task (mean over subject x session of mean2 - mean0)."""
    cm = cell_means(Z, M, ["subject", "session", "task", "level"])
    out = {}
    for t in ("nback", "matb"):
        d = [cm[(s, ses, t, 2)] - cm[(s, ses, t, 0)] for (s, ses, tt, lv) in cm if tt == t and lv == 0
             and (s, ses, t, 2) in cm]
        if d:
            out[t] = np.mean(d, 0)
    return out


def time_dirs(Z, M):
    """Within-run slope on time, and between-run slot coefficient (level effects removed per task)."""
    slopes = []
    for _, ix in M.groupby(["subject", "session", "task", "level"]).indices.items():
        t = M.t_in_run.to_numpy()[ix]; t = t - t.mean()
        if (t @ t) > 0:
            slopes.append(t @ (Z[ix] - Z[ix].mean(0)) / (t @ t))
    tau_in = np.mean(slopes, 0)
    run = M.groupby(["subject", "session", "task", "level"]).agg(slot=("slot", "first")).reset_index()
    R = np.array([Z[ix].mean(0) for ix in M.groupby(["subject", "session", "task", "level"]).indices.values()])
    D = pd.get_dummies(run.task + "_" + run.level.astype(str)).to_numpy(float)
    A = np.column_stack([D, run.slot.to_numpy(float)])
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    return np.vstack([tau_in, coef[-1]])


def run_target(sid, allX, allM, K, fam):
    rows = []
    pop = allM.subject != sid
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        own = (allM.subject == sid) & (allM.task == src) & allM.session.isin(["S1", "S2"])
        tr = (pop | own) & allM.level.isin([0, 2])
        te = ((allM.subject == sid) & (allM.task == tgt) & (allM.session == "S3")).to_numpy()
        Xtr, Mtr = allX[tr.to_numpy()], allM[tr].reset_index(drop=True)
        mu0 = Xtr.mean(0)
        sub = np.random.default_rng(sid).choice(len(Xtr), min(len(Xtr), 6000), replace=False)
        _, _, Vt = randomized_svd(Xtr[sub] - mu0, K, random_state=0)
        P = lambda X: (X - mu0) @ Vt.T
        Ztr, Zte, Mte = P(Xtr), P(allX[te]), allM[te].reset_index(drop=True)
        cm = cell_means(Ztr, Mtr, ["subject", "session", "task", "level"])
        keys = list(zip(Mtr.subject, Mtr.session, Mtr.task, Mtr.level))
        Sig = OAS().fit(Ztr - np.array([cm[k] for k in keys])).covariance_
        ev, evec = np.linalg.eigh(Sig)
        Wh = evec @ np.diag(ev ** -0.5) @ evec.T                      # Sigma^-1/2
        Sh = evec @ np.diag(ev ** 0.5) @ evec.T                       # Sigma^+1/2
        popM, ownM = (Mtr.subject != sid).to_numpy(), (Mtr.subject == sid).to_numpy()
        dp = directions(Ztr[popM], Mtr[popM]); do = directions(Ztr[ownM], Mtr[ownM])
        mus = {"pop_nback": dp["nback"], "pop_matb": dp["matb"], f"own_{src}": do[src]}
        U = np.array([Wh @ m for m in mus.values()])
        Tw = Wh @ time_dirs(Ztr, Mtr).T                                # whitened time directions [K, 2]
        Q, _ = np.linalg.qr(Tw)
        U_t = U - (U @ Q) @ Q.T
        res = {}
        c, a = min_norm_hull(U); res["M_MM"] = (Wh @ c, np.linalg.norm(c), a)
        c, a = min_norm_hull(U_t); res["M_MMT"] = (Wh @ c, np.linalg.norm(c), a)
        res["B_MEAN"] = (Wh @ (Wh @ sum(mus.values())), np.nan, None)
        # realised standardised separation of the test task (uses test labels for EVALUATION only)
        tm = directions(Zte, Mte)[tgt]
        for name, (w, cert, a) in res.items():
            r = fit_eval_score(Zte @ w, Mte)
            r.update(certified=cert, realised=float(w @ tm / np.sqrt(w @ Sig @ w)),
                     time_corr_w=float(np.abs((Sh @ w) @ Q).max() / np.linalg.norm(Sh @ w)),   # |cos| of c=Sigma^1/2 w with whitened time dirs
                     weights=None if a is None else np.round(a, 3).tolist())
            rows.append(dict(subject=sid, family=fam, method=name, train=src, test=tgt, **r))
        ytr = (Mtr.level == 2).to_numpy()
        for name, mask in (("B_SUBJ", ownM), ("B_POOL", np.ones(len(Mtr), bool))):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xtr[mask], ytr[mask])
            r = fit_eval_score(clf.decision_function(allX[te]), Mte)
            rows.append(dict(subject=sid, family=fam, method=name, train=src, test=tgt, **r))
    return rows


def fit_eval_score(s, mte):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    lv = mte.level.to_numpy(); hl = lv != 1
    tot = mte.slot.to_numpy() * 1e4 + mte.t_in_run.to_numpy()
    return dict(auc=roc_auc_score(lv[hl] == 2, s[hl]), rho=spearmanr(s, lv)[0],
                time_leak=np.nanmean([spearmanr(s[lv == L], tot[lv == L])[0] for L in (0, 1, 2)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ALL,RIEM,BP,ALL_noNWCF,NWCF"); ap.add_argument("--K", type=int, default=100)
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    data = {s: load(s) for s in dev}
    rows = []
    for fam in a.families.split(","):
        parts = [build(data[s], fam) for s in dev]
        allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
        rows += [r for rr in Parallel(n_jobs=4)(delayed(run_target)(s, allX, allM, a.K, fam) for s in dev) for r in rr]
        print(fam, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"M01_minimax_dev_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    for met in ("auc", "rho", "time_leak"):
        print(f"\n== {met}")
        print(df.groupby(["family", "train", "test", "method"])[met].mean().unstack("method").round(3).to_string())
    print("\n== certified vs realised (minimax)")
    print(df[df.method.isin(["M_MM", "M_MMT"])].groupby(["family", "method"])[["certified", "realised", "time_corr_w"]]
          .mean().round(3).to_string())
