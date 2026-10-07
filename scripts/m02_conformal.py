"""
M02 — Minimax task-general load decoder with a CONFORMAL (subject-level) certificate.

Core estimator = M01 (minimax over {population N-back, population MATB, own source-task} load directions in the
whitened PCA space; optional exact time-direction nulling).  The M01 hull certificate ||c*|| assumes the new
person's load direction lies in the hull; it does not (realised << certified).  M02 calibrates it:
  for target subject s, every other development subject j is treated in turn as a pseudo-target
  (population = dev \ {s, j}; j's own source task S1+S2; j's unseen task in S3 for evaluation), giving the
  nonconformity score  r_j = certified_j - realised_j.
  Conformal certificate for s:  cert_s - q,  q = ceil((n+1)(1-alpha))-th smallest r_j  (n = 10, alpha = 0.1).
Under exchangeability of subjects this gives P(realised_s >= conformal certificate) >= 1 - alpha (split-
conformal guarantee, finite-sample).  Realised separation for s uses s's test labels for EVALUATION only.

Usage: python scripts/m02_conformal.py [--family RIEM] [--K 100] [--alpha 0.1]
"""
import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import OAS
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import DOCS, OUT  # noqa: E402
from m01_minimax import build, cell_means, directions, fit_eval_score, min_norm_hull, time_dirs  # noqa: E402
from x10_crosstask_baselines import load  # noqa: E402


def core(allX, allM, target, pop, src, tgt, K, time_null):
    """Fit on population `pop` (both tasks, levels 0/2) + target's source task S1+S2; evaluate target's tgt S3."""
    own = (allM.subject == target) & (allM.task == src) & allM.session.isin(["S1", "S2"])
    tr = (allM.subject.isin(pop) | own) & allM.level.isin([0, 2])
    te = ((allM.subject == target) & (allM.task == tgt) & (allM.session == "S3")).to_numpy()
    Xtr, Mtr = allX[tr.to_numpy()], allM[tr].reset_index(drop=True)
    mu0 = Xtr.mean(0)
    sub = np.random.default_rng(int(target)).choice(len(Xtr), min(len(Xtr), 6000), replace=False)
    _, _, Vt = randomized_svd(Xtr[sub] - mu0, K, random_state=0)
    Ztr, Zte, Mte = (Xtr - mu0) @ Vt.T, (allX[te] - mu0) @ Vt.T, allM[te].reset_index(drop=True)
    cm = cell_means(Ztr, Mtr, ["subject", "session", "task", "level"])
    Sig = OAS().fit(Ztr - np.array([cm[k] for k in zip(Mtr.subject, Mtr.session, Mtr.task, Mtr.level)])).covariance_
    ev, evec = np.linalg.eigh(Sig)
    Wh = evec @ np.diag(ev ** -0.5) @ evec.T
    popM, ownM = (Mtr.subject != target).to_numpy(), (Mtr.subject == target).to_numpy()
    dp, do = directions(Ztr[popM], Mtr[popM]), directions(Ztr[ownM], Mtr[ownM])
    U = np.array([Wh @ m for m in (dp["nback"], dp["matb"], do[src])])
    if time_null:
        Q, _ = np.linalg.qr(Wh @ time_dirs(Ztr, Mtr).T)
        U = U - (U @ Q) @ Q.T
    c, a = min_norm_hull(U)
    w = Wh @ c
    tm = directions(Zte, Mte)[tgt]
    realised = float(w @ tm / np.sqrt(w @ Sig @ w))
    return dict(w=w, cert=float(np.linalg.norm(c)), realised=realised, score=Zte @ w, Mte=Mte, weights=a)


def run_target(s, dev, allX, allM, K, alpha):
    rows = []
    for src, tgt in (("nback", "matb"), ("matb", "nback")):
        for tn in (False, True):
            main = core(allX, allM, s, [j for j in dev if j != s], src, tgt, K, tn)
            r = [None] * 0
            for j in dev:
                if j == s:
                    continue
                cj = core(allX, allM, j, [k for k in dev if k not in (s, j)], src, tgt, K, tn)
                r.append(cj["cert"] - cj["realised"])
            r = np.sort(r); n = len(r)
            kq = ceil((n + 1) * (1 - alpha))
            q = r[kq - 1] if kq <= n else np.inf
            m = fit_eval_score(main["score"], main["Mte"])
            rows.append(dict(subject=s, method="M_MMT" if tn else "M_MM", train=src, test=tgt, **m,
                             hull_cert=main["cert"], conformal_cert=main["cert"] - q, realised=main["realised"],
                             covered=bool(main["realised"] >= main["cert"] - q), weights=np.round(main["weights"], 3).tolist()))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="RIEM"); ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    parts = [build(load(s), a.family) for s in dev]
    allX = np.vstack([p[0] for p in parts]); allM = pd.concat([p[1] for p in parts], ignore_index=True)
    rows = [r for rr in Parallel(n_jobs=4)(delayed(run_target)(s, dev, allX, allM, a.K, a.alpha) for s in dev) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"M02_conformal_dev_{a.family}_K{a.K}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["method", "train", "test"])[["auc", "rho", "time_leak", "hull_cert", "conformal_cert",
                                                   "realised", "covered"]].mean().round(3).to_string())
    print("\noverall coverage:", df.groupby("method").covered.mean().round(3).to_dict(), " target >=", 1 - a.alpha)
