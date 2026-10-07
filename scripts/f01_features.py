"""
F01 — window-level feature families for the load-signature study (4-s windows, identical to the frozen
NWCF v3.1 windowing: per level, blocks in sorted order, non-overlapping 512-sample windows).

Families (one row per window):
  BP    log relative band power theta/alpha/beta per channel           (NWCF v3.1 cache, unchanged)
  APER  aperiodic offset + exponent per channel (robust log-log fit 2-40 Hz with iterative peak exclusion)
  OSC   periodic power above the aperiodic fit, theta/alpha/beta per channel
  RIEM  theta/alpha/beta OAS covariances -> tangent space at the session Riemannian mean (re-centred)
  NWCF  frozen NWCF v3.1 window descriptors, aec_orth + wpli, theta/alpha/beta, M58 (cache, unchanged)
  TOPO  classical Rips PH + graph + clique-topology descriptors from the same cache (closest topology baseline)
Metadata per window: subject, session, task (nback|matb), level 0/1/2, block, slot (position of the
level's run in the session order from notebook.mat; 0..7), t_in_run (s), t_session_rank.

Usage:  python scripts/f01_features.py [--subjects dev|all|1,3,...] [--jobs 7]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.covariance import OAS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eegls.baselines import riemann_mean, tangent_space  # noqa: E402
from eegls.paths import CACHE, CLEAN, DOCS, NOTEBOOK, NWCF_FEATURES  # noqa: E402

FS, WIN = 128, 512
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
LEVELS = {"nback0": ("nback", 0), "nback1": ("nback", 1), "nback2": ("nback", 2),
          "matb_easy": ("matb", 0), "matb_med": ("matb", 1), "matb_diff": ("matb", 2)}
NOTEBOOK_CODE = {"nback2": 3, "nback1": 4, "nback0": 5, "matb_easy": 6, "matb_med": 7, "matb_diff": 8}


def session_order(sid, ses):
    nb = loadmat(NOTEBOOK, squeeze_me=True, struct_as_record=False)["notebook"]
    order = list(getattr(getattr(nb, f"SBJ_{sid}"), f"SESS_{ses[-1]}").Order)
    return {lvl: order.index(code) for lvl, code in NOTEBOOK_CODE.items()}


def aperiodic_fit(f, logp, n_iter=3, thr=0.1):
    """Robust 1/f fit in log-log space, vectorised. logp [..., F] (log10 power).
    Iteratively excludes frequencies whose residual exceeds thr (log10 units) above the line (peaks)."""
    x = np.log10(f)
    mask = np.ones(logp.shape, bool)
    for _ in range(n_iter):
        w = mask.astype(float)
        sw = w.sum(-1); mx = (w * x).sum(-1) / sw; my = (w * logp).sum(-1) / sw
        cov = (w * (x - mx[..., None]) * (logp - my[..., None])).sum(-1)
        var = (w * (x - mx[..., None]) ** 2).sum(-1)
        slope = cov / var; off = my - slope * mx
        resid = logp - (off[..., None] + slope[..., None] * x)
        mask = resid < thr
    return off, -slope, resid


def run(sid, ses):
    z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz")
    segs = {}
    for key in z.files:
        lvl, k = key.split("|")
        segs.setdefault(lvl, {})[int(k)] = z[key].astype(float)
    slot = session_order(sid, ses)
    X, meta = [], []
    for lvl in segs:
        wi = 0
        for k in sorted(segs[lvl]):
            s = segs[lvl][k]
            for w in range(len(s) // WIN):
                X.append(s[w * WIN:(w + 1) * WIN].T)
                meta.append(dict(subject=sid, session=ses, level_name=lvl, task=LEVELS[lvl][0],
                                 level=LEVELS[lvl][1], block=k, window=wi, slot=slot[lvl],
                                 t_in_run=(sum(len(segs[lvl][j]) for j in sorted(segs[lvl]) if j < k) + w * WIN) / FS))
                wi += 1
    X = np.array(X); meta = pd.DataFrame(meta)
    # spectra
    f, p = welch(X, fs=FS, nperseg=256, axis=-1)
    sel = (f >= 2) & (f <= 40)
    logp = np.log10(p[..., sel])
    off, expo, resid = aperiodic_fit(f[sel], logp)
    fs_ = f[sel]
    osc = np.concatenate([resid[..., (fs_ >= lo) & (fs_ < hi)].clip(min=0).mean(-1) for lo, hi in BANDS.values()], axis=1)
    aper = np.concatenate([off, expo], axis=1)
    # Riemannian (re-centred per session)
    riem = []
    for lo, hi in BANDS.values():
        Xf = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), X, axis=-1)
        C = np.array([OAS().fit(x.T).covariance_ for x in Xf])
        riem.append(tangent_space(C, riemann_mean(C)))
    riem = np.concatenate(riem, axis=1)
    # NWCF / BP / classical topology from the frozen cache
    d = pickle.load(open(NWCF_FEATURES / f"sub-{sid:02d}_{ses}.pkl", "rb"))
    wdf = d["windows"]; wdf = wdf[wdf.montage == "M58"]
    nwcf_cols = [c for c in wdf.columns if c.startswith("nwcf_")]
    topo_cols = [c for c in wdf.columns if c.startswith(("rips_", "graph_", "ct_"))]
    wide = wdf.pivot_table(index=["level", "window"], columns=["band", "fc"], values=nwcf_cols + topo_cols)
    wide.columns = [f"{v}|{b}|{fc}" for v, b, fc in wide.columns]
    bp = d["bp"].set_index(["level", "window"])[[c for c in d["bp"].columns if c.startswith("bp_")]]
    key = pd.MultiIndex.from_frame(meta[["level_name", "window"]])
    nw = wide.reindex(key); bpv = bp.reindex(key)
    assert not nw.isna().any().any() and not bpv.isna().any().any(), f"cache misalignment sub-{sid} {ses}"
    out = dict(meta=meta,
               BP=bpv.to_numpy(float), APER=aper, OSC=osc, RIEM=riem,
               NWCF=nw[[c for c in nw.columns if c.startswith("nwcf_")]].to_numpy(float),
               TOPO=nw[[c for c in nw.columns if not c.startswith("nwcf_")]].to_numpy(float))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"F01_sub-{sid:02d}_{ses}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return sid, ses, {k: v.shape for k, v in out.items() if k != "meta"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="dev"); ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    split = json.loads((DOCS / "subject_split.json").read_text())
    subs = split["development"] if a.subjects == "dev" else (
        split["development"] + split["sealed_test"] if a.subjects == "all" else [int(s) for s in a.subjects.split(",")])
    for r in Parallel(n_jobs=a.jobs, verbose=0)(delayed(run)(s, ses) for s in subs for ses in ("S1", "S2", "S3")):
        print(*r, flush=True)
