"""
F02 — RIEM_T: theta/alpha/beta OAS covariances re-centred at the Riemannian mean of each (session, task) run set
(all levels pooled, label-free) instead of the whole session. Same windows/order as F01; appended to the F01 cache.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from sklearn.covariance import OAS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eegls.baselines import riemann_mean, tangent_space  # noqa: E402
from eegls.paths import CACHE, CLEAN, DOCS  # noqa: E402

FS, WIN = 128, 512
BANDS = ((4, 8), (8, 13), (13, 30))


def run(sid, ses):
    p = CACHE / f"F01_sub-{sid:02d}_{ses}.pkl"
    d = pickle.load(open(p, "rb")); m = d["meta"]
    z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz")
    segs = {}
    for key in z.files:
        lvl, k = key.split("|"); segs.setdefault(lvl, {})[int(k)] = z[key].astype(float)
    X = []
    for r in m.itertuples():
        s = segs[r.level_name][r.block]
        off = int(round(r.t_in_run * FS)) - sum(len(segs[r.level_name][j]) for j in sorted(segs[r.level_name]) if j < r.block)
        X.append(s[off:off + WIN].T)
    X = np.array(X)
    out = np.zeros((len(m), 0))
    feats = []
    for lo, hi in BANDS:
        Xf = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), X, axis=-1)
        C = np.array([OAS().fit(x.T).covariance_ for x in Xf])
        T = np.zeros((len(m), C.shape[1] * (C.shape[1] + 1) // 2))
        for t in ("nback", "matb"):
            ix = (m.task == t).to_numpy()
            T[ix] = tangent_space(C[ix], riemann_mean(C[ix]))
        feats.append(T)
    d["RIEM_T"] = np.hstack(feats)
    pickle.dump(d, open(p, "wb"))
    return sid, ses, d["RIEM_T"].shape


if __name__ == "__main__":
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    for r in Parallel(n_jobs=7)(delayed(run)(s, ses) for s in dev for ses in ("S1", "S2", "S3")):
        print(*r, flush=True)
