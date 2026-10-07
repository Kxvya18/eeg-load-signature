"""
B20 — deep baselines (EEGNet-8,2; ShallowConvNet) in the M01 scenario: target subject, unseen task, S3.
Training windows: population (other dev subjects, both tasks, all sessions) + target's source task S1+S2,
levels 0 vs 2; raw 4-s windows (58 ch x 512), identical to the feature windows (F01 order).
Normalisation: per-channel z-scoring within each (subject, session, task) run set (label-free, mirrors the
feature pipeline). Early stopping on 2 held-out population subjects (fixed per target, seeded).
Architecture/training exactly as eegls/deep.py (published hyperparameters; Adam 1e-3, batch 64, patience 10).
Results checkpointed per (model, target, direction) for resume.

Usage: python scripts/b20_deep.py --model EEGNet|ShallowConvNet
"""
import torch  # noqa: F401  (import first: avoids the torch/sklearn segfault seen on this machine)
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.deep import MODEL_DEVICE, MODELS, _train  # noqa: E402
from eegls.paths import CACHE, CLEAN, DOCS, OUT  # noqa: E402
from m01_minimax import fit_eval_score  # noqa: E402
import pickle  # noqa: E402


def raw_windows(sid):
    X, M = [], []
    for ses in ("S1", "S2", "S3"):
        m = pickle.load(open(CACHE / f"F01_sub-{sid:02d}_{ses}.pkl", "rb"))["meta"]
        z = np.load(CLEAN / f"sub-{sid:02d}_{ses}.npz"); segs = {}
        for key in z.files:
            l, k = key.split("|"); segs.setdefault(l, {})[int(k)] = z[key].astype(np.float32)
        W = []
        for r in m.itertuples():
            off = int(round(r.t_in_run * 128)) - sum(len(segs[r.level_name][j]) for j in sorted(segs[r.level_name]) if j < r.block)
            W.append(segs[r.level_name][r.block][off:off + 512].T)
        W = np.array(W)
        for t in ("nback", "matb"):
            ix = (m.task == t).to_numpy()
            W[ix] = (W[ix] - W[ix].mean(axis=(0, 2), keepdims=True)) / (W[ix].std(axis=(0, 2), keepdims=True) + 1e-6)
        X.append(W); M.append(m)
    return np.concatenate(X), pd.concat(M, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="EEGNet")
    a = ap.parse_args()
    dev = json.loads((DOCS / "subject_split.json").read_text())["development"]
    device = MODEL_DEVICE[a.model]
    if device.type == "cpu":
        torch.set_num_threads(6)
    ck = CACHE / "b20_ckpt"; ck.mkdir(parents=True, exist_ok=True)
    parts = {s: raw_windows(s) for s in dev}
    print("loaded", {s: p[0].shape for s, p in parts.items()}, flush=True)
    rows = []
    for sid in dev:
        for src, tgt in (("nback", "matb"), ("matb", "nback")):
            f = ck / f"{a.model}_sub-{sid:02d}_{src}-{tgt}.json"
            if f.exists():
                rows.append(json.loads(f.read_text())); continue
            pops = [j for j in dev if j != sid]
            val_subs = list(np.random.default_rng(sid).choice(pops, 2, replace=False))
            Xs, ys, Xv, yv = [], [], [], []
            for j in dev:
                X, M = parts[j]
                keep = M.level.isin([0, 2]).to_numpy()
                if j == sid:
                    keep &= ((M.task == src) & M.session.isin(["S1", "S2"])).to_numpy()
                (Xv if j in val_subs else Xs).append(X[keep]); (yv if j in val_subs else ys).append((M.level[keep] == 2).to_numpy())
            Xtr, ytr = np.concatenate(Xs)[:, None], np.concatenate(ys).astype(np.float32)
            Xva, yva = np.concatenate(Xv)[:, None], np.concatenate(yv).astype(np.float32)
            torch.manual_seed(sid); np.random.seed(sid)
            model = MODELS[a.model](Xtr.shape[2], Xtr.shape[3]).to(device)
            model = _train(model, Xtr, ytr, Xva, yva, sid, device)
            X, M = parts[sid]
            te = ((M.task == tgt) & (M.session == "S3")).to_numpy()
            model.eval()
            with torch.no_grad():
                xt = torch.tensor(X[te][:, None], device=device)
                s = np.concatenate([model(xt[i:i + 256]).cpu().numpy() for i in range(0, len(xt), 256)])
            r = dict(subject=sid, method=a.model, train=src, test=tgt, **fit_eval_score(s, M[te].reset_index(drop=True)))
            r = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}
            f.write_text(json.dumps(r)); rows.append(r)
            print(sid, src, "->", tgt, round(r["auc"], 3), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / f"B20_{a.model}_dev.csv", index=False)
    print(df.groupby(["train", "test"])[["auc", "rho", "time_leak"]].mean().round(3).to_string())
