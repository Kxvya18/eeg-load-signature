"""
SOTA deep-learning baselines (PREREGISTRATION_v31.md, amendment A2).

EEGNet-8,2   Lawhern et al., J. Neural Eng. 15:056013 (2018)
ShallowConvNet  Schirrmeister et al., Hum. Brain Mapp. 38:5391 (2017)
Published hyperparameters; temporal kernel/pool lengths scaled to 128 Hz.
Training: Adam 1e-3, batch 64, <= 100 epochs, early stopping (patience 10) on one held-out
group of training subjects; per-channel z-scoring with training statistics.
Recording score = mean window logit.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

_MPS = torch.backends.mps.is_available()
MODEL_DEVICE = {"EEGNet": torch.device("mps" if _MPS else "cpu"),   # fastest per profiling (A3)
                "ShallowConvNet": torch.device("cpu")}


class EEGNet(nn.Module):
    def __init__(self, n_ch, n_t, f1=8, d=2, f2=16, k1=64, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * d), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n_feat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_feat, 1)

    def forward(self, x):
        return self.fc(self.net(x)).squeeze(1)


class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    def __init__(self, n_ch, n_t, nf=40, k=13, pool=38, stride=8, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, nf, (1, k)), nn.Conv2d(nf, nf, (n_ch, 1), bias=False), nn.BatchNorm2d(nf),
            Square(), nn.AvgPool2d((1, pool), stride=(1, stride)), SafeLog(), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n_feat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_feat, 1)

    def forward(self, x):
        return self.fc(self.net(x)).squeeze(1)


MODELS = {"EEGNet": EEGNet, "ShallowConvNet": ShallowConvNet}


def _train(model, xtr, ytr, xva, yva, seed, device, max_epochs=100, patience=10, batch=64):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    xva_t = torch.tensor(xva, device=device); yva_t = torch.tensor(yva, device=device)
    best, best_state, wait = np.inf, None, 0
    n = len(xtr)
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).numpy()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.tensor(xtr[idx], device=device); yb = torch.tensor(ytr[idx], device=device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([lossf(model(xva_t[i:i + 256]), yva_t[i:i + 256]).item()
                               for i in range(0, len(xva), 256)]))
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def subject_folds(subjects, k=5):
    """Fixed grouped k-fold over sorted subject ids (shared by nets and feature models)."""
    subjects = np.sort(np.unique(subjects))
    return [subjects[te] for _, te in GroupKFold(k).split(subjects, groups=subjects)]


def cv_scores(X, y_win, rec_id, groups_win, model_name, seed=0, k=5, verbose=True, ckpt_dir=None):
    """X: [n_windows, n_ch, n_t] float32; y_win, rec_id, groups_win per window.
    Returns {rec_id: mean logit} from grouped k-fold training (A3)."""
    out = {}
    device = MODEL_DEVICE[model_name]
    if device.type == "cpu":
        torch.set_num_threads(6)                 # speed only; EEGNet runs on MPS concurrently
    folds = subject_folds(groups_win, k)
    for fi, held in enumerate(folds):
        ck = (ckpt_dir / f"{model_name}_fold{fi}.json") if ckpt_dir is not None else None
        if ck is not None and ck.exists():                  # resume: fold already trained
            out.update(json.loads(ck.read_text()))
            continue
        tr, te = np.where(~np.isin(groups_win, held))[0], np.where(np.isin(groups_win, held))[0]
        a, b = next(GroupKFold(5).split(tr, y_win[tr], groups_win[tr]))
        fit_i, val_i = tr[a], tr[b]
        mu = X[fit_i].mean(axis=(0, 2), keepdims=True); sd = X[fit_i].std(axis=(0, 2), keepdims=True) + 1e-6
        z = lambda i: ((X[i] - mu) / sd)[:, None].astype(np.float32)
        torch.manual_seed(seed + fi); np.random.seed(seed + fi)
        model = MODELS[model_name](X.shape[1], X.shape[2]).to(device)
        model = _train(model, z(fit_i), y_win[fit_i].astype(np.float32), z(val_i),
                       y_win[val_i].astype(np.float32), seed + fi, device)
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(z(te), device=device)
            logits = np.concatenate([model(xt[i:i + 256]).cpu().numpy() for i in range(0, len(te), 256)])
        fold_out = {str(r): float(logits[rec_id[te] == r].mean()) for r in np.unique(rec_id[te])}
        out.update(fold_out)
        if ck is not None:
            ck.write_text(json.dumps(fold_out))
        if verbose:
            print(f"  {model_name} fold {fi + 1}/{len(folds)} done", flush=True)
    return out
