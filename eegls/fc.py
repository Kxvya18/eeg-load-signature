"""Functional-connectivity operators on analytic (Hilbert) signals, shape [T, N].

aec_orth follows the pairwise orthogonalization of Hipp et al. (2012) as
implemented in MNE `envelope_correlation(orthogonalize="pairwise")`:
Y_perp = imag(Y * conj(X) / |X|), corr(|X|, |Y_perp|), symmetrized.
wpli follows Vinck et al. (2011) over time samples of the window.
All operators return similarity in [0, 1] with unit diagonal.
"""
from __future__ import annotations

import numpy as np


def _corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0); b = b - b.mean(0)
    num = a.T @ b
    den = np.sqrt((a ** 2).sum(0))[:, None] * np.sqrt((b ** 2).sum(0))[None, :]
    return num / np.where(den > 0, den, np.inf)


def _finish(m: np.ndarray) -> np.ndarray:
    m = 0.5 * (m + m.T)
    m = np.clip(np.nan_to_num(m), 0.0, 1.0)
    np.fill_diagonal(m, 1.0)
    return m


def aec(z: np.ndarray) -> np.ndarray:
    env = np.abs(z)
    return _finish(_corr_cols(env, env))


def aec_orth(z: np.ndarray) -> np.ndarray:
    n = z.shape[1]
    env = np.abs(z)
    out = np.zeros((n, n))
    for i in range(n):
        unit = np.conj(z[:, i]) / np.where(env[:, i] > 0, env[:, i], np.inf)
        y_perp = np.abs(np.imag(z * unit[:, None]))          # [T, N], all j orthogonalized to i
        out[i] = _corr_cols(env[:, [i]], y_perp)[0]
    return _finish(out)


def plv(z: np.ndarray) -> np.ndarray:
    u = z / np.where(np.abs(z) > 0, np.abs(z), np.inf)
    return _finish(np.abs(u.T @ np.conj(u)) / z.shape[0])


def wpli(z: np.ndarray) -> np.ndarray:
    n = z.shape[1]
    out = np.zeros((n, n))
    for i in range(n):
        im = np.imag(z[:, [i]] * np.conj(z))                  # [T, N]
        den = np.abs(im).sum(0)
        out[i] = np.abs(im.sum(0)) / np.where(den > 0, den, np.inf)
    return _finish(out)


FC_OPERATORS = {"aec_orth": aec_orth, "wpli": wpli, "aec": aec, "plv": plv}
