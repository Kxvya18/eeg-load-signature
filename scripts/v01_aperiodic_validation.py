"""V01 — validate the fast robust aperiodic fit (F01) against specparam (fixed mode) on a random
subsample of windows x channels from development subjects. Reports Pearson r and median abs diff."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from specparam import SpectralModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eegls.paths import CLEAN, OUT  # noqa: E402
from f01_features import FS, WIN, aperiodic_fit  # noqa: E402

rng = np.random.default_rng(20261007)
rows = []
for sid in (1, 4, 9, 16):
    z = np.load(CLEAN / f"sub-{sid:02d}_S1.npz")
    for key in rng.choice(z.files, 3, replace=False):
        s = z[key].astype(float)
        for w in rng.choice(len(s) // WIN, 5, replace=False):
            x = s[w * WIN:(w + 1) * WIN].T
            f, p = welch(x, fs=FS, nperseg=256, axis=-1)
            sel = (f >= 2) & (f <= 40)
            off, expo, _ = aperiodic_fit(f[sel], np.log10(p[:, sel]))
            for ch in rng.choice(x.shape[0], 6, replace=False):
                fm = SpectralModel(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
                fm.fit(f, p[ch], [2, 40])
                ap = fm.results.get_params("aperiodic")
                rows.append((off[ch], expo[ch], ap[0], ap[1]))
r = np.array(rows)
out = (f"n={len(r)}  offset r={np.corrcoef(r[:,0], r[:,2])[0,1]:.3f} median|d|={np.median(np.abs(r[:,0]-r[:,2])):.3f}  "
       f"exponent r={np.corrcoef(r[:,1], r[:,3])[0,1]:.3f} median|d|={np.median(np.abs(r[:,1]-r[:,3])):.3f}")
print(out); (OUT / "V01_aperiodic_validation.txt").write_text(out + "\n")
