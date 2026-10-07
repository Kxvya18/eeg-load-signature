"""Same as X03 but in sensor-space log relative band power (theta/alpha/beta per channel): fixed
coordinates, no whitening/re-centering, so a low across-session cosine cannot come from a
session-specific coordinate rotation. Each session's features are z-scored within session."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b01_baselines import ROOT, bandpower  # noqa: E402
from x01_block_confound import load  # noqa: E402
from x03_direction_stability import cos  # noqa: E402


def run(sid):
    D, half = {}, {}
    for ses in ("S1", "S2", "S3"):
        X, lvl, blk = load(sid, ses)
        F = bandpower(X); F = (F - F.mean(0)) / F.std(0)
        D[ses] = {"nback": F[lvl == "nback2"].mean(0) - F[lvl == "nback0"].mean(0),
                  "matb": F[lvl == "matb_diff"].mean(0) - F[lvl == "matb_easy"].mean(0)}
        half[ses] = [F[(lvl == "nback2") & (blk == b)].mean(0) - F[(lvl == "nback0") & (blk == b)].mean(0) for b in (0, 1)]
    return dict(subject=sid, nback_within=np.mean([cos(*half[s]) for s in half]),
                nback_across=np.mean([cos(D[a]["nback"], D[b]["nback"]) for a, b in combinations(D, 2)]),
                matb_across=np.mean([cos(D[a]["matb"], D[b]["matb"]) for a, b in combinations(D, 2)]))


if __name__ == "__main__":
    dev = json.loads((ROOT / "docs" / "subject_split.json").read_text())["development"]
    df = pd.DataFrame(Parallel(n_jobs=7)(delayed(run)(s) for s in dev))
    df.to_csv(ROOT / "outputs" / "X03b_direction_bandpower_dev.csv", index=False)
    print(df.round(3).to_string(index=False)); print("mean:\n", df.drop(columns="subject").mean().round(3).to_string())
