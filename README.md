# eeg-load-signature

Task-general and time-on-task-separated EEG workload signatures (COG-BCI MATB / N-back),
with NWCF network-topology, band-power and Riemannian feature families competing as inputs.

* Data are NOT in this repository. Set `EEGLS_CLEAN` to the folder of cleaned per-session `.npz` files.
* Subject split (development vs sealed confirmation) is fixed in `docs/subject_split.json`; sealed subjects
  are not touched before the pre-registered confirmation.
* `docs/DEVELOPMENT_LOG.md` records every development analysis and its outcome, including negative ones.

Layout: `eegls/` library (NWCF operator, baselines, connectivity, nulls) · `scripts/` analyses · `docs/` protocol and logs.
