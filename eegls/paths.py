"""Data locations. Data never lives in the repository; override with environment variables."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLEAN = Path(os.environ.get("EEGLS_CLEAN", "/Volumes/Seagate/simulations/ALL_PROJECTS/stew/nwcf_v31/cache/clean"))
OUT = REPO / "outputs"
DOCS = REPO / "docs"
NWCF_FEATURES = Path(os.environ.get("EEGLS_NWCF", "/Volumes/Seagate/simulations/ALL_PROJECTS/stew/nwcf_v31/outputs/features"))
NOTEBOOK = Path(os.environ.get("EEGLS_NOTEBOOK", "/Volumes/Seagate/simulations/ALL_PROJECTS/cogbci/raw/notebook.mat"))
CACHE = Path(os.environ.get("EEGLS_CACHE", "/Volumes/Seagate/simulations/ALL_PROJECTS/eegls_cache"))
