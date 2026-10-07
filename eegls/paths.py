"""Data locations. Data never lives in the repository; override with environment variables."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLEAN = Path(os.environ.get("EEGLS_CLEAN", "/Volumes/Seagate/simulations/ALL_PROJECTS/stew/nwcf_v31/cache/clean"))
OUT = REPO / "outputs"
DOCS = REPO / "docs"
