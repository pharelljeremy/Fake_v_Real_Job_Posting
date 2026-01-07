import os
import logging
from pathlib import Path

# Project root (directory containing this file's parent or adjust as needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents and Path(__file__).resolve().parents[1]) else Path.cwd()

# Data directories (relative to project root)
RAW_DIR = PROJECT_ROOT / "raw_data"
CLEANED_DIR = PROJECT_ROOT / "cleaned_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Default cleaned filenames (used by train/analyze/visualise)
CLEANED_FILENAME = "jobs_clean.csv"
CLEANED_PII_FREE = "jobs_clean_pii_free.csv"

# Full paths
CLEANED_PATH = CLEANED_DIR / CLEANED_FILENAME
CLEANED_PII_PATH = CLEANED_DIR / CLEANED_PII_FREE

# Logging
LOG_FMT = "%(asctime)s %(levelname)s %(name)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("project")

def ensure_dirs():
    """
    Create required directories if they don't exist.
    Call at the start of scripts that write files.
    """
    for p in (RAW_DIR, CLEANED_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # best-effort: log and continue
            logging.getLogger("project").warning("Could not create dir %s: %s", p, e)

