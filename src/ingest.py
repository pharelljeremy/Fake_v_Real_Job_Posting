from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, ensure_dirs, logger

ensure_dirs()

# Update RAW_FILENAME to whatever your raw file is called (example: raw_jobs.csv)
RAW_FILENAME = "raw_jobs.csv"
RAW_PATH = RAW_DIR / RAW_FILENAME

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load raw data CSV into a DataFrame. If path doesn't exist, raise an informative error.
    """
    if not path.exists():
        logger.error("Raw data file not found: %s", path)
        raise FileNotFoundError(f"Raw data not found at {path}. Place your raw CSV at this location.")
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df

if __name__ == "__main__":
    df = load_raw()
    # Save a copy into raw_data/raw_jobs_loaded.csv for reproducibility
    out = RAW_DIR / "raw_jobs_loaded.csv"
    df.to_csv(out, index=False)
    logger.info("Saved a copy of loaded raw data to %s", out)

