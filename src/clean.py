import pandas as pd
import numpy as np
import re
from pathlib import Path
from src.config import CLEANED_PATH, CLEANED_PII_PATH, ensure_dirs, logger

ensure_dirs()

def _basic_text_prep(text: str) -> str:
    if pd.isna(text):
        return ""
    # simple cleanup: remove excessive whitespace, basic PII placeholders
    txt = str(text)
    txt = re.sub(r"\s+", " ", txt).strip()
    # optional: remove emails and phone numbers
    txt = re.sub(r"\S+@\S+\.\S+", " ", txt)  # email -> space
    txt = re.sub(r"\b(?:\+?\d[\d\-\s]{6,}\d)\b", " ", txt)  # crude phone removal
    return txt

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe and return cleaned df. This function:
    * creates a combined_text column used for modeling
    * computes text_length
    * drops exact duplicate rows
    """
    logger.info("Starting cleaning on dataframe with %d rows", len(df))
    df = df.copy()

    # Fill missing textual fields with empty string
    text_cols = [c for c in df.columns if c.lower() in ("title", "description", "job_description", "requirements", "company", "location")]
    # If typical columns not present, try to find any text-like columns
    if not text_cols:
        text_cols = [c for c in df.columns if df[c].dtype == object][:5]

    # Apply basic cleanup to each text column
    for c in text_cols:
        df[c] = df[c].apply(_basic_text_prep)

    # Create combined_text for modeling
    df["combined_text"] = df[text_cols].agg(" ".join, axis=1).fillna("").astype(str)

    # Compute text length
    df["text_length"] = df["combined_text"].str.len()

    # Drop exact duplicates on combined_text
    before = len(df)
    df = df.drop_duplicates(subset=["combined_text"])
    after = len(df)
    logger.info("Dropped %d duplicate rows", before - after)

    # Optionally drop columns with PII if present (email, phone columns)
    pii_cols = [c for c in df.columns if any(k in c.lower() for k in ("email", "phone", "contact"))]
    if pii_cols:
        logger.info("Dropping potential PII columns: %s", pii_cols)
        df_pii_free = df.drop(columns=pii_cols, errors="ignore")
    else:
        df_pii_free = df.copy()

    # Save cleaned files
    ensure_dirs()
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANED_PII_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    df_pii_free.to_csv(CLEANED_PII_PATH, index=False)
    logger.info("Saved cleaned data to %s and PII-free to %s", CLEANED_PATH, CLEANED_PII_PATH)

    return df

if __name__ == "__main__":
    # run a quick check if raw_data/raw_jobs.csv exists
    from src.ingest import load_raw
    df_raw = load_raw()
    df_clean = clean(df_raw)
    logger.info("Cleaning complete. Cleaned rows: %d", len(df_clean))

