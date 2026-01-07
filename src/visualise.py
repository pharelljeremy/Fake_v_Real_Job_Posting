import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import CLEANED_PATH, OUTPUTS_DIR, FIGURES_DIR, ensure_dirs, logger

ensure_dirs()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_clean(path: Path = CLEANED_PATH) -> pd.DataFrame:
    if not path.exists():
        logger.error("Cleaned file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def plot_class_balance(df: pd.DataFrame):
    if "is_fake" not in df.columns:
        logger.warning("No is_fake column present; skipping class balance plot")
        return
    counts = df["is_fake"].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title("Fake vs Real job posts")
    plt.xlabel("is_fake")
    plt.ylabel("count")
    path = FIGURES_DIR / "class_balance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)

def plot_text_length(df: pd.DataFrame):
    if "text_length" not in df.columns:
        logger.warning("No text_length column present; skipping text length plot")
        return
    plt.figure(figsize=(8,4))
    sns.histplot(df["text_length"].dropna(), bins=40, kde=True)
    plt.title("Combined text length distribution")
    path = FIGURES_DIR / "text_length_dist.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)

def run_visuals():
    ensure_dirs()
    df = load_clean()
    plot_class_balance(df)
    plot_text_length(df)

if __name__ == "__main__":
    run_visuals()

