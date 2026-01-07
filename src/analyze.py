import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import CLEANED_PATH, OUTPUTS_DIR, ensure_dirs, logger

ensure_dirs()

STATS_FILE = OUTPUTS_DIR / "stats.json"
TOP_TOKENS_FILE = OUTPUTS_DIR / "top_tokens.csv"

def load_clean(path: Path = CLEANED_PATH) -> pd.DataFrame:
    if not path.exists():
        logger.error("Cleaned file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def basic_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["n_rows"] = int(len(df))
    stats["n_columns"] = int(df.shape[1])
    if "is_fake" in df.columns:
        stats["fake_count"] = int(df["is_fake"].fillna(0).sum())
        stats["real_count"] = int((df["is_fake"] == 0).sum())
    if "text_length" in df.columns:
        stats["text_length_mean"] = float(df["text_length"].mean())
        stats["text_length_median"] = float(df["text_length"].median())
    return stats

def top_tokens(df: pd.DataFrame, n=25):
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(df["combined_text"].fillna(""))
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    top_idx = np.argsort(sums)[::-1][:n]
    return [(terms[i], float(sums[i])) for i in top_idx]

def run_analysis():
    df = load_clean()
    stats = basic_stats(df)
    ensure_dirs()
    # save stats
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved stats to %s", STATS_FILE)

    tokens = top_tokens(df, n=50)
    pd.DataFrame(tokens, columns=["token", "tfidf_sum"]).to_csv(TOP_TOKENS_FILE, index=False)
    logger.info("Saved top tokens to %s", TOP_TOKENS_FILE)

    # Print a concise text summary to console
    logger.info("Basic stats: %s", stats)
    logger.info("Top tokens (first 10): %s", tokens[:10])

if __name__ == "__main__":
    run_analysis()

