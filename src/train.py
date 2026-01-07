import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from src.config import CLEANED_PATH, MODELS_DIR, ensure_dirs, logger

ensure_dirs()

def train_batch(random_state=42):
    """
    Batch training using TF-IDF + LogisticRegression — saves a single pipeline joblib.
    """
    if not CLEANED_PATH.exists():
        logger.error("Cleaned data not found at %s", CLEANED_PATH)
        raise FileNotFoundError(CLEANED_PATH)
    df = pd.read_csv(CLEANED_PATH)
    df = df[df["is_fake"].notna()]
    X = df["combined_text"].fillna("")
    y = df["is_fake"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    vec = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1,2))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    pipeline = make_pipeline(vec, clf)
    logger.info("Fitting batch model (TF-IDF + LogisticRegression)...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    logger.info("Classification report: %s", report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "tfidf_logreg_pipeline.joblib"
    joblib.dump(pipeline, out_path)
    # save metrics
    metrics_path = MODELS_DIR / "tfidf_logreg_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved batch pipeline to %s and metrics to %s", out_path, metrics_path)

def train_incremental(chunksize=2000, random_state=42):
    """
    Incremental training using HashingVectorizer + SGDClassifier via partial_fit.
    """
    if not CLEANED_PATH.exists():
        logger.error("Cleaned data not found at %s", CLEANED_PATH)
        raise FileNotFoundError(CLEANED_PATH)
    # read labels to get class set
    df_all = pd.read_csv(CLEANED_PATH, usecols=["is_fake"])
    classes = np.unique(df_all["is_fake"].dropna().astype(int).values)
    classes = np.sort(classes)
    logger.info("Detected classes: %s", classes)

    vec = HashingVectorizer(n_features=2**20, alternate_sign=False, stop_words="english", ngram_range=(1,2))
    clf = SGDClassifier(loss="log", class_weight="balanced", random_state=random_state)

    first_batch = True
    for i, chunk in enumerate(pd.read_csv(CLEANED_PATH, dtype=str, chunksize=chunksize)):
        logger.info("Training on chunk %d rows=%d", i, len(chunk))
        chunk = chunk[chunk["is_fake"].notna()]
        X_chunk = chunk["combined_text"].fillna("").values
        y_chunk = chunk["is_fake"].astype(int).values

        X_trans = vec.transform(X_chunk)
        if first_batch:
            clf.partial_fit(X_trans, y_chunk, classes=classes)
            first_batch = False
        else:
            clf.partial_fit(X_trans, y_chunk)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, MODELS_DIR / "hashing_vectorizer.joblib")
    joblib.dump(clf, MODELS_DIR / "sgd_clf.joblib")
    logger.info("Saved incremental model to %s", MODELS_DIR)

if __name__ == "__main__":
    # default to batch training; change as needed
    train_batch()

