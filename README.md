"""
Notebook template: EDA, model loading and sample inference.
Run as script or convert to notebook.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report

CLEANED = Path("cleaned_data/jobs_clean.csv")
MODEL = Path("models/tfidf_logreg_pipeline.joblib")
METRICS = Path("models/tfidf_logreg_metrics.json")

# 1. Load cleaned data
df = pd.read_csv(CLEANED)
display(df.head())

# 2. Basic EDA
print("Rows:", len(df))
if "is_fake" in df.columns:
    print(df["is_fake"].value_counts())

if "text_length" in df.columns:
    print(df["text_length"].describe())

# 3. Show top tokens
top = pd.read_csv("outputs/top_tokens.csv")
display(top.head(20))

# 4. Load model and run predictions on a sample
pipeline = joblib.load(MODEL)
sample_texts = df["combined_text"].fillna("").astype(str).head(5).tolist()
preds = pipeline.predict(sample_texts)
probs = None
if hasattr(pipeline, "predict_proba"):
    probs = pipeline.predict_proba(sample_texts)
print("Preds:", preds)
if probs is not None:
    print("Probabilities (per class):")
    print(probs)

# 5. If you have a holdout set, compute full report (example: use metrics file)
import json
with open(METRICS) as f:
    print(json.load(f))

