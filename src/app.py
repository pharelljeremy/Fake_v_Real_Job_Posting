import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` imports work in different run contexts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import joblib
import pandas as pd
import json
from src.config import MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, CLEANED_PATH, ensure_dirs, logger

ensure_dirs()

# Helpers
def load_batch_pipeline():
    p = MODELS_DIR / "tfidf_logreg_pipeline.joblib"
    if not p.exists():
        st.warning(f"Batch pipeline not found at {p}. Run training first.")
        return None
    return joblib.load(p)

def load_incremental():
    vec_p = MODELS_DIR / "hashing_vectorizer.joblib"
    clf_p = MODELS_DIR / "sgd_clf.joblib"
    if not vec_p.exists() or not clf_p.exists():
        st.warning("Incremental model files not found. Run incremental training first.")
        return None, None
    vec = joblib.load(vec_p)
    clf = joblib.load(clf_p)
    return vec, clf

def load_metrics():
    p = MODELS_DIR / "tfidf_logreg_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def predict_with_pipeline(pipe, text: str):
    label = None
    proba_1 = None
    try:
        label = int(pipe.predict([text])[0])
        if hasattr(pipe, "predict_proba"):
            proba_1 = float(pipe.predict_proba([text])[0][1])
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return label, proba_1

def predict_incremental(vec, clf, text: str):
    X = vec.transform([text])
    label = int(clf.predict(X)[0])
    proba_1 = None
    if hasattr(clf, "predict_proba"):
        proba_1 = float(clf.predict_proba(X)[0][1])
    return label, proba_1

# UI
st.set_page_config(page_title="Fake vs Real Job Posting — Demo", layout="wide")
st.title("Fake vs Real Job Posting — Demo")

st.sidebar.header("Model & Data")
model_mode = st.sidebar.selectbox("Model to use", ["batch_pipeline", "incremental"], index=0)
show_metrics = st.sidebar.checkbox("Show saved metrics", value=True)
show_figures = st.sidebar.checkbox("Show output figures", value=True)

if show_metrics:
    metrics = load_metrics()
    if metrics:
        st.sidebar.markdown("### Batch model metrics")
        st.sidebar.code(json.dumps(metrics, indent=2))
    else:
        st.sidebar.info("No batch metrics found (models/tfidf_logreg_metrics.json)")

st.header("Predict single job posting")
default_text = ""
if CLEANED_PATH.exists():
    try:
        df_sample = pd.read_csv(CLEANED_PATH, nrows=5)
        if "combined_text" in df_sample.columns:
            default_text = df_sample["combined_text"].iloc[0]
    except Exception:
        default_text = ""
text_in = st.text_area("Paste the job posting text here", value=default_text, height=250)
col1, col2 = st.columns([1,1])

if col1.button("Predict"):
    if not text_in.strip():
        st.warning("Please enter job post text first")
    else:
        label_map = {0: "Real", 1: "Fake"}
        if model_mode == "batch_pipeline":
            pipe = load_batch_pipeline()
            if pipe is not None:
                label, proba = predict_with_pipeline(pipe, text_in)
                st.success(f"Predicted label: {label} ({label_map.get(label, label)})")
                if proba is not None:
                    st.info(f"P(is_fake = 1): {proba:.4f}")
        else:
            vec, clf = load_incremental()
            if vec is not None:
                label, proba = predict_incremental(vec, clf, text_in)
                st.success(f"Predicted label: {label} ({label_map.get(label, label)})")
                if proba is not None:
                    st.info(f"P(is_fake = 1): {proba:.4f}")

st.header("Batch predict from CSV")
st.markdown("Upload a CSV file with a column named `combined_text` (or a text column). The app will attempt to use `combined_text` or the first text-like column.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    if "combined_text" in df_up.columns:
        text_col = "combined_text"
    else:
        text_cols = [c for c in df_up.columns if df_up[c].dtype == object]
        if not text_cols:
            st.error("No text columns found in uploaded CSV")
            st.stop()
        text_col = text_cols[0]
        st.info(f"Using column `{text_col}` for predictions")

    if st.button("Run batch predictions on uploaded CSV"):
        if model_mode == "batch_pipeline":
            pipe = load_batch_pipeline()
            if pipe is None:
                st.stop()
            preds = pipe.predict(df_up[text_col].fillna("").astype(str).tolist())
            df_up["predicted_label"] = preds
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(df_up[text_col].fillna("").astype(str).tolist())
                df_up["pred_prob_is_fake"] = [float(p[1]) for p in probs]
            out_path = OUTPUTS_DIR / "predictions.csv"
            df_up.to_csv(out_path, index=False)
            st.success(f"Saved predictions to {out_path}")
            st.dataframe(df_up.head())
        else:
            vec, clf = load_incremental()
            if vec is None:
                st.stop()
            X = vec.transform(df_up[text_col].fillna("").astype(str).tolist())
            preds = clf.predict(X)
            df_up["predicted_label"] = preds
            out_path = OUTPUTS_DIR / "predictions.csv"
            df_up.to_csv(out_path, index=False)
            st.success(f"Saved predictions to {out_path}")
            st.dataframe(df_up.head())

if show_figures:
    st.header("Saved figures")
    fig_files = list(FIGURES_DIR.glob("*"))
    if not fig_files:
        st.info("No figures found in outputs/figures/. Run visualise script first.")
    else:
        for f in fig_files:
            st.image(str(f), caption=f.name, use_container_width=True)

st.header("Top tokens / features")
top_tokens_path = OUTPUTS_DIR / "top_tokens.csv"
if top_tokens_path.exists():
    df_tokens = pd.read_csv(top_tokens_path)
    st.table(df_tokens.head(30))
else:
    st.info("No top tokens file found; run analysis first to generate top_tokens.csv")

st.markdown("---")
st.caption("This Streamlit demo loads models saved in models/ and outputs from outputs/. Run `python run_pipeline.py` to regenerate artifacts.")

