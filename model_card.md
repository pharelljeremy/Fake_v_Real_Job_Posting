Model Card — Fake vs Real Job Posting
Model: TF-IDF + LogisticRegression (batch)
Date trained: see models/model_metadata.json
Dataset: Proprietary CSV uploaded to raw_data/raw_jobs.csv (cleaned and saved to cleaned_data/jobs_clean.csv)
Task: Binary classification — predict if a job posting is fake (1) or real (0)
Performance (example)
	▪	Accuracy (holdout): see models/tfidf_logreg_metrics.json 	▪	Notes: metrics reported are on an 80/20 train/test split. Class imbalance handled with class_weight='balanced' for the batch model.
Intended use
	▪	Educational / portfolio demonstration of an end-to-end NLP pipeline. 	▪	Not intended for production use without further validation and bias/audit testing.
Limitations
	▪	Model trained on a single dataset — may not generalise to other job posting sources. 	▪	Simple TF-IDF features — consider fine-tuning larger transformer models for improved robustness. 	▪	PII removal is heuristic (emails/phones) — verify removal before public release.
