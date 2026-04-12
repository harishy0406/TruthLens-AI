"""
train_fake_models.py
─────────────────────
Trains Logistic Regression, CNN, and BiLSTM on the fake-news dataset.
Selects the best model and saves it to models/final/best_fake_model.pkl.

Usage:
    python -m src.training.train_fake_models
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.feature_engineering import (
    prepare_fake_dataset, build_tfidf, build_tokenizer,
    texts_to_sequences, split_data, MODELS_FINAL,
)
from src.models.logistic_model import LogisticModel
from src.models.cnn_model       import CNNModel
from src.models.bilstm_model    import BiLSTMModel
from src.evaluation.metrics     import compute_metrics, save_results

FAKE_MODELS_DIR = os.path.join("models", "fake_models")
os.makedirs(FAKE_MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_FINAL,    exist_ok=True)

RESULTS_FILE = os.path.join(FAKE_MODELS_DIR, "results.json")


def train_all():
    # ── 1. Load / prepare data ─────────────────────────────────────────────────
    processed = os.path.join("data", "processed", "fake_news_combined.csv")
    if os.path.exists(processed):
        print(f"[INFO] Loading preprocessed data from {processed} …")
        df = pd.read_csv(processed)
    else:
        df = prepare_fake_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── 2. TF-IDF ──────────────────────────────────────────────────────────────
    vec_path = os.path.join(MODELS_FINAL, "vectorizer.pkl")
    if os.path.exists(vec_path):
        print("[INFO] Loading existing TF-IDF vectorizer …")
        tfidf = joblib.load(vec_path)
    else:
        tfidf = build_tfidf(X_train)

    X_tr_tfidf  = tfidf.transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_te_tfidf  = tfidf.transform(X_test)

    # ── 3. Tokenizer / sequences ───────────────────────────────────────────────
    tok_path = os.path.join(MODELS_FINAL, "tokenizer.pkl")
    if os.path.exists(tok_path):
        print("[INFO] Loading existing tokenizer …")
        tokenizer = joblib.load(tok_path)
    else:
        tokenizer = build_tokenizer(X_train)

    X_tr_seq  = texts_to_sequences(tokenizer, X_train)
    X_val_seq = texts_to_sequences(tokenizer, X_val)
    X_te_seq  = texts_to_sequences(tokenizer, X_test)

    scores = {}

    # ── 4. Logistic Regression ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Model 1 — Logistic Regression")
    print("="*60)
    lr = LogisticModel()
    lr.fit(X_tr_tfidf, y_train)
    lr_results = lr.evaluate(X_te_tfidf, y_test)
    lr.save(os.path.join(FAKE_MODELS_DIR, "logistic.pkl"))
    scores["logistic"] = lr_results["accuracy"]

    # ── 5. CNN ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Model 2 — CNN")
    print("="*60)
    cnn = CNNModel()
    cnn.fit(X_tr_seq, y_train, X_val_seq, y_val)
    cnn_results = cnn.evaluate(X_te_seq, y_test)
    cnn.save(os.path.join(FAKE_MODELS_DIR, "cnn.h5"))
    scores["cnn"] = cnn_results["accuracy"]

    # Save training history
    joblib.dump(cnn_results["history"], os.path.join(FAKE_MODELS_DIR, "cnn_history.pkl"))

    # ── 6. BiLSTM ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Model 3 — BiLSTM")
    print("="*60)
    bilstm = BiLSTMModel()
    bilstm.fit(X_tr_seq, y_train, X_val_seq, y_val)
    bilstm_results = bilstm.evaluate(X_te_seq, y_test)
    bilstm.save(os.path.join(FAKE_MODELS_DIR, "bilstm.h5"))
    scores["bilstm"] = bilstm_results["accuracy"]

    joblib.dump(bilstm_results["history"], os.path.join(FAKE_MODELS_DIR, "bilstm_history.pkl"))

    # ── 7. Select best ─────────────────────────────────────────────────────────
    best_name = max(scores, key=scores.get)
    print(f"\n[INFO] Scores: {scores}")
    print(f"[INFO] Best fake model: {best_name.upper()} ({scores[best_name]:.4f})")

    # Copy best model artefact to models/final/
    import shutil
    src_map = {
        "logistic": os.path.join(FAKE_MODELS_DIR, "logistic.pkl"),
        "cnn":      os.path.join(FAKE_MODELS_DIR, "cnn.h5"),
        "bilstm":   os.path.join(FAKE_MODELS_DIR, "bilstm.h5"),
    }
    ext = ".pkl" if best_name == "logistic" else ".h5"
    best_dst = os.path.join(MODELS_FINAL, f"best_fake_model{ext}")
    shutil.copy2(src_map[best_name], best_dst)
    print(f"[OK] Best model saved → {best_dst}")

    # Save metadata
    meta = {
        "best_model": best_name,
        "best_accuracy": scores[best_name],
        "all_scores": scores,
    }
    with open(os.path.join(MODELS_FINAL, "fake_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(RESULTS_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n[DONE] Fake-news model training complete.\n")
    return meta


if __name__ == "__main__":
    train_all()
