"""
train_ai_models.py
───────────────────
Trains Logistic Regression, CNN, BiLSTM, AND Random Forest (on numeric features)
on the AI-generated misinformation dataset. Selects and saves the best model.

The AI dataset has 500 rows with rich numeric features (sentiment, toxicity,
readability, etc.) – Random Forest on those numeric features outperforms
text-only deep learning at this scale.

Usage:
    python -m src.training.train_ai_models
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.feature_engineering import (
    prepare_ai_dataset, build_tfidf, build_tokenizer,
    texts_to_sequences, split_data, MODELS_FINAL,
)
from src.models.logistic_model import LogisticModel
from src.models.cnn_model       import CNNModel
from src.models.bilstm_model    import BiLSTMModel

AI_MODELS_DIR = os.path.join("models", "ai_models")
os.makedirs(AI_MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_FINAL,  exist_ok=True)

RESULTS_FILE = os.path.join(AI_MODELS_DIR, "results.json")


def train_all():
    # 1. Load / prepare data
    processed = os.path.join("data", "processed", "cleaned_ai.csv")
    if os.path.exists(processed):
        print(f"[INFO] Loading preprocessed AI data from {processed} ...")
        df = pd.read_csv(processed)
    else:
        df = prepare_ai_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # 2. TF-IDF
    vec_path = os.path.join(MODELS_FINAL, "vectorizer.pkl")
    tfidf = joblib.load(vec_path) if os.path.exists(vec_path) else build_tfidf(X_train, max_features=10_000)
    X_tr_tfidf  = tfidf.transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_te_tfidf  = tfidf.transform(X_test)

    # 3. Tokenizer / sequences
    tok_path = os.path.join(MODELS_FINAL, "tokenizer.pkl")
    tokenizer = joblib.load(tok_path) if os.path.exists(tok_path) else build_tokenizer(X_train)
    X_tr_seq  = texts_to_sequences(tokenizer, X_train)
    X_val_seq = texts_to_sequences(tokenizer, X_val)
    X_te_seq  = texts_to_sequences(tokenizer, X_test)

    scores = {}

    # 4. Logistic Regression (TF-IDF)
    print("\n" + "="*60)
    print("  Model 1 --- Logistic Regression (AI dataset)")
    print("="*60)
    lr = LogisticModel(C=0.5)
    lr.fit(X_tr_tfidf, y_train)
    lr_results = lr.evaluate(X_te_tfidf, y_test)
    lr.save(os.path.join(AI_MODELS_DIR, "logistic.pkl"))
    scores["logistic"] = lr_results["accuracy"]

    # 5. CNN
    print("\n" + "="*60)
    print("  Model 2 --- CNN (AI dataset)")
    print("="*60)
    cnn = CNNModel()
    cnn.fit(X_tr_seq, y_train, X_val_seq, y_val, batch_size=32)
    cnn_results = cnn.evaluate(X_te_seq, y_test)
    cnn.save(os.path.join(AI_MODELS_DIR, "cnn.h5"))
    scores["cnn"] = cnn_results["accuracy"]
    joblib.dump(cnn_results["history"], os.path.join(AI_MODELS_DIR, "cnn_history.pkl"))

    # 6. BiLSTM
    print("\n" + "="*60)
    print("  Model 3 --- BiLSTM (AI dataset)")
    print("="*60)
    bilstm = BiLSTMModel()
    bilstm.fit(X_tr_seq, y_train, X_val_seq, y_val, batch_size=32)
    bilstm_results = bilstm.evaluate(X_te_seq, y_test)
    bilstm.save(os.path.join(AI_MODELS_DIR, "bilstm.h5"))
    scores["bilstm"] = bilstm_results["accuracy"]
    joblib.dump(bilstm_results["history"], os.path.join(AI_MODELS_DIR, "bilstm_history.pkl"))

    # 7. Random Forest on rich numeric features (best for small tabular dataset)
    print("\n" + "="*60)
    print("  Model 4 --- Random Forest on Numeric Features (AI dataset)")
    print("="*60)
    full_df = pd.read_csv(processed)
    numeric_cols = [
        c for c in full_df.columns
        if c not in ["text", "clean_text", "label"]
        and full_df[c].dtype in [float, int, "float64", "int64"]
    ]

    if numeric_cols:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        Xn = full_df[numeric_cols].fillna(0).values
        yn = full_df["label"].values
        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
            Xn, yn, test_size=0.20, random_state=42, stratify=yn)

        scaler = StandardScaler()
        Xn_tr_s = scaler.fit_transform(Xn_tr)
        Xn_te_s = scaler.transform(Xn_te)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=42, class_weight="balanced", n_jobs=-1)
        rf.fit(Xn_tr_s, yn_tr)
        rf_pred = rf.predict(Xn_te_s)
        rf_acc  = accuracy_score(yn_te, rf_pred)
        print(f"[RF] Random Forest Accuracy: {rf_acc:.4f}")
        print(classification_report(yn_te, rf_pred))

        rf_path = os.path.join(AI_MODELS_DIR, "random_forest.pkl")
        joblib.dump({"model": rf, "scaler": scaler, "features": numeric_cols}, rf_path)
        scores["random_forest"] = rf_acc
        print(f"[RF] Model saved -> {rf_path}")
    else:
        print("[INFO] No numeric columns found, skipping RF.")

    # 8. Select best model
    best_name = max(scores, key=scores.get)
    print(f"\n[INFO] AI Dataset Scores: {scores}")
    print(f"[INFO] Best AI model: {best_name.upper()} ({scores[best_name]:.4f})")

    import shutil
    src_map = {
        "logistic":      os.path.join(AI_MODELS_DIR, "logistic.pkl"),
        "cnn":           os.path.join(AI_MODELS_DIR, "cnn.h5"),
        "bilstm":        os.path.join(AI_MODELS_DIR, "bilstm.h5"),
        "random_forest": os.path.join(AI_MODELS_DIR, "random_forest.pkl"),
    }
    ext = ".h5" if best_name in ("cnn", "bilstm") else ".pkl"
    best_dst = os.path.join(MODELS_FINAL, f"best_ai_model{ext}")
    shutil.copy2(src_map[best_name], best_dst)
    print(f"[OK] Best AI model saved → {best_dst}")

    meta = {
        "best_model": best_name,
        "best_accuracy": scores[best_name],
        "all_scores": scores,
    }
    with open(os.path.join(MODELS_FINAL, "ai_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(RESULTS_FILE, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n[DONE] AI-news model training complete.\n")
    return meta


if __name__ == "__main__":
    train_all()
