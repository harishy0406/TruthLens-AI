"""
logistic_model.py
─────────────────
TF-IDF + Logistic Regression wrapper for binary text classification.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class LogisticModel:
    """Logistic Regression on TF-IDF features."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        self.vectorizer = None

    def fit(self, X_train_tfidf, y_train):
        print("[LR] Training Logistic Regression …")
        self.model.fit(X_train_tfidf, y_train)
        return self

    def predict(self, X_tfidf) -> np.ndarray:
        return self.model.predict(X_tfidf)

    def predict_proba(self, X_tfidf) -> np.ndarray:
        """Returns probability of the positive class (Fake)."""
        return self.model.predict_proba(X_tfidf)[:, 1]

    def evaluate(self, X_tfidf, y_true) -> dict:
        y_pred = self.predict(X_tfidf)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"[LR] Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred))
        return {"accuracy": acc, "report": report, "y_pred": y_pred}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[LR] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticModel":
        obj = cls()
        obj.model = joblib.load(path)
        return obj
