"""
model_fusion.py
───────────────
Fuses the best fake-news model and best AI model into a unified
prediction using weighted averaging + optional stacking.

Saves fused_model.pkl (stacking meta-learner) to models/final/.

Usage:
    python -m src.fusion.model_fusion
"""

import os, sys, json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.fusion.attention_layer import attention_fusion

MODELS_FINAL = os.path.join("models", "final")


# ── Weight tables ──────────────────────────────────────────────────────────────
# Loaded from the meta JSON files produced during training.
def _load_weights() -> tuple:
    """Return (fake_weight, ai_weight) normalised to sum to 1."""
    fake_acc, ai_acc = 0.5, 0.5

    fake_meta_path = os.path.join(MODELS_FINAL, "fake_meta.json")
    ai_meta_path   = os.path.join(MODELS_FINAL, "ai_meta.json")

    if os.path.exists(fake_meta_path):
        with open(fake_meta_path) as f:
            fake_acc = json.load(f).get("best_accuracy", 0.5)
    if os.path.exists(ai_meta_path):
        with open(ai_meta_path) as f:
            ai_acc = json.load(f).get("best_accuracy", 0.5)

    total = fake_acc + ai_acc
    return fake_acc / total, ai_acc / total


class FusedModel:
    """
    Ensemble of best_fake_model + best_ai_model.

    Strategy: weighted average fusion  (weights ∝ individual accuracy)
    with stacking meta-learner as a secondary option.
    """

    def __init__(self):
        self.fake_weight, self.ai_weight = _load_weights()
        self.meta_learner = None  # Optional stacking model

    # ── Weighted Average Fusion ────────────────────────────────────────────────
    def fuse(self, fake_prob: float, ai_prob: float) -> dict:
        """
        Weighted average of two model probabilities.

        Returns
        -------
        dict: { fused_prob, label, confidence }
        """
        fused = self.fake_weight * fake_prob + self.ai_weight * ai_prob
        label = int(fused >= 0.5)
        confidence = fused if label == 1 else 1 - fused
        return {
            "fused_prob":  round(fused, 4),
            "label":       label,
            "confidence":  round(float(confidence), 4),
            "fake_prob":   round(fake_prob, 4),
            "ai_prob":     round(ai_prob, 4),
            "fake_weight": round(self.fake_weight, 4),
            "ai_weight":   round(self.ai_weight, 4),
        }

    # ── Attention Fusion ───────────────────────────────────────────────────────
    def fuse_attention(self, fake_prob: float, ai_prob: float,
                       fake_temp: float = None, ai_temp: float = None) -> dict:
        """Soft-attention weighted fusion."""
        fake_temp = fake_temp or (self.fake_weight * 2)
        ai_temp   = ai_temp   or (self.ai_weight   * 2)
        fused = attention_fusion(
            np.array([fake_prob, ai_prob]),
            np.array([fake_temp, ai_temp]),
        )
        label = int(fused >= 0.5)
        confidence = fused if label == 1 else 1 - fused
        return {
            "fused_prob": round(fused, 4),
            "label":      label,
            "confidence": round(float(confidence), 4),
        }

    # ── Stacking Meta-Learner ──────────────────────────────────────────────────
    def fit_stacking(self, fake_probs: np.ndarray, ai_probs: np.ndarray,
                     y_true: np.ndarray):
        """Train a meta-learner on concatenated model outputs."""
        X_meta = np.column_stack([fake_probs, ai_probs])
        self.meta_learner = LogisticRegression(max_iter=500, C=1.0)
        cv_scores = cross_val_score(self.meta_learner, X_meta, y_true, cv=5)
        self.meta_learner.fit(X_meta, y_true)
        print(f"[Fusion] Stacking CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return float(cv_scores.mean())

    def stacking_predict_proba(self, fake_prob: float, ai_prob: float) -> float:
        if self.meta_learner is None:
            return self.fuse(fake_prob, ai_prob)["fused_prob"]
        X = np.array([[fake_prob, ai_prob]])
        return float(self.meta_learner.predict_proba(X)[0, 1])

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str = None):
        path = path or os.path.join(MODELS_FINAL, "fused_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[Fusion] Fused model saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "FusedModel":
        path = path or os.path.join(MODELS_FINAL, "fused_model.pkl")
        if os.path.exists(path):
            return joblib.load(path)
        return cls()


if __name__ == "__main__":
    fused = FusedModel()
    print(f"[Fusion] Fake weight: {fused.fake_weight:.4f}")
    print(f"[Fusion] AI   weight: {fused.ai_weight:.4f}")
    # Demo fusion
    result = fused.fuse(fake_prob=0.82, ai_prob=0.74)
    print(f"[Fusion] Demo result: {result}")
    fused.save()
    print("[Fusion] Done.")
