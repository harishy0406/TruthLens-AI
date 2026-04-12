"""
predict.py
──────────
Unified prediction pipeline for TruthLens.
Loads all saved models and artefacts; exposes a single predict() function.

Usage:
    from src.utils.predict import TruthLensPredictor
    predictor = TruthLensPredictor()
    result = predictor.predict("Article text goes here...")
    print(result)
"""

import os, sys, json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.clean_text import clean_text
from src.fusion.model_fusion import FusedModel

MODELS_FINAL  = os.path.join("models", "final")
FAKE_MODELS   = os.path.join("models", "fake_models")
AI_MODELS     = os.path.join("models", "ai_models")

VOCAB_SIZE = 4000
MAX_LEN    = 120


def _risk_level(confidence: float) -> str:
    if confidence >= 0.85:  return "🔴 High Risk"
    if confidence >= 0.60:  return "🟠 Medium Risk"
    if confidence >= 0.40:  return "🟡 Low Risk"
    return "🟢 Likely Real"


class TruthLensPredictor:
    """
    Loads all model artefacts once and exposes a predict() method.
    Falls back gracefully if individual models are missing.
    """

    def __init__(self):
        self._vectorizer  = None
        self._tokenizer   = None
        self._fake_model  = None  # best fake model (LR or keras)
        self._ai_model    = None  # best AI model
        self._fused       = None
        self._fake_type   = None  # "logistic" | "keras"
        self._ai_type     = None
        self._loaded      = False

    # ── Lazy loading ───────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return
        print("[Predictor] Loading artefacts …")

        # TF-IDF
        vec_path = os.path.join(MODELS_FINAL, "vectorizer.pkl")
        if os.path.exists(vec_path):
            self._vectorizer = joblib.load(vec_path)

        # Tokenizer
        tok_path = os.path.join(MODELS_FINAL, "tokenizer.pkl")
        if os.path.exists(tok_path):
            self._tokenizer = joblib.load(tok_path)

        # Best Fake model
        self._fake_model, self._fake_type = self._load_best("fake")

        # Best AI model
        self._ai_model, self._ai_type = self._load_best("ai")

        # Fused model
        self._fused = FusedModel.load()

        self._loaded = True
        print("[Predictor] Ready.")

    def _load_best(self, which: str):
        """Load best fake or AI model. Returns (model, type_str)."""
        meta_path = os.path.join(MODELS_FINAL, f"{which}_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        best_name = meta.get("best_model", "logistic")
        # For AI models, prefer random_forest or logistic (pkl) since
        # deep learning doesn't converge well on small/synthetic AI datasets
        if which == "ai":
            # Try random_forest pkl first (numeric features), fallback to logistic
            rf_path = os.path.join(f"models/{which}_models", "random_forest.pkl")
            if os.path.exists(rf_path):
                m = joblib.load(rf_path)
                return m, "random_forest"
            best_name = "logistic"  # fallback for AI

        ext = ".pkl" if best_name in ("logistic", "random_forest") else ".h5"

        candidates = [
            os.path.join(MODELS_FINAL, f"best_{which}_model{ext}"),
            os.path.join(f"models/{which}_models", f"{best_name}{ext}"),
        ]
        for path in candidates:
            if os.path.exists(path):
                if ext == ".pkl":
                    m = joblib.load(path)
                    return m, "logistic"
                else:
                    from tensorflow.keras.models import load_model
                    from src.fusion.attention_layer import AttentionLayer
                    m = load_model(path, custom_objects={"AttentionLayer": AttentionLayer})
                    return m, "keras"

        print(f"[Predictor] WARNING: No {which} model found — using fallback.")
        return None, None

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _tfidf_transform(self, text: str):
        if self._vectorizer is None:
            return None
        return self._vectorizer.transform([text])

    def _tokenize(self, text: str):
        if self._tokenizer is None:
            return None
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = self._tokenizer.texts_to_sequences([text])
        return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    def _predict_one(self, model, model_type: str, clean: str, raw_text: str = "") -> float:
        """Return probability of Fake (float 0–1)."""
        if model is None:
            return 0.5  # Neutral fallback

        if model_type == "logistic":
            X = self._tfidf_transform(clean)
            if X is None:
                return 0.5
            return float(model.predict_proba(X)[0, 1])

        elif model_type == "keras":
            X = self._tokenize(clean)
            if X is None:
                return 0.5
            raw_prob = float(model.predict(X, verbose=0).flatten()[0])
            # NOTE: In the fake-news training dataset, label 0 = FAKE, label 1 = REAL
            # So raw_prob = P(REAL). We return P(FAKE) = 1 - raw_prob.
            return 1.0 - raw_prob

        elif model_type == "random_forest":
            # Build basic numeric features from available text properties
            rf_data  = model  # dict with model, scaler, features
            rf_model = rf_data["model"]
            scaler   = rf_data["scaler"]
            features = rf_data["features"]
            # Construct a feature vector with available signals
            text     = raw_text or clean
            feat_vec = {
                "text_length":               len(text),
                "token_count":               len(text.split()),
                "readability_score":         min(len(text.split()) / max(len(text), 1) * 10, 10),
                "sentiment_score":           0.0,
                "toxicity_score":            0.5,
                "engagement":                0,
                "num_urls":                  text.count("http"),
                "num_mentions":              text.count("@"),
                "num_hashtags":              text.count("#"),
                "detected_synthetic_score":  0.5,
                "embedding_sim_to_facts":    0.5,
                "source_domain_reliability": 0.5,
                "author_followers":          0,
                "author_verified":           0,
            }
            import numpy as np
            X_arr = np.array([[feat_vec.get(f, 0.0) for f in features]])
            X_arr = scaler.transform(X_arr)
            return float(rf_model.predict_proba(X_arr)[0, 1])

        return 0.5

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Predict whether `text` is fake/misinformation.

        Returns
        -------
        dict:
            label          : 0 (Real) | 1 (Fake)
            verdict        : "REAL" | "FAKE"
            confidence     : float  (0–1, how sure the model is)
            fake_score     : probability from fake-news detector
            ai_score       : probability from AI-news detector
            fused_prob     : weighted ensemble probability
            risk_level     : descriptive risk string
            clean_text     : pre-processed text snippet
        """
        self._load()

        if not text or not text.strip():
            return {"error": "Empty input text."}

        cleaned = clean_text(text)

        fake_prob = self._predict_one(self._fake_model, self._fake_type, cleaned, text)
        ai_prob   = self._predict_one(self._ai_model,   self._ai_type,   cleaned, text)

        fusion = self._fused.fuse(fake_prob, ai_prob)
        fused_prob = fusion["fused_prob"]
        label      = fusion["label"]
        confidence = fusion["confidence"]

        return {
            "label":      label,
            "verdict":    "FAKE" if label == 1 else "REAL",
            "confidence": confidence,
            "fake_score": fake_prob,
            "ai_score":   ai_prob,
            "fused_prob": fused_prob,
            "risk_level": _risk_level(fused_prob if label == 1 else 1 - fused_prob),
            "clean_text": cleaned[:300],
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_predictor: TruthLensPredictor = None


def get_predictor() -> TruthLensPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TruthLensPredictor()
    return _predictor


def predict(text: str) -> dict:
    """Convenience function — uses the module-level predictor singleton."""
    return get_predictor().predict(text)


if __name__ == "__main__":
    sample = (
        "Scientists have confirmed that drinking bleach cures COVID-19 "
        "according to a new study published yesterday by the White House."
    )
    result = predict(sample)
    for k, v in result.items():
        print(f"  {k:15s}: {v}")
