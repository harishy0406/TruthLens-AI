"""
feature_engineering.py
───────────────────────
Builds TF-IDF vectorizer and Keras tokenizer/padder for both datasets.
Saves processed CSVs to data/processed/ and artefacts to models/final/.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from src.preprocessing.clean_text import clean_text

# ── Constants ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 4000
MAX_LEN    = 120
TEST_SIZE  = 0.20
RANDOM_STATE = 42

RAW_PATH      = os.path.join("data", "raw", "news_dataset_raw.csv")
AI_PATH       = os.path.join("data", "ai_news", "ai_news_extended.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_FINAL  = os.path.join("models", "final")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_FINAL,  exist_ok=True)


# ── Fake News Dataset ──────────────────────────────────────────────────────────
def prepare_fake_dataset() -> pd.DataFrame:
    """Load, clean, and save the fake-news combined dataset."""
    print("[INFO] Loading fake news dataset …")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # Keep only rows that belong to the fake/true news split
    df = df[df["is_misinformation"].isna()].copy()
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    print(f"[INFO] Fake dataset raw size : {len(df):,}")
    print(f"[INFO] Label distribution    :\n{df['label'].value_counts()}\n")

    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df["text"].progress_apply(clean_text)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, "fake_news_combined.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved  → {out_path}  ({len(df):,} rows)\n")
    return df


# ── AI News Dataset ────────────────────────────────────────────────────────────
def prepare_ai_dataset() -> pd.DataFrame:
    """Load, clean, and save the AI-generated misinformation dataset."""
    print("[INFO] Loading AI news dataset …")
    df = pd.read_csv(AI_PATH)

    # Numeric feature columns present in the AI dataset
    numeric_cols = [
        "text_length", "token_count", "readability_score",
        "sentiment_score", "toxicity_score", "engagement",
        "num_urls", "num_mentions", "num_hashtags",
        "detected_synthetic_score", "embedding_sim_to_facts",
        "source_domain_reliability", "author_followers", "author_verified",
    ]
    available = [c for c in numeric_cols if c in df.columns]

    df = df[["text"] + available + ["is_misinformation"]].dropna(subset=["text", "is_misinformation"])
    df["is_misinformation"] = df["is_misinformation"].astype(int)
    df = df.rename(columns={"is_misinformation": "label"})

    print(f"[INFO] AI dataset raw size   : {len(df):,}")
    print(f"[INFO] Label distribution    :\n{df['label'].value_counts()}\n")

    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df["text"].progress_apply(clean_text)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, "cleaned_ai.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved  → {out_path}  ({len(df):,} rows)\n")
    return df


# ── TF-IDF Vectorizer ──────────────────────────────────────────────────────────
def build_tfidf(texts: pd.Series, max_features: int = 50_000) -> TfidfVectorizer:
    """Fit TF-IDF on training texts and save."""
    print("[INFO] Fitting TF-IDF vectorizer …")
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    vec.fit(texts)
    vec_path = os.path.join(MODELS_FINAL, "vectorizer.pkl")
    joblib.dump(vec, vec_path)
    print(f"[OK] TF-IDF saved → {vec_path}\n")
    return vec


# ── Keras Tokenizer ────────────────────────────────────────────────────────────
def build_tokenizer(texts: pd.Series) -> Tokenizer:
    """Fit Keras tokenizer and save."""
    print("[INFO] Fitting Keras tokenizer …")
    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    tok_path = os.path.join(MODELS_FINAL, "tokenizer.pkl")
    joblib.dump(tok, tok_path)
    print(f"[OK] Tokenizer saved → {tok_path}\n")
    return tok


def texts_to_sequences(tokenizer: Tokenizer, texts: pd.Series) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


# ── Train / Val / Test splits ──────────────────────────────────────────────────
def split_data(df: pd.DataFrame, text_col: str = "clean_text"):
    X = df[text_col]
    y = df["label"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE * 2, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_df = prepare_fake_dataset()
    ai_df   = prepare_ai_dataset()

    # Build shared artefacts on the larger (fake) dataset
    build_tfidf(fake_df["clean_text"])
    build_tokenizer(fake_df["clean_text"])

    print("[DONE] Feature engineering complete.")
