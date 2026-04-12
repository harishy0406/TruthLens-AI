"""
clean_text.py
─────────────
NLP preprocessing utilities for TruthLens.
Handles lowercasing, URL removal, stopword removal, lemmatization.
"""

import re
import string
import nltk

# Download NLTK resources (silent if already present)
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Regex patterns compiled once for performance
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Full NLP cleaning pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation & numbers
    5. Remove stopwords
    6. Lemmatize tokens
    Returns cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def extract_features(text: str) -> dict:
    """
    Extract lightweight numeric features from raw text.
    Returns dict with text_length, token_count.
    """
    if not isinstance(text, str):
        return {"text_length": 0, "token_count": 0}
    tokens = text.split()
    return {
        "text_length": len(text),
        "token_count": len(tokens),
    }


if __name__ == "__main__":
    sample = "BREAKING: U.S. Scientists <b>discover</b> that eating 10 apples/day cures cancer! Visit http://fake.com for more."
    print("Raw:", sample)
    print("Clean:", clean_text(sample))
    print("Features:", extract_features(sample))
