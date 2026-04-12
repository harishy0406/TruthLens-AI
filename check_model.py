"""Quick check of the fake model output on known fake/real samples."""
import sys, os
sys.path.insert(0, '.')
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing.clean_text import clean_text

# Load artefacts
vec  = joblib.load('models/final/vectorizer.pkl')
tok  = joblib.load('models/final/tokenizer.pkl')
fake_bilstm = load_model('models/final/best_fake_model.h5')
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 120

samples = [
    (1, "Breaking: Scientists confirm drinking bleach cures covid. Mainstream media is suppressing this secret government study. Share before Big Pharma deletes it!"),
    (0, "The Federal Reserve held interest rates steady. Powell said the committee will act based on incoming economic data and inflation trends."),
    (1, "Vaccines contain microchips to track citizens. Bill Gates is funding a secret program to control the global population using 5G towers."),
    (0, "NASA discovered three new exoplanets using the James Webb Space Telescope, located in the habitable zone approximately 40 light-years away."),
]

print(f"{'True':4s}  {'Clean text snippet':50s}  {'BiLSTM_prob':11s}")
print("-"*70)
for label, text in samples:
    cleaned = clean_text(text)
    seq = tok.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = fake_bilstm.predict(padded, verbose=0).flatten()[0]
    pred = 1 if prob >= 0.5 else 0
    m = '✅' if pred == label else '❌'
    print(f"{m} [{label}]  {cleaned[:50]:50s}  prob={prob:.4f}")
