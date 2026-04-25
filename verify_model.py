"""Verify model on in-distribution data.""" 
sys.path.insert(0, '.')
import pandas as pd, numpy as np, joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok    = joblib.load('models/final/tokenizer.pkl')
bilstm = load_model('models/final/best_fake_model.h5')
df     = pd.read_csv('data/processed/fake_news_combined.csv').sample(8, random_state=42)

print("Model check on actual processed training data (in-distribution):")
print(f"{'True':4}  {'Pred':4}  {'Prob':6}  Text snippet")
print("-"*80)
correct = 0
for _, row in df.iterrows():
    seq = tok.texts_to_sequences([row['clean_text']])
    padded = pad_sequences(seq, maxlen=120, padding='post', truncating='post')
    prob = float(bilstm.predict(padded, verbose=0).flatten()[0])
    pred = 1 if prob >= 0.5 else 0
    ok = pred == int(row['label'])
    correct += int(ok)
    mark = 'OK' if ok else 'WRONG'
    print(f"  [{int(row['label'])}] -> [{pred}]  {prob:.4f}  {mark}  {row['clean_text'][:60]}")

print(f"\nAccuracy: {correct}/{len(df)} = {correct/len(df)*100:.0f}%")
print("\nNote: label 0 = FAKE, label 1 = REAL in this dataset")
