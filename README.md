# 🔍 TruthLens — AI Misinformation Detection System

> **Detecting fake news before it spreads — whether human-written or AI-generated.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Overview

TruthLens is a production-ready AI system that classifies news articles as **Real (0)** or **Fake/Misinformation (1)**. It operates on two detection tracks simultaneously:

| Track | Dataset | Purpose |
|---|---|---|
| 🧠 Human Fake News | Fake.csv + True.csv (44,898 articles) | Detects human-written misinformation |
| 🤖 AI News | ai_news_dataset.csv (500 samples) | Detects AI-generated misinformation |

Both tracks are fused using a **weighted attention mechanism** into a single explainable prediction.

---

## 🏗️ Architecture

```
TruthLens/
├── data/
│   ├── raw/                   ← news_dataset_raw.csv (combined 45K)
│   ├── ai_news/               ← ai_news_dataset.csv (500 rows)
│   └── processed/             ← cleaned CSVs (generated)
│
├── models/
│   ├── fake_models/           ← LR + CNN + BiLSTM for fake detection
│   ├── ai_models/             ← LR + CNN + BiLSTM for AI detection
│   └── final/                 ← best models + vectorizer + fused model
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_text.py      ← NLP pipeline (lemma, stopwords, URL removal)
│   │   └── feature_engineering.py  ← TF-IDF + Keras tokenizer
│   ├── models/
│   │   ├── logistic_model.py  ← LR wrapper
│   │   ├── cnn_model.py       ← 1D-CNN with BatchNorm
│   │   └── bilstm_model.py    ← Bidirectional LSTM
│   ├── training/
│   │   ├── train_fake_models.py
│   │   └── train_ai_models.py
│   ├── evaluation/
│   │   ├── metrics.py         ← Accuracy, F1, AUC, confusion matrix
│   │   └── visualize.py       ← Matplotlib/Seaborn plots
│   ├── fusion/
│   │   ├── attention_layer.py ← Keras attention + NumPy soft-attention
│   │   └── model_fusion.py    ← Weighted average + stacking fusion
│   └── utils/
│       ├── predict.py         ← Unified prediction API
│       └── scraper.py         ← RSS news scraper (10 sources)
│
├── app/
│   ├── streamlit_app.py       ← Main dashboard (4 pages)
│   ├── components/            ← Navbar, cards, charts, footer
│   └── assets/styles.css      ← Premium design system
│
├── notebooks/                 ← Jupyter EDA & training notebooks
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd TruthLens-AI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Download NLTK resources

```python
python -c "import nltk; [nltk.download(r) for r in ['stopwords','wordnet','punkt','omw-1.4']]"
```

### 3. Train models

```bash
# Train fake-news detector (LR + CNN + BiLSTM on 44K articles)
python -m src.training.train_fake_models

# Train AI misinformation detector (LR + CNN + BiLSTM on 500 articles)
python -m src.training.train_ai_models
```

### 4. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

Visit **http://localhost:8501** 🚀

---

## 🤖 Models

### Training Pipeline

```
Raw Text
  ↓ Lowercase · URL removal · Punctuation stripping
  ↓ Stopword removal · Lemmatization (NLTK WordNet)
  ↓
TF-IDF (Logistic Regression)        Tokenizer + Padding (CNN / BiLSTM)
  ↓                                   ↓ vocab=4000 · max_len=120
Logistic Regression                 Embedding(64d)
  C=1.0 · class_weight=balanced       ↓
                                    [CNN]  Conv1D×3 → GlobalMaxPool
                                    [BiLSTM] BiLSTM→BiLSTM → Dense
```

### Model Selection

After training, the highest-accuracy model from each dataset is automatically selected:

- `models/final/best_fake_model.*`
- `models/final/best_ai_model.*`

### Fusion

```
best_fake_model  ─→  fake_prob
                           ↘
                      Weighted Average Fusion  →  fused_prob  →  verdict
                           ↗
best_ai_model    ─→  ai_prob

weights ∝ individual model validation accuracy
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | True Fake / All predicted Fake |
| Recall | True Fake / All actual Fake |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under the ROC curve |

Target accuracy range: **75–85%** (realistic for production).

---

## 🌐 Dashboard Features

| Feature | Description |
|---|---|
| 🏠 Landing Page | Project overview, pipeline, tech stack, team |
| 🧪 Simulation Mode | Paste text → instant prediction + gauge + charts |
| 🌐 Real-Time Mode | Fetch 20 live articles → analyze all → pie chart + export CSV |
| 📊 Model Analytics | Training curves, accuracy comparison, fusion weights |
| 🌙 Dark Mode | Toggle via sidebar |
| 📥 Export CSV | Download real-time results |
| 🔍 Search & Filter | Filter news cards by verdict or keyword |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.12+ / Keras |
| Classical ML | Scikit-learn |
| NLP | NLTK (lemmatization, stopwords) |
| Dashboard | Streamlit 1.28+ |
| Charts | Plotly |
| Data | Pandas, NumPy |
| Scraping | FeedParser, BeautifulSoup4, Requests |
| Serialisation | Joblib |

---

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app/streamlit_app.py`

> **Note:** Pre-train models locally and commit the `models/final/` directory since cloud instances have limited compute.

### Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
```

---

## 📘 Dataset Sources

| Dataset | Source |
|---|---|
| Fake News | LIAR / ISOT Fake News Dataset |
| AI News | Synthetic AI misinformation benchmark |

> Raw files must be placed in `data/raw/` and `data/ai_news/` before training.

---

## 👥 Team

Built by **Team TruthLens** · © 2026 · MIT License
