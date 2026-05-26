# 🔍 TruthLens — An Attention-Enhanced Framework for AI-Generated Fake News

> **Unmasking misinformation in the age of AI — detecting fake news before it influences the world.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![Explainable AI](https://img.shields.io/badge/AI-ExplainableAI-yellow)]()
[![Pandas](https://img.shields.io/badge/Pandas-DataAnalysis-purple?logo=pandas)](https://pandas.pydata.org) 
[![NumPy](https://img.shields.io/badge/NumPy-NumericalComputing-blue?logo=numpy)](https://numpy.org)

---

## 🎯 Overview
<p align="center">
  <img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/2ad49893-9b9a-4835-8449-a57d89c67be0" />
  <br>
  <em>TruthLens Project Overview</em>
</p>


TruthLens is a production-ready AI system that classifies news articles as **Real (0)** or **Fake/Misinformation (1)**. The framework is specifically designed to detect both **human-written fake news** and **AI-generated misinformation** using a hybrid Machine Learning and Deep Learning pipeline.

The system operates on two independent detection tracks and combines their intelligence using a **weighted attention-based fusion mechanism** to generate accurate and explainable predictions.

| Track | Dataset | Purpose |
|---|---|---|
| 🧠 Human Fake News | Fake.csv + True.csv (44,898 articles) | Detects human-written misinformation |
| 🤖 AI News | ai_news_dataset.csv (9,891 samples) | Detects AI-generated misinformation |

TruthLens trains a total of **6 models** across both datasets:

- Logistic Regression  
- Convolutional Neural Network (CNN)  
- Bidirectional LSTM (BiLSTM)  

The best-performing models from each dataset are selected and fused into a unified architecture enhanced with an **attention mechanism** for improved interpretability and contextual understanding.

The platform also includes a **premium Streamlit dashboard** featuring:

- 🧪 Simulation Mode for custom news testing  
- 🌐 Real-Time Mode for live news analysis  
- 📊 Confidence scores and visual analytics  
- 🧠 Explainable predictions with reasoning and sources  

TruthLens combines NLP, Deep Learning, Explainable AI, and real-time inference to create a scalable and intelligent misinformation detection system for modern digital media environments.


---

## 🤖 Models

### Training Pipeline
<img width="1247" height="350" alt="image" src="https://github.com/user-attachments/assets/fdd63446-ec7d-4375-92d5-06b48789d442" />


### Model Selection

After training, the highest-accuracy model from each dataset is automatically selected:

- `models/final/best_fake_model.*`
- `models/final/best_ai_model.*`

### Fusion
<img width="1247" height="350" alt="image" src="https://github.com/user-attachments/assets/a840fec0-13a7-4746-8a8d-9107cf23e2d9" />


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

## 📸 Project Snapshots


### 📊 Project Overview Dashboard
The dashboard displays dataset statistics, number of trained models, and supported detection categories for both human-written and AI-generated fake news.

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/30378f9c-6464-40a0-9857-67aa6810556a" />




### 🧪 Simulation Mode
Simulation Mode allows users to manually input news content and analyze whether the news is real or fake along with confidence scores and explainable reasoning.

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/2e9830ac-11c5-4656-9c09-9578f3c341a9" />
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/bbb14c7a-7738-4400-99ff-1294fcc93af9" />




### 🌐 Real-Time News Analysis
The Real-Time Mode fetches live news articles from online sources and performs instant misinformation detection using the TruthLens inference pipeline.

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/7b09805d-ef03-43be-a879-18000b775be7" />
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/ef16a563-5670-4664-a204-fbf757775000" />
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/19633fe5-153c-4423-bfe7-f109de1af533" />



### 📈 Prediction
The system provides prediction confidence scores, attention-based explanations, and interactive visualizations for better interpretability and user understanding.

<p align="center">
  <img src="https://github.com/user-attachments/assets/86d10e47-d159-465f-859f-e51a4c0f64d1" width="48%" />
  <img src="https://github.com/user-attachments/assets/da06f942-bb02-47fc-8282-0ca8aba5e9d6" width="48%" />
</p>


---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="20"/> |
| Deep Learning | TensorFlow 2.12+ / Keras <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="20"/> |
| Classical ML | Scikit-learn <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="22"/> |
| NLP | NLTK (lemmatization, stopwords) <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*YM2HXc7f4v02pZBEO8h-qw.png" width="22"/> |
| Dashboard | Streamlit 1.28+ <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="70"/> |
| Charts | Plotly <img src="https://images.plot.ly/logo/new-branding/plotly-logomark.png" width="20"/> |
| Data | Pandas <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="20"/> & NumPy <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="20"/> |
| Scraping | BeautifulSoup4  Requests <img src="https://requests.readthedocs.io/en/latest/_static/requests-sidebar.png" width="22"/> |
| Serialisation | Joblib |
---
## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | True Fake / All predicted Fake |
| Recall | True Fake / All actual Fake |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under the ROC curve |

### Target accuracy range: **75–85%** .
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
<div align="center">

**Made with ❤️ by M Harish Gautham**

⭐ If you find this project helpful, please star it! ⭐

</div>
