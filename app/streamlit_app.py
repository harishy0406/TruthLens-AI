"""
streamlit_app.py
─────────────────
TruthLens – AI Misinformation Detection System
Main Streamlit dashboard entry point.

Run:
    streamlit run app/streamlit_app.py
"""

import os, sys, json, time
import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.components.navbar  import render_navbar
from app.components.footer  import render_footer
from app.components.cards   import verdict_card, news_card, stat_card
from app.components.charts  import (
    fake_real_pie, confidence_gauge, score_bar,
    model_accuracy_bar, history_chart,
)

# ══════════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TruthLens – AI Misinformation Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Lazy imports (heavy) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load prediction pipeline once and cache across sessions."""
    try:
        from src.utils.predict import TruthLensPredictor
        p = TruthLensPredictor()
        p._load()
        return p, None
    except Exception as e:
        return None, str(e)


def _predictor_available() -> bool:
    model_dir = os.path.join(ROOT, "models", "final")
    return (
        os.path.exists(os.path.join(model_dir, "vectorizer.pkl")) and
        os.path.exists(os.path.join(model_dir, "tokenizer.pkl"))
    )


# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px">
            <span style="font-size:2.5rem">🔍</span><br>
            <span style="font-size:1.3rem; font-weight:800; color:#0055A4;">
                Truth<span style="color:#27aeef">Lens</span>
            </span>
        </div>
        <hr style="border-color:rgba(0,85,164,0.15); margin:12px 0">
        """, unsafe_allow_html=True)

        st.markdown("**📍 Navigation**")
        page = st.radio(
            "",
            ["🏠 Home", "🧪 Simulation Mode", "🌐 Real-Time Mode", "📊 Model Analytics"],
            key="nav_radio",
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(0,85,164,0.10)'>", unsafe_allow_html=True)
        st.markdown("**⚙️ Settings**")

        threshold = st.slider(
            "Detection Threshold", 0.30, 0.80, 0.50, 0.05,
            help="Probability threshold above which an article is flagged as Fake."
        )

        dark_mode = st.toggle("🌙 Dark Mode", value=False)
        if dark_mode:
            st.markdown(
                "<style>:root { --bg:#0d1117; --bg-card:#161b22; "
                "--text-main:#e6edf3; --text-muted:#7d8590; }</style>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:rgba(0,85,164,0.10)'>", unsafe_allow_html=True)

        # Model status
        st.markdown("**🤖 Model Status**")
        trained = _predictor_available()
        if trained:
            st.success("✅ Models loaded & ready")
        else:
            st.warning("⚠️ Models not trained yet")
            st.info("Run the training scripts first:\n```\npython -m src.training.train_fake_models\npython -m src.training.train_ai_models\n```")

        return page.split(" ", 1)[1].strip(), threshold


# ════════════════════════════════════════════════════════════════
# PAGE: HOME / LANDING
# ════════════════════════════════════════════════════════════════
def page_home():
    render_navbar()

    # Hero
    st.markdown("""
    <div class="tl-hero">
        <div class="tl-hero-title">
            Can You Trust What<br>You Read? <span>We Can.</span>
        </div>
        <div class="tl-hero-sub">
            TruthLens is an AI-powered misinformation detection system that identifies
            fake news — whether human-written or AI-generated — with explainable confidence scores.
        </div>
        <div class="tl-badge-row">
            <span class="tl-badge">✅ Real-Time Detection</span>
            <span class="tl-badge">🤖 AI-Fake Aware</span>
            <span class="tl-badge">🧠 Explainable AI</span>
            <span class="tl-badge">⚡ BiLSTM + CNN</span>
            <span class="tl-badge">🔗 Multi-Model Fusion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview stats ──────────────────────────────────────────
    st.markdown("""
    <div class="tl-section-header">
        <span class="tl-section-icon">📊</span>
        <div>
            <div class="tl-section-title">Project Overview</div>
            <div class="tl-section-sub">Dataset &amp; Model Statistics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Training Articles", "44,898", "📰", "Fake + Real News")
    with c2: stat_card("AI News Samples",   "10000",    "🤖", "Misinformation labeled")
    with c3: stat_card("ML/DL Models",      "6",      "🧠", "LR + CNN + BiLSTM × 2")
    with c4: stat_card("Detection Types",   "2",      "🎯", "Human & AI Fake News")

    # ── How It Works ────────────────────────────────────────────
    st.markdown("""
    <div class="tl-section-header" style="margin-top:40px">
        <span class="tl-section-icon">⚙️</span>
        <div>
            <div class="tl-section-title">How It Works</div>
            <div class="tl-section-sub">5-step detection pipeline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("NLP Preprocessing",   "Text is lowercased, URLs/HTML removed, stopwords filtered, and lemmatized."),
        ("Feature Extraction",  "TF-IDF vectorization for Logistic Regression; Token padding for CNN & BiLSTM."),
        ("Dual Detection",      "Fake-news detector & AI-misinformation detector run independently."),
        ("Model Fusion",        "Weighted attention fusion combines both detector scores into one probability."),
        ("Verdict + Confidence","Final label (Real/Fake), confidence %, and risk level are returned."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="tl-step">
            <div class="tl-step-num">{i}</div>
            <div>
                <div class="tl-step-title">{title}</div>
                <div class="tl-step-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Technologies Used ────────────────────────────────────────
    st.markdown("""
    <div class="tl-section-header" style="margin-top:40px">
        <span class="tl-section-icon">🛠️</span>
        <div>
            <div class="tl-section-title">Technologies Used</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tech_list = [
        ("Python 3.10+", "Core language"),
        ("TensorFlow / Keras", "CNN & BiLSTM models"),
        ("Scikit-learn", "Logistic Regression & TF-IDF"),
        ("NLTK", "NLP preprocessing & lemmatization"),
        ("Streamlit", "Interactive web dashboard"),
        ("Plotly", "Dynamic visualisations"),
        ("Pandas / NumPy", "Data pipeline"),
        ("FeedParser", "RSS news scraping"),
        ("BeautifulSoup4", "Article text extraction"),
        ("Joblib", "Model serialisation"),
    ]

    cols = st.columns(5)
    for i, (tech, desc) in enumerate(tech_list):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="tl-feature-card" style="padding:14px 12px; margin-bottom:10px">
                <div class="tl-feature-title" style="font-size:0.9rem">{tech}</div>
                <div class="tl-feature-text">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Models Used ──────────────────────────────────────────────
    st.markdown("""
    <div class="tl-section-header" style="margin-top:40px">
        <span class="tl-section-icon">🤖</span>
        <div>
            <div class="tl-section-title">Models Used</div>
            <div class="tl-section-sub">3 models × 2 datasets = 6 total models with fusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mcols = st.columns(3)
    models_info = [
        ("📈", "Logistic Regression", "TF-IDF features with L2 regularisation. Fast, interpretable baseline."),
        ("🌀", "CNN",                 "1-D Convolutional layers detect local n-gram patterns with high efficiency."),
        ("🔁", "BiLSTM",              "Bidirectional LSTM captures long-range sequential dependencies both ways."),
    ]
    for col, (icon, name, desc) in zip(mcols, models_info):
        with col:
            st.markdown(f"""
            <div class="tl-feature-card">
                <div class="tl-feature-icon">{icon}</div>
                <div class="tl-feature-title">{name}</div>
                <div class="tl-feature-text">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Fusion
    st.markdown("""
    <div style="background:linear-gradient(135deg,#e8f4ff,#f0f9ff);
                border-radius:12px; padding:20px 24px; margin-top:12px;
                border:1px solid rgba(0,85,164,0.12)">
        <b style="color:#0055A4">⚡ Model Fusion</b> — The best model from each dataset is combined
        using <b>weighted attention averaging</b> (weights proportional to individual accuracy),
        plus an optional <b>stacking meta-learner</b> for maximum performance.
    </div>
    """, unsafe_allow_html=True)

    # ── Team ────────────────────────────────────────────────────
    st.markdown("""
    <div class="tl-section-header" style="margin-top:40px">
        <span class="tl-section-icon">👥</span>
        <div><div class="tl-section-title">Team TruthLens</div></div>
    </div>
    """, unsafe_allow_html=True)

    team = [
        ("https://media.licdn.com/dms/image/v2/D5603AQEd_Zab6PG6JA/profile-displayphoto-scale_200_200/B56Z01b_IuK0AY-/0/1774718027623?e=1777507200&v=beta&t=bi1CCaik0M9uz4WX7fQPVVJQ_-Mug5PGMvLxvDpFmHE", "M Harish Gautham",      "ML pipeline, model training, fusion, Streamlit dashboard, CSS design system", "https://linkedin.com/in/mharishy46"),
        ("https://media.licdn.com/dms/image/v2/D4D03AQHM6hOWzsTRVQ/profile-displayphoto-shrink_200_200/B4DZRmCyVBHYAY-/0/1736878794662?e=1777507200&v=beta&t=AWEVf8pRohvna31NUFbNMugRIlSPHD5BLfHmfoJDt2M", "Prasurjya Boruah",      "EDA, feature engineering, evaluation, Text preprocessing, NLTK, embeddings", "https://www.linkedin.com/in/prasurjya-boruah-b70153347/"),
    ]
    tcols = st.columns(4)
    for col, (avatar_url, role, desc, linkedin) in zip(tcols, team):
        with col:
            st.markdown(f"""
            <div class="tl-team-card" style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
                <div>
                    <div class="tl-team-avatar" style="padding:0; background:none;">
                        <img src="{avatar_url}" style="width:100%; height:100%; border-radius:50%; object-fit:cover;">
                    </div>
                    <div class="tl-team-name">{role}</div>
                    <div class="tl-feature-text" style="margin-top:8px;font-size:0.78rem">{desc}</div>
                </div>
                <div style="margin-top:20px;">
                    <a href="{linkedin}" target="_blank" style="display:inline-block; padding:8px 24px; background:#0a66c2; color:white; text-decoration:none; border-radius:50px; font-weight:600; font-size:0.85rem; transition:all 0.3s; width:100%; box-sizing:border-box;">
                        🔗 LinkedIn
                    </a>
                </div>
            </div>""", unsafe_allow_html=True)

    render_footer()


# ════════════════════════════════════════════════════════════════
# PAGE: SIMULATION MODE
# ════════════════════════════════════════════════════════════════
def page_simulation(threshold: float = 0.5):
    render_navbar()

    st.markdown("""
    <div class="tl-section-header">
        <span class="tl-section-icon">🧪</span>
        <div>
            <div class="tl-section-title">Simulation Mode</div>
            <div class="tl-section-sub">Paste any article text and get an instant prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _predictor_available():
        st.error("⚠️ **Models not yet trained.** Run training scripts before using Simulation Mode.")
        st.code("python -m src.training.train_fake_models\npython -m src.training.train_ai_models", language="bash")
        render_footer()
        return

    predictor, err = load_predictor()
    if err:
        st.error(f"Failed to load predictor: {err}")
        render_footer()
        return

    # ── Input ──────────────────────────────────────────────────
    col_in, col_sample = st.columns([3, 1])
    with col_in:
        user_text = st.text_area(
            "📝 Paste your news article text here:",
            height=220,
            placeholder=(
                "Paste a full news article or headline here. "
                "Minimum 20 words recommended for accurate detection."
            ),
            key="sim_input",
        )

    def set_fake_sample():
        st.session_state["sim_input"] = (
            "BREAKING: Scientists confirm that drinking bleach mixed with lemon "
            "juice cures COVID-19 in 24 hours. A secret government study suppressed "
            "by mainstream media reveals this miracle cure that Big Pharma doesn't "
            "want you to know about. Share this before it gets deleted!"
        )

    def set_real_sample():
        st.session_state["sim_input"] = (
            "The Federal Reserve announced Wednesday that it would hold interest "
            "rates steady, citing a need to observe the effects of previous rate "
            "increases on inflation. Fed Chair Jerome Powell said the committee "
            "remains attentive to risks and will act as appropriate based on "
            "incoming data and the evolving economic outlook."
        )

    with col_sample:
        st.markdown("**💡 Try a sample:**")
        st.button("🟥 Fake Sample", use_container_width=True, on_click=set_fake_sample)
        st.button("🟩 Real Sample", use_container_width=True, on_click=set_real_sample)

    word_count = len(user_text.split()) if user_text else 0
    if user_text:
        st.caption(f"📏 {word_count} words · {len(user_text)} characters")

    predict_btn = st.button("🔍 Analyze Article", use_container_width=False, key="predict_btn")

    if predict_btn:
        if not user_text or word_count < 5:
            st.warning("Please enter at least 5 words to analyze.")
        else:
            with st.spinner("🧠 Analyzing article through TruthLens pipeline…"):
                time.sleep(0.6)  # Brief UX pause
                result = predictor.predict(user_text)

            if "error" in result:
                st.error(result["error"])
            else:
                # Apply custom threshold
                fused = result.get("fused_prob", 0.5)
                result["label"]   = int(fused >= threshold)
                result["verdict"] = "FAKE" if result["label"] == 1 else "REAL"

                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")

                # Verdict card
                verdict_card(result)

                # Charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    gauge = confidence_gauge(result["confidence"], result["verdict"])
                    st.plotly_chart(gauge, use_container_width=True, key="sim_gauge")

                with chart_col2:
                    scores = {
                        "Fake Detector": result.get("fake_score", 0.5),
                        "AI Detector":   result.get("ai_score",   0.5),
                        "Fused Score":   result.get("fused_prob", 0.5),
                    }
                    bar = score_bar(scores)
                    st.plotly_chart(bar, use_container_width=True, key="sim_bar")

                # Explainability
                with st.expander("🔬 Explain This Prediction", expanded=False):
                    st.markdown(f"""
                    | Component | Score |
                    |---|---|
                    | 🧠 Human Fake-News Detector | `{result['fake_score']*100:.1f}%` |
                    | 🤖 AI-Generated Misinformation Detector | `{result['ai_score']*100:.1f}%` |
                    | ⚡ Fused Ensemble Probability | `{result['fused_prob']*100:.1f}%` |
                    | 🎯 Detection Threshold | `{threshold*100:.0f}%` |
                    | ✅ Final Verdict | **{result['verdict']}** |
                    | 📊 Confidence | **{result['confidence']*100:.1f}%** |
                    """)

                    st.info(
                        "TruthLens runs two independent detectors: one trained on human-written "
                        "fake news, the other on AI-generated misinformation. Their outputs are "
                        "weighted by each model's validation accuracy and fused into a final score."
                    )

    # ── History ────────────────────────────────────────────────
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []

    if predict_btn and "result" in dir() and "error" not in result:
        st.session_state["analysis_history"].append({
            "text":    user_text[:100] + "…",
            "verdict": result["verdict"],
            "confidence": f"{result['confidence']*100:.1f}%",
            "fused":   f"{result['fused_prob']*100:.1f}%",
        })

    if st.session_state["analysis_history"]:
        st.markdown("---")
        st.markdown("### 📋 Session History")
        hist_df = pd.DataFrame(st.session_state["analysis_history"])
        # Color verdicts
        def color_verdict(val):
            color = "#dc2626" if val == "FAKE" else "#16a34a"
            return f"color: {color}; font-weight: bold"
        st.dataframe(
            hist_df.style.applymap(color_verdict, subset=["verdict"]),
            use_container_width=True, hide_index=True,
        )
        if st.button("🗑️ Clear History"):
            st.session_state["analysis_history"] = []
            st.rerun()

    render_footer()


# ════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME MODE
# ════════════════════════════════════════════════════════════════
def page_realtime(threshold: float = 0.5):
    render_navbar()

    st.markdown("""
    <div class="tl-section-header">
        <span class="tl-section-icon">🌐</span>
        <div>
            <div class="tl-section-title">Real-Time Mode</div>
            <div class="tl-section-sub">Live news articles analyzed as they come in</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    with ctrl_col1:
        n_articles = st.selectbox("Articles to fetch", [10, 15, 20], index=2)
    with ctrl_col2:
        filter_verdict = st.selectbox("Filter by Verdict", ["All", "FAKE", "REAL"])
    with ctrl_col3:
        search_term = st.text_input("🔍 Search articles", placeholder="e.g. climate, politics…")

    fetch_btn   = st.button("📡 Fetch & Analyze Live News", use_container_width=True, type="primary")
    export_btn  = st.button("📥 Export Results CSV", use_container_width=False)

    if fetch_btn:
        if not _predictor_available():
            st.error("⚠️ Run training scripts first before using Real-Time Mode.")
            render_footer()
            return

        predictor, err = load_predictor()
        if err:
            st.error(f"Model loading error: {err}")
            render_footer()
            return

        st.session_state["rt_articles"] = None
        st.session_state["rt_results"]  = None

        with st.spinner("📡 Fetching live news articles…"):
            try:
                from src.utils.scraper import fetch_news
                articles = fetch_news(n=n_articles)
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                render_footer()
                return

        if not articles:
            st.warning("No articles found. Check your internet connection.")
            render_footer()
            return

        results = []
        prog = st.progress(0, text="🧠 Analyzing articles…")
        for i, article in enumerate(articles):
            text = article.get("summary", "") or article.get("title", "")
            res  = predictor.predict(text) if text else {"label": 0, "verdict": "REAL", "confidence": 0.5}

            # Apply threshold
            fused = res.get("fused_prob", 0.5)
            res["label"]   = int(fused >= threshold)
            res["verdict"] = "FAKE" if res["label"] == 1 else "REAL"

            results.append(res)
            prog.progress((i + 1) / len(articles), text=f"Analyzed {i+1}/{len(articles)}…")

        prog.empty()
        st.session_state["rt_articles"] = articles
        st.session_state["rt_results"]  = results

    # ── Display results ────────────────────────────────────────
    articles = st.session_state.get("rt_articles")
    results  = st.session_state.get("rt_results")

    if articles and results:
        # Stats row
        fake_count = sum(1 for r in results if r["label"] == 1)
        real_count = len(results) - fake_count
        avg_conf   = sum(r.get("confidence", 0.5) for r in results) / len(results)

        s1, s2, s3, s4 = st.columns(4)
        with s1: stat_card("Total Analyzed",  str(len(results)), "📰")
        with s2: stat_card("Flagged FAKE",    str(fake_count),   "⚠️", f"{fake_count/len(results)*100:.0f}%")
        with s3: stat_card("Verified REAL",   str(real_count),   "✅", f"{real_count/len(results)*100:.0f}%")
        with s4: stat_card("Avg Confidence",  f"{avg_conf*100:.0f}%", "📊")

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            pie = fake_real_pie(fake_count, real_count)
            st.plotly_chart(pie, use_container_width=True, key="rt_pie")
        with ch2:
            source_counts = {}
            for a, r in zip(articles, results):
                src = a.get("source", "Unknown")
                if src not in source_counts:
                    source_counts[src] = {"FAKE": 0, "REAL": 0}
                source_counts[src][r["verdict"]] += 1

            import plotly.graph_objects as go
            srcs   = list(source_counts.keys())
            fakes  = [source_counts[s]["FAKE"] for s in srcs]
            reals  = [source_counts[s]["REAL"] for s in srcs]
            fig_src = go.Figure(data=[
                go.Bar(name="FAKE", x=srcs, y=fakes, marker_color="#ef4444"),
                go.Bar(name="REAL", x=srcs, y=reals, marker_color="#10b981"),
            ])
            fig_src.update_layout(
                barmode="stack", title="Results by Source",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#333333", margin=dict(l=20, r=20, t=40, b=60),
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_src, use_container_width=True, key="rt_src")

        # ── Article cards ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Article Results")

        pairs = list(zip(articles, results))
        if filter_verdict != "All":
            pairs = [(a, r) for a, r in pairs if r["verdict"] == filter_verdict]
        if search_term:
            pairs = [(a, r) for a, r in pairs
                     if search_term.lower() in a.get("title", "").lower()
                     or search_term.lower() in a.get("source", "").lower()]

        if not pairs:
            st.info("No articles match the current filter.")
        else:
            for i, (article, result) in enumerate(pairs):
                news_card(article, result, i)

        # Export
        if export_btn or st.session_state.get("export_trigger"):
            rows = []
            for a, r in zip(articles, results):
                rows.append({
                    "Title":      a.get("title", ""),
                    "Source":     a.get("source", ""),
                    "URL":        a.get("url", ""),
                    "Published":  a.get("published_at", ""),
                    "Verdict":    r.get("verdict", ""),
                    "Confidence": f"{r.get('confidence', 0)*100:.1f}%",
                    "FakeScore":  f"{r.get('fake_score', 0)*100:.1f}%",
                    "AIScore":    f"{r.get('ai_score', 0)*100:.1f}%",
                })
            df_export = pd.DataFrame(rows)
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download CSV",
                data=csv_bytes,
                file_name="truthlens_results.csv",
                mime="text/csv",
            )

    render_footer()


# ════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ════════════════════════════════════════════════════════════════
def page_analytics():
    render_navbar()

    st.markdown("""
    <div class="tl-section-header">
        <span class="tl-section-icon">📊</span>
        <div>
            <div class="tl-section-title">Model Analytics</div>
            <div class="tl-section-sub">Training results, accuracy comparisons, and history charts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def _load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    fake_results = _load_json(os.path.join(ROOT, "models", "fake_models", "results.json"))
    ai_results   = _load_json(os.path.join(ROOT, "models", "ai_models",   "results.json"))
    fake_meta    = _load_json(os.path.join(ROOT, "models", "final", "fake_meta.json"))
    ai_meta      = _load_json(os.path.join(ROOT, "models", "final", "ai_meta.json"))

    if not fake_results and not ai_results:
        st.warning("⚠️ No training results found. Run the training scripts to populate this page.")
        st.code("python -m src.training.train_fake_models\npython -m src.training.train_ai_models", language="bash")
        render_footer()
        return

    tab1, tab2, tab3 = st.tabs(["📰 Fake News Models", "🤖 AI News Models", "⚡ Model Fusion"])

    with tab1:
        if fake_results:
            st.markdown(f"**Best model:** `{fake_meta.get('best_model','—').upper()}` "
                        f"— Accuracy `{fake_meta.get('best_accuracy',0)*100:.2f}%`")
            fig = model_accuracy_bar(fake_results, "Fake News Dataset — Model Comparison")
            st.plotly_chart(fig, use_container_width=True, key="fake_acc_bar")

            # Training history charts
            import joblib
            for model_name in ["cnn", "bilstm"]:
                hist_path = os.path.join(ROOT, "models", "fake_models", f"{model_name}_history.pkl")
                if os.path.exists(hist_path):
                    hist = joblib.load(hist_path)
                    fig_h = history_chart(hist, f"{model_name.upper()} Training History (Fake Dataset)")
                    st.plotly_chart(fig_h, use_container_width=True, key=f"fake_{model_name}_hist")
        else:
            st.info("Train fake-news models to see results here.")

    with tab2:
        if ai_results:
            st.markdown(f"**Best model:** `{ai_meta.get('best_model','—').upper()}` "
                        f"— Accuracy `{ai_meta.get('best_accuracy',0)*100:.2f}%`")
            fig = model_accuracy_bar(ai_results, "AI News Dataset — Model Comparison")
            st.plotly_chart(fig, use_container_width=True, key="ai_acc_bar")

            import joblib
            for model_name in ["cnn", "bilstm"]:
                hist_path = os.path.join(ROOT, "models", "ai_models", f"{model_name}_history.pkl")
                if os.path.exists(hist_path):
                    hist = joblib.load(hist_path)
                    fig_h = history_chart(hist, f"{model_name.upper()} Training History (AI Dataset)")
                    st.plotly_chart(fig_h, use_container_width=True, key=f"ai_{model_name}_hist")
        else:
            st.info("Train AI-news models to see results here.")

    with tab3:
        st.markdown("### ⚡ Fusion Strategy")

        if fake_meta and ai_meta:
            fa = fake_meta.get("best_accuracy", 0.5)
            aa = ai_meta.get("best_accuracy",   0.5)
            total = fa + aa
            fw = fa / total; aw = aa / total

            st.markdown(f"""
            | Model | Best Accuracy | Fusion Weight |
            |---|---|---|
            | 🧠 Fake News ({fake_meta.get('best_model','?').upper()}) | `{fa*100:.2f}%` | `{fw:.3f}` |
            | 🤖 AI News ({ai_meta.get('best_model','?').upper()})   | `{aa*100:.2f}%` | `{aw:.3f}` |
            """)

            import plotly.graph_objects as go
            fig_w = go.Figure(go.Pie(
                labels=["Fake Detector Weight", "AI Detector Weight"],
                values=[fw, aw], hole=0.5,
                marker_colors=["#0055A4", "#27aeef"],
            ))
            fig_w.update_layout(
                title="Fusion Weight Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#333333",
            )
            st.plotly_chart(fig_w, use_container_width=True, key="fusion_weight_pie")
        else:
            st.info("Train both model sets to see fusion analytics.")

    render_footer()


# ════════════════════════════════════════════════════════════════
# Main Router
# ════════════════════════════════════════════════════════════════
def main():
    page, threshold = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Simulation Mode":
        page_simulation(threshold)
    elif page == "Real-Time Mode":
        page_realtime(threshold)
    elif page == "Model Analytics":
        page_analytics()


if __name__ == "__main__":
    main()
