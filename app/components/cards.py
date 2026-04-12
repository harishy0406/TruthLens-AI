"""
cards.py — Reusable card components for TruthLens
"""
import streamlit as st


def verdict_card(result: dict):
    """Displays the main verdict after prediction."""
    label      = result.get("label", 0)
    verdict    = result.get("verdict", "REAL")
    confidence = result.get("confidence", 0.5)
    risk       = result.get("risk_level", "")
    fake_score = result.get("fake_score", 0.5)
    ai_score   = result.get("ai_score", 0.5)

    is_fake = label == 1
    card_class = "verdict-fake" if is_fake else "verdict-real"
    icon       = "⚠️" if is_fake else "✅"
    label_text = "FAKE / MISINFORMATION" if is_fake else "REAL / CREDIBLE"

    st.markdown(f"""
    <div class="verdict-card {card_class}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-label">{label_text}</div>
        <div class="verdict-confidence">Confidence: {confidence*100:.1f}%</div>
        <div class="verdict-risk">{risk}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score breakdown columns
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="score-pill">
            <div class="score-pill-label">🧠 Human Fake</div>
            <div class="score-pill-value">{fake_score*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="score-pill">
            <div class="score-pill-label">🤖 AI-Generated</div>
            <div class="score-pill-value">{ai_score*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        fused = result.get("fused_prob", 0.5)
        st.markdown(f"""
        <div class="score-pill">
            <div class="score-pill-label">⚡ Fused Score</div>
            <div class="score-pill-value">{fused*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)


def news_card(article: dict, result: dict = None, idx: int = 0):
    """Compact card for a single news article in real-time mode."""
    title   = article.get("title", "Untitled")[:120]
    source  = article.get("source", "Unknown")
    url     = article.get("url", "#")
    pub_at  = article.get("published_at", "")
    summary = article.get("summary", "")[:200]

    if result:
        is_fake   = result.get("label", 0) == 1
        confidence= result.get("confidence", 0.5)
        verdict   = result.get("verdict", "REAL")
        badge_cls = "badge-fake" if is_fake else "badge-real"
        badge_txt = f"{'⚠️ FAKE' if is_fake else '✅ REAL'} · {confidence*100:.0f}%"
    else:
        badge_cls = "badge-pending"
        badge_txt = "⏳ Analyzing…"

    st.markdown(f"""
    <div class="news-card" id="news-card-{idx}">
        <div class="news-card-header">
            <span class="news-source">{source}</span>
            <span class="news-date">{pub_at}</span>
        </div>
        <div class="news-title">
            <a href="{url}" target="_blank">{title}</a>
        </div>
        <div class="news-summary">{summary}</div>
        <div class="news-footer">
            <span class="news-badge {badge_cls}">{badge_txt}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value: str, icon: str = "📊", delta: str = ""):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {f'<div class="stat-delta">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)
