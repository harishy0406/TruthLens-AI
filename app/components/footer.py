"""
footer.py — TruthLens footer component
"""
import streamlit as st


def render_footer():
    st.markdown("""
    <footer class="tl-footer">
        <div class="tl-footer-inner">
            <div class="tl-footer-brand">
                <span class="tl-logo-icon">🔍</span>
                <span class="tl-footer-name">TruthLens</span>
            </div>
            <div class="tl-footer-desc">
                AI-Powered Misinformation Detection — Combining Human & AI Fake News Detection
            </div>
            <div class="tl-footer-stack">
                <span class="tl-tech-tag">Python</span>
                <span class="tl-tech-tag">TensorFlow</span>
                <span class="tl-tech-tag">Scikit-learn</span>
                <span class="tl-tech-tag">Streamlit</span>
                <span class="tl-tech-tag">NLTK</span>
                <span class="tl-tech-tag">Plotly</span>
            </div>
            <div class="tl-footer-copy">
                © 2026 TruthLens · Built with ❤️ by Team TruthLens
            </div>
        </div>
    </footer>
    """, unsafe_allow_html=True)
