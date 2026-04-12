"""
navbar.py — TruthLens navigation bar component
"""
import streamlit as st


def render_navbar(current_page: str = "home"):
    st.markdown("""
    <nav class="tl-navbar">
        <div class="tl-nav-brand">
            <span class="tl-logo-icon">🔍</span>
            <span class="tl-logo-text">Truth<span class="tl-logo-accent">Lens</span></span>
        </div>
        <div class="tl-nav-tagline">AI Misinformation Detection System</div>
    </nav>
    """, unsafe_allow_html=True)
