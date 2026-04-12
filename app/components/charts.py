"""
charts.py — Plotly chart components for TruthLens
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


BLUE  = "#0055A4"
CYAN  = "#27aeef"
RED   = "#e05c5c"
GREEN = "#2ecc71"
DARK  = "#333333"
LIGHT = "#F5F5F5"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font_family  ="'Inter', 'DejaVu Sans', sans-serif",
    font_color   =DARK,
    margin       =dict(l=20, r=20, t=40, b=20),
)


def fake_real_pie(fake_count: int, real_count: int) -> go.Figure:
    """Pie chart of Fake vs Real breakdown."""
    fig = go.Figure(go.Pie(
        labels=["Fake / Mis.", "Real / Credible"],
        values=[fake_count, real_count],
        marker_colors=[RED, GREEN],
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} articles<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Fake vs Real Breakdown", font_size=16, font_color=BLUE),
        showlegend=True,
        **CHART_LAYOUT,
    )
    return fig


def confidence_gauge(confidence: float, verdict: str) -> go.Figure:
    """Gauge chart for single-article confidence."""
    color = RED if verdict == "FAKE" else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={"reference": 50, "increasing": {"color": RED}, "decreasing": {"color": GREEN}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#d4edda"},
                {"range": [40, 60], "color": "#fff3cd"},
                {"range": [60, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": DARK, "width": 3},
                "thickness": 0.8, "value": 50,
            },
        },
        title={"text": f"Confidence — {verdict}", "font": {"size": 16, "color": BLUE}},
    ))
    fig.update_layout(height=260, **CHART_LAYOUT)
    return fig


def score_bar(scores: dict) -> go.Figure:
    """Horizontal bar chart for individual model scores."""
    labels = list(scores.keys())
    values = [v * 100 for v in scores.values()]
    colors = [RED if v > 50 else GREEN for v in values]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Model Score Breakdown", font_size=16, font_color=BLUE),
        xaxis=dict(range=[0, 110], ticksuffix="%"),
        **CHART_LAYOUT,
    )
    return fig


def model_accuracy_bar(scores: dict, title: str = "Model Accuracy Comparison") -> go.Figure:
    """Compare multiple model accuracies."""
    names = list(scores.keys())
    vals  = [round(v * 100, 2) for v in scores.values()]
    colors = [BLUE, CYAN, "#5E533E"]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colors[:len(names)],
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font_size=16, font_color=BLUE),
        yaxis=dict(range=[0, 110], ticksuffix="%"),
        **CHART_LAYOUT,
    )
    return fig


def history_chart(history: dict, title: str = "Training History") -> go.Figure:
    """Training vs Validation loss + accuracy curves."""
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=epochs, y=history.get("loss", []),
        name="Train Loss", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_loss", []),
        name="Val Loss", line=dict(color=RED, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("accuracy", []),
        name="Train Acc", line=dict(color=CYAN, width=2),
        yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_accuracy", []),
        name="Val Acc", line=dict(color=GREEN, width=2, dash="dash"),
        yaxis="y2"))

    fig.update_layout(
        title=dict(text=title, font_size=16, font_color=BLUE),
        xaxis_title="Epoch",
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
        **CHART_LAYOUT,
    )
    return fig
