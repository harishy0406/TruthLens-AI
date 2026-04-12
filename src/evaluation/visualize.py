"""
visualize.py
────────────
Generates evaluation plots: confusion matrix, ROC curve,
Precision-Recall curve, and training history curves.
All plots saved to models/<subdir>/ as PNG files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
)


# ── Shared style ───────────────────────────────────────────────────────────────
BLUE  = "#0055A4"
CYAN  = "#27aeef"
DARK  = "#333333"
LIGHT = "#F5F5F5"

plt.rcParams.update({
    "figure.facecolor": LIGHT,
    "axes.facecolor":   LIGHT,
    "font.family":      "DejaVu Sans",
    "axes.labelcolor":  DARK,
    "xtick.color":      DARK,
    "ytick.color":      DARK,
})


def plot_confusion_matrix(y_true, y_pred, save_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax,
                linewidths=0.5)
    ax.set_title(title, fontsize=14, color=BLUE, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[viz] Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_proba, save_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, color=BLUE, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[viz] ROC curve saved → {save_path}")


def plot_pr_curve(y_true, y_proba, save_path: str, title: str = "Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=CYAN, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=14, color=BLUE, pad=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[viz] PR curve saved → {save_path}")


def plot_training_history(history: dict, save_path: str, title: str = "Training History"):
    """Plot training vs. validation loss and accuracy from Keras history dict."""
    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.get("loss", []), color=BLUE, label="Train Loss")
    axes[0].plot(history.get("val_loss", []), color=CYAN, label="Val Loss", linestyle="--")
    axes[0].set_title("Loss", color=BLUE); axes[0].legend(); axes[0].set_xlabel("Epoch")

    # Accuracy
    axes[1].plot(history.get("accuracy", []), color=BLUE, label="Train Acc")
    axes[1].plot(history.get("val_accuracy", []), color=CYAN, label="Val Acc", linestyle="--")
    axes[1].set_title("Accuracy", color=BLUE); axes[1].legend(); axes[1].set_xlabel("Epoch")

    fig.suptitle(title, fontsize=14, color=DARK, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Training history saved → {save_path}")


def plot_model_comparison(scores: dict, save_path: str, title: str = "Model Comparison"):
    """Bar chart comparing model accuracies."""
    names = list(scores.keys())
    vals  = [scores[k] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=[BLUE, CYAN, "#5E533E"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=14, color=BLUE, pad=12)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=DARK)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[viz] Model comparison saved → {save_path}")
