"""
metrics.py
──────────
Unified evaluation metrics for all TruthLens models.
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None) -> dict:
    """
    Compute full classification metric suite.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc,
                    confusion_matrix, report
    """
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":    classification_report(y_true, y_pred, output_dict=True),
    }
    if y_proba is not None:
        try:
            m["auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            m["auc"] = None
    return m


def save_results(metrics: dict, path: str):
    """Save metrics dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Saved → {path}")


def print_summary(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation summary."""
    border = "─" * 50
    print(f"\n{border}")
    print(f"  {model_name} Evaluation Summary")
    print(border)
    print(f"  Accuracy  : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision : {metrics.get('precision', 0):.4f}")
    print(f"  Recall    : {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score  : {metrics.get('f1', 0):.4f}")
    if metrics.get("auc") is not None:
        print(f"  ROC-AUC   : {metrics.get('auc', 0):.4f}")
    print(border + "\n")
