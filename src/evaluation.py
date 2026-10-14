"""Comprehensive model evaluation and credit risk metrics module.

Provides:
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score)
- Financial cost matrix evaluation (asymmetric FP vs FN costs)
- Threshold optimization
- FICO-style credit score mapping
- Visualization exports (Confusion Matrix, ROC & PR curves)
- DVC metrics JSON serialization
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Calculates comprehensive classification and financial loss metrics."""
    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0))
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = float(cost_fp * fp + cost_fn * fn)
    metrics["Financial_Cost"] = total_cost

    if y_pred_proba is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["PR-AUC"] = float(average_precision_score(y_true, y_pred_proba))
        metrics["Brier_Score"] = float(brier_score_loss(y_true, y_pred_proba))

    print(f"\n================ {model_name} Performance ================")
    for k, v in metrics.items():
        if k == "Financial_Cost":
            print(f"{k:16s}: {v:,.0f} (FP={fp}, FN={fn})")
        else:
            print(f"{k:16s}: {v:.4f}")
    print("==========================================================")
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0
) -> Dict[str, Any]:
    """Finds the decision threshold minimizing total financial cost."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_cost = float("inf")
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)
            best_metrics = {
                "threshold": best_thresh,
                "cost": float(cost),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0))
            }

    return best_metrics


def probability_to_credit_score(
    prob_default: np.ndarray,
    base_score: int = 600,
    pdo: int = 40,
    min_score: int = 300,
    max_score: int = 850
) -> np.ndarray:
    """Translates default probability into standard credit score (FICO style).
    
    Formula: Score = base_score + pdo * log2((1 - p) / p)
    """
    p = np.clip(prob_default, 1e-6, 1.0 - 1e-6)
    odds = (1.0 - p) / p
    scores = base_score + pdo * np.log2(odds)
    return np.clip(np.round(scores), min_score, max_score).astype(int)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> None:
    """Plots and optionally saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    models_probas: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plots overlaid ROC curves for multiple models."""
    plt.figure(figsize=(8, 6))
    for name, proba in models_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Comparison - ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def save_metrics_report(metrics_dict: Dict[str, Any], save_path: str = "reports/metrics.json") -> None:
    """Saves metrics dictionary to a JSON file for DVC tracking."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)