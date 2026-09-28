"""Evaluation metrics and formatting utilities for the Match Outcome Predictor."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

from .preprocessing import CLASS_LABELS, INT_TO_CLASS, INT_TO_HUMAN


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    """Calculate multiclass Brier score (mean squared error of predicted probabilities).

    Lower is better (0 = perfect calibration, ~0.667 = uniform guessing).
    """
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def evaluate_match_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate multiclass match predictions across standard statistical metrics.

    Args:
        y_true: Ground truth integer labels (0: H, 1: D, 2: A).
        y_pred: Predicted integer labels (0: H, 1: D, 2: A).
        y_prob: Predicted probability array of shape (N, 3).

    Returns:
        Dictionary of evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Log loss
    ll = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))

    # Multiclass ROC-AUC (One-vs-Rest, weighted)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        roc_auc_ovr = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc_ovr = float("nan")

    # Brier score
    brier = multiclass_brier_score(y_true, y_prob, n_classes=3)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Class-wise report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Home Win (H)", "Draw (D)", "Away Win (A)"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "log_loss": ll,
        "roc_auc_ovr": roc_auc_ovr,
        "brier_score": brier,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def format_classification_report_table(report_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format classification report dictionary into a clean pandas DataFrame."""
    rows = []
    class_keys = [
        ("Home Win (H)", "H", "Home Win"),
        ("Draw (D)", "D", "Draw"),
        ("Away Win (A)", "A", "Away Win"),
    ]
    for key, code, name in class_keys:
        if key in report_dict:
            rows.append({
                "Class": f"{name} ({code})",
                "Precision": f"{report_dict[key]['precision']:.4f}",
                "Recall": f"{report_dict[key]['recall']:.4f}",
                "F1": f"{report_dict[key]['f1-score']:.4f}",
                "Support": int(report_dict[key]['support']),
            })

    # Summary rows
    if "macro avg" in report_dict:
        rows.append({
            "Class": "Macro Average",
            "Precision": f"{report_dict['macro avg']['precision']:.4f}",
            "Recall": f"{report_dict['macro avg']['recall']:.4f}",
            "F1": f"{report_dict['macro avg']['f1-score']:.4f}",
            "Support": int(report_dict['macro avg']['support']),
        })
    if "weighted avg" in report_dict:
        rows.append({
            "Class": "Weighted Average",
            "Precision": f"{report_dict['weighted avg']['precision']:.4f}",
            "Recall": f"{report_dict['weighted avg']['recall']:.4f}",
            "F1": f"{report_dict['weighted avg']['f1-score']:.4f}",
            "Support": int(report_dict['weighted avg']['support']),
        })

    return pd.DataFrame(rows)
