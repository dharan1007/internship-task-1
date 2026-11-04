"""Declared multiclass and common binary endpoints."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def five_class_metrics(truth: list[int], predicted: list[int]) -> dict[str, object]:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Evaluation needs matching nonempty truth and predictions")
    if not set(truth + predicted).issubset(set(range(5))):
        raise ValueError("Five-grade metric received an out-of-range label")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(5)), zero_division=0
    )
    kappa = cohen_kappa_score(truth, predicted, labels=list(range(5)), weights="quadratic")
    if not np.isfinite(kappa):
        raise ValueError("QWK undefined for this label distribution")
    return {
        "count": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "undefined_precision_policy": "zero when a class has no predicted examples",
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)), "qwk": float(kappa),
        "per_class": [{"grade": i, "support": int(support[i]), "precision": float(precision[i]),
                       "recall": float(recall[i]), "f1": float(f1[i])} for i in range(5)],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(5))).tolist(),
    }


def referable_metrics(truth: list[int], predicted: list[int], *, original_messidor: bool) -> dict[str, object]:
    """Common endpoint: EyePACS ICDR >=2; original Messidor R2/R3."""
    if not truth or len(truth) != len(predicted):
        raise ValueError("Binary evaluation needs matching nonempty labels")
    if not set(truth).issubset(set(range(4 if original_messidor else 5))):
        raise ValueError("Truth outside its source grading scheme")
    if not set(predicted).issubset(set(range(5))):
        raise ValueError("Prediction outside EyePACS grading scheme")
    true_binary = [int(grade >= 2) for grade in truth]
    pred_binary = [int(grade >= 2) for grade in predicted]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_binary, pred_binary, labels=[0, 1], zero_division=0
    )
    return {
        "definition": ("Messidor R2/R3 versus model ICDR 2/3/4" if original_messidor
                       else "ICDR 2/3/4 versus ICDR 0/1"),
        "count": len(truth), "accuracy": float(accuracy_score(true_binary, pred_binary)),
        "precision_referable": float(precision[1]), "recall_referable": float(recall[1]),
        "f1_referable": float(f1[1]), "referable_support": int(support[1]),
        "confusion_matrix": confusion_matrix(true_binary, pred_binary, labels=[0, 1]).tolist(),
    }
