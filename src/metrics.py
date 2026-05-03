"""Evaluation metrics for URL phishing classifiers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.utils import safe_divide


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, float]:
    """Return common binary classification metrics."""

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "true_positive_rate": round(safe_divide(tp, tp + fn), 4),
        "false_positive_rate": round(safe_divide(fp, fp + tn), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def average_latency_seconds(items: list[Any], predict_fn: Callable[[Any], Any]) -> float:
    """Measure average prediction latency for a callable."""

    if not items:
        return 0.0
    start = time.perf_counter()
    for item in items:
        predict_fn(item)
    elapsed = time.perf_counter() - start
    return round(elapsed / len(items), 6)
