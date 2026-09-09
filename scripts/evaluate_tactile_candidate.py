from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def binary_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    if ground_truth.shape != prediction.shape:
        raise ValueError("mask shapes must match")
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)
    union = np.logical_or(ground_truth, prediction).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(ground_truth, prediction).sum() / union)


def summarize_presence(pairs: Sequence[tuple[bool, bool]]) -> dict[str, float | int | None]:
    if not pairs:
        raise ValueError("presence pairs are required")
    true_positive = sum(truth and prediction for truth, prediction in pairs)
    false_negative = sum(truth and not prediction for truth, prediction in pairs)
    false_positive = sum(not truth and prediction for truth, prediction in pairs)
    true_negative = sum(not truth and not prediction for truth, prediction in pairs)
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None,
        "false_positive_rate": false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else None,
    }
