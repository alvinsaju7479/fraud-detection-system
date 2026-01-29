# src/utils/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def pr_auc(y_true, y_proba) -> float:
    return float(average_precision_score(y_true, y_proba))

def find_threshold_by_cost(y_true, y_proba, cost_fn=500, cost_fp=5):
    """
    Choose threshold minimizing expected cost:
      cost = FN*cost_fn + FP*cost_fp
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    best = {"threshold": 0.5, "cost": float("inf"), "fp": None, "fn": None}

    y_true = np.asarray(y_true)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "fp": fp, "fn": fn}

    return best

def precision_recall_at_threshold(y_true, y_proba, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall), tp, fp, fn
