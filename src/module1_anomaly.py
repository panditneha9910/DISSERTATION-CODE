"""
module1_anomaly.py
Module 1 — value-level anomaly detection.

Five detectors, built in the order set by the project plan:
  1. Isolation Forest  (global, tree-based)
  2. Local Outlier Factor (local, density-based; novelty mode)
  3. Z-score baseline  (univariate, 3 standard deviations)
  4. IQR baseline      (univariate, 1.5 x IQR)
  5. Autoencoder       (PyTorch)

Design:
- The ML detectors (iForest, LOF) are FIT on the clean reference feature matrix and
  then SCORE the incoming feature matrix. This matches the framework's "compare
  incoming to reference" design and avoids fitting on corrupted data.
- All detectors return an anomaly score where HIGHER = more anomalous, plus a binary
  flag. Scores are min-max normalised to [0,1] so they can feed the integration layer.
- Ground truth for evaluation is the injected-anomaly mask from inject_anomalies
  (optionally combined with the real Is_Laundering label).

Author: Neha Pandit 
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

SEED = 42


def _normalise(scores):
    """Min-max scale scores to [0,1]. Higher = more anomalous."""
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo == 0:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ----------------------------------------------------------------------
# 1. Isolation Forest
# ----------------------------------------------------------------------
def fit_isolation_forest(X_ref, contamination="auto", n_estimators=200):
    """Fit an Isolation Forest on the clean reference features."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_ref)
    return model


def score_isolation_forest(model, X):
    """
    Anomaly score in [0,1], higher = more anomalous.
    sklearn's score_samples gives higher values for normal points, so we negate.
    """
    raw = -model.score_samples(X)
    return _normalise(raw)


# ----------------------------------------------------------------------
# 2. Local Outlier Factor
# ----------------------------------------------------------------------
def fit_lof(X_ref, n_neighbors=20):
    """Fit LOF in novelty mode on the clean reference features."""
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_ref)
    return model


def score_lof(model, X):
    """Anomaly score in [0,1], higher = more anomalous."""
    raw = -model.score_samples(X)
    return _normalise(raw)


# ----------------------------------------------------------------------
# 3. Z-score baseline 
# ----------------------------------------------------------------------
def zscore_scores(values, ref_values=None, threshold=3.0):
    """
    Z-score anomaly score on a 1-D array (e.g. log amount).
    Mean and std are taken from ref_values (the clean reference) if provided,
    otherwise from `values` itself.

    Returns (score01, flags) where score01 in [0,1] and flags is |z| > threshold.
    """
    values = np.asarray(values, dtype=float)
    base = np.asarray(ref_values, dtype=float) if ref_values is not None else values
    mu, sigma = base.mean(), base.std()
    if sigma == 0:
        z = np.zeros_like(values)
    else:
        z = np.abs((values - mu) / sigma)
    flags = z > threshold
    return _normalise(z), flags


# ----------------------------------------------------------------------
# 4. IQR baseline
# ----------------------------------------------------------------------
def iqr_scores(values, ref_values=None, k=1.5):
    """
    IQR anomaly score on a 1-D array. Q1/Q3 taken from ref_values if provided.
    Returns (score01, flags) where flags mark values beyond Q1-k*IQR or Q3+k*IQR.
    """
    values = np.asarray(values, dtype=float)
    base = np.asarray(ref_values, dtype=float) if ref_values is not None else values
    q1, q3 = np.percentile(base, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    # distance outside the fence (0 if inside), used as a graded score
    dist = np.maximum(np.maximum(lower - values, values - upper), 0.0)
    flags = (values < lower) | (values > upper)
    return _normalise(dist), flags


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(scores, ground_truth, threshold=0.5, flags=None):
    
    y = np.asarray(ground_truth).astype(int)
    pred = np.asarray(flags).astype(int) if flags is not None else (np.asarray(scores) >= threshold).astype(int)
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    try:
        out["roc_auc"] = roc_auc_score(y, scores)
        out["pr_auc"] = average_precision_score(y, scores)
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def evaluate_topk(scores, ground_truth, k):
   
    scores = np.asarray(scores)
    y = np.asarray(ground_truth).astype(bool)
    thr = np.quantile(scores, 1 - k)
    pred = scores >= thr
    tp = np.sum(pred & y)
    precision = tp / max(pred.sum(), 1)
    recall = tp / max(y.sum(), 1)
    return {"precision": float(precision), "recall": float(recall), "threshold": float(thr)}
