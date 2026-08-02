"""
integration.py
Integration layer — combines the three module scores into one data-quality score
and a pass/fail decision.

Layer 1 — Purpose profiles: baseline weights chosen by the pipeline's purpose.
Layer 2 — Confidence-based adaptive weighting: a module whose score is far from 0.5
          is confident and gets more weight; a score near 0.5 is uncertain and gets less.
          Grounded in the entropy-confidence idea of Almarshad et al. (2025).

Formula (verified against the worked example in the project file):
    confidence_i   = abs(score_i - 0.5)
    raw_weight_i   = baseline_i * (0.5 + confidence_i)
    final_weight_i = raw_weight_i / sum(raw_weights)
    combined_score = sum(final_weight_i * score_i)
All module scores must be in [0,1]. Pass if combined < threshold (default 0.5), else fail.

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd
from scipy import stats

# Layer 1 — purpose profiles (each sums to 1.0)
PROFILES = {
    "general":    {"anomaly": 0.3, "drift": 0.5, "missing": 0.2},
    "fraud":      {"anomaly": 0.5, "drift": 0.3, "missing": 0.2},
    "compliance": {"anomaly": 0.2, "drift": 0.3, "missing": 0.5},
}


# ======================================================================
# Core: Layer 1 + Layer 2 combination
# ======================================================================
def combine_scores(scores, mode="general", threshold=0.5, custom_weights=None):
    """
    Combine per-module batch scores into one data-quality score and decision.

    scores : dict with keys 'anomaly', 'drift', 'missing', each a float in [0,1].
    mode   : 'general' | 'fraud' | 'compliance' | 'custom'.
    custom_weights : required if mode == 'custom'; dict of baselines summing to 1.

    Returns a dict: combined_score, decision ('PASS'/'FAIL'), final weights, baselines.
    """
    baselines = custom_weights if mode == "custom" else PROFILES[mode]

    raw = {}
    for k, s in scores.items():
        confidence = abs(s - 0.5)
        raw[k] = baselines[k] * (0.5 + confidence)
    total = sum(raw.values())
    final = {k: raw[k] / total for k in raw}

    combined = sum(final[k] * scores[k] for k in scores)
    decision = "FAIL" if combined >= threshold else "PASS"
    return {
        "combined_score": float(combined),
        "decision": decision,
        "weights": {k: float(v) for k, v in final.items()},
        "baselines": baselines,
        "module_scores": dict(scores),
    }


# ======================================================================
# Batch-level module scores (each reduces a module's output to one [0,1] value)
# ======================================================================
def batch_anomaly_score(batch_amount, ref_mean, ref_std, z_thresh=3.0):
    """Fraction of batch rows whose log-amount is > z_thresh SDs from the reference."""
    vals = np.log1p(np.clip(batch_amount.values, 0, None))
    if ref_std == 0:
        return 0.0
    z = np.abs((vals - ref_mean) / ref_std)
    return float((z > z_thresh).mean())


def batch_drift_score(drift_clf, feature_vector):
    """Random Forest drift probability for the batch (already in [0,1])."""
    return float(drift_clf.predict_proba(feature_vector.reshape(1, -1))[0, 1])


def batch_missing_score(missing_clf, X_batch):
    """Mean predicted null-risk across the batch (Module 3 XGBoost), in [0,1]."""
    return float(missing_clf.predict_proba(X_batch)[:, 1].mean())


# ======================================================================
# DQFramework — orchestrates the three modules + integration
# ======================================================================
class DQFramework:
    """
    Entry point: DQFramework(reference_data, mode, threshold).
    Fits the three modules on the clean reference, then .assess(batch) returns a
    full data-quality report for an incoming batch.

    The fitted module components are passed in (already trained) to keep this class
    focused on orchestration; see the notebook for how they are built.
    """

    def __init__(self, reference_data, fitted, mode="general", threshold=0.5):
        self.reference = reference_data
        self.fitted = fitted            # dict of fitted module objects
        self.mode = mode
        self.threshold = threshold

    def assess(self, batch):
        f = self.fitted
        anomaly = batch_anomaly_score(batch["Amount Paid"],
                                      f["ref_log_mean"], f["ref_log_std"])
        drift = batch_drift_score(f["drift_clf"],
                                  f["drift_feature_fn"](self.reference, batch))
        missing = batch_missing_score(f["missing_clf"],
                                      f["missing_feature_fn"](batch, f["missing_target"])
                                      .reindex(columns=f["missing_columns"], fill_value=0))
        report = combine_scores({"anomaly": anomaly, "drift": drift, "missing": missing},
                                mode=self.mode, threshold=self.threshold)
        return report
