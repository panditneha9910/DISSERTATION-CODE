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
def combine_scores(scores, mode="general", threshold=0.5, custom_weights=None,
                   combine_mode="weighted"):
    """
    Combine per-module batch scores into one data-quality score and decision.

    scores : dict with keys 'anomaly', 'drift', 'missing', each a float in [0,1].
    mode   : 'general' | 'fraud' | 'compliance' | 'custom'.
    custom_weights : required if mode == 'custom'; dict of baselines summing to 1.
    combine_mode :
        'weighted' (default) — Layer 1 profile x Layer 2 confidence weighted average.
                    This averages the dimensions, which DILUTES a single-dimension fault
                    (a batch bad on only one axis is pulled towards the middle).
        'max'      — gate semantics: the combined score is the strongest single dimension,
                    so the batch FAILs if ANY dimension is bad. This is the configuration
                    the ablation study validates for a multi-dimension quality gate.
    The confidence weights are always returned (informative) even under 'max'.

    Returns a dict: combined_score, decision ('PASS'/'FAIL'), final weights, baselines.
    """
    baselines = custom_weights if mode == "custom" else PROFILES[mode]

    raw = {}
    for k, s in scores.items():
        confidence = abs(s - 0.5)
        raw[k] = baselines[k] * (0.5 + confidence)
    total = sum(raw.values())
    final = {k: raw[k] / total for k in raw}

    if combine_mode == "max":
        combined = max(scores.values())
    elif combine_mode == "weighted":
        combined = sum(final[k] * scores[k] for k in scores)
    else:
        raise ValueError("combine_mode must be 'weighted' or 'max'")
    decision = "FAIL" if combined >= threshold else "PASS"
    return {
        "combined_score": float(combined),
        "decision": decision,
        "combine_mode": combine_mode,
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

    def __init__(self, reference_data, fitted, mode="general", threshold=0.5,
                 amount_col="Amount Paid", combine_mode="weighted", z_thresh=3.0,
                 anomaly_sig=None, missing_observed_col=None, missing_sig=0.05,
                 drift_baseline=None):
        self.reference = reference_data
        self.fitted = fitted            # dict of fitted module objects
        self.mode = mode
        self.threshold = threshold
        self.amount_col = amount_col     # 'Amount Paid' (IBM) or 'Amount' (Credit Card)
        # --- scoring controls (defaults reproduce the original weighted-average behaviour) ---
        self.combine_mode = combine_mode          # 'weighted' (old default) or 'max' (gate)
        self.z_thresh = z_thresh                  # z cut-off for the anomaly fraction
        self.anomaly_sig = anomaly_sig            # if set, anomaly fraction -> severity min(1, frac/sig)
        self.missing_observed_col = missing_observed_col  # column whose OBSERVED null-rate is the gate signal
        self.missing_sig = missing_sig            # null-rate that maps to missing-severity 1
        self.drift_baseline = drift_baseline      # if set, drift proba is rescaled RELATIVE to
        #                                           normal clean variation: severity =
        #                                           (proba - baseline)/(1 - baseline), clipped to
        #                                           [0,1]. Stops mild/normal drift from firing.

    def assess(self, batch):
        """
        Return a data-quality report for one incoming batch.

        Gate signals (all in [0,1], comparable):
          anomaly : fraction of rows beyond z_thresh SDs on log-amount. If anomaly_sig is
                    set, rescaled to a severity (min(1, fraction/anomaly_sig)) so a single-
                    dimension fault is not lost on a tiny fraction scale.
          drift   : Random Forest drift probability.
          missing : if missing_observed_col is set, the OBSERVED null-rate severity of that
                    column (direct data-quality signal); otherwise the ML model's predicted
                    null-risk (legacy behaviour).
        The Module 3 ML model is always reported separately as 'missing_risk_warning'
        (forward-looking early warning), independent of what drives the gate.
        """
        f = self.fitted
        frac = batch_anomaly_score(batch[self.amount_col],
                                   f["ref_log_mean"], f["ref_log_std"], z_thresh=self.z_thresh)
        anomaly = min(1.0, frac / self.anomaly_sig) if self.anomaly_sig else frac

        drift_raw = batch_drift_score(f["drift_clf"],
                                      f["drift_feature_fn"](self.reference, batch))
        if self.drift_baseline is not None:
            drift = min(1.0, max(0.0, (drift_raw - self.drift_baseline)
                                 / (1.0 - self.drift_baseline + 1e-9)))
        else:
            drift = drift_raw

        # predicted null-risk (early warning) — available whenever the model is present
        risk = None
        if f.get("missing_clf") is not None and f.get("missing_feature_fn") is not None:
            Xb = (f["missing_feature_fn"](batch, f["missing_target"])
                  .reindex(columns=f["missing_columns"], fill_value=0))
            risk = batch_missing_score(f["missing_clf"], Xb)

        # gate missing signal
        if self.missing_observed_col and self.missing_observed_col in batch.columns:
            mrate = float(batch[self.missing_observed_col].isna().mean())
            missing = min(1.0, mrate / self.missing_sig) if self.missing_sig else mrate
        else:
            missing = risk if risk is not None else 0.0

        report = combine_scores({"anomaly": anomaly, "drift": drift, "missing": missing},
                                mode=self.mode, threshold=self.threshold,
                                combine_mode=self.combine_mode)
        report["missing_risk_warning"] = risk
        return report
