"""
diagnostics.py
Explainability layer for the DQ framework.

When the framework blocks a batch, this turns the raw module signals into concrete,
human-readable findings: which check fired, which column is responsible, how severe it is,
and a plain-language reason. It is what lets a developer / QA / business reader see WHY a
batch was returned rather than just a PASS/FAIL flag.

diagnose(batch, reference, ...) -> a report dict:
  {
    'decision': 'FAIL'|'PASS', 'combined': float,
    'findings': [ {dimension, column, severity, headline, detail}, ... ],  # only real issues
    'checks': {'anomaly': sev, 'drift': sev, 'missing': sev},               # 0..1 per dimension
  }

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
from scipy import stats


def _anomaly_findings(batch, reference, numeric_cols, z_thresh=5.0, frac_sig=0.005):
    """Fraction of rows whose value is > z_thresh SDs from the reference (log scale)."""
    out = []
    for col in numeric_cols:
        if col not in batch.columns or col not in reference.columns:
            continue
        ref = np.log1p(np.clip(reference[col].dropna().values, 0, None))
        if ref.std() == 0:
            continue
        val = np.log1p(np.clip(batch[col].dropna().values, 0, None))
        frac = float((np.abs((val - ref.mean()) / ref.std()) > z_thresh).mean())
        if frac >= frac_sig:
            sev = 0.5 + 0.5 * min(1.0, (frac / frac_sig - 1) / 4)  # material finding -> [0.5, 1]
            out.append(dict(dimension="anomaly", column=col, severity=sev,
                            headline=f"{frac*100:.1f}% of '{col}' values are extreme outliers",
                            detail=f"{frac*100:.2f}% of rows in '{col}' sit more than {z_thresh:.0f} "
                                   f"standard deviations from normal — a sign of corrupted or fraudulent values."))
    return out


def _drift_findings(batch, reference, numeric_cols, ks_sig=0.1):
    """
    Per-column KS distance between batch and reference distributions.
    The KS test is run on log1p(value): financial amounts are extremely heavy-tailed, so on
    the raw scale a real multiplicative shift barely moves the statistic (the tail dominates
    the range). Working on the log scale makes the test sensitive to genuine shifts and
    matches the log-scale evidence chart.
    """
    out = []
    for col in numeric_cols:
        if col not in batch.columns or col not in reference.columns:
            continue
        ref = np.log1p(np.clip(reference[col].dropna().values, 0, None))
        cur = np.log1p(np.clip(batch[col].dropna().values, 0, None))
        if len(ref) == 0 or len(cur) == 0:
            continue
        ks, _ = stats.ks_2samp(ref, cur)
        if ks >= ks_sig:
            sev = 0.5 + 0.5 * min(1.0, (ks / ks_sig - 1) / 2)  # material finding -> [0.5, 1]
            out.append(dict(dimension="drift", column=col, severity=sev,
                            headline=f"'{col}' distribution has shifted (KS = {ks:.2f})",
                            detail=f"The pattern of '{col}' no longer matches recent data (KS distance {ks:.2f}). "
                                   f"Something upstream that produces '{col}' has likely changed."))
    return out


def _missing_findings(batch, columns, rate_sig=0.02):
    """Per-column observed null rate."""
    out = []
    for col in columns:
        if col not in batch.columns:
            continue
        rate = float(batch[col].isna().mean())
        if rate >= rate_sig:
            sev = 0.5 + 0.5 * min(1.0, (rate / rate_sig - 1) / 9)  # material finding -> [0.5, 1]
            out.append(dict(dimension="missing", column=col, severity=sev,
                            headline=f"{rate*100:.0f}% of '{col}' is missing",
                            detail=f"{rate*100:.1f}% of rows have no value for '{col}'. Downstream steps that "
                                   f"rely on '{col}' will break or silently drop records."))
    return out


def diagnose(batch, reference, numeric_cols, missing_cols, threshold=0.5,
             z_thresh=5.0, frac_sig=0.005, ks_sig=0.1, rate_sig=0.02):
    """
    Produce a human-readable fault report for one batch (see module docstring).
    numeric_cols : columns checked for anomalies + drift.
    missing_cols : columns checked for missing values.
    """
    a = _anomaly_findings(batch, reference, numeric_cols, z_thresh, frac_sig)
    d = _drift_findings(batch, reference, numeric_cols, ks_sig)
    m = _missing_findings(batch, missing_cols, rate_sig)
    findings = sorted(a + d + m, key=lambda f: f["severity"], reverse=True)

    checks = {
        "anomaly": max([f["severity"] for f in a], default=0.0),
        "drift":   max([f["severity"] for f in d], default=0.0),
        "missing": max([f["severity"] for f in m], default=0.0),
    }
    combined = max(checks.values())
    return dict(decision="FAIL" if combined >= threshold else "PASS",
                combined=float(combined), findings=findings, checks=checks)
