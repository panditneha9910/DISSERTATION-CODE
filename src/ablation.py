"""
ablation.py
Ablation study — measures each module's individual contribution to the combined
framework, and compares weighting schemes.

Experiments (per the project plan):
  A1: remove Module 1 (anomaly), run drift + missing only
  A2: remove Module 2 (drift),   run anomaly + missing only
  A3: remove Module 3 (missing), run anomaly + drift only
  A4: compare weighting schemes — equal (1/3 each) vs severity (0.3/0.5/0.2 fixed)
      vs adaptive (severity baselines + Layer 2 confidence weighting)

Method: build a labelled set of batches (clean -> should PASS, corrupted -> should
FAIL). Corrupted batches carry one of several fault types so different modules are
responsible for catching different faults. Run each framework variant, threshold the
combined score at 0.5, and score the PASS/FAIL decision against the true label with F1.
The drop in F1 when a module is removed is that module's contribution.

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import module2_drift as m2
import integration as ig

SEED = 42
SEVERITY = {"anomaly": 0.3, "drift": 0.5, "missing": 0.2}


# ----------------------------------------------------------------------
# Corruption: make a batch that a specific module should catch
# ----------------------------------------------------------------------
def corrupt_batch(batch, kind, seed=42):
    """
    kind: 'anomaly' (extreme values), 'drift' (distribution shift),
          'missing' (inject missing values), or 'mixed'.
    """
    b = batch.copy()
    rng = np.random.default_rng(seed)
    if kind in ("anomaly", "mixed"):
        # very large multiplier so anomalies clear the heavy natural tail (see Module 1
        # finding): x1000 is only ~1.9 SD on the log scale, so we use a much larger factor.
        n = int(len(b) * 0.02)
        idx = rng.choice(b.index, size=n, replace=False)
        b.loc[idx, "Amount Paid"] = b.loc[idx, "Amount Paid"] * 1_000_000_000
    if kind in ("drift", "mixed"):
        b = m2.inject_distribution_shift(b, column="Amount Paid", rate=0.25,
                                         factor=50, seed=int(rng.integers(1e9)))
    if kind in ("missing", "mixed"):
        n = int(len(b) * 0.20)
        idx = rng.choice(b.index, size=n, replace=False)
        b.loc[idx, "Receiving Currency"] = np.nan
    return b


# ----------------------------------------------------------------------
# Batch -> three module scores
# ----------------------------------------------------------------------
def compute_scores(batch, reference, fitted):
    """
    Return {'anomaly','drift','missing'} batch severity scores in [0,1], where 0 is
    clean and ~1 is a clear single-dimension fault. The raw signals are on different
    natural scales (a flagged-fraction, a probability, a missing-rate), so anomaly and
    missing are rescaled to a common severity by a 'significant level': 1% of rows
    flagged, or 5% missing, maps to severity 1.
    """
    # z_thresh=5 so natural heavy-tail extremes (max ~2.4e10 on this data sit below 5 SD
    # on the log scale) do not register as anomalies; only injected extremes do.
    frac_anom = ig.batch_anomaly_score(batch["Amount Paid"],
                                       fitted["ref_log_mean"], fitted["ref_log_std"],
                                       z_thresh=5.0)
    anomaly = min(1.0, frac_anom / 0.01)
    drift = ig.batch_drift_score(fitted["drift_clf"],
                                 m2.drift_feature_vector(reference, batch))
    missing_rate = float(batch["Receiving Currency"].isna().mean())
    missing = min(1.0, missing_rate / 0.05)
    return {"anomaly": anomaly, "drift": drift, "missing": missing}


def build_labelled_scores(reference, ref_pool, fitted, n_each=15, seed=42):
    """
    Build labelled batches: clean (label 0) and corrupted (label 1) of each fault type.
    Returns list of (scores_dict, label).
    """
    rng = np.random.default_rng(seed)
    data = []
    kinds = ["anomaly", "drift", "missing"]
    for i in range(n_each):
        # clean
        clean = ref_pool.sample(20000, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        data.append((compute_scores(clean, reference, fitted), 0))
        # one corrupted batch per fault type
        for kind in kinds:
            base = ref_pool.sample(20000, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            corr = corrupt_batch(base, kind, seed=int(rng.integers(1e9)))
            data.append((compute_scores(corr, reference, fitted), 1))
    return data


# ----------------------------------------------------------------------
# Weighting schemes
# ----------------------------------------------------------------------
def _combine_fixed(scores, baselines):
    """Weighted average with fixed (normalised) baselines — no confidence weighting."""
    keys = list(scores.keys())
    total = sum(baselines[k] for k in keys)
    return sum((baselines[k] / total) * scores[k] for k in keys)


def combined_score(scores, scheme, included):
    """
    Compute the combined score for a subset of modules under a weighting scheme.

    Weighted schemes (equal / severity / adaptive) average the dimension scores, which
    dilutes a single-dimension fault. The 'max' scheme takes the strongest dimension
    signal, giving proper gate semantics: fail if ANY dimension is bad. The ablation
    compares them to show that averaging is unsuitable for a multi-dimension gate.
    """
    sub = {k: scores[k] for k in included}
    if scheme == "equal":
        return float(np.mean([sub[k] for k in sub]))
    if scheme == "severity":
        return _combine_fixed(sub, SEVERITY)
    if scheme == "adaptive":
        return ig.combine_scores(sub, mode="general")["combined_score"]
    if scheme == "max":
        return float(max(sub.values()))
    raise ValueError(scheme)


# ----------------------------------------------------------------------
# Run the experiments
# ----------------------------------------------------------------------
def evaluate_variant(labelled, scheme, included, threshold=0.5):
    """F1 and accuracy of the PASS/FAIL decision for one framework variant."""
    y = [lab for _, lab in labelled]
    pred = [1 if combined_score(s, scheme, included) >= threshold else 0
            for s, _ in labelled]
    return {"f1": f1_score(y, pred, zero_division=0),
            "accuracy": accuracy_score(y, pred)}


def run_ablation(labelled, decision_scheme="max"):
    """
    Run A1-A4 and return a results table.

    A1-A3 use the `decision_scheme` (default 'max', the proper gate rule) and remove one
    module at a time, so the drop in F1 shows that module's contribution.
    A4 compares all weighting schemes on the full module set.
    """
    all_mods = ["anomaly", "drift", "missing"]
    rows = []
    # Full framework
    full = evaluate_variant(labelled, decision_scheme, all_mods)
    rows.append([f"A0 Full framework ({decision_scheme})", full["f1"], full["accuracy"]])
    # A1-A3: remove each module
    labels = {"anomaly": "A1 Remove Module 1 (anomaly)",
              "drift": "A2 Remove Module 2 (drift)",
              "missing": "A3 Remove Module 3 (missing)"}
    for removed in all_mods:
        kept = [m for m in all_mods if m != removed]
        r = evaluate_variant(labelled, decision_scheme, kept)
        rows.append([labels[removed], r["f1"], r["accuracy"]])
    # A4: weighting schemes (all three modules)
    for scheme in ["equal", "severity", "adaptive", "max"]:
        r = evaluate_variant(labelled, scheme, all_mods)
        rows.append([f"A4 Weighting: {scheme}", r["f1"], r["accuracy"]])
    return pd.DataFrame(rows, columns=["experiment", "f1", "accuracy"])
