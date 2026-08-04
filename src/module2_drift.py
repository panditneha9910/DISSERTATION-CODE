"""
module2_drift.py
Module 2 — schema and distribution drift detection.

Three steps, per the project plan:
  Step 1 — Structural check: compare column names and dtypes between the clean
           reference batch and the incoming batch. Deterministic; catches dropped
           or added columns and dtype changes.
  Step 2 — Statistical check: for each numeric column a two-sample Kolmogorov-Smirnov
           test; for each categorical column a Chi-squared test on value frequencies.
           Also record mean shift, variance ratio, and null-rate change per column.
  Step 3 — Random Forest drift classifier: trained on the drift statistics from Step 2
           to predict whether a batch has drifted. Trained on injected drift at 5% and
           10% rates; evaluated at 20% and 30% with different seeds (reduces circularity).

Known limitation (state in methodology): the RF classifier is trained and evaluated on
synthetically injected drift, because no public financial dataset carries naturally
labelled drift. Mitigated by different injection rates and seeds for train vs evaluation.

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

SEED = 42

# Column roles for the IBM HI-Small_Trans.csv schema
NUMERIC_COLS = ["Amount Paid", "Amount Received"]
CATEGORICAL_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]


# ======================================================================
# STEP 1 — Structural check
# ======================================================================
def structural_check(reference_df, incoming_df):
    """
    Compare column names and dtypes between the reference and incoming batches.

    Returns a change_log dict recording dropped columns, added columns, and dtype
    changes. An empty dict means the structure is unchanged.
    """
    change_log = {}
    ref_cols, inc_cols = set(reference_df.columns), set(incoming_df.columns)

    dropped = sorted(ref_cols - inc_cols)
    added = sorted(inc_cols - ref_cols)
    if dropped:
        change_log["dropped_columns"] = dropped
    if added:
        change_log["added_columns"] = added

    dtype_changes = {}
    for col in ref_cols & inc_cols:
        if str(reference_df[col].dtype) != str(incoming_df[col].dtype):
            dtype_changes[col] = {
                "from": str(reference_df[col].dtype),
                "to": str(incoming_df[col].dtype),
            }
    if dtype_changes:
        change_log["dtype_changes"] = dtype_changes

    return change_log


# ======================================================================
# STEP 2 — Statistical check
# ======================================================================
def statistical_check(reference_df, incoming_df,
                      numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS,
                      alpha=0.05):
    """
    Per-column distribution comparison between reference and incoming.

    For numeric columns: two-sample KS test (scipy.stats.ks_2samp).
    For categorical columns: Chi-squared test on aligned value-frequency tables.
    Also records mean shift, variance ratio, and null-rate change per numeric column.

    Returns a dict: {column: {stat, p_value, drifted, ...}}. `drifted` is p < alpha.
    Columns absent from either frame are skipped (structural drift handles those).
    """
    results = {}

    for col in numeric_cols:
        if col not in reference_df.columns or col not in incoming_df.columns:
            continue
        ref = reference_df[col].dropna().values
        inc = incoming_df[col].dropna().values
        if len(ref) == 0 or len(inc) == 0:
            continue
        ks_stat, p = stats.ks_2samp(ref, inc)
        ref_std = ref.std() if ref.std() > 0 else 1.0
        ref_var = ref.var() if ref.var() > 0 else 1.0
        results[col] = {
            "type": "numeric",
            "ks_stat": float(ks_stat),
            "p_value": float(p),
            "drifted": bool(p < alpha),
            "mean_shift": float(abs(inc.mean() - ref.mean()) / ref_std),
            "variance_ratio": float(inc.var() / ref_var),
            "null_rate_change": float(incoming_df[col].isna().mean()
                                      - reference_df[col].isna().mean()),
        }

    for col in categorical_cols:
        if col not in reference_df.columns or col not in incoming_df.columns:
            continue
        # align categories across both batches
        cats = sorted(set(reference_df[col].dropna().unique())
                      | set(incoming_df[col].dropna().unique()))
        ref_counts = reference_df[col].value_counts().reindex(cats, fill_value=0).values
        inc_counts = incoming_df[col].value_counts().reindex(cats, fill_value=0).values
        table = np.vstack([ref_counts, inc_counts]) + 1  # +1 avoids zero cells
        chi2, p, _, _ = stats.chi2_contingency(table)
        results[col] = {
            "type": "categorical",
            "chi2_stat": float(chi2),
            "p_value": float(p),
            "drifted": bool(p < alpha),
            "null_rate_change": float(incoming_df[col].isna().mean()
                                      - reference_df[col].isna().mean()),
        }

    return results


def drift_feature_vector(reference_df, incoming_df,
                         numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS):
    """
    Turn the Step 2 statistics into a fixed-length numeric feature vector for the
    Random Forest classifier. Order is fixed so the vector is comparable across batches.
    """
    stats_map = statistical_check(reference_df, incoming_df, numeric_cols, categorical_cols)
    feats = []
    for col in numeric_cols:
        s = stats_map.get(col, {})
        feats += [s.get("ks_stat", 0.0), s.get("p_value", 1.0),
                  s.get("mean_shift", 0.0), s.get("variance_ratio", 1.0),
                  s.get("null_rate_change", 0.0)]
    for col in categorical_cols:
        s = stats_map.get(col, {})
        feats += [s.get("chi2_stat", 0.0), s.get("p_value", 1.0),
                  s.get("null_rate_change", 0.0)]
    return np.array(feats, dtype=float)


# ======================================================================
# Distribution-shift injection (drift ground truth for Step 3)
# ======================================================================
def inject_distribution_shift(df, column="Amount Paid", rate=0.10, factor=10.0, seed=42):
    """
    Inject a distribution shift by multiplying a random `rate` fraction of a numeric
    column by `factor`. This changes the column's distribution (detectable by the KS
    test) without changing the schema. Returns the drifted copy.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = int(len(df) * rate)
    idx = rng.choice(df.index, size=n, replace=False)
    df.loc[idx, column] = df.loc[idx, column] * factor
    return df


# ======================================================================
# STEP 3 — Random Forest drift classifier
# ======================================================================
def build_drift_dataset(reference_df, pool_df, rates, batch_size=20000,
                        n_per_rate=20, seed=42,
                        numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS,
                        drift_col="Amount Paid"):
    """
    Build a batch-level training/evaluation set for the drift classifier.

    For each rate in `rates`, generate `n_per_rate` drifted batches (distribution shift
    injected at that rate) and `n_per_rate` clean batches, each sampled from pool_df.
    Each example is one drift_feature_vector; label 1 = drifted, 0 = clean.

    IMPORTANT: pool_df must be drawn from the SAME period as the reference batch (i.e.
    the reference pool, days 1-3). If clean batches come from a later period they carry
    natural temporal drift, which makes them look drifted and confuses the classifier.
    Using same-period clean batches isolates the injected drift as the only signal.

    Returns (X, y).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for rate in rates:
        for i in range(n_per_rate):
            # clean batch
            batch = pool_df.sample(n=batch_size, random_state=int(rng.integers(1e9)))
            X.append(drift_feature_vector(reference_df, batch,
                                          numeric_cols, categorical_cols)); y.append(0)
            # drifted batch
            batch = pool_df.sample(n=batch_size, random_state=int(rng.integers(1e9)))
            drifted = inject_distribution_shift(batch, column=drift_col, rate=rate,
                                                seed=int(rng.integers(1e9)))
            X.append(drift_feature_vector(reference_df, drifted,
                                          numeric_cols, categorical_cols)); y.append(1)
    return np.array(X), np.array(y)


def train_drift_classifier(X, y):
    """Train a Random Forest on the batch-level drift feature vectors."""
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X, y)
    return clf


def evaluate_drift_classifier(clf, X, y):
    """Precision, Recall, F1, ROC-AUC and confusion matrix for the drift classifier."""
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
