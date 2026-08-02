"""
module3_missing.py
Module 3 — missing-value prediction (early warning).

Predicts WHETHER a value will be missing in a target column, using the other columns
as features. This is different from imputation (DataWig), which predicts what the value
should be. One XGBoost classifier is trained per target column.

Design note on the missingness mechanism (important — ties to Rubin, 1976):
- The project's default injection is missing-completely-at-random (MCAR). MCAR missingness
  is, by definition, independent of the other variables, so it CANNOT be predicted — a
  classifier scores ~0.5. This is a correct theoretical result, not a failure.
- Real pipeline missingness is usually missing-at-random (MAR) or not-at-random (MNAR):
  it depends on other observed values. inject_missing_values_mar below makes missingness
  depend on the transaction amount, which is realistic and learnable. Module 3 is evaluated
  primarily on MAR, with MCAR shown as a contrast.

Class imbalance is handled with scale_pos_weight = (non-missing count) / (missing count).

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

SEED = 42

NUMERIC_COLS = ["Amount Paid", "Amount Received"]
CATEGORICAL_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]
TIME_COL = "Timestamp"


# ======================================================================
# MAR missing-value injection (learnable) — ground truth for Module 3
# ======================================================================
def inject_missing_values_mar(df, target_column, rate=0.10, driver="Amount Paid",
                              power=2.0, seed=42):
    """
    Inject MISSING-AT-RANDOM values: the probability that `target_column` is missing
    depends on the `driver` column (larger amounts are more likely to be missing).
    This creates a pattern a classifier can learn from the other columns.

    `power` controls how strongly missingness depends on the driver: the selection
    weight is proportional to (amount rank)**power. power=1 is a gentle dependence,
    power=2 (default) a moderate, realistic one; higher makes missingness concentrate
    in the largest transactions. Chosen calibration: power=2 gives ROC-AUC ~0.76 on
    this data, clearly learnable while remaining realistic.

    Returns (corrupted_df, ground_truth) where ground_truth is True/False per row.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_missing = int(len(df) * rate)

    ranks = df[driver].rank(method="first").values
    weights = ranks ** power
    weights = weights / weights.sum()
    missing_idx = rng.choice(df.index, size=n_missing, replace=False, p=weights)

    ground_truth = pd.Series(False, index=df.index)
    ground_truth.loc[missing_idx] = True
    df.loc[missing_idx, target_column] = np.nan
    return df, ground_truth


# ======================================================================
# Feature building (all columns except the target)
# ======================================================================
def build_module3_features(df, target_col):
    """
    Build a numeric feature matrix from every column EXCEPT the target column.
    log-transformed amounts + time features + one-hot encoded categoricals (excluding
    the target). Returns a DataFrame X aligned to df.index.
    """
    feats = pd.DataFrame(index=df.index)

    # numeric amounts (log), excluding the target if it is numeric
    for col in NUMERIC_COLS:
        if col != target_col and col in df.columns:
            feats[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # time features
    if TIME_COL in df.columns:
        ts = pd.to_datetime(df[TIME_COL], errors="coerce")
        feats["hour"] = ts.dt.hour
        feats["dayofweek"] = ts.dt.dayofweek
        feats["day"] = ts.dt.day

    # one-hot categoricals, excluding the target
    cat_cols = [c for c in CATEGORICAL_COLS if c != target_col and c in df.columns]
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], dummy_na=False)
        feats = pd.concat([feats, dummies], axis=1)

    return feats


# ======================================================================
# Train / evaluate one per-column classifier
# ======================================================================
def train_missing_classifier(X, y):
    """
    Train an XGBoost classifier to predict missingness, with class-imbalance handling
    via scale_pos_weight = non-missing / missing.
    """
    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    n_neg = int(len(y) - y.sum())
    spw = n_neg / n_pos
    clf = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def evaluate_missing(clf, X, y, threshold=0.5):
    """Precision, Recall, F1, ROC-AUC, confusion matrix for one target column."""
    y = np.asarray(y).astype(int)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def run_module3(train_df, test_df, target_cols, mechanism="mar", rate=0.10, seed=42):
    """
    Full Module 3 run: for each target column, inject missingness (MAR or MCAR) into the
    train and test batches, train an XGBoost classifier, and evaluate on the test batch.

    Returns a dict {target_col: metrics}. `mechanism` is 'mar' (learnable) or 'mcar'
    (random, unpredictable — used as a contrast).
    """
    from injection import inject_missing_values  # MCAR
    results = {}
    for col in target_cols:
        if mechanism == "mar":
            tr_c, tr_gt = inject_missing_values_mar(train_df, col, rate=rate, seed=seed)
            te_c, te_gt = inject_missing_values_mar(test_df, col, rate=rate, seed=seed + 1)
        else:
            tr_c, tr_gt = inject_missing_values(train_df, col, rate=rate, seed=seed)
            te_c, te_gt = inject_missing_values(test_df, col, rate=rate, seed=seed + 1)
        Xtr = build_module3_features(tr_c, col)
        Xte = build_module3_features(te_c, col)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)  # align one-hot columns
        clf = train_missing_classifier(Xtr, tr_gt.values)
        results[col] = evaluate_missing(clf, Xte, te_gt.values)
    return results
