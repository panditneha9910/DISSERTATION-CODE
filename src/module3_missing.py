"""
module3_missing.py
Module 3 — missing-value prediction.

Predicts WHETHER a value will be missing in a target column, using the other columns
as features. This is different from imputation (DataWig), which predicts what the value
should be. One XGBoost classifier is trained per target column.

Design note on the missing mechanism:
- The project's default injection is missing-completely-at-random (MCAR). MCAR missingness
  is, by definition, independent of the other variables, so it CANNOT be predicted — a
  classifier scores ~0.5. This is a correct theoretical result, not a failure.
- Real pipeline missingness is usually missing-at-random (MAR) or not-at-random (MNAR):
  it depends on other observed values. inject_missing_values_mar below makes missingness
  depend on the transaction amount, which is realistic and learnable. Module 3 is evaluated
  primarily on MAR, with MCAR shown as a contrast.

Class imbalance is handled with scale_pos_weight = (non-missing count) / (missing count).

Author: Neha Pandit 
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
def build_module3_features(df, target_col, log_cols=None, passthrough_cols=None,
                           categorical_cols=None, time_col=TIME_COL):
    
    log_cols = NUMERIC_COLS if log_cols is None else log_cols
    passthrough_cols = [] if passthrough_cols is None else passthrough_cols
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols

    feats = pd.DataFrame(index=df.index)

    # amount-like numeric columns (log), excluding the target if it is one of them
    for col in log_cols:
        if col != target_col and col in df.columns:
            feats[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # already-numeric columns used as-is, excluding the target
    for col in passthrough_cols:
        if col != target_col and col in df.columns:
            feats[col] = df[col].astype(float)

    # time features (only if a real timestamp column is supplied)
    if time_col and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        feats["hour"] = ts.dt.hour
        feats["dayofweek"] = ts.dt.dayofweek
        feats["day"] = ts.dt.day

    # one-hot categoricals, excluding the target
    cat_cols = [c for c in categorical_cols if c != target_col and c in df.columns]
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


def run_module3(train_df, test_df, target_cols, mechanism="mar", rate=0.10, seed=42,
                driver="Amount Paid", feature_kwargs=None):
    
    from injection import inject_missing_values  # MCAR
    feature_kwargs = feature_kwargs or {}
    results = {}
    for col in target_cols:
        if mechanism == "mar":
            tr_c, tr_gt = inject_missing_values_mar(train_df, col, rate=rate,
                                                    driver=driver, seed=seed)
            te_c, te_gt = inject_missing_values_mar(test_df, col, rate=rate,
                                                    driver=driver, seed=seed + 1)
        else:
            tr_c, tr_gt = inject_missing_values(train_df, col, rate=rate, seed=seed)
            te_c, te_gt = inject_missing_values(test_df, col, rate=rate, seed=seed + 1)
        Xtr = build_module3_features(tr_c, col, **feature_kwargs)
        Xte = build_module3_features(te_c, col, **feature_kwargs)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)  # align one-hot columns
        clf = train_missing_classifier(Xtr, tr_gt.values)
        results[col] = evaluate_missing(clf, Xte, te_gt.values)
    return results
