"""
preprocessing.py
Data loading, temporal reference/incoming split, and feature construction for the
DQ monitoring framework.

Design decisions (grounded in EDA of HI-Small_Trans.csv, 27 July 2026):
- Amounts are extremely heavy-tailed (skew ~219, max ~1.4e11), so the amount is
  log-transformed. This stabilises the scale for density- and reconstruction-based
  detectors and turns a multiplicative anomaly into a consistent additive shift.
- Account / Account.1 are near-unique identifiers (~298k distinct in 500k rows),
  so they are dropped from the model feature set rather than encoded.
- Reference vs incoming is split temporally: the earliest fraction of transactions
  is treated as the clean reference; later transactions form the incoming batches.
  This mirrors a pipeline that monitors new data against history.
- Transformers (the scaler) are fit on the reference batch only, then applied to
  incoming batches, to avoid leakage from corrupted incoming data.

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Column roles for the IBM HI-Small_Trans.csv schema
AMOUNT_COL = "Amount Paid"
TIME_COL = "Timestamp"
LABEL_COL = "Is Laundering"
LOW_CARD_CATS = ["Receiving Currency", "Payment Currency", "Payment Format"]
ID_COLS = ["Account", "Account.1"]  # near-unique, dropped from features


def load_transactions(path, nrows=None):
    """Load the transactions CSV. Set nrows for a fast development sample."""
    return pd.read_csv(path, nrows=nrows)


def temporal_split(df, reference_days=3, max_days=10, time_col=TIME_COL):
    """
    Split into a clean reference batch (transactions in the first
    `reference_days` calendar days) and an incoming batch (the remaining days).

    Splitting by day, not by row fraction, matters for this dataset: daily
    volumes are very uneven (day 1 alone holds >1M rows), so a row-fraction
    split would give the reference a time span of only minutes and collapse the
    variance of the time features. A day-based split gives the reference a real
    multi-day span.

    `max_days` drops the negligible long tail (the dataset has a few hundred
    stray transactions on days 11-18); set to None to keep everything.

    Returns
    -------
    reference_df, incoming_df : pandas.DataFrame
        Both sorted by time, with a fresh RangeIndex.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    start_day = df[time_col].dt.normalize().min()
    ref_cutoff = start_day + pd.Timedelta(days=reference_days)
    reference_df = df[df[time_col] < ref_cutoff].reset_index(drop=True)

    incoming = df[df[time_col] >= ref_cutoff]
    if max_days is not None:
        end_cutoff = start_day + pd.Timedelta(days=max_days)
        incoming = incoming[incoming[time_col] < end_cutoff]
    incoming_df = incoming.reset_index(drop=True)
    return reference_df, incoming_df


def add_time_features(df, time_col=TIME_COL):
    """Derive numeric features from the timestamp: hour, day of week, day."""
    df = df.copy()
    ts = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["day"] = ts.dt.day
    return df


def build_module1_features(df, scaler=None, fit=False, amount_col=AMOUNT_COL):
    """
    Build the numeric feature matrix for Module 1 (anomaly detection):
    log1p(amount) plus time features, standardised.

    Fit the scaler on the reference batch (fit=True); pass the returned scaler
    back in (fit=False) to transform incoming batches on the same scale.

    Parameters
    ----------
    df : pandas.DataFrame
        A batch (reference or incoming). May contain injected anomalies.
    scaler : sklearn StandardScaler or None
        Fitted scaler to reuse. Ignored when fit=True.
    fit : bool
        If True, fit a new StandardScaler on this batch.

    Returns
    -------
    X : numpy.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        The fitted scaler (fit it on the reference batch, reuse on incoming).
    feature_names : list of str
        Column order of X.
    """
    df = add_time_features(df)
    feats = pd.DataFrame(index=df.index)
    feats["log_amount"] = np.log1p(df[amount_col].clip(lower=0))
    feats["hour"] = df["hour"].astype(float)
    feats["dayofweek"] = df["dayofweek"].astype(float)
    feats["day"] = df["day"].astype(float)

    if fit or scaler is None:
        scaler = StandardScaler().fit(feats.values)
    X = scaler.transform(feats.values)
    return X, scaler, list(feats.columns)
