"""
injection.py
Controlled fault-injection functions that generate ground truth for the three
detection modules of the DQ monitoring framework.

Each function takes a clean reference DataFrame, makes a copy, corrupts the copy
in a controlled and logged way, and returns exactly two things: the corrupted
DataFrame and a record of what was changed. The original DataFrame is never
modified in place.

Dataset column reference (IBM HI-Small_Trans.csv, verified 27 July 2026):
    Timestamp, From Bank, Account, To Bank, Account.1,
    Amount Received, Receiving Currency, Amount Paid, Payment Currency,
    Payment Format, Is Laundering
Numeric columns suitable for anomaly/missing injection: 'Amount Paid',
'Amount Received'. Label column: 'Is Laundering'.

Author: Neha Pandit
"""

import numpy as np
import pandas as pd


def inject_anomalies(df, column, rate=0.02, multiplier=50, seed=42):
   
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_anomalies = int(len(df) * rate)
    anomaly_idx = rng.choice(df.index, size=n_anomalies, replace=False)

    ground_truth = pd.Series(False, index=df.index)
    ground_truth.loc[anomaly_idx] = True

    df.loc[anomaly_idx, column] = df.loc[anomaly_idx, column] * multiplier
    return df, ground_truth


def inject_schema_drift(df, drop_column=None, rename_map=None, dtype_change=None):
   
    df = df.copy()
    change_log = {}

    if drop_column:
        df = df.drop(columns=[drop_column])
        change_log["dropped_column"] = drop_column

    if rename_map:
        df = df.rename(columns=rename_map)
        change_log["renamed_columns"] = rename_map

    if dtype_change:
        col, new_type = dtype_change
        df[col] = df[col].astype(new_type)
        change_log["dtype_changed"] = {col: new_type}

    return df, change_log


def inject_missing_values(df, column, rate=0.10, seed=42):
  
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    n_missing = int(n_rows * rate)
    missing_idx = rng.choice(df.index, size=n_missing, replace=False)

    ground_truth = pd.Series(False, index=df.index)
    ground_truth.loc[missing_idx] = True

    df.loc[missing_idx, column] = np.nan
    return df, ground_truth
