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

Author: Neha Pandit | MSc Data Science | University of Surrey
"""

import numpy as np
import pandas as pd


def inject_anomalies(df, column, rate=0.02, multiplier=50, seed=42):
    """
    Inject value-level anomalies by multiplying a chosen numeric column by a
    large factor on a random subset of rows. Feeds Module 1 ground truth.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean reference data. Not modified in place.
    column : str
        Name of the numeric column to corrupt (e.g. 'Amount Paid').
    rate : float, default 0.02
        Fraction of rows to turn into anomalies.
    multiplier : float, default 50
        Factor applied to the chosen column on the selected rows.
    seed : int, default 42
        Seed for the random generator, for reproducibility.

    Returns
    -------
    corrupted_df : pandas.DataFrame
        Copy of df with anomalies injected into `column`.
    ground_truth : pandas.Series
        Boolean Series aligned to df.index. True where a row was corrupted.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_anomalies = int(len(df) * rate)
    anomaly_idx = rng.choice(df.index, size=n_anomalies, replace=False)

    ground_truth = pd.Series(False, index=df.index)
    ground_truth.loc[anomaly_idx] = True

    df.loc[anomaly_idx, column] = df.loc[anomaly_idx, column] * multiplier
    return df, ground_truth


def inject_schema_drift(df, drop_column=None, rename_map=None, dtype_change=None):
    """
    Inject structural schema drift by dropping, renaming, or retyping columns.
    Call separately for each drift type being tested. Feeds Module 2 ground truth.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean reference data. Not modified in place.
    drop_column : str or None
        Column name to drop, or None.
    rename_map : dict or None
        Mapping {old_name: new_name}, or None.
    dtype_change : tuple or None
        (column_name, new_type) to cast a column, or None.

    Returns
    -------
    drifted_df : pandas.DataFrame
        Copy of df with the requested structural change applied.
    change_log : dict
        Record of what changed at batch level.
    """
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
    """
    Inject missing values by setting a random subset of rows in a chosen column
    to NaN. Feeds Module 3 ground truth. Call once per target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean reference data. Not modified in place.
    column : str
        Name of the column in which to insert missing values.
    rate : float, default 0.10
        Fraction of rows to set to NaN.
    seed : int, default 42
        Seed for the random generator, for reproducibility.

    Returns
    -------
    corrupted_df : pandas.DataFrame
        Copy of df with NaNs inserted into `column`.
    ground_truth : pandas.Series
        Boolean Series aligned to df.index. True where a value was removed.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    n_missing = int(n_rows * rate)
    missing_idx = rng.choice(df.index, size=n_missing, replace=False)

    ground_truth = pd.Series(False, index=df.index)
    ground_truth.loc[missing_idx] = True

    df.loc[missing_idx, column] = np.nan
    return df, ground_truth
