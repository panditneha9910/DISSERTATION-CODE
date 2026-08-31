"""
framework.py
Single entry point for the ML-based data-quality monitoring framework.

This is the developer/tester-facing orchestrator. Instead of assembling the three
modules by hand in a notebook, a caller does:

    import framework as fw
    pipeline, summary, stats = fw.run_pipeline('../data/HI-Small_Trans.csv', fw.IBM_CONFIG)

or, to build the framework and assess batches manually:

    reference, incoming = ...            
    dq = fw.build_framework(reference, fw.IBM_CONFIG)
    report = dq.assess(incoming_batch)   # -> combined score + PASS/FAIL

Design choices (see PROJECT_CONTEXT 4e/4f):
- The framework is built in the ABLATION-VALIDATED configuration by default:
  max-gate combination + severity-scaled anomaly signal + observed missing-rate as the
  gate signal. The Module 3 ML model is still fitted and reported as a forward-looking
  'missing_risk_warning', but the gate decision uses the directly observed null rate so
  all three dimensions are comparable severities.
- run_pipeline streams the ENTIRE incoming period in batches (not a sample), so the
  framework is exercised on the whole dataset.
- LOF is deliberately NOT part of the runtime gate- the gate's anomaly signal is a fast,
  scalable log-amount z-score. LOF is near-quadratic and is used only in the Module 1
  offline comparison (notebook 03).

The same code runs on both datasets; only the column config differs.

Author: Neha Pandit 
"""

import numpy as np
import pandas as pd

import preprocessing as pp
import module2_drift as m2
import module3_missing as m3
import integration as ig



IBM_CONFIG = dict(
    name="IBM HI-Small",
    amount_col="Amount Paid",
    time_col="Timestamp",
    m2_numeric=["Amount Paid", "Amount Received"],
    m2_categorical=["Receiving Currency", "Payment Currency", "Payment Format"],
    drift_col="Amount Paid",
    m3_target="Receiving Currency",
    m3_driver="Amount Paid",
    m3_feature_kwargs={},                       # IBM defaults inside module3
    missing_observed_col="Receiving Currency",  # gate signal = observed null-rate here
)

CC_CONFIG = dict(
    name="Credit Card Fraud",
    amount_col="Amount",
    time_col=None,                              # 'Time' is seconds-from-start, not a stamp
    m2_numeric=["Amount"] + [f"V{i}" for i in range(1, 7)],
    m2_categorical=[],
    drift_col="Amount",
    m3_target="Amount",
    m3_driver="Amount",
    m3_feature_kwargs=dict(log_cols=["Amount"], passthrough_cols=[f"V{i}" for i in range(1, 29)],
                           categorical_cols=[], time_col=None),
    missing_observed_col="V1",
)


# ======================================================================
# Build a ready-to-use framework from a clean reference
# ======================================================================
def build_framework(reference_df, config, combine_mode="max", threshold=0.5,
                    z_thresh=5.0, anomaly_sig=0.01, missing_sig=0.05,
                    drift_rates=(0.05, 0.10), ks_sample=3000,
                    fit_sample=200000, mode="general", seed=42):
    """
    Fit all three modules on the clean reference and return a configured DQFramework.

    By default the framework is returned in the ablation-validated gate configuration
    (max combine, severity-scaled anomaly, observed missing-rate gate). Pass
    combine_mode='weighted' / anomaly_sig=None to reproduce the original averaging behaviour.

    `fit_sample` caps the reference used to TRAIN the missing-risk model (a sample of a few
    hundred thousand rows characterises the reference well and keeps fitting fast on a
    multi-million-row reference); set None to fit on the entire reference. The incoming
    stream is always processed in full by run_pipeline regardless of this setting.
    """
    cfg = config

    # Reference sample for the drift KS test. IMPORTANT: the KS test's power grows with
    # sample size, so on 50k-row batches even a trivial distribution difference reads as
    # drift (p ~ 0). We therefore fix a MODERATE sample size `ks_sample` for the drift
    # comparison in BOTH training and runtime, so only material effect sizes register.
    drift_ref = reference_df.sample(min(ks_sample, len(reference_df)),
                                    random_state=seed).reset_index(drop=True)

    def _subsample(d):
        return d.sample(ks_sample, random_state=seed).reset_index(drop=True) if len(d) > ks_sample else d

    def drift_feature_fn(r, b):
        return m2.drift_feature_vector(_subsample(r), _subsample(b),
                                       cfg["m2_numeric"], cfg["m2_categorical"])

    # reference used to fit the missing-risk model (capped for speed on huge references)
    fit_ref = (reference_df if fit_sample is None or len(reference_df) <= fit_sample
               else reference_df.sample(fit_sample, random_state=seed).reset_index(drop=True))

    # --- Module 2: drift classifier (trained on injected shift, KS at the fixed sample) ---
    Xd, yd = m2.build_drift_dataset(
        drift_ref, reference_df, rates=list(drift_rates),
        batch_size=ks_sample, n_per_rate=20,
        numeric_cols=cfg["m2_numeric"], categorical_cols=cfg["m2_categorical"],
        drift_col=cfg["drift_col"], seed=seed)
    drift_clf = m2.train_drift_classifier(Xd, yd)
    # calibrate the "normal variation" drift level: proba the classifier gives to CLEAN
    # same-period batches. The 95th percentile is the upper edge of normal drift; anything
    # materially above it is real drift. This de-sensitises the gate to mild variation.
    clean_proba = drift_clf.predict_proba(Xd[yd == 0])[:, 1]
    drift_baseline = float(np.quantile(clean_proba, 0.95)) if len(clean_proba) else None

    # --- Module 3: missing-risk model (early-warning; not the gate signal) ---
    mc, mg = m3.inject_missing_values_mar(fit_ref, cfg["m3_target"], rate=0.10,
                                          driver=cfg["m3_driver"], seed=seed)
    Xm = m3.build_module3_features(mc, cfg["m3_target"], **cfg["m3_feature_kwargs"])
    miss_clf = m3.train_missing_classifier(Xm, mg.values)

    # --- Module 1: reference log-amount statistics for the z-score gate signal ---
    logamt = np.log1p(reference_df[cfg["amount_col"]].clip(lower=0))

    fitted = {
        "ref_log_mean": float(logamt.mean()), "ref_log_std": float(logamt.std()),
        "drift_clf": drift_clf,
        "drift_feature_fn": drift_feature_fn,
        "missing_clf": miss_clf,
        "missing_feature_fn": (lambda b, t: m3.build_module3_features(
            b, t, **cfg["m3_feature_kwargs"])),
        "missing_target": cfg["m3_target"], "missing_columns": list(Xm.columns),
    }

    return ig.DQFramework(
        drift_ref, fitted, mode=mode, threshold=threshold,
        amount_col=cfg["amount_col"], combine_mode=combine_mode,
        z_thresh=z_thresh, anomaly_sig=anomaly_sig,
        missing_observed_col=cfg["missing_observed_col"], missing_sig=missing_sig,
        drift_baseline=drift_baseline)


# ======================================================================
# Whole dataset run: stream the ENTIRE incoming period in batches
# ======================================================================
def run_pipeline(data_path, config, batch_size=50000, reference_days=3, max_days=10,
                 nrows=None, max_batches=None, verbose=True,
                 drift_reference="rolling", rolling_window=None, **build_kwargs):
    """
    Load -> split -> build -> stream every incoming batch through the framework.

    drift_reference controls what each batch's DRIFT is measured against:
      'rolling' (default) — the recent preceding window (last `rolling_window` rows, default
                 = batch_size). This is what a real monitor does: it flags a batch that
                 differs from RECENT history, so gradual temporal evolution does not fire
                 the alarm and only abrupt drift fails. Fixes the fixed-reference failure
                 mode where every later-period batch looks drifted versus a distant origin.
      'fixed'  — the built day-1..reference_days reference (the naive baseline; kept so the
                 contrast can be shown).
    Anomaly and missing gates always use the fixed reference statistics — only the drift
    comparison rolls.

    Returns (pipeline, summary_df, stats). The incoming stream is processed in full (all
    rows, in `batch_size` chunks). `nrows` / `max_batches` are for quick dev runs only.
    """
    import time
    t0 = time.time()
    rolling_window = rolling_window or batch_size

    df = pp.load_transactions(data_path, nrows=nrows)
    if config.get("time_col") and config["time_col"] in df.columns:
        reference, incoming = pp.temporal_split(df, reference_days=reference_days,
                                                max_days=max_days, time_col=config["time_col"])
    else:
        k = int(len(df) * 0.4)
        reference = df.iloc[:k].reset_index(drop=True)
        incoming = df.iloc[k:].reset_index(drop=True)

    if verbose:
        print("[%s] reference=%d  incoming=%d  batch_size=%d  drift_reference=%s"
              % (config.get("name", "?"), len(reference), len(incoming), batch_size, drift_reference))

    pipeline = build_framework(reference, config, **build_kwargs)
    fixed_reference = pipeline.reference     # the built reference sample (day 1..k)
    # rolling buffer seeded with the END of the reference period, so batch 0 is compared to
    # the most recent history rather than a cold start.
    buffer = fixed_reference.tail(rolling_window).reset_index(drop=True)

    rows = []
    n = len(incoming)
    for bi, start in enumerate(range(0, n, batch_size)):
        batch = incoming.iloc[start:start + batch_size]
        pipeline.reference = buffer if drift_reference == "rolling" else fixed_reference
        rep = pipeline.assess(batch)
        rows.append({"batch": bi, "rows": len(batch),
                     **{k: round(v, 4) for k, v in rep["module_scores"].items()},
                     "combined": round(rep["combined_score"], 4),
                     "decision": rep["decision"],
                     "missing_risk": (round(rep["missing_risk_warning"], 4)
                                      if rep["missing_risk_warning"] is not None else None)})
        # update the rolling reference only with batches that PASSED, so a bad batch does
        # not become the baseline (which would false-fail the next, normal batch).
        if drift_reference == "rolling" and rep["decision"] == "PASS":
            buffer = (pd.concat([buffer, batch], ignore_index=True)
                      .tail(rolling_window).reset_index(drop=True))
        if verbose and (bi % 10 == 0):
            print("  batch %3d/%d  rows=%d  combined=%.3f  %s"
                  % (bi, (n + batch_size - 1) // batch_size, len(batch),
                     rep["combined_score"], rep["decision"]))
        if max_batches and bi + 1 >= max_batches:
            break

    pipeline.reference = fixed_reference     # restore for any later manual use
    summary = pd.DataFrame(rows)
    stats = dict(reference_rows=len(reference), incoming_rows=len(incoming),
                 drift_reference=drift_reference,
                 batches=len(summary), fail_batches=int((summary["decision"] == "FAIL").sum()),
                 time_s=round(time.time() - t0, 1))
    if verbose:
        print("[done] %d batches, %d FAIL, %.1fs"
              % (stats["batches"], stats["fail_batches"], stats["time_s"]))
    return pipeline, summary, stats
