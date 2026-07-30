# PROJECT CONTEXT — DQ Monitoring Framework (implementation handoff)
# Neha Pandit | MSc Data Science, University of Surrey | Submission: 4pm, 1 September 2026
#
# PURPOSE OF THIS FILE
# This is the single source of truth for the implementation. If you open a fresh
# Claude (e.g. on the website inside the HPC) or hand this to anyone, point them
# here first. It captures every decision, every data fact, and the build plan, so
# no context is trapped in a chat. Keep it in the repo root and update it as you go.

---

## 1. WHAT WE ARE BUILDING (one paragraph)

An automated quality gate that sits between development and QA in a financial data
pipeline. A batch of incoming data is compared against a clean reference batch and
passed through three ML-based checks that run in parallel: value-level anomaly
detection (Module 1), schema and distribution drift detection (Module 2), and
missing-value prediction (Module 3). Their outputs are combined by an integration
layer into one score in [0,1]; a batch scoring at or above 0.5 fails and returns to
the developer, below 0.5 passes to QA. An ablation study measures each module's
contribution.

Entry point: `DQFramework(batch_data, reference_data, mode="general", threshold=0.5)`

---

## 2. DATASETS (verified from the actual files, 27 July 2026)

### Primary — IBM HI-Small_Trans.csv
- Location on Mac: `DISSERTATION/DATASET/archive (2)/HI-Small_Trans.csv`
- Rows: 5,078,345. Size: 454 MB.
- REAL columns (the project file's assumed schema was wrong — use these):
  `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
   Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`
  (pandas auto-renames the second `Account` to `Account.1`.)
- Numeric anomaly/missing target: `Amount Paid` (or `Amount Received`).
- Label: `Is Laundering` (real fraud, ~0.1% positive).

### Secondary — creditcard.csv (ULB)
- Location: `DISSERTATION/DATASET/Secondary -Credit Card Fraud/creditcard.csv`
- Rows: 284,807. Columns: `Time, V1..V28, Amount, Class`. Label: `Class`.
- Scope: Module 1 full; Module 2 statistical drift only (V1-V28 are PCA, no business
  meaning, so no structural drift check); Module 3 full.

---

## 3. DATA FACTS DISCOVERED (these correct or extend the project file)

1. NO natural missing values in the IBM data → Module 3 relies entirely on injected
   missingness. (Confirms project design.)
2. `Amount Paid` is extremely heavy-tailed: skew ~219, median ~2,130, max ~1.4e11 →
   log-transform is essential.
3. `Amount Paid` == `Amount Received` for ~99.4% of rows → use one amount column.
4. `Account` / `Account.1` are near-unique (~298k distinct in 500k) → DROP as features.
5. Low-cardinality categoricals: Receiving/Payment Currency (15), Payment Format (7).
6. Class imbalance: laundering ~0.1%.
7. TIME SPAN CORRECTION: project file says 10 days. The file actually runs 1-18 Sep,
   but days 11-18 are a negligible tail (a few hundred rows total). Days 1-10 carry
   the volume (200k-1.1M/day). The pipeline drops the tail (`max_days=10`).
8. DEV TRAP: reading a head-sample (`pd.read_csv(nrows=N)`) returns almost only day 1,
   because daily volumes are uneven. Anything involving the temporal split MUST use
   the full file (fine on the HPC).

---

## 4. FINALISED PREPROCESSING DECISIONS (grounded in EDA, in src/preprocessing.py)

- Reference vs incoming split = TEMPORAL by calendar day: days 1-3 = clean reference
  (~2.08M rows), days 4-10 = incoming batches (~3.0M rows). Mirrors real monitoring.
- Module 1 features = `log1p(Amount Paid)` + hour, day-of-week, day; standardised.
- No leakage: the StandardScaler is fit on the REFERENCE batch only, then applied to
  incoming batches. Same pattern for any future transformer.
- Fixed random seed = 42 everywhere for reproducibility.

---

## 4b. ANOMALY INJECTION CALIBRATION (decided 27 Jul 2026 — for methodology chapter)

The project file's default anomaly injection (rate 2%, multiplier x50) was tested on the
real data and found INSUFFICIENT: amounts are so heavy-tailed (reference 99.9th percentile
~592M, max ~24bn) that a x50 anomaly typically lands below the 88th percentile of natural
amounts — it is not an outlier. Injecting 2% also makes anomalies cluster, which breaks LOF.

DECISION: use a RARE + EXTREME injection for Module 1 — rate 0.5%, multiplier 1000 — so
anomalies are genuine outliers without clustering. Report ROC-AUC and PR-AUC, not a fixed
0.5 threshold. Document this calibration in the methodology as a data-driven adjustment.

Verified Module 1 results at this setting (100k reference / 100k incoming sample):
  Z-score ROC-AUC 0.88 (best) · Isolation Forest 0.81 · IQR 0.59 · LOF 0.18 (fails).
The LOF failure is a genuine finding, not a bug: LOF is a local detector and the anomalies
are global extremes. It is direct evidence for the multi-detector design (Xu et al. 2023;
Han et al. 2022) — report it, do not hide it. PR-AUC is low across detectors because natural
extremes compete with injected anomalies; this is an honest limitation of the data.

## 4c. MODULE 2 FINDINGS (28 Jul 2026 — for methodology/results)

All three steps work. Verified on real data:
- Step 1 (structural): detects dropped columns and dtype changes exactly (deterministic).
- Step 2 (statistical): KS flags shifted numeric columns, Chi-squared flags categorical
  shifts, and unchanged columns are correctly left alone.
- Step 3 (RF drift classifier): F1 ~0.99, ROC-AUC ~1.00 on the controlled experiment
  (train drift at 5/10%, evaluate at 20/30%, different seeds).

IMPORTANT design finding: there is NATURAL temporal drift between the reference period
(days 1-3) and later days (e.g. Payment Format mix changes over the 10 days). So the RF's
"clean" batches must be sampled from the SAME period as the reference (days 1-3), otherwise
natural drift makes clean batches look drifted and the classifier fails (ROC-AUC ~0.5).
Report this: (a) it is why the controlled experiment uses same-period clean batches;
(b) the framework does detect genuine temporal drift, which is a positive, not a bug.
Honest caveat: the ~0.99 F1 is on synthetic injected drift (circularity limitation);
it shows the pipeline detects the injected drift types, not that drift is "solved" in the wild.

## 5. BUILD ORDER (strict — do not skip)

- [DONE] Stage 0: environment + data load check.
- [DONE] Stage 1: injection functions (src/injection.py) — all three tested.
- [DONE] Preprocessing (src/preprocessing.py) + EDA (notebook 02) — validated.
- [NEXT] Stage 2: Module 1 — Isolation Forest, then LOF, then Z-score, then IQR,
  then Autoencoder (PyTorch, LAST; drop if behind schedule).
- Stage 3: Module 2 — structural check, then KS + Chi-squared, then RF classifier.
  Train the RF on injected drift at 5% and 10%; evaluate at 20% and 30% with
  DIFFERENT seeds. Test four drift types separately (drop, rename, dtype, distribution).
- Stage 4: Module 3 — one XGBoost per target column; scale_pos_weight =
  non-missing / missing; evaluate per column then average.
- Stage 5: Integration layer — Layer 1 purpose profiles, then Layer 2 confidence
  weighting; threshold 0.5; normalise all module scores to [0,1] first.
- Stage 6: Ablation (A1-A4) + full evaluation + all plots.

Minimum viable if time runs short: iForest + LOF + Z/IQR for M1; steps 1-2 of M2;
XGBoost for M3; equal-weighted integration; all four ablation experiments.

---

## 6. INTEGRATION LAYER SPEC (for Stage 5)

Layer 1 baseline weights by mode (all sum to 1.0):
- general (default): drift 0.5, anomaly 0.3, missing 0.2
- fraud: anomaly 0.5, drift 0.3, missing 0.2
- compliance: missing 0.5, drift 0.3, anomaly 0.2

Layer 2 (confidence-based adaptive, applied per batch; grounded in Almarshad et al. 2025):
    confidence_i   = abs(score_i - 0.5)
    raw_weight_i   = baseline_i * (0.5 + confidence_i)
    final_weight_i = raw_weight_i / sum(all raw_weights)
    combined_score = sum(final_weight_i * score_i)
All module scores MUST be normalised to [0,1] before this. Threshold 0.5: pass if
combined < 0.5, fail if >= 0.5.

---

## 7. EVALUATION / ABLATION (Stage 6)

- Per module: Precision, Recall, F1, ROC-AUC (reported individually) + confusion matrix.
- Ablation: A1 remove M1; A2 remove M2; A3 remove M3; A4 equal vs severity vs adaptive
  weights. Report F1 change per experiment as % change from the full framework.
- Plots: confusion matrix per module; ROC curves (all M1 methods on one plot); RF and
  XGBoost feature importances; ablation bar chart.

---

## 8. CODING CONVENTIONS

- seed = 42 everywhere.
- Never modify the original DataFrame in place (functions .copy() first).
- Fit transformers/models on the reference (clean) batch; apply to incoming.
- Clean, commented Python; reusable functions in src/, runnable notebooks in notebooks/.
- Do not commit the datasets to Git (they are large); see .gitignore.

---

## 9. REPO / HPC WORKFLOW

- Files created on the Mac live in `DISSERTATION/CODE/`.
- Sync to the Surrey HPC via Git: push from Mac, pull on HPC, run in VS Code.
- Datasets are NOT in Git — copy them to the HPC once (scp/rsync) or point paths at
  their HPC location. Set `DATA_PATH` in each notebook accordingly.

---

## 10. KNOWN LIMITATIONS TO STATE HONESTLY (methodology/limitations chapter)

Module 2 circularity (RF trained + evaluated on injected drift); secondary-dataset PCA
(no structural drift check); no streaming evaluation; manual integration weights;
threshold 0.5 not empirically tuned; injection is MCAR only (simplest mechanism);
datasets synthetic/anonymised. Also correct the "10-day span" claim (see 3.7).
