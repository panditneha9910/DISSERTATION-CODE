# Project Context — Data Quality Monitoring Framework

Neha Pandit, MSc Data Science, University of Surrey. Submission: 1 September 2026.

This is my working record for the implementation. It keeps the design decisions, the facts I
found in the data, and the results in one place so the reasoning behind the code is not scattered
across notebooks. I update it as the project moves.

---

## 1. What I am building

An automated quality gate that sits between development and QA in a financial data pipeline. A
batch of incoming data is compared against a clean reference batch and passed through three
ML-based checks that run in parallel: value-level anomaly detection (Module 1), schema and
distribution drift detection (Module 2), and missing-value prediction (Module 3). An integration
layer combines their outputs into a single score in [0,1]. A batch scoring at or above 0.5 fails
and goes back to the developer; below 0.5 it passes to QA. An ablation study then measures how
much each module actually contributes.

The entry point is `DQFramework(batch_data, reference_data, mode="general", threshold=0.5)`.

---

## 2. Datasets (checked against the actual files, 27 July 2026)

### Primary: IBM HI-Small_Trans.csv
- 5,078,345 rows, 454 MB.
- The real columns are `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
  Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`. My original
  plan assumed a different schema, so I use these. (Pandas auto-renames the second `Account` to
  `Account.1`.)
- Numeric target for anomalies and missingness: `Amount Paid` (or `Amount Received`).
- Label: `Is Laundering`, real fraud at roughly 0.1% positive.

### Secondary: creditcard.csv (ULB Credit Card Fraud)
- 284,807 rows. Columns `Time, V1..V28, Amount, Class`, label `Class`.
- Scope: Module 1 in full; Module 2 statistical drift only (V1 to V28 are PCA components with no
  business meaning, so the structural check does not apply); Module 3 in full.

---

## 3. What I found in the data (corrects or extends my original plan)

1. There are no natural missing values in the IBM data, so Module 3 works entirely on injected
   missingness. This matches the design.
2. `Amount Paid` is extremely heavy-tailed (skew around 219, median about 2,130, max about
   1.4e11), so a log transform is essential.
3. `Amount Paid` equals `Amount Received` for about 99.4% of rows, so I use one amount column.
4. `Account` and `Account.1` are close to unique (around 298k distinct in 500k rows), so I drop
   them as features.
5. The low-cardinality categoricals are Receiving and Payment Currency (15 each) and Payment
   Format (7).
6. The classes are imbalanced: laundering is about 0.1%.
7. The time span needed correcting. My plan said 10 days. The file actually spans 1 to 18
   September, but days 11 to 18 are a tiny tail of a few hundred rows. Days 1 to 10 carry the
   volume (200k to 1.1M per day), so the pipeline drops the tail with `max_days=10`.
8. A trap I hit early: reading a head sample with `pd.read_csv(nrows=N)` returns almost only day
   one, because the daily volumes are so uneven. Anything that uses the temporal split has to read
   the full file, which is fine on the HPC.

---

## 4. Preprocessing decisions (from the EDA, in src/preprocessing.py)

- The reference/incoming split is temporal, by calendar day: days 1 to 3 are the clean reference
  (about 2.08M rows), days 4 to 10 are the incoming batches (about 3.0M rows). This mirrors real
  monitoring.
- Module 1 features are `log1p(Amount Paid)` plus hour, day of week, and day, all standardised.
- To avoid leakage, the StandardScaler is fit on the reference batch only and then applied to the
  incoming batches. I use the same pattern for any transformer.
- I fix the random seed at 42 throughout for reproducibility.

---

## 4b. Anomaly injection calibration (27 July 2026, for the methodology chapter)

My original injection settings (2% rate, times 50 multiplier) turned out to be too weak on the
real data. The amounts are so heavy-tailed (the reference 99.9th percentile is around 592M and the
max around 24bn) that a times-50 anomaly usually lands below the 88th percentile of natural
amounts, so it is not really an outlier. Injecting at 2% also makes the anomalies cluster, which
breaks LOF.

So I switched to a rare and extreme injection for Module 1: 0.5% rate, times 1000 multiplier. That
makes the anomalies genuine outliers without clustering. I report ROC-AUC and PR-AUC rather than a
fixed 0.5 threshold, and I document this calibration in the methodology as a data-driven
adjustment.

Module 1 results at this setting (100k reference, 100k incoming sample): Z-score ROC-AUC 0.88
(best), Isolation Forest 0.81, IQR 0.59, LOF 0.18 (fails). The LOF failure is a real finding, not
a bug. LOF is a local density detector and these anomalies are global extremes, which it is blind
to. That is direct evidence for the multi-detector design (Xu et al. 2023; Han et al. 2022), so I
report it rather than hide it. PR-AUC is low across the detectors because the natural extremes
compete with the injected anomalies, which is an honest limitation of this data.

## 4c. Module 2 findings (28 July 2026)

All three steps work on the real data.
- The structural check detects dropped columns and dtype changes exactly, and it is deterministic.
- The statistical check flags shifted numeric columns with the KS test and categorical shifts with
  Chi-squared, and it correctly leaves unchanged columns alone.
- The RF drift classifier reaches F1 around 0.99 and ROC-AUC around 1.00 on the controlled
  experiment (trained on drift at 5% and 10%, evaluated at 20% and 30% with different seeds).

One design finding matters here. There is natural temporal drift between the reference period
(days 1 to 3) and the later days, for example the Payment Format mix changes over the ten days. So
the classifier's "clean" batches have to be sampled from the same period as the reference,
otherwise natural drift makes clean batches look drifted and the classifier collapses to ROC-AUC
around 0.5. I report this two ways: it explains why the controlled experiment uses same-period
clean batches, and it shows the framework genuinely detects real temporal drift, which is a
positive rather than a fault. Honest caveat: the 0.99 F1 is measured on synthetic injected drift,
so it shows the pipeline detects the injected drift types, not that drift is solved in the wild.

## 4d. Module 3 findings (28 July 2026)

A per-column XGBoost predicts whether a value will be missing (an early warning), rather than what
it should be, which is the opposite of imputation tools like DataWig. Class imbalance is handled
with `scale_pos_weight` set to non-missing over missing.

The key decision ties back to Rubin (1976) in my literature review. The default injection is MCAR
(missing completely at random). By definition MCAR is independent of the other columns, so it
cannot be predicted, and the classifier scores ROC-AUC around 0.50. That is a correct theoretical
result, not a failure. Real pipeline missingness is usually MAR or MNAR, where it depends on other
values, so I added `inject_missing_values_mar`, where the chance of a value being missing depends
on the transaction amount (strength = rank to the power, power 2 by default). That makes the task
realistic and learnable. Module 3 is evaluated on MAR, with MCAR shown as a contrast.

Results (80k train, 40k test, targets Receiving Currency and Payment Format):
- MAR: ROC-AUC 0.764, average F1 0.30. Recall is high (about 0.79) because of `scale_pos_weight`,
  and precision is low (about 0.18) because missingness is rare and the signal is only moderate,
  so ROC-AUC is the fair metric here.
- MCAR: ROC-AUC around 0.50, unpredictable, exactly as the theory predicts.

I report both. The MCAR-versus-MAR contrast shows I understand the missingness mechanisms and that
the module works when a real pattern exists.

## 4e. Integration layer (28 July 2026)

`combine_scores()` reproduces my worked example exactly: anomaly 0.53, drift 0.92, missing 0.51 in
general mode gives a combined 0.776 with weights 0.22, 0.64, 0.14. All three purpose profiles sum
to 1.0. The Layer 2 confidence weighting down-weights scores that sit near 0.5.

Each module is reduced to a single [0,1] batch score:
- anomaly is the fraction of batch rows more than 3 SD from the reference on log-amount,
- drift is the Module 2 RF drift probability,
- missing is the mean predicted null-risk from the Module 3 XGBoost.

End-to-end (general mode, threshold 0.5): a clean batch gives a combined 0.098 and passes, a
corrupted (amount-drift) batch gives 0.535 and fails. The purpose profiles change the decision:
the drift-corrupted batch fails under the general profile (which weights drift) but passes under
the fraud profile (which weights anomalies). That is the intended behaviour of configurable
weighting, and a good discussion point. `DQFramework(reference, fitted, mode, threshold).assess(
batch)` returns the full report.

## 4f. Ablation study (28 July 2026, the critical findings for Chapters 5 and 6)

Method: I build labelled batches (clean ones should pass, corrupted ones should fail), each
corrupted with a single fault type (anomaly, drift, or missing), then score the pass/fail decision
against the true label with F1. The decision rule for A1 to A3 is "max", which fails a batch if
any dimension looks bad.

Results (F1):
- A0, full framework (max): 0.99, so the framework works.
- A1, remove Module 1 (anomaly): 0.99, no drop, so Module 1 is redundant with Module 2 here.
- A2, remove Module 2 (drift): 0.80, a drop, so Module 2 contributes.
- A3, remove Module 3 (missing): 0.79, a drop, so Module 3 contributes on its own.
- A4, equal / severity / adaptive averaging: 0.50, so averaging dilutes single-dimension faults.
- A4, max: 0.99, so a fail-if-any-bad gate is the right rule.

The findings I care about, reported honestly:
1. Module 1 is redundant with Module 2 on this dataset. An extreme-value anomaly is also a
   distributional shift, so the drift module catches it too. That is a genuine insight about
   amount-driven financial data.
2. The weighted-averaging integration, including the confidence-adaptive scheme, fails on
   single-dimension faults (F1 0.50), because one clean dimension pulls a real fault below the
   threshold. A max/OR rule works (F1 0.99). This empirically confirms the caveat from my
   literature review that RABEM's confidence weighting suits fusing same-task models, not gating
   across independent quality dimensions.
3. For the write-up: the confidence-weighted layer is useful for interpretable per-dimension
   severity and purpose-based prioritisation, but the final pass/fail gate should use max/OR.

On scales: the ablation uses severity scores in [0,1] (anomaly is flagged-fraction over 0.01,
capped; drift is the RF probability; missing is missing-rate over 0.05, capped), with a z threshold
of 5 for anomaly flagging (the natural max, around 2.4e10, stays under 5 SD on the log scale, so
only injected extremes register). Anomaly corruption uses times 1e9 so the injected anomalies
clearly exceed the natural tail.

## 4g. Secondary dataset cross-validation (28 July 2026, Credit Card Fraud)

This tests whether the framework generalises. Credit Card has real fraud labels (`Class`), so
Module 1 is evaluated against genuine anomalies with no injection. Its amount skew is only about
17, against 858 for IBM.

Results from the full framework running through the same parameterised `src` (notebook 08, 4
August 2026):
- Module 1 against real fraud: Isolation Forest ROC-AUC 0.953, LOF 0.956, Z-score (amount) 0.705,
  Autoencoder 0.956 (PR-AUC 0.521, the highest of the four on this imbalanced set).
- Module 2 statistical (KS, first half against second half): every tested column drifts (V3 KS
  0.51, V1 0.42), which is natural temporal drift.
- Module 2 RF drift classifier (trained on injected shift over the reference period, evaluated at
  held-out 20% and 30% rates): ROC-AUC 1.000, F1 1.000. Caveat: this is near-perfect because an
  injected distribution shift is trivially separable, which is the known circularity limitation. I
  present it as "detects injected drift reliably", not "perfect drift detection".
- Module 3 (MAR missing on Amount, predicted from V1 to V28): ROC-AUC 0.722.
- Integration (general profile): a clean batch gives a combined 0.107 and passes, a corrupted
  batch gives 0.556 and fails. Note that the corrupted batch (times 1e9 on Amount) is caught by
  the drift module (0.94), not the anomaly module (0.021), because the times-1e9 shift moves the
  distribution. The framework catches the fault, but I do not claim the anomaly module caught it.
- Ablation (max scheme): the full framework scores F1 1.000; removing Module 2 or Module 3 drops
  it to 0.80 (each contributes); removing Module 1 leaves it at 1.000 (the anomaly fault overlaps
  with what drift catches here, so anomaly looks redundant on Credit Card too, which is honest).
  The weighting schemes are equal 0.500, severity 0.594, adaptive 0.571, max 1.000, the same
  dilution finding as on IBM.

The generalisation claim now holds: the same `src` modules run end-to-end on both datasets
(Modules 1 to 3, integration, and ablation), driven only by a column config (IBM defaults against
the Credit Card config). Backward compatibility is checked, since the IBM notebooks call the
changed functions with defaults only, and those defaults equal the values that used to be
hardcoded.

The strongest finding for Chapters 5 and 6: LOF fails on IBM (0.18) but works on Credit Card
(0.956). The reason is that IBM anomalies are global amount extremes, which is LOF's weak point,
whereas Credit Card fraud lives in the multi-dimensional PCA space where LOF's local density is
strong. This explains when each detector helps and supports the multi-detector design (Xu et al.
2023; Han et al. 2022). Confirmed limitation: V1 to V28 are anonymous PCA columns, so the Module 2
structural check has no business meaning here and only the statistical check applies.

## 4h. Autoencoder, the fifth Module 1 method (28 July 2026)

`module1_autoencoder.py` is a PyTorch autoencoder. It trains on the clean reference and scores
anomalies by reconstruction error. The bottleneck is sized from the input (`input_dim // 2`) so it
is strictly smaller than the number of features. A bottleneck equal to the input gives no
compression and the model just copies its input to its output, which produces no anomaly signal.
That was a real bug, now fixed.

The finding, which is about dimensionality:
- On IBM (Module 1 has only 4 features), the autoencoder reaches ROC-AUC around 0.50, near random.
  Autoencoders need many correlated features, and four (three near-constant time features plus
  amount) is too few. This is an honest limitation.
- On Credit Card (29 features, V1 to V28 plus log amount, against real fraud), reconstruction-based
  detection reaches ROC-AUC around 0.95, which is strong.

So the autoencoder is not bad, it is dimensionality-dependent: weak on the low-dimensional IBM
feature set, strong on the high-dimensional Credit Card space. That is a clean demonstration of
when autoencoders help, and it is why the autoencoder is the droppable "build last" method for the
IBM analysis. It appears in notebook 03 (IBM, around 0.5) and notebook 08 (Credit Card, around
0.95). It runs on the cluster after `pip install torch`. I did not run it end-to-end in the
development sandbox because torch would not install there, but the reconstruction method is checked
with a PCA proxy on both datasets.

## 4i. Framework orchestrator, scoring fix, and full-scale run (4 August 2026)

`src/framework.py` is the single entry point a developer or tester would use.
- `build_framework(reference, config)` fits all three modules and returns a ready `DQFramework` in
  the configuration the ablation validated (max combine, severity-scaled anomaly, observed
  missing-rate gate). The same code runs on both datasets through `IBM_CONFIG` or `CC_CONFIG`.
- `run_pipeline(data_path, config)` streams the entire incoming period in batches (a whole-dataset
  run) and returns a per-batch pass/fail summary plus throughput.

The scoring fix in `integration.py` is backward compatible, so the old defaults reproduce notebook
06:
- `combine_scores` gains a `combine_mode` of `weighted` (the old behaviour) or `max` (the gate).
  The ablation showed averaging dilutes a single-dimension fault, and max is the right rule.
- `DQFramework` now rescales the anomaly fraction to a severity and uses the observed null-rate as
  the missing gate signal, while the Module 3 model is reported separately as an early-warning
  `missing_risk`. This makes the shipped framework match what the ablation validates. LOF is left
  out of the runtime gate because it does not scale, and the runtime anomaly signal is the
  log-amount z-score.

The drift fix (notebook 09) is a real finding, and I present it as one rather than as a limitation.
- The naive fixed-reference gate flagged all 61 incoming batches (100%) as drift, for two reasons:
  the fixed day-1-to-3 reference conflates natural temporal drift with quality problems, and the
  KS test is over-powered at 50k rows, so trivial differences read as drift with p near 0.
- The fix is a rolling reference (each batch is compared to recent history), the KS test run on a
  fixed sample of about 3k rows to control its power, the drift calibrated against normal clean
  variation, and failed batches excluded from the rolling buffer.
- On controlled synthetic data with known faults in known batches, it caught all three fault types
  and raised no false alarms on clean batches, even with gradual drift present.
- On the full real run (HI-Small, all 3,000,485 incoming rows, about 219k rows per second) the
  fixed reference flagged 12 of 61 (20%) and the rolling reference flagged 3 of 61 (5%). I state
  the decomposition honestly: the KS subsampling and calibration took it from 100% to 20%, and the
  rolling reference took it from 20% to 5%, so the rolling reference did not do all the work alone.
- Caveats: this run mainly exercises the drift dimension, because the real incoming data has no
  injected anomaly or missing faults, so those scores are near 0 and are validated instead in the
  ablation and the synthetic tests. The three rolling failures are batches that differ materially
  from recent history; confirming they are genuine quality events would need domain ground truth.

---

## 5. Build order

- Stage 0, environment and data load check. Done.
- Stage 1, injection functions in `src/injection.py`, all three tested. Done.
- Preprocessing in `src/preprocessing.py` plus the EDA in notebook 02. Done and validated.
- Stage 2, Module 1: Isolation Forest, then LOF, then Z-score, then IQR, then the autoencoder
  (PyTorch, built last, droppable if time runs short).
- Stage 3, Module 2: the structural check, then KS plus Chi-squared, then the RF classifier.
  Train the RF on injected drift at 5% and 10%, evaluate at 20% and 30% with different seeds, and
  test each drift type separately (drop, rename, dtype, distribution).
- Stage 4, Module 3: one XGBoost per target column, `scale_pos_weight` set to non-missing over
  missing, evaluated per column and then averaged.
- Stage 5, integration: Layer 1 purpose profiles, then Layer 2 confidence weighting, threshold
  0.5, with all module scores normalised to [0,1] first.
- Stage 6, ablation (A1 to A4), full evaluation, and all plots.

If time had run short, the minimum viable version was iForest, LOF, and Z/IQR for Module 1, steps
1 and 2 of Module 2, XGBoost for Module 3, equal-weighted integration, and all four ablation
experiments.

---

## 6. Integration layer spec

Layer 1 baseline weights by mode (each sums to 1.0):
- general (default): drift 0.5, anomaly 0.3, missing 0.2
- fraud: anomaly 0.5, drift 0.3, missing 0.2
- compliance: missing 0.5, drift 0.3, anomaly 0.2

Layer 2 is confidence-based and applied per batch, grounded in Almarshad et al. (2025):

    confidence_i   = abs(score_i - 0.5)
    raw_weight_i   = baseline_i * (0.5 + confidence_i)
    final_weight_i = raw_weight_i / sum(all raw_weights)
    combined_score = sum(final_weight_i * score_i)

All module scores are normalised to [0,1] before this. With a threshold of 0.5, a batch passes if
the combined score is below 0.5 and fails at or above it.

---

## 7. Evaluation and ablation

- Per module: Precision, Recall, F1, and ROC-AUC reported individually, plus a confusion matrix.
- Ablation: A1 removes Module 1, A2 removes Module 2, A3 removes Module 3, A4 compares equal,
  severity, and adaptive weighting. I report the F1 change per experiment as a percentage change
  from the full framework.
- Plots: a confusion matrix per module, ROC curves for all Module 1 methods on one plot, the RF
  and XGBoost feature importances, and the ablation bar chart.

---

## 8. Coding conventions

- The seed is 42 everywhere.
- Functions never modify the original DataFrame in place; they `.copy()` first.
- Transformers and models are fit on the clean reference batch and applied to the incoming data.
- Reusable functions live in `src/`, runnable notebooks in `notebooks/`, and the code is
  commented.
- The datasets are not committed to Git because they are large; see `.gitignore`.

---

## 9. Repository and HPC workflow

- Files are created on the Mac in `DISSERTATION/CODE/`.
- They sync to the Surrey HPC through Git: push from the Mac, pull on the HPC, run in VS Code.
- The datasets are not in Git. I copy them to the HPC once and point `DATA_PATH` in each notebook
  at their location there.

---

## 10. Known limitations to state honestly (for the limitations chapter)

Module 2 has a circularity limitation, since the RF is trained and evaluated on injected drift. On
the secondary dataset the PCA columns mean there is no structural drift check. There is no
streaming evaluation. The integration weights are set manually. The 0.5 threshold is not
empirically tuned. The default injection mechanism is MCAR, the simplest one. Both datasets are
synthetic or anonymised. I also correct the earlier "10-day span" claim, as noted in Section 3.
