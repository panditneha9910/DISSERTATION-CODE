# Results, organised by stage

This folder is a reader-friendly copy of `results/`, grouped by the notebook that
produces each figure. It is for viewing and for pulling figures into the report. The
notebooks themselves still read from and write to the flat `results/` folder, so that
one is the working copy and should not be renamed.

- 02_EDA — amount distribution before and after the log transform.
- 03_Module1_anomaly — ROC for the anomaly detectors on the IBM data.
- 04_Module2_drift — drift classifier confusion matrix.
- 05_Module3_missing — feature importance for the missing-value model.
- 07_Ablation — F1 for each ablation experiment.
- 08_Secondary_CreditCard — EDA, Module 1 ROC, and ablation on the Credit Card data.
- 09_Fullscale_run — per-batch summaries (fixed and rolling) and the drift comparison.
- 10_Dashboard — the presentation figures (gate donut, before/after, detectors, ablation, summary table).
- 11_Fault_reports — the three demonstration reports and the three real flagged-batch reports.
