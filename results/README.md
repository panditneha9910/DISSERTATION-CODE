# Results

This folder holds every figure and data file the notebooks produce. The files are named
by stage, so a prefix tells you where a file comes from: `module1/2/3` for the primary
IBM modules, `ablation` for the ablation study, `fullscale` for the whole-stream run,
`dash` for the presentation dashboard, `fault` for the fault reports, and `secondary`
for the Credit Card dataset. This guide lists them in the order the report uses them.

## Start here

If you only look at three files, look at these. `dash_gate_donut.png` is the headline:
the share of the incoming data the gate passed automatically versus the share it sent for
review. `dash_summary_table.png` says the same thing in plain language for a non-technical
reader. `dash_before_after.png` shows the drift fix that made the gate usable.

## Primary dataset (IBM transactions)

| File | What it shows | Notebook |
|---|---|---|
| amount_distribution.png | Amount Paid before and after the log transform (the heavy skew) | 02 |
| module1_roc.png | Module 1 ROC on the primary data | 03 |
| module1_roc_all.png | ROC for all five anomaly detectors | 03 |
| module2_confusion.png | Drift classifier confusion matrix | 04 |
| module3_importance.png | Feature importance for the missing-value model | 05 |
| ablation_f1.png | F1 for each ablation experiment | 07 |

## Full-scale run (entire incoming stream)

| File | What it shows | Notebook |
|---|---|---|
| fullscale_fixed.csv | Per-batch summary, naive fixed reference | 09 |
| fullscale_rolling.csv | Per-batch summary, rolling reference (the fix) | 09 |
| fullscale_fixed_vs_rolling.png | Per-batch drift score, fixed versus rolling | 09 |

## Presentation dashboard

| File | What it shows | Notebook |
|---|---|---|
| dash_gate_donut.png | Share of the stream passed versus flagged | 10 |
| dash_before_after.png | Flagged rate before and after the drift fix | 10 |
| dash_module1_detectors.png | Anomaly detectors compared on real fraud | 10 |
| dash_ablation.png | Module and combining-rule contribution | 10 |
| dash_summary_table.png | Plain-language summary of the findings | 10 |

## Fault reports

| File | What it shows | Notebook |
|---|---|---|
| fault_demo_anomaly.png | Example report for an anomaly fault | 11 |
| fault_demo_drift.png | Example report for a drift fault | 11 |
| fault_demo_missing.png | Example report for a missing-value fault | 11 |
| fault_real_batch55.png | Report for a batch the full-scale run flagged | 11 |
| fault_real_batch56.png | Report for a batch the full-scale run flagged | 11 |
| fault_real_batch57.png | Report for a batch the full-scale run flagged | 11 |

## Secondary dataset (Credit Card Fraud)

| File | What it shows | Notebook |
|---|---|---|
| secondary_eda.png | EDA of the Credit Card data | 08 |
| secondary_module1_roc.png | Module 1 ROC against real fraud labels | 08 |
| secondary_ablation.png | Ablation on the Credit Card data | 08 |
