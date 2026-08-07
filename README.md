# Data Quality Monitoring Framework

Code for my MSc Data Science dissertation at the University of Surrey.

The project builds an automated quality gate for a financial data pipeline. An incoming batch of
data is compared against a clean reference batch and checked along three dimensions at once:
value-level anomalies, schema and distribution drift, and missing values. The three checks are
combined into a single score, and the batch is either passed on to QA or sent back to the
developer.

## Layout

- `src/` holds the reusable Python modules: fault injection, preprocessing, the three detection
  modules, the integration layer, the framework orchestrator, and the diagnostics.
- `notebooks/` holds the runnable notebooks in build order. They open cleanly in VS Code.
- `data/` is where the datasets go. They are large, so they are kept out of Git.
- `results/` holds the saved metrics, figures, and run summaries.

## Where to start

Read `PROJECT_CONTEXT.md` first. It records the design decisions, what I found in the data, and
the results, so the reasoning behind the code is all in one place.
