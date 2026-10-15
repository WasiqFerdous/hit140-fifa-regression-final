# HIT140 FIFA regression final working version

The two required original eight-input OLS regressions, fixed random test splits,
coefficients and original test scores are retained. All five source CSV files
are unchanged. The executed notebook contains 25 code cells and six figures.

## Run

Open FIFA_World_Cup_2026_Regression_Final.ipynb and run all cells from top to
bottom, with the source CSVs in the same folder. In Colab upload the three raw
CSVs and the notebook into the session. Locally install requirements.txt and
run `python run_analysis.py`. Generated files go into analysis_outputs.

## Methods and small improvements

- Task 2.1 has 104 fixtures and an 83/21 train/test partition.
- Task 2.2 has 208 team rows and a 166/42 partition, grouped by match ID.
- Five-fold CV uses only training records, with grouped folds for task 2.2.
- Compare a training-mean benchmark, original OLS and scaled Ridge on the same
  eight inputs and same folds. The mean is a reference, not a replacement for
  either required eight-input regression.
- Try Ridge alpha 1, 10 and 100. Select each task's alpha using lowest mean CV
  RMSE. StandardScaler is fitted within each fold through a pipeline. The CV
  values are selection scores; they are not unbiased post-selection estimates.
- Freeze the Ridge choice before evaluating its original test partition. The
  original test outcomes were inspected in earlier drafts and are not a new
  blind test. No further tuning follows test diagnostics.
- Residual plots compare OLS and Ridge; the original coefficient tables remain
  OLS. Positive residuals mean underprediction. Raw predictions are not clipped.

The mean benchmark is explicit in supplied Week 9 Document20.pdf, Task 1.
Ridge is one extension beyond the supplied lecture notes, supported by the
official scikit-learn documentation:
https://scikit-learn.org/1.8/modules/linear_model.html#ridge-regression-and-classification
https://scikit-learn.org/1.8/modules/compose.html#pipeline

See comparison_cv_folds.csv, comparison_cv_summary.csv, comparison_selection.json,
model_comparison.csv and diagnostic_summary.csv in analysis_outputs. All-record
EDA is descriptive and was not used to select the predictors. Random validation
does not establish future forecasting performance. Source defaults, estimated
valuations and incomplete provenance remain in FIFA_Data_Source_Checks.md.

AI assistance must be disclosed and all members must understand the code. The
brief limits report assistance to outlining and final proofreading; the group
remains responsible for its own final wording and genuine contributions.
