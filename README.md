# Team.blue assignment
This repository contains a customer-level churn modelling workflow for the team.blue data-science assignment. It uses invoice, subscription, product, acquisition and monthly recurring revenue (MRR) signals from a customer's first two months to predict whether that customer will churn within 12 months.

The analysis is notebook-led for readability, with reusable and tested logic under src/team_blue/.

# Repository

## Layout

Repository layout

```
.
├── data/          # Storage of input data (training and validation sets)
├── notebooks/     # Notebooks executed to explore the data and model it
├── src/team_blue/ # Folder project with features to run the notebooks
├── artifacts/     # Folder that stores all tracking results from models
└── outputs/       # Folder with the file to deliver for this case study
```

## Setup

The project requires:

- Python 3.13
- [uv](https://docs.astral.sh/uv/).

Verify the required tools:

```bash
python --version
uv --version
```

The Python command should report `Python 3.13.x`.

From the repository root:

```{bash}
uv sync
```

Place the assignment files at:
- data/training.csv
- data/validation.csv

The loader also accepts a different data directory when one is passed to the Python API or the --data-dir CLI option.

## Recommended notebook workflow

Start JupyterLab without changing the locked project dependencies:

Run the notebooks in order:

- 01_exploratory_data_analysis.ipynb inspects entity grain, churn prevalence, data quality, cohorts, MRR and categorical overlap.
- 02_customer_level_modelling.ipynb builds customer features, compares the majority baseline, logistic regression and XGBoost, and tunes the two trainable models on out-of-time folds.
- 03_model_validation_and_delivery.ipynb evaluates the frozen XGBoost configuration once on June 2023, refits on all labelled customers and creates the validation predictions.

The final notebook reads artifacts/tuning/xgboost/best_config.json by default and writes:
- outputs/predictions.csv
- artifacts/final/xgboost/xgboost_pipeline.joblib
- artifacts/final/xgboost/holdout_metrics.csv
- artifacts/final/xgboost/metadata.json

## Data decisions

- All modelling occurs at customer level; invoice-product rows are never treated as independent observations.
- One customer with conflicting target values is removed because no deterministic outcome can be assigned.
- Rows whose TENURE is outside the defined {0, 1} observation window are removed and counted in a cleaning report.
- Negative MRR values are retained as possible credit or refund signals.
- Missing acquisition values are represented explicitly as Missing.
- Customer, invoice and subscription identifiers are used for grouping and continuity calculations, not as raw predictors.
- Encoders ignore unseen categories, and validation features are aligned to the training schema.

## Customer-level features

features.py creates one row per customer using only the first two tenure months. The feature families include:
- Invoice, subscription and product-diversity measures
- Total, average, minimum and maximum MRR
- Tenure-month 0 and 1 summaries and changes between them
- Negative-MRR incidence and amount
- Retained, added and removed subscriptions plus subscription Jaccard similarity
- Product-subcategory presence flags
- Acquisition group/subgroup and business/private status.
- Dates are retained as cohort metadata for temporal splitting rather than passed directly into the model.

## Validation and model selection

The training cohorts run from January to June 2023, while the unlabeled validation cohorts are from January to March 2024. To better represent deployment through time:

1. Expanding-window folds train only on cohorts earlier than each validation cohort
2. April and May 2023 are used for model and hyperparameter comparison
3. Optuna maximises mean temporal ROC AUC
4. The binary threshold is selected separately from out-of-time predictions, using balanced accuracy by default
5. June 2023 remains untouched until the model family, parameters and threshold are frzen.

This design reduces temporal leakage and gives a more realistic estimate than a random row split. It does not remove all overfitting risk, so fold variability, train-to-holdout gaps and future drift remain important.

## Prediction contract

* `outputs/predictions.csv` contains exactly: FK_DWH_CUSTOMERS,Predictions
There is one unique row per validation customer, and Predictions is a Boolean classification rather than a probability.

## Tests and code quality

Run the automated checks from the repository root:

```{bash}
uv run pytest
uv run ruff check src tests
```

The tests use small synthetic frames and do not require the assignment CSV files. They cover data contracts and cleaning, feature aggregation/alignment, temporal folds, evaluation, model construction and probability handling.

## Reproducibility and limitations

- Random seeds are fixed at 42 in tuning and supported model constructors.
- Saved tuning files contain parameters, threshold, fold periods, metrics and trial history.
- Saved final metadata records the configuration, customer counts, cleaning report and holdout result.
- Product and acquisition associations are descriptive, not causal churn drivers.
- Only six labelled acquisition cohorts are available, so temporal performance has material uncertainty.
- The 72.1% positive rate on validation is above the historical 62.1% customer churn rate. Revenue forecasting should therefore use calibrated probabilities and observed outcomes when available, rather than treating every positive prediction as certain churn.
