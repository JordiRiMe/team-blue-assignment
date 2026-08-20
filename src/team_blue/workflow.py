"""Shared preparation workflow for tuning and final model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from team_blue.data import (
    CleaningReport,
    clean_training_data,
    load_datasets,
    read_invoice_csv,
    resolve_dataset_paths,
)
from team_blue.features import (
    align_validation_features,
    build_customer_features,
    model_inputs,
    scoring_inputs,
)


@dataclass(frozen=True)
class PreparedTrainingData:
    """Customer-level matrices used for model selection."""

    X_train: pd.DataFrame
    y_train: pd.Series
    cohorts: pd.Series
    cleaning_report: CleaningReport | None


@dataclass(frozen=True)
class PreparedModelingData(PreparedTrainingData):
    """Customer-level matrices used for final fitting and scoring."""

    X_validation: pd.DataFrame


def prepare_training_data(
    data_dir: str | Path | None = None,
) -> PreparedTrainingData:
    """Load and aggregate training data without inspecting validation data."""

    training_path = resolve_dataset_paths(data_dir).training
    raw_training = read_invoice_csv(training_path, "training")
    training, cleaning_report = clean_training_data(raw_training)
    training_customers = build_customer_features(training, include_target=True)
    X_train, y_train, cohorts = model_inputs(training_customers)

    return PreparedTrainingData(
        X_train=X_train,
        y_train=y_train,
        cohorts=cohorts,
        cleaning_report=cleaning_report,
    )


def prepare_modeling_data(
    data_dir: str | Path | None = None,
) -> PreparedModelingData:
    """Load, clean, aggregate, and align both assignment datasets."""

    datasets = load_datasets(data_dir)
    training_customers = build_customer_features(
        datasets.training,
        include_target=True,
    )
    validation_customers = build_customer_features(
        datasets.validation,
        include_target=False,
    )
    aligned_validation = align_validation_features(
        training_customers,
        validation_customers,
    )

    X_train, y_train, cohorts = model_inputs(training_customers)
    X_validation = scoring_inputs(aligned_validation)
    if X_train.columns.tolist() != X_validation.columns.tolist():
        raise ValueError("Training and validation feature columns are not aligned.")

    return PreparedModelingData(
        X_train=X_train,
        y_train=y_train,
        cohorts=cohorts,
        cleaning_report=datasets.cleaning_report,
        X_validation=X_validation,
    )