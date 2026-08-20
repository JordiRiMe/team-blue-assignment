"""Modelling pipelines and out-of-time evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

from team_blue.features import CATEGORICAL_FEATURE_COLUMNS

type ModelName = Literal["logistic", "xgboost"]
type ModelParameter = float | int | str | None

type LogisticPenalty = Literal["l1", "l2"]
type LogisticClassWeight = Literal["balanced"] | None


@dataclass(frozen=True)
class TemporalFold:
    """Indices for one expanding-window evaluation fold."""

    validation_period: pd.Period
    train_index: pd.Index
    validation_index: pd.Index


@dataclass(frozen=True)
class TemporalEvaluation:
    """Per-fold metrics and out-of-time customer predictions."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create leakage-safe numeric and categorical transformations."""

    categorical_columns = [
        column for column in CATEGORICAL_FEATURE_COLUMNS if column in X.columns
    ]
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]
    if not numeric_columns:
        raise ValueError("At least one numeric feature is required.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                    missing_values=pd.NA, # type: ignore
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_dummy_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create a majority-class baseline using the same preprocessing contract."""

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )


def make_logistic_pipeline(
    X: pd.DataFrame,
    *,
    C: float = 1.0,
    l1_ratio: float = 0.0,
    dual: bool = False,
    class_weight: str | None = None,
) -> Pipeline:
    """
    Create an interpretable regularised logistic-regression baseline.
    More restrictive model due to linear relationship limitation.
    """
    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        dual=dual,
        solver="liblinear",
        max_iter=2_000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            ("model", model),
        ]
    )

def make_xgboost_pipeline(
    X: pd.DataFrame,
    *,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    min_child_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
) -> Pipeline:
    """
    XGBoost can capture non-linear relationships not seen by Logistic regression.
    Although interpretation is more limited to the SHAP values.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            ("model", model),
        ]
    )

def make_expanding_window_folds(
    cohorts: pd.Series,
    *,
    minimum_training_periods: int = 3,
) -> list[TemporalFold]:
    """Build chronological folds where every validation month is in the future."""

    if not cohorts.index.is_unique:
        raise ValueError("Cohort index must contain one row per customer.")
    periods = sorted(pd.Period(value, freq="M") for value in cohorts.dropna().unique())
    if len(periods) <= minimum_training_periods:
        raise ValueError(
            "Not enough customer cohorts for expanding-window validation: "
            f"found {len(periods)}, need more than {minimum_training_periods}."
        )

    folds: list[TemporalFold] = []
    for validation_period in periods[minimum_training_periods:]:
        train_index = cohorts.index[cohorts < validation_period]
        validation_index = cohorts.index[cohorts == validation_period]
        if len(train_index) and len(validation_index):
            folds.append(
                TemporalFold(
                    validation_period=validation_period,
                    train_index=train_index,
                    validation_index=validation_index,
                )
            )
    return folds


def positive_class_probability(estimator: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    probabilities = estimator.predict_proba(X)  # type: ignore[attr-defined]
    classes = list(estimator.classes_)  # type: ignore[attr-defined]
    positive_position = classes.index(True) if True in classes else classes.index(1)
    return pd.Series(probabilities[:, positive_position], index=X.index)


def _classification_metrics(
    y_true: pd.Series,
    predicted: pd.Series,
    probability: pd.Series,
) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
    }


def evaluate_temporal_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cohorts: pd.Series,
    *,
    model_name: str,
    minimum_training_periods: int = 3,
    folds: Sequence[TemporalFold] | None = None,
    threshold: float = 0.5,
) -> TemporalEvaluation:
    """Evaluate a model using expanding out-of-time customer cohorts."""

    if not X.index.equals(y.index) or not X.index.equals(cohorts.index):
        raise ValueError(
            "X, y, and cohorts must share the same ordered customer index."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    # Use explicitly supplied folds during tuning or create all temporal
    # folds when the caller does not provide them.
    evaluation_folds = (
        list(folds)
        if folds is not None
        else make_expanding_window_folds(
            cohorts,
            minimum_training_periods=minimum_training_periods,
        )
    )

    if not evaluation_folds:
        raise ValueError(
            "At least one temporal evaluation fold is required."
        )

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in evaluation_folds:
        fitted = clone(estimator)

        X_train = X.loc[fold.train_index]
        y_train = y.loc[fold.train_index]
        X_validation = X.loc[fold.validation_index]
        y_validation = y.loc[fold.validation_index]

        fitted.fit(X_train, y_train)

        train_probability = positive_class_probability(
            fitted,
            X_train,
        )
        validation_probability = positive_class_probability(
            fitted,
            X_validation,
        )

        train_prediction = train_probability.ge(threshold)
        validation_prediction = validation_probability.ge(threshold)

        train_metrics = _classification_metrics(
            y_train,
            train_prediction,
            train_probability,
        )

        validation_metrics = _classification_metrics(
            y_validation,
            validation_prediction,
            validation_probability,
        )

        metric_rows.append(
            {
                "model": model_name,
                "validation_period": str(fold.validation_period),
                "training_customers": len(fold.train_index),
                "validation_customers": len(fold.validation_index),
                **validation_metrics,
                **{
                    f"train_{metric}": value
                    for metric, value in train_metrics.items()
                },
                "roc_auc_gap": (
                    train_metrics["roc_auc"] - validation_metrics["roc_auc"]
                ),
                "average_precision_gap": (
                    train_metrics["average_precision"] - validation_metrics["average_precision"]
                ),
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "validation_period": str(
                        fold.validation_period
                    ),
                    "target": y_validation,
                    "prediction": validation_prediction,
                    "probability": validation_probability,
                }
            )
        )

    return TemporalEvaluation(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(
            prediction_frames,
            axis=0,
        ).sort_index(),
    )

def _float_parameter(
    parameters: Mapping[str, ModelParameter],
    name: str,
    default: float,
) -> float:
    value = parameters.get(name, default)

    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"Model parameter {name!r} must be numeric.")

    return float(value)


def _int_parameter(
    parameters: Mapping[str, ModelParameter],
    name: str,
    default: int,
) -> int:
    value = parameters.get(name, default)

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Model parameter {name!r} must be an integer.")

    return value


def make_model_pipeline(
    X: pd.DataFrame,
    model_name: ModelName,
    parameters: Mapping[str, ModelParameter] | None = None,
) -> Pipeline:
    """Build a model pipeline from an Optuna-compatible parameter mapping."""

    values = dict(parameters or {})

    if model_name == "logistic":
        allowed = {"C", "penalty", "class_weight"}
        unknown = sorted(set(values) - allowed)

        if unknown:
            raise ValueError(
                f"Unknown logistic-regression parameters: {unknown}."
            )

        penalty = values.get("penalty", "l2")
        class_weight = values.get("class_weight")

        if penalty not in {"l1", "l2"}:
            raise ValueError(
                "Logistic-regression penalty must be 'l1' or 'l2'."
            )

        if class_weight not in {None, "balanced"}:
            raise ValueError(
                "Logistic-regression class_weight must be "
                "None or 'balanced'."
            )

        return make_logistic_pipeline(
            X,
            C=_float_parameter(values, "C", 1.0),
            l1_ratio=_float_parameter(values, "l1_ratio", 0.0),
            class_weight=cast(
                LogisticClassWeight,
                class_weight,
            ),
        )

    if model_name == "xgboost":
        allowed = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        }
        unknown = sorted(set(values) - allowed)

        if unknown:
            raise ValueError(f"Unknown XGBoost parameters: {unknown}.")

        return make_xgboost_pipeline(
            X,
            n_estimators=_int_parameter(
                values,
                "n_estimators",
                400,
            ),
            learning_rate=_float_parameter(
                values,
                "learning_rate",
                0.05,
            ),
            max_depth=_int_parameter(
                values,
                "max_depth",
                4,
            ),
            min_child_weight=_float_parameter(
                values,
                "min_child_weight",
                1.0,
            ),
            subsample=_float_parameter(
                values,
                "subsample",
                0.8,
            ),
            colsample_bytree=_float_parameter(
                values,
                "colsample_bytree",
                0.8,
            ),
            reg_alpha=_float_parameter(
                values,
                "reg_alpha",
                0.0,
            ),
            reg_lambda=_float_parameter(
                values,
                "reg_lambda",
                1.0,
            ),
        )

    raise ValueError(f"Unsupported model: {model_name!r}.")
