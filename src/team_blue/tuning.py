"""Optuna-based hyperparameter tuning with a final temporal holdout."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from team_blue.modeling import (
    ModelName,
    ModelParameter,
    TemporalFold,
    evaluate_temporal_model,
    make_expanding_window_folds,
    make_model_pipeline,
)
from team_blue.workflow import prepare_training_data

ThresholdMetric = Literal["accuracy", "balanced_accuracy", "f1"]


class TrialLike(Protocol):
    """Subset of the Optuna Trial API used by this module."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float: ...

    def suggest_int(self, name: str, low: int, high: int) -> int: ...

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[ModelParameter],
    ) -> ModelParameter: ...

    def set_user_attr(self, key: str, value: float | str) -> None: ...


@dataclass(frozen=True)
class TuningArtifacts:
    """Files produced by one model-family tuning run."""

    configuration: Path
    tuning_metrics: Path
    trials: Path


def suggest_model_parameters(
    trial: TrialLike,
    model_name: ModelName,
) -> dict[str, ModelParameter]:
    """Define bounded, model-specific hyperparameter search spaces."""

    if model_name == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.001,
                0.2,
                log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                100.0,
                log=True,
            ),
        }
    raise ValueError(f"Unsupported model: {model_name!r}.")


def _calculate_threshold_score(
    actual: pd.Series,
    predicted: pd.Series,
    metric: ThresholdMetric,
) -> float:
    """Calculate a threshold metric as a native Python float."""

    if metric == "accuracy":
        return float(
            accuracy_score(
                actual,
                predicted,
            )
        )

    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        )

    if metric == "f1":
        return float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        )

    raise ValueError(f"Unsupported threshold metric: {metric!r}.")


def select_classification_threshold(
    target: pd.Series,
    probability: pd.Series,
    *,
    metric: ThresholdMetric = "balanced_accuracy",
) -> tuple[float, float]:
    """Select a binary threshold from out-of-time predictions only."""

    if not target.index.equals(probability.index):
        raise ValueError("Target and probability must share the same ordered index.")
    if not bool(probability.between(0.0, 1.0).all()):
        raise ValueError("Probabilities must be between 0 and 1.")

    candidates: list[tuple[float, float, float]] = []

    for threshold in np.linspace(0.05, 0.95, 181):
        threshold_value = float(threshold)
        predicted = probability.ge(threshold_value)

        score = _calculate_threshold_score(
            target,
            predicted,
            metric,
        )

        # The second value breaks score ties by choosing the threshold
        # closest to 0.5.
        candidates.append(
            (
                score,
                -abs(threshold_value - 0.5),
                threshold_value,
            )
        )

    best_score, _, best_threshold = max(candidates)

    return best_threshold, best_score


def make_objective(
    model_name: ModelName,
    X: pd.DataFrame,
    y: pd.Series,
    cohorts: pd.Series,
    tuning_folds: Sequence[TemporalFold],
) -> Callable[[TrialLike], float]:
    """Build an Optuna objective maximizing mean temporal ROC AUC."""

    def objective(trial: TrialLike) -> float:
        parameters = suggest_model_parameters(trial, model_name)
        estimator = make_model_pipeline(X, model_name, parameters)
        evaluation = evaluate_temporal_model(
            estimator,
            X,
            y,
            cohorts,
            model_name=model_name,
            folds=tuning_folds,
        )

        metrics = evaluation.metrics
        validation_roc_auc = pd.Series(
            metrics.loc[:, "roc_auc"],
            dtype=float,
        )

        train_roc_auc = pd.Series(
            metrics.loc[:, "train_roc_auc"],
            dtype=float,
        )
        roc_auc_gap = train_roc_auc - validation_roc_auc

        scores = pd.Series(evaluation.metrics.loc[:, "roc_auc"], dtype=float)
        trial.set_user_attr("fold_roc_auc_std", float(scores.std(ddof=0)))
        trial.set_user_attr("worst_fold_roc_auc", float(scores.min()))
        trial.set_user_attr("mean_roc_auc_gap", float(roc_auc_gap.mean()))
        return float(scores.mean())

    return objective


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a metrics frame to records containing native JSON values."""

    return json.loads(frame.to_json(orient="records"))


def tune_model(
    model_name: ModelName,
    *,
    data_dir: str | Path | None = None,
    output_dir: str | Path = "artifacts/tuning",
    n_trials: int = 50,
    seed: int = 42,
    threshold_metric: ThresholdMetric = "balanced_accuracy",
) -> TuningArtifacts:
    """Tune one model family while leaving the latest cohort untouched."""

    try:
        import optuna
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Optuna is not installed. Run `uv sync` after adding optuna."
        ) from error

    prepared = prepare_training_data(data_dir)
    folds = make_expanding_window_folds(prepared.cohorts)
    if len(folds) < 2:
        raise ValueError(
            "At least two temporal folds are required to reserve a final holdout."
        )
    tuning_folds = folds[:-1]
    reserved_holdout_fold = folds[-1]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{model_name}_temporal_roc_auc",
    )
    study.optimize(
        make_objective(
            model_name,
            prepared.X_train,
            prepared.y_train,
            prepared.cohorts,
            tuning_folds,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    best_parameters = dict(study.best_trial.params)
    selected_estimator = make_model_pipeline(
        prepared.X_train,
        model_name,
        best_parameters,
    )
    tuning_evaluation = evaluate_temporal_model(
        selected_estimator,
        prepared.X_train,
        prepared.y_train,
        prepared.cohorts,
        model_name=model_name,
        folds=tuning_folds,
    )
    tuning_target = pd.Series(
        tuning_evaluation.predictions.loc[:, "target"],
        index=tuning_evaluation.predictions.index,
        dtype=bool,
    )
    tuning_probability = pd.Series(
        tuning_evaluation.predictions.loc[:, "probability"],
        index=tuning_evaluation.predictions.index,
        dtype=float,
    )
    threshold, threshold_score = select_classification_threshold(
        tuning_target,
        tuning_probability,
        metric=threshold_metric,
    )
    destination = (
        Path(output_dir)
        .expanduser()
        .resolve()
        / model_name
    )
    destination.mkdir(parents=True, exist_ok=True)
    configuration_path = destination / "best_config.json"
    tuning_metrics_path = destination / "tuning_metrics.csv"
    trials_path = destination / "trials.csv"

    summary = {
        "schema_version": 1,
        "model_name": model_name,
        "selection_metric": "roc_auc",
        "best_tuning_score": float(study.best_value),
        "parameters": best_parameters,
        "threshold": threshold,
        "threshold_metric": threshold_metric,
        "threshold_tuning_score": threshold_score,
        "tuning_periods": [str(fold.validation_period) for fold in tuning_folds],
        "reserved_holdout_period": str(reserved_holdout_fold.validation_period),
        "tuning_metrics": _records(tuning_evaluation.metrics),
    }
    configuration_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tuning_evaluation.metrics.to_csv(tuning_metrics_path, index=False)
    study.trials_dataframe().to_csv(trials_path, index=False)

    return TuningArtifacts(
        configuration=configuration_path,
        tuning_metrics=tuning_metrics_path,
        trials=trials_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a churn model using expanding temporal folds.",
    )
    parser.add_argument("--model", choices=("logistic", "xgboost"), required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-metric",
        choices=("accuracy", "balanced_accuracy", "f1"),
        default="balanced_accuracy",
    )
    return parser.parse_args()


def main() -> None:
    """Run tuning from the command line."""

    args = _parse_args()
    artifacts = tune_model(
        args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.trials,
        seed=args.seed,
        threshold_metric=args.threshold_metric,
    )
    print(f"Best configuration: {artifacts.configuration}")


if __name__ == "__main__":
    main()
