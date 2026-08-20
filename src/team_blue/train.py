"""Fit a selected churn model and create the assignment predictions CSV."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import pandas as pd

from team_blue.data import CUSTOMER_ID_COLUMN
from team_blue.modeling import (
    ModelName,
    ModelParameter,
    evaluate_temporal_model,
    make_expanding_window_folds,
    make_model_pipeline,
    positive_class_probability,
)
from team_blue.workflow import prepare_modeling_data

PREDICTION_COLUMN = "Predictions"


@dataclass(frozen=True)
class SelectedModelConfiguration:
    """Validated model and threshold selected by temporal tuning."""

    model_name: ModelName
    parameters: dict[str, ModelParameter]
    threshold: float


@dataclass(frozen=True)
class TrainingArtifacts:
    """Files produced by final model fitting."""

    predictions: Path
    fitted_model: Path
    metadata: Path
    holdout_metrics: Path


def load_selected_configuration(path: str | Path) -> SelectedModelConfiguration:
    """Read and validate a configuration produced by ``tuning.py``."""

    configuration_path = Path(path).expanduser().resolve()
    payload = json.loads(configuration_path.read_text(encoding="utf-8"))

    model_name = payload.get("model_name")
    if model_name not in {"logistic", "xgboost"}:
        raise ValueError("Configuration contains an unsupported model_name.")
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise TypeError("Configuration parameters must be a JSON object.")
    parameters: dict[str, ModelParameter] = {}
    for name, value in raw_parameters.items():
        if not isinstance(name, str) or not (
            value is None or isinstance(value, (float, int, str))
        ):
            raise ValueError("Configuration contains an invalid model parameter.")
        parameters[name] = value

    threshold = float(payload.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Configuration threshold must be between 0 and 1.")
    return SelectedModelConfiguration(
        model_name=cast(ModelName, model_name),
        parameters=parameters,
        threshold=threshold,
    )


def train_and_predict(
    configuration_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    predictions_path: str | Path = "outputs/predictions.csv",
    model_path: str | Path = "artifacts/final_model.joblib",
) -> TrainingArtifacts:
    """Refit the selected model on all training customers and score validation."""

    configuration = load_selected_configuration(configuration_path)
    prepared = prepare_modeling_data(data_dir)
    estimator = make_model_pipeline(
        prepared.X_train,
        configuration.model_name,
        configuration.parameters,
    )
    folds = make_expanding_window_folds(prepared.cohorts)
    holdout_evaluation = evaluate_temporal_model(
        estimator,
        prepared.X_train,
        prepared.y_train,
        prepared.cohorts,
        model_name=configuration.model_name,
        folds=[folds[-1]],
        threshold=configuration.threshold,
    )
    estimator.fit(prepared.X_train, prepared.y_train)
    probability = positive_class_probability(estimator, prepared.X_validation)
    prediction = probability.ge(configuration.threshold)

    output = pd.DataFrame(
        {
            CUSTOMER_ID_COLUMN: prepared.X_validation.index.to_numpy(),
            PREDICTION_COLUMN: prediction.to_numpy(dtype=bool),
        }
    )
    if bool(output[CUSTOMER_ID_COLUMN].duplicated().any()):
        raise ValueError("Prediction output contains duplicate customer IDs.")
    if output.columns.tolist() != [CUSTOMER_ID_COLUMN, PREDICTION_COLUMN]:
        raise ValueError("Prediction output does not match the required schema.")

    predictions_destination = Path(predictions_path).expanduser().resolve()
    model_destination = Path(model_path).expanduser().resolve()
    metadata_destination = model_destination.with_suffix(".json")
    holdout_destination = model_destination.with_name("final_holdout_metrics.csv")
    predictions_destination.parent.mkdir(parents=True, exist_ok=True)
    model_destination.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(predictions_destination, index=False)
    joblib.dump(estimator, model_destination)
    holdout_evaluation.metrics.to_csv(holdout_destination, index=False)
    metadata = {
        "schema_version": 1,
        "model_name": configuration.model_name,
        "parameters": configuration.parameters,
        "threshold": configuration.threshold,
        "training_customers": len(prepared.X_train),
        "validation_customers": len(prepared.X_validation),
        "positive_predictions": int(prediction.sum()),
        "positive_prediction_rate": float(prediction.mean()),
        "holdout_metrics": json.loads(
            holdout_evaluation.metrics.to_json(orient="records")
        ),
        "cleaning_report": (
            prepared.cleaning_report.as_dict()
            if prepared.cleaning_report is not None
            else None
        ),
    }
    metadata_destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TrainingArtifacts(
        predictions=predictions_destination,
        fitted_model=model_destination,
        metadata=metadata_destination,
        holdout_metrics=holdout_destination,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected churn model and score validation customers.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions.csv"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/final_model.joblib"),
    )
    return parser.parse_args()


def main() -> None:
    """Run final fitting and scoring from the command line."""

    args = _parse_args()
    artifacts = train_and_predict(
        args.config,
        data_dir=args.data_dir,
        predictions_path=args.predictions,
        model_path=args.model_output,
    )
    print(f"Predictions: {artifacts.predictions}")
    print(f"Fitted model: {artifacts.fitted_model}")
    print(f"Final holdout metrics: {artifacts.holdout_metrics}")


if __name__ == "__main__":
    main()