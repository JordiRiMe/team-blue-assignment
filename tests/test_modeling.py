"""Tests for preprocessing and out-of-time model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from team_blue.modeling import (
    evaluate_temporal_model,
    make_dummy_pipeline,
    make_expanding_window_folds,
    make_logistic_pipeline,
)


@pytest.fixture
def temporal_model_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    periods = pd.period_range("2023-01", "2023-06", freq="M")
    index = pd.Index(range(100, 124), name="FK_DWH_CUSTOMERS")
    cohorts = pd.Series(
        np.repeat(periods, 4),
        index=index,
        name="customer_start_month",
    )
    numeric_signal = np.tile([0.0, 1.0, 2.0, 3.0], len(periods))
    X = pd.DataFrame(
        {
            "mrr_sum_t0": numeric_signal,
            "subscription_count": np.tile([1, 1, 2, 2], len(periods)),
            "origin_group": np.tile(
                ["Organic", "Paid", "Organic", "Paid"], len(periods)
            ),
            "origin_subgroup": np.tile(["SEO", "PPC", "SEO", "PPC"], len(periods)),
        },
        index=index,
    )
    y = pd.Series(
        np.tile([False, False, True, True], len(periods)),
        index=index,
        name="noise",
    )
    return X, y, cohorts


def test_expanding_window_folds_are_strictly_temporal(
    temporal_model_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    _, _, cohorts = temporal_model_data

    folds = make_expanding_window_folds(cohorts)

    assert [str(fold.validation_period) for fold in folds] == [
        "2023-04",
        "2023-05",
        "2023-06",
    ]
    for fold in folds:
        assert (cohorts.loc[fold.train_index] < fold.validation_period).all()
        assert (cohorts.loc[fold.validation_index] == fold.validation_period).all()


@pytest.mark.parametrize("model_name", ["Dummy", "Logistic"])
def test_temporal_evaluation_returns_metrics_and_predictions(
    temporal_model_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    model_name: str,
) -> None:
    X, y, cohorts = temporal_model_data
    estimator = (
        make_dummy_pipeline(X) if model_name == "Dummy" else make_logistic_pipeline(X)
    )

    evaluation = evaluate_temporal_model(
        estimator,
        X,
        y,
        cohorts,
        model_name=model_name,
    )

    assert len(evaluation.metrics) == 3
    assert len(evaluation.predictions) == 12
    assert set(evaluation.metrics["validation_period"]) == {
        "2023-04",
        "2023-05",
        "2023-06",
    }
    assert evaluation.metrics["accuracy"].between(0, 1).all()
    assert evaluation.predictions["probability"].between(0, 1).all()


def test_temporal_evaluation_rejects_misaligned_indices(
    temporal_model_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, cohorts = temporal_model_data
    misaligned_y = y.sample(frac=1, random_state=42)

    with pytest.raises(ValueError, match="same ordered customer index"):
        evaluate_temporal_model(
            make_dummy_pipeline(X),
            X,
            misaligned_y,
            cohorts,
            model_name="Dummy",
        )
