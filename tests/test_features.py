"""Tests for customer-level feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from team_blue.data import CUSTOMER_ID_COLUMN
from team_blue.features import (
    PRODUCT_FEATURE_PREFIX,
    FeatureValidationError,
    align_validation_features,
    build_customer_features,
    model_inputs,
)


def _invoice_row(
    customer_id: int,
    tenure: int,
    subscription_id: int,
    product: str,
    mrr: float,
    target: bool,
) -> dict[str, object]:
    return {
        "PK_INVOICES": subscription_id + 1_000,
        "FK_DATE_INVOICE": pd.Timestamp("2023-01-15"),
        "CUSTOMER_START_DATE": pd.Timestamp("2023-01-01"),
        "TENURE": tenure,
        "FLG_DWH_CUSTOMER_IS_BUSINESS": customer_id % 2,
        CUSTOMER_ID_COLUMN: customer_id,
        "FK_SUBSCRIPTIONS": subscription_id,
        "PRODUCT_GROUP": product,
        "PRODUCT_SUBCATEGORY": product,
        "mrr": mrr,
        "ORIGIN_SUBGROUP": "PPC",
        "ORIGIN_GROUP": "Paid Traffic",
        "noise": target,
    }


@pytest.fixture
def clean_invoice_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _invoice_row(1, 0, 10, "Domain", 1.0, True),
            _invoice_row(1, 1, 10, "Domain", 2.0, True),
            _invoice_row(1, 1, 11, "Email", -1.0, True),
            _invoice_row(2, 0, 20, "Hosting", 3.0, False),
        ]
    )


def test_build_customer_features_returns_one_row_per_customer(
    clean_invoice_frame: pd.DataFrame,
) -> None:
    features = build_customer_features(clean_invoice_frame, include_target=True)

    assert features.index.tolist() == [1, 2]
    assert features.index.is_unique
    assert features.loc[1, "noise"]
    assert not features.loc[2, "noise"]


def test_build_customer_features_calculates_monthly_change_and_continuity(
    clean_invoice_frame: pd.DataFrame,
) -> None:
    features = build_customer_features(clean_invoice_frame, include_target=True)

    assert features.at[1, "mrr_sum_t0"] == pytest.approx(1.0)
    assert features.at[1, "mrr_sum_t1"] == pytest.approx(1.0)
    assert features.at[1, "mrr_sum_delta_t1_t0"] == pytest.approx(0.0)
    assert features.at[1, "negative_mrr_rows_t1"] == 1
    assert features.at[1, "retained_subscriptions_t1"] == 1
    assert features.at[1, "added_subscriptions_t1"] == 1
    assert features.at[1, "removed_subscriptions_t1"] == 0
    assert features.at[1, "subscription_jaccard_t0_t1"] == pytest.approx(0.5)
    assert features.at[1, f"{PRODUCT_FEATURE_PREFIX}Domain"] == 1
    assert features.at[1, f"{PRODUCT_FEATURE_PREFIX}Email"] == 1


def test_feature_builder_rejects_unclean_tenure(
    clean_invoice_frame: pd.DataFrame,
) -> None:
    invalid = clean_invoice_frame.copy()
    invalid.loc[invalid.index[0], "TENURE"] = -1

    with pytest.raises(FeatureValidationError, match="unsupported tenure"):
        build_customer_features(invalid, include_target=True)


def test_align_validation_uses_training_product_vocabulary(
    clean_invoice_frame: pd.DataFrame,
) -> None:
    training = build_customer_features(clean_invoice_frame, include_target=True)
    validation_rows = pd.DataFrame(
        [
            _invoice_row(3, 0, 30, "Domain", 4.0, False),
            _invoice_row(3, 1, 31, "New Product", 5.0, False),
        ]
    ).drop(columns="noise")
    validation = build_customer_features(validation_rows, include_target=False)

    aligned = align_validation_features(training, validation)

    assert aligned.columns.tolist() == training.drop(columns="noise").columns.tolist()
    assert f"{PRODUCT_FEATURE_PREFIX}New Product" not in aligned
    assert aligned.loc[3, f"{PRODUCT_FEATURE_PREFIX}Email"] == 0


def test_model_inputs_excludes_target_and_start_date(
    clean_invoice_frame: pd.DataFrame,
) -> None:
    features = build_customer_features(clean_invoice_frame, include_target=True)

    X, y, cohorts = model_inputs(features)

    assert "noise" not in X
    assert "customer_start_date" not in X
    assert y.index.equals(X.index)
    assert cohorts.index.equals(X.index)
    assert set(cohorts.astype(str)) == {"2023-01"}
