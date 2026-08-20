"""Customer-level feature engineering for churn modelling."""

from __future__ import annotations

import pandas as pd

from team_blue.data import (
    CUSTOMER_ID_COLUMN,
    TARGET_COLUMN,
    TENURE_COLUMN,
    VALID_TENURES,
)

CUSTOMER_START_COLUMN = "customer_start_date"
CATEGORICAL_FEATURE_COLUMNS = ("origin_group", "origin_subgroup")
PRODUCT_FEATURE_PREFIX = "has_product_subcategory::"


class FeatureValidationError(ValueError):
    """Raised when invoice rows cannot be safely aggregated by customer."""


def _validate_feature_input(invoice_data: pd.DataFrame, include_target: bool) -> None:
    required_columns = {
        "PK_INVOICES",
        "FK_DATE_INVOICE",
        "CUSTOMER_START_DATE",
        TENURE_COLUMN,
        "FLG_DWH_CUSTOMER_IS_BUSINESS",
        CUSTOMER_ID_COLUMN,
        "FK_SUBSCRIPTIONS",
        "PRODUCT_GROUP",
        "PRODUCT_SUBCATEGORY",
        "mrr",
        "ORIGIN_SUBGROUP",
        "ORIGIN_GROUP",
    }
    if include_target:
        required_columns.add(TARGET_COLUMN)

    missing = sorted(required_columns - set(invoice_data.columns))
    if missing:
        raise FeatureValidationError(f"Feature input is missing columns: {missing}.")

    invalid_tenures = sorted(
        set(invoice_data[TENURE_COLUMN].dropna().unique()) - VALID_TENURES
    )
    if invalid_tenures:
        raise FeatureValidationError(
            f"Feature input contains unsupported tenure values: {invalid_tenures}."
        )

    if include_target:
        target_counts = invoice_data.groupby(CUSTOMER_ID_COLUMN)[
            TARGET_COLUMN
        ].nunique()
        conflicting_ids = target_counts.index[target_counts > 1]
        if len(conflicting_ids):
            raise FeatureValidationError(
                "Feature input contains customers with conflicting targets: "
                f"{conflicting_ids.tolist()}."
            )


def _monthly_features(invoice_data: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.DataFrame(
        invoice_data.groupby([CUSTOMER_ID_COLUMN, TENURE_COLUMN])
        .agg(
            rows=("PK_INVOICES", "size"),
            invoices=("PK_INVOICES", "nunique"),
            subscriptions=("FK_SUBSCRIPTIONS", "nunique"),
            product_groups=("PRODUCT_GROUP", "nunique"),
            product_subcategories=("PRODUCT_SUBCATEGORY", "nunique"),
            mrr_sum=("mrr", "sum"),
            mrr_mean=("mrr", "mean"),
            mrr_min=("mrr", "min"),
            mrr_max=("mrr", "max"),
            negative_mrr_rows=("_negative_mrr_flag", "sum"),
            negative_mrr_total=("_negative_mrr_value", "sum"),
        )
        .unstack(TENURE_COLUMN)
    )
    monthly.columns = [f"{metric}_t{int(tenure)}" for metric, tenure in monthly.columns]

    expected_columns = [
        f"{metric}_t{tenure}"
        for metric in (
            "rows",
            "invoices",
            "subscriptions",
            "product_groups",
            "product_subcategories",
            "mrr_sum",
            "mrr_mean",
            "mrr_min",
            "mrr_max",
            "negative_mrr_rows",
            "negative_mrr_total",
        )
        for tenure in sorted(VALID_TENURES)
    ]
    return monthly.reindex(columns=expected_columns, fill_value=0).fillna(0)


def _subscription_continuity(invoice_data: pd.DataFrame) -> pd.DataFrame:
    presence = invoice_data[
        [CUSTOMER_ID_COLUMN, TENURE_COLUMN, "FK_SUBSCRIPTIONS"]
    ].drop_duplicates()
    counts = (
        presence.groupby([CUSTOMER_ID_COLUMN, TENURE_COLUMN])
        .size()
        .unstack(TENURE_COLUMN, fill_value=0)
        .reindex(columns=sorted(VALID_TENURES), fill_value=0)
    )
    counts = counts.set_axis(
        [
            f"subscription_count_t{tenure}"
            for tenure in sorted(VALID_TENURES)
        ],
        axis="columns",
    )

    subscriptions_t0 = presence.loc[
        presence[TENURE_COLUMN] == 0,
        [CUSTOMER_ID_COLUMN, "FK_SUBSCRIPTIONS"],
    ]
    subscriptions_t1 = presence.loc[
        presence[TENURE_COLUMN] == 1,
        [CUSTOMER_ID_COLUMN, "FK_SUBSCRIPTIONS"],
    ]
    retained = (
        subscriptions_t0.merge(
            subscriptions_t1,
            on=[CUSTOMER_ID_COLUMN, "FK_SUBSCRIPTIONS"],
            how="inner",
        )
        .groupby(CUSTOMER_ID_COLUMN)
        .size()
        .rename("retained_subscriptions_t1")
    )

    continuity = counts.join(retained, how="left").fillna(0)
    continuity["retained_subscriptions_t1"] = continuity[
        "retained_subscriptions_t1"
    ].astype(int)
    continuity["added_subscriptions_t1"] = (
        continuity["subscription_count_t1"] - continuity["retained_subscriptions_t1"]
    )
    continuity["removed_subscriptions_t1"] = (
        continuity["subscription_count_t0"] - continuity["retained_subscriptions_t1"]
    )
    union_count = (
        continuity["subscription_count_t0"]
        + continuity["subscription_count_t1"]
        - continuity["retained_subscriptions_t1"]
    )
    continuity["subscription_jaccard_t0_t1"] = continuity[
        "retained_subscriptions_t1"
    ].div(union_count.where(union_count > 0, 1))
    return continuity.drop(columns=["subscription_count_t0", "subscription_count_t1"])


def _product_presence(invoice_data: pd.DataFrame) -> pd.DataFrame:
    """Create product-presence features"""
    presence = pd.crosstab(
        invoice_data[CUSTOMER_ID_COLUMN],
        invoice_data["PRODUCT_SUBCATEGORY"],
    ).clip(upper=1)
    presence.columns = [
        f"{PRODUCT_FEATURE_PREFIX}{value}" for value in presence.columns
    ]
    presence.columns.name = None
    return presence.astype("int8")


def build_customer_features(
    invoice_data: pd.DataFrame,
    *,
    include_target: bool,
) -> pd.DataFrame:
    """
    Aggregate invoice-product observations to one row per customer.

    The input should already have passed through ``clean_training_data`` (for
    training) or validation in ``load_datasets``. Dates remain metadata for
    temporal splitting and are not included automatically as model features.
    """

    _validate_feature_input(invoice_data, include_target)

    prepared = invoice_data.copy()
    prepared["_negative_mrr_flag"] = prepared["mrr"].lt(0).astype("int8")
    prepared["_negative_mrr_value"] = prepared["mrr"].where(prepared["mrr"].lt(0), 0.0)
    prepared["ORIGIN_GROUP"] = prepared["ORIGIN_GROUP"].fillna("Missing")
    prepared["ORIGIN_SUBGROUP"] = prepared["ORIGIN_SUBGROUP"].fillna("Missing")

    aggregations: dict[str, tuple[str, str | object]] = {
        "row_count": ("PK_INVOICES", "size"),
        "invoice_count": ("PK_INVOICES", "nunique"),
        "subscription_count": ("FK_SUBSCRIPTIONS", "nunique"),
        "product_group_count": ("PRODUCT_GROUP", "nunique"),
        "product_subcategory_count": ("PRODUCT_SUBCATEGORY", "nunique"),
        "total_mrr": ("mrr", "sum"),
        "mean_mrr": ("mrr", "mean"),
        "min_mrr": ("mrr", "min"),
        "max_mrr": ("mrr", "max"),
        "negative_mrr_rows": ("_negative_mrr_flag", "sum"),
        "negative_mrr_total": ("_negative_mrr_value", "sum"),
        "is_business": ("FLG_DWH_CUSTOMER_IS_BUSINESS", "max"),
        CUSTOMER_START_COLUMN: ("CUSTOMER_START_DATE", "min"),
        "origin_group": ("ORIGIN_GROUP", "first"),
        "origin_subgroup": ("ORIGIN_SUBGROUP", "first"),
    }
    if include_target:
        aggregations[TARGET_COLUMN] = (TARGET_COLUMN, "first")

    customer_features = prepared.groupby(CUSTOMER_ID_COLUMN).agg(**aggregations)
    customer_features = customer_features.join(
        [
            _monthly_features(prepared),
            _subscription_continuity(prepared),
            _product_presence(prepared),
        ],
        how="left",
    )

    for metric in (
        "rows",
        "invoices",
        "subscriptions",
        "product_groups",
        "product_subcategories",
        "mrr_sum",
        "negative_mrr_rows",
        "negative_mrr_total",
    ):
        customer_features[f"{metric}_delta_t1_t0"] = (
            customer_features[f"{metric}_t1"] - customer_features[f"{metric}_t0"]
        )

    customer_features["observed_t0"] = customer_features["rows_t0"].gt(0).astype("int8")
    customer_features["observed_t1"] = customer_features["rows_t1"].gt(0).astype("int8")

    if not customer_features.index.is_unique:
        raise FeatureValidationError("Customer feature index is not unique.")
    if len(customer_features) != invoice_data[CUSTOMER_ID_COLUMN].nunique():
        raise FeatureValidationError("Customer aggregation changed the entity count.")

    product_columns = customer_features.columns.str.startswith(PRODUCT_FEATURE_PREFIX)
    customer_features.loc[:, product_columns] = customer_features.loc[
        :, product_columns
    ].fillna(0)
    return customer_features.sort_index()


def align_validation_features(
    training_customers: pd.DataFrame,
    validation_customers: pd.DataFrame,
) -> pd.DataFrame:
    """Align scoring features to the columns learned from training data.

    Product categories present only in validation are intentionally discarded;
    categories absent from validation are added as zero-valued presence flags.
    """

    expected_columns = [
        column for column in training_customers.columns if column != TARGET_COLUMN
    ]
    required_non_product = {
        column
        for column in expected_columns
        if not column.startswith(PRODUCT_FEATURE_PREFIX)
    }
    missing_required = sorted(required_non_product - set(validation_customers.columns))
    if missing_required:
        raise FeatureValidationError(
            f"Validation customer features are missing required columns: {missing_required}."
        )

    aligned = validation_customers.reindex(columns=expected_columns, fill_value=0)
    return aligned


def model_inputs(
    customer_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a training customer table into model features, target, and cohort."""

    required = {TARGET_COLUMN, CUSTOMER_START_COLUMN}
    missing = sorted(required - set(customer_features.columns))
    if missing:
        raise FeatureValidationError(f"Customer table is missing columns: {missing}.")

    X = customer_features.drop(columns=[TARGET_COLUMN, CUSTOMER_START_COLUMN])
    y = customer_features[TARGET_COLUMN].astype(bool).rename(TARGET_COLUMN)
    cohorts = pd.to_datetime(customer_features[CUSTOMER_START_COLUMN]).dt.to_period("M")
    cohorts = cohorts.rename("customer_start_month")
    return X, y, cohorts


def scoring_inputs(customer_features: pd.DataFrame) -> pd.DataFrame:
    """Remove non-predictive metadata from an aligned validation customer table."""

    if CUSTOMER_START_COLUMN not in customer_features:
        raise FeatureValidationError(
            f"Customer table is missing {CUSTOMER_START_COLUMN!r}."
        )
    return customer_features.drop(columns=CUSTOMER_START_COLUMN)
