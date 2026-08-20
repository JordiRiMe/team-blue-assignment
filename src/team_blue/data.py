"""Load, validate, and clean the invoice-level churn datasets.

This module owns the boundary between the source CSV files and the rest of the
project. It intentionally does not aggregate rows to customer-level features;
that transformation belongs in ``features.py``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from team_blue.typing import (
    ConflictingTargetPolicy,
    DatasetKind,
    InvalidTenurePolicy,
)

CUSTOMER_ID_COLUMN = "FK_DWH_CUSTOMERS"
TARGET_COLUMN = "noise"
TENURE_COLUMN = "TENURE"
VALID_TENURES = frozenset({0, 1})

FEATURE_COLUMNS = (
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
)
TRAINING_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
DATE_COLUMNS = ("FK_DATE_INVOICE", "CUSTOMER_START_DATE")


class DataValidationError(ValueError):
    """Raised when a source dataset violates an expected data contract."""


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved locations of the two source datasets."""

    training: Path
    validation: Path


@dataclass(frozen=True)
class CleaningReport:
    """Audit information for a training-data cleaning operation."""

    invalid_tenure_policy: InvalidTenurePolicy
    conflicting_target_policy: ConflictingTargetPolicy
    input_rows: int
    output_rows: int
    input_customers: int
    output_customers: int
    invalid_tenure_rows: int
    invalid_tenure_customers: int
    conflicting_target_customers: int

    @property
    def removed_rows(self) -> int:
        """Number of rows removed by all cleaning rules."""

        return self.input_rows - self.output_rows

    @property
    def removed_customers(self) -> int:
        """Number of customer IDs removed by all cleaning rules."""

        return self.input_customers - self.output_customers

    def as_dict(self) -> dict[str, int | str]:
        """Return a serialization-friendly representation for logs/notebooks."""

        return {
            "invalid_tenure_policy": self.invalid_tenure_policy,
            "conflicting_target_policy": self.conflicting_target_policy,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "removed_rows": self.removed_rows,
            "input_customers": self.input_customers,
            "output_customers": self.output_customers,
            "removed_customers": self.removed_customers,
            "invalid_tenure_rows": self.invalid_tenure_rows,
            "invalid_tenure_customers": self.invalid_tenure_customers,
            "conflicting_target_customers": self.conflicting_target_customers,
        }


@dataclass(frozen=True)
class InvoiceDatasets:
    """Loaded training and validation frames plus the training cleaning audit."""

    training: pd.DataFrame
    validation: pd.DataFrame
    cleaning_report: CleaningReport | None


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml``.

    Parameters
    ----------
    start:
        Directory from which to begin searching. The current working directory
        is used by default, which supports notebooks launched from either the
        repository root or ``notebooks/``.
    """

    start_path = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    if start_path.is_file():
        start_path = start_path.parent

    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    module_root = Path(__file__).resolve().parents[2]
    if (module_root / "pyproject.toml").is_file():
        return module_root

    raise FileNotFoundError(
        f"Could not find a project root containing pyproject.toml from {start_path}."
    )


def resolve_dataset_paths(data_dir: str | Path | None = None) -> DatasetPaths:
    """Resolve the conventional training and validation CSV paths."""

    directory = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else find_project_root() / "data"
    )
    return DatasetPaths(
        training=directory / "training.csv",
        validation=directory / "validation.csv",
    )


def _expected_columns(dataset_kind: DatasetKind) -> tuple[str, ...]:
    if dataset_kind == "training":
        return TRAINING_COLUMNS
    if dataset_kind == "validation":
        return FEATURE_COLUMNS
    raise ValueError(f"Unknown dataset kind: {dataset_kind!r}")


def _csv_dtypes(dataset_kind: DatasetKind) -> Mapping[Hashable, str]:
    dtypes: dict[Hashable, str] = {
        "PK_INVOICES": "int64",
        "FK_DATE_INVOICE": "int64",
        "CUSTOMER_START_DATE": "int64",
        TENURE_COLUMN: "int64",
        "FLG_DWH_CUSTOMER_IS_BUSINESS": "int8",
        CUSTOMER_ID_COLUMN: "int64",
        "FK_SUBSCRIPTIONS": "int64",
        "PRODUCT_GROUP": "string",
        "PRODUCT_SUBCATEGORY": "string",
        "mrr": "float64",
        "ORIGIN_SUBGROUP": "string",
        "ORIGIN_GROUP": "string",
    }
    if dataset_kind == "training":
        dtypes[TARGET_COLUMN] = "boolean"
    return dtypes


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in DATE_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(result[column].dtype):
            try:
                result[column] = pd.to_datetime(
                    result[column].astype("string"),
                    format="%Y%m%d",
                    errors="raise",
                )
            except (TypeError, ValueError) as error:
                raise DataValidationError(
                    f"Column {column!r} contains values that are not valid YYYYMMDD dates."
                ) from error
    return result


def validate_invoice_data(df: pd.DataFrame, dataset_kind: DatasetKind) -> None:
    """Validate schema and invariant fields without changing the input frame.

    Missing acquisition fields and negative MRR values are accepted because
    both occur in the supplied data and have plausible business meanings.
    Invalid tenure values and conflicting customer targets are handled by the
    explicit training cleaning policies rather than hidden inside validation.
    """

    expected = set(_expected_columns(dataset_kind))
    actual = set(df.columns)
    missing_columns = sorted(expected - actual)
    unexpected_columns = sorted(actual - expected)
    if missing_columns or unexpected_columns:
        raise DataValidationError(
            "Unexpected dataset schema. "
            f"Missing columns: {missing_columns or 'none'}; "
            f"unexpected columns: {unexpected_columns or 'none'}."
        )

    nullable_columns = {"ORIGIN_SUBGROUP", "ORIGIN_GROUP"}
    required_columns = expected - nullable_columns
    missing_required = df[list(required_columns)].isna().sum()
    missing_required = missing_required.loc[missing_required > 0]
    if not missing_required.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_required.items()
        )
        raise DataValidationError(
            f"Required columns contain missing values: {details}."
        )

    business_values = set(df["FLG_DWH_CUSTOMER_IS_BUSINESS"].unique())
    if not business_values <= {0, 1}:
        raise DataValidationError(
            "FLG_DWH_CUSTOMER_IS_BUSINESS must contain only 0 or 1; "
            f"found {sorted(business_values)}."
        )

    if not np.isfinite(df["mrr"].to_numpy(dtype=float)).all():
        raise DataValidationError("mrr must contain only finite numeric values.")

    if dataset_kind == "training":
        target_values = set(df[TARGET_COLUMN].unique())
        if not target_values <= {False, True}:
            raise DataValidationError(
                f"{TARGET_COLUMN} must be binary; found {sorted(target_values)}."
            )

    # Parse into a temporary frame so invalid source dates fail validation even
    # when the caller asks read_invoice_csv to preserve their integer encoding.
    _parse_date_columns(df[list(DATE_COLUMNS)])


def read_invoice_csv(
    path: str | Path,
    dataset_kind: DatasetKind,
    *,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Read one source CSV with stable dtypes and validate its data contract."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, dtype=_csv_dtypes(dataset_kind))
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            f"Could not parse {csv_path.name}: {error}"
        ) from error

    validate_invoice_data(frame, dataset_kind)
    return _parse_date_columns(frame) if parse_dates else frame


def _conflicting_target_ids(training: pd.DataFrame) -> pd.Index:
    target_counts = training.groupby(CUSTOMER_ID_COLUMN)[TARGET_COLUMN].nunique()
    return target_counts.index[target_counts > 1]


def clean_training_data(
    training: pd.DataFrame,
    *,
    invalid_tenure_policy: InvalidTenurePolicy = "drop_rows",
    conflicting_target_policy: ConflictingTargetPolicy = "drop_customers",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply explicit, auditable cleaning rules to training invoice rows.

    ``drop_rows`` is the primary tenure policy selected after EDA. The optional
    ``drop_customers`` policy supports the planned sensitivity analysis without
    duplicating cleaning logic elsewhere.
    """

    valid_invalid_policies = {"drop_rows", "drop_customers", "raise"}
    if invalid_tenure_policy not in valid_invalid_policies:
        raise ValueError(
            f"invalid_tenure_policy must be one of {sorted(valid_invalid_policies)}."
        )
    valid_target_policies = {"drop_customers", "raise"}
    if conflicting_target_policy not in valid_target_policies:
        raise ValueError(
            f"conflicting_target_policy must be one of {sorted(valid_target_policies)}."
        )

    validate_invoice_data(training, "training")
    input_rows = len(training)
    input_customers = training[CUSTOMER_ID_COLUMN].nunique()

    invalid_tenure_mask = ~training[TENURE_COLUMN].isin(VALID_TENURES)
    invalid_tenure_ids = training.loc[invalid_tenure_mask, CUSTOMER_ID_COLUMN].unique()
    conflict_ids = _conflicting_target_ids(training)

    if invalid_tenure_mask.any() and invalid_tenure_policy == "raise":
        invalid_values = sorted(
            training.loc[invalid_tenure_mask, TENURE_COLUMN].unique()
        )
        raise DataValidationError(
            f"Found {int(invalid_tenure_mask.sum())} rows across "
            f"{len(invalid_tenure_ids)} customers with invalid tenure values: "
            f"{invalid_values}."
        )
    if len(conflict_ids) and conflicting_target_policy == "raise":
        raise DataValidationError(
            f"Found {len(conflict_ids)} customers with conflicting target values: "
            f"{conflict_ids.tolist()}."
        )

    cleaned = training.copy()
    if invalid_tenure_policy == "drop_rows":
        cleaned = cleaned.loc[~invalid_tenure_mask]
    elif invalid_tenure_policy == "drop_customers":
        cleaned = cleaned.loc[~cleaned[CUSTOMER_ID_COLUMN].isin(invalid_tenure_ids)]

    if conflicting_target_policy == "drop_customers":
        cleaned = cleaned.loc[~cleaned[CUSTOMER_ID_COLUMN].isin(conflict_ids)]

    cleaned = cleaned.reset_index(drop=True)
    if not cleaned[TENURE_COLUMN].isin(VALID_TENURES).all():
        raise DataValidationError("Invalid tenure values remain after cleaning.")
    if (_conflicting_target_ids(cleaned).size) > 0:
        raise DataValidationError("Conflicting customer targets remain after cleaning.")

    report = CleaningReport(
        invalid_tenure_policy=invalid_tenure_policy,
        conflicting_target_policy=conflicting_target_policy,
        input_rows=input_rows,
        output_rows=len(cleaned),
        input_customers=input_customers,
        output_customers=cleaned[CUSTOMER_ID_COLUMN].nunique(),
        invalid_tenure_rows=int(invalid_tenure_mask.sum()),
        invalid_tenure_customers=len(invalid_tenure_ids),
        conflicting_target_customers=len(conflict_ids),
    )
    return cleaned, report


def load_datasets(
    data_dir: str | Path | None = None,
    *,
    parse_dates: bool = True,
    clean_training: bool = True,
    invalid_tenure_policy: InvalidTenurePolicy = "drop_rows",
    conflicting_target_policy: ConflictingTargetPolicy = "drop_customers",
) -> InvoiceDatasets:
    """Load both source datasets and optionally clean the training frame."""

    paths = resolve_dataset_paths(data_dir)
    training = read_invoice_csv(
        paths.training,
        "training",
        parse_dates=parse_dates,
    )
    validation = read_invoice_csv(
        paths.validation,
        "validation",
        parse_dates=parse_dates,
    )

    invalid_validation_tenure = ~validation[TENURE_COLUMN].isin(VALID_TENURES)
    if invalid_validation_tenure.any():
        invalid_values = sorted(
            validation.loc[invalid_validation_tenure, TENURE_COLUMN].unique()
        )
        raise DataValidationError(
            "Validation contains tenure values outside the supported first-two-month "
            f"window: {invalid_values}."
        )

    report = None
    if clean_training:
        training, report = clean_training_data(
            training,
            invalid_tenure_policy=invalid_tenure_policy,
            conflicting_target_policy=conflicting_target_policy,
        )

    return InvoiceDatasets(
        training=training,
        validation=validation,
        cleaning_report=report,
    )
