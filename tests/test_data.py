"""Tests for the invoice-data ingestion and cleaning contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from team_blue.data import (
    CUSTOMER_ID_COLUMN,
    DataValidationError,
    clean_training_data,
    load_datasets,
    read_invoice_csv,
    validate_invoice_data,
)


def _row(
    customer_id: int,
    tenure: int,
    target: bool,
    *,
    invoice_id: int | None = None,
) -> dict[str, object]:
    unique_id = invoice_id if invoice_id is not None else customer_id * 10 + tenure
    return {
        "PK_INVOICES": unique_id,
        "FK_DATE_INVOICE": 20230115,
        "CUSTOMER_START_DATE": 20230101,
        "TENURE": tenure,
        "FLG_DWH_CUSTOMER_IS_BUSINESS": 1,
        CUSTOMER_ID_COLUMN: customer_id,
        "FK_SUBSCRIPTIONS": unique_id + 100,
        "PRODUCT_GROUP": "Domain",
        "PRODUCT_SUBCATEGORY": "Domain Names",
        "mrr": 1.5,
        "ORIGIN_SUBGROUP": "PPC",
        "ORIGIN_GROUP": "Paid Traffic",
        "noise": target,
    }


def _training_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(1, 0, False),
            _row(1, 1, False),
            _row(2, -1, True),
            _row(2, 0, True),
            _row(3, 0, False),
            _row(3, 1, True),
        ]
    )


class ValidateInvoiceDataTests(unittest.TestCase):
    def test_rejects_missing_columns(self) -> None:
        frame = _training_fixture().drop(columns="mrr")

        with self.assertRaisesRegex(DataValidationError, "Missing columns"):
            validate_invoice_data(frame, "training")

    def test_allows_missing_acquisition_values_and_negative_mrr(self) -> None:
        frame = _training_fixture()
        frame.loc[0, ["ORIGIN_SUBGROUP", "ORIGIN_GROUP"]] = None
        frame.loc[0, "mrr"] = -2.5

        validate_invoice_data(frame, "training")

    def test_rejects_non_binary_business_flag(self) -> None:
        frame = _training_fixture()
        frame.loc[0, "FLG_DWH_CUSTOMER_IS_BUSINESS"] = 2

        with self.assertRaisesRegex(DataValidationError, "must contain only 0 or 1"):
            validate_invoice_data(frame, "training")


class CleanTrainingDataTests(unittest.TestCase):
    def test_default_policy_drops_invalid_rows_and_conflicting_customers(self) -> None:
        original = _training_fixture()

        cleaned, report = clean_training_data(original)

        self.assertEqual(set(cleaned[CUSTOMER_ID_COLUMN]), {1, 2})
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(set(cleaned["TENURE"]), {0, 1})
        self.assertEqual(report.invalid_tenure_rows, 1)
        self.assertEqual(report.invalid_tenure_customers, 1)
        self.assertEqual(report.conflicting_target_customers, 1)
        self.assertEqual(report.removed_rows, 3)
        self.assertEqual(report.removed_customers, 1)
        self.assertEqual(len(original), 6, "Cleaning must not mutate the input frame")

    def test_sensitivity_policy_drops_entire_invalid_tenure_customer(self) -> None:
        cleaned, report = clean_training_data(
            _training_fixture(),
            invalid_tenure_policy="drop_customers",
        )

        self.assertEqual(set(cleaned[CUSTOMER_ID_COLUMN]), {1})
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report.removed_customers, 2)

    def test_raise_policy_reports_invalid_tenure(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "invalid tenure values"):
            clean_training_data(
                _training_fixture(),
                invalid_tenure_policy="raise",
            )


class LoadingTests(unittest.TestCase):
    def test_read_invoice_csv_parses_business_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            _training_fixture().to_csv(path, index=False)

            frame = read_invoice_csv(path, "training")

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["FK_DATE_INVOICE"]))
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(frame["CUSTOMER_START_DATE"])
        )

    def test_load_datasets_returns_clean_training_and_untouched_validation(
        self,
    ) -> None:
        training = _training_fixture()
        validation = training.drop(columns="noise").query("TENURE in [0, 1]")

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            training.to_csv(data_dir / "training.csv", index=False)
            validation.to_csv(data_dir / "validation.csv", index=False)

            datasets = load_datasets(data_dir)

        self.assertEqual(len(datasets.training), 3)
        self.assertEqual(len(datasets.validation), len(validation))
        self.assertIsNotNone(datasets.cleaning_report)
