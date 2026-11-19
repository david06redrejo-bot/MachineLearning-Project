"""Automated verification tests for F1 scrubbing, temporal splits, and zero leakage.

Run via:
    python -m unittest tests/test_pipeline.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FORBIDDEN_INRACE_COLUMNS, build_scrubbed_base_table, load_raw_tables
from src.features import (
    ALL_FEATURES,
    CV_VAL_SEASONS,
    FEATURE_GROUPS,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    build_feature_dataset,
    get_temporal_splits,
    verify_no_leakage,
)
from src.utils import DATA_PROCESSED_DIR


class TestF1PipelineIntegrity(unittest.TestCase):
    """End-to-end methodological and data-hygiene regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = load_raw_tables()
        cls.base_df = build_scrubbed_base_table(cls.tables, history_start_year=1995)
        cls.mod_df = build_feature_dataset(
            cls.base_df, cls.tables["results"], cls.tables["races"]
        )

    def test_period_and_target_balance(self) -> None:
        self.assertEqual(int(self.mod_df["year"].min()), 2010)
        self.assertEqual(int(self.mod_df["year"].max()), 2024)
        pos_rate = float(self.mod_df["top10"].mean())
        self.assertGreater(pos_rate, 0.45)
        self.assertLess(pos_rate, 0.50)

    def test_pit_lane_scrubbing(self) -> None:
        pit_rows = self.mod_df[self.mod_df["is_pit_lane_start"] == 1]
        self.assertGreater(len(pit_rows), 50)
        # Every pit-lane start must be mapped strictly behind the last normal grid slot
        self.assertTrue((pit_rows["grid_clean"] > pit_rows["n_entries"]).all())
        self.assertTrue((self.mod_df["grid_clean"] >= 1.0).all())

    def test_feature_groups_disjoint_and_no_inrace_columns(self) -> None:
        seen: set[str] = set()
        for grp_name, cols in FEATURE_GROUPS.items():
            for col in cols:
                self.assertNotIn(col, seen, f"Duplicate feature {col} in {grp_name}")
                self.assertNotIn(col, FORBIDDEN_INRACE_COLUMNS, f"Forbidden column {col}")
                seen.add(col)
        self.assertEqual(seen, set(ALL_FEATURES))

    def test_strict_temporal_split_monotonicity(self) -> None:
        train_dev, test_df, cv_splits = get_temporal_splits(self.mod_df)
        self.assertLess(int(train_dev["year"].max()), int(test_df["year"].min()))
        self.assertEqual(int(train_dev["year"].max()), TRAIN_END_YEAR)
        self.assertEqual(int(test_df["year"].min()), TEST_START_YEAR)
        self.assertEqual(len(cv_splits), len(CV_VAL_SEASONS))
        for (tr_idx, va_idx), val_yr in zip(cv_splits, CV_VAL_SEASONS):
            self.assertLess(
                int(train_dev.iloc[tr_idx]["year"].max()),
                int(train_dev.iloc[va_idx]["year"].min()),
            )
            self.assertEqual(int(train_dev.iloc[va_idx]["year"].unique()[0]), val_yr)

    def test_adversarial_zero_leakage(self) -> None:
        report = verify_no_leakage(
            self.base_df, self.tables["results"], self.tables["races"]
        )
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["forbidden_overlap_count"], 0)
        self.assertEqual(report["future_mutation_max_abs_diff"], 0.0)
        self.assertEqual(report["same_race_mutation_max_abs_diff"], 0.0)
        self.assertGreater(report["next_race_rolling_sensitivity_diff"], 0.0)

    def test_processed_artifacts_exist(self) -> None:
        self.assertTrue((DATA_PROCESSED_DIR / "f1_modeling_2010_2024.csv").exists())
        self.assertTrue((DATA_PROCESSED_DIR / "feature_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
