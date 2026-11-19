"""Reproducible CLI entry point for Follow-up #1 and Follow-up #2 pipelines.

Usage:
    python run_all.py --stage followup1   # Obtain, Scrub, Features, Leakage Audit, EDA
    python run_all.py --stage followup2   # Modeling, Temporal CV, 7-Subset Ablation, Eval
    python run_all.py --stage all         # End-to-end reproduction of both deliveries
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_scrubbed_base_table, load_raw_tables, run_obtain_and_scrub
from src.eda import run_followup1_eda
from src.evaluation import run_followup2_evaluation
from src.features import build_feature_dataset, save_processed_dataset, verify_no_leakage
from src.models import run_all_modeling
from src.utils import DATA_PROCESSED_DIR, ensure_directories, get_logger

logger = get_logger("f1_ml.cli")


def run_stage_1() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute Delivery 1 (Follow-up #1): Obtain, Scrub, Features, Leakage Audit & EDA."""
    logger.info("=== STARTING DELIVERY 1 (FOLLOW-UP #1): OBTAIN, SCRUB & EXPLORE ===")
    ensure_directories()
    base_df, audit = run_obtain_and_scrub(save_audit=True)
    tables = load_raw_tables()
    leakage_report = verify_no_leakage(base_df, tables["results"], tables["races"])
    mod_df = build_feature_dataset(base_df, tables["results"], tables["races"])
    save_processed_dataset(mod_df, leakage_report)
    run_followup1_eda(mod_df, audit)
    logger.info("=== DELIVERY 1 COMPLETE ===")
    return mod_df, base_df, tables


def run_stage_2(
    mod_df: pd.DataFrame | None = None,
    base_df: pd.DataFrame | None = None,
    tables: dict | None = None,
) -> None:
    """Execute Delivery 2 (Follow-up #2): Modeling, Ablation, What-If & Interpretation."""
    logger.info("=== STARTING DELIVERY 2 (FOLLOW-UP #2): MODEL & INTERPRET ===")
    ensure_directories()
    if tables is None:
        tables = load_raw_tables()
    if base_df is None:
        base_df = build_scrubbed_base_table(tables, history_start_year=1995)
    if mod_df is None:
        parquet_path = DATA_PROCESSED_DIR / "f1_modeling_2010_2024.parquet"
        csv_path = DATA_PROCESSED_DIR / "f1_modeling_2010_2024.csv"
        if parquet_path.exists():
            mod_df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            mod_df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                "Processed dataset not found. Run `--stage followup1` first."
            )

    run_all_modeling(mod_df, base_df, tables["results"], tables["races"])
    run_followup2_evaluation(mod_df)
    logger.info("=== DELIVERY 2 COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Formula 1 Machine Learning project deliveries."
    )
    parser.add_argument(
        "--stage",
        choices=("followup1", "followup2", "all"),
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    args = parser.parse_args()

    if args.stage == "followup1":
        run_stage_1()
    elif args.stage == "followup2":
        run_stage_2()
    else:
        mod_df, base_df, tables = run_stage_1()
        run_stage_2(mod_df=mod_df, base_df=base_df, tables=tables)


if __name__ == "__main__":
    main()
