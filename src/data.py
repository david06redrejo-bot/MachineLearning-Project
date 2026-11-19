"""Data acquisition, relational auditing, and scrubbing for the F1 ML project.

Follows the Obtain & Scrub stages of the OSEMN workflow:
- Loads the 14 relational Ergast/Kaggle Formula 1 CSV tables (1950-2024).
- Documents and enforces strict domain constraints:
  * Restricts modeling target rows to the modern Top-10 points era (2010-2024),
    while preserving pre-2010 history strictly for warm-starting past-only
    driver/constructor/circuit statistics.
  * Excludes all in-race tables/variables (lap_times, pit_stops, target-race
    status, fastest lap, race time, post-race standings) from predictive features.
  * Scrubs Ergast-specific anomalies: '\\N' null tokens, pit-lane starts
    encoded as grid == 0, missing qualifying rows, and constructor rebrands
    (e.g., Sauber <-> Alfa Romeo <-> Sauber, AlphaTauri -> RB).
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED_DIR, DATA_RAW_DIR, get_logger, save_json

logger = get_logger("f1_ml.data")

KAGGLE_ZIP_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rohanrao/formula-1-world-championship-1950-2020"
)

REQUIRED_TABLES = (
    "circuits.csv",
    "constructors.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "drivers.csv",
    "driver_standings.csv",
    "qualifying.csv",
    "races.csv",
    "results.csv",
    "seasons.csv",
    "sprint_results.csv",
    "status.csv",
)

# Forbidden in-race/post-race columns that MUST NEVER enter feature matrices
FORBIDDEN_INRACE_COLUMNS: frozenset[str] = frozenset(
    {
        "position",
        "positionText",
        "positionOrder",
        "points",
        "laps",
        "time",
        "milliseconds",
        "fastestLap",
        "rank",
        "fastestLapTime",
        "fastestLapSpeed",
        "statusId",
        "status",
        "status_category",
        "is_dnf",
        "is_mech_dnf",
        "is_incident_dnf",
        "modern_points_equiv",
        "positions_gained",
        "top10",
    }
)

# Modern FIA Top-10 points schedule (introduced in 2010)
MODERN_POINTS_MAP: dict[int, float] = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}

# Canonical constructor lineage mapping across 2006-2024 team rebrands so
# rolling constructor form and circuit history remain longitudinally continuous.
# Key: Ergast constructorId -> Canonical lineage slug
CONSTRUCTOR_LINEAGE_MAP: dict[int, str] = {
    # Enstone lineage: Renault -> Lotus F1 -> Renault -> Alpine
    4: "enstone_alpine",
    208: "enstone_alpine",
    214: "enstone_alpine",
    # Faenza lineage: Toro Rosso -> AlphaTauri -> RB F1 Team
    5: "faenza_rb",
    213: "faenza_rb",
    215: "faenza_rb",
    # Silverstone lineage: Force India -> Racing Point -> Aston Martin
    10: "silverstone_aston",
    211: "silverstone_aston",
    117: "silverstone_aston",
    # Hinwil lineage: BMW Sauber -> Sauber -> Alfa Romeo -> Kick Sauber (reverted to ID 15 in 2024!)
    2: "hinwil_sauber",
    15: "hinwil_sauber",
    51: "hinwil_sauber",
    # Brackley lineage: Honda / Brawn -> Mercedes
    11: "brackley_mercedes",
    23: "brackley_mercedes",
    131: "brackley_mercedes",
    # Banbury backmarker lineage: Virgin -> Marussia -> Manor Marussia
    166: "banbury_manor",
    206: "banbury_manor",
    209: "banbury_manor",
    # Hingham backmarker lineage: Lotus -> Caterham
    205: "hingham_caterham",
    207: "hingham_caterham",
    # Stable major teams
    1: "woking_mclaren",
    3: "grove_williams",
    6: "maranello_ferrari",
    9: "milton_keynes_redbull",
    210: "kannapolis_haas",
    164: "madrid_hrt",
}

INCIDENT_STATUSES: frozenset[str] = frozenset(
    {
        "Accident",
        "Collision",
        "Collision damage",
        "Spun off",
        "Damage",
        "Debris",
        "Fatal accident",
    }
)

DNS_DNQ_DSQ_STATUSES: frozenset[str] = frozenset(
    {
        "Disqualified",
        "Excluded",
        "Withdrew",
        "Did not qualify",
        "Did not prequalify",
        "107% Rule",
        "Not restarted",
        "Not classified",
        "Illness",
        "Injury",
        "Injured",
        "Driver unwell",
        "Physical",
    }
)

_LAPPED_RE = re.compile(r"^\+\d+\s+Laps?$")


def download_raw_data(raw_dir: Path = DATA_RAW_DIR, force: bool = False) -> Path:
    """Download and extract the Kaggle F1 World Championship dataset if missing."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_TABLES if not (raw_dir / name).exists()]
    if not missing and not force:
        logger.info("All required raw CSV tables already exist in %s", raw_dir)
        return raw_dir

    zip_path = raw_dir / "f1_dataset.zip"
    logger.info("Downloading F1 dataset from Kaggle to %s ...", zip_path)
    urllib.request.urlretrieve(KAGGLE_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)
    logger.info("Extracted F1 dataset into %s", raw_dir)
    return raw_dir


def load_raw_tables(raw_dir: Path = DATA_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load core relational CSV tables with Ergast '\\N' null decoding."""
    raw_dir = Path(raw_dir)
    missing = [name for name in REQUIRED_TABLES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw CSV tables in {raw_dir}: {missing}. "
            "Run download_raw_data() first."
        )

    tables: dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_TABLES:
        key = filename.replace(".csv", "")
        tables[key] = pd.read_csv(raw_dir / filename, na_values=["\\N", ""])
    return tables


def parse_lap_time_to_ms(value: Any) -> float:
    """Parse an Ergast qualifying time string ('1:26.572' or '58.312') to ms."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text == "\\N":
        return np.nan
    try:
        if ":" in text:
            mins, secs = text.split(":", maxsplit=1)
            return (int(mins) * 60.0 + float(secs)) * 1000.0
        return float(text) * 1000.0
    except ValueError:
        return np.nan


def classify_status(status_text: Any) -> str:
    """Categorize Ergast finish status into finished / incident / mechanical / other.

    Used EXCLUSIVELY on strictly historical prior races (reliability / attrition)
    and post-hoc test error analysis — NEVER as a target-race feature.
    """
    if pd.isna(status_text):
        return "other"
    s = str(status_text).strip()
    if s == "Finished" or _LAPPED_RE.match(s):
        return "finished"
    if s in INCIDENT_STATUSES:
        return "incident_dnf"
    if s in DNS_DNQ_DSQ_STATUSES:
        return "dns_dnq_dsq"
    return "mechanical_dnf"


def map_constructor_lineage(constructor_id: int, constructor_ref: str) -> str:
    """Map an Ergast constructorId to a canonical longitudinal team lineage."""
    if constructor_id in CONSTRUCTOR_LINEAGE_MAP:
        return CONSTRUCTOR_LINEAGE_MAP[constructor_id]
    return f"team_{constructor_ref}"


def build_scrubbed_base_table(
    tables: dict[str, pd.DataFrame],
    history_start_year: int = 1995,
) -> pd.DataFrame:
    """Merge and scrub relational F1 tables chronologically from history_start_year+.

    Keeping 1995-2009 rows in the base table allows strictly past warm-starting of
    driver experience, circuit overtaking correlation, and constructor/driver
    circuit history for the 2010 season, while final modeling & evaluation strictly
    restricts to year >= 2010 (the unified Top-10 points era).
    """
    races = tables["races"][
        ["raceId", "year", "round", "circuitId", "name", "date"]
    ].rename(columns={"name": "race_name"})
    races["date"] = pd.to_datetime(races["date"])

    circuits = tables["circuits"][
        ["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt"]
    ].rename(columns={"name": "circuit_name"})

    drivers = tables["drivers"][
        ["driverId", "driverRef", "code", "forename", "surname", "dob", "nationality"]
    ].rename(columns={"nationality": "driver_nationality"})
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    drivers["dob"] = pd.to_datetime(drivers["dob"], errors="coerce")

    constructors = tables["constructors"][
        ["constructorId", "constructorRef", "name", "nationality"]
    ].rename(
        columns={
            "name": "constructor_name",
            "nationality": "constructor_nationality",
        }
    )

    status = tables["status"][["statusId", "status"]]

    quali = tables["qualifying"][
        ["raceId", "driverId", "position", "q1", "q2", "q3"]
    ].rename(columns={"position": "qualy_pos_raw"})
    # Guard against rare duplicate (raceId, driverId) rows in qualifying
    quali = quali.sort_values("qualy_pos_raw").drop_duplicates(
        subset=["raceId", "driverId"], keep="first"
    )
    for col in ("q1", "q2", "q3"):
        quali[f"{col}_ms"] = quali[col].map(parse_lap_time_to_ms)
    quali["best_q_ms"] = quali[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1)
    quali["reached_q2"] = quali["q2_ms"].notna().astype(int)
    quali["reached_q3"] = quali["q3_ms"].notna().astype(int)

    sprint_race_ids = set(tables["sprint_results"]["raceId"].unique())

    # Assemble base relational join starting from results
    base = (
        tables["results"][
            [
                "resultId",
                "raceId",
                "driverId",
                "constructorId",
                "grid",
                "position",
                "positionText",
                "positionOrder",
                "points",
                "laps",
                "statusId",
            ]
        ]
        .merge(races, on="raceId", how="inner")
        .merge(circuits, on="circuitId", how="left")
        .merge(drivers, on="driverId", how="left")
        .merge(constructors, on="constructorId", how="left")
        .merge(status, on="statusId", how="left")
        .merge(
            quali[
                [
                    "raceId",
                    "driverId",
                    "qualy_pos_raw",
                    "best_q_ms",
                    "reached_q2",
                    "reached_q3",
                ]
            ],
            on=["raceId", "driverId"],
            how="left",
        )
    )

    # Filter to history window and sort strictly chronologically
    base = base[base["year"] >= history_start_year].copy()
    base = base.sort_values(
        ["year", "round", "date", "raceId", "positionOrder", "resultId"]
    ).reset_index(drop=True)

    # Canonical constructor lineage
    base["constructor_lineage"] = [
        map_constructor_lineage(int(cid), str(cref))
        for cid, cref in zip(base["constructorId"], base["constructorRef"])
    ]

    # Race field size (pre-race entry count per Grand Prix)
    race_entries = base.groupby("raceId")["driverId"].transform("count")
    base["n_entries"] = race_entries.astype(int)
    base["is_sprint_weekend"] = base["raceId"].isin(sprint_race_ids).astype(int)

    # Scrubbing grid == 0 (pit-lane start / DNS / DNQ) -> back of grid (n_entries + 1)
    base["is_pit_lane_start"] = (base["grid"] == 0).astype(int)
    base["grid_clean"] = np.where(
        base["grid"] == 0, base["n_entries"] + 1, base["grid"]
    ).astype(float)

    # Driver age at race date (strictly pre-race biographical info)
    base["driver_age"] = (base["date"] - base["dob"]).dt.days / 365.25

    # Target & outcome fields (STRICTLY segregated — never used as features for race t!)
    base["top10"] = (base["positionOrder"] <= 10).astype(int)
    base["modern_points_equiv"] = (
        base["positionOrder"].map(MODERN_POINTS_MAP).fillna(0.0).astype(float)
    )
    base["positions_gained"] = base["grid_clean"] - base["positionOrder"].astype(float)
    base["status_category"] = base["status"].map(classify_status)
    base["is_dnf"] = (base["status_category"] != "finished").astype(int)
    base["is_mech_dnf"] = (base["status_category"] == "mechanical_dnf").astype(int)
    base["is_incident_dnf"] = (base["status_category"] == "incident_dnf").astype(int)

    logger.info(
        "Built scrubbed base table (%d-%d): %d rows across %d races (2010+: %d rows, %d races)",
        int(base["year"].min()),
        int(base["year"].max()),
        len(base),
        base["raceId"].nunique(),
        int((base["year"] >= 2010).sum()),
        int(base.loc[base["year"] >= 2010, "raceId"].nunique()),
    )
    return base


def compute_relational_audit(
    tables: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    modeling_start_year: int = 2010,
) -> dict[str, Any]:
    """Produce a comprehensive Obtain & Scrub audit report for Follow-up #1."""
    all_results = tables["results"].merge(
        tables["races"][["raceId", "year"]], on="raceId", how="inner"
    )
    mod = base_df[base_df["year"] >= modeling_start_year].copy()

    # Check raw table shapes & missingness
    table_summary = {
        name: {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
        }
        for name, df in tables.items()
    }

    # Historical scoring eras comparison
    era_stats = []
    for label, y0, y1 in [
        ("1950-1959 (Top 5)", 1950, 1959),
        ("1960-2002 (Top 6)", 1960, 2002),
        ("2003-2009 (Top 8)", 2003, 2009),
        ("2010-2024 (Top 10)", 2010, 2024),
    ]:
        sub = all_results[(all_results["year"] >= y0) & (all_results["year"] <= y1)]
        era_stats.append(
            {
                "era": label,
                "rows": int(len(sub)),
                "races": int(sub["raceId"].nunique()),
                "points_scoring_rate": float((sub["points"] > 0).mean()),
                "top10_rate": float((sub["positionOrder"] <= 10).mean()),
                "mean_field_size": float(
                    sub.groupby("raceId")["resultId"].count().mean()
                ),
            }
        )

    raw_grid_baseline_acc = float(((mod["grid"] <= 10) == mod["top10"]).mean())
    clean_grid_baseline_acc = float(((mod["grid_clean"] <= 10) == mod["top10"]).mean())

    # Season-by-season overview (2010-2024)
    season_rows = []
    for yr, grp in mod.groupby("year"):
        season_rows.append(
            {
                "year": int(yr),
                "races": int(grp["raceId"].nunique()),
                "entries": int(len(grp)),
                "mean_field_size": float(len(grp) / grp["raceId"].nunique()),
                "top10_rate": float(grp["top10"].mean()),
                "dnf_rate": float(grp["is_dnf"].mean()),
                "pit_lane_starts": int(grp["is_pit_lane_start"].sum()),
                "grid10_clean_acc": float(
                    ((grp["grid_clean"] <= 10) == grp["top10"]).mean()
                ),
            }
        )

    audit: dict[str, Any] = {
        "raw_tables": table_summary,
        "full_span_years": [int(all_results["year"].min()), int(all_results["year"].max())],
        "modeling_span_years": [int(mod["year"].min()), int(mod["year"].max())],
        "modeling_rows": int(len(mod)),
        "modeling_races": int(mod["raceId"].nunique()),
        "modeling_drivers": int(mod["driverId"].nunique()),
        "modeling_constructors_raw": int(mod["constructorId"].nunique()),
        "modeling_constructor_lineages": int(mod["constructor_lineage"].nunique()),
        "modeling_circuits": int(mod["circuitId"].nunique()),
        "target_positive_rate": float(mod["top10"].mean()),
        "target_discrepancy_posOrder_vs_points": int(
            ((mod["positionOrder"] <= 10) != (mod["points"] > 0)).sum()
        ),
        "scrubbing_findings": {
            "grid_zero_entries": int(mod["is_pit_lane_start"].sum()),
            "grid_zero_top10_finishes": int(
                mod.loc[mod["is_pit_lane_start"] == 1, "top10"].sum()
            ),
            "missing_qualifying_rows": int(mod["qualy_pos_raw"].isna().sum()),
            "missing_best_q_ms": int(mod["best_q_ms"].isna().sum()),
            "missing_circuit_alt": int(mod["alt"].isna().sum()),
            "raw_grid_le_10_accuracy": raw_grid_baseline_acc,
            "scrubbed_grid_le_10_accuracy": clean_grid_baseline_acc,
            "scrubbing_baseline_gain_pp": (clean_grid_baseline_acc - raw_grid_baseline_acc)
            * 100.0,
            "status_breakdown": mod["status_category"].value_counts().to_dict(),
        },
        "scoring_eras": era_stats,
        "season_summary": season_rows,
    }
    return audit


def run_obtain_and_scrub(
    save_audit: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute Obtain + Scrub stage and persist the audit summary."""
    download_raw_data(DATA_RAW_DIR)
    tables = load_raw_tables(DATA_RAW_DIR)
    base_df = build_scrubbed_base_table(tables, history_start_year=1995)
    audit = compute_relational_audit(tables, base_df, modeling_start_year=2010)
    if save_audit:
        save_json(audit, DATA_PROCESSED_DIR / "data_audit_followup1.json")
    return base_df, audit
