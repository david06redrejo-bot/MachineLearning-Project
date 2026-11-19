"""Leakage-free temporal feature engineering for the three F1 factor groups.

Partitions all 26 pre-race features into three strictly disjoint groups:
- Group A (G_qual,  8 features): Qualifying & Starting Grid Context
- Group B (G_form, 11 features): Rolling Constructor & Driver Momentum (strictly prior races)
- Group C (G_track, 7 features): Historical Track & Circuit Characteristics (strictly prior races)

Includes an automated adversarial leakage audit (verify_no_leakage) proving
zero in-race leakage, zero same-race teammate leakage, and zero future leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import FORBIDDEN_INRACE_COLUMNS
from src.utils import DATA_PROCESSED_DIR, get_logger, save_json

logger = get_logger("f1_ml.features")

# Strict temporal split definition (by season — never random!)
MODELING_START_YEAR: int = 2010
TRAIN_END_YEAR: int = 2021        # 2010-2021: Development set (12 seasons)
TEST_START_YEAR: int = 2022       # 2022-2024: Held-out Ground-Effect Test set (3 seasons)
CV_VAL_SEASONS: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021)  # 5-fold expanding window

# Disjoint feature group specification
FEATURE_GROUPS: dict[str, list[str]] = {
    "A_qualifying": [
        "grid_clean",
        "grid_pct",
        "is_pit_lane_start",
        "qualy_pos",
        "grid_penalty_drop",
        "reached_q3",
        "qualy_gap_to_pole_pct",
        "qualy_delta_teammate_pct",
    ],
    "B_form": [
        "const_roll5_top10_rate",
        "const_roll5_pts_per_car",
        "const_roll10_avg_pos",
        "const_momentum_delta",
        "const_roll10_mech_dnf_rate",
        "const_season_pts_per_race",
        "drv_roll5_top10_rate",
        "drv_roll5_avg_pos",
        "drv_roll5_pos_gained",
        "drv_roll10_dnf_rate",
        "drv_career_starts_log",
    ],
    "C_track": [
        "circuit_alt_log",
        "circuit_hist_grid_corr",
        "circuit_hist_attrition_rate",
        "circuit_hist_top10_retention",
        "drv_circuit_hist_top10_rate",
        "drv_circuit_prior_starts",
        "const_circuit_hist_top10_rate",
    ],
}

ALL_FEATURES: list[str] = (
    FEATURE_GROUPS["A_qualifying"]
    + FEATURE_GROUPS["B_form"]
    + FEATURE_GROUPS["C_track"]
)

FEATURE_LABELS: dict[str, str] = {
    "grid_clean": "Starting Grid Position (scrubbed)",
    "grid_pct": "Normalized Grid Percentile",
    "is_pit_lane_start": "Pit-Lane Start Indicator",
    "qualy_pos": "Qualifying Rank",
    "grid_penalty_drop": "Grid Penalty Drop (Places Lost)",
    "reached_q3": "Reached Q3 Shootout",
    "qualy_gap_to_pole_pct": "Qualifying Gap to Pole (%)",
    "qualy_delta_teammate_pct": "Qualifying Gap vs Teammate (%)",
    "const_roll5_top10_rate": "Constructor Top-10 Rate (Last 5)",
    "const_roll5_pts_per_car": "Constructor Pts/Car (Last 5)",
    "const_roll10_avg_pos": "Constructor Avg Finish (Last 10)",
    "const_momentum_delta": "Constructor Momentum (3 vs 10 Races)",
    "const_roll10_mech_dnf_rate": "Constructor Mech. DNF Rate (Last 10)",
    "const_season_pts_per_race": "Constructor Season Pts/Race (Prior)",
    "drv_roll5_top10_rate": "Driver Top-10 Rate (Last 5)",
    "drv_roll5_avg_pos": "Driver Avg Finish (Last 5)",
    "drv_roll5_pos_gained": "Driver Avg Positions Gained (Last 5)",
    "drv_roll10_dnf_rate": "Driver DNF Rate (Last 10)",
    "drv_career_starts_log": "Log Driver Prior Career Starts",
    "circuit_alt_log": "Log Circuit Altitude (m)",
    "circuit_hist_grid_corr": "Circuit Grid-to-Finish Corr. (Prior)",
    "circuit_hist_attrition_rate": "Circuit Historical DNF Rate (Prior)",
    "circuit_hist_top10_retention": "Circuit P1-P10 Retention (Prior)",
    "drv_circuit_hist_top10_rate": "Driver Circuit Top-10 Rate (Prior)",
    "drv_circuit_prior_starts": "Driver Prior Starts at Circuit",
    "const_circuit_hist_top10_rate": "Constructor Circuit Top-10 Rate (Prior)",
}

# Identifiers & metadata retained for traceability and post-hoc error analysis
METADATA_COLUMNS: list[str] = [
    "resultId",
    "raceId",
    "year",
    "round",
    "date",
    "circuitId",
    "circuitRef",
    "circuit_name",
    "race_name",
    "driverId",
    "driver_name",
    "code",
    "constructorId",
    "constructor_name",
    "constructor_lineage",
    "n_entries",
    "is_sprint_weekend",
    "grid",
    "positionOrder",
    "positionText",
    "points",
    "status",
    "status_category",
    "is_dnf",
    "is_mech_dnf",
    "is_incident_dnf",
    "top10",
]


def _safe_spearman(grid_vals: pd.Series, finish_vals: pd.Series) -> float:
    """Compute Spearman rank correlation safely for a single race."""
    if len(grid_vals) < 5 or grid_vals.nunique() < 2 or finish_vals.nunique() < 2:
        return np.nan
    rho, _ = spearmanr(grid_vals, finish_vals)
    return float(rho) if np.isfinite(rho) else np.nan


def _compute_group_a_qualifying(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Group A (Qualifying & Grid Context) — 100% Saturday/pre-race."""
    out = df.copy()
    out["grid_pct"] = (out["grid_clean"] - 1.0) / np.maximum(out["n_entries"] - 1.0, 1.0)
    out["qualy_pos"] = out["qualy_pos_raw"].fillna(out["grid_clean"]).astype(float)
    # Non-negative grid drop isolates penalties/sprint drops without exact linear rank deficiency
    out["grid_penalty_drop"] = np.maximum(0.0, out["grid_clean"] - out["qualy_pos"])
    out["reached_q3"] = out["reached_q3"].fillna(0).astype(float)

    # Scale-invariant qualifying gap to pole (%) within each raceId
    pole_ms = out.groupby("raceId")["best_q_ms"].transform("min")
    raw_gap_pct = (out["best_q_ms"] - pole_ms) / pole_ms * 100.0
    # Winsorize extreme wet/107% qualifying outliers while preserving NaN for train-fold imputation
    out["qualy_gap_to_pole_pct"] = raw_gap_pct.clip(lower=0.0, upper=12.0)

    # Gap vs teammate in the exact same qualifying session (pure Saturday pace comparison)
    team_valid_sum = out.groupby(["raceId", "constructorId"])[
        "qualy_gap_to_pole_pct"
    ].transform("sum")
    team_valid_cnt = out.groupby(["raceId", "constructorId"])[
        "qualy_gap_to_pole_pct"
    ].transform("count")
    has_teammate = (team_valid_cnt >= 2) & out["qualy_gap_to_pole_pct"].notna()
    teammate_mean = np.where(
        has_teammate,
        (team_valid_sum - out["qualy_gap_to_pole_pct"]) / np.maximum(team_valid_cnt - 1, 1),
        np.nan,
    )
    out["qualy_delta_teammate_pct"] = (
        (out["qualy_gap_to_pole_pct"] - teammate_mean).clip(lower=-5.0, upper=5.0)
    )
    out["qualy_delta_teammate_pct"] = out["qualy_delta_teammate_pct"].fillna(0.0)
    return out


def _compute_group_b_form(
    df: pd.DataFrame,
    raw_results_all_years: pd.DataFrame | None = None,
    raw_races_all_years: pd.DataFrame | None = None,
    roll_short: int = 3,
    roll_main: int = 5,
    roll_long: int = 10,
) -> pd.DataFrame:
    """Compute Group B (Rolling Constructor & Driver Momentum) strictly from prior races.

    Constructor features are aggregated to the (constructor_lineage, raceId) race level
    and shifted strictly by 1 race BEFORE rolling windows so teammate outcomes from the
    same race can never leak into each other.
    """
    out = df.copy()

    # 1) Race-level constructor aggregation (one row per team per Grand Prix)
    const_race = (
        out.groupby(["constructor_lineage", "year", "round", "raceId"], as_index=False)
        .agg(
            team_entries=("driverId", "count"),
            team_top10_sum=("top10", "sum"),
            team_pts_mean=("modern_points_equiv", "mean"),
            team_pos_mean=("positionOrder", "mean"),
            team_mech_dnf_sum=("is_mech_dnf", "sum"),
        )
        .sort_values(["constructor_lineage", "year", "round", "raceId"])
        .reset_index(drop=True)
    )

    g_const = const_race.groupby("constructor_lineage", sort=False)

    # Strictly shifted race-level streams (race t only sees races <= t-1)
    s_entries = g_const["team_entries"].shift(1)
    s_top10 = g_const["team_top10_sum"].shift(1)
    s_pts = g_const["team_pts_mean"].shift(1)
    s_pos = g_const["team_pos_mean"].shift(1)
    s_mech = g_const["team_mech_dnf_sum"].shift(1)

    # Rolling sums / means grouped by constructor_lineage on already-shifted series
    const_race["_s_entries"] = s_entries
    const_race["_s_top10"] = s_top10
    const_race["_s_pts"] = s_pts
    const_race["_s_pos"] = s_pos
    const_race["_s_mech"] = s_mech

    gc = const_race.groupby("constructor_lineage", sort=False)

    r_main_top10 = gc["_s_top10"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).sum()
    )
    r_main_entries = gc["_s_entries"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).sum()
    )
    # Empirical Bayes smoothing with fixed neutral prior (0.50, pseudo-count m=2 entries)
    const_race["const_roll5_top10_rate"] = (
        r_main_top10.fillna(0.0) + 2.0 * 0.50
    ) / (r_main_entries.fillna(0.0) + 2.0)

    const_race["const_roll5_pts_per_car"] = gc["_s_pts"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).mean()
    )
    const_race["const_roll3_pts_per_car"] = gc["_s_pts"].transform(
        lambda s: s.rolling(roll_short, min_periods=1).mean()
    )
    const_race["const_roll10_pts_per_car"] = gc["_s_pts"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).mean()
    )
    const_race["const_roll10_avg_pos"] = gc["_s_pos"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).mean()
    )
    const_race["const_momentum_delta"] = (
        const_race["const_roll3_pts_per_car"] - const_race["const_roll10_pts_per_car"]
    ).fillna(0.0)

    r_long_mech = gc["_s_mech"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).sum()
    )
    r_long_entries = gc["_s_entries"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).sum()
    )
    const_race["const_roll10_mech_dnf_rate"] = (
        r_long_mech.fillna(0.0) + 2.0 * 0.10
    ) / (r_long_entries.fillna(0.0) + 2.0)

    # Current-season cumulative constructor points/car entering round r
    # At round 1 (no prior rounds this season), fall back to prior-season recent pace
    gs = const_race.groupby(["constructor_lineage", "year"], sort=False)
    season_prior_pts = gs["team_pts_mean"].transform(lambda s: s.shift(1).expanding().mean())
    const_race["const_season_pts_per_race"] = season_prior_pts.fillna(
        const_race["const_roll5_pts_per_car"]
    )

    const_feature_cols = [
        "constructor_lineage",
        "raceId",
        "const_roll5_top10_rate",
        "const_roll5_pts_per_car",
        "const_roll10_avg_pos",
        "const_momentum_delta",
        "const_roll10_mech_dnf_rate",
        "const_season_pts_per_race",
    ]
    out = out.merge(
        const_race[const_feature_cols],
        on=["constructor_lineage", "raceId"],
        how="left",
    )

    # 2) Driver-level rolling features (1 row per (driverId, raceId), shifted strictly by 1)
    out = out.sort_values(["driverId", "year", "round", "raceId"]).reset_index(drop=True)
    gd = out.groupby("driverId", sort=False)

    out["_d_s_top10"] = gd["top10"].shift(1)
    out["_d_s_pos"] = gd["positionOrder"].shift(1)
    out["_d_s_gain"] = gd["positions_gained"].shift(1)
    out["_d_s_dnf"] = gd["is_dnf"].shift(1)

    gd2 = out.groupby("driverId", sort=False)
    d_r5_top10_sum = gd2["_d_s_top10"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).sum()
    )
    d_r5_cnt = gd2["_d_s_top10"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).count()
    )
    out["drv_roll5_top10_rate"] = (
        d_r5_top10_sum.fillna(0.0) + 2.0 * 0.50
    ) / (d_r5_cnt.fillna(0.0) + 2.0)

    out["drv_roll5_avg_pos"] = gd2["_d_s_pos"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).mean()
    )
    out["drv_roll5_pos_gained"] = gd2["_d_s_gain"].transform(
        lambda s: s.rolling(roll_main, min_periods=1).mean()
    ).fillna(0.0)

    d_r10_dnf_sum = gd2["_d_s_dnf"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).sum()
    )
    d_r10_cnt = gd2["_d_s_dnf"].transform(
        lambda s: s.rolling(roll_long, min_periods=1).count()
    )
    out["drv_roll10_dnf_rate"] = (
        d_r10_dnf_sum.fillna(0.0) + 2.0 * 0.15
    ) / (d_r10_cnt.fillna(0.0) + 2.0)

    # Exact prior career starts (using full 1950+ history if provided)
    if raw_results_all_years is not None and raw_races_all_years is not None:
        all_starts = (
            raw_results_all_years[["raceId", "driverId"]]
            .merge(
                raw_races_all_years[["raceId", "year", "round"]],
                on="raceId",
                how="inner",
            )
            .drop_duplicates(subset=["raceId", "driverId"])
            .sort_values(["driverId", "year", "round", "raceId"])
        )
        all_starts["_prior_career_starts"] = all_starts.groupby("driverId").cumcount()
        out = out.merge(
            all_starts[["raceId", "driverId", "_prior_career_starts"]],
            on=["raceId", "driverId"],
            how="left",
        )
        out["drv_career_starts_log"] = np.log1p(
            out["_prior_career_starts"].fillna(0.0).astype(float)
        )
        out = out.drop(columns=["_prior_career_starts"])
    else:
        out["drv_career_starts_log"] = np.log1p(gd.cumcount().astype(float))

    out = out.drop(columns=["_d_s_top10", "_d_s_pos", "_d_s_gain", "_d_s_dnf"])
    return out


def _compute_group_c_track(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Group C (Historical Track & Circuit Characteristics) strictly from prior races."""
    out = df.copy()

    # Static physical geography
    out["circuit_alt_log"] = np.log1p(np.clip(out["alt"].fillna(0.0).astype(float), 0.0, None))

    # 1) Race-level circuit aggregates (one row per circuit per Grand Prix)
    race_records = []
    for (cid, yr, rnd, rid), grp in out.groupby(
        ["circuitId", "year", "round", "raceId"], sort=True
    ):
        top10_starters = grp[grp["grid_clean"] <= 10]
        retention = (
            float(top10_starters["top10"].mean()) if len(top10_starters) > 0 else np.nan
        )
        race_records.append(
            {
                "circuitId": cid,
                "year": yr,
                "round": rnd,
                "raceId": rid,
                "race_grid_corr": _safe_spearman(grp["grid_clean"], grp["positionOrder"]),
                "race_dnf_rate": float(grp["is_dnf"].mean()),
                "race_top10_retention": retention,
            }
        )
    circ_race = (
        pd.DataFrame(race_records)
        .sort_values(["circuitId", "year", "round", "raceId"])
        .reset_index(drop=True)
    )

    # Shift strictly by 1 race at circuitId level before expanding averages
    g_circ = circ_race.groupby("circuitId", sort=False)
    circ_race["_s_corr"] = g_circ["race_grid_corr"].shift(1)
    circ_race["_s_dnf"] = g_circ["race_dnf_rate"].shift(1)
    circ_race["_s_ret"] = g_circ["race_top10_retention"].shift(1)

    gc = circ_race.groupby("circuitId", sort=False)
    # Empirical Bayes shrinkage toward neutral domain priors (m = 2 races)
    corr_sum = gc["_s_corr"].transform(lambda s: s.expanding().sum()).fillna(0.0)
    corr_cnt = gc["_s_corr"].transform(lambda s: s.expanding().count()).fillna(0.0)
    circ_race["circuit_hist_grid_corr"] = (corr_sum + 2.0 * 0.65) / (corr_cnt + 2.0)

    dnf_sum = gc["_s_dnf"].transform(lambda s: s.expanding().sum()).fillna(0.0)
    dnf_cnt = gc["_s_dnf"].transform(lambda s: s.expanding().count()).fillna(0.0)
    circ_race["circuit_hist_attrition_rate"] = (dnf_sum + 2.0 * 0.18) / (dnf_cnt + 2.0)

    ret_sum = gc["_s_ret"].transform(lambda s: s.expanding().sum()).fillna(0.0)
    ret_cnt = gc["_s_ret"].transform(lambda s: s.expanding().count()).fillna(0.0)
    circ_race["circuit_hist_top10_retention"] = (ret_sum + 2.0 * 0.75) / (ret_cnt + 2.0)

    out = out.merge(
        circ_race[
            [
                "circuitId",
                "raceId",
                "circuit_hist_grid_corr",
                "circuit_hist_attrition_rate",
                "circuit_hist_top10_retention",
            ]
        ],
        on=["circuitId", "raceId"],
        how="left",
    )

    # 2) Driver-at-circuit historical affinity (strictly prior visits)
    out = out.sort_values(["driverId", "circuitId", "year", "round", "raceId"]).reset_index(
        drop=True
    )
    g_dc = out.groupby(["driverId", "circuitId"], sort=False)
    prior_dc_starts = g_dc.cumcount().astype(float)
    prior_dc_top10 = g_dc["top10"].transform(lambda s: s.shift(1).cumsum()).fillna(0.0)
    out["drv_circuit_prior_starts"] = prior_dc_starts
    # Empirical Bayes smoothing (m = 3 prior visits, neutral prior = 0.50)
    out["drv_circuit_hist_top10_rate"] = (prior_dc_top10 + 3.0 * 0.50) / (
        prior_dc_starts + 3.0
    )

    # 3) Constructor-lineage-at-circuit historical affinity (aggregated at race level first!)
    cc_race = (
        out.groupby(
            ["constructor_lineage", "circuitId", "year", "round", "raceId"],
            as_index=False,
        )
        .agg(
            cc_entries=("driverId", "count"),
            cc_top10_sum=("top10", "sum"),
        )
        .sort_values(["constructor_lineage", "circuitId", "year", "round", "raceId"])
        .reset_index(drop=True)
    )
    g_cc = cc_race.groupby(["constructor_lineage", "circuitId"], sort=False)
    prior_cc_entries = g_cc["cc_entries"].transform(lambda s: s.shift(1).cumsum()).fillna(0.0)
    prior_cc_top10 = g_cc["cc_top10_sum"].transform(lambda s: s.shift(1).cumsum()).fillna(0.0)
    # Empirical Bayes smoothing (m = 4 prior car-entries, neutral prior = 0.50)
    cc_race["const_circuit_hist_top10_rate"] = (prior_cc_top10 + 4.0 * 0.50) / (
        prior_cc_entries + 4.0
    )

    out = out.merge(
        cc_race[
            [
                "constructor_lineage",
                "circuitId",
                "raceId",
                "const_circuit_hist_top10_rate",
            ]
        ],
        on=["constructor_lineage", "circuitId", "raceId"],
        how="left",
    )
    return out


def build_feature_dataset(
    base_df: pd.DataFrame,
    raw_results_all_years: pd.DataFrame | None = None,
    raw_races_all_years: pd.DataFrame | None = None,
    modeling_start_year: int = MODELING_START_YEAR,
    roll_short: int = 3,
    roll_main: int = 5,
    roll_long: int = 10,
) -> pd.DataFrame:
    """Build the complete leakage-free feature matrix for the 2010-2024 period."""
    df = _compute_group_a_qualifying(base_df)
    df = _compute_group_b_form(
        df,
        raw_results_all_years=raw_results_all_years,
        raw_races_all_years=raw_races_all_years,
        roll_short=roll_short,
        roll_main=roll_main,
        roll_long=roll_long,
    )
    df = _compute_group_c_track(df)

    # Restrict strictly to modern Top-10 points era (2010-2024)
    mod = (
        df[df["year"] >= modeling_start_year]
        .sort_values(["year", "round", "date", "raceId", "grid_clean", "driverId"])
        .reset_index(drop=True)
    )

    # Tag temporal split
    mod["split"] = np.where(mod["year"] <= TRAIN_END_YEAR, "train_dev", "test")

    keep_cols = METADATA_COLUMNS + ["split"] + ALL_FEATURES
    # Deduplicate while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))
    mod = mod[keep_cols].copy()

    logger.info(
        "Built modeling feature dataset (%d-%d): %d rows x %d features "
        "(train_dev=%d, test=%d)",
        int(mod["year"].min()),
        int(mod["year"].max()),
        len(mod),
        len(ALL_FEATURES),
        int((mod["split"] == "train_dev").sum()),
        int((mod["split"] == "test").sum()),
    )
    return mod


def get_temporal_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Return train_dev DF, held-out test DF, and expanding-window seasonal CV folds."""
    train_dev = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    test_df = df[df["year"] >= TEST_START_YEAR].reset_index(drop=True)

    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for val_year in CV_VAL_SEASONS:
        tr_idx = np.where(train_dev["year"].values < val_year)[0]
        va_idx = np.where(train_dev["year"].values == val_year)[0]
        assert len(tr_idx) > 0 and len(va_idx) > 0, f"Empty CV fold for {val_year}"
        assert train_dev.iloc[tr_idx]["year"].max() < train_dev.iloc[va_idx]["year"].min()
        cv_splits.append((tr_idx, va_idx))

    return train_dev, test_df, cv_splits


def verify_no_leakage(
    base_df: pd.DataFrame,
    raw_results_all_years: pd.DataFrame | None = None,
    raw_races_all_years: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run automated adversarial assertions proving zero target/same-race/future leakage.

    Tests:
    1. Schema check: zero forbidden in-race columns in ALL_FEATURES; groups strictly disjoint.
    2. Future-outcome mutation test: invert ALL outcomes in 2022-2024 and verify every
       single feature value in 2010-2021 is bit-for-bit identical (max_abs_diff == 0).
    3. Same-race outcome mutation test: invert ALL outcomes of a single mid-season race
       (2023 Round 10) and verify every feature value in that exact race (including both
       teammates) and all prior races is bit-for-bit identical (max_abs_diff == 0).
    4. Temporal split boundary check: train max year < test min year & monotonic CV folds.
    """
    # Check 1: Schema & disjointness
    group_sets = [set(cols) for cols in FEATURE_GROUPS.values()]
    assert len(ALL_FEATURES) == sum(len(s) for s in group_sets), "Duplicate feature across groups!"
    forbidden_overlap = set(ALL_FEATURES).intersection(FORBIDDEN_INRACE_COLUMNS)
    assert not forbidden_overlap, f"Forbidden in-race columns found: {forbidden_overlap}"

    clean_mod = build_feature_dataset(
        base_df,
        raw_results_all_years=raw_results_all_years,
        raw_races_all_years=raw_races_all_years,
    )

    # Check 2: Future-outcome mutation invariance (mutate 2022-2024 outcomes)
    mut_future = base_df.copy()
    mask_future = mut_future["year"] >= TEST_START_YEAR
    mut_future.loc[mask_future, "top10"] = 1 - mut_future.loc[mask_future, "top10"]
    mut_future.loc[mask_future, "positionOrder"] = (
        mut_future.loc[mask_future, "n_entries"] + 1 - mut_future.loc[mask_future, "positionOrder"]
    )
    mut_future.loc[mask_future, "modern_points_equiv"] = 25.0 - mut_future.loc[
        mask_future, "modern_points_equiv"
    ]
    mut_future.loc[mask_future, "positions_gained"] = -mut_future.loc[
        mask_future, "positions_gained"
    ]
    mut_future.loc[mask_future, "is_dnf"] = 1 - mut_future.loc[mask_future, "is_dnf"]
    mut_future.loc[mask_future, "is_mech_dnf"] = 1 - mut_future.loc[mask_future, "is_mech_dnf"]

    mod_mut_future = build_feature_dataset(
        mut_future,
        raw_results_all_years=raw_results_all_years,
        raw_races_all_years=raw_races_all_years,
    )
    orig_past = clean_mod[clean_mod["year"] < TEST_START_YEAR][ALL_FEATURES].fillna(-9999.0)
    mut_past = mod_mut_future[mod_mut_future["year"] < TEST_START_YEAR][ALL_FEATURES].fillna(-9999.0)
    future_max_diff = float((orig_past.values - mut_past.values).__abs__().max())
    assert future_max_diff == 0.0, f"Future leakage detected! max_diff={future_max_diff}"

    # Check 3: Same-race outcome mutation invariance (mutate 2023 Round 10)
    target_year, target_round = 2023, 10
    mut_same = base_df.copy()
    mask_race = (mut_same["year"] == target_year) & (mut_same["round"] == target_round)
    assert mask_race.sum() > 0, "Probe race 2023 Round 10 not found"
    mut_same.loc[mask_race, "top10"] = 1 - mut_same.loc[mask_race, "top10"]
    mut_same.loc[mask_race, "positionOrder"] = (
        mut_same.loc[mask_race, "n_entries"] + 1 - mut_same.loc[mask_race, "positionOrder"]
    )
    mut_same.loc[mask_race, "modern_points_equiv"] = 25.0 - mut_same.loc[
        mask_race, "modern_points_equiv"
    ]
    mut_same.loc[mask_race, "positions_gained"] = -mut_same.loc[
        mask_race, "positions_gained"
    ]
    mut_same.loc[mask_race, "is_dnf"] = 1 - mut_same.loc[mask_race, "is_dnf"]
    mut_same.loc[mask_race, "is_mech_dnf"] = 1 - mut_same.loc[mask_race, "is_mech_dnf"]

    mod_mut_same = build_feature_dataset(
        mut_same,
        raw_results_all_years=raw_results_all_years,
        raw_races_all_years=raw_races_all_years,
    )
    up_to_race = (clean_mod["year"] < target_year) | (
        (clean_mod["year"] == target_year) & (clean_mod["round"] <= target_round)
    )
    orig_same = clean_mod.loc[up_to_race, ALL_FEATURES].fillna(-9999.0)
    pert_same = mod_mut_same.loc[up_to_race, ALL_FEATURES].fillna(-9999.0)
    same_race_max_diff = float((orig_same.values - pert_same.values).__abs__().max())
    assert same_race_max_diff == 0.0, f"Same-race leakage detected! max_diff={same_race_max_diff}"

    # Verify that AFTER the perturbed race, future rolling features DO change (sanity check!)
    after_race = (clean_mod["year"] == target_year) & (clean_mod["round"] == target_round + 1)
    orig_next = clean_mod.loc[after_race, FEATURE_GROUPS["B_form"]].fillna(-9999.0)
    pert_next = mod_mut_same.loc[after_race, FEATURE_GROUPS["B_form"]].fillna(-9999.0)
    next_race_diff = float((orig_next.values - pert_next.values).__abs__().max())
    assert next_race_diff > 0.0, "Rolling form failed to update after prior race!"

    # Check 4: Temporal split monotonicity
    train_dev, test_df, cv_splits = get_temporal_splits(clean_mod)
    assert train_dev["year"].max() < test_df["year"].min()

    report = {
        "status": "PASSED",
        "n_features_total": len(ALL_FEATURES),
        "group_counts": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "forbidden_overlap_count": len(forbidden_overlap),
        "future_mutation_max_abs_diff": future_max_diff,
        "same_race_mutation_max_abs_diff": same_race_max_diff,
        "next_race_rolling_sensitivity_diff": next_race_diff,
        "train_years": [int(train_dev["year"].min()), int(train_dev["year"].max())],
        "test_years": [int(test_df["year"].min()), int(test_df["year"].max())],
        "cv_val_seasons": list(CV_VAL_SEASONS),
        "train_rows": int(len(train_dev)),
        "test_rows": int(len(test_df)),
    }
    logger.info("Leakage audit PASSED: %s", report)
    return report


def save_processed_dataset(
    mod_df: pd.DataFrame,
    leakage_report: dict[str, Any],
) -> None:
    """Persist processed modeling dataset and feature schema metadata."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mod_df.to_csv(DATA_PROCESSED_DIR / "f1_modeling_2010_2024.csv", index=False)
    try:
        mod_df.to_parquet(DATA_PROCESSED_DIR / "f1_modeling_2010_2024.parquet", index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not write parquet (CSV saved successfully): %s", exc)

    metadata = {
        "feature_groups": FEATURE_GROUPS,
        "all_features": ALL_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "temporal_split": {
            "modeling_start_year": MODELING_START_YEAR,
            "train_end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR,
            "cv_val_seasons": list(CV_VAL_SEASONS),
        },
        "leakage_audit": leakage_report,
    }
    save_json(metadata, DATA_PROCESSED_DIR / "feature_metadata.json")
