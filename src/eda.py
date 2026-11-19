"""Exploratory Data Analysis (Follow-up #1) statistics and publication figures.

Strictly computes feature-target associations on the development split (2010-2021)
to avoid any test-set peeking prior to Follow-up #2 modeling, while visualizing the
overall 2010-2024 temporal structure and season-based train/test boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features import (
    ALL_FEATURES,
    CV_VAL_SEASONS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
)
from src.utils import (
    FIGURES_F1_DIR,
    PALETTE,
    RANDOM_STATE,
    REPORTS_DIR,
    configure_plot_style,
    get_logger,
    save_json,
)

logger = get_logger("f1_ml.eda")

GROUP_COLORS: dict[str, str] = {
    "A_qualifying": PALETTE["blue"],
    "B_form": PALETTE["orange"],
    "C_track": PALETTE["green"],
}

GROUP_SHORT_NAMES: dict[str, str] = {
    "A_qualifying": "Group A: Qualifying & Grid",
    "B_form": "Group B: Rolling Form",
    "C_track": "Group C: Circuit History",
}


def _wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Compute 95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_univariate_statistics(train_dev: pd.DataFrame) -> pd.DataFrame:
    """Compute univariate predictive signal, hypothesis tests, and VIF on train_dev."""
    y = train_dev["top10"].values
    X_raw = train_dev[ALL_FEATURES].copy()
    medians = X_raw.median()
    X_imp = X_raw.fillna(medians)

    # Mutual information (fixed seed for exact reproducibility)
    mi_vals = mutual_info_classif(
        X_imp.values, y, discrete_features=False, random_state=RANDOM_STATE
    )

    # Standardize for numerically stable VIF computation
    X_std = (X_imp - X_imp.mean()) / X_imp.std().replace(0.0, 1.0)
    X_std_arr = X_std.values
    vifs = [
        float(variance_inflation_factor(X_std_arr, i))
        for i in range(X_std_arr.shape[1])
    ]

    feat_to_group = {
        feat: grp for grp, cols in FEATURE_GROUPS.items() for feat in cols
    }

    rows = []
    for idx, feat in enumerate(ALL_FEATURES):
        vals = X_imp[feat].values
        pos = vals[y == 1]
        neg = vals[y == 0]
        raw_auc = float(roc_auc_score(y, vals))
        oriented_auc = max(raw_auc, 1.0 - raw_auc)
        direction = "+" if raw_auc >= 0.5 else "-"
        rho, rho_p = spearmanr(vals, y)
        _, mwu_p = mannwhitneyu(pos, neg, alternative="two-sided")

        rows.append(
            {
                "feature": feat,
                "label": FEATURE_LABELS[feat],
                "group": feat_to_group[feat],
                "missing_count_train": int(X_raw[feat].isna().sum()),
                "mean_top10": float(np.mean(pos)),
                "mean_non_top10": float(np.mean(neg)),
                "spearman_r": float(rho),
                "spearman_p": float(rho_p),
                "univariate_auc": oriented_auc,
                "direction": direction,
                "mutual_info_nats": float(mi_vals[idx]),
                "mannwhitney_p": float(mwu_p),
                "vif": vifs[idx],
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False).reset_index(drop=True)
    return stats_df


def plot_fig01_eras_and_split(audit: dict[str, Any], out_dir: Path) -> Path:
    """Figure 1: Historical scoring eras & 2010-2024 temporal train/CV/test split."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.0, 1.45]})

    # Panel A: Scoring rules across F1 eras
    ax = axes[0]
    eras = pd.DataFrame(audit["scoring_eras"])
    x = np.arange(len(eras))
    width = 0.36
    bars1 = ax.bar(
        x - width / 2,
        eras["points_scoring_rate"] * 100,
        width,
        color=PALETTE["blue"],
        label="Actual Points-Scoring Rate (%)",
        alpha=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        eras["top10_rate"] * 100,
        width,
        color=PALETTE["sky"],
        label="Top-10 Finish Rate (%)",
        alpha=0.85,
    )
    ax.axhline(50, color=PALETTE["gray"], linestyle=":", linewidth=1.0, label="50% Balanced Target")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["1950-59\n(Top 5)", "1960-02\n(Top 6)", "2003-09\n(Top 8)", "2010-24\n(Top 10)"],
        fontsize=9.5,
    )
    ax.set_ylabel("Share of Driver Entries (%)")
    ax.set_ylim(0, 62)
    ax.set_title("A. Scoring Rule Regimes Across F1 History")
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    # Highlight 2010-2024 era
    ax.axvspan(2.55, 3.45, color=PALETTE["green"], alpha=0.10, zorder=0)
    ax.text(3.0, 56.5, "Selected\nEra", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["green"])
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel B: 2010-2024 season-by-season timeline & split boundaries
    ax = axes[1]
    seas = pd.DataFrame(audit["season_summary"])
    years = seas["year"].values

    # Shaded temporal zones
    ax.axvspan(2009.5, 2016.5, color=PALETTE["sky"], alpha=0.10, label="Core Train Only (2010-16)")
    ax.axvspan(2016.5, 2021.5, color=PALETTE["orange"], alpha=0.12, label="Expanding-Window CV Val (2017-21)")
    ax.axvspan(2021.5, 2024.5, color=PALETTE["vermillion"], alpha=0.11, label="Held-Out Test: Ground-Effect Era (2022-24)")
    ax.axvline(2021.5, color=PALETTE["vermillion"], linestyle="--", linewidth=1.6)

    ax.plot(
        years,
        seas["grid10_clean_acc"] * 100,
        marker="o",
        linewidth=2.2,
        color=PALETTE["blue"],
        label="Scrubbed Baseline (grid <= 10) Acc (%)",
    )
    ax.plot(
        years,
        seas["top10_rate"] * 100,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color=PALETTE["green"],
        label="Top-10 Target Rate (%)",
    )
    ax.plot(
        years,
        seas["dnf_rate"] * 100,
        marker="^",
        linewidth=1.6,
        linestyle=":",
        color=PALETTE["purple"],
        label="Race DNF / Attrition Rate (%)",
    )

    ax.set_xticks(years)
    ax.set_xticklabels([str(y)[2:] for y in years], fontsize=9)
    ax.set_xlabel("Season (20XX)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(5, 96)
    ax.set_title("B. Season Dynamics & Strict Temporal Split (2010-2024)")
    ax.text(2015.5, 90.0, "Development (n=5,077)", ha="center", fontsize=8.8, fontweight="bold", color=PALETTE["dark"])
    ax.text(2023.0, 90.0, "Test (n=1,359)", ha="center", fontsize=8.8, fontweight="bold", color=PALETTE["vermillion"])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=8.0,
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    out_path = out_dir / "fig01_scoring_eras_and_temporal_split.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig02_grid_probability_and_penalties(train_dev: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2: P(Top-10 | Grid) with Wilson 95% CIs + Grid-Penalty Recovery effect."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Empirical P(Top-10 | Grid) for P1..P20 + Pit Lane
    ax = axes[0]
    records = []
    for g in range(1, 21):
        sub = train_dev[(train_dev["is_pit_lane_start"] == 0) & (train_dev["grid"] == g)]
        n = len(sub)
        k = sub["top10"].sum()
        lo, hi = _wilson_ci(k, n)
        records.append({"label": f"P{g}", "pos": g, "p": k / n, "lo": lo, "hi": hi, "n": n})
    pit = train_dev[train_dev["is_pit_lane_start"] == 1]
    lo, hi = _wilson_ci(pit["top10"].sum(), len(pit))
    records.append({"label": "Pit\n(g=0)", "pos": 21.5, "p": pit["top10"].mean(), "lo": lo, "hi": hi, "n": len(pit)})
    gdf = pd.DataFrame(records)

    # Shade P8-P13 midfield uncertainty band
    ax.axvspan(7.5, 13.5, color=PALETTE["orange"], alpha=0.14, label="Midfield Transition Zone (P8-P13)")
    ax.axvline(10.5, color=PALETTE["vermillion"], linestyle="--", linewidth=1.4, label="Heuristic Cutoff (grid <= 10)")
    ax.axhline(0.50, color=PALETTE["gray"], linestyle=":", linewidth=1.0)

    # Plot grid P1..P20
    normal = gdf.iloc[:-1]
    ax.errorbar(
        normal["pos"],
        normal["p"] * 100,
        yerr=[(normal["p"] - normal["lo"]) * 100, (normal["hi"] - normal["p"]) * 100],
        fmt="o-",
        color=PALETTE["blue"],
        capsize=3,
        linewidth=2.0,
        markersize=5.5,
        label="Grid Start P(Top-10) +/- 95% CI",
    )
    # Plot Pit lane separately in vermillion
    pl = gdf.iloc[-1]
    ax.errorbar(
        [pl["pos"]],
        [pl["p"] * 100],
        yerr=[[(pl["p"] - pl["lo"]) * 100], [(pl["hi"] - pl["p"]) * 100]],
        fmt="D",
        color=PALETTE["vermillion"],
        capsize=4,
        markersize=7.5,
        label=f"Pit-Lane Start (grid=0, n={int(pl['n'])})",
    )

    ax.set_xticks(list(range(1, 21, 2)) + [10, 21.5])
    ax.set_xticklabels([f"P{i}" for i in range(1, 21, 2)] + ["P10", "Pit\n(g=0)"], fontsize=8.5)
    ax.set_xlabel("Starting Grid Position (2010-2021 Development Set)")
    ax.set_ylabel("Empirical Top-10 Finish Rate (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("A. Points Probability by Grid Slot & Pit-Lane Scrubbing")
    ax.legend(loc="upper right", fontsize=8.2)

    # Panel B: Where grid <= 10 fails — Fast car penalized outside Top-10 vs Weak car inside Top-10
    ax = axes[1]
    df_b = train_dev.copy()
    # Define pre-race constructor tier via rolling 5-race points/car tertiles on train_dev
    q33, q67 = df_b["const_roll5_pts_per_car"].quantile([0.33, 0.67])
    df_b["team_tier"] = np.where(
        df_b["const_roll5_pts_per_car"] >= q67,
        "Top-Tier Team",
        np.where(df_b["const_roll5_pts_per_car"] >= q33, "Midfield Team", "Backmarker Team"),
    )

    # Compare P11-P20 starters: Regular vs Penalized (grid_penalty_drop >= 3) by team tier
    outside = df_b[df_b["grid_clean"] > 10].copy()
    outside["scenario"] = np.where(
        outside["grid_penalty_drop"] >= 3, "Grid Penalty Drop\n(Drop >= +3)", "Qualified P11+\n(No Major Penalty)"
    )

    tiers = ["Top-Tier Team", "Midfield Team", "Backmarker Team"]
    scenarios = ["Qualified P11+\n(No Major Penalty)", "Grid Penalty Drop\n(Drop >= +3)"]
    x = np.arange(len(tiers))
    w = 0.36
    colors = [PALETTE["sky"], PALETTE["orange"]]

    for s_idx, scen in enumerate(scenarios):
        vals, errs_lo, errs_hi, ns = [], [], [], []
        for tier in tiers:
            sub = outside[(outside["team_tier"] == tier) & (outside["scenario"] == scen)]
            n = len(sub)
            k = sub["top10"].sum()
            p = k / n if n > 0 else 0.0
            lo, hi = _wilson_ci(k, n)
            vals.append(p * 100)
            errs_lo.append((p - lo) * 100)
            errs_hi.append((hi - p) * 100)
            ns.append(n)
        xpos = x + (s_idx - 0.5) * w
        ax.bar(xpos, vals, w, yerr=[errs_lo, errs_hi], capsize=3.5, color=colors[s_idx], alpha=0.9, label=scen)
        for bx, val, e_hi, n in zip(xpos, vals, errs_hi, ns):
            ax.text(bx, val + e_hi + 2.2, f"{val:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax.axhline(50, color=PALETTE["vermillion"], linestyle="--", linewidth=1.2, label="50% Decision Boundary")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=9)
    ax.set_ylabel("Top-10 Recovery Rate from P11+ (%)")
    ax.set_ylim(0, 94)
    ax.set_title("B. P11+ Recovery Rate by Team Form & Grid Penalties")
    ax.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    out_path = out_dir / "fig02_grid_vs_top10_and_pitlane_scrub.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig03_univariate_signal(stats_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3: Univariate ROC-AUC & Mutual Information ranked across the 3 factor groups."""
    configure_plot_style()
    df = stats_df.sort_values("univariate_auc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))
    colors = [GROUP_COLORS[g] for g in df["group"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Left panel: Oriented Univariate ROC-AUC
    ax = axes[0]
    ax.axvline(0.50, color=PALETTE["gray"], linestyle=":", linewidth=1.0)
    ax.hlines(y_pos, 0.50, df["univariate_auc"], color=colors, linewidth=2.2, alpha=0.85)
    ax.scatter(df["univariate_auc"], y_pos, color=colors, s=55, zorder=3)
    for y_i, row in df.iterrows():
        ax.text(
            row["univariate_auc"] + 0.006,
            y_i,
            f"{row['univariate_auc']:.3f} ({row['direction']})",
            va="center",
            fontsize=8.0,
            color=PALETTE["dark"],
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=8.8)
    ax.set_xlim(0.48, 0.94)
    ax.set_xlabel("Oriented Univariate ROC-AUC (2010-2021 Dev Set)")
    ax.set_title("A. Univariate Discriminative Power (ROC-AUC)")

    # Right panel: Mutual Information (nats)
    ax = axes[1]
    ax.barh(y_pos, df["mutual_info_nats"], color=colors, alpha=0.85, height=0.66)
    for y_i, row in df.iterrows():
        ax.text(
            row["mutual_info_nats"] + 0.004,
            y_i,
            f"{row['mutual_info_nats']:.3f}",
            va="center",
            fontsize=8.0,
        )
    ax.set_xlim(0, df["mutual_info_nats"].max() * 1.20)
    ax.set_xlabel("Mutual Information with Top-10 Target (nats)")
    ax.set_title("B. Non-Linear Dependence (Mutual Information)")

    handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_SHORT_NAMES[g])
        for g in ("A_qualifying", "B_form", "C_track")
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8.8, frameon=True)

    fig.suptitle(
        "Univariate Predictive Signal Across the Three Pre-Race Factor Groups (Train/Dev 2010-2021)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / "fig03_factor_group_signal_strength.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig04_constructor_form_interaction(
    train_dev: pd.DataFrame,
    full_mod: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Figure 4: Grid x Constructor Form interaction heatmap + lineage continuity trajectories."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel A: 2D Heatmap of P(Top-10) across Grid Band x Constructor Rolling Pts/Car Quintile
    ax = axes[0]
    tmp = train_dev.dropna(subset=["const_roll5_pts_per_car"]).copy()
    tmp["grid_band"] = pd.cut(
        tmp["grid_clean"],
        bins=[0, 5, 10, 15, 30],
        labels=["P1-P5", "P6-P10", "P11-P15", "P16+ / Pit"],
    )
    tmp["form_q"] = pd.qcut(
        tmp["const_roll5_pts_per_car"].rank(method="first"),
        q=5,
        labels=["Q1\n(Weakest)", "Q2", "Q3", "Q4", "Q5\n(Strongest)"],
    )
    piv_rate = tmp.pivot_table(index="grid_band", columns="form_q", values="top10", aggfunc="mean", observed=False) * 100
    piv_cnt = tmp.pivot_table(index="grid_band", columns="form_q", values="top10", aggfunc="count", observed=False)

    im = ax.imshow(piv_rate.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for r in range(piv_rate.shape[0]):
        for c in range(piv_rate.shape[1]):
            val = piv_rate.values[r, c]
            cnt = piv_cnt.values[r, c]
            txt_color = "white" if val > 58 else PALETTE["dark"]
            ax.text(c, r, f"{val:.0f}%\n(n={cnt})", ha="center", va="center", fontsize=8.2, color=txt_color, fontweight="bold")
    ax.set_xticks(range(piv_rate.shape[1]))
    ax.set_xticklabels(piv_rate.columns, fontsize=8.8)
    ax.set_yticks(range(piv_rate.shape[0]))
    ax.set_yticklabels(piv_rate.index, fontsize=9.2)
    ax.set_xlabel("Constructor Rolling Form Quintile (Pts/Car, Last 5 Races)")
    ax.set_ylabel("Starting Grid Band")
    ax.set_title("A. Grid x Constructor Form Interaction (Dev 2010-21)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Top-10 Finish Rate (%)", fontsize=9)

    # Panel B: Longitudinal lineage continuity across rebrands (full 2010-2024 timeline)
    ax = axes[1]
    lineages = [
        ("silverstone_aston", "Force India -> Racing Point -> Aston Martin", PALETTE["green"]),
        ("hinwil_sauber", "Sauber -> Alfa Romeo -> Kick Sauber", PALETTE["vermillion"]),
        ("faenza_rb", "Toro Rosso -> AlphaTauri -> RB F1", PALETTE["blue"]),
    ]
    for slug, label, color in lineages:
        sub = (
            full_mod[full_mod["constructor_lineage"] == slug]
            .groupby("year")["const_roll5_pts_per_car"]
            .mean()
            .reset_index()
        )
        ax.plot(sub["year"], sub["const_roll5_pts_per_car"], marker="o", linewidth=2.1, color=color, label=label)

    # Mark key rebrand years
    ax.axvline(2019, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(2019.05, 8.6, "2019: Alfa Romeo\n& Racing Point", fontsize=7.6, color=PALETTE["gray"])
    ax.axvline(2021, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(2021.05, 7.6, "2021: Aston Martin\n& Alpine", fontsize=7.6, color=PALETTE["gray"])
    ax.axvline(2024, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(2023.85, 6.2, "2024: RB &\nSauber ID revert", fontsize=7.6, ha="right", color=PALETTE["gray"])

    ax.set_xlabel("Season")
    ax.set_ylabel("Season Mean Rolling Constructor Pts/Car")
    ax.set_title("B. Continuous Rolling Form via Canonical Team Lineages")
    ax.set_ylim(-0.3, 10.0)
    ax.legend(loc="upper left", fontsize=8.0)

    fig.tight_layout()
    out_path = out_dir / "fig04_constructor_momentum_and_lineage.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig05_circuit_dynamics(train_dev: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 5: Circuit overtaking difficulty (grid-finish corr) vs attrition rate."""
    configure_plot_style()
    circuit_labels = {
        "albert_park": "Melbourne",
        "americas": "Austin (COTA)",
        "bahrain": "Bahrain",
        "baku": "Baku",
        "catalunya": "Barcelona",
        "hockenheimring": "Hockenheim",
        "hungaroring": "Hungary",
        "interlagos": "Interlagos",
        "marina_bay": "Singapore",
        "monaco": "Monaco",
        "monza": "Monza",
        "red_bull_ring": "Spielberg",
        "rodriguez": "Mexico City",
        "sepang": "Sepang",
        "shanghai": "Shanghai",
        "silverstone": "Silverstone",
        "sochi": "Sochi",
        "spa": "Spa",
        "suzuka": "Suzuka",
        "villeneuve": "Montreal",
        "yas_marina": "Abu Dhabi",
    }
    records = []
    for cid, grp in train_dev.groupby("circuitId"):
        n_races = grp["raceId"].nunique()
        if n_races < 5:
            continue
        race_corrs = []
        for _, rgrp in grp.groupby("raceId"):
            rho, _ = spearmanr(rgrp["grid_clean"], rgrp["positionOrder"])
            if np.isfinite(rho):
                race_corrs.append(rho)
        cref = str(grp["circuitRef"].iloc[0])
        short_name = circuit_labels.get(cref, grp["circuit_name"].iloc[0])
        records.append(
            {
                "circuit": short_name,
                "n_races": n_races,
                "mean_grid_corr": float(np.mean(race_corrs)),
                "dnf_rate": float(grp["is_dnf"].mean()) * 100,
                "grid10_acc": float(((grp["grid_clean"] <= 10) == grp["top10"]).mean()) * 100,
            }
        )
    cdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    sc = ax.scatter(
        cdf["mean_grid_corr"],
        cdf["dnf_rate"],
        s=cdf["n_races"] * 22,
        c=cdf["grid10_acc"],
        cmap="viridis",
        edgecolors=PALETTE["dark"],
        linewidths=0.8,
        alpha=0.90,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Baseline (grid <= 10) Accuracy (%)", fontsize=10)

    # Quadrant medians
    med_x = cdf["mean_grid_corr"].median()
    med_y = cdf["dnf_rate"].median()
    ax.axvline(med_x, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(med_y, color=PALETTE["gray"], linestyle="--", linewidth=1.0, alpha=0.7)

    # Non-overlapping curated annotation coordinates with leader lines
    offsets: dict[str, tuple[float, float]] = {
        "Baku": (10, 4),
        "Melbourne": (12, -2),
        "Monaco": (12, 4),
        "Sepang": (12, 8),
        "Singapore": (-52, -14),
        "Spa": (-34, 8),
        "Hockenheim": (-12, -18),
        "Austin (COTA)": (-68, 10),
        "Sochi": (10, -10),
        "Bahrain": (16, 16),
        "Spielberg": (-56, 16),
        "Silverstone": (14, 10),
        "Montreal": (12, 2),
        "Monza": (14, -14),
        "Hungary": (-58, 12),
        "Interlagos": (-62, -4),
        "Abu Dhabi": (-28, -26),
        "Barcelona": (22, 16),
        "Suzuka": (18, -2),
        "Mexico City": (12, 6),
        "Shanghai": (-52, -12),
    }
    bold_tracks = {"Monaco", "Baku", "Singapore", "Monza", "Barcelona", "Spa", "Melbourne"}
    for _, row in cdf.iterrows():
        dx, dy = offsets.get(row["circuit"], (10, 6))
        ax.annotate(
            row["circuit"],
            (row["mean_grid_corr"], row["dnf_rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            color=PALETTE["dark"],
            fontweight="bold" if row["circuit"] in bold_tracks else "normal",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.65, "shrinkA": 0, "shrinkB": 5},
        )

    ax.set_xlabel("Mean Historical Grid-to-Finish Spearman Correlation (Higher = Harder to Overtake)")
    ax.set_ylabel("Historical Race Attrition / DNF Rate (%)")
    ax.set_title("Circuit Structural Regimes: Overtaking Difficulty vs Attrition (Dev 2010-2021, >=5 Races)")
    ax.set_xlim(cdf["mean_grid_corr"].min() - 0.045, cdf["mean_grid_corr"].max() + 0.055)
    ax.set_ylim(cdf["dnf_rate"].min() - 2.8, cdf["dnf_rate"].max() + 2.8)

    fig.tight_layout()
    out_path = out_dir / "fig05_circuit_overtaking_vs_attrition.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig06_correlation_and_vif(
    train_dev: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Figure 6: Group-ordered Spearman correlation heatmap + top multicollinearity VIFs."""
    configure_plot_style()
    X_imp = train_dev[ALL_FEATURES].fillna(train_dev[ALL_FEATURES].median())
    corr = X_imp.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.0), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Panel A: Correlation heatmap ordered by factor group (A, B, C)
    ax = axes[0]
    short_ticks = [f.replace("const_", "c_").replace("drv_", "d_").replace("circuit_", "trk_") for f in ALL_FEATURES]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ALL_FEATURES)))
    ax.set_xticklabels(short_ticks, rotation=90, fontsize=7.6)
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels(short_ticks, fontsize=7.6)

    # Draw group boundary lines (8 | 11 | 7)
    n_a = len(FEATURE_GROUPS["A_qualifying"])
    n_b = len(FEATURE_GROUPS["B_form"])
    for bnd in (n_a - 0.5, n_a + n_b - 0.5):
        ax.axhline(bnd, color=PALETTE["dark"], linewidth=1.8)
        ax.axvline(bnd, color=PALETTE["dark"], linewidth=1.8)

    ax.set_title("A. Spearman Feature Correlation Matrix (Ordered A | B | C)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Spearman r", fontsize=9)

    # Panel B: Top-12 Variance Inflation Factors (motivating L1/L2 Regularization)
    ax = axes[1]
    top_vif = stats_df.sort_values("vif", ascending=False).head(12).iloc[::-1]
    colors = [GROUP_COLORS[g] for g in top_vif["group"]]
    y_pos = np.arange(len(top_vif))
    ax.barh(y_pos, top_vif["vif"], color=colors, alpha=0.88, height=0.66)
    ax.axvline(5.0, color=PALETTE["orange"], linestyle="--", linewidth=1.3, label="VIF = 5 (Moderate)")
    ax.axvline(10.0, color=PALETTE["vermillion"], linestyle="--", linewidth=1.3, label="VIF = 10 (High Collinearity)")
    for y_i, val in enumerate(top_vif["vif"]):
        ax.text(val + 1.5, y_i, f"{val:.1f}", va="center", fontsize=8.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_vif["label"], fontsize=8.6)
    ax.set_xlabel("Variance Inflation Factor (VIF, Train/Dev 2010-2021)")
    ax.set_xlim(0, top_vif["vif"].max() * 1.18)
    ax.set_title("B. Within-Group Collinearity (Top-12 VIFs)")
    ax.legend(loc="lower right", fontsize=8.2)

    fig.tight_layout()
    out_path = out_dir / "fig06_correlation_matrix_and_multicollinearity.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_followup1_eda(
    mod_df: pd.DataFrame,
    audit: dict[str, Any],
    out_dir: Path = FIGURES_F1_DIR,
) -> dict[str, Any]:
    """Execute all Follow-up #1 EDA statistics and generate all 6 publication figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dev = mod_df[mod_df["year"] <= TRAIN_END_YEAR].copy()
    stats_df = compute_univariate_statistics(train_dev)
    stats_df.to_csv(metrics_dir / "followup1_univariate_stats.csv", index=False)

    # Group-level univariate summary
    group_summary = (
        stats_df.groupby("group")
        .agg(
            n_features=("feature", "count"),
            mean_auc=("univariate_auc", "mean"),
            max_auc=("univariate_auc", "max"),
            mean_mi=("mutual_info_nats", "mean"),
            max_mi=("mutual_info_nats", "max"),
            max_vif=("vif", "max"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    figures = {
        "fig01": str(plot_fig01_eras_and_split(audit, out_dir)),
        "fig02": str(plot_fig02_grid_probability_and_penalties(train_dev, out_dir)),
        "fig03": str(plot_fig03_univariate_signal(stats_df, out_dir)),
        "fig04": str(plot_fig04_constructor_form_interaction(train_dev, mod_df, out_dir)),
        "fig05": str(plot_fig05_circuit_dynamics(train_dev, out_dir)),
        "fig06": str(plot_fig06_correlation_and_vif(train_dev, stats_df, out_dir)),
    }

    # Midfield zone & grid penalty quantitative callouts
    midfield = train_dev[(train_dev["grid_clean"] >= 8) & (train_dev["grid_clean"] <= 13)]
    non_mid = train_dev[(train_dev["grid_clean"] < 8) | (train_dev["grid_clean"] > 13)]
    baseline_mid_acc = float(((midfield["grid_clean"] <= 10) == midfield["top10"]).mean())
    baseline_non_mid_acc = float(((non_mid["grid_clean"] <= 10) == non_mid["top10"]).mean())

    summary = {
        "train_dev_rows": int(len(train_dev)),
        "test_rows": int((mod_df["year"] >= TEST_START_YEAR).sum()),
        "cv_val_seasons": list(CV_VAL_SEASONS),
        "group_univariate_summary": group_summary,
        "top_5_features_by_auc": stats_df.head(5)[
            ["feature", "group", "univariate_auc", "mutual_info_nats", "spearman_r"]
        ].to_dict(orient="records"),
        "baseline_grid10_diagnostics_train_dev": {
            "overall_acc": float(((train_dev["grid_clean"] <= 10) == train_dev["top10"]).mean()),
            "midfield_P8_P13_acc": baseline_mid_acc,
            "midfield_P8_P13_share": float(len(midfield) / len(train_dev)),
            "outside_midfield_acc": baseline_non_mid_acc,
        },
        "figures": figures,
    }
    save_json(summary, metrics_dir / "followup1_eda_summary.json")
    logger.info("Follow-up #1 EDA complete. Saved 6 figures to %s", out_dir)
    return summary
