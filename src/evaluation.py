"""Interpretability, error analysis, case studies, and Follow-up #2 figures.

Generates 6 publication-quality figures in reports/figures/followup2/ and
exports structured interpretability & error-decomposition tables to reports/metrics/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.eda import GROUP_COLORS, GROUP_SHORT_NAMES
from src.features import ALL_FEATURES, FEATURE_GROUPS, FEATURE_LABELS, get_temporal_splits
from src.utils import (
    FIGURES_F2_DIR,
    MODELS_DIR,
    PALETTE,
    RANDOM_STATE,
    REPORTS_DIR,
    configure_plot_style,
    get_logger,
    load_json,
    save_json,
)

logger = get_logger("f1_ml.evaluation")

FAMILY_COLORS: dict[str, str] = {
    "Baseline": PALETTE["gray"],
    "Linear": PALETTE["blue"],
    "Probabilistic": PALETTE["purple"],
    "Instance-Based": PALETTE["sky"],
    "Non-Linear Kernel": PALETTE["vermillion"],
    "Tree": PALETTE["yellow"],
    "Tree Ensemble": PALETTE["orange"],
}


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("-", "_")
    )


def plot_fig07_model_comparison(bm_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 7: Benchmark comparison across all 12 models (CV vs Test Accuracy, ROC-AUC, Log-Loss)."""
    configure_plot_style()
    df = bm_df[bm_df["model"] != "Baseline: Majority Class"].copy()
    df = df.sort_values("test_roc_auc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))
    colors = [FAMILY_COLORS.get(f, PALETTE["gray"]) for f in df["family"]]

    grid10_acc = float(
        bm_df.loc[bm_df["model"] == "Baseline: grid <= 10 (Scrubbed)", "test_accuracy"].iloc[0]
    ) * 100.0

    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 5.4), sharey=True, gridspec_kw={"width_ratios": [1.25, 1.05, 0.95]}
    )

    # Panel A: CV vs Test Accuracy
    ax = axes[0]
    ax.axvline(grid10_acc, color=PALETTE["vermillion"], linestyle="--", linewidth=1.4, label=f"grid <= 10 ({grid10_acc:.1f}%)")
    ax.barh(y_pos + 0.16, df["test_accuracy"] * 100, height=0.32, color=colors, alpha=0.92, label="Held-Out Test (2022-24)")
    ax.barh(y_pos - 0.16, df["cv_accuracy"] * 100, height=0.32, color=colors, alpha=0.38, edgecolor=colors, linewidth=0.8, label="Temporal CV (2017-21)")
    for i, row in df.iterrows():
        ax.text(row["test_accuracy"] * 100 + 0.35, i + 0.16, f"{row['test_accuracy']*100:.1f}%", va="center", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model"], fontsize=8.8)
    ax.set_xlim(66, 86.5)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("A. Accuracy: Temporal CV vs Test")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.6)

    # Panel B: Test ROC-AUC
    ax = axes[1]
    ax.hlines(y_pos, 0.75, df["test_roc_auc"], color=colors, linewidth=2.4, alpha=0.85)
    ax.scatter(df["test_roc_auc"], y_pos, color=colors, s=62, zorder=3)
    for i, row in df.iterrows():
        ax.text(row["test_roc_auc"] + 0.004, i, f"{row['test_roc_auc']:.3f}", va="center", fontsize=8.2, fontweight="bold")
    ax.set_xlim(0.75, 0.925)
    ax.set_xlabel("Held-Out Test ROC-AUC (2022-24)")
    ax.set_title("B. Discriminative Ranking (ROC-AUC)")

    # Panel C: Test Log-Loss (lower is better; NaN for hard rule)
    ax = axes[2]
    for i, row in df.iterrows():
        ll = row["test_log_loss"]
        if np.isfinite(ll):
            ax.barh(i, ll, height=0.58, color=colors[i], alpha=0.88)
            ax.text(ll + 0.015, i, f"{ll:.3f}", va="center", fontsize=8.0)
        else:
            ax.text(0.05, i, "N/A (hard rule)", va="center", fontsize=7.8, style="italic", color=PALETTE["gray"])
    ax.set_xlim(0, max(df["test_log_loss"].dropna().max() * 1.18, 0.85))
    ax.set_xlabel("Held-Out Test Log-Loss (Lower = Better)")
    ax.set_title("C. Probabilistic Calibration (Log-Loss)")

    # Shared family legend below
    used_families = ["Baseline", "Linear", "Non-Linear Kernel", "Tree Ensemble", "Tree", "Instance-Based", "Probabilistic"]
    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in used_families]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.11, 1, 1])

    out_path = out_dir / "fig07_model_comparison_cv_vs_test.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig08_ablation_and_shapley(
    ablation_df: pd.DataFrame,
    shapley_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Figure 8: Centerpiece 7-subset Factor Group Ablation Matrix & Exact Shapley Attribution."""
    configure_plot_style()
    subset_order = ["C", "B", "A", "B+C", "A+C", "A+B", "A+B+C"]
    model_order = [
        "Logistic Regression (L2)",
        "Linear SVM",
        "RBF Kernel SVM",
        "Random Forest",
        "Gradient Boosting",
    ]
    short_models = ["LogReg\n(L2)", "Linear\nSVM", "RBF\nSVM", "Random\nForest", "Gradient\nBoosting"]

    piv_auc = ablation_df.pivot(index="subset", columns="model", values="test_roc_auc").loc[subset_order, model_order]
    piv_acc = ablation_df.pivot(index="subset", columns="model", values="test_accuracy").loc[subset_order, model_order] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: 7x5 Ablation Matrix Heatmap
    ax = axes[0]
    im = ax.imshow(piv_auc.values, cmap="YlGnBu", vmin=0.66, vmax=0.895, aspect="auto")
    for r in range(piv_auc.shape[0]):
        for c in range(piv_auc.shape[1]):
            auc_v = piv_auc.values[r, c]
            acc_v = piv_acc.values[r, c]
            txt_c = "white" if auc_v > 0.81 else PALETTE["dark"]
            ax.text(c, r, f"{auc_v:.3f}\n({acc_v:.1f}%)", ha="center", va="center", fontsize=8.3, color=txt_c, fontweight="bold")

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(short_models, fontsize=9.0)
    ax.set_yticks(range(len(subset_order)))
    subset_labels = [
        "C  (Track, 7)",
        "B  (Form, 11)",
        "A  (Qualy, 8)",
        "B+C  (18)",
        "A+C  (15)",
        "A+B  (19)",
        "A+B+C  (26)",
    ]
    ax.set_yticklabels(subset_labels, fontsize=9.0)
    ax.set_title("A. 7-Subset Ablation Matrix: Test ROC-AUC (and Accuracy %)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cbar.set_label("Held-Out Test ROC-AUC", fontsize=9.2)

    # Panel B: Exact Shapley Group Attribution (% share of ROC-AUC gain over null across models)
    ax = axes[1]
    sh_auc = shapley_df[shapley_df["metric"] == "test_roc_auc"].set_index("model").loc[model_order].reset_index()
    y_pos = np.arange(len(model_order))
    s_a = sh_auc["share_A_pct"].values
    s_b = sh_auc["share_B_pct"].values
    s_c = sh_auc["share_C_pct"].values

    ax.barh(y_pos, s_a, height=0.56, color=GROUP_COLORS["A_qualifying"], label="Group A: Qualifying & Grid")
    ax.barh(y_pos, s_b, left=s_a, height=0.56, color=GROUP_COLORS["B_form"], label="Group B: Rolling Form")
    ax.barh(y_pos, s_c, left=s_a + s_b, height=0.56, color=GROUP_COLORS["C_track"], label="Group C: Circuit History")

    for i in range(len(model_order)):
        ax.text(s_a[i] / 2, i, f"{s_a[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        ax.text(s_a[i] + s_b[i] / 2, i, f"{s_b[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        if s_c[i] >= 6.5:
            ax.text(s_a[i] + s_b[i] + s_c[i] / 2, i, f"{s_c[i]:.1f}%", ha="center", va="center", color="white", fontsize=8.4, fontweight="bold")
        else:
            ax.text(min(s_a[i] + s_b[i] + s_c[i] + 1.5, 102.0), i, f"{s_c[i]:.1f}%", ha="left", va="center", color=PALETTE["dark"], fontsize=8.0, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("\n", " ") for m in short_models], fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Exact Shapley Attribution of ROC-AUC Gain Over Null (%)")
    ax.set_title("B. Shapley Contribution of Each Factor Group")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.4, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = out_dir / "fig08_ablation_7subsets_and_shapley.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig09_regularization_and_sparsity(whatif: dict[str, Any], out_dir: Path) -> Path:
    """Figure 9: Regularization bias-variance curves (L2 & L1) + L1 sparse feature selection."""
    configure_plot_style()
    reg_df = pd.DataFrame(whatif["regularization_sweep"])
    l2 = reg_df[reg_df["penalty"] == "L2 (Ridge)"]
    l1 = reg_df[reg_df["penalty"] == "L1 (Lasso)"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))

    # Panel A: L2 Bias-Variance Curve (Train vs Temporal CV vs Test Log-Loss)
    ax = axes[0]
    ax.semilogx(l2["C"], l2["train_logloss"], "o--", color=PALETTE["sky"], linewidth=1.8, label="L2 Train Log-Loss")
    ax.semilogx(l2["C"], l2["cv_logloss"], "s-", color=PALETTE["blue"], linewidth=2.3, label="L2 Temporal-CV Log-Loss")
    ax.semilogx(l2["C"], l2["test_logloss"], "D-", color=PALETTE["vermillion"], linewidth=2.2, label="L2 Held-Out Test Log-Loss")

    best_idx = int(l2["cv_logloss"].idxmin())
    best_c = l2.loc[best_idx, "C"]
    ax.axvline(best_c, color=PALETTE["green"], linestyle=":", linewidth=1.8, label=f"CV-Optimal C = {best_c:g}")
    ax.annotate(
        "Underfitting\n(High Bias)",
        xy=(0.0022, l2["cv_logloss"].max() - 0.008),
        fontsize=8.3,
        color=PALETTE["gray"],
        fontweight="bold",
    )
    ax.annotate(
        "Collinearity Variance\n(Train-CV Gap Widens)",
        xy=(12.0, l2["train_logloss"].min() + 0.028),
        ha="center",
        fontsize=8.2,
        color=PALETTE["gray"],
        fontweight="bold",
    )
    ax.set_xlabel("Inverse Regularization Strength C (log scale; Smaller = Stronger Regularization)")
    ax.set_ylabel("Binary Cross-Entropy Log-Loss (Lower = Better)")
    ax.set_title("A. L2 Regularization & Bias-Variance Trade-off")
    ax.legend(loc="upper right", fontsize=8.1)

    # Panel B: L1 Lasso Sparsity vs Predictive Performance
    ax = axes[1]
    ax2 = ax.twinx()
    l_cv = ax.semilogx(l1["C"], l1["cv_auc"], "s-", color=PALETTE["blue"], linewidth=2.2, label="L1 Temporal-CV ROC-AUC")
    l_te = ax.semilogx(l1["C"], l1["test_auc"], "D-", color=PALETTE["vermillion"], linewidth=2.1, label="L1 Held-Out Test ROC-AUC")
    l_nz = ax2.semilogx(l1["C"], l1["n_nonzero"], "^--", color=PALETTE["orange"], linewidth=2.0, label="Active Non-Zero Features (/26)")

    # Annotate sparse sweet spot at C=0.03
    row_03 = l1[np.isclose(l1["C"], 0.03)].iloc[0]
    ax.axvline(0.03, color=PALETTE["orange"], linestyle=":", linewidth=1.4)
    ax.annotate(
        f"C=0.03: {int(row_03['n_nonzero'])}/26 features\n(0 from Group C!)\nTest AUC = {row_03['test_auc']:.3f}",
        xy=(0.03, row_03["test_auc"]),
        xytext=(0.08, 0.72),
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["dark"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": PALETTE["orange"]},
        arrowprops={"arrowstyle": "->", "color": PALETTE["dark"], "lw": 1.0},
    )

    ax.set_xlabel("L1 Lasso Inverse Regularization Strength C (log scale)")
    ax.set_ylabel("ROC-AUC", color=PALETTE["blue"])
    ax2.set_ylabel("Active Non-Zero Features", color=PALETTE["orange"])
    ax2.set_ylim(0, 29)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PALETTE["orange"])
    ax.set_title("B. L1 (Lasso) Sparsity Path & Automatic Feature Selection")

    lines = l_cv + l_te + l_nz
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="lower right", fontsize=8.1)

    fig.tight_layout()
    out_path = out_dir / "fig09_regularization_and_bias_variance.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig10_interpretability(
    train_dev: pd.DataFrame,
    test_df: pd.DataFrame,
    best_linear_model,
    rf_model,
    out_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """Figure 10: Standardized Logistic Regression Coefficients + Test Permutation Importance."""
    configure_plot_style()
    X_dev = train_dev[ALL_FEATURES]
    y_dev = train_dev["top10"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["top10"].values

    feat_to_group = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}

    # 1) Standardized Logistic Regression coefficients + bootstrap 95% CIs (300 resamples on train_dev)
    coefs = best_linear_model.named_steps["clf"].coef_[0]
    rng = np.random.default_rng(RANDOM_STATE)
    boot_coefs = np.empty((200, len(ALL_FEATURES)), dtype=float)
    from sklearn.base import clone

    n_dev = len(X_dev)
    for b in range(200):
        idx = rng.integers(0, n_dev, size=n_dev)
        m_b = clone(best_linear_model)
        m_b.fit(X_dev.iloc[idx], y_dev[idx])
        boot_coefs[b] = m_b.named_steps["clf"].coef_[0]
    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

    # 2) Held-out Test Permutation Importance (30 repeats, ROC-AUC drop) for both Linear & RF
    perm_lin = permutation_importance(
        best_linear_model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_rf = permutation_importance(
        rf_model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
    )

    interp_df = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "label": [FEATURE_LABELS[f] for f in ALL_FEATURES],
            "group": [feat_to_group[f] for f in ALL_FEATURES],
            "std_coef": coefs,
            "odds_ratio_per_sd": np.exp(coefs),
            "coef_ci_lo": ci_lo,
            "coef_ci_hi": ci_hi,
            "perm_auc_linear_mean": perm_lin.importances_mean,
            "perm_auc_linear_std": perm_lin.importances_std,
            "perm_auc_rf_mean": perm_rf.importances_mean,
            "perm_auc_rf_std": perm_rf.importances_std,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))

    # Panel A: Top-14 Standardized Coefficients by |beta|
    ax = axes[0]
    top_coef = interp_df.reindex(interp_df["std_coef"].abs().sort_values(ascending=False).index).head(14).iloc[::-1]
    y_pos = np.arange(len(top_coef))
    colors = [GROUP_COLORS[g] for g in top_coef["group"]]
    ax.axvline(0.0, color=PALETTE["dark"], linewidth=1.0)
    ax.barh(y_pos, top_coef["std_coef"], color=colors, alpha=0.88, height=0.62)
    ax.errorbar(
        top_coef["std_coef"],
        y_pos,
        xerr=[
            np.maximum(0.0, top_coef["std_coef"] - top_coef["coef_ci_lo"]),
            np.maximum(0.0, top_coef["coef_ci_hi"] - top_coef["std_coef"]),
        ],
        fmt="none",
        ecolor=PALETTE["dark"],
        capsize=2.5,
        linewidth=1.1,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_coef["label"], fontsize=8.6)
    ax.set_xlabel("Standardized Log-Odds Coefficient (+/- 95% Bootstrap CI)")
    ax.set_title("A. Logistic Regression (L2) Standardized Weights (Top 14)")

    # Panel B: Top-12 Permutation Importance on Held-Out Test Set (Linear vs RF)
    ax = axes[1]
    top_perm = interp_df.sort_values("perm_auc_linear_mean", ascending=False).head(12).iloc[::-1]
    y_pos2 = np.arange(len(top_perm))
    ax.barh(
        y_pos2 + 0.17,
        top_perm["perm_auc_linear_mean"],
        height=0.33,
        xerr=top_perm["perm_auc_linear_std"],
        capsize=2.5,
        color=PALETTE["blue"],
        alpha=0.90,
        label="Logistic Regression (L2)",
    )
    ax.barh(
        y_pos2 - 0.17,
        top_perm["perm_auc_rf_mean"],
        height=0.33,
        xerr=top_perm["perm_auc_rf_std"],
        capsize=2.5,
        color=PALETTE["orange"],
        alpha=0.85,
        label="Random Forest",
    )
    ax.axvline(0.0, color=PALETTE["dark"], linewidth=0.9)
    ax.set_yticks(y_pos2)
    ax.set_yticklabels(top_perm["label"], fontsize=8.6)
    ax.set_xlabel("Held-Out Test ROC-AUC Drop When Permuted (+/- 1 SD)")
    ax.set_title("B. Out-of-Sample Permutation Importance (2022-2024 Test)")
    ax.legend(loc="lower right", fontsize=8.5)

    handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_SHORT_NAMES[g])
        for g in ("A_qualifying", "B_form", "C_track")
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8.0)

    fig.tight_layout()
    out_path = out_dir / "fig10_interpretability_weights_and_permutation.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, interp_df


def plot_fig11_roc_calibration_and_boundary(
    train_dev: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Figure 11: ROC Curves, Calibration Curves, and 2D Decision Boundary (Grid vs Form)."""
    configure_plot_style()
    y_test = pred_df["top10"].values

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))

    # Panel A: ROC Curves
    ax = axes[0]
    roc_models = [
        ("Logistic Regression (L2)", PALETTE["blue"], "-", 2.3),
        ("Linear SVM", PALETTE["sky"], "--", 2.0),
        ("Random Forest", PALETTE["orange"], "-", 2.0),
        ("Gradient Boosting", PALETTE["green"], "-.", 1.9),
        ("Gaussian Naive Bayes", PALETTE["purple"], ":", 1.9),
        ("Baseline: Grid-Only Logistic", PALETTE["gray"], "--", 1.5),
    ]
    for mname, col, ls, lw in roc_models:
        pcol = f"prob_{_slug(mname)}"
        if pcol not in pred_df.columns:
            continue
        probs = pred_df[pcol].values
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        short = mname.replace("Baseline: ", "").replace("Regression ", "Reg ")
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f"{short} ({auc:.3f})")

    # Mark exact operating point of hard heuristic grid <= 10
    g_pred = pred_df["pred_grid10"].values
    tn, fp, fn, tp = confusion_matrix(y_test, g_pred).ravel()
    ax.scatter(
        [fp / (fp + tn)],
        [tp / (tp + fn)],
        marker="*",
        s=190,
        color=PALETTE["vermillion"],
        edgecolors="black",
        zorder=5,
        label=f"grid <= 10 Rule (TPR={tp/(tp+fn):.2f}, FPR={fp/(fp+tn):.2f})",
    )
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.9, alpha=0.5)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("A. Test ROC Curves (2022-2024)")
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel B: Reliability / Calibration Curves
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.1, label="Perfect Calibration")
    cal_models = [
        ("Logistic Regression (L2)", PALETTE["blue"], "o-"),
        ("Random Forest", PALETTE["orange"], "s-"),
        ("Gradient Boosting", PALETTE["green"], "^-"),
        ("Gaussian Naive Bayes", PALETTE["purple"], "d:"),
    ]
    for mname, col, fmt in cal_models:
        pcol = f"prob_{_slug(mname)}"
        if pcol not in pred_df.columns:
            continue
        probs = pred_df[pcol].values
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=8, strategy="quantile")
        brier = brier_score_loss(y_test, probs)
        short = mname.replace("Regression ", "Reg ")
        ax.plot(mean_pred, frac_pos, fmt, color=col, linewidth=1.9, markersize=5.5, label=f"{short} (Brier={brier:.3f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Top-10 Finishes")
    ax.set_title("B. Probability Calibration (Test 2022-2024)")
    ax.legend(loc="upper left", fontsize=7.8)

    # Panel C: 2D Interpretable Linear Decision Boundary in (Grid Position, Constructor Form)
    ax = axes[2]
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feat_2d = ["grid_clean", "const_roll5_pts_per_car"]
    m2d = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(l1_ratio=0.0, C=0.1)),
        ]
    )
    m2d.fit(train_dev[feat_2d], train_dev["top10"].values)

    xx, yy = np.meshgrid(np.linspace(1, 21, 180), np.linspace(0, 21, 180))
    grid_df = pd.DataFrame({"grid_clean": xx.ravel(), "const_roll5_pts_per_car": yy.ravel()})
    zz = m2d.predict_proba(grid_df)[:, 1].reshape(xx.shape)

    cf = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.45)
    ax.contour(xx, yy, zz, levels=[0.5], colors=[PALETTE["dark"]], linewidths=2.2)
    ax.axvline(10.5, color=PALETTE["vermillion"], linestyle="--", linewidth=1.6, label="grid <= 10 Cutoff")

    # Overlay subsample of 2022-2024 test points
    rng = np.random.default_rng(RANDOM_STATE)
    jitter_x = test_df["grid_clean"].values + rng.normal(0, 0.18, len(test_df))
    pts_pos = test_df["top10"].values == 1
    ax.scatter(
        jitter_x[pts_pos],
        test_df["const_roll5_pts_per_car"].values[pts_pos],
        s=11,
        alpha=0.45,
        color=PALETTE["blue"],
        label="Actual Top-10",
    )
    ax.scatter(
        jitter_x[~pts_pos],
        test_df["const_roll5_pts_per_car"].values[~pts_pos],
        s=11,
        alpha=0.40,
        color=PALETTE["vermillion"],
        marker="x",
        label="Actual Outside Top-10",
    )
    ax.text(14.5, 16.5, "Fast car P11+\n-> Predicted Top-10!", fontsize=7.8, fontweight="bold", color=PALETTE["dark"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": PALETTE["blue"]})
    ax.text(5.2, 1.3, "Weak car P8-10\n-> Predicted Outside", fontsize=7.8, fontweight="bold", color=PALETTE["dark"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": PALETTE["vermillion"]})
    ax.set_xlim(0.5, 21.2)
    ax.set_ylim(-0.3, 21.0)
    ax.set_xlabel("Starting Grid Position (Scrubbed)")
    ax.set_ylabel("Constructor Rolling Pts/Car (Last 5 Races)")
    ax.set_title("C. Tilted Decision Boundary vs Vertical Cutoff")
    ax.legend(loc="upper right", fontsize=7.4)

    fig.tight_layout()
    out_path = out_dir / "fig11_roc_pr_calibration_and_boundary.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compute_error_analysis_and_plot_fig12(
    pred_df: pd.DataFrame,
    best_model_name: str,
    out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Figure 12: Confusion Matrix, Irreducible Incident Error Decomposition, & Slice Robustness."""
    configure_plot_style()
    m_slug = _slug(best_model_name)
    y_true = pred_df["top10"].values
    y_pred = pred_df[f"pred_{m_slug}"].values
    y_prob = pred_df[f"prob_{m_slug}"].values
    g_pred = pred_df["pred_grid10"].values

    # Post-hoc error decomposition (ONLY after predictions — never as features!)
    err_df = pred_df.copy()
    err_df["y_pred"] = y_pred
    err_df["y_prob"] = y_prob
    err_df["error_type"] = np.where(
        (err_df["top10"] == 1) & (err_df["y_pred"] == 1),
        "TP",
        np.where(
            (err_df["top10"] == 0) & (err_df["y_pred"] == 0),
            "TN",
            np.where((err_df["top10"] == 0) & (err_df["y_pred"] == 1), "FP", "FN"),
        ),
    )

    fps = err_df[err_df["error_type"] == "FP"]
    fns = err_df[err_df["error_type"] == "FN"]

    # For FPs: why did a predicted Top-10 driver fail to score? (Crash DNF vs Mech DNF vs DNS/DSQ vs Classified P11+)
    fp_breakdown = {
        "Crash / Incident DNF": int((fps["status_category"] == "incident_dnf").sum()),
        "Mechanical / Reliability DNF": int((fps["status_category"] == "mechanical_dnf").sum()),
        "DNS / DSQ / Excluded": int((fps["status_category"] == "dns_dnq_dsq").sum()),
        "Finished Classified P11+": int((fps["status_category"] == "finished").sum()),
    }
    fp_irreducible = (
        fp_breakdown["Crash / Incident DNF"]
        + fp_breakdown["Mechanical / Reliability DNF"]
        + fp_breakdown["DNS / DSQ / Excluded"]
    )

    # For FNs: count races where >=3 top-10 starters DNF'd (attrition promotion) vs normal finish
    race_top10_dnfs = (
        err_df[err_df["grid_clean"] <= 10]
        .groupby("raceId")["is_dnf"]
        .sum()
        .rename("ahead_top10_dnfs")
    )
    err_df = err_df.merge(race_top10_dnfs, on="raceId", how="left")
    fns = err_df[err_df["error_type"] == "FN"]
    fn_breakdown = {
        "High-Attrition Race (>=2 Top-10 DNFs)": int((fns["ahead_top10_dnfs"] >= 2).sum()),
        "Midfield Grid Border (P9-P13)": int(((fns["ahead_top10_dnfs"] < 2) & (fns["grid_clean"] <= 13)).sum()),
        "Back-of-Grid Pace/Strategy Surge (P14+)": int(((fns["ahead_top10_dnfs"] < 2) & (fns["grid_clean"] >= 14)).sum()),
    }

    # Iconic case studies where model beat grid <= 10 AND high-confidence irreducible misses
    overturns = err_df[(err_df["y_pred"] == err_df["top10"]) & (err_df["pred_grid10"] != err_df["top10"])].copy()
    overturns["confidence_margin"] = np.abs(overturns["y_prob"] - 0.5)
    top_wins = (
        overturns.sort_values("confidence_margin", ascending=False)
        .head(8)[
            [
                "year",
                "race_name",
                "driver_name",
                "constructor_name",
                "grid",
                "qualy_pos",
                "positionOrder",
                "const_roll5_pts_per_car",
                "y_prob",
                "y_pred",
                "top10",
            ]
        ]
        .to_dict(orient="records")
    )

    biggest_misses = (
        err_df[err_df["y_pred"] != err_df["top10"]]
        .assign(err_conf=lambda d: np.abs(d["y_prob"] - d["top10"]))
        .sort_values("err_conf", ascending=False)
        .head(8)[
            [
                "year",
                "race_name",
                "driver_name",
                "constructor_name",
                "grid",
                "positionOrder",
                "status",
                "status_category",
                "y_prob",
                "y_pred",
                "top10",
            ]
        ]
        .to_dict(orient="records")
    )

    # Slice robustness: 2022, 2023, 2024, Midfield P8-P13, Grid Penalty Drop >= 3
    slices = [
        ("2022 Season", err_df["year"] == 2022),
        ("2023 Season", err_df["year"] == 2023),
        ("2024 Season", err_df["year"] == 2024),
        ("Midfield (P8-P13)", (err_df["grid_clean"] >= 8) & (err_df["grid_clean"] <= 13)),
        ("Grid Penalty (Drop>=3)", err_df["grid_penalty_drop"] >= 3),
        ("Overall Test (2022-24)", pd.Series(True, index=err_df.index)),
    ]
    slice_rows = []
    for s_label, mask in slices:
        sub = err_df[mask]
        slice_rows.append(
            {
                "slice": s_label,
                "n": int(len(sub)),
                "grid10_acc": float(accuracy_score(sub["top10"], sub["pred_grid10"])) * 100.0,
                "model_acc": float(accuracy_score(sub["top10"], sub["y_pred"])) * 100.0,
                "model_auc": float(roc_auc_score(sub["top10"], sub["y_prob"])),
                "gain_pp": (
                    float(accuracy_score(sub["top10"], sub["y_pred"]))
                    - float(accuracy_score(sub["top10"], sub["pred_grid10"]))
                )
                * 100.0,
            }
        )
    slice_df = pd.DataFrame(slice_rows)

    # Plot Figure 12 (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), gridspec_kw={"width_ratios": [0.85, 1.05, 1.25]})

    # Panel A: Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            v = cm[r, c]
            pct = v / cm.sum() * 100.0
            tc = "white" if v > cm.max() * 0.55 else PALETTE["dark"]
            ax.text(c, r, f"{labels_cm[r][c]}\n{v}\n({pct:.1f}%)", ha="center", va="center", color=tc, fontsize=10.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Outside", "Pred Top-10"], fontsize=9.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Outside", "Actual Top-10"], fontsize=9.2)
    ax.set_title(f"A. Confusion Matrix\n({best_model_name})")

    # Panel B: Post-Hoc Error Decomposition (FP Causes + FN Context)
    ax = axes[1]
    fp_vals = list(fp_breakdown.values())
    fp_keys = ["Crash/Incident\nDNF", "Mechanical\nDNF", "DNS / DSQ", "Classified\nP11+ Pace"]
    fp_cols = [PALETTE["vermillion"], PALETTE["orange"], PALETTE["purple"], PALETTE["sky"]]
    x_pos = np.arange(len(fp_keys))
    bars = ax.bar(x_pos, fp_vals, color=fp_cols, alpha=0.90, width=0.62)
    for b, val in zip(bars, fp_vals):
        pct = val / max(len(fps), 1) * 100.0
        ax.text(b.get_x() + b.get_width() / 2, val + 1.8, f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fp_keys, fontsize=8.5)
    ax.set_ylabel("Number of False Positive Errors")
    ax.set_ylim(0, max(fp_vals) * 1.26)
    ax.set_title(
        f"B. False Positive Decomposition (n={len(fps)})\n"
        f"{fp_irreducible/max(len(fps),1)*100:.0f}% Caused by Unpredictable DNFs/DNS"
    )

    # Panel C: Slice Robustness (Model vs grid <= 10 Baseline)
    ax = axes[2]
    y_s = np.arange(len(slice_df))
    ax.barh(y_s - 0.18, slice_df["grid10_acc"], height=0.34, color=PALETTE["gray"], alpha=0.72, label="Baseline (grid <= 10)")
    ax.barh(y_s + 0.18, slice_df["model_acc"], height=0.34, color=PALETTE["blue"], alpha=0.92, label=best_model_name)
    for i, row in slice_df.iterrows():
        sign = "+" if row["gain_pp"] >= 0 else ""
        max_x = max(row["grid10_acc"], row["model_acc"])
        ax.text(
            max_x + 0.8,
            i,
            f"{sign}{row['gain_pp']:.1f} pp",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color=PALETTE["green"] if row["gain_pp"] >= 0 else PALETTE["vermillion"],
        )
    ax.set_yticks(y_s)
    ax.set_yticklabels([f"{r['slice']} (n={r['n']})" for _, r in slice_df.iterrows()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(45, 98)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("C. Temporal & Hard-Slice Robustness vs Baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.2)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = out_dir / "fig12_error_analysis_and_regime_robustness.png"
    fig.savefig(out_path)
    plt.close(fig)

    error_summary = {
        "evaluated_model": best_model_name,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "false_positive_total": int(len(fps)),
        "false_positive_irreducible_dnf_share": float(fp_irreducible / max(len(fps), 1)),
        "false_positive_breakdown": fp_breakdown,
        "false_negative_total": int(len(fns)),
        "false_negative_breakdown": fn_breakdown,
        "overturn_wins_vs_grid10": int(len(overturns)),
        "overturn_losses_vs_grid10": int(((err_df["y_pred"] != err_df["top10"]) & (err_df["pred_grid10"] == err_df["top10"])).sum()),
        "slice_robustness": slice_rows,
        "iconic_overturn_wins": top_wins,
        "highest_confidence_misses": biggest_misses,
    }
    return out_path, error_summary


def run_followup2_evaluation(
    mod_df: pd.DataFrame,
    out_dir: Path = FIGURES_F2_DIR,
) -> dict[str, Any]:
    """Run full Follow-up #2 interpretability, error analysis, and figure generation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = REPORTS_DIR / "metrics"

    train_dev, test_df, _ = get_temporal_splits(mod_df)
    bundle = load_json(metrics_dir / "followup2_modeling_results.json")
    bm_df = pd.read_csv(metrics_dir / "followup2_model_benchmark.csv")
    ablation_df = pd.read_csv(metrics_dir / "followup2_ablation_7subsets.csv")
    shapley_df = pd.read_csv(metrics_dir / "followup2_ablation_shapley.csv")
    pred_df = pd.read_csv(metrics_dir / "followup2_test_predictions.csv")

    best_linear_name = bundle["best_linear_model"]
    best_tree_name = bundle["best_tree_model"]
    lin_slug = _slug(best_linear_name)
    # Load L2 Logistic for coefficient interpretability (always saved if best linear, else fallback)
    l2_path = MODELS_DIR / "logistic_regression_l2.joblib"
    best_lin_path = MODELS_DIR / f"{lin_slug}.joblib"
    if l2_path.exists():
        lin_model = joblib.load(l2_path)
        coef_model_name = "Logistic Regression (L2)"
    else:
        lin_model = joblib.load(best_lin_path)
        coef_model_name = best_linear_name

    rf_path = MODELS_DIR / "random_forest.joblib"
    if not rf_path.exists():
        # Fall back to best tree model
        rf_path = MODELS_DIR / f"{_slug(best_tree_name)}.joblib"
    tree_model = joblib.load(rf_path)

    fig07 = plot_fig07_model_comparison(bm_df, out_dir)
    fig08 = plot_fig08_ablation_and_shapley(ablation_df, shapley_df, out_dir)
    fig09 = plot_fig09_regularization_and_sparsity(bundle["whatif"], out_dir)
    fig10, interp_df = plot_fig10_interpretability(train_dev, test_df, lin_model, tree_model, out_dir)
    fig11 = plot_fig11_roc_calibration_and_boundary(train_dev, test_df, pred_df, out_dir)
    fig12, error_summary = compute_error_analysis_and_plot_fig12(pred_df, coef_model_name, out_dir)

    interp_df.to_csv(metrics_dir / "followup2_interpretability_table.csv", index=False)

    eval_summary = {
        "interpreted_model": coef_model_name,
        "error_analysis": error_summary,
        "top_coefficients": interp_df.reindex(
            interp_df["std_coef"].abs().sort_values(ascending=False).index
        )
        .head(10)
        .to_dict(orient="records"),
        "figures": {
            "fig07": str(fig07),
            "fig08": str(fig08),
            "fig09": str(fig09),
            "fig10": str(fig10),
            "fig11": str(fig11),
            "fig12": str(fig12),
        },
    }
    save_json(eval_summary, metrics_dir / "followup2_evaluation_summary.json")
    logger.info("Follow-up #2 evaluation complete. Saved 6 figures to %s", out_dir)
    return eval_summary
