"""Modeling, temporal expanding-window CV, ablation, and what-if studies (Follow-up #2).

Implements and compares machine learning algorithms covered in the UAB
Fundamentals of Machine Learning curriculum:
- Baselines: Majority Dummy, Scrubbed Heuristic (grid <= 10), Single-Feature Grid Logistic
- Linear / Regularized: Logistic Regression (L2 Ridge), Logistic Regression (L1 Lasso),
  Linear SVM (with Platt calibration)
- Probabilistic & Instance-Based: Gaussian Naive Bayes, k-Nearest Neighbors (CBR)
- Non-Linear Kernel & Tree Ensembles: RBF Kernel SVM, Decision Tree (Gini/Entropy),
  Random Forest, Gradient Boosting

Enforces strict season-based temporal validation:
- Development set: 2010-2021 (5-fold expanding-window CV on seasons 2017..2021)
- Held-out Test set: 2022-2024 (Ground-Effect era, n=1,359, touched only after tuning)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.features import (
    ALL_FEATURES,
    CV_VAL_SEASONS,
    FEATURE_GROUPS,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    build_feature_dataset,
    get_temporal_splits,
)
from src.utils import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, get_logger, save_json

logger = get_logger("f1_ml.models")

ABLATION_SUBSETS: dict[str, list[str]] = {
    "A": FEATURE_GROUPS["A_qualifying"],
    "B": FEATURE_GROUPS["B_form"],
    "C": FEATURE_GROUPS["C_track"],
    "A+B": FEATURE_GROUPS["A_qualifying"] + FEATURE_GROUPS["B_form"],
    "A+C": FEATURE_GROUPS["A_qualifying"] + FEATURE_GROUPS["C_track"],
    "B+C": FEATURE_GROUPS["B_form"] + FEATURE_GROUPS["C_track"],
    "A+B+C": ALL_FEATURES,
}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a comprehensive set of binary classification & calibration metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        probs = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
        metrics["log_loss"] = float(log_loss(y_true, probs))
        metrics["brier"] = float(brier_score_loss(y_true, probs))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred))
        metrics["log_loss"] = np.nan
        metrics["brier"] = np.nan
    return metrics


def mcnemar_exact_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """Exact two-sided McNemar test comparing two classifiers on paired samples.

    b_wins: count where model B is correct and model A is wrong.
    a_wins: count where model A is correct and model B is wrong.
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b_wins = int(np.sum((~a_correct) & b_correct))
    a_wins = int(np.sum(a_correct & (~b_correct)))
    n_disc = b_wins + a_wins
    if n_disc == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(b_wins, n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "b_correct_a_wrong": b_wins,
        "a_correct_b_wrong": a_wins,
        "net_gain_cases": b_wins - a_wins,
        "discordant_pairs": n_disc,
        "p_value": p_val,
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    pred_or_prob_a: np.ndarray,
    pred_or_prob_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Compute paired bootstrap 95% CI for metric(B) - metric(A)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = metric_fn(yt, pred_or_prob_b[idx]) - metric_fn(yt, pred_or_prob_a[idx])
    diffs = diffs[np.isfinite(diffs)]
    point = float(metric_fn(y_true, pred_or_prob_b) - metric_fn(y_true, pred_or_prob_a))
    return {
        "delta": point,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_boot_two_sided": float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
    }


def _make_model_specs() -> list[dict[str, Any]]:
    """Define model pipelines and temporal CV hyperparameter grids (sklearn 1.9 clean)."""
    return [
        {
            "name": "Logistic Regression (L2)",
            "family": "Linear",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            l1_ratio=0.0,
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        },
        {
            "name": "Logistic Regression (L1)",
            "family": "Linear",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            l1_ratio=1.0,
                            solver="liblinear",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {"clf__C": [0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        },
        {
            "name": "Linear SVM",
            "family": "Linear",
            "needs_calibration": True,
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearSVC(
                            dual="auto",
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            },
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Probabilistic",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", GaussianNB()),
                ]
            ),
            "param_grid": {"clf__var_smoothing": [1e-9, 1e-6, 1e-4, 1e-2, 1e-1]},
        },
        {
            "name": "k-Nearest Neighbors",
            "family": "Instance-Based",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "clf__n_neighbors": [15, 25, 35, 51, 75],
                "clf__weights": ["uniform", "distance"],
            },
        },
        {
            "name": "RBF Kernel SVM",
            "family": "Non-Linear Kernel",
            "needs_calibration": True,
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__C": [0.1, 0.5, 1.0, 3.0],
                "clf__gamma": ["scale", 0.01, 0.03],
            },
        },
        {
            "name": "Decision Tree",
            "family": "Tree",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 4, 5, 6, 8],
                "clf__min_samples_leaf": [15, 30, 50],
            },
        },
        {
            "name": "Random Forest",
            "family": "Tree Ensemble",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            n_jobs=1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__max_depth": [5, 8, 12],
                "clf__min_samples_leaf": [10, 20, 35],
                "clf__max_features": ["sqrt", 0.4],
            },
        },
        {
            "name": "Gradient Boosting",
            "family": "Tree Ensemble",
            "pipe": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "clf__n_estimators": [80, 150],
                "clf__learning_rate": [0.03, 0.06],
                "clf__max_depth": [2, 3],
                "clf__min_samples_leaf": [20, 40],
                "clf__subsample": [0.8],
            },
        },
    ]


def _wrap_for_probabilities(best_pipe: Pipeline, needs_calibration: bool):
    """Wrap decision-function SVM pipelines in Platt calibration cleanly (sklearn 1.9+)."""
    if not needs_calibration:
        return best_pipe
    # Calibrate only the final SVM step inside the same preprocessing pipeline
    steps = [(name, clone(step)) for name, step in best_pipe.steps[:-1]]
    clf_name, clf_step = best_pipe.steps[-1]
    steps.append(
        (
            clf_name,
            CalibratedClassifierCV(
                estimator=clone(clf_step),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
        )
    )
    return Pipeline(steps)


def evaluate_out_of_fold_cv(
    estimator,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute strict expanding-window out-of-fold predictions & per-season CV metrics."""
    oof_probs = np.full(len(y_dev), np.nan, dtype=float)
    oof_preds = np.full(len(y_dev), -1, dtype=int)
    fold_metrics = []

    for tr_idx, va_idx in cv_splits:
        est = clone(estimator)
        est.fit(X_dev.iloc[tr_idx], y_dev[tr_idx])
        probs = est.predict_proba(X_dev.iloc[va_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[va_idx] = probs
        oof_preds[va_idx] = preds
        fold_metrics.append(compute_classification_metrics(y_dev[va_idx], preds, probs))

    mean_cv = {
        f"cv_{k}": float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0].keys()
    }
    std_cv = {
        f"cv_{k}_std": float(np.std([m[k] for m in fold_metrics], ddof=1))
        for k in ("accuracy", "roc_auc", "f1", "log_loss")
    }
    mean_cv.update(std_cv)
    return mean_cv, oof_preds, oof_probs


def run_benchmark_models(
    train_dev: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Train and evaluate all baselines + 9 course models with temporal CV & test holdout."""
    X_dev = train_dev[ALL_FEATURES]
    y_dev = train_dev["top10"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["top10"].values

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    test_probs_map: dict[str, np.ndarray] = {}
    test_preds_map: dict[str, np.ndarray] = {}

    # 1) Baseline 0: Majority Class Dummy
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_cv, _, _ = evaluate_out_of_fold_cv(dummy, X_dev, y_dev, cv_splits)
    dummy.fit(X_dev, y_dev)
    d_prob = dummy.predict_proba(X_test)[:, 1]
    d_pred = dummy.predict(X_test)
    d_test = compute_classification_metrics(y_test, d_pred, d_prob)
    rows.append(
        {
            "model": "Baseline: Majority Class",
            "family": "Baseline",
            "best_params": "{}",
            **dummy_cv,
            **{f"test_{k}": v for k, v in d_test.items()},
        }
    )
    test_probs_map["Baseline: Majority Class"] = d_prob
    test_preds_map["Baseline: Majority Class"] = d_pred

    # 2) Baseline 1: Mandatory Scrubbed Heuristic (grid_clean <= 10)
    grid_cv_folds = []
    for _, va_idx in cv_splits:
        p_va = (train_dev.iloc[va_idx]["grid_clean"].values <= 10).astype(int)
        grid_cv_folds.append(compute_classification_metrics(y_dev[va_idx], p_va, None))
    grid_cv = {
        f"cv_{k}": float(np.nanmean([m[k] for m in grid_cv_folds]))
        if not np.all(np.isnan([m[k] for m in grid_cv_folds]))
        else np.nan
        for k in grid_cv_folds[0].keys()
    }
    grid_cv["cv_accuracy_std"] = float(np.std([m["accuracy"] for m in grid_cv_folds], ddof=1))
    grid_cv["cv_roc_auc_std"] = float(np.std([m["roc_auc"] for m in grid_cv_folds], ddof=1))
    grid_cv["cv_f1_std"] = float(np.std([m["f1"] for m in grid_cv_folds], ddof=1))
    grid_cv["cv_log_loss_std"] = np.nan

    g_pred = (test_df["grid_clean"].values <= 10).astype(int)
    g_test = compute_classification_metrics(y_test, g_pred, None)
    rows.append(
        {
            "model": "Baseline: grid <= 10 (Scrubbed)",
            "family": "Baseline",
            "best_params": "rule: 1 <= grid <= 10",
            **grid_cv,
            **{f"test_{k}": v for k, v in g_test.items()},
        }
    )
    test_preds_map["Baseline: grid <= 10 (Scrubbed)"] = g_pred

    # 3) Baseline 2: Single-Feature Grid Logistic Regression
    grid_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(l1_ratio=0.0, C=1.0, max_iter=1000)),
        ]
    )
    glr_cv, _, _ = evaluate_out_of_fold_cv(grid_lr, train_dev[["grid_clean"]], y_dev, cv_splits)
    grid_lr.fit(train_dev[["grid_clean"]], y_dev)
    glr_prob = grid_lr.predict_proba(test_df[["grid_clean"]])[:, 1]
    glr_pred = (glr_prob >= 0.5).astype(int)
    glr_test = compute_classification_metrics(y_test, glr_pred, glr_prob)
    rows.append(
        {
            "model": "Baseline: Grid-Only Logistic",
            "family": "Baseline",
            "best_params": "{'feature': 'grid_clean'}",
            **glr_cv,
            **{f"test_{k}": v for k, v in glr_test.items()},
        }
    )
    fitted_models["Baseline: Grid-Only Logistic"] = grid_lr
    test_probs_map["Baseline: Grid-Only Logistic"] = glr_prob
    test_preds_map["Baseline: Grid-Only Logistic"] = glr_pred

    # 4) Full Course Model Suite with Temporal Expanding-Window GridSearchCV
    for spec in _make_model_specs():
        name = spec["name"]
        logger.info("Tuning %s via 5-fold expanding-window seasonal CV...", name)
        gs = GridSearchCV(
            estimator=spec["pipe"],
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv_splits,
            refit=True,
            n_jobs=-1,
        )
        gs.fit(X_dev, y_dev)
        best_est = _wrap_for_probabilities(
            gs.best_estimator_, bool(spec.get("needs_calibration", False))
        )

        cv_metrics, _, _ = evaluate_out_of_fold_cv(best_est, X_dev, y_dev, cv_splits)
        best_est.fit(X_dev, y_dev)
        y_prob = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        test_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        rows.append(
            {
                "model": name,
                "family": spec["family"],
                "best_params": str(gs.best_params_),
                **cv_metrics,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        fitted_models[name] = best_est
        test_probs_map[name] = y_prob
        test_preds_map[name] = y_pred

    results_df = pd.DataFrame(rows)
    return results_df, fitted_models, test_preds_map, test_probs_map


def run_ablation_study(
    train_dev: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete 7-subset (A, B, C, A+B, A+C, B+C, A+B+C) ablation matrix.

    Evaluates 5 representative model families spanning linear, kernel, and tree ensembles,
    retuning each model strictly on temporal CV within each feature subset, and computes
    exact Shapley group attributions across Accuracy, ROC-AUC, and Log-Loss.
    """
    ablation_models = [
        spec
        for spec in _make_model_specs()
        if spec["name"]
        in {
            "Logistic Regression (L2)",
            "Linear SVM",
            "RBF Kernel SVM",
            "Random Forest",
            "Gradient Boosting",
        }
    ]
    y_dev = train_dev["top10"].values
    y_test = test_df["top10"].values

    records = []
    for subset_name, cols in ABLATION_SUBSETS.items():
        X_dev_sub = train_dev[cols]
        X_test_sub = test_df[cols]
        for spec in ablation_models:
            mname = spec["name"]
            gs = GridSearchCV(
                estimator=clone(spec["pipe"]),
                param_grid=spec["param_grid"],
                scoring="roc_auc",
                cv=cv_splits,
                refit=True,
                n_jobs=-1,
            )
            gs.fit(X_dev_sub, y_dev)
            best_est = _wrap_for_probabilities(
                gs.best_estimator_, bool(spec.get("needs_calibration", False))
            )
            cv_m, _, _ = evaluate_out_of_fold_cv(best_est, X_dev_sub, y_dev, cv_splits)
            best_est.fit(X_dev_sub, y_dev)
            y_prob = best_est.predict_proba(X_test_sub)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            test_m = compute_classification_metrics(y_test, y_pred, y_prob)

            records.append(
                {
                    "subset": subset_name,
                    "n_features": len(cols),
                    "model": mname,
                    "family": spec["family"],
                    "best_params": str(gs.best_params_),
                    "cv_accuracy": cv_m["cv_accuracy"],
                    "cv_roc_auc": cv_m["cv_roc_auc"],
                    "cv_log_loss": cv_m["cv_log_loss"],
                    "test_accuracy": test_m["accuracy"],
                    "test_f1": test_m["f1"],
                    "test_roc_auc": test_m["roc_auc"],
                    "test_pr_auc": test_m["pr_auc"],
                    "test_log_loss": test_m["log_loss"],
                    "test_brier": test_m["brier"],
                }
            )
        logger.info("Completed ablation subset %-5s (%2d features)", subset_name, len(cols))

    ablation_df = pd.DataFrame(records)

    # Compute exact 3-player Shapley values for each model over test_roc_auc, test_accuracy, test_log_loss
    # Baseline empty coalition v(empty): accuracy=0.500, roc_auc=0.500, log_loss=ln(2)=0.69315
    v_empty = {
        "test_accuracy": 0.50,
        "test_roc_auc": 0.50,
        "test_log_loss": float(np.log(2.0)),
    }
    players = ["A", "B", "C"]

    def _subset_key(s: set[str]) -> str:
        return "+".join(p for p in players if p in s)

    shapley_rows = []
    for mname, grp in ablation_df.groupby("model"):
        lookup = grp.set_index("subset").to_dict(orient="index")
        for metric in ("test_roc_auc", "test_accuracy", "test_log_loss"):
            # For log_loss, report reduction (empty - v) so positive = better!
            sign = -1.0 if metric == "test_log_loss" else 1.0

            def val(s: set[str]) -> float:
                if not s:
                    return 0.0
                return sign * (lookup[_subset_key(s)][metric] - v_empty[metric])

            # Shapley formula for 3 players: weights |S|=0 -> 1/3, |S|=1 -> 1/6, |S|=2 -> 1/3
            weights = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}
            phis: dict[str, float] = {}
            for p in players:
                others = [q for q in players if q != p]
                coalitions = [set(), {others[0]}, {others[1]}, set(others)]
                phi = sum(weights[len(c)] * (val(c | {p}) - val(c)) for c in coalitions)
                phis[p] = float(phi)
            total = sum(phis.values())
            shapley_rows.append(
                {
                    "model": mname,
                    "metric": metric,
                    "phi_A": phis["A"],
                    "phi_B": phis["B"],
                    "phi_C": phis["C"],
                    "share_A_pct": (phis["A"] / total * 100.0) if total != 0 else np.nan,
                    "share_B_pct": (phis["B"] / total * 100.0) if total != 0 else np.nan,
                    "share_C_pct": (phis["C"] / total * 100.0) if total != 0 else np.nan,
                    "total_gain_over_null": total,
                }
            )

    shapley_df = pd.DataFrame(shapley_rows)
    return ablation_df, shapley_df


def run_whatif_experiments(
    base_df: pd.DataFrame,
    raw_results: pd.DataFrame,
    raw_races: pd.DataFrame,
    train_dev: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    best_linear_model,
) -> dict[str, Any]:
    """Run 4 targeted 'what-if' scientific experiments for Follow-up #2."""
    X_dev = train_dev[ALL_FEATURES]
    y_dev = train_dev["top10"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["top10"].values

    # 1) Regularization & Bias-Variance Sweep (L2 vs L1 across C grid)
    c_grid = [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
    reg_rows = []
    for penalty_label, l1_ratio, solver in [
        ("L2 (Ridge)", 0.0, "lbfgs"),
        ("L1 (Lasso)", 1.0, "liblinear"),
    ]:
        for c_val in c_grid:
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            l1_ratio=l1_ratio,
                            C=c_val,
                            solver=solver,
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            cv_m, _, _ = evaluate_out_of_fold_cv(pipe, X_dev, y_dev, cv_splits)
            pipe.fit(X_dev, y_dev)
            tr_prob = pipe.predict_proba(X_dev)[:, 1]
            te_prob = pipe.predict_proba(X_test)[:, 1]
            coefs = pipe.named_steps["clf"].coef_[0]
            reg_rows.append(
                {
                    "penalty": penalty_label,
                    "C": c_val,
                    "n_nonzero": int(np.sum(np.abs(coefs) > 1e-4)),
                    "l2_norm": float(np.linalg.norm(coefs)),
                    "train_auc": float(roc_auc_score(y_dev, tr_prob)),
                    "cv_auc": cv_m["cv_roc_auc"],
                    "test_auc": float(roc_auc_score(y_test, te_prob)),
                    "train_logloss": float(log_loss(y_dev, tr_prob)),
                    "cv_logloss": cv_m["cv_log_loss"],
                    "test_logloss": float(log_loss(y_test, te_prob)),
                }
            )
    reg_df = pd.DataFrame(reg_rows)

    # Track which exact features survive strong L1 sparsity (at C=0.01 and C=0.03)
    l1_sparse_features = {}
    for c_probe in (0.01, 0.03, 0.1):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        l1_ratio=1.0,
                        C=c_probe,
                        solver="liblinear",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        pipe.fit(X_dev, y_dev)
        coefs = pipe.named_steps["clf"].coef_[0]
        active = [
            {"feature": f, "coef": float(w)}
            for f, w in sorted(zip(ALL_FEATURES, coefs), key=lambda kv: abs(kv[1]), reverse=True)
            if abs(w) > 1e-4
        ]
        l1_sparse_features[f"C={c_probe}"] = active

    # 2) Rolling Window Horizon Sensitivity (W = 3 vs 5 vs 10 races)
    window_rows = []
    for w_short, w_main, w_long, label in [
        (2, 3, 6, "Short (W=3)"),
        (3, 5, 10, "Standard (W=5)"),
        (5, 10, 15, "Long (W=10)"),
    ]:
        mod_w = build_feature_dataset(
            base_df,
            raw_results_all_years=raw_results,
            raw_races_all_years=raw_races,
            roll_short=w_short,
            roll_main=w_main,
            roll_long=w_long,
        )
        tr_w, te_w, cv_w = get_temporal_splits(mod_w)
        est = clone(best_linear_model)
        cv_m, _, _ = evaluate_out_of_fold_cv(est, tr_w[ALL_FEATURES], tr_w["top10"].values, cv_w)
        est.fit(tr_w[ALL_FEATURES], tr_w["top10"].values)
        p_te = est.predict_proba(te_w[ALL_FEATURES])[:, 1]
        y_te = te_w["top10"].values
        tm = compute_classification_metrics(y_te, (p_te >= 0.5).astype(int), p_te)
        window_rows.append(
            {
                "window_config": label,
                "cv_roc_auc": cv_m["cv_roc_auc"],
                "cv_accuracy": cv_m["cv_accuracy"],
                "test_roc_auc": tm["roc_auc"],
                "test_accuracy": tm["accuracy"],
                "test_log_loss": tm["log_loss"],
            }
        )

    # 3) Constructor Lineage Canonicalization vs Naive Raw constructorId
    base_raw_id = base_df.copy()
    base_raw_id["constructor_lineage"] = base_raw_id["constructorId"].astype(str)
    mod_raw_id = build_feature_dataset(
        base_raw_id,
        raw_results_all_years=raw_results,
        raw_races_all_years=raw_races,
    )
    tr_r, te_r, cv_r = get_temporal_splits(mod_raw_id)
    est_r = clone(best_linear_model)
    cv_m_r, _, _ = evaluate_out_of_fold_cv(est_r, tr_r[ALL_FEATURES], tr_r["top10"].values, cv_r)
    est_r.fit(tr_r[ALL_FEATURES], tr_r["top10"].values)
    p_te_r = est_r.predict_proba(te_r[ALL_FEATURES])[:, 1]
    tm_r = compute_classification_metrics(te_r["top10"].values, (p_te_r >= 0.5).astype(int), p_te_r)

    # Compare specifically on rebranded team transition seasons (2022-2024: RB 2024, Sauber 2024, Aston, Alpine)
    rebrand_mask = te_r["constructor_name"].isin(
        ["RB F1 Team", "Sauber", "Alfa Romeo", "AlphaTauri", "Aston Martin", "Alpine F1 Team"]
    ).values
    p_canon = best_linear_model.predict_proba(X_test)[:, 1]
    lineage_comparison = {
        "canonical_lineage": {
            "cv_roc_auc": float(evaluate_out_of_fold_cv(clone(best_linear_model), X_dev, y_dev, cv_splits)[0]["cv_roc_auc"]),
            "test_roc_auc": float(roc_auc_score(y_test, p_canon)),
            "test_accuracy": float(accuracy_score(y_test, (p_canon >= 0.5).astype(int))),
            "test_log_loss": float(log_loss(y_test, p_canon)),
            "rebrand_teams_2022_2024_logloss": float(log_loss(y_test[rebrand_mask], p_canon[rebrand_mask])),
            "rebrand_teams_2022_2024_auc": float(roc_auc_score(y_test[rebrand_mask], p_canon[rebrand_mask])),
        },
        "naive_raw_constructorId": {
            "cv_roc_auc": cv_m_r["cv_roc_auc"],
            "test_roc_auc": tm_r["roc_auc"],
            "test_accuracy": tm_r["accuracy"],
            "test_log_loss": tm_r["log_loss"],
            "rebrand_teams_2022_2024_logloss": float(log_loss(y_test[rebrand_mask], p_te_r[rebrand_mask])),
            "rebrand_teams_2022_2024_auc": float(roc_auc_score(y_test[rebrand_mask], p_te_r[rebrand_mask])),
        },
    }

    # 4) Threshold Optimization (strictly on out-of-fold CV predictions!)
    _, _, oof_probs = evaluate_out_of_fold_cv(clone(best_linear_model), X_dev, y_dev, cv_splits)
    valid_oof = np.isfinite(oof_probs)
    thresh_grid = np.linspace(0.30, 0.70, 81)
    best_t_acc = float(
        thresh_grid[
            int(np.argmax([accuracy_score(y_dev[valid_oof], (oof_probs[valid_oof] >= t).astype(int)) for t in thresh_grid]))
        ]
    )
    best_t_f1 = float(
        thresh_grid[
            int(np.argmax([f1_score(y_dev[valid_oof], (oof_probs[valid_oof] >= t).astype(int)) for t in thresh_grid]))
        ]
    )
    threshold_analysis = {
        "default_0.50": compute_classification_metrics(y_test, (p_canon >= 0.50).astype(int), p_canon),
        "cv_opt_acc_threshold": best_t_acc,
        "cv_opt_acc_test": compute_classification_metrics(y_test, (p_canon >= best_t_acc).astype(int), p_canon),
        "cv_opt_f1_threshold": best_t_f1,
        "cv_opt_f1_test": compute_classification_metrics(y_test, (p_canon >= best_t_f1).astype(int), p_canon),
    }

    return {
        "regularization_sweep": reg_df.to_dict(orient="records"),
        "l1_sparse_features": l1_sparse_features,
        "window_sensitivity": window_rows,
        "lineage_canonicalization": lineage_comparison,
        "threshold_analysis": threshold_analysis,
    }


def run_all_modeling(
    mod_df: pd.DataFrame,
    base_df: pd.DataFrame,
    raw_results: pd.DataFrame,
    raw_races: pd.DataFrame,
) -> dict[str, Any]:
    """Execute the full Follow-up #2 modeling, ablation, significance, and what-if suite."""
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_dev, test_df, cv_splits = get_temporal_splits(mod_df)
    y_test = test_df["top10"].values

    # 1) Full model benchmark
    benchmark_df, fitted_models, test_preds, test_probs = run_benchmark_models(
        train_dev, test_df, cv_splits
    )
    benchmark_df.to_csv(metrics_dir / "followup2_model_benchmark.csv", index=False)

    # Identify best overall model (by CV ROC-AUC) and best linear / best tree-ensemble models
    ml_only = benchmark_df[benchmark_df["family"] != "Baseline"]
    best_overall_name = ml_only.sort_values("cv_roc_auc", ascending=False).iloc[0]["model"]
    best_linear_name = (
        ml_only[ml_only["family"] == "Linear"]
        .sort_values("cv_roc_auc", ascending=False)
        .iloc[0]["model"]
    )
    best_tree_name = (
        ml_only[ml_only["family"] == "Tree Ensemble"]
        .sort_values("cv_roc_auc", ascending=False)
        .iloc[0]["model"]
    )

    # 2) Statistical significance tests (McNemar + Paired Bootstrap 95% CIs)
    grid_pred = test_preds["Baseline: grid <= 10 (Scrubbed)"]
    grid_lr_prob = test_probs["Baseline: Grid-Only Logistic"]
    significance: dict[str, Any] = {}
    for mname in ml_only["model"]:
        significance[f"{mname} vs grid<=10"] = {
            "mcnemar": mcnemar_exact_test(y_test, grid_pred, test_preds[mname]),
            "bootstrap_acc_delta": paired_bootstrap_ci(
                y_test, grid_pred, test_preds[mname], accuracy_score
            ),
            "bootstrap_auc_delta_vs_grid_lr": paired_bootstrap_ci(
                y_test, grid_lr_prob, test_probs[mname], roc_auc_score
            ),
        }
    significance[f"{best_linear_name} vs {best_tree_name}"] = {
        "mcnemar": mcnemar_exact_test(y_test, test_preds[best_tree_name], test_preds[best_linear_name]),
        "bootstrap_acc_delta": paired_bootstrap_ci(
            y_test, test_preds[best_tree_name], test_preds[best_linear_name], accuracy_score
        ),
        "bootstrap_auc_delta": paired_bootstrap_ci(
            y_test, test_probs[best_tree_name], test_probs[best_linear_name], roc_auc_score
        ),
    }

    # 3) Full 7-subset Factor Group Ablation Study + Exact Shapley Group Attributions
    ablation_df, shapley_df = run_ablation_study(train_dev, test_df, cv_splits)
    ablation_df.to_csv(metrics_dir / "followup2_ablation_7subsets.csv", index=False)
    shapley_df.to_csv(metrics_dir / "followup2_ablation_shapley.csv", index=False)

    # 4) What-If Experiments
    whatif = run_whatif_experiments(
        base_df=base_df,
        raw_results=raw_results,
        raw_races=raw_races,
        train_dev=train_dev,
        test_df=test_df,
        cv_splits=cv_splits,
        best_linear_model=fitted_models[best_linear_name],
    )

    # 5) Persist serialized best models + test predictions table for interpretation
    models_to_save = {
        best_linear_name,
        best_tree_name,
        "Logistic Regression (L2)",
        "Logistic Regression (L1)",
        "Linear SVM",
        "RBF Kernel SVM",
        "Random Forest",
        "Gradient Boosting",
    }
    for mname in sorted(models_to_save):
        if mname in fitted_models:
            slug = mname.lower().replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(fitted_models[mname], MODELS_DIR / f"{slug}.joblib")

    pred_table = test_df[
        [
            "raceId",
            "year",
            "round",
            "race_name",
            "circuit_name",
            "driver_name",
            "code",
            "constructor_name",
            "grid",
            "grid_clean",
            "qualy_pos",
            "grid_penalty_drop",
            "const_roll5_pts_per_car",
            "positionOrder",
            "positionText",
            "status",
            "status_category",
            "is_dnf",
            "top10",
        ]
    ].copy()
    pred_table["pred_grid10"] = grid_pred
    for mname, probs in test_probs.items():
        slug = mname.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("-", "_")
        pred_table[f"prob_{slug}"] = probs
        pred_table[f"pred_{slug}"] = test_preds[mname]
    pred_table.to_csv(metrics_dir / "followup2_test_predictions.csv", index=False)

    bundle = {
        "best_overall_model": best_overall_name,
        "best_linear_model": best_linear_name,
        "best_tree_model": best_tree_name,
        "benchmark": benchmark_df.to_dict(orient="records"),
        "significance": significance,
        "shapley_summary": shapley_df.to_dict(orient="records"),
        "whatif": whatif,
    }
    save_json(bundle, metrics_dir / "followup2_modeling_results.json")
    logger.info(
        "Follow-up #2 modeling complete! Best overall=%s | Best linear=%s | Best tree=%s",
        best_overall_name,
        best_linear_name,
        best_tree_name,
    )
    return bundle
