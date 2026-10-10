"""Fold-local radiomics selection and development stability selection."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


@dataclass
class RadiomicsSelection:
    selected_features: List[str]
    selected_l1_c: Optional[float]
    n_input: int
    n_after_correlation: int
    inner_auc_mean: float = np.nan
    inner_auc_std: float = np.nan
    inner_log_loss_mean: float = np.nan
    inner_log_loss_std: float = np.nan
    selection_method: str = "none"
    search_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SharedRadiomicsSelections:
    """Radiomics selections fitted once and reused before creating model inputs."""

    by_cv_fold: Dict[int, RadiomicsSelection]
    final_development: RadiomicsSelection


def analyze_feature_origin(feature_list):
    n_rad = sum(
        "Radiomics___" in feature or feature == "Radiomics_Probability"
        for feature in feature_list
    )
    n_clin = sum(
        "Clinical___" in feature or feature == "Clinical_Probability"
        for feature in feature_list
    )
    n_mus = sum(
        "Muscle___" in feature or feature == "Muscle_Probability"
        for feature in feature_list
    )
    return f"[Rad: {n_rad} | Clin: {n_clin} | Mus: {n_mus}]"


def process_radiomics_specifically(
        X_train_rad: pd.DataFrame,
        y_train,
        target_n_min: int = 10,
        target_n_max: int = 15,
        corr_threshold: float = 0.90,
        alpha_grid=None,
        inner_cv_folds: int = 3,
        refinement_points: int = 25,
        max_scored_alphas: int = 12,
) -> RadiomicsSelection:
    """
    Select Radiomics L1-logistic C by inner-CV AUC using training data only.

    The 10-15 range is a sparsity constraint, not the optimization target.
    Among feasible C values, a fixed Logistic Regression downstream model chooses
    the C with the highest inner-CV AUC. Log-loss remains an audit metric.
    Near-zero-variance/correlation filtering,
    scaling and L1 logistic selection are re-fitted inside every inner-training fold.
    Clinical and Muscle
    columns never enter this function.
    """
    y_train = np.asarray(y_train, dtype=int)
    if X_train_rad.empty:
        return RadiomicsSelection([], None, 0, 0, selection_method="no-radiomics")
    if X_train_rad.shape[1] < 2:
        features = X_train_rad.columns.tolist()
        return RadiomicsSelection(
            features, None, len(features), len(features), selection_method="single-feature"
        )

    def correlation_filtered_columns(X_fit: pd.DataFrame) -> List[str]:
        informative_columns = [
            column for column in X_fit.columns
            if X_fit[column].nunique(dropna=False) > 1
            and np.isfinite(X_fit[column].var())
            and float(X_fit[column].var()) > 1e-12
        ]
        if not informative_columns:
            return []
        corr_matrix = X_fit[informative_columns].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if (upper[column] > corr_threshold).any()]
        return [column for column in informative_columns if column not in to_drop]

    def selected_positions_at_alpha(X_scaled, y_fit, alpha: float):
        selector = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=float(alpha),
            max_iter=10000,
            random_state=42,
        )
        selector.fit(X_scaled, y_fit)
        return np.flatnonzero(np.abs(selector.coef_.ravel()) > 1e-8)

    full_filtered_columns = correlation_filtered_columns(X_train_rad)

    if len(full_filtered_columns) <= target_n_max:
        return RadiomicsSelection(
            selected_features=full_filtered_columns,
            selected_l1_c=None,
            n_input=X_train_rad.shape[1],
            n_after_correlation=len(full_filtered_columns),
            selection_method="correlation-only",
        )

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 1, 60)

    # Correlation filtering and scaling for the current training scope are
    # fitted once. L1 logistic selection is still re-fitted for every C value.
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X_train_rad[full_filtered_columns])
    alpha_candidates_by_value = {}

    def add_full_scope_candidate(alpha: float, grid_stage: str):
        alpha = float(alpha)
        alpha_key = round(alpha, 15)
        if alpha_key in alpha_candidates_by_value:
            return
        selected_positions = selected_positions_at_alpha(X_full_scaled, y_train, alpha)
        selected = [full_filtered_columns[position] for position in selected_positions]
        n_selected = len(selected)
        if n_selected == 0:
            return
        if n_selected < target_n_min:
            distance = target_n_min - n_selected
        elif n_selected > target_n_max:
            distance = n_selected - target_n_max
        else:
            distance = 0
        alpha_candidates_by_value[alpha_key] = {
            "alpha": float(alpha),
            "selected_features": selected,
            "n_selected_full_scope": n_selected,
            "count_distance": distance,
            "is_feasible": distance == 0,
            "grid_stage": grid_stage,
        }

    initial_grid = sorted({float(alpha) for alpha in alpha_grid if float(alpha) > 0})
    for alpha in initial_grid:
        add_full_scope_candidate(alpha, "initial")

    alpha_candidates = sorted(
        alpha_candidates_by_value.values(), key=lambda candidate: candidate["alpha"]
    )

    if not alpha_candidates:
        raise RuntimeError(
            "Radiomics L1 logistic regression did not select any feature "
            "for the configured C grid."
        )

    # A 50-point log grid can jump across the narrow 10-15 feature region.
    # Refine only its local neighbourhood, rather than expanding the whole grid.
    feasible = [candidate for candidate in alpha_candidates if candidate["is_feasible"]]
    if len(feasible) < 3 and len(alpha_candidates) > 1 and refinement_points > 0:
        if feasible:
            relevant_indices = [
                index for index, candidate in enumerate(alpha_candidates)
                if candidate["is_feasible"]
            ]
        else:
            closest_distance = min(
                candidate["count_distance"] for candidate in alpha_candidates
            )
            relevant_indices = [
                index for index, candidate in enumerate(alpha_candidates)
                if candidate["count_distance"] == closest_distance
            ]
        lower_index = max(min(relevant_indices) - 1, 0)
        upper_index = min(max(relevant_indices) + 1, len(alpha_candidates) - 1)
        lower_alpha = alpha_candidates[lower_index]["alpha"]
        upper_alpha = alpha_candidates[upper_index]["alpha"]
        if upper_alpha > lower_alpha:
            for alpha in np.geomspace(lower_alpha, upper_alpha, refinement_points):
                add_full_scope_candidate(float(alpha), "adaptive-refinement")
        alpha_candidates = sorted(
            alpha_candidates_by_value.values(), key=lambda candidate: candidate["alpha"]
        )
        feasible = [candidate for candidate in alpha_candidates if candidate["is_feasible"]]

    if feasible:
        candidates_to_score = feasible
    else:
        closest_distance = min(candidate["count_distance"] for candidate in alpha_candidates)
        candidates_to_score = [
            candidate for candidate in alpha_candidates
            if candidate["count_distance"] == closest_distance
        ]

    # C values producing exactly the same full-scope feature set are redundant.
    unique_feature_sets = {}
    for candidate in candidates_to_score:
        feature_key = tuple(candidate["selected_features"])
        existing = unique_feature_sets.get(feature_key)
        if existing is None or candidate["alpha"] < existing["alpha"]:
            unique_feature_sets[feature_key] = candidate
    candidates_to_score = sorted(
        unique_feature_sets.values(), key=lambda candidate: candidate["alpha"]
    )
    if len(candidates_to_score) > max_scored_alphas:
        sampled_indices = np.unique(np.round(
            np.linspace(0, len(candidates_to_score) - 1, max_scored_alphas)
        ).astype(int))
        candidates_to_score = [candidates_to_score[index] for index in sampled_indices]

    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_inner_splits = min(inner_cv_folds, min_class_count)
    if n_inner_splits < 2:
        raise ValueError("Radiomics inner CV requires at least two samples in each class.")
    inner_splitter = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=137
    )

    # Cache honest inner-fold preprocessing once. This preserves fold isolation
    # while avoiding repeated correlation/scaling work for every candidate alpha.
    inner_fold_cache = []
    for inner_train_index, inner_validation_index in inner_splitter.split(
            X_train_rad, y_train
    ):
        X_inner_train = X_train_rad.iloc[inner_train_index]
        X_inner_validation = X_train_rad.iloc[inner_validation_index]
        y_inner_train = y_train[inner_train_index]
        y_inner_validation = y_train[inner_validation_index]
        inner_columns = correlation_filtered_columns(X_inner_train)
        inner_scaler = StandardScaler()
        X_inner_train_scaled = inner_scaler.fit_transform(X_inner_train[inner_columns])
        X_inner_validation_scaled = inner_scaler.transform(
            X_inner_validation[inner_columns]
        )
        inner_fold_cache.append({
            "X_train_scaled": X_inner_train_scaled,
            "X_validation_scaled": X_inner_validation_scaled,
            "y_train": y_inner_train,
            "y_validation": y_inner_validation,
        })

    search_rows = []
    for candidate_number, candidate in enumerate(candidates_to_score, start=1):
        fold_auc = []
        fold_log_loss = []
        alpha = candidate["alpha"]
        for inner_fold in inner_fold_cache:
            selected_positions = selected_positions_at_alpha(
                inner_fold["X_train_scaled"], inner_fold["y_train"], alpha
            )
            if len(selected_positions) == 0:
                prevalence = float(np.mean(inner_fold["y_train"]))
                probability = np.repeat(prevalence, len(inner_fold["y_validation"]))
                fold_auc.append(0.5)
                fold_log_loss.append(log_loss(
                    inner_fold["y_validation"], probability, labels=[0, 1]
                ))
                continue
            selector_model = LogisticRegression(
                solver="liblinear", max_iter=5000, random_state=42
            )
            selector_model.fit(
                inner_fold["X_train_scaled"][:, selected_positions],
                inner_fold["y_train"],
            )
            probability = selector_model.predict_proba(
                inner_fold["X_validation_scaled"][:, selected_positions]
            )[:, 1]
            fold_auc.append(roc_auc_score(inner_fold["y_validation"], probability))
            fold_log_loss.append(log_loss(
                inner_fold["y_validation"], probability, labels=[0, 1]
            ))

        candidate["inner_auc_mean"] = float(np.mean(fold_auc))
        candidate["inner_auc_std"] = (
            float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0
        )
        candidate["inner_log_loss_mean"] = float(np.mean(fold_log_loss))
        candidate["inner_log_loss_std"] = (
            float(np.std(fold_log_loss, ddof=1))
            if len(fold_log_loss) > 1 else 0.0
        )
        search_rows.append({
            "Candidate": candidate_number,
            "L1_Logistic_C": alpha,
            "Grid_Stage": candidate["grid_stage"],
            "N_Selected_Full_Scope": candidate["n_selected_full_scope"],
            "Within_Target_10_15": candidate["is_feasible"],
            "Inner_CV_AUC_Mean": candidate["inner_auc_mean"],
            "Inner_CV_AUC_SD": candidate["inner_auc_std"],
            "Inner_CV_LogLoss_Mean": candidate["inner_log_loss_mean"],
            "Inner_CV_LogLoss_SD": candidate["inner_log_loss_std"],
            "Selected_Features_Full_Scope": " | ".join(candidate["selected_features"]),
        })

    target_midpoint = (target_n_min + target_n_max) / 2
    best_candidate = max(
        candidates_to_score,
        key=lambda candidate: (
            candidate["inner_auc_mean"],
            -candidate["inner_auc_std"],
            -candidate["inner_log_loss_mean"],
            -abs(candidate["n_selected_full_scope"] - target_midpoint),
            -candidate["alpha"],
        ),
    )
    selected_l1_c = best_candidate["alpha"]
    selected_features = best_candidate["selected_features"]
    print(
        f"      [Radiomics] {X_train_rad.shape[1]} -> {len(full_filtered_columns)} after corr -> "
        f"{len(selected_features)} after L1-logistic (C={selected_l1_c:.6g}, "
        f"inner AUC={best_candidate['inner_auc_mean']:.4f}, "
        f"LogLoss={best_candidate['inner_log_loss_mean']:.4f}, "
        f"scored C values={len(candidates_to_score)})"
    )
    return RadiomicsSelection(
        selected_features=selected_features,
        selected_l1_c=selected_l1_c,
        n_input=X_train_rad.shape[1],
        n_after_correlation=len(full_filtered_columns),
        inner_auc_mean=best_candidate["inner_auc_mean"],
        inner_auc_std=best_candidate["inner_auc_std"],
        inner_log_loss_mean=best_candidate["inner_log_loss_mean"],
        inner_log_loss_std=best_candidate["inner_log_loss_std"],
        selection_method=(
            "adaptive inner-CV-AUC NZV+corr+L1-logistic (cached preprocessing)"
        ),
        search_results=search_rows,
    )


def select_features_for_training_split(
        X_train: pd.DataFrame,
        y_train,
        fitted_radiomics_selection: Optional[RadiomicsSelection] = None,
) -> tuple[List[str], RadiomicsSelection]:
    """Fuse selected Radiomics with every Clinical/Muscle feature."""
    radiomics_columns = [column for column in X_train.columns if column.startswith("Radiomics___")]
    non_radiomics_columns = [column for column in X_train.columns if not column.startswith("Radiomics___")]
    if radiomics_columns:
        radiomics_selection = fitted_radiomics_selection
        if radiomics_selection is None:
            radiomics_selection = process_radiomics_specifically(X_train[radiomics_columns], y_train)
        missing_selected = [
            feature for feature in radiomics_selection.selected_features
            if feature not in radiomics_columns
        ]
        if missing_selected:
            raise ValueError(
                "Cached Radiomics selection contains columns unavailable in this dataset: "
                f"{missing_selected[:20]}"
            )
    else:
        radiomics_selection = RadiomicsSelection(
            [], None, 0, 0, selection_method="not-applicable"
        )
    selected_set = set(radiomics_selection.selected_features + non_radiomics_columns)
    selected_features = [column for column in X_train.columns if column in selected_set]
    return selected_features, radiomics_selection


def prepare_shared_radiomics_selections(
        X_radiomics_development: pd.DataFrame,
        y_development: np.ndarray,
        cv_folds: int,
        output_dir: Path,
) -> SharedRadiomicsSelections:
    """
    Finish Radiomics selection before building Radiomics/Fusion model inputs.

    A separate selection is required for each CV training fold to keep its
    validation fold unseen. A full-development selection is kept for audit.
    Final Radiomics/Fusion models use features selected in at least 60% of the
    outer training folds.
    """
    radiomics_columns = [
        column for column in X_radiomics_development.columns
        if column.startswith("Radiomics___")
    ]
    if not radiomics_columns:
        raise ValueError("Shared Radiomics preparation received no Radiomics columns.")

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    by_cv_fold = {}
    audit_rows = []
    alpha_search_rows = []
    print("\n[Phase 1] Preparing shared Radiomics selections before fusion...")
    for fold_number, (train_index, _) in enumerate(
            splitter.split(X_radiomics_development, y_development), start=1
    ):
        selection = process_radiomics_specifically(
            X_radiomics_development.iloc[train_index][radiomics_columns],
            y_development[train_index],
        )
        by_cv_fold[fold_number] = selection
        audit_rows.append({
            "Scope": f"CV_Fold_{fold_number}_Train",
            "Radiomics_Input": selection.n_input,
            "Radiomics_After_Correlation": selection.n_after_correlation,
            "Radiomics_Selected": len(selection.selected_features),
            "Radiomics_L1_Logistic_C": selection.selected_l1_c,
            "Inner_CV_AUC_Mean": selection.inner_auc_mean,
            "Inner_CV_AUC_SD": selection.inner_auc_std,
            "Inner_CV_LogLoss_Mean": selection.inner_log_loss_mean,
            "Inner_CV_LogLoss_SD": selection.inner_log_loss_std,
            "Selection_Method": selection.selection_method,
            "Selected_Features": " | ".join(selection.selected_features),
        })
        for search_row in selection.search_results:
            alpha_search_rows.append({"Scope": f"CV_Fold_{fold_number}_Train", **search_row})

    full_development_audit = process_radiomics_specifically(
        X_radiomics_development[radiomics_columns],
        y_development,
    )
    audit_rows.append({
        "Scope": "Full_Development_AUC_Audit",
        "Radiomics_Input": full_development_audit.n_input,
        "Radiomics_After_Correlation": full_development_audit.n_after_correlation,
        "Radiomics_Selected": len(full_development_audit.selected_features),
        "Radiomics_L1_Logistic_C": full_development_audit.selected_l1_c,
        "Inner_CV_AUC_Mean": full_development_audit.inner_auc_mean,
        "Inner_CV_AUC_SD": full_development_audit.inner_auc_std,
        "Inner_CV_LogLoss_Mean": full_development_audit.inner_log_loss_mean,
        "Inner_CV_LogLoss_SD": full_development_audit.inner_log_loss_std,
        "Selection_Method": full_development_audit.selection_method,
        "Selected_Features": " | ".join(full_development_audit.selected_features),
    })
    for search_row in full_development_audit.search_results:
        alpha_search_rows.append({
            "Scope": "Full_Development_AUC_Audit", **search_row
        })

    radiomics_counter = Counter(
        feature
        for selection in by_cv_fold.values()
        for feature in selection.selected_features
    )
    minimum_fold_count = int(np.ceil(0.60 * cv_folds))
    stable_features = [
        feature for feature in radiomics_columns
        if radiomics_counter[feature] >= minimum_fold_count
    ]
    if stable_features:
        stability_method = f"outer-fold-stability>={minimum_fold_count}/{cv_folds}"
    else:
        stable_features = full_development_audit.selected_features.copy()
        stability_method = "fallback-full-development-inner-CV-AUC"

    final_selection = RadiomicsSelection(
        selected_features=stable_features,
        selected_l1_c=None,
        n_input=full_development_audit.n_input,
        n_after_correlation=full_development_audit.n_after_correlation,
        inner_auc_mean=full_development_audit.inner_auc_mean,
        inner_auc_std=full_development_audit.inner_auc_std,
        inner_log_loss_mean=full_development_audit.inner_log_loss_mean,
        inner_log_loss_std=full_development_audit.inner_log_loss_std,
        selection_method=stability_method,
    )
    audit_rows.append({
        "Scope": "Final_Development_Stable",
        "Radiomics_Input": final_selection.n_input,
        "Radiomics_After_Correlation": final_selection.n_after_correlation,
        "Radiomics_Selected": len(final_selection.selected_features),
        "Radiomics_L1_Logistic_C": np.nan,
        "Inner_CV_AUC_Mean": final_selection.inner_auc_mean,
        "Inner_CV_AUC_SD": final_selection.inner_auc_std,
        "Inner_CV_LogLoss_Mean": final_selection.inner_log_loss_mean,
        "Inner_CV_LogLoss_SD": final_selection.inner_log_loss_std,
        "Selection_Method": final_selection.selection_method,
        "Selected_Features": " | ".join(final_selection.selected_features),
    })

    stable_set = set(stable_features)
    stability_rows = [{
        "Feature": feature,
        "Fold_Count": radiomics_counter[feature],
        "Selection_Rate": radiomics_counter[feature] / cv_folds,
        "Used_In_Final_Model": feature in stable_set,
    } for feature in radiomics_columns]
    pd.DataFrame(stability_rows).sort_values(
        ["Fold_Count", "Feature"], ascending=[False, True]
    ).to_csv(output_dir / "radiomics_stability.csv", index=False)
    pd.DataFrame(alpha_search_rows).to_csv(
        output_dir / "radiomics_l1_c_search.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "radiomics_selection_shared.csv", index=False
    )
    print(
        f"   shared final Radiomics set: {len(final_selection.selected_features)} features; "
        f"stability rule={stability_method}; now ready for Radiomics-only and Full Fusion."
    )
    return SharedRadiomicsSelections(
        by_cv_fold=by_cv_fold,
        final_development=final_selection,
    )
