# ============================================================
# ML9 AUC-FIRST STANDARD-MODEL PIPELINE (2026-09-13)
# Single / pairwise / full early fusion + pairwise / full OOF late fusion
# Sampler ablation (none / class-weight / train-fold SMOTE)
# Full standard-model meta benchmark; no custom estimators
# Radiomics L1-logistic C by adaptive inner-CV AUC
#
# Two cohorts, not three. The 20% internal hold-out of the 2026-08-17 run is
# gone: every aligned internal case is development data, which is what CV,
# feature selection, Optuna, the final fits and the OOF decision threshold all
# see. The external cohort is the only test set, scored once and never allowed
# to influence selection. SHAP therefore explains external cases.
# ============================================================

import json
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# Repository layout: train/train.py, data/processed/, fusion_subsidence/,
# outputs/ and artifacts/. PROJECT_ROOT is the repository root, found from this
# file's location so the script runs from any working directory. The web app's
# config.py resolves the same root, so both read artifacts/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SRC_DIR = PROJECT_ROOT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The publication labels sit next to this file. Add that directory explicitly so
# the import works whether this script is run directly or loaded by path from
# one of the regeneration scripts.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shap_publication_labels as shap_labels
from fusion_subsidence.model_components import SelectiveStandardScaler


# --- OPTIONAL LIBRARIES SETUP ---
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = SMOTENC = ImbPipeline = None
    IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost will be skipped.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not installed. LightGBM will be skipped.")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Winners will use default parameters.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# 1. DATA FUSION
# ============================================================

class DataFusionPipeline:
    def __init__(self, sources: Dict[str, str]):
        self.sources = sources
        self.dfs: Dict[str, pd.DataFrame] = {}
        self.common_index = None
        self.label_col = None

    def load_and_align(self):
        print("[DataFusion] Loading data...")
        loaded_indices = []
        for name, path in self.sources.items():
            try:
                df = pd.read_csv(path, index_col=0).dropna(axis=1, how="all")
            except Exception as exc:
                raise RuntimeError(f"Cannot load {name}: {path} ({exc})") from exc
            if "Label" not in df.columns:
                raise ValueError(f"Internal {name} data has no Label column: {path}")
            if df.index.has_duplicates:
                raise ValueError(f"Internal {name} data contains duplicate case IDs.")
            self.dfs[name] = df
            loaded_indices.append(df.index)
            print(f"   loaded {name}: {df.shape}")

        self.common_index = loaded_indices[0]
        for index in loaded_indices[1:]:
            self.common_index = self.common_index.intersection(index, sort=False)
        if len(self.common_index) == 0:
            raise ValueError("No common internal cases across data sources.")

        first_source = next(iter(self.sources))
        reference_label = self.dfs[first_source].loc[self.common_index, "Label"].astype(int)
        for name, df in self.dfs.items():
            source_label = df.loc[self.common_index, "Label"].astype(int)
            if not source_label.equals(reference_label):
                mismatch = reference_label.index[source_label.to_numpy() != reference_label.to_numpy()].tolist()
                raise ValueError(f"Label mismatch between {first_source} and {name}: {mismatch[:10]}")
        unexpected_labels = set(reference_label.unique()) - {0, 1}
        if unexpected_labels:
            raise ValueError(f"Internal Label must be binary 0/1; found {unexpected_labels}")

        self.label_col = reference_label
        counts = reference_label.value_counts().sort_index().to_dict()
        print(f"   aligned samples: {len(self.common_index)} | Label distribution: {counts}")

    def get_merged_dataset(self, combination_keys: List[str]) -> pd.DataFrame:
        if self.common_index is None:
            raise RuntimeError("Call load_and_align() before get_merged_dataset().")
        merged = []
        for name in combination_keys:
            df = self.dfs[name].loc[self.common_index].drop(columns=["Label"]).copy()
            df.columns = [f"{name}___{column}" for column in df.columns]
            merged.append(df)
        X = pd.concat(merged, axis=1)
        X["Label"] = self.label_col
        return X


# ============================================================
# 2. RADIOMICS-ONLY FEATURE SELECTION
# ============================================================

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


# ============================================================
# 3. TRAIN-FOLD SCALING AND OPTIONAL IMBALANCE HANDLING
# ============================================================

CLINICAL_CATEGORICAL_BASE_COLUMNS = {
    "Level",
    "Gender",
    "Tobacco",
    "Drink",
    "Diabetes",
    "Pos_lat",
    "Pos_AP",
    "Cage_type",
    "DBM",
    "Condition",
}


def get_categorical_columns(columns) -> List[str]:
    expected = {f"Clinical___{column}" for column in CLINICAL_CATEGORICAL_BASE_COLUMNS}
    return [column for column in columns if column in expected]


SAMPLER_PREFERENCE = {"none": 2, "class_weight": 1, "smote": 0}


def get_sampler_choices(estimator_template: SklearnPipeline) -> List[str]:
    """Return valid imbalance strategies for an estimator, simplest first."""
    estimator_name, _ = estimator_template.steps[-1]
    choices = ["none"]
    if estimator_name in {"rf", "svm", "lr", "lgbm", "xgb"}:
        choices.append("class_weight")
    if IMBLEARN_AVAILABLE:
        choices.append("smote")
    return choices


def build_training_pipeline(
        estimator_template: SklearnPipeline,
        feature_columns,
        sampler_choice: str = "none",
        y_train=None,
) -> Any:
    """Build a leakage-safe preprocessing/model pipeline for one sampler choice."""
    if sampler_choice not in {"none", "class_weight", "smote"}:
        raise ValueError(f"Unknown sampler choice: {sampler_choice}")
    categorical_columns = get_categorical_columns(feature_columns)
    model_steps = [(name, clone(model)) for name, model in estimator_template.steps]
    preprocessing_steps = [
        ("pre", SelectiveStandardScaler(categorical_columns=categorical_columns))
    ]
    if sampler_choice == "smote":
        if not IMBLEARN_AVAILABLE:
            raise ImportError("sampler='smote' requires imbalanced-learn.")
        if categorical_columns:
            sampler = SMOTENC(
                categorical_features=categorical_columns,
                sampling_strategy="auto",
                random_state=42,
                k_neighbors=5,
            )
        else:
            sampler = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
        pipeline = ImbPipeline([
            *preprocessing_steps,
            ("smote", sampler),
            *model_steps,
        ])
    else:
        pipeline = SklearnPipeline([*preprocessing_steps, *model_steps])

    if sampler_choice == "class_weight":
        if y_train is None:
            raise ValueError("class_weight requires the current training labels.")
        estimator_name = model_steps[-1][0]
        if estimator_name == "xgb":
            counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
            if counts[1] == 0:
                raise ValueError("XGBoost class weighting requires both classes.")
            pipeline.set_params(
                **{"xgb__scale_pos_weight": float(counts[0] / counts[1])}
            )
        elif estimator_name in {"rf", "svm", "lr", "lgbm"}:
            pipeline.set_params(**{f"{estimator_name}__class_weight": "balanced"})
        else:
            raise ValueError(
                f"class_weight is unsupported for estimator step '{estimator_name}'."
            )
    return pipeline


# ============================================================
# 4. CROSS-VALIDATION, MODEL SELECTION, AND FINAL OPTUNA
# ============================================================

@dataclass
class PreparedFold:
    fold: int
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    selected_features: List[str]
    radiomics_selection: RadiomicsSelection


@dataclass
class CVEvaluation:
    metrics_cv_mean: pd.Series
    metrics_cv_std: pd.Series
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    mean_n_features: float
    std_n_features: float


@dataclass
class MLResults:
    fitted_model: Any
    X_train_final: pd.DataFrame
    selected_features: List[str]
    best_params: Dict[str, Any]
    sampler_choice: str
    optuna_best_cv_auc: float
    decision_threshold: float
    selected_cv_evaluation: CVEvaluation


def compute_metrics(y_true, probability, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, probability),
        "Sens": recall_score(y_true, prediction, zero_division=0),
        "Spec": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "Prec": precision_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "Acc": accuracy_score(y_true, prediction),
        "Brier": brier_score_loss(y_true, probability),
        "LogLoss": log_loss(y_true, probability, labels=[0, 1]),
    }


def build_auc_comparison_table(
        prediction_table: pd.DataFrame,
        reference_dataset: str = "Clinical",
        n_bootstrap: int = 2000,
        random_state: int = 42,
) -> pd.DataFrame:
    """Paired stratified-bootstrap AUC differences against a reference branch."""
    required = {
        "Dataset", "Case_ID", "True_Label", "Probability_Subsidence"
    }
    missing = required - set(prediction_table.columns)
    if missing:
        raise ValueError(f"AUC comparison table is missing columns: {sorted(missing)}")
    reference = prediction_table.loc[
        prediction_table["Dataset"] == reference_dataset,
        ["Case_ID", "True_Label", "Probability_Subsidence"],
    ].rename(columns={"Probability_Subsidence": "Reference_Probability"})
    if reference.empty:
        raise ValueError(f"Reference dataset '{reference_dataset}' has no predictions.")
    if reference["Case_ID"].duplicated().any():
        raise ValueError(f"Reference dataset '{reference_dataset}' has duplicate case IDs.")

    comparison_rows = []
    for dataset_number, dataset_name in enumerate(
            prediction_table["Dataset"].drop_duplicates(), start=1
    ):
        candidate = prediction_table.loc[
            prediction_table["Dataset"] == dataset_name,
            ["Case_ID", "True_Label", "Probability_Subsidence"],
        ].rename(columns={
            "True_Label": "Candidate_Label",
            "Probability_Subsidence": "Candidate_Probability",
        })
        if candidate["Case_ID"].duplicated().any():
            raise ValueError(f"Dataset '{dataset_name}' has duplicate case IDs.")
        aligned = reference.merge(candidate, on="Case_ID", how="inner", validate="one_to_one")
        if len(aligned) != len(reference) or len(aligned) != len(candidate):
            raise ValueError(
                f"Cannot pair every prediction for {dataset_name} against {reference_dataset}."
            )
        if not np.array_equal(
                aligned["True_Label"].to_numpy(dtype=int),
                aligned["Candidate_Label"].to_numpy(dtype=int),
        ):
            raise ValueError(f"Label mismatch in paired AUC comparison for {dataset_name}.")

        y = aligned["True_Label"].to_numpy(dtype=int)
        reference_probability = aligned["Reference_Probability"].to_numpy(dtype=float)
        candidate_probability = aligned["Candidate_Probability"].to_numpy(dtype=float)
        reference_auc = roc_auc_score(y, reference_probability)
        candidate_auc = roc_auc_score(y, candidate_probability)
        positive_indices = np.flatnonzero(y == 1)
        negative_indices = np.flatnonzero(y == 0)
        rng = np.random.default_rng(random_state + dataset_number)
        candidate_bootstrap = []
        difference_bootstrap = []
        for _ in range(n_bootstrap):
            sample_indices = np.concatenate([
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            ])
            sampled_y = y[sample_indices]
            sampled_reference_auc = roc_auc_score(
                sampled_y, reference_probability[sample_indices]
            )
            sampled_candidate_auc = roc_auc_score(
                sampled_y, candidate_probability[sample_indices]
            )
            candidate_bootstrap.append(sampled_candidate_auc)
            difference_bootstrap.append(
                sampled_candidate_auc - sampled_reference_auc
            )
        candidate_bootstrap = np.asarray(candidate_bootstrap)
        difference_bootstrap = np.asarray(difference_bootstrap)
        comparison_rows.append({
            "Reference_Dataset": reference_dataset,
            "Candidate_Dataset": dataset_name,
            "N_External": len(aligned),
            "Reference_AUC": reference_auc,
            "Candidate_AUC": candidate_auc,
            "Candidate_AUC_CI_Lower": np.quantile(candidate_bootstrap, 0.025),
            "Candidate_AUC_CI_Upper": np.quantile(candidate_bootstrap, 0.975),
            "AUC_Difference_vs_Reference": candidate_auc - reference_auc,
            "AUC_Difference_CI_Lower": np.quantile(difference_bootstrap, 0.025),
            "AUC_Difference_CI_Upper": np.quantile(difference_bootstrap, 0.975),
            "Bootstrap_Probability_Candidate_Better": np.mean(
                difference_bootstrap > 0
            ),
            "Bootstrap_Replicates": n_bootstrap,
        })
    return pd.DataFrame(comparison_rows)


def choose_oof_threshold(y_true, probability) -> float:
    """Choose a classification cutoff on development OOF predictions by Youden J."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    finite_thresholds = thresholds[finite]
    youden_j = true_positive_rate[finite] - false_positive_rate[finite]
    best_j = np.max(youden_j)
    tied = np.flatnonzero(np.isclose(youden_j, best_j))
    best_index = tied[np.argmin(np.abs(finite_thresholds[tied] - 0.5))]
    return float(np.clip(finite_thresholds[best_index], 0.0, 1.0))


def safe_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def prepare_cv_folds(
        X_development: pd.DataFrame,
        y_development: np.ndarray,
        combo_name: str,
        cv_folds: int,
        output_dir: Path,
        shared_radiomics_selections: Optional[SharedRadiomicsSelections] = None,
) -> List[PreparedFold]:
    """Build fold inputs after Radiomics has already been selected."""
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    prepared_folds = []
    feature_counter = Counter()
    selection_rows = []

    for fold_number, (train_index, validation_index) in enumerate(
            splitter.split(X_development, y_development), start=1
    ):
        X_train = X_development.iloc[train_index].copy()
        X_validation = X_development.iloc[validation_index].copy()
        y_train = y_development[train_index]
        y_validation = y_development[validation_index]
        has_radiomics = any(
            column.startswith("Radiomics___") for column in X_train.columns
        )
        if has_radiomics:
            if shared_radiomics_selections is None:
                raise ValueError(
                    f"{combo_name} contains Radiomics but no shared pre-fusion selection was supplied."
                )
            fitted_radiomics_selection = shared_radiomics_selections.by_cv_fold[fold_number]
        else:
            fitted_radiomics_selection = None
        selected_features, rad_selection = select_features_for_training_split(
            X_train,
            y_train,
            fitted_radiomics_selection=fitted_radiomics_selection,
        )
        feature_counter.update(selected_features)
        selection_rows.append({
            "Dataset": combo_name,
            "Fold": fold_number,
            "Radiomics_Input": rad_selection.n_input,
            "Radiomics_After_Correlation": rad_selection.n_after_correlation,
            "Radiomics_Selected": len(rad_selection.selected_features),
            "Radiomics_L1_Logistic_C": rad_selection.selected_l1_c,
            "Radiomics_Inner_CV_AUC": rad_selection.inner_auc_mean,
            "Radiomics_Inner_CV_LogLoss": rad_selection.inner_log_loss_mean,
            "Radiomics_Selection_Method": rad_selection.selection_method,
            "Clinical_Selected": sum(feature.startswith("Clinical___") for feature in selected_features),
            "Muscle_Selected": sum(feature.startswith("Muscle___") for feature in selected_features),
            "Total_Selected": len(selected_features),
        })
        prepared_folds.append(PreparedFold(
            fold=fold_number,
            X_train=X_train[selected_features],
            X_validation=X_validation[selected_features],
            y_train=y_train,
            y_validation=y_validation,
            selected_features=selected_features,
            radiomics_selection=rad_selection,
        ))

    safe_combo = safe_file_name(combo_name)
    feature_track = pd.DataFrame(feature_counter.most_common(), columns=["Feature", "Fold_Count"])
    if not feature_track.empty:
        feature_track["Selection_Rate"] = feature_track["Fold_Count"] / cv_folds
        feature_track["Origin"] = feature_track["Feature"].str.split("___").str[0]
    feature_track.to_csv(output_dir / f"features_track_{safe_combo}.csv", index=False)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(
            output_dir / f"model_inputs_{safe_combo}.csv", index=False
        )
    return prepared_folds


def build_late_fusion_table(
        probability_frames: Dict[str, pd.DataFrame],
        modalities: List[str],
        require_fold: bool,
) -> pd.DataFrame:
    """Align modality probabilities by case ID and verify labels/folds exactly."""
    missing_modalities = [name for name in modalities if name not in probability_frames]
    if missing_modalities:
        raise ValueError(f"Late Fusion is missing base predictions for: {missing_modalities}")

    reference = None
    for modality in modalities:
        required_columns = {"Case_ID", "True_Label", "Probability_Subsidence"}
        if require_fold:
            required_columns.add("Fold")
        frame = probability_frames[modality].copy()
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Late Fusion {modality} predictions are missing columns: {sorted(missing_columns)}"
            )
        frame["Case_ID"] = frame["Case_ID"].astype(str)
        if frame["Case_ID"].duplicated().any():
            duplicates = frame.loc[frame["Case_ID"].duplicated(), "Case_ID"].tolist()
            raise ValueError(f"Duplicate {modality} Late Fusion case IDs: {duplicates[:20]}")
        frame = frame.set_index("Case_ID").sort_index()

        if reference is None:
            reference_columns = ["True_Label"] + (["Fold"] if require_fold else [])
            reference = frame[reference_columns].copy()
        else:
            if not frame.index.equals(reference.index):
                missing = reference.index.difference(frame.index).tolist()
                extra = frame.index.difference(reference.index).tolist()
                raise ValueError(
                    f"Late Fusion case mismatch for {modality}; "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
            if not np.array_equal(
                    frame["True_Label"].to_numpy(dtype=int),
                    reference["True_Label"].to_numpy(dtype=int),
            ):
                raise ValueError(f"Late Fusion label mismatch for {modality}.")
            if require_fold and not np.array_equal(
                    frame["Fold"].to_numpy(dtype=int),
                    reference["Fold"].to_numpy(dtype=int),
            ):
                raise ValueError(f"Late Fusion fold mismatch for {modality}.")

        reference[f"{modality}_Probability"] = frame[
            "Probability_Subsidence"
        ].to_numpy(dtype=float)

    return reference


def prepare_late_fusion_folds(
        stacking_table: pd.DataFrame,
        modalities: List[str],
) -> List[PreparedFold]:
    """Create meta-model folds from already out-of-fold base probabilities."""
    feature_columns = [f"{modality}_Probability" for modality in modalities]
    prepared_folds = []
    for fold_number in sorted(stacking_table["Fold"].astype(int).unique()):
        validation_mask = stacking_table["Fold"].astype(int) == fold_number
        training_mask = ~validation_mask
        X_train = stacking_table.loc[training_mask, feature_columns].copy()
        X_validation = stacking_table.loc[validation_mask, feature_columns].copy()
        y_train = stacking_table.loc[training_mask, "True_Label"].to_numpy(dtype=int)
        y_validation = stacking_table.loc[validation_mask, "True_Label"].to_numpy(dtype=int)
        prepared_folds.append(PreparedFold(
            fold=int(fold_number),
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            selected_features=feature_columns,
            radiomics_selection=RadiomicsSelection(
                [], None, 0, 0, selection_method="not-applicable-to-late-fusion"
            ),
        ))
    return prepared_folds


def evaluate_model_cv(
        prepared_folds: List[PreparedFold],
        estimator_template: SklearnPipeline,
        model_params: Optional[Dict[str, Any]] = None,
        sampler_choice: str = "none",
        threshold: float = 0.5,
) -> CVEvaluation:
    fold_rows = []
    oof_rows = []
    n_features = []
    for prepared in prepared_folds:
        print(f"      fold {prepared.fold}/{len(prepared_folds)}", end="\r", flush=True)
        model = build_training_pipeline(
            estimator_template,
            prepared.selected_features,
            sampler_choice=sampler_choice,
            y_train=prepared.y_train,
        )
        if model_params:
            model.set_params(**model_params)
        model.fit(prepared.X_train, prepared.y_train)
        probability = model.predict_proba(prepared.X_validation)[:, 1]
        row = {
            "Fold": prepared.fold,
            "Sampler": sampler_choice,
            "Threshold": threshold,
            **compute_metrics(prepared.y_validation, probability, threshold=threshold),
        }
        fold_rows.append(row)
        for case_id, true_label, case_probability in zip(
                prepared.X_validation.index,
                prepared.y_validation,
                probability,
        ):
            oof_rows.append({
                "Fold": prepared.fold,
                "Case_ID": str(case_id),
                "True_Label": int(true_label),
                "Probability_Subsidence": float(case_probability),
            })
        n_features.append(len(prepared.selected_features))
    print(" " * 40, end="\r", flush=True)

    fold_metrics = pd.DataFrame(fold_rows)
    metric_columns = [
        "AUC", "Sens", "Spec", "Prec", "F1", "Acc", "Brier", "LogLoss"
    ]
    return CVEvaluation(
        metrics_cv_mean=fold_metrics[metric_columns].mean(),
        metrics_cv_std=fold_metrics[metric_columns].std(ddof=1),
        fold_metrics=fold_metrics,
        oof_predictions=pd.DataFrame(oof_rows),
        mean_n_features=float(np.mean(n_features)),
        std_n_features=float(np.std(n_features, ddof=1)) if len(n_features) > 1 else 0.0,
    )


def evaluation_rank_key(evaluation: CVEvaluation):
    """Rank finished candidates by the prespecified discrimination endpoint, AUC."""
    return (
        evaluation.metrics_cv_mean["AUC"],
        -evaluation.metrics_cv_std["AUC"],
        -evaluation.metrics_cv_mean["Brier"],
    )


def choose_sampler_candidate(
        candidates: Dict[str, CVEvaluation],
        auc_tolerance: float = 0.005,
) -> tuple[str, CVEvaluation]:
    """Select by AUC; prefer simpler imbalance handling for near-ties."""
    if not candidates:
        raise ValueError("No valid sampler candidate was supplied.")
    best_auc = max(
        evaluation.metrics_cv_mean["AUC"] for evaluation in candidates.values()
    )
    eligible = {
        sampler: evaluation
        for sampler, evaluation in candidates.items()
        if evaluation.metrics_cv_mean["AUC"] >= best_auc - auc_tolerance
    }
    selected_sampler, selected_evaluation = max(
        eligible.items(),
        key=lambda item: (
            SAMPLER_PREFERENCE[item[0]],
            evaluation_rank_key(item[1]),
        ),
    )
    return selected_sampler, selected_evaluation


def tune_model_candidate(
        combo_name: str,
        model_name: str,
        prepared_folds: List[PreparedFold],
        estimator_template: SklearnPipeline,
        optuna_fn: Optional[Callable],
        n_trials: int,
        output_dir: Path,
        baseline_evaluation: CVEvaluation,
        sampler_choice: str,
) -> Dict[str, Any]:
    """Tune one shortlisted model by mean validation-fold AUC."""
    selected_params: Dict[str, Any] = {}
    selected_sampler = sampler_choice
    selected_evaluation = baseline_evaluation
    selected_stage = "Default screening"
    optuna_best_cv_auc = np.nan

    if OPTUNA_AVAILABLE and optuna_fn is not None:
        def objective(trial):
            params = optuna_fn(trial)
            fold_auc = []
            for prepared in prepared_folds:
                model = build_training_pipeline(
                    estimator_template,
                    prepared.selected_features,
                    sampler_choice=selected_sampler,
                    y_train=prepared.y_train,
                )
                model.set_params(**params)
                model.fit(prepared.X_train, prepared.y_train)
                probability = model.predict_proba(prepared.X_validation)[:, 1]
                fold_auc.append(
                    roc_auc_score(prepared.y_validation, probability)
                )
            return float(np.mean(fold_auc))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def report_trial_progress(current_study, completed_trial):
            print(
                f"      Optuna trial {completed_trial.number + 1}/{n_trials}: "
                f"AUC={completed_trial.value:.4f}, best={current_study.best_value:.4f}"
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[report_trial_progress],
        )
        tuned_params = dict(study.best_trial.params)
        optuna_best_cv_auc = float(study.best_value)
        trials_path = output_dir / (
            f"optuna_trials_{safe_file_name(combo_name)}_{safe_file_name(model_name)}.csv"
        )
        study.trials_dataframe().to_csv(trials_path, index=False)
        tuned_evaluation = evaluate_model_cv(
            prepared_folds,
            estimator_template,
            model_params=tuned_params,
            sampler_choice=selected_sampler,
        )
        if (
                tuned_evaluation.metrics_cv_mean["AUC"]
                >= baseline_evaluation.metrics_cv_mean["AUC"] - 1e-12
        ):
            selected_params = tuned_params
            selected_evaluation = tuned_evaluation
            selected_stage = "Optuna"
        else:
            selected_stage = "Default fallback (Optuna AUC lower)"
        print(
            f"      Optuna best CV AUC={optuna_best_cv_auc:.4f} | "
            f"selected={selected_stage}, sampler={selected_sampler}, params={selected_params}"
        )
    else:
        print(
            "      Optuna unavailable/not applicable: retaining default parameters "
            f"with sampler={selected_sampler}."
        )

    return {
        "model_name": model_name,
        "sampler_choice": selected_sampler,
        "best_params": selected_params,
        "evaluation": selected_evaluation,
        "stage": selected_stage,
        "optuna_best_cv_auc": optuna_best_cv_auc,
    }


def fit_final_model(
        combo_name: str,
        model_name: str,
        X_development: pd.DataFrame,
        y_development: np.ndarray,
        estimator_template: SklearnPipeline,
        selected_evaluation: CVEvaluation,
        best_params: Dict[str, Any],
        optuna_best_cv_auc: float,
        output_dir: Path,
        sampler_choice: str,
        final_radiomics_selection: Optional[RadiomicsSelection] = None,
) -> MLResults:
    """Fit the selected configuration on development data and retain its OOF cutoff."""
    oof = selected_evaluation.oof_predictions
    decision_threshold = choose_oof_threshold(
        oof["True_Label"], oof["Probability_Subsidence"]
    )

    final_features, rad_selection = select_features_for_training_split(
        X_development,
        y_development,
        fitted_radiomics_selection=final_radiomics_selection,
    )
    print(f"      Final features: {len(final_features)} {analyze_feature_origin(final_features)}")
    print(f"      OOF Youden threshold: {decision_threshold:.4f}")
    final_model = build_training_pipeline(
        estimator_template,
        final_features,
        sampler_choice=sampler_choice,
        y_train=y_development,
    )
    if best_params:
        final_model.set_params(**best_params)
    X_train_final = X_development[final_features].copy()
    final_model.fit(X_train_final, y_development)

    feature_rows = []
    for feature in final_features:
        feature_origin = (
            feature.removesuffix("_Probability")
            if feature.endswith("_Probability")
            else feature.split("___", 1)[0]
        )
        feature_rows.append({
            "Dataset": combo_name,
            "Model": model_name,
            "Feature": feature,
            "Origin": feature_origin,
            "Radiomics_L1_Logistic_C": rad_selection.selected_l1_c,
            "Radiomics_Selection_Method": rad_selection.selection_method,
        })
    pd.DataFrame(feature_rows).to_csv(
        output_dir / f"final_features_{safe_file_name(combo_name)}.csv", index=False
    )
    return MLResults(
        fitted_model=final_model,
        X_train_final=X_train_final,
        selected_features=final_features,
        best_params=best_params,
        sampler_choice=sampler_choice,
        optuna_best_cv_auc=optuna_best_cv_auc,
        decision_threshold=decision_threshold,
        selected_cv_evaluation=selected_evaluation,
    )


def _to_builtin_scalar(value):
    """Convert numpy/pandas scalar values into portable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_clinical_form_schema(clinical_result: MLResults) -> List[Dict[str, Any]]:
    """Describe exactly the selected Clinical inputs required by the web form."""
    schema = []
    for feature in clinical_result.selected_features:
        if not feature.startswith("Clinical___"):
            continue
        source_name = feature.split("___", 1)[1]
        series = clinical_result.X_train_final[feature].dropna()
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        categorical = source_name in CLINICAL_CATEGORICAL_BASE_COLUMNS
        integer_like = bool(
            len(numeric_series)
            and np.allclose(
                numeric_series.to_numpy(dtype=float),
                np.round(numeric_series.to_numpy(dtype=float)),
            )
        )
        field = {
            "name": feature,
            "source_name": source_name,
            "label": source_name.replace("_", " "),
            "input_type": "select" if categorical else "number",
            "value_type": "integer" if integer_like else "number",
            "required": True,
        }
        if categorical:
            field["choices"] = [
                _to_builtin_scalar(value)
                for value in sorted(series.unique().tolist())
            ]
        elif len(numeric_series):
            # These are informational training ranges, not hard validation limits.
            field["observed_min"] = float(numeric_series.min())
            field["observed_max"] = float(numeric_series.max())
            field["step"] = 1 if integer_like else "any"
        schema.append(field)
    return schema


def export_late_fusion_bundle(
        final_results: Dict[str, MLResults],
        base_winner_names: Dict[str, str],
        late_winner_names: Dict[str, str],
        output_dir: Path,
        deployment_dir: Path,
) -> Path:
    """Persist deployable inference routes that always include Clinical.

    Clinical-only uses its strongest base model. Each Clinical-containing pair
    uses whichever selected early- or late-fusion winner has the higher
    development CV AUC. All three inputs use the selected full Late_Fusion
    model. Muscle-only, Radiomics-only, and Muscle+Radiomics are intentionally
    unavailable for deployment.

    ``deployment_dir`` is where the bundle and route artifacts land. The main
    training workflow writes directly to the project's artifacts directory.
    """
    modalities = ["Clinical", "Muscle", "Radiomics"]
    missing_base_winners = [
        modality for modality in modalities if modality not in base_winner_names
    ]
    if missing_base_winners:
        raise RuntimeError(
            f"Cannot export deployment bundle; missing base winners: "
            f"{missing_base_winners}"
        )
    if "Late_Fusion" not in late_winner_names:
        raise RuntimeError(
            "Cannot export deployment bundle; full Late_Fusion winner is missing."
        )

    base_models = {}
    for modality in modalities:
        model_name = base_winner_names[modality]
        result_key = f"{modality}|{model_name}"
        if result_key not in final_results:
            raise RuntimeError(
                f"Cannot export deployment bundle; result {result_key!r} is missing."
            )
        result = final_results[result_key]
        base_models[modality] = {
            "model_name": model_name,
            "pipeline": result.fitted_model,
            "features": list(result.selected_features),
            "decision_threshold": float(result.decision_threshold),
        }

    full_late_model_name = late_winner_names["Late_Fusion"]
    late_result_key = f"Late_Fusion|{full_late_model_name}"
    if late_result_key not in final_results:
        raise RuntimeError(
            f"Cannot export deployment bundle; result {late_result_key!r} is missing."
        )
    late_result = final_results[late_result_key]
    expected_meta_features = [
        "Clinical_Probability",
        "Muscle_Probability",
        "Radiomics_Probability",
    ]
    if list(late_result.selected_features) != expected_meta_features:
        raise RuntimeError(
            "Full Late_Fusion deployment features are not the expected ordered "
            f"three probabilities: {late_result.selected_features}"
        )

    def route_payload(
            dataset_name: str,
            model_name: str,
            result: MLResults,
            route_modalities: Sequence[str],
            strategy: str,
    ) -> Dict[str, Any]:
        return {
            "dataset": dataset_name,
            "modalities": list(route_modalities),
            "strategy": strategy,
            "model_name": model_name,
            "pipeline": result.fitted_model,
            "features": list(result.selected_features),
            "decision_threshold": float(result.decision_threshold),
            "selection_cv_auc": float(
                result.selected_cv_evaluation.metrics_cv_mean["AUC"]
            ),
        }

    clinical_model_name = base_winner_names["Clinical"]
    clinical_result = final_results[f"Clinical|{clinical_model_name}"]
    routing_models = {
        "Clinical": route_payload(
            "Clinical",
            clinical_model_name,
            clinical_result,
            ["Clinical"],
            "direct",
        ),
    }

    pair_datasets = {
        ("Clinical", "Muscle"): (
            "Clinical_Muscle", "Late_Fusion_Clinical_Muscle"
        ),
        ("Clinical", "Radiomics"): (
            "Clinical_Radiomics", "Late_Fusion_Clinical_Radiomics"
        ),
    }
    for pair_modalities, (early_dataset, late_dataset) in pair_datasets.items():
        candidates = []
        early_keys = [
            key for key in final_results if key.startswith(f"{early_dataset}|")
        ]
        if len(early_keys) == 1:
            early_model_name = early_keys[0].split("|", 1)[1]
            early_result = final_results[early_keys[0]]
            candidates.append((
                early_result.selected_cv_evaluation.metrics_cv_mean["AUC"],
                early_dataset,
                early_model_name,
                early_result,
                "direct",
            ))
        late_model_name = late_winner_names.get(late_dataset)
        late_key = (
            f"{late_dataset}|{late_model_name}" if late_model_name else None
        )
        if late_key in final_results:
            pair_late_result = final_results[late_key]
            candidates.append((
                pair_late_result.selected_cv_evaluation.metrics_cv_mean["AUC"],
                late_dataset,
                late_model_name,
                pair_late_result,
                "late_fusion",
            ))
        if not candidates:
            raise RuntimeError(
                f"Cannot export deployment route for pair {pair_modalities}."
            )
        _, dataset_name, model_name, result, strategy = max(
            candidates, key=lambda candidate: candidate[0]
        )
        route_key = "+".join(pair_modalities)
        routing_models[route_key] = route_payload(
            dataset_name, model_name, result, pair_modalities, strategy
        )

    routing_models["Clinical+Muscle+Radiomics"] = route_payload(
        "Late_Fusion",
        full_late_model_name,
        late_result,
        modalities,
        "late_fusion",
    )

    bundle = {
        "bundle_type": "fusion_subsidence_full_late_fusion",
        "bundle_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": 1,
        "base_models": base_models,
        "late_fusion": {
            "model_name": full_late_model_name,
            "pipeline": late_result.fitted_model,
            "features": list(late_result.selected_features),
            "decision_threshold": float(late_result.decision_threshold),
        },
        "routing_models": routing_models,
        "clinical_form_schema": build_clinical_form_schema(
            final_results[f"Clinical|{base_winner_names['Clinical']}"]
        ),
        "metadata": {
            "experiment_output": str(output_dir),
            "training_script": "train/train.py",
            "inference_note": (
                "Clinical is required for every prediction. An optional Muscle "
                "and/or Radiomics input selects the strongest matching "
                "development-validated route; late-fusion routes consume "
                "ordered base probabilities."
            ),
        },
    }

    deployment_dir = Path(deployment_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = deployment_dir / "late_fusion_bundle.pkl"
    temporary_path = deployment_dir / "late_fusion_bundle.pkl.tmp"
    joblib.dump(bundle, temporary_path, compress=3)
    temporary_path.replace(bundle_path)

    route_file_names = {
        "Clinical": "clinical.pkl",
        "Clinical+Muscle": "clinical_muscle.pkl",
        "Clinical+Radiomics": "clinical_radiomics.pkl",
        "Clinical+Muscle+Radiomics": "late_fusion.pkl",
    }
    route_artifact_paths = {}
    for route_key, file_name in route_file_names.items():
        route = routing_models[route_key]
        route_artifact = {
            "bundle_type": "fusion_subsidence_route",
            "bundle_version": 1,
            "created_at_utc": bundle["created_at_utc"],
            "positive_class": 1,
            "route_key": route_key,
            "base_models": {
                modality: base_models[modality]
                for modality in route["modalities"]
            },
            "route_model": route,
            "clinical_form_schema": (
                bundle["clinical_form_schema"]
                if "Clinical" in route["modalities"] else []
            ),
        }
        route_path = deployment_dir / file_name
        route_temporary_path = deployment_dir / f"{file_name}.tmp"
        joblib.dump(route_artifact, route_temporary_path, compress=3)
        route_temporary_path.replace(route_path)
        route_artifact_paths[route_key] = route_path

    # Remove route artifacts that are no longer valid when rerunning into an
    # existing artifacts directory.
    for obsolete_file_name in (
            "muscle.pkl", "radiomics.pkl", "muscle_radiomics.pkl",
    ):
        (deployment_dir / obsolete_file_name).unlink(missing_ok=True)

    manifest = {
        "bundle_type": bundle["bundle_type"],
        "bundle_version": bundle["bundle_version"],
        "created_at_utc": bundle["created_at_utc"],
        "artifact_path": str(bundle_path),
        "positive_class": bundle["positive_class"],
        "base_models": {
            modality: {
                "model_name": payload["model_name"],
                "features": payload["features"],
                "decision_threshold": payload["decision_threshold"],
            }
            for modality, payload in base_models.items()
        },
        "late_fusion": {
            "model_name": full_late_model_name,
            "features": late_result.selected_features,
            "decision_threshold": late_result.decision_threshold,
        },
        "routing_models": {
            route_key: {
                key: value
                for key, value in payload.items()
                if key != "pipeline"
            }
            for route_key, payload in routing_models.items()
        },
        "route_artifacts": {
            route_key: str(path)
            for route_key, path in route_artifact_paths.items()
        },
    }
    with open(
            deployment_dir / "late_fusion_bundle_manifest.json",
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"  Adaptive deployment bundle saved to: {bundle_path}")
    print(
        "  Route artifacts saved: "
        + ", ".join(path.name for path in route_artifact_paths.values())
    )
    return bundle_path


# ============================================================
# 5. MODEL CONFIGURATIONS (EBM REMOVED FROM ML9)
# ============================================================

def get_rf_config():
    pipe = SklearnPipeline([("rf", RandomForestClassifier(n_jobs=1, random_state=42))])

    def opt(trial):
        return {
            "rf__n_estimators": trial.suggest_int("rf__n_estimators", 200, 800),
            "rf__max_depth": trial.suggest_int("rf__max_depth", 2, 10),
            "rf__min_samples_leaf": trial.suggest_int("rf__min_samples_leaf", 2, 20),
            "rf__max_features": trial.suggest_categorical(
                "rf__max_features", ["sqrt", "log2", 0.5, 1.0]
            ),
        }

    return pipe, opt


def get_knn_config():
    pipe = SklearnPipeline([("knn", KNeighborsClassifier())])

    def opt(trial):
        return {
            "knn__n_neighbors": trial.suggest_int("knn__n_neighbors", 3, 15),
            "knn__weights": trial.suggest_categorical("knn__weights", ["uniform", "distance"]),
            "knn__p": trial.suggest_int("knn__p", 1, 2),
        }

    return pipe, opt


def get_svm_config():
    pipe = SklearnPipeline([("svm", SVC(probability=True, random_state=42))])

    def opt(trial):
        kernel = trial.suggest_categorical("svm__kernel", ["linear", "rbf"])
        params = {
            "svm__kernel": kernel,
            "svm__C": trial.suggest_float("svm__C", 0.1, 50, log=True),
        }
        if kernel == "rbf":
            params["svm__gamma"] = trial.suggest_float("svm__gamma", 1e-3, 1, log=True)
        return params

    return pipe, opt


def get_lr_config():
    pipe = SklearnPipeline([
        ("lr", LogisticRegression(solver="liblinear", max_iter=5000, random_state=42))
    ])

    def opt(trial):
        return {
            "lr__C": trial.suggest_float("lr__C", 0.01, 10, log=True),
            "lr__penalty": trial.suggest_categorical("lr__penalty", ["l1", "l2"]),
        }

    return pipe, opt


def get_lgbm_config():
    if not LIGHTGBM_AVAILABLE:
        return None
    pipe = SklearnPipeline([
        ("lgbm", lgb.LGBMClassifier(n_jobs=1, random_state=42, verbose=-1, objective="binary"))
    ])

    def opt(trial):
        return {
            "lgbm__n_estimators": trial.suggest_int("lgbm__n_estimators", 100, 500),
            "lgbm__learning_rate": trial.suggest_float("lgbm__learning_rate", 1e-3, 0.3, log=True),
            "lgbm__num_leaves": trial.suggest_int("lgbm__num_leaves", 2, 31),
            "lgbm__max_depth": trial.suggest_int("lgbm__max_depth", 2, 7),
            "lgbm__min_child_samples": trial.suggest_int(
                "lgbm__min_child_samples", 10, 80
            ),
            "lgbm__subsample": trial.suggest_float("lgbm__subsample", 0.6, 1.0),
            "lgbm__colsample_bytree": trial.suggest_float(
                "lgbm__colsample_bytree", 0.5, 1.0
            ),
            "lgbm__reg_alpha": trial.suggest_float("lgbm__reg_alpha", 1e-3, 10.0, log=True),
            "lgbm__reg_lambda": trial.suggest_float("lgbm__reg_lambda", 1e-3, 10.0, log=True),
        }

    return pipe, opt


def get_xgb_config():
    if not XGBOOST_AVAILABLE:
        return None
    pipe = SklearnPipeline([
        ("xgb", xgb.XGBClassifier(
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        ))
    ])

    def opt(trial):
        return {
            "xgb__n_estimators": trial.suggest_int("xgb__n_estimators", 100, 500),
            "xgb__learning_rate": trial.suggest_float("xgb__learning_rate", 1e-3, 0.3, log=True),
            "xgb__max_depth": trial.suggest_int("xgb__max_depth", 1, 5),
            "xgb__min_child_weight": trial.suggest_float(
                "xgb__min_child_weight", 1.0, 20.0, log=True
            ),
            "xgb__subsample": trial.suggest_float("xgb__subsample", 0.6, 1.0),
            "xgb__colsample_bytree": trial.suggest_float(
                "xgb__colsample_bytree", 0.5, 1.0
            ),
            "xgb__gamma": trial.suggest_float("xgb__gamma", 1e-4, 5.0, log=True),
            "xgb__reg_alpha": trial.suggest_float(
                "xgb__reg_alpha", 1e-4, 10.0, log=True
            ),
            "xgb__reg_lambda": trial.suggest_float(
                "xgb__reg_lambda", 1e-3, 20.0, log=True
            ),
        }

    return pipe, opt


def get_models_map():
    models = {
        "RF": get_rf_config(),
        "KNN": get_knn_config(),
        "SVM": get_svm_config(),
        "LR": get_lr_config(),
    }
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = get_lgbm_config()
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = get_xgb_config()
    return {name: config for name, config in models.items() if config is not None}


def get_meta_models_map():
    """Use the same standard model zoo for base learning and late fusion."""
    return get_models_map()


# ============================================================
# 6. EXTERNAL VALIDATION AND LABEL AUDIT
# ============================================================

def _normalize_external_case_id(case_id):
    """Normalize external IDs to the shared 8-digit MRN."""
    normalized = str(case_id).strip()
    normalized = re.sub(r"_20\d{6}$", "", normalized)
    prefixed_match = re.fullmatch(r"\d+_(\d{8})", normalized)
    if prefixed_match:
        normalized = prefixed_match.group(1)
    return normalized


def _read_external_source(path: Path, source_name: str):
    if not path.exists():
        raise FileNotFoundError(f"External {source_name} file not found: {path}")
    raw = pd.read_csv(path, index_col=0).dropna(axis=1, how="all")
    original_names = pd.Series(raw.index.astype(str), index=raw.index)
    normalized_index = [_normalize_external_case_id(case_id) for case_id in raw.index]
    raw.index = normalized_index
    original_names.index = normalized_index
    if raw.index.has_duplicates:
        duplicates = raw.index[raw.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate normalized external IDs in {source_name}: {duplicates}")
    return raw, original_names


def load_external_validation_dataset(
        external_dir: Path,
        internal_fusion: DataFusionPipeline,
        combination_keys: List[str],
        subsidence_threshold_mm: float = 3.0,
):
    """
    Load predictors for one modality combination and labels from Clinical.

    When Sub is available, positive class is explicitly verified as:
      Label 1 = subsidence, Sub > 3 mm
      Label 0 = non-subsidence, Sub <= 3 mm
    When only Label is available, use it as supplied and mark it as
    not independently verified in the label audit.
    Clinical is read for the labels even for Radiomics-only or Muscle-only.
    """
    source_files = {
        "Radiomics": external_dir / "radiomics_features_EXT_new.csv",
        "Clinical": external_dir / "clinical_features_EXT_new.csv",
        "Muscle": external_dir / "CSA_muscles_features_EXT_new.csv",
    }

    clinical_raw, clinical_names = _read_external_source(source_files["Clinical"], "Clinical")
    label_from_sub = None
    if "Sub" in clinical_raw.columns:
        sub_values = pd.to_numeric(clinical_raw["Sub"], errors="coerce")
        if sub_values.isna().any():
            bad_ids = sub_values.index[sub_values.isna()].tolist()
            raise ValueError(f"External Sub is missing/non-numeric for: {bad_ids}")
        label_from_sub = (sub_values > subsidence_threshold_mm).astype(int)

    if "Label" in clinical_raw.columns:
        label_series = pd.to_numeric(clinical_raw["Label"], errors="coerce")
        if label_series.isna().any():
            raise ValueError("External Label contains missing/non-numeric values.")
        label_series = label_series.astype(int)
        if label_from_sub is not None and not label_series.equals(label_from_sub):
            mismatches = label_series.index[label_series != label_from_sub].tolist()
            raise ValueError(
                "External Label is inconsistent with positive class Sub > 3 mm for: "
                f"{mismatches[:20]}"
            )
        if label_from_sub is not None:
            label_source = "Label (verified against Sub > 3 mm)"
            label_rule = f"Sub > {subsidence_threshold_mm:g} mm"
        else:
            label_source = "Provided Label (Sub unavailable; not independently verified)"
            label_rule = "Provided binary Label"
    elif label_from_sub is not None:
        label_series = label_from_sub
        label_source = "Derived from Sub > 3 mm"
        label_rule = f"Sub > {subsidence_threshold_mm:g} mm"
    else:
        raise ValueError("External Clinical data must contain Label or Sub for validation.")

    unexpected_labels = set(label_series.unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"External labels must be binary 0/1; found {unexpected_labels}")

    label_audit = pd.DataFrame({
        "Case_ID": clinical_names,
        "Normalized_Case_ID": clinical_raw.index,
        "Sub_mm": sub_values if label_from_sub is not None else np.nan,
        "Derived_Label": label_series,
        "Positive_Class": "1 = subsidence",
        "Negative_Class": "0 = non-subsidence",
        "Label_Rule": label_rule,
        "Label_Source": label_source,
    }, index=clinical_raw.index)

    prepared = {}
    print(f"\n[ExternalValidation] Sources: {combination_keys}")
    for source_name in combination_keys:
        if source_name == "Clinical":
            raw = clinical_raw
        else:
            raw, _ = _read_external_source(source_files[source_name], source_name)
            missing_ids = label_series.index.difference(raw.index, sort=False).tolist()
            extra_ids = raw.index.difference(label_series.index, sort=False).tolist()
            if missing_ids or extra_ids:
                raise ValueError(
                    f"External {source_name} IDs do not match Clinical: "
                    f"missing={missing_ids[:20]}, extra={extra_ids[:20]}"
                )
            if "Label" in raw.columns:
                source_labels = pd.to_numeric(raw["Label"], errors="coerce")
                if source_labels.isna().any():
                    bad_ids = source_labels.index[source_labels.isna()].tolist()
                    raise ValueError(
                        f"External {source_name} Label is missing/non-numeric for: "
                        f"{bad_ids[:20]}"
                    )
                source_labels = source_labels.astype(int)
                label_mismatches = source_labels.index[
                    source_labels != label_series.loc[source_labels.index]
                ].tolist()
                if label_mismatches:
                    raise ValueError(
                        f"External {source_name} Label disagrees with Clinical for: "
                        f"{label_mismatches[:20]}"
                    )

        expected_columns = [
            column for column in internal_fusion.dfs[source_name].columns if column != "Label"
        ]
        missing_columns = [column for column in expected_columns if column not in raw.columns]
        if missing_columns:
            raise ValueError(
                f"External {source_name} is missing {len(missing_columns)} required predictors: "
                f"{missing_columns[:20]}"
            )
        source_predictors = raw[expected_columns].copy()
        source_predictors.columns = [f"{source_name}___{column}" for column in expected_columns]
        prepared[source_name] = source_predictors
        print(f"   prepared {source_name}: {source_predictors.shape}")

    common_index = prepared[combination_keys[0]].index
    for source_name in combination_keys[1:]:
        common_index = common_index.intersection(prepared[source_name].index, sort=False)
    common_index = common_index.intersection(label_series.index, sort=False)
    if len(common_index) == 0:
        raise ValueError("No aligned external cases after ID normalization.")

    X_external = pd.concat(
        [prepared[source_name].loc[common_index] for source_name in combination_keys], axis=1
    )
    y_external = label_series.loc[common_index].astype(int).to_numpy()
    external_names = clinical_names.loc[common_index].to_numpy()
    aligned_audit = label_audit.loc[common_index].reset_index(drop=True)

    counts = pd.Series(y_external).value_counts().sort_index().to_dict()
    print(f"   aligned external samples: {len(common_index)} | Label distribution: {counts}")
    return X_external, y_external, external_names, aligned_audit


# ============================================================
# 7. POSITIVE-CLASS SHAP FOR THE FINAL WINNERS
# ============================================================

SHAP_GLOBAL_TOP_FEATURES = 10
# Matched to the global figures so a reader moves between them without
# re-learning how many features each one shows. A model with fewer features
# than this displays all of them; the direct late-fusion meta-model has three.
SHAP_LOCAL_TOP_FEATURES = 10
SHAP_PAIR_COUNT = 5
SHAP_HIGH_PROBABILITY = 0.85
SHAP_LOW_PROBABILITY = 0.15
SHAP_FIGURE_DPI = 600
# SHAP's own default for force_plot, which is the canvas these figures have
# always used. A narrower canvas prints the hard-coded 12 pt labels larger once
# the figure is scaled into a journal column, but changes the layout, so the
# default is kept deliberately.
SHAP_FORCE_FIGURE_SIZE = (20, 3)


class ShapInputMismatchError(ValueError):
    """Raised when SHAP cases diverge from audited evaluation predictions."""


def validate_shap_output_probability(
        explanation: Any,
        probability: Sequence[float],
        model_key: str,
        tolerance: float = 1e-5,
):
    """Require base value plus SHAP contributions to reproduce probability."""
    shap_output = (
        np.asarray(explanation.base_values, dtype=float).reshape(-1)
        + np.asarray(explanation.values, dtype=float).sum(axis=1)
    )
    probability = np.asarray(probability, dtype=float)
    if not np.allclose(
            shap_output, probability, rtol=tolerance, atol=tolerance,
    ):
        max_difference = float(np.max(np.abs(shap_output - probability)))
        raise ShapInputMismatchError(
            f"{model_key} SHAP output is not on the audited probability scale; "
            f"max difference={max_difference:.3e}."
        )


def shap_positive_class_explanation(
        explainer: Any,
        shap_values: Any,
        X_plot: pd.DataFrame,
):
    """Normalize old/new SHAP classifier outputs to class 1 only."""
    n_samples, n_features = X_plot.shape
    if isinstance(shap_values, list):
        if len(shap_values) < 2:
            raise ValueError("Binary classifier SHAP output has no positive class.")
        values = np.asarray(shap_values[1], dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            if values.shape[:2] == (n_samples, n_features):
                if values.shape[2] < 2:
                    raise ValueError("Binary classifier SHAP output has no class 1 axis.")
                values = values[:, :, 1]
            elif values.shape[1:] == (n_samples, n_features):
                if values.shape[0] < 2:
                    raise ValueError("Binary classifier SHAP output has no class 1 axis.")
                values = values[1]
            else:
                raise ValueError(
                    f"Unexpected three-dimensional SHAP shape: {values.shape}."
                )
    if values.ndim == 1 and n_samples == 1:
        values = values.reshape(1, -1)
    if values.shape != (n_samples, n_features):
        raise ValueError(
            f"SHAP shape {values.shape} does not match data shape "
            f"{(n_samples, n_features)}."
        )

    expected_value = np.asarray(explainer.expected_value, dtype=float)
    if expected_value.ndim == 0 or expected_value.size == 1:
        positive_base_value = float(expected_value.ravel()[0])
    else:
        positive_base_value = float(expected_value.ravel()[1])
    return shap.Explanation(
        values=values,
        base_values=np.repeat(positive_base_value, n_samples),
        data=X_plot.to_numpy(),
        feature_names=X_plot.columns.tolist(),
    )


def select_shap_explanation_rows(
        probability: Sequence[float],
        max_explain_cases: int,
        pair_count: int = SHAP_PAIR_COUNT,
        high_cutoff: float = SHAP_HIGH_PROBABILITY,
        low_cutoff: float = SHAP_LOW_PROBABILITY,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Keep up to five strict high/low-probability pairs in the SHAP sample."""
    probability = np.asarray(probability, dtype=float)
    high_positions = np.flatnonzero(probability > high_cutoff)
    low_positions = np.flatnonzero(probability < low_cutoff)
    high_positions = high_positions[np.argsort(probability[high_positions])[::-1]]
    low_positions = low_positions[np.argsort(probability[low_positions])]
    n_pairs = min(pair_count, len(high_positions), len(low_positions))
    pairs = [
        (pair_number, int(high_positions[pair_number - 1]),
         int(low_positions[pair_number - 1]))
        for pair_number in range(1, n_pairs + 1)
    ]

    required_positions = []
    for _, high_position, low_position in pairs:
        required_positions.extend([high_position, low_position])
    required_positions = list(dict.fromkeys(required_positions))
    remaining_positions = np.setdiff1d(
        np.arange(len(probability), dtype=int),
        np.asarray(required_positions, dtype=int),
        assume_unique=False,
    )
    remaining_count = max(0, min(
        max_explain_cases - len(required_positions), len(remaining_positions)
    ))
    if remaining_count:
        rng = np.random.default_rng(42)
        sampled_positions = rng.choice(
            remaining_positions, size=remaining_count, replace=False
        ).tolist()
    else:
        sampled_positions = []
    explain_positions = np.asarray(
        required_positions + sampled_positions, dtype=int
    )
    return explain_positions, pairs


def shap_local_top_feature_explanation(
        explanation: Any,
        row_index: int,
        max_features: int = SHAP_LOCAL_TOP_FEATURES,
):
    """Select one sign-balanced local feature set for Force and Waterfall.

    When meaningful positive and negative SHAP contributions both exist, retain
    the strongest feature from each direction before filling the remaining slots
    by absolute importance. Contributions are never fabricated merely to add a
    plot color. Omitted contributions are absorbed into the displayed baseline.
    """
    row = explanation[row_index]
    values = np.asarray(row.values, dtype=float)
    n_keep = min(max_features, len(values))
    ranked_positions = np.argsort(np.abs(values))[::-1]
    max_absolute_value = float(np.max(np.abs(values))) if len(values) else 0.0
    sign_tolerance = max(1e-12, max_absolute_value * 1e-6)
    positive_positions = [
        int(position) for position in ranked_positions
        if values[position] > sign_tolerance
    ]
    negative_positions = [
        int(position) for position in ranked_positions
        if values[position] < -sign_tolerance
    ]

    if n_keep >= 2 and positive_positions and negative_positions:
        selected_positions = [positive_positions[0], negative_positions[0]]
        for position in ranked_positions:
            position = int(position)
            if position not in selected_positions:
                selected_positions.append(position)
            if len(selected_positions) == n_keep:
                break
        top_positions = np.asarray(sorted(
            selected_positions,
            key=lambda position: abs(values[position]),
            reverse=True,
        ), dtype=int)
    else:
        top_positions = ranked_positions[:n_keep]
    omitted_contribution = float(values.sum() - values[top_positions].sum())
    adjusted_base_value = float(np.asarray(row.base_values).ravel()[0]) + omitted_contribution
    local_explanation = shap.Explanation(
        values=values[top_positions],
        base_values=adjusted_base_value,
        data=np.asarray(row.data)[top_positions],
        feature_names=np.asarray(row.feature_names, dtype=object)[top_positions].tolist(),
    )
    return local_explanation, top_positions, omitted_contribution


def shap_plot_explanation_with_two_decimals(explanation: Any):
    """Copy one local explanation ready to plot.

    Feature values are formatted to two decimals and the programming names are
    swapped for the publication labels used by the global beeswarm, so Waterfall
    and Force figures read in the same terms as the summary figure. The audit
    CSVs keep the original column names.
    """
    formatted_data = []
    for value in np.asarray(explanation.data, dtype=object):
        try:
            numeric_value = float(value)
            formatted_data.append(
                f"{numeric_value:.2f}" if math.isfinite(numeric_value) else str(value)
            )
        except (TypeError, ValueError):
            formatted_data.append(str(value))
    return shap.Explanation(
        values=np.asarray(explanation.values, dtype=float),
        base_values=float(np.asarray(explanation.base_values).ravel()[0]),
        data=np.asarray(formatted_data, dtype=object),
        feature_names=[
            shap_labels.label_for(name) for name in explanation.feature_names
        ],
    )


def format_force_plot_axis_two_decimals(figure: Any, base_value: float):
    """Use two decimals and hide the tick printed directly below base value."""
    figure.canvas.draw()
    for axis in figure.axes:
        ticks = np.asarray(axis.get_xticks(), dtype=float)
        if not len(ticks):
            continue
        base_tick = float(ticks[np.argmin(np.abs(ticks - base_value))])
        tolerance = max(1e-10, np.ptp(ticks) * 1e-7)
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _position, hidden=base_tick, atol=tolerance: (
                "" if np.isclose(value, hidden, atol=atol) else f"{value:.2f}"
            )
        ))


def format_shap_x_axes_two_decimals(figure: Any):
    """Format numeric SHAP output axes without replacing feature-name labels."""
    if figure.axes:
        figure.axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def format_waterfall_bar_labels_two_decimals(figure: Any):
    """Print every Waterfall bar label with two decimals.

    SHAP formats a bar with "%+0.02f" and then strips trailing zeros, so a
    contribution of exactly 0.10 prints as "+0.1" beside "+0.29" and the figure
    mixes one and two decimals. The bar labels are the only text objects on the
    main axis, so they can be rewritten in place. SHAP's unicode minus is kept.
    """
    if not figure.axes:
        return
    for label in figure.axes[0].texts:
        text = label.get_text().replace("−", "-")
        try:
            value = float(text)
        except ValueError:
            continue
        label.set_text(f"{value:+.2f}".replace("-", "−"))


def format_waterfall_axes_two_decimals(
        figure: Any,
        explanation: Any,
):
    """Preserve SHAP labels and prevent base/output ticks from overlapping."""
    figure.canvas.draw()
    if not figure.axes:
        return
    base_value = float(np.asarray(explanation.base_values).ravel()[0])
    output_value = base_value + float(np.asarray(explanation.values).sum())
    main_axis = figure.axes[0]
    main_ticks = np.asarray(main_axis.get_xticks(), dtype=float)
    hidden_ticks = []
    if len(main_ticks):
        for marker in (base_value, output_value):
            hidden_ticks.append(
                float(main_ticks[np.argmin(np.abs(main_ticks - marker))])
            )
    tolerance = max(1e-10, np.ptp(main_ticks) * 1e-7) if len(main_ticks) else 1e-10
    main_axis.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _position, hidden=tuple(hidden_ticks), atol=tolerance: (
            "" if any(np.isclose(value, item, atol=atol) for item in hidden)
            else f"{value:.2f}"
        )
    ))

    # SHAP adds one auxiliary axis for E[f(X)] and one for f(x). Retain those
    # labels while rounding only their numeric component.
    for auxiliary_axis in figure.axes[1:]:
        tick_positions = auxiliary_axis.get_xticks()
        tick_labels = []
        label_colors = []
        for label in auxiliary_axis.get_xticklabels():
            text = label.get_text()
            match = re.search(r"(-?\d+(?:\.\d+)?)", text)
            if match:
                rounded = f"{float(match.group(1)):.2f}"
                text = text[:match.start()] + rounded + text[match.end():]
            tick_labels.append(text)
            label_colors.append(label.get_color())
        auxiliary_axis.set_xticks(tick_positions, labels=tick_labels)
        for label, color in zip(
                auxiliary_axis.get_xticklabels(), label_colors
        ):
            label.set_color(color)


def save_positive_class_shap_outputs(
        model_key: str,
        explanation: Any,
        X_plot: pd.DataFrame,
        probability: Sequence[float],
        pair_rows: List[tuple[int, int, int]],
        output_dir: Path,
        save_global_plots: bool = True,
        save_local_plots: bool = True,
        high_eligibility_rule: Optional[str] = None,
        low_eligibility_rule: Optional[str] = None,
):
    """Save top-12 global plots and sign-aware top-6 local plots for class 1.

    The summary beeswarm keeps every feature on the figure: the top 12 get
    their own row and the rest are summed into a final "Sum of N other
    features" row, so a reader can see how much signal sits outside the top 12
    rather than having to assume it is negligible.
    """
    safe_name = safe_file_name(model_key)
    global_dir = output_dir / "global"
    local_dir = output_dir / "local"
    if save_local_plots:
        local_dir.mkdir(parents=True, exist_ok=True)
    probability = np.asarray(probability, dtype=float)
    high_eligibility_rule = high_eligibility_rule or (
        f"probability > {SHAP_HIGH_PROBABILITY}"
    )
    low_eligibility_rule = low_eligibility_rule or (
        f"probability < {SHAP_LOW_PROBABILITY}"
    )

    if save_global_plots:
        global_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 7))
        # beeswarm rather than the legacy summary_plot: it keeps the features
        # past the top 12 on the figure instead of dropping them silently,
        # collapsing them into a bottom "Sum of N other features" row. One
        # extra display slot is asked for so that row is additional to the 12.
        # With 12 or fewer features nothing is cut and no extra row appears.
        labelled_names = [
            shap_labels.label_for(name) for name in explanation.feature_names
        ]
        shap.plots.beeswarm(
            shap.Explanation(
                values=np.asarray(explanation.values, dtype=float),
                base_values=np.asarray(explanation.base_values, dtype=float),
                data=np.asarray(explanation.data, dtype=float),
                feature_names=labelled_names,
            ),
            max_display=SHAP_GLOBAL_TOP_FEATURES + 1,
            group_remaining_features=True,
            show=False,
            plot_size=(12, 7),
        )
        summary_figure = plt.gcf()
        format_shap_x_axes_two_decimals(summary_figure)
        # beeswarm writes SHAP's own axis label; the manuscript uses one that
        # names the outcome the value acts on.
        summary_figure.axes[0].set_xlabel(
            shap_labels.SHAP_VALUE_AXIS_LABEL, fontsize=13
        )
        plt.tight_layout()
        summary_figure.savefig(
            global_dir / (
                f"shap_summary_top{SHAP_GLOBAL_TOP_FEATURES}_{safe_name}.tiff"
            ),
            format="tiff", dpi=SHAP_FIGURE_DPI,
            bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close()

        plt.figure(figsize=(13, 8))
        shap.decision_plot(
            float(np.asarray(explanation.base_values).ravel()[0]),
            explanation.values,
            features=X_plot.set_axis(labelled_names, axis="columns"),
            feature_names=labelled_names,
            feature_order="importance",
            feature_display_range=slice(
                -1, -SHAP_GLOBAL_TOP_FEATURES - 1, -1
            ),
            show=False,
            ignore_warnings=True,
        )
        decision_figure = plt.gcf()
        decision_figure.set_size_inches(13, 8)
        format_shap_x_axes_two_decimals(decision_figure)
        plt.tight_layout()
        decision_figure.savefig(
            global_dir / (
                f"shap_decision_top{SHAP_GLOBAL_TOP_FEATURES}_{safe_name}.tiff"
            ),
            format="tiff", dpi=SHAP_FIGURE_DPI,
            bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(decision_figure)

    if not save_local_plots:
        return

    manifest_rows = []
    contribution_rows = []
    for pair_number, high_row, low_row in pair_rows:
        for risk_group, row_index, eligibility_rule in (
                ("high", high_row, high_eligibility_rule),
                ("low", low_row, low_eligibility_rule),
        ):
            case_id = str(X_plot.index[row_index])
            case_safe_name = safe_file_name(case_id)
            local_explanation, top_positions, omitted_contribution = (
                shap_local_top_feature_explanation(explanation, row_index)
            )
            plot_explanation = shap_plot_explanation_with_two_decimals(
                local_explanation
            )
            shap_output = float(
                np.asarray(explanation[row_index].base_values).ravel()[0]
                + np.asarray(explanation[row_index].values).sum()
            )
            manifest_rows.append({
                "Model": model_key,
                "Pair": pair_number,
                "Risk_Group": risk_group,
                "Case_ID": case_id,
                "Probability_Positive_Class": probability[row_index],
                "Eligibility_Rule": eligibility_rule,
                "Displayed_Features": len(top_positions),
                "SHAP_Output": shap_output,
                "Omitted_SHAP_Contribution_Absorbed_Into_Baseline": omitted_contribution,
            })
            for rank, feature_position in enumerate(top_positions, start=1):
                feature_shap = float(explanation.values[row_index, feature_position])
                contribution_rows.append({
                    "Model": model_key,
                    "Pair": pair_number,
                    "Risk_Group": risk_group,
                    "Case_ID": case_id,
                    "Probability_Positive_Class": probability[row_index],
                    "Rank_By_Absolute_SHAP": rank,
                    "Feature": X_plot.columns[feature_position],
                    "Feature_Value": X_plot.iloc[row_index, feature_position],
                    "Positive_Class_SHAP": feature_shap,
                    "Direction": (
                        "toward positive/subsidence"
                        if feature_shap > 0
                        else (
                            "toward negative/non-subsidence"
                            if feature_shap < 0 else "neutral"
                        )
                    ),
                })

            shap.plots.waterfall(
                plot_explanation,
                max_display=SHAP_LOCAL_TOP_FEATURES,
                show=False,
            )
            waterfall_figure = plt.gcf()
            waterfall_figure.set_size_inches(12, 7.5)
            for waterfall_axis in waterfall_figure.axes:
                waterfall_axis.margins(x=0.12)
            format_waterfall_axes_two_decimals(
                waterfall_figure, plot_explanation
            )
            format_waterfall_bar_labels_two_decimals(waterfall_figure)
            waterfall_figure.subplots_adjust(
                left=0.36, right=0.94, bottom=0.13, top=0.90
            )
            waterfall_figure.savefig(
                local_dir / (
                    f"shap_waterfall_top{SHAP_LOCAL_TOP_FEATURES}_{safe_name}_"
                    f"pair{pair_number:02d}_"
                    f"{risk_group}_{case_safe_name}.tiff"
                ),
                format="tiff", dpi=SHAP_FIGURE_DPI,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"},
            )
            plt.close(waterfall_figure)

            force_base_value = float(
                np.asarray(plot_explanation.base_values).ravel()[0]
            )
            # force_plot builds its own figure and ignores a pre-created one,
            # so the size is passed here rather than through plt.figure().
            shap.force_plot(
                force_base_value,
                plot_explanation.values,
                plot_explanation.data,
                feature_names=plot_explanation.feature_names,
                matplotlib=True,
                show=False,
                text_rotation=45,
                figsize=SHAP_FORCE_FIGURE_SIZE,
            )
            force_figure = plt.gcf()
            format_force_plot_axis_two_decimals(
                force_figure, force_base_value
            )
            force_figure.savefig(
                local_dir / (
                    f"shap_force_top{SHAP_LOCAL_TOP_FEATURES}_{safe_name}_"
                    f"pair{pair_number:02d}_"
                    f"{risk_group}_{case_safe_name}.tiff"
                ),
                format="tiff", dpi=SHAP_FIGURE_DPI,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"},
            )
            plt.close()

    manifest_columns = [
        "Model", "Pair", "Risk_Group", "Case_ID",
        "Probability_Positive_Class", "Eligibility_Rule", "Displayed_Features",
        "SHAP_Output", "Omitted_SHAP_Contribution_Absorbed_Into_Baseline",
    ]
    contribution_columns = [
        "Model", "Pair", "Risk_Group", "Case_ID",
        "Probability_Positive_Class", "Rank_By_Absolute_SHAP", "Feature",
        "Feature_Value", "Positive_Class_SHAP", "Direction",
    ]
    pd.DataFrame(manifest_rows, columns=manifest_columns).to_csv(
        local_dir / f"shap_local_case_manifest_{safe_name}.csv", index=False
    )
    pd.DataFrame(contribution_rows, columns=contribution_columns).to_csv(
        local_dir / (
            f"shap_local_top{SHAP_LOCAL_TOP_FEATURES}_contributions_"
            f"{safe_name}.csv"
        ),
        index=False,
    )


def generate_shap_plots(
        results_dict: Dict[str, MLResults],
        output_dir: Path,
        max_explain_cases: int = 100,
        model_keys: Optional[Sequence[str]] = None,
        explanation_inputs: Optional[Dict[str, pd.DataFrame]] = None,
        expected_probabilities: Optional[Dict[str, pd.Series]] = None,
        background_size: int = 20,
):
    if not SHAP_AVAILABLE:
        print("SHAP not installed: explanation plots skipped.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    all_legend_rows = []

    selected_keys = (
        list(model_keys) if model_keys is not None else list(results_dict)
    )
    for model_key in selected_keys:
        result = results_dict.get(model_key)
        if result is None:
            print(f"  Positive-class SHAP skipped; missing: {model_key}")
            continue
        print(f"  Positive-class SHAP for: {model_key}...")
        try:
            model = result.fitted_model
            estimator_name = next(
                (name for name in ["rf", "xgb", "lgbm", "svm", "lr", "knn"]
                 if name in model.named_steps),
                None,
            )
            if estimator_name is None:
                print(f"    estimator step not found for {model_key}")
                continue

            if explanation_inputs is None or model_key not in explanation_inputs:
                raise ShapInputMismatchError(
                    f"Held-out explanation inputs are required for {model_key}; "
                    "in-sample SHAP is disabled."
                )
            if (
                    expected_probabilities is None
                    or model_key not in expected_probabilities
            ):
                raise ShapInputMismatchError(
                    f"Audited held-out probabilities are required for {model_key}."
                )
            X_candidates = explanation_inputs[model_key].copy()
            model_probability = model.predict_proba(X_candidates)[:, 1]
            expected_probability = expected_probabilities[model_key].copy()
            expected_probability.index = expected_probability.index.astype(str)
            candidate_index = X_candidates.index.astype(str)
            if not expected_probability.index.equals(candidate_index):
                raise ShapInputMismatchError(
                    f"SHAP case order does not match audited predictions for "
                    f"{model_key}."
                )
            candidate_probability = expected_probability.to_numpy(dtype=float)
            if not np.allclose(
                    model_probability, candidate_probability,
                    rtol=1e-10, atol=1e-12,
            ):
                max_difference = float(np.max(np.abs(
                    model_probability - candidate_probability
                )))
                raise ShapInputMismatchError(
                    f"SHAP probabilities do not match audited predictions for "
                    f"{model_key}; max difference={max_difference:.3e}."
                )
            explain_positions, source_pairs = select_shap_explanation_rows(
                candidate_probability, max_explain_cases=max_explain_cases
            )
            if len(source_pairs) < SHAP_PAIR_COUNT:
                print(
                    f"    warning: only {len(source_pairs)}/{SHAP_PAIR_COUNT} "
                    "strict probability pairs are available "
                    f"(>{SHAP_HIGH_PROBABILITY} versus <{SHAP_LOW_PROBABILITY})."
                )
            X_explain = X_candidates.iloc[explain_positions].copy()
            explain_probability = candidate_probability[explain_positions]
            source_to_explain_row = {
                int(source_position): row_index
                for row_index, source_position in enumerate(explain_positions)
            }
            pair_rows = [
                (
                    pair_number,
                    source_to_explain_row[high_position],
                    source_to_explain_row[low_position],
                )
                for pair_number, high_position, low_position in source_pairs
            ]

            preprocessor = model.named_steps["pre"]
            X_encoded = preprocessor.transform(X_explain)
            if not isinstance(X_encoded, pd.DataFrame):
                X_encoded = pd.DataFrame(
                    X_encoded,
                    columns=preprocessor.get_feature_names_out(),
                    index=X_explain.index,
                )
            X_plot = X_explain.loc[:, X_encoded.columns].copy()
            for original_name in X_plot.columns:
                all_legend_rows.append({
                    "Model": model_key,
                    "Display_Name": original_name,
                    "Original_Feature_Name": original_name,
                })

            estimator = model.named_steps[estimator_name]
            if estimator_name in {"rf", "xgb", "lgbm"}:
                background_n = min(background_size, len(result.X_train_final))
                X_background = result.X_train_final.sample(
                    n=background_n, random_state=42
                )
                X_background_encoded = preprocessor.transform(X_background)
                explainer = shap.TreeExplainer(
                    estimator,
                    data=X_background_encoded,
                    feature_perturbation="interventional",
                    model_output="probability",
                )
                shap_values = explainer.shap_values(X_encoded)
            elif estimator_name == "lr" or (
                    estimator_name == "svm"
                    and getattr(estimator, "kernel", "rbf") == "linear"
            ):
                explainer = shap.LinearExplainer(
                    estimator, shap.maskers.Independent(X_encoded)
                )
                shap_values = explainer.shap_values(X_encoded)
            else:
                background = shap.kmeans(X_encoded, min(10, len(X_encoded)))
                explainer = shap.KernelExplainer(
                    estimator.predict_proba, background
                )
                shap_values = explainer.shap_values(
                    X_encoded, nsamples=100, l1_reg=0
                )

            explanation = shap_positive_class_explanation(
                explainer, shap_values, X_plot
            )
            validate_shap_output_probability(
                explanation,
                explain_probability,
                model_key,
            )
            save_positive_class_shap_outputs(
                model_key,
                explanation,
                X_plot,
                explain_probability,
                pair_rows,
                output_dir,
                save_local_plots=False,
            )
        except ShapInputMismatchError:
            raise
        except Exception as exc:
            print(f"    SHAP error for {model_key}: {exc}")

    if all_legend_rows:
        pd.DataFrame(all_legend_rows).to_csv(
            output_dir / "shap_feature_legend.csv", index=False
        )


def generate_late_fusion_end_to_end_shap(
        results_dict: Dict[str, MLResults],
        base_winner_names: Dict[str, str],
        late_winner_name: str,
        modalities: List[str],
        holdout_feature_frames: Dict[str, pd.DataFrame],
        holdout_meta_inputs: pd.DataFrame,
        expected_probability: pd.Series,
        output_dir: Path,
        background_size: int = 20,
        explain_size: int = 100,
        nsamples: int = 200,
):
    """Explain final late-fusion predictions in terms of original base features.

    Both direct and end-to-end SHAP explain the same held-out cases - the
    external cohort - and their audited probabilities. Development data are
    used only as SHAP background. This black-box explanation evaluates the
    complete path: original features -> base probabilities -> meta-model.
    """
    if not SHAP_AVAILABLE:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    late_key = f"Late_Fusion|{late_winner_name}"
    if late_key not in results_dict:
        print(
            "  End-to-end Late Fusion SHAP skipped: final meta-model "
            f"{late_winner_name} is missing."
        )
        return

    missing_winners = [
        modality for modality in modalities if modality not in base_winner_names
    ]
    if missing_winners:
        print(
            "  End-to-end Late Fusion SHAP skipped: missing base winners for "
            f"{missing_winners}."
        )
        return

    base_results = {
        modality: results_dict.get(
            f"{modality}|{base_winner_names[modality]}"
        )
        for modality in modalities
    }
    missing_results = [
        modality for modality, result in base_results.items() if result is None
    ]
    if missing_results:
        print(
            "  End-to-end Late Fusion SHAP skipped: fitted results are missing for "
            f"{missing_results}."
        )
        return

    missing_holdout_frames = [
        modality for modality in modalities
        if modality not in holdout_feature_frames
    ]
    if missing_holdout_frames:
        raise ValueError(
            "End-to-end Late Fusion SHAP is missing held-out feature frames for "
            f"{missing_holdout_frames}."
        )

    # Development rows are never explained. They provide only the Kernel SHAP
    # reference distribution.
    reference_index = base_results[modalities[0]].X_train_final.index
    for modality in modalities[1:]:
        if not base_results[modality].X_train_final.index.equals(reference_index):
            raise ValueError(
                f"End-to-end Late Fusion SHAP development index mismatch for "
                f"{modality}."
            )

    X_background_source = pd.concat([
        base_results[modality].X_train_final.loc[
            :, base_results[modality].selected_features
        ]
        for modality in modalities
    ], axis=1)
    if X_background_source.columns.duplicated().any():
        duplicates = X_background_source.columns[
            X_background_source.columns.duplicated()
        ].tolist()
        raise ValueError(
            "End-to-end Late Fusion SHAP has duplicate development features: "
            f"{duplicates[:20]}"
        )

    holdout_meta_inputs = holdout_meta_inputs.copy()
    holdout_meta_inputs.index = holdout_meta_inputs.index.astype(str)
    if holdout_meta_inputs.index.duplicated().any():
        raise ValueError("Duplicate held-out Case_ID values for end-to-end SHAP.")
    holdout_reference_index = holdout_meta_inputs.index
    aligned_holdout_frames = []
    for modality in modalities:
        base_result = base_results[modality]
        frame = holdout_feature_frames[modality].copy()
        frame.index = frame.index.astype(str)
        if frame.index.duplicated().any():
            raise ValueError(
                f"Duplicate held-out Case_ID values for SHAP {modality}."
            )
        missing_cases = holdout_reference_index.difference(frame.index).tolist()
        extra_cases = frame.index.difference(holdout_reference_index).tolist()
        if missing_cases or extra_cases:
            raise ValueError(
                f"End-to-end SHAP held-out cases do not match for {modality}; "
                f"missing={missing_cases[:10]}, extra={extra_cases[:10]}."
            )
        aligned_holdout_frames.append(
            frame.loc[
                holdout_reference_index, base_result.selected_features
            ]
        )
    X_original = pd.concat(aligned_holdout_frames, axis=1)
    if X_original.columns.duplicated().any():
        duplicates = X_original.columns[
            X_original.columns.duplicated()
        ].tolist()
        raise ValueError(
            "End-to-end Late Fusion SHAP has duplicate held-out features: "
            f"{duplicates[:20]}"
        )

    original_columns = X_background_source.columns.tolist()
    if X_original.columns.tolist() != original_columns:
        raise ValueError(
            "End-to-end SHAP held-out and development feature columns differ."
        )
    late_result = results_dict[late_key]
    late_feature_columns = [
        f"{modality}_Probability" for modality in modalities
    ]

    def build_meta_inputs(X):
        if isinstance(X, pd.DataFrame):
            X_frame = X.copy()
            X_frame.columns = original_columns
        else:
            X_frame = pd.DataFrame(np.asarray(X), columns=original_columns)

        meta_inputs = {}
        for modality in modalities:
            base_result = base_results[modality]
            meta_inputs[f"{modality}_Probability"] = (
                base_result.fitted_model.predict_proba(
                    X_frame.loc[:, base_result.selected_features]
                )[:, 1]
            )
        return pd.DataFrame(meta_inputs, index=X_frame.index)

    def predict_end_to_end(X):
        X_meta = build_meta_inputs(X)
        return late_result.fitted_model.predict_proba(
            X_meta.loc[:, late_feature_columns]
        )[:, 1]

    # The recomputed end-to-end meta inputs and final probabilities must be the
    # exact values already used for the direct LightGBM external-test analysis.
    recomputed_meta_inputs = build_meta_inputs(X_original).loc[
        :, late_feature_columns
    ]
    audited_meta_inputs = holdout_meta_inputs.loc[:, late_feature_columns]
    if not np.allclose(
            recomputed_meta_inputs.to_numpy(dtype=float),
            audited_meta_inputs.to_numpy(dtype=float),
            rtol=1e-10, atol=1e-12,
    ):
        max_difference = float(np.max(np.abs(
            recomputed_meta_inputs.to_numpy(dtype=float)
            - audited_meta_inputs.to_numpy(dtype=float)
        )))
        raise ValueError(
            "End-to-end base probabilities do not match the direct LightGBM "
            f"held-out inputs; max difference={max_difference:.3e}."
        )

    expected_probability = expected_probability.copy()
    expected_probability.index = expected_probability.index.astype(str)
    if not expected_probability.index.equals(holdout_reference_index):
        raise ValueError(
            "End-to-end SHAP case order does not match held-out predictions."
        )
    candidate_probability = predict_end_to_end(X_original)
    audited_probability = expected_probability.to_numpy(dtype=float)
    if not np.allclose(
            candidate_probability, audited_probability,
            rtol=1e-10, atol=1e-12,
    ):
        max_difference = float(np.max(np.abs(
            candidate_probability - audited_probability
        )))
        raise ValueError(
            "End-to-end SHAP probabilities do not match the audited LightGBM "
            f"held-out predictions; max difference={max_difference:.3e}."
        )
    candidate_probability = audited_probability
    explain_positions, source_pairs = select_shap_explanation_rows(
        candidate_probability, max_explain_cases=explain_size
    )
    if len(source_pairs) < SHAP_PAIR_COUNT:
        print(
            f"    warning: only {len(source_pairs)}/{SHAP_PAIR_COUNT} "
            "strict end-to-end probability pairs are available "
            f"(>{SHAP_HIGH_PROBABILITY} versus <{SHAP_LOW_PROBABILITY})."
        )
    X_explain = X_original.iloc[explain_positions].copy()
    explain_probability = np.asarray(candidate_probability)[explain_positions]
    source_to_explain_row = {
        int(source_position): row_index
        for row_index, source_position in enumerate(explain_positions)
    }
    pair_rows = [
        (
            pair_number,
            source_to_explain_row[high_position],
            source_to_explain_row[low_position],
        )
        for pair_number, high_position, low_position in source_pairs
    ]
    background_n = min(background_size, len(X_background_source))
    background = X_background_source.sample(
        n=background_n, random_state=42
    )

    print(
        "  SHAP for: Late_Fusion|End_to_End "
        f"({len(original_columns)} original features, {len(X_explain)} cases)..."
    )
    try:
        explainer = shap.KernelExplainer(predict_end_to_end, background)
        shap_values = explainer.shap_values(
            X_explain,
            nsamples=nsamples,
            # See generate_ml9_external_local_shap.py: the Kernel SHAP default
            # of l1_reg="num_features(10)" zeroes all but ten features per case.
            l1_reg=0,
            silent=True,
        )
        X_plot = X_explain.copy()
        explanation = shap_positive_class_explanation(
            explainer, shap_values, X_plot
        )
        validate_shap_output_probability(
            explanation,
            explain_probability,
            "Late_Fusion|End_to_End",
            tolerance=1e-4,
        )
        pd.DataFrame([{
            "Model": "Late_Fusion|End_to_End",
            "Display_Name": column,
            "Original_Feature_Name": column,
            "Modality": column.split("___", 1)[0],
        } for column in original_columns]).to_csv(
            output_dir / "shap_feature_legend_Late_Fusion_End_to_End.csv",
            index=False,
        )
        save_positive_class_shap_outputs(
            "Late_Fusion|End_to_End",
            explanation,
            X_plot,
            explain_probability,
            pair_rows,
            output_dir,
            save_local_plots=False,
        )
    except ShapInputMismatchError:
        raise
    except Exception as exc:
        print(f"    End-to-end Late Fusion SHAP error: {exc}")


# ============================================================
# 8. PUBLICATION EVALUATION (POST-TRAINING; NO RETRAINING)
# ============================================================

PUBLICATION_MODEL_ORDER = (
    "Clinical",
    "Muscle",
    "Radiomics",
    "Clinical_Radiomics",
    "Clinical_Muscle",
    "Radiomics_Muscle",
    "Full_Fusion",
    "Late_Fusion_Clinical_Radiomics",
    "Late_Fusion_Clinical_Muscle",
    "Late_Fusion_Radiomics_Muscle",
    "Late_Fusion",
)
PUBLICATION_META_MODEL_ORDER = (
    "RF", "KNN", "SVM", "LR", "LightGBM", "XGBoost",
)
PUBLICATION_META_MODEL_COLORS = {
    "RF": "#0072B2",
    "KNN": "#56B4E9",
    "SVM": "#009E73",
    "LR": "#E69F00",
    "LightGBM": "#CC79A7",
    "XGBoost": "#D55E00",
}
PUBLICATION_META_MODEL_LINE_STYLES = {
    "RF": "-",
    "KNN": "--",
    "SVM": "-.",
    "LR": ":",
    "LightGBM": "-",
    "XGBoost": "--",
}
PUBLICATION_META_MODEL_MARKERS = {
    "RF": "o",
    "KNN": "s",
    "SVM": "^",
    "LR": "D",
    "LightGBM": "P",
    "XGBoost": "X",
}
PUBLICATION_DISPLAY_NAMES = {
    "Clinical": "Clinical",
    "Muscle": "Muscle",
    "Radiomics": "Radiomics",
    "Clinical_Radiomics": "Clinical + Radiomics (Early)",
    "Clinical_Muscle": "Clinical + Muscle (Early)",
    "Radiomics_Muscle": "Radiomics + Muscle (Early)",
    "Full_Fusion": "Full Fusion (Early)",
    "Late_Fusion_Clinical_Radiomics": "Clinical + Radiomics (Late)",
    "Late_Fusion_Clinical_Muscle": "Clinical + Muscle (Late)",
    "Late_Fusion_Radiomics_Muscle": "Radiomics + Muscle (Late)",
    "Late_Fusion": "Full Late Fusion (Stacking)",
}
@dataclass(frozen=True)
class PublicationMetricValues:
    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    ppv: float
    npv: float
    tn: int
    fp: int
    fn: int
    tp: int


def publication_output_directories(output_dir: Path) -> Dict[str, Path]:
    """Create publication subfolders inside the existing experiment output."""
    directories = {
        "tables": output_dir / "performance_tables",
        "roc": output_dir / "roc_curves",
        "calibration": output_dir / "supplementary" / "calibration",
        "dca": output_dir / "supplementary" / "decision_curve_analysis",
        "confusion": output_dir / "confusion_matrix",
        "summary": output_dir / "summary",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def publication_validate_predictions(
        prediction_table: pd.DataFrame,
        selected_models: pd.DataFrame,
        split: str,
) -> pd.DataFrame:
    """Validate winner predictions and their leakage-safe threshold provenance."""
    threshold_column = (
        "Decision_Threshold" if split == "cv"
        else "Decision_Threshold_From_OOF"
    )
    required = {
        "Dataset", "Model", "Case_ID", "True_Label",
        "Probability_Subsidence", threshold_column,
    }
    if split == "cv":
        required.add("Fold")
    missing = required - set(prediction_table.columns)
    if missing:
        raise ValueError(
            f"Publication {split} predictions are missing: {sorted(missing)}"
        )

    table = prediction_table.loc[
        prediction_table["Dataset"].isin(PUBLICATION_MODEL_ORDER)
    ].copy()
    table["Dataset"] = table["Dataset"].astype(str)
    table["Model"] = table["Model"].astype(str)
    table["Case_ID"] = table["Case_ID"].astype(str)
    table["True_Label"] = pd.to_numeric(
        table["True_Label"], errors="raise"
    ).astype(int)
    table["Probability_Subsidence"] = pd.to_numeric(
        table["Probability_Subsidence"], errors="raise"
    ).astype(float)
    table[threshold_column] = pd.to_numeric(
        table[threshold_column], errors="raise"
    ).astype(float)

    missing_models = set(PUBLICATION_MODEL_ORDER) - set(table["Dataset"])
    if missing_models:
        raise ValueError(
            f"Publication {split} predictions are missing winners: "
            f"{sorted(missing_models)}"
        )
    if table[["Dataset", "Case_ID"]].duplicated().any():
        raise ValueError(f"Publication {split} predictions contain duplicate cases.")
    if not set(table["True_Label"].unique()).issubset({0, 1}):
        raise ValueError(f"Publication {split} labels must be binary 0/1.")
    probabilities = table["Probability_Subsidence"]
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError(
            f"Publication {split} probabilities must be finite and within [0, 1]."
        )
    thresholds = table[threshold_column]
    if not np.isfinite(thresholds).all() or not thresholds.between(0, 1, inclusive="neither").all():
        raise ValueError(f"Publication {split} decision thresholds are invalid.")

    winner_lookup = selected_models.set_index("Dataset")
    reference_cases = None
    for dataset in PUBLICATION_MODEL_ORDER:
        group = table.loc[table["Dataset"] == dataset]
        if group["True_Label"].nunique() != 2:
            raise ValueError(f"{split}/{dataset} does not contain both outcome classes.")
        if group["Model"].nunique() != 1:
            raise ValueError(f"{split}/{dataset} contains multiple winner algorithms.")
        expected_model = str(winner_lookup.loc[dataset, "Best_Model"])
        if str(group["Model"].iloc[0]) != expected_model:
            raise ValueError(
                f"{split}/{dataset} model does not match selected_models.csv."
            )
        if group[threshold_column].nunique() != 1:
            raise ValueError(f"{split}/{dataset} contains multiple thresholds.")
        observed_threshold = float(group[threshold_column].iloc[0])
        expected_threshold = float(
            winner_lookup.loc[dataset, "OOF_Decision_Threshold"]
        )
        if not math.isclose(
                observed_threshold, expected_threshold,
                rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(
                f"{split}/{dataset} does not use its development OOF threshold."
            )
        aligned = group[["Case_ID", "True_Label"]].sort_values(
            "Case_ID"
        ).reset_index(drop=True)
        if reference_cases is None:
            reference_cases = aligned
        elif not aligned.equals(reference_cases):
            raise ValueError(
                f"Publication {split} cases/labels are not aligned across models."
            )
        if split == "cv" and group["Fold"].nunique() != 5:
            raise ValueError(f"CV/{dataset} must contain exactly five folds.")
    return table


def publication_safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def publication_metrics(
        y_true: Sequence[int],
        probability: Sequence[float],
        threshold: float,
) -> PublicationMetricValues:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()
    return PublicationMetricValues(
        auc=float(roc_auc_score(y_true, probability)),
        accuracy=float(accuracy_score(y_true, prediction)),
        sensitivity=publication_safe_ratio(tp, tp + fn),
        specificity=publication_safe_ratio(tn, tn + fp),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        ppv=publication_safe_ratio(tp, tp + fp),
        npv=publication_safe_ratio(tn, tn + fn),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def publication_bootstrap_auc_ci(
        y_true: Sequence[int],
        probability: Sequence[float],
        iterations: int = 2000,
        random_seed: int = 20260809,
) -> tuple[float, float, float]:
    """Stratified nonparametric percentile-bootstrap 95% CI for ROC AUC."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if not len(positive) or not len(negative):
        raise ValueError("AUC bootstrap requires both outcome classes.")
    rng = np.random.default_rng(random_seed)
    auc_values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        indices = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        auc_values[iteration] = roc_auc_score(
            y_true[indices], probability[indices]
        )
    lower, upper = np.percentile(auc_values, [2.5, 97.5])
    return (
        float(roc_auc_score(y_true, probability)),
        float(lower),
        float(upper),
    )


def publication_model_name(dataset: str, selected_models: pd.DataFrame) -> str:
    model = selected_models.set_index("Dataset").loc[dataset, "Best_Model"]
    return f"{PUBLICATION_DISPLAY_NAMES[dataset]} ({model})"


def publication_mean_sd(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=float)
    return f"{np.nanmean(values):.3f} ± {np.nanstd(values, ddof=1):.3f}"


def publication_cv_table(
        cv_predictions: pd.DataFrame,
        selected_models: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in PUBLICATION_MODEL_ORDER:
        group = cv_predictions.loc[cv_predictions["Dataset"] == dataset]
        threshold = float(group["Decision_Threshold"].iloc[0])
        fold_metrics = [
            publication_metrics(
                fold["True_Label"], fold["Probability_Subsidence"], threshold
            )
            for _, fold in group.groupby("Fold", sort=True)
        ]
        rows.append({
            "Model": publication_model_name(dataset, selected_models),
            "AUC (mean ± SD)": publication_mean_sd([x.auc for x in fold_metrics]),
            "Accuracy (mean ± SD)": publication_mean_sd([x.accuracy for x in fold_metrics]),
            "Sensitivity (mean ± SD)": publication_mean_sd([x.sensitivity for x in fold_metrics]),
            "Specificity (mean ± SD)": publication_mean_sd([x.specificity for x in fold_metrics]),
            "F1 score (mean ± SD)": publication_mean_sd([x.f1 for x in fold_metrics]),
            "PPV (mean ± SD)": publication_mean_sd([x.ppv for x in fold_metrics]),
            "NPV (mean ± SD)": publication_mean_sd([x.npv for x in fold_metrics]),
        })
    return pd.DataFrame(rows)


def publication_test_table(
        predictions: pd.DataFrame,
        selected_models: pd.DataFrame,
        bootstrap_iterations: int,
        random_seed: int,
) -> pd.DataFrame:
    rows = []
    for dataset in PUBLICATION_MODEL_ORDER:
        group = predictions.loc[predictions["Dataset"] == dataset]
        threshold = float(group["Decision_Threshold_From_OOF"].iloc[0])
        metrics = publication_metrics(
            group["True_Label"], group["Probability_Subsidence"], threshold
        )
        auc, lower, upper = publication_bootstrap_auc_ci(
            group["True_Label"], group["Probability_Subsidence"],
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        rows.append({
            "Model": publication_model_name(dataset, selected_models),
            "AUC (95% CI)": f"{auc:.3f} ({lower:.3f}-{upper:.3f})",
            "Accuracy": f"{metrics.accuracy:.3f}",
            "Sensitivity": f"{metrics.sensitivity:.3f}",
            "Specificity": f"{metrics.specificity:.3f}",
            "F1 score": f"{metrics.f1:.3f}",
            "PPV": f"{metrics.ppv:.3f}",
            "NPV": f"{metrics.npv:.3f}",
        })
    return pd.DataFrame(rows)


def publication_validate_late_model_predictions(
        predictions: pd.DataFrame,
        split: str,
) -> pd.DataFrame:
    """Validate full Late_Fusion predictions compared across algorithms."""
    required = {
        "Dataset", "Model", "Case_ID", "True_Label", "Probability_Subsidence"
    }
    if split == "cv":
        required.add("Fold")
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Late_Fusion {split} model comparison is missing: {sorted(missing)}"
        )
    table = predictions.loc[predictions["Dataset"] == "Late_Fusion"].copy()
    table["Model"] = table["Model"].astype(str)
    table["Case_ID"] = table["Case_ID"].astype(str)
    table["True_Label"] = pd.to_numeric(
        table["True_Label"], errors="raise"
    ).astype(int)
    table["Probability_Subsidence"] = pd.to_numeric(
        table["Probability_Subsidence"], errors="raise"
    ).astype(float)
    if table.empty:
        raise ValueError(f"Late_Fusion {split} model comparison is empty.")
    unknown_models = set(table["Model"]) - set(PUBLICATION_META_MODEL_ORDER)
    if unknown_models:
        raise ValueError(
            f"Unknown Late_Fusion meta-models: {sorted(unknown_models)}"
        )
    if table[["Model", "Case_ID"]].duplicated().any():
        raise ValueError(
            f"Late_Fusion {split} comparison contains duplicate model/case rows."
        )
    probability = table["Probability_Subsidence"]
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError(
            f"Late_Fusion {split} probabilities must be finite within [0, 1]."
        )

    reference_cases = None
    for model_name in publication_available_meta_models(table):
        group = table.loc[table["Model"] == model_name]
        if group["True_Label"].nunique() != 2:
            raise ValueError(
                f"Late_Fusion {split}/{model_name} lacks both outcome classes."
            )
        aligned = group[["Case_ID", "True_Label"]].sort_values(
            "Case_ID"
        ).reset_index(drop=True)
        if reference_cases is None:
            reference_cases = aligned
        elif not aligned.equals(reference_cases):
            raise ValueError(
                f"Late_Fusion {split} cases are not aligned across meta-models."
            )
        if split == "cv" and group["Fold"].nunique() != 5:
            raise ValueError(
                f"Late_Fusion CV/{model_name} must contain exactly five folds."
            )
    return table


def publication_available_meta_models(
        predictions: pd.DataFrame,
) -> List[str]:
    available = set(predictions["Model"].astype(str).unique())
    return [
        model_name for model_name in PUBLICATION_META_MODEL_ORDER
        if model_name in available
    ]


def publication_late_model_auc_table(
        cv_predictions: pd.DataFrame,
        external_predictions: pd.DataFrame,
        bootstrap_iterations: int,
        random_seed: int,
) -> pd.DataFrame:
    """AUC summary for algorithms fitted to the same full Late_Fusion inputs."""
    rows = []
    for model_name in publication_available_meta_models(cv_predictions):
        cv_group = cv_predictions.loc[cv_predictions["Model"] == model_name]
        fold_auc = [
            roc_auc_score(fold["True_Label"], fold["Probability_Subsidence"])
            for _, fold in cv_group.groupby("Fold", sort=True)
        ]
        external_group = external_predictions.loc[
            external_predictions["Model"] == model_name
        ]
        external_auc, external_lower, external_upper = (
            publication_bootstrap_auc_ci(
                external_group["True_Label"],
                external_group["Probability_Subsidence"],
                iterations=bootstrap_iterations,
                random_seed=random_seed,
            )
        )
        rows.append({
            "Meta_Model": model_name,
            "Five_Fold_CV_AUC_Mean": float(np.mean(fold_auc)),
            "Five_Fold_CV_AUC_SD": float(np.std(fold_auc, ddof=1)),
            "External_AUC": external_auc,
            "External_AUC_95CI_Lower": external_lower,
            "External_AUC_95CI_Upper": external_upper,
        })
    return pd.DataFrame(rows)


def publication_figure_style() -> Dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def publication_save_figure(fig, tiff_path: Path, dpi: int = 600):
    """Save publication figures exclusively as LZW-compressed 600-dpi TIFF."""
    tiff_path = tiff_path.with_suffix(".tiff")
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def publication_plot_cv_roc_axis(
        axis: Any,
        cv_predictions: pd.DataFrame,
):
    """Compare full Late_Fusion meta-models using validation-fold ROC curves."""
    mean_fpr = np.linspace(0, 1, 201)
    for model_name in publication_available_meta_models(cv_predictions):
        group = cv_predictions.loc[cv_predictions["Model"] == model_name]
        fold_tprs, fold_aucs = [], []
        for _, fold in group.groupby("Fold", sort=True):
            fpr, tpr, _ = roc_curve(
                fold["True_Label"], fold["Probability_Subsidence"]
            )
            interpolated = np.interp(mean_fpr, fpr, tpr)
            interpolated[0] = 0.0
            fold_tprs.append(interpolated)
            fold_aucs.append(roc_auc_score(
                fold["True_Label"], fold["Probability_Subsidence"]
            ))
        mean_tpr = np.mean(fold_tprs, axis=0)
        mean_tpr[-1] = 1.0
        color = PUBLICATION_META_MODEL_COLORS[model_name]
        axis.plot(
            mean_fpr, mean_tpr,
            color=color,
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            linewidth=1.7,
            label=(
                f"{model_name} "
                f"(AUC {np.mean(fold_aucs):.3f} ± "
                f"{np.std(fold_aucs, ddof=1):.3f})"
            ),
        )


def publication_plot_test_roc_axis(
        axis: Any,
        predictions: pd.DataFrame,
        bootstrap_iterations: int,
        random_seed: int,
):
    """Compare full Late_Fusion meta-models on one held-out cohort."""
    for model_name in publication_available_meta_models(predictions):
        group = predictions.loc[predictions["Model"] == model_name]
        fpr, tpr, _ = roc_curve(
            group["True_Label"], group["Probability_Subsidence"]
        )
        auc, lower, upper = publication_bootstrap_auc_ci(
            group["True_Label"], group["Probability_Subsidence"],
            iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        axis.plot(
            fpr, tpr,
            color=PUBLICATION_META_MODEL_COLORS[model_name],
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            linewidth=1.7,
            label=(
                f"{model_name} "
                f"(AUC {auc:.3f} [{lower:.3f}–{upper:.3f}])"
            ),
        )


def publication_curve_title(
        curve_name: str,
        cohort_name: str,
        is_cross_validation: bool,
) -> str:
    if is_cross_validation:
        return f"Mean {curve_name} Curves from 5-fold Cross-Validation"
    # The external cohort is the only test set, so its name is the only one
    # a non-CV curve can carry.
    return f"{curve_name} Curves for External Test set"


def publication_plot_late_model_roc(
        predictions: pd.DataFrame,
        cohort_name: str,
        is_cross_validation: bool,
        output_path: Path,
        bootstrap_iterations: int,
        random_seed: int,
        dpi: int,
):
    """Save one cohort-specific ROC comparison across Late_Fusion algorithms."""
    with plt.rc_context(publication_figure_style()):
        fig, axis = plt.subplots(figsize=(8.0, 6.8))
        if is_cross_validation:
            publication_plot_cv_roc_axis(axis, predictions)
        else:
            publication_plot_test_roc_axis(
                axis, predictions,
                bootstrap_iterations, random_seed,
            )
        axis.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.0)
        axis.set(
            xlim=(0, 1), ylim=(0, 1.02),
            xlabel="False positive rate (1 − specificity)",
            ylabel="True positive rate (sensitivity)",
            title=publication_curve_title(
                "ROC", cohort_name, is_cross_validation
            ),
        )
        axis.grid(alpha=0.15, linewidth=0.5)
        axis.legend(
            loc="lower right", fontsize=8, ncol=1,
            frameon=True, framealpha=0.92, facecolor="white",
        )
        fig.tight_layout()
        publication_save_figure(fig, output_path, dpi)


def publication_fold_calibration(
        y_true: Sequence[int],
        probability: Sequence[float],
        n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count calibration bins with a fixed bin count for fold averaging."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    ordered_indices = np.argsort(probability, kind="stable")
    bins = np.array_split(ordered_indices, n_bins)
    mean_probability = np.asarray([
        probability[indices].mean() for indices in bins if len(indices)
    ])
    observed_probability = np.asarray([
        y_true[indices].mean() for indices in bins if len(indices)
    ])
    return mean_probability, observed_probability


def publication_plot_cv_calibration_axis(
        axis: Any,
        cv_predictions: pd.DataFrame,
        n_bins: int,
):
    """Mean five-fold calibration across full Late_Fusion meta-models."""
    for model_name in publication_available_meta_models(cv_predictions):
        cv_group = cv_predictions.loc[cv_predictions["Model"] == model_name]
        folds = [fold for _, fold in cv_group.groupby("Fold", sort=True)]
        effective_bins = min(n_bins, *(len(fold) for fold in folds))
        fold_predicted, fold_observed, fold_brier, fold_auc = [], [], [], []
        for fold in folds:
            predicted, observed = publication_fold_calibration(
                fold["True_Label"],
                fold["Probability_Subsidence"],
                n_bins=effective_bins,
            )
            fold_predicted.append(predicted)
            fold_observed.append(observed)
            fold_brier.append(brier_score_loss(
                fold["True_Label"], fold["Probability_Subsidence"]
            ))
            fold_auc.append(roc_auc_score(
                fold["True_Label"], fold["Probability_Subsidence"]
            ))
        mean_predicted = np.mean(fold_predicted, axis=0)
        mean_observed = np.mean(fold_observed, axis=0)
        color = PUBLICATION_META_MODEL_COLORS[model_name]
        axis.plot(
            mean_predicted, mean_observed,
            marker=PUBLICATION_META_MODEL_MARKERS[model_name],
            markersize=3.5,
            linewidth=1.5,
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            color=color,
            label=(
                f"{model_name} "
                f"(AUC {np.mean(fold_auc):.3f}; "
                f"Brier {np.mean(fold_brier):.3f})"
            ),
        )


def publication_plot_test_calibration_axis(
        axis: Any,
        predictions: pd.DataFrame,
        n_bins: int,
):
    """Held-out calibration compared across full Late_Fusion meta-models."""
    for model_name in publication_available_meta_models(predictions):
        group = predictions.loc[predictions["Model"] == model_name]
        observed, predicted = calibration_curve(
            group["True_Label"],
            group["Probability_Subsidence"],
            n_bins=min(n_bins, len(group)),
            strategy="quantile",
        )
        brier = brier_score_loss(
            group["True_Label"], group["Probability_Subsidence"]
        )
        auc = roc_auc_score(
            group["True_Label"], group["Probability_Subsidence"]
        )
        axis.plot(
            predicted, observed,
            marker=PUBLICATION_META_MODEL_MARKERS[model_name],
            markersize=3.5,
            linewidth=1.5,
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            color=PUBLICATION_META_MODEL_COLORS[model_name],
            label=(
                f"{model_name} "
                f"(AUC {auc:.3f}; Brier {brier:.3f})"
            ),
        )


def publication_plot_late_model_calibration(
        predictions: pd.DataFrame,
        cohort_name: str,
        is_cross_validation: bool,
        output_path: Path,
        n_bins: int,
        dpi: int,
):
    """Save one cohort-specific calibration comparison of Late_Fusion models."""
    with plt.rc_context(publication_figure_style()):
        fig, axis = plt.subplots(figsize=(8.0, 6.8))
        if is_cross_validation:
            publication_plot_cv_calibration_axis(axis, predictions, n_bins)
        else:
            publication_plot_test_calibration_axis(
                axis, predictions, n_bins
            )
        axis.plot(
            [0, 1], [0, 1], "--", color="0.35", linewidth=1.1,
            label="Perfect calibration",
        )
        axis.set(
            xlim=(0, 1), ylim=(0, 1.02),
            xlabel="Mean predicted probability",
            ylabel="Observed event proportion",
            title=publication_curve_title(
                "Calibration", cohort_name, is_cross_validation
            ),
        )
        axis.grid(alpha=0.15, linewidth=0.5)
        axis.legend(
            loc="lower right", fontsize=8, ncol=1,
            frameon=True, framealpha=0.92, facecolor="white",
        )
        fig.tight_layout()
        publication_save_figure(fig, output_path, dpi)


def publication_net_benefit(
        y_true: Sequence[int],
        probability: Sequence[float],
        thresholds: Sequence[float],
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    benefits = np.empty(len(thresholds), dtype=float)
    for index, threshold in enumerate(thresholds):
        prediction = probability >= threshold
        tp = np.sum(prediction & (y_true == 1))
        fp = np.sum(prediction & (y_true == 0))
        benefits[index] = (
            tp / len(y_true)
            - fp / len(y_true) * threshold / (1 - threshold)
        )
    return benefits


def publication_plot_cv_dca_axis(
        axis: Any,
        cv_predictions: pd.DataFrame,
        thresholds: np.ndarray,
) -> float:
    """Mean validation-fold net benefit across Late_Fusion meta-models."""
    model_order = publication_available_meta_models(cv_predictions)
    reference = cv_predictions.loc[
        cv_predictions["Model"] == model_order[0]
    ]
    fold_prevalence = [
        float(fold["True_Label"].mean())
        for _, fold in reference.groupby("Fold", sort=True)
    ]
    treat_all = np.mean([
        prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        for prevalence in fold_prevalence
    ], axis=0)
    axis.axhline(0, color="black", linewidth=1.1, label="Treat none")
    axis.plot(
        thresholds, treat_all, "--", color="0.5", linewidth=1.3,
        label="Treat all",
    )
    upper_values = [float(np.max(treat_all))]
    for model_name in model_order:
        group = cv_predictions.loc[cv_predictions["Model"] == model_name]
        fold_benefits = []
        for _, fold in group.groupby("Fold", sort=True):
            fold_benefits.append(publication_net_benefit(
                fold["True_Label"], fold["Probability_Subsidence"], thresholds
            ))
        mean_benefit = np.mean(fold_benefits, axis=0)
        upper_values.append(float(np.max(mean_benefit)))
        axis.plot(
            thresholds, mean_benefit,
            color=PUBLICATION_META_MODEL_COLORS[model_name],
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            linewidth=1.6,
            # A decision curve compares net benefit, not discrimination, so the
            # legend carries the model name only.
            label=model_name,
        )
    return max(upper_values)


def publication_plot_test_dca_axis(
        axis: Any,
        predictions: pd.DataFrame,
        thresholds: np.ndarray,
) -> float:
    """Held-out net benefit compared across Late_Fusion meta-models."""
    model_order = publication_available_meta_models(predictions)
    reference = predictions.loc[
        predictions["Model"] == model_order[0], "True_Label"
    ].to_numpy(int)
    prevalence = float(reference.mean())
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    axis.axhline(0, color="black", linewidth=1.1, label="Treat none")
    axis.plot(
        thresholds, treat_all, "--", color="0.5", linewidth=1.3,
        label="Treat all",
    )
    upper_values = [float(np.max(treat_all))]
    for model_name in model_order:
        group = predictions.loc[predictions["Model"] == model_name]
        benefit = publication_net_benefit(
            group["True_Label"], group["Probability_Subsidence"], thresholds
        )
        upper_values.append(float(np.max(benefit)))
        axis.plot(
            thresholds, benefit,
            color=PUBLICATION_META_MODEL_COLORS[model_name],
            linestyle=PUBLICATION_META_MODEL_LINE_STYLES[model_name],
            linewidth=1.6,
            label=model_name,
        )
    return max(upper_values)


def publication_plot_late_model_dca(
        predictions: pd.DataFrame,
        cohort_name: str,
        is_cross_validation: bool,
        output_path: Path,
        min_threshold: float,
        max_threshold: float,
        dpi: int,
):
    """Save one cohort-specific DCA comparison of Late_Fusion algorithms."""
    thresholds = np.linspace(min_threshold, max_threshold, 160)
    with plt.rc_context(publication_figure_style()):
        fig, axis = plt.subplots(figsize=(8.0, 6.8))
        if is_cross_validation:
            upper = publication_plot_cv_dca_axis(
                axis, predictions, thresholds
            )
        else:
            upper = publication_plot_test_dca_axis(
                axis, predictions, thresholds
            )
        axis.set(
            xlim=(min_threshold, max_threshold),
            ylim=(-0.10, min(1.0, upper + 0.05)),
            xlabel="Threshold probability",
            ylabel="Net benefit",
            title=publication_curve_title(
                "DCA", cohort_name, is_cross_validation
            ),
        )
        axis.grid(alpha=0.15, linewidth=0.5)
        # Decision curves fall from left to right, so the upper right is the
        # empty corner here - the opposite of the ROC and calibration axes,
        # whose curves run along the top and keep their legend lower right.
        axis.legend(
            loc="upper right", fontsize=8, ncol=1,
            frameon=True, framealpha=0.92, facecolor="white",
        )
        fig.tight_layout()
        publication_save_figure(fig, output_path, dpi)


def publication_plot_confusion_matrices(
        predictions: pd.DataFrame,
        selected_models: pd.DataFrame,
        cohort_name: str,
        output_path: Path,
        dpi: int,
):
    n_columns = 3
    n_rows = math.ceil(len(PUBLICATION_MODEL_ORDER) / n_columns)
    with plt.rc_context(publication_figure_style()):
        fig, axes = plt.subplots(
            n_rows, n_columns,
            figsize=(10.2, 3.2 * n_rows),
            squeeze=False,
        )
        for axis, dataset in zip(axes.flat, PUBLICATION_MODEL_ORDER):
            group = predictions.loc[predictions["Dataset"] == dataset]
            threshold = float(group["Decision_Threshold_From_OOF"].iloc[0])
            metric = publication_metrics(
                group["True_Label"], group["Probability_Subsidence"], threshold
            )
            matrix = np.array([[metric.tn, metric.fp], [metric.fn, metric.tp]])
            axis.imshow(
                matrix, cmap="Blues",
                norm=Normalize(vmin=0, vmax=max(1, int(matrix.max()))),
            )
            names = (("TN", "FP"), ("FN", "TP"))
            midpoint = matrix.max() / 2
            for row in range(2):
                for column in range(2):
                    axis.text(
                        column, row,
                        f"{names[row][column]}\n{matrix[row, column]}",
                        ha="center", va="center", fontsize=11,
                        color=(
                            "white" if matrix[row, column] > midpoint
                            else "black"
                        ),
                    )
            axis.set_xticks([0, 1], labels=["Negative", "Positive"])
            axis.set_yticks([0, 1], labels=["Negative", "Positive"])
            axis.set_xlabel("Predicted outcome")
            axis.set_ylabel("Observed outcome")
            axis.set_title(
                f"{publication_model_name(dataset, selected_models)}\n"
                f"Threshold = {threshold:.2f}",
                fontsize=9.5,
            )
        for axis in list(axes.flat)[len(PUBLICATION_MODEL_ORDER):]:
            axis.remove()
        fig.suptitle(f"{cohort_name} confusion matrices", fontsize=12, y=1.01)
        fig.tight_layout()
        publication_save_figure(fig, output_path, dpi)


def publication_write_table(table: pd.DataFrame, path: Path):
    table.to_csv(path, index=False)
    table.to_excel(path.with_suffix(".xlsx"), index=False, engine="openpyxl")


def publication_cohort_description(
        predictions: pd.DataFrame,
) -> str:
    group = predictions.loc[
        predictions["Dataset"] == PUBLICATION_MODEL_ORDER[0]
    ]
    positives = int(group["True_Label"].sum())
    return (
        f"n={len(group)}, events={positives}, "
        f"non-events={len(group) - positives}, "
        f"prevalence={positives / len(group):.3f}"
    )


def publication_write_summary(
        output_dir: Path,
        selected_models: pd.DataFrame,
        cv_predictions: pd.DataFrame,
        external_predictions: pd.DataFrame,
        generated_files: Sequence[Path],
        bootstrap_iterations: int,
        random_seed: int,
):
    winner_lookup = selected_models.set_index("Dataset")
    lines = [
        "PUBLICATION EVALUATION SUMMARY",
        "=" * 72,
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Evaluation output directory: {output_dir.resolve()}",
        "Training/retraining performed by evaluation: NO",
        "Test-set threshold optimization performed: NO",
        "",
        "COHORTS",
        f"- Five-fold CV development cohort: {publication_cohort_description(cv_predictions)}",
        f"- External test cohort: {publication_cohort_description(external_predictions)}",
        "- No internal hold-out: every aligned internal case is development data.",
        "",
        "EVALUATED WINNERS",
    ]
    for dataset in PUBLICATION_MODEL_ORDER:
        row = winner_lookup.loc[dataset]
        lines.append(
            f"- {PUBLICATION_DISPLAY_NAMES[dataset]}: {row['Best_Model']}; "
            f"development OOF threshold="
            f"{float(row['OOF_Decision_Threshold']):.6f}"
        )
    lines.extend([
        "",
        "STATISTICAL METHODS",
        "- CV metrics: arithmetic mean ± sample SD across five validation folds.",
        "- ROC, calibration, and DCA figures compare algorithms within the full "
        "three-modality Late_Fusion model; modality branches are not curves.",
        "- Five-fold CV and external test ROC/calibration/DCA are exported as "
        "separate figures.",
        "- All publication figures are LZW-compressed TIFF at 600 dpi.",
        "- CV ROC: mean validation ROC; the former shaded ±1 SD band is omitted "
        "for visual clarity.",
        "- External AUC CI: stratified nonparametric percentile bootstrap, "
        f"{bootstrap_iterations} resamples, seed={random_seed}.",
        "- Threshold metrics: development OOF Youden threshold saved by training.",
        "- Supplementary calibration: five-fold mean curves and held-out "
        "quantile-binned curves are shown without uncertainty shading.",
        "- Supplementary DCA: CV uses mean validation-fold net benefit; the "
        "external cohort uses cohort-level net benefit; TP/N - FP/N × pt/(1-pt), "
        "plus treat-all and treat-none.",
        "",
        "INTERPRETATION NOTE",
        "- CV uses validation-fold OOF predictions only, never training predictions.",
        "- External labels never select a model, parameter, or threshold.",
        "- Final performance claims should rest on the held-out external cohort; "
        "CV describes development behaviour only.",
        "",
        "GENERATED FILES",
    ])
    lines.extend(
        f"- {path.resolve().relative_to(output_dir.resolve())}"
        for path in sorted(generated_files)
    )
    summary_path = output_dir / "summary" / "evaluation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_publication_evaluation(
        output_dir: Path,
        selected_models: pd.DataFrame,
        cv_predictions: pd.DataFrame,
        external_predictions: pd.DataFrame,
        late_model_cv_predictions: Optional[pd.DataFrame] = None,
        late_model_external_predictions: Optional[pd.DataFrame] = None,
        bootstrap_iterations: int = 2000,
        random_seed: int = 20260809,
        calibration_bins: int = 10,
        dca_min_threshold: float = 0.01,
        dca_max_threshold: float = 0.80,
        figure_dpi: int = 600,
):
    """Generate manuscript outputs from saved predictions, without fitting models."""
    print("\n[Publication Evaluation] Validating saved predictions...")
    required_winner_columns = {
        "Dataset", "Best_Model", "OOF_Decision_Threshold"
    }
    missing = required_winner_columns - set(selected_models.columns)
    if missing:
        raise ValueError(f"selected_models is missing columns: {sorted(missing)}")
    if selected_models["Dataset"].duplicated().any():
        raise ValueError("selected_models contains duplicate datasets.")
    missing_winners = set(PUBLICATION_MODEL_ORDER) - set(selected_models["Dataset"])
    if missing_winners:
        raise ValueError(f"selected_models is missing: {sorted(missing_winners)}")

    cv_predictions = publication_validate_predictions(
        cv_predictions, selected_models, "cv"
    )
    external_predictions = publication_validate_predictions(
        external_predictions, selected_models, "external"
    )
    if (
            late_model_cv_predictions is None
            or late_model_external_predictions is None
            or late_model_external_predictions.empty
    ):
        raise ValueError(
            "Late_Fusion meta-model CV and external predictions are required."
        )
    late_model_cv_predictions = publication_validate_late_model_predictions(
        late_model_cv_predictions, "cv"
    )
    late_model_external_predictions = publication_validate_late_model_predictions(
        late_model_external_predictions, "external"
    )
    expected_meta_models = publication_available_meta_models(
        late_model_cv_predictions
    )
    if publication_available_meta_models(
            late_model_external_predictions
    ) != expected_meta_models:
        raise ValueError(
            "Late_Fusion CV/external meta-model sets do not match."
        )

    directories = publication_output_directories(output_dir)
    cv_table = publication_cv_table(cv_predictions, selected_models)
    external_table = publication_test_table(
        external_predictions, selected_models,
        bootstrap_iterations, random_seed,
    )
    publication_write_table(
        cv_table, directories["tables"] / "cv_performance_table.csv"
    )
    publication_write_table(
        external_table,
        directories["tables"] / "external_test_performance_table.csv",
    )
    late_model_auc_table = publication_late_model_auc_table(
        late_model_cv_predictions,
        late_model_external_predictions,
        bootstrap_iterations,
        random_seed,
    )
    publication_write_table(
        late_model_auc_table,
        directories["tables"] / "late_fusion_meta_model_AUC_comparison.csv",
    )

    publication_plot_late_model_roc(
        late_model_cv_predictions, "Five-fold cross-validation", True,
        directories["roc"] / "late_fusion_models_ROC_5fold_CV.tiff",
        bootstrap_iterations, random_seed, figure_dpi,
    )
    publication_plot_late_model_roc(
        late_model_external_predictions, "External test cohort", False,
        directories["roc"] / "late_fusion_models_ROC_external.tiff",
        bootstrap_iterations, random_seed, figure_dpi,
    )
    publication_plot_late_model_calibration(
        late_model_cv_predictions, "Five-fold cross-validation", True,
        directories["calibration"] /
        "late_fusion_models_calibration_5fold_CV.tiff",
        calibration_bins, figure_dpi,
    )
    publication_plot_late_model_calibration(
        late_model_external_predictions, "External test cohort", False,
        directories["calibration"] /
        "late_fusion_models_calibration_external.tiff",
        calibration_bins, figure_dpi,
    )
    publication_plot_late_model_dca(
        late_model_cv_predictions, "Five-fold cross-validation", True,
        directories["dca"] / "late_fusion_models_DCA_5fold_CV.tiff",
        dca_min_threshold, dca_max_threshold, figure_dpi,
    )
    publication_plot_late_model_dca(
        late_model_external_predictions, "External test cohort", False,
        directories["dca"] / "late_fusion_models_DCA_external.tiff",
        dca_min_threshold, dca_max_threshold, figure_dpi,
    )
    publication_plot_confusion_matrices(
        external_predictions, selected_models, "External test",
        directories["confusion"] / "external_confusion_matrix.tiff",
        figure_dpi,
    )

    summary_path = directories["summary"] / "evaluation_summary.txt"
    generated_files = [
        path
        for directory in directories.values()
        for path in directory.rglob("*")
        if path.is_file() and path != summary_path
    ]
    publication_write_summary(
        output_dir, selected_models, cv_predictions, external_predictions,
        generated_files, bootstrap_iterations, random_seed,
    )
    print(
        f"[Publication Evaluation] DONE. "
        f"Generated {len(generated_files) + 1} files inside: {output_dir}"
    )


# ============================================================
# 9. MAIN
# ============================================================

def format_cv_metric(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def metric_summary_row(dataset, model_name, n_features, metrics, extra=None):
    row = {
        "Dataset": dataset,
        "Model": model_name,
        "N_Features": n_features,
        "AUC": float(metrics["AUC"]),
        "F1": float(metrics["F1"]),
        "Acc": float(metrics["Acc"]),
        "Sens (Recall)": float(metrics["Sens"]),
        "Spec": float(metrics["Spec"]),
        "Brier (Calib)": float(metrics["Brier"]),
        "LogLoss": float(metrics["LogLoss"]),
    }
    if extra:
        row.update(extra)
    return row


def cv_summary_row(
        dataset,
        model_name,
        sampler_choice,
        evaluation: CVEvaluation,
        stage,
        feature_selection,
        best_params=None,
        optuna_best_cv_auc=np.nan,
):
    mean = evaluation.metrics_cv_mean
    std = evaluation.metrics_cv_std
    return {
        "Dataset": dataset,
        "Model": model_name,
        "Stage": stage,
        "Feature_Selection": feature_selection,
        "Sampler": sampler_choice,
        "N_Features_Mean": evaluation.mean_n_features,
        "N_Features_SD": evaluation.std_n_features,
        "AUC_Mean": mean["AUC"],
        "AUC_SD": std["AUC"],
        "AUC": format_cv_metric(mean["AUC"], std["AUC"]),
        "F1": format_cv_metric(mean["F1"], std["F1"]),
        "Acc": format_cv_metric(mean["Acc"], std["Acc"]),
        "Sens (Recall)": format_cv_metric(mean["Sens"], std["Sens"]),
        "Spec": format_cv_metric(mean["Spec"], std["Spec"]),
        "Brier_Mean": mean["Brier"],
        "Brier_SD": std["Brier"],
        "Brier (Calib)": format_cv_metric(mean["Brier"], std["Brier"]),
        "LogLoss_Mean": mean["LogLoss"],
        "LogLoss_SD": std["LogLoss"],
        "LogLoss": format_cv_metric(mean["LogLoss"], std["LogLoss"]),
        "Optuna_Best_CV_AUC": optuna_best_cv_auc,
        "Best_Params": json.dumps(best_params or {}, ensure_ascii=False, sort_keys=True),
    }


def main():
    data_sources = {
        "Radiomics": str(DATA_DIR / "radiomics_features_new.csv"),
        "Clinical": str(DATA_DIR / "clinical_features_new.csv"),
        "Muscle": str(DATA_DIR / "CSA_muscles_features_new.csv"),
    }
    combinations = {
        "Clinical": ["Clinical"],
        "Muscle": ["Muscle"],
        "Radiomics": ["Radiomics"],
        "Clinical_Radiomics": ["Clinical", "Radiomics"],
        "Clinical_Muscle": ["Clinical", "Muscle"],
        "Radiomics_Muscle": ["Radiomics", "Muscle"],
        "Full_Fusion": ["Radiomics", "Clinical", "Muscle"],
    }
    n_trials = 50
    cv_folds = 5
    top_models_to_tune = 2
    top_meta_models_to_tune = 2
    # External is the only hold-out now, so turning it off would leave the run
    # with no test cohort at all. The flag stays for provenance, not as a knob.
    run_external_validation = True
    external_dir = DATA_DIR / "external_validation_WF"
    if not run_external_validation:
        raise RuntimeError(
            "External validation is the only test cohort in this pipeline; "
            "disabling it would leave nothing to evaluate the winners on."
        )
    late_fusion_sets = {
        "Late_Fusion_Clinical_Radiomics": ["Clinical", "Radiomics"],
        "Late_Fusion_Clinical_Muscle": ["Clinical", "Muscle"],
        "Late_Fusion_Radiomics_Muscle": ["Radiomics", "Muscle"],
        "Late_Fusion": ["Clinical", "Muscle", "Radiomics"],
    }
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "2026-09-13_ml9_dev_all_external_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion = DataFusionPipeline(data_sources)
    fusion.load_and_align()
    models_map = get_models_map()
    meta_models_map = get_meta_models_map()
    if not models_map:
        raise RuntimeError("No model configuration is available.")

    # Every aligned internal case is development data. The external cohort is
    # the only hold-out, so nothing is carved out here.
    all_labels = fusion.label_col.to_numpy(dtype=int)
    cohort_audit = pd.DataFrame({
        "Case_ID": np.asarray(fusion.common_index).astype(str),
        "Label": all_labels,
        "Split": "Development",
        "Positive_Class": "1 = subsidence",
    })
    cohort_audit.to_csv(output_dir / "development_cohort_audit.csv", index=False)

    # Phase 1 is completed once before any Radiomics/Fusion model is built.
    # Fold-specific selections preserve honest CV; the final-development
    # selection is shared by the two final model branches.
    radiomics_merged = fusion.get_merged_dataset(["Radiomics"])
    X_radiomics_all = radiomics_merged.drop(columns=["Label"])
    X_radiomics_development = X_radiomics_all.copy()
    shared_radiomics_selections = prepare_shared_radiomics_selections(
        X_radiomics_development,
        y_development=all_labels,
        cv_folds=cv_folds,
        output_dir=output_dir,
    )

    default_cv_rows = []
    screening_cv_rows = []
    tuned_cv_rows = []
    final_cv_rows = []
    selected_model_rows = []
    external_summary_rows = []
    external_prediction_rows = []
    late_model_comparison_config_rows = []
    late_model_comparison_cv_rows = []
    late_model_comparison_external_rows = []
    external_label_audit = None
    selected_oof_rows = []
    final_results: Dict[str, MLResults] = {}
    base_oof_probability_frames: Dict[str, pd.DataFrame] = {}
    base_external_probability_frames: Dict[str, pd.DataFrame] = {}
    base_external_feature_frames: Dict[str, pd.DataFrame] = {}
    base_winner_names: Dict[str, str] = {}
    full_late_external_inputs: Optional[pd.DataFrame] = None
    full_late_external_probability: Optional[pd.Series] = None

    print(
        "\nSTARTING ML9 AUC-FIRST: SINGLE/PAIRWISE/FULL EARLY FUSION "
        "+ ALL-STANDARD-MODEL LATE FUSION + AUC OPTUNA"
    )
    print(
        f"Development cohort: all {len(fusion.common_index)} aligned internal "
        "cases (no internal hold-out)."
    )
    print(
        f"External test cohort: {external_dir} "
        "(scoring only; never used for model selection)."
    )
    for combo_name, source_keys in combinations.items():
        print(f"\n{'=' * 70}\nDataset: {combo_name} {source_keys}")
        merged = fusion.get_merged_dataset(source_keys)
        X_all = merged.drop(columns=["Label"])
        y_all = merged["Label"].to_numpy(dtype=int)
        X_development = X_all.copy()
        y_development = y_all

        prepared_folds = prepare_cv_folds(
            X_development,
            y_development,
            combo_name,
            cv_folds,
            output_dir=output_dir,
            shared_radiomics_selections=(
                shared_radiomics_selections if "Radiomics" in source_keys else None
            ),
        )

        feature_selection_label = (
            "Radiomics-only NZV+corr+L1-logistic C by inner-CV AUC"
            if "Radiomics" in source_keys else "None"
        )
        screened_models = {}
        for model_name, (estimator_template, optuna_fn) in models_map.items():
            sampler_evaluations = {}
            for sampler_choice in get_sampler_choices(estimator_template):
                print(
                    f"  -> Default CV: {combo_name} | {model_name} | "
                    f"sampler={sampler_choice}"
                )
                try:
                    evaluation = evaluate_model_cv(
                        prepared_folds,
                        estimator_template,
                        sampler_choice=sampler_choice,
                    )
                except Exception as exc:
                    print(
                        f"     skipped {model_name}/{sampler_choice} because CV failed: {exc}"
                    )
                    continue
                sampler_evaluations[sampler_choice] = evaluation
                stage_label = f"Default sampler ablation ({sampler_choice})"
                evaluation.fold_metrics.assign(
                    Dataset=combo_name,
                    Model=model_name,
                    Stage=stage_label,
                ).to_csv(
                    output_dir / (
                        f"cv_folds_default_{safe_file_name(combo_name)}_"
                        f"{safe_file_name(model_name)}_{sampler_choice}.csv"
                    ),
                    index=False,
                )
                default_cv_rows.append(cv_summary_row(
                    combo_name,
                    model_name,
                    sampler_choice,
                    evaluation,
                    stage=stage_label,
                    feature_selection=feature_selection_label,
                ))

            if not sampler_evaluations:
                continue
            selected_sampler, selected_evaluation = choose_sampler_candidate(
                sampler_evaluations
            )
            screening_cv_rows.append(cv_summary_row(
                combo_name,
                model_name,
                selected_sampler,
                selected_evaluation,
                stage="Selected sampler after ablation",
                feature_selection=feature_selection_label,
            ))
            screened_models[model_name] = {
                "sampler_choice": selected_sampler,
                "evaluation": selected_evaluation,
                "estimator_template": estimator_template,
                "optuna_fn": optuna_fn,
            }
            print(
                f"     selected sampler={selected_sampler} for {model_name}; "
                f"AUC={selected_evaluation.metrics_cv_mean['AUC']:.4f}, "
                f"LogLoss={selected_evaluation.metrics_cv_mean['LogLoss']:.4f}"
            )

        if not screened_models:
            raise RuntimeError(f"All models failed for {combo_name}.")

        shortlist = sorted(
            screened_models.items(),
            key=lambda item: evaluation_rank_key(item[1]["evaluation"]),
            reverse=True,
        )[:top_models_to_tune]
        print(
            "  Top models entering Optuna: "
            + ", ".join(
                f"{name} (AUC={info['evaluation'].metrics_cv_mean['AUC']:.4f}, "
                f"sampler={info['sampler_choice']})"
                for name, info in shortlist
            )
        )

        tuned_candidates = {}
        for model_name, screened in shortlist:
            print(f"  -> Optuna: {combo_name} | {model_name}")
            candidate = tune_model_candidate(
                combo_name,
                model_name,
                prepared_folds,
                screened["estimator_template"],
                screened["optuna_fn"],
                n_trials,
                output_dir,
                baseline_evaluation=screened["evaluation"],
                sampler_choice=screened["sampler_choice"],
            )
            tuned_candidates[model_name] = candidate
            candidate_evaluation = candidate["evaluation"]
            candidate_evaluation.fold_metrics.assign(
                Dataset=combo_name,
                Model=model_name,
                Stage=candidate["stage"],
            ).to_csv(
                output_dir / (
                    f"cv_folds_tuned_{safe_file_name(combo_name)}_"
                    f"{safe_file_name(model_name)}.csv"
                ),
                index=False,
            )
            tuned_cv_rows.append(cv_summary_row(
                combo_name,
                model_name,
                candidate["sampler_choice"],
                candidate_evaluation,
                stage=candidate["stage"],
                feature_selection=feature_selection_label,
                best_params=candidate["best_params"],
                optuna_best_cv_auc=candidate["optuna_best_cv_auc"],
            ))

        # cv_results.csv contains the best available post-selection result for
        # every model: tuned/fallback for top 2, sampler-selected default otherwise.
        for model_name, screened in screened_models.items():
            if model_name in tuned_candidates:
                candidate = tuned_candidates[model_name]
                final_cv_rows.append(cv_summary_row(
                    combo_name,
                    model_name,
                    candidate["sampler_choice"],
                    candidate["evaluation"],
                    stage=candidate["stage"],
                    feature_selection=feature_selection_label,
                    best_params=candidate["best_params"],
                    optuna_best_cv_auc=candidate["optuna_best_cv_auc"],
                ))
            else:
                final_cv_rows.append(cv_summary_row(
                    combo_name,
                    model_name,
                    screened["sampler_choice"],
                    screened["evaluation"],
                    stage="Sampler-selected default (not shortlisted)",
                    feature_selection=feature_selection_label,
                ))

        winner_name, winner_candidate = max(
            tuned_candidates.items(),
            key=lambda item: evaluation_rank_key(item[1]["evaluation"]),
        )
        winner_evaluation = winner_candidate["evaluation"]
        winner_template = models_map[winner_name][0]
        print(
            f"  Final winner: {winner_name} | sampler={winner_candidate['sampler_choice']} | "
            f"AUC={winner_evaluation.metrics_cv_mean['AUC']:.4f} "
            f"± {winner_evaluation.metrics_cv_std['AUC']:.4f}"
        )
        result = fit_final_model(
            combo_name,
            winner_name,
            X_development,
            y_development,
            winner_template,
            selected_evaluation=winner_evaluation,
            best_params=winner_candidate["best_params"],
            optuna_best_cv_auc=winner_candidate["optuna_best_cv_auc"],
            output_dir=output_dir,
            sampler_choice=winner_candidate["sampler_choice"],
            final_radiomics_selection=(
                shared_radiomics_selections.final_development
                if "Radiomics" in source_keys else None
            ),
        )
        final_results[f"{combo_name}|{winner_name}"] = result
        oof = result.selected_cv_evaluation.oof_predictions.copy()
        if combo_name in {"Clinical", "Muscle", "Radiomics"}:
            base_winner_names[combo_name] = winner_name
            base_oof_probability_frames[combo_name] = oof[[
                "Fold", "Case_ID", "True_Label", "Probability_Subsidence"
            ]].copy()
        oof_metrics = compute_metrics(
            oof["True_Label"],
            oof["Probability_Subsidence"],
            threshold=result.decision_threshold,
        )
        selected_model_rows.append({
            "Dataset": combo_name,
            "Best_Model": winner_name,
            "Selection_Stage": winner_candidate["stage"],
            "Sampler": result.sampler_choice,
            "Selection_CV_AUC_Mean": winner_evaluation.metrics_cv_mean["AUC"],
            "Selection_CV_AUC_SD": winner_evaluation.metrics_cv_std["AUC"],
            "Selection_CV_Brier_Mean": winner_evaluation.metrics_cv_mean["Brier"],
            "Selection_CV_LogLoss_Mean": winner_evaluation.metrics_cv_mean["LogLoss"],
            "Optuna_Best_CV_AUC": result.optuna_best_cv_auc,
            "OOF_Decision_Threshold": result.decision_threshold,
            "OOF_Pooled_AUC": oof_metrics["AUC"],
            "OOF_Pooled_Acc_At_Threshold": oof_metrics["Acc"],
            "OOF_Pooled_Sens_At_Threshold": oof_metrics["Sens"],
            "OOF_Pooled_Spec_At_Threshold": oof_metrics["Spec"],
            "Final_N_Features": len(result.selected_features),
            "Best_Params": json.dumps(result.best_params, ensure_ascii=False, sort_keys=True),
        })

        oof["Dataset"] = combo_name
        oof["Model"] = winner_name
        oof["Sampler"] = result.sampler_choice
        oof["Decision_Threshold"] = result.decision_threshold
        oof["Prediction_Threshold_0.5"] = (
            oof["Probability_Subsidence"] >= 0.5
        ).astype(int)
        oof["Prediction_Optimized_Threshold"] = (
            oof["Probability_Subsidence"] >= result.decision_threshold
        ).astype(int)
        selected_oof_rows.extend(oof.to_dict("records"))

        if run_external_validation:
            X_external, y_external, _, aligned_audit = (
                load_external_validation_dataset(
                    external_dir,
                    fusion,
                    source_keys,
                )
            )
            if external_label_audit is None:
                external_label_audit = aligned_audit.copy()
            external_probability = result.fitted_model.predict_proba(
                X_external[result.selected_features]
            )[:, 1]
            external_case_ids = X_external.index.astype(str)
            if combo_name in {"Clinical", "Muscle", "Radiomics"}:
                # The feature frame feeds end-to-end SHAP; the probability
                # frame feeds late fusion. Both are held-out external cases.
                external_features = X_external.loc[:, result.selected_features].copy()
                external_features.index = pd.Index(
                    external_case_ids, name="Case_ID"
                )
                base_external_feature_frames[combo_name] = external_features
                base_external_probability_frames[combo_name] = pd.DataFrame({
                    "Case_ID": external_case_ids,
                    "True_Label": y_external,
                    "Probability_Subsidence": external_probability,
                })
            external_metrics = compute_metrics(
                y_external,
                external_probability,
                threshold=result.decision_threshold,
            )
            external_summary_rows.append(metric_summary_row(
                combo_name,
                winner_name,
                len(result.selected_features),
                external_metrics,
                extra={
                    "N_External": len(y_external),
                    "Sampler": result.sampler_choice,
                    "Decision_Threshold_From_OOF": result.decision_threshold,
                },
            ))
            for case_id, true_label, probability in zip(
                    X_external.index, y_external, external_probability
            ):
                external_prediction_rows.append({
                    "Dataset": combo_name,
                    "Model": winner_name,
                    "Case_ID": str(case_id),
                    "True_Label": int(true_label),
                    "Probability_Subsidence": float(probability),
                    "Decision_Threshold_From_OOF": result.decision_threshold,
                    "Prediction_Threshold_0.5": int(probability >= 0.5),
                    "Prediction_Optimized_Threshold": int(
                        probability >= result.decision_threshold
                    ),
                })

    # ------------------------------------------------------------
    # Late fusion ablation: all pairwise modality sets and the three-way set.
    # Inputs are cross-fitted base probabilities; the external cohort is not
    # used for sampler, model, hyperparameter, or meta-coefficient selection.
    # ------------------------------------------------------------
    late_fusion_run_config = {}
    late_winner_names = {}
    for late_dataset_name, late_modalities in late_fusion_sets.items():
        print(f"\n{'=' * 70}\nDataset: {late_dataset_name} {late_modalities}")
        safe_late_dataset = safe_file_name(late_dataset_name)
        late_oof_table = build_late_fusion_table(
            base_oof_probability_frames,
            late_modalities,
            require_fold=True,
        )
        late_external_table = None
        if run_external_validation:
            late_external_table = build_late_fusion_table(
                base_external_probability_frames,
                late_modalities,
                require_fold=False,
            )
        late_feature_columns = [
            f"{modality}_Probability" for modality in late_modalities
        ]
        late_oof_table.reset_index().to_csv(
            output_dir / f"{safe_late_dataset}_oof_inputs.csv", index=False
        )
        if late_external_table is not None:
            late_external_table.reset_index().to_csv(
                output_dir / f"{safe_late_dataset}_external_inputs.csv", index=False
            )

        late_prepared_folds = prepare_late_fusion_folds(
            late_oof_table, late_modalities
        )
        late_feature_selection_label = (
            f"{len(late_modalities)} aligned base-modality OOF probabilities"
        )
        late_screened_models = {}
        for model_name, (estimator_template, optuna_fn) in meta_models_map.items():
            sampler_evaluations = {}
            for sampler_choice in get_sampler_choices(estimator_template):
                print(
                    f"  -> Default CV: {late_dataset_name} | {model_name} | "
                    f"sampler={sampler_choice}"
                )
                try:
                    evaluation = evaluate_model_cv(
                        late_prepared_folds,
                        estimator_template,
                        sampler_choice=sampler_choice,
                    )
                except Exception as exc:
                    print(
                        f"     skipped {model_name}/{sampler_choice} because CV failed: {exc}"
                    )
                    continue
                sampler_evaluations[sampler_choice] = evaluation
                stage_label = f"Meta sampler ablation ({sampler_choice})"
                evaluation.fold_metrics.assign(
                    Dataset=late_dataset_name,
                    Model=model_name,
                    Stage=stage_label,
                ).to_csv(
                    output_dir / (
                        f"cv_folds_default_{safe_late_dataset}_"
                        f"{safe_file_name(model_name)}_{sampler_choice}.csv"
                    ),
                    index=False,
                )
                default_cv_rows.append(cv_summary_row(
                    late_dataset_name,
                    model_name,
                    sampler_choice,
                    evaluation,
                    stage=stage_label,
                    feature_selection=late_feature_selection_label,
                ))

            if not sampler_evaluations:
                continue
            selected_sampler, selected_evaluation = choose_sampler_candidate(
                sampler_evaluations
            )
            screening_cv_rows.append(cv_summary_row(
                late_dataset_name,
                model_name,
                selected_sampler,
                selected_evaluation,
                stage="Selected meta sampler after ablation",
                feature_selection=late_feature_selection_label,
            ))
            late_screened_models[model_name] = {
                "sampler_choice": selected_sampler,
                "evaluation": selected_evaluation,
                "estimator_template": estimator_template,
                "optuna_fn": optuna_fn,
            }

        if not late_screened_models:
            raise RuntimeError(f"All meta-models failed for {late_dataset_name}.")

        late_shortlist = sorted(
            late_screened_models.items(),
            key=lambda item: evaluation_rank_key(item[1]["evaluation"]),
            reverse=True,
        )[:top_meta_models_to_tune]
        print(
            "  Meta-models entering final tuning: "
            + ", ".join(
                f"{name} (AUC={info['evaluation'].metrics_cv_mean['AUC']:.4f}, "
                f"sampler={info['sampler_choice']})"
                for name, info in late_shortlist
            )
        )

        late_tuned_candidates = {}
        for model_name, screened in late_shortlist:
            print(f"  -> Final tuning: {late_dataset_name} | {model_name}")
            candidate = tune_model_candidate(
                late_dataset_name,
                model_name,
                late_prepared_folds,
                screened["estimator_template"],
                screened["optuna_fn"],
                n_trials,
                output_dir,
                baseline_evaluation=screened["evaluation"],
                sampler_choice=screened["sampler_choice"],
            )
            late_tuned_candidates[model_name] = candidate
            candidate_evaluation = candidate["evaluation"]
            candidate_evaluation.fold_metrics.assign(
                Dataset=late_dataset_name,
                Model=model_name,
                Stage=candidate["stage"],
            ).to_csv(
                output_dir / (
                    f"cv_folds_tuned_{safe_late_dataset}_"
                    f"{safe_file_name(model_name)}.csv"
                ),
                index=False,
            )
            tuned_cv_rows.append(cv_summary_row(
                late_dataset_name,
                model_name,
                candidate["sampler_choice"],
                candidate_evaluation,
                stage=candidate["stage"],
                feature_selection=late_feature_selection_label,
                best_params=candidate["best_params"],
                optuna_best_cv_auc=candidate["optuna_best_cv_auc"],
            ))

        for model_name, screened in late_screened_models.items():
            if model_name in late_tuned_candidates:
                candidate = late_tuned_candidates[model_name]
                final_cv_rows.append(cv_summary_row(
                    late_dataset_name,
                    model_name,
                    candidate["sampler_choice"],
                    candidate["evaluation"],
                    stage=candidate["stage"],
                    feature_selection=late_feature_selection_label,
                    best_params=candidate["best_params"],
                    optuna_best_cv_auc=candidate["optuna_best_cv_auc"],
                ))
            else:
                final_cv_rows.append(cv_summary_row(
                    late_dataset_name,
                    model_name,
                    screened["sampler_choice"],
                    screened["evaluation"],
                    stage="Sampler-selected meta default (not shortlisted)",
                    feature_selection=late_feature_selection_label,
                ))

        late_winner_name, late_candidate = max(
            late_tuned_candidates.items(),
            key=lambda item: evaluation_rank_key(item[1]["evaluation"]),
        )
        late_evaluation = late_candidate["evaluation"]
        late_template = late_screened_models[late_winner_name]["estimator_template"]
        print(
            f"  Final {late_dataset_name} winner: {late_winner_name} | "
            f"sampler={late_candidate['sampler_choice']} | "
            f"AUC={late_evaluation.metrics_cv_mean['AUC']:.4f} "
            f"± {late_evaluation.metrics_cv_std['AUC']:.4f}"
        )

        X_late_development = late_oof_table[late_feature_columns].copy()
        y_late_development = late_oof_table["True_Label"].to_numpy(dtype=int)
        late_result = fit_final_model(
            late_dataset_name,
            late_winner_name,
            X_late_development,
            y_late_development,
            late_template,
            selected_evaluation=late_evaluation,
            best_params=late_candidate["best_params"],
            optuna_best_cv_auc=late_candidate["optuna_best_cv_auc"],
            output_dir=output_dir,
            sampler_choice=late_candidate["sampler_choice"],
        )
        final_results[f"{late_dataset_name}|{late_winner_name}"] = late_result
        late_winner_names[late_dataset_name] = late_winner_name

        fitted_steps = late_result.fitted_model.named_steps
        if "lr" in fitted_steps:
            late_preprocessor = fitted_steps["pre"]
            late_estimator = fitted_steps["lr"]
            scaled_coefficients = late_estimator.coef_.ravel()
            scale_lookup = dict(zip(
                late_preprocessor.scale_columns_, late_preprocessor.scaler_.scale_
            ))
            mean_lookup = dict(zip(
                late_preprocessor.scale_columns_, late_preprocessor.scaler_.mean_
            ))
            original_scale_coefficients = np.asarray([
                coefficient / scale_lookup.get(feature, 1.0)
                for feature, coefficient in zip(
                    late_feature_columns, scaled_coefficients
                )
            ])
            original_scale_intercept = float(
                late_estimator.intercept_[0]
                - sum(
                    coefficient * mean_lookup.get(feature, 0.0)
                    / scale_lookup.get(feature, 1.0)
                    for feature, coefficient in zip(
                        late_feature_columns, scaled_coefficients
                    )
                )
            )
            pd.DataFrame([{
                "Feature": feature,
                "Base_Model": base_winner_names[
                    feature.removesuffix("_Probability")
                ],
                "Standardized_Coefficient": float(scaled_coefficient),
                "Original_Probability_Scale_Coefficient": float(
                    original_coefficient
                ),
                "Original_Probability_Scale_Intercept": original_scale_intercept,
            } for feature, scaled_coefficient, original_coefficient in zip(
                late_feature_columns,
                scaled_coefficients,
                original_scale_coefficients,
            )]).to_csv(
                output_dir / f"late_fusion_coefficients_{safe_late_dataset}.csv",
                index=False,
            )
        late_oof = late_result.selected_cv_evaluation.oof_predictions.copy()
        late_oof_metrics = compute_metrics(
            late_oof["True_Label"],
            late_oof["Probability_Subsidence"],
            threshold=late_result.decision_threshold,
        )
        selected_model_rows.append({
            "Dataset": late_dataset_name,
            "Best_Model": late_winner_name,
            "Selection_Stage": late_candidate["stage"],
            "Sampler": late_result.sampler_choice,
            "Selection_CV_AUC_Mean": late_evaluation.metrics_cv_mean["AUC"],
            "Selection_CV_AUC_SD": late_evaluation.metrics_cv_std["AUC"],
            "Selection_CV_Brier_Mean": late_evaluation.metrics_cv_mean["Brier"],
            "Selection_CV_LogLoss_Mean": late_evaluation.metrics_cv_mean["LogLoss"],
            "Optuna_Best_CV_AUC": late_result.optuna_best_cv_auc,
            "OOF_Decision_Threshold": late_result.decision_threshold,
            "OOF_Pooled_AUC": late_oof_metrics["AUC"],
            "OOF_Pooled_Acc_At_Threshold": late_oof_metrics["Acc"],
            "OOF_Pooled_Sens_At_Threshold": late_oof_metrics["Sens"],
            "OOF_Pooled_Spec_At_Threshold": late_oof_metrics["Spec"],
            "Final_N_Features": len(late_result.selected_features),
            "Best_Params": json.dumps(
                late_result.best_params, ensure_ascii=False, sort_keys=True
            ),
        })
        late_oof["Dataset"] = late_dataset_name
        late_oof["Model"] = late_winner_name
        late_oof["Sampler"] = late_result.sampler_choice
        late_oof["Decision_Threshold"] = late_result.decision_threshold
        late_oof["Prediction_Threshold_0.5"] = (
            late_oof["Probability_Subsidence"] >= 0.5
        ).astype(int)
        late_oof["Prediction_Optimized_Threshold"] = (
            late_oof["Probability_Subsidence"] >= late_result.decision_threshold
        ).astype(int)
        selected_oof_rows.extend(late_oof.to_dict("records"))

        if late_external_table is not None:
            X_late_external = late_external_table[late_feature_columns].copy()
            y_late_external = late_external_table["True_Label"].to_numpy(dtype=int)
            late_external_probability = late_result.fitted_model.predict_proba(
                X_late_external
            )[:, 1]
            if late_dataset_name == "Late_Fusion":
                full_late_external_inputs = X_late_external.copy()
                full_late_external_probability = pd.Series(
                    late_external_probability,
                    index=X_late_external.index.astype(str),
                    name="Probability_Subsidence",
                )
            late_external_metrics = compute_metrics(
                y_late_external,
                late_external_probability,
                threshold=late_result.decision_threshold,
            )
            external_summary_rows.append(metric_summary_row(
                late_dataset_name,
                late_winner_name,
                len(late_result.selected_features),
                late_external_metrics,
                extra={
                    "N_External": len(y_late_external),
                    "Sampler": late_result.sampler_choice,
                    "Decision_Threshold_From_OOF": late_result.decision_threshold,
                },
            ))
            for case_id, true_label, probability in zip(
                    X_late_external.index,
                    y_late_external,
                    late_external_probability,
            ):
                external_prediction_rows.append({
                    "Dataset": late_dataset_name,
                    "Model": late_winner_name,
                    "Case_ID": str(case_id),
                    "True_Label": int(true_label),
                    "Probability_Subsidence": float(probability),
                    "Decision_Threshold_From_OOF": late_result.decision_threshold,
                    "Prediction_Threshold_0.5": int(probability >= 0.5),
                    "Prediction_Optimized_Threshold": int(
                        probability >= late_result.decision_threshold
                    ),
                })

        # The publication ROC/calibration/DCA figures compare algorithms, not
        # modality branches. Build development-selected predictions for every
        # available meta-model only for the full three-modality Late_Fusion.
        if late_dataset_name == "Late_Fusion":
            for model_name, screened in late_screened_models.items():
                if model_name in late_tuned_candidates:
                    comparison_candidate = late_tuned_candidates[model_name]
                    comparison_evaluation = comparison_candidate["evaluation"]
                    comparison_sampler = comparison_candidate["sampler_choice"]
                    comparison_params = comparison_candidate["best_params"]
                    comparison_stage = comparison_candidate["stage"]
                else:
                    comparison_evaluation = screened["evaluation"]
                    comparison_sampler = screened["sampler_choice"]
                    comparison_params = {}
                    comparison_stage = "Sampler-selected default (not shortlisted)"

                comparison_oof = comparison_evaluation.oof_predictions.copy()
                comparison_threshold = choose_oof_threshold(
                    comparison_oof["True_Label"],
                    comparison_oof["Probability_Subsidence"],
                )
                comparison_oof["Dataset"] = "Late_Fusion"
                comparison_oof["Model"] = model_name
                comparison_oof["Sampler"] = comparison_sampler
                comparison_oof["Decision_Threshold"] = comparison_threshold
                late_model_comparison_cv_rows.extend(
                    comparison_oof.to_dict("records")
                )

                if model_name == late_winner_name:
                    comparison_model = late_result.fitted_model
                else:
                    comparison_model = build_training_pipeline(
                        screened["estimator_template"],
                        late_feature_columns,
                        sampler_choice=comparison_sampler,
                        y_train=y_late_development,
                    )
                    if comparison_params:
                        comparison_model.set_params(**comparison_params)
                    comparison_model.fit(X_late_development, y_late_development)

                if late_external_table is not None:
                    comparison_external_probability = (
                        comparison_model.predict_proba(X_late_external)[:, 1]
                    )
                    for case_id, true_label, probability in zip(
                            X_late_external.index,
                            y_late_external,
                            comparison_external_probability,
                    ):
                        late_model_comparison_external_rows.append({
                            "Dataset": "Late_Fusion",
                            "Model": model_name,
                            "Case_ID": str(case_id),
                            "True_Label": int(true_label),
                            "Probability_Subsidence": float(probability),
                            "Decision_Threshold_From_OOF": comparison_threshold,
                        })

                late_model_comparison_config_rows.append({
                    "Dataset": "Late_Fusion",
                    "Model": model_name,
                    "Stage": comparison_stage,
                    "Sampler": comparison_sampler,
                    "CV_AUC_Mean": comparison_evaluation.metrics_cv_mean["AUC"],
                    "CV_AUC_SD": comparison_evaluation.metrics_cv_std["AUC"],
                    "OOF_Decision_Threshold": comparison_threshold,
                    "Best_Params": json.dumps(
                        comparison_params, ensure_ascii=False, sort_keys=True
                    ),
                })

        late_fusion_run_config[late_dataset_name] = {
            "base_modalities": late_modalities,
            "selected_meta_model": late_winner_name,
            "selected_sampler": late_result.sampler_choice,
            "base_winner_models": {
                modality: base_winner_names[modality]
                for modality in late_modalities
            },
        }

    # Export only after single, pairwise, and full winners have been fitted.
    # The web backend loads these adaptive routes once; it never retrains.
    # Save directly in artifacts/, matching the web app's default bundle path.
    deployment_bundle_path = export_late_fusion_bundle(
        final_results=final_results,
        base_winner_names=base_winner_names,
        late_winner_names=late_winner_names,
        output_dir=output_dir,
        deployment_dir=PROJECT_ROOT / "artifacts",
    )

    pd.DataFrame(default_cv_rows).to_csv(
        output_dir / "cv_results_default.csv", index=False
    )
    pd.DataFrame(screening_cv_rows).to_csv(
        output_dir / "cv_results_screening.csv", index=False
    )
    pd.DataFrame(tuned_cv_rows).to_csv(
        output_dir / "cv_results_tuned.csv", index=False
    )
    pd.DataFrame(final_cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(selected_model_rows).to_csv(output_dir / "selected_models.csv", index=False)
    if run_external_validation:
        external_summary_table = pd.DataFrame(external_summary_rows)
        external_summary_table.to_csv(
            output_dir / "external_validation_results.csv", index=False
        )
        external_prediction_table = pd.DataFrame(external_prediction_rows)
        external_prediction_table.to_csv(
            output_dir / "external_validation_predictions.csv", index=False
        )
        build_auc_comparison_table(
            external_prediction_table,
            reference_dataset="Clinical",
        ).to_csv(
            output_dir / "auc_comparisons_external_vs_clinical.csv", index=False
        )
        if external_label_audit is not None:
            external_label_audit.to_csv(
                output_dir / "external_label_audit.csv", index=False
            )
    pd.DataFrame(selected_oof_rows).to_csv(
        output_dir / "oof_predictions_selected_models.csv", index=False
    )
    late_model_comparison_config_table = pd.DataFrame(
        late_model_comparison_config_rows
    )
    late_model_comparison_cv_table = pd.DataFrame(
        late_model_comparison_cv_rows
    )
    late_model_comparison_external_table = pd.DataFrame(
        late_model_comparison_external_rows
    )
    late_model_comparison_config_table.to_csv(
        output_dir / "late_fusion_model_comparison_config.csv", index=False
    )
    late_model_comparison_cv_table.to_csv(
        output_dir / "late_fusion_model_comparison_5fold_predictions.csv",
        index=False,
    )
    late_model_comparison_external_table.to_csv(
        output_dir / "late_fusion_model_comparison_external_predictions.csv",
        index=False,
    )
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as file:
        json.dump({
            "cv_folds": cv_folds,
            "optuna_trials_per_shortlisted_model": n_trials,
            "top_models_tuned_per_modality": top_models_to_tune,
            "top_meta_models_tuned": top_meta_models_to_tune,
            "sampler_candidates": ["none", "class_weight", "smote"],
            "sampler_selection_rule": (
                "highest CV AUC; prefer simpler sampler when within 0.005 AUC"
            ),
            "optuna_objective": "maximize mean validation-fold AUC",
            "final_model_ranking": (
                "mean CV AUC, then lower AUC SD, then lower Brier"
            ),
            "early_fusion_combinations": combinations,
            "radiomics_target_feature_count": [10, 15],
            "radiomics_regularization_objective": (
                "adaptive L1-logistic C by inner-CV AUC with cached preprocessing; "
                "LogLoss retained for audit"
            ),
            "radiomics_final_stability_rule": "selected in >=60% outer folds",
            "late_fusion_inputs": "aligned base-model OOF probabilities",
            "deployment_bundle": str(deployment_bundle_path),
            "stacking_cv_scope": (
                "cross-fitted development OOF; base configurations are selected on "
                "development CV, so final performance claims must use the "
                "untouched external cohort"
            ),
            "late_fusion_sets": late_fusion_run_config,
            "meta_model_candidates": list(meta_models_map),
            "custom_estimators": "none",
            "cohorts": {
                "development": (
                    "every aligned internal case; no internal hold-out is "
                    "carved out in this pipeline"
                ),
                "test": "the external cohort, scored once",
            },
            "selection_data": (
                "development folds only; the external cohort is untouched"
            ),
            "external_validation": (
                "the only test cohort; scored once and never used for model, "
                "sampler, hyperparameter, feature, or threshold selection"
            ),
            "external_validation_dir": str(external_dir),
            "publication_figures": {
                "scope": "full three-modality Late_Fusion only",
                "curves": "meta-model algorithms, not modality branches",
                "separate_files": [
                    "five-fold cross-validation",
                    "external test cohort",
                ],
                "legend_location": "lower right inside each axis",
                "cv_uncertainty_shading": False,
                "format": "LZW-compressed TIFF, 600 dpi",
                "supplementary_figures": ["calibration", "decision curve analysis"],
            },
            "shap_outputs": {
                "explained_class": "positive class (1 = subsidence)",
                "model_scope": [
                    "Late_Fusion|LightGBM",
                    "Late_Fusion|End_to_End",
                ],
                "plot_types": [
                    "summary",
                    "decision",
                    "waterfall",
                    "force",
                ],
                "global_top_features": SHAP_GLOBAL_TOP_FEATURES,
                "local_top_features": SHAP_LOCAL_TOP_FEATURES,
                "requested_high_low_pairs": SHAP_PAIR_COUNT,
                "high_probability_rule": f"> {SHAP_HIGH_PROBABILITY}",
                "low_probability_rule": f"< {SHAP_LOW_PROBABILITY}",
                "local_display_decimals": 2,
                "figure_format": "LZW-compressed TIFF, 600 dpi",
            },
        }, file, ensure_ascii=False, indent=2)

    # Post-training only: reuse the OOF and external probabilities already
    # produced above. No estimator is fitted and no test threshold is optimized.
    generate_publication_evaluation(
        output_dir=output_dir,
        selected_models=pd.DataFrame(selected_model_rows),
        cv_predictions=pd.DataFrame(selected_oof_rows),
        external_predictions=external_prediction_table,
        late_model_cv_predictions=late_model_comparison_cv_table,
        late_model_external_predictions=late_model_comparison_external_table,
    )

    print("\nFinal CV results (post tuning for shortlisted models):")
    print(pd.DataFrame(final_cv_rows)[[
        "Dataset", "Model", "Stage", "Sampler", "AUC", "LogLoss"
    ]])
    print("\nSelected models:")
    print(pd.DataFrame(selected_model_rows))
    print("\nExternal test results:")
    print(pd.DataFrame(external_summary_rows))

    shap_output_dir = output_dir / "shap"
    # Manuscript scope: positive-class explanations for the full late-fusion
    # model and its end-to-end original-feature view. The explained algorithm
    # is whichever one development CV actually selected: hard-coding LightGBM
    # here made a run whose winner was RF produce no SHAP output at all, and
    # say nothing about it. The external cohort is the only held-out data, so
    # it is what SHAP sees; development cases would be in-sample for the final
    # fit.
    full_late_winner = late_winner_names["Late_Fusion"]
    late_shap_key = f"Late_Fusion|{full_late_winner}"
    if late_shap_key not in final_results:
        raise RuntimeError(
            f"The selected Late_Fusion winner {full_late_winner} has no fitted "
            f"result under '{late_shap_key}'; SHAP cannot run."
        )
    if full_late_external_inputs is None or full_late_external_probability is None:
        raise RuntimeError(
            "External Late Fusion inputs are unavailable for SHAP."
        )
    print("")
    print(f"SHAP explains the selected Late_Fusion winner: {full_late_winner}")
    external_explain_size = len(full_late_external_inputs)
    generate_shap_plots(
        final_results,
        shap_output_dir,
        max_explain_cases=external_explain_size,
        model_keys=[late_shap_key],
        explanation_inputs={
            late_shap_key: full_late_external_inputs,
        },
        expected_probabilities={
            late_shap_key: full_late_external_probability,
        },
    )
    generate_late_fusion_end_to_end_shap(
        results_dict=final_results,
        base_winner_names=base_winner_names,
        late_winner_name=full_late_winner,
        modalities=late_fusion_sets["Late_Fusion"],
        holdout_feature_frames=base_external_feature_frames,
        holdout_meta_inputs=full_late_external_inputs,
        expected_probability=full_late_external_probability,
        output_dir=shap_output_dir,
        explain_size=external_explain_size,
    )
    print(f"\nDONE. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
