"""Validate saved predictions and calculate publication tables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import math

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import numpy as np
import pandas as pd


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
