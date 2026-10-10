"""Cross-validation result types, metrics, thresholds and result rows."""

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import re

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
import numpy as np
import pandas as pd

from .feature_selection import RadiomicsSelection


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
