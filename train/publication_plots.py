"""Draw ROC, calibration, decision and confusion-matrix figures."""

from pathlib import Path
from typing import Any, Dict, Sequence
import math

from matplotlib.colors import Normalize
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .publication_metrics import (
    PUBLICATION_META_MODEL_COLORS,
    PUBLICATION_META_MODEL_LINE_STYLES,
    PUBLICATION_META_MODEL_MARKERS,
    PUBLICATION_MODEL_ORDER,
    publication_available_meta_models,
    publication_bootstrap_auc_ci,
    publication_metrics,
    publication_model_name,
)


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
