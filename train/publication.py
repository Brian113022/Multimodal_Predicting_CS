"""Assemble publication tables, figures and the evaluation summary."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .publication_metrics import (
    PUBLICATION_DISPLAY_NAMES,
    PUBLICATION_MODEL_ORDER,
    publication_available_meta_models,
    publication_cv_table,
    publication_late_model_auc_table,
    publication_output_directories,
    publication_test_table,
    publication_validate_late_model_predictions,
    publication_validate_predictions,
)
from .publication_plots import (
    publication_plot_confusion_matrices,
    publication_plot_late_model_calibration,
    publication_plot_late_model_dca,
    publication_plot_late_model_roc,
)


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
