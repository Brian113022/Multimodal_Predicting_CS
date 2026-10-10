"""Export inference models, run metadata and evaluation tables and figures."""

import json

import pandas as pd

from .export import export_late_fusion_bundle
from .metrics import build_auc_comparison_table
from .publication import generate_publication_evaluation
from .settings import TrainingConfig
from .shap_utils import (
    SHAP_GLOBAL_TOP_FEATURES,
    SHAP_HIGH_PROBABILITY,
    SHAP_LOCAL_TOP_FEATURES,
    SHAP_LOW_PROBABILITY,
    SHAP_PAIR_COUNT,
)
from .state import TrainingState


def write_training_outputs(
    combinations: dict, meta_models_map: dict,
    config: TrainingConfig, state: TrainingState,
):
    output_dir = config.output_dir
    external_dir = config.data_dir / "external_validation_WF"
    run_external_validation = True
    cv_folds = config.cv_folds
    n_trials = config.n_trials
    top_models_to_tune = config.top_models_to_tune
    top_meta_models_to_tune = config.top_meta_models_to_tune
    deployment_bundle_path = export_late_fusion_bundle(
        final_results=state.final_results,
        base_winner_names=state.base_winner_names,
        late_winner_names=state.late_winner_names,
        output_dir=output_dir,
        deployment_dir=config.artifacts_dir,
    )

    pd.DataFrame(state.default_cv_rows).to_csv(
        output_dir / "cv_results_default.csv", index=False
    )
    pd.DataFrame(state.screening_cv_rows).to_csv(
        output_dir / "cv_results_screening.csv", index=False
    )
    pd.DataFrame(state.tuned_cv_rows).to_csv(
        output_dir / "cv_results_tuned.csv", index=False
    )
    pd.DataFrame(state.final_cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(state.selected_model_rows).to_csv(output_dir / "selected_models.csv", index=False)
    if run_external_validation:
        external_summary_table = pd.DataFrame(state.external_summary_rows)
        external_summary_table.to_csv(
            output_dir / "external_validation_results.csv", index=False
        )
        external_prediction_table = pd.DataFrame(state.external_prediction_rows)
        external_prediction_table.to_csv(
            output_dir / "external_validation_predictions.csv", index=False
        )
        build_auc_comparison_table(
            external_prediction_table,
            reference_dataset="Clinical",
            n_bootstrap=config.bootstrap_iterations,
        ).to_csv(
            output_dir / "auc_comparisons_external_vs_clinical.csv", index=False
        )
        if state.external_label_audit is not None:
            state.external_label_audit.to_csv(
                output_dir / "external_label_audit.csv", index=False
            )
    pd.DataFrame(state.selected_oof_rows).to_csv(
        output_dir / "oof_predictions_selected_models.csv", index=False
    )
    late_model_comparison_config_table = pd.DataFrame(
        state.late_model_comparison_config_rows
    )
    late_model_comparison_cv_table = pd.DataFrame(
        state.late_model_comparison_cv_rows
    )
    late_model_comparison_external_table = pd.DataFrame(
        state.late_model_comparison_external_rows
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
            "run_mode": "smoke_test" if config.smoke_test else "full",
            "data_dir": str(config.data_dir),
            "output_dir": str(output_dir),
            "artifacts_dir": str(config.artifacts_dir),
            "bootstrap_iterations": config.bootstrap_iterations,
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
            "late_fusion_sets": state.late_fusion_run_config,
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
                    f"Late_Fusion|{state.late_winner_names['Late_Fusion']}",
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
                "max_explain_cases": config.shap_max_cases,
                "background_size": config.shap_background_size,
                "end_to_end_nsamples": config.shap_nsamples,
            },
        }, file, ensure_ascii=False, indent=2)

    # Post-training only: reuse the OOF and external probabilities already
    # produced above. No estimator is fitted and no test threshold is optimized.
    generate_publication_evaluation(
        output_dir=output_dir,
        selected_models=pd.DataFrame(state.selected_model_rows),
        cv_predictions=pd.DataFrame(state.selected_oof_rows),
        external_predictions=external_prediction_table,
        late_model_cv_predictions=late_model_comparison_cv_table,
        late_model_external_predictions=late_model_comparison_external_table,
        bootstrap_iterations=config.bootstrap_iterations,
    )

    print("\nFinal CV results (post tuning for shortlisted models):")
    print(pd.DataFrame(state.final_cv_rows)[[
        "Dataset", "Model", "Stage", "Sampler", "AUC", "LogLoss"
    ]])
    print("\nSelected models:")
    print(pd.DataFrame(state.selected_model_rows))
    print("\nExternal test results:")
    print(pd.DataFrame(state.external_summary_rows))

