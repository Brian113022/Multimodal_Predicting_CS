"""Train and externally validate pairwise and full late-fusion models."""

import json

import numpy as np
import pandas as pd

from .cross_validation import (
    build_late_fusion_table,
    choose_sampler_candidate,
    evaluate_model_cv,
    evaluation_rank_key,
    fit_final_model,
    prepare_late_fusion_folds,
    tune_model_candidate,
)
from .metrics import (
    choose_oof_threshold,
    compute_metrics,
    cv_summary_row,
    metric_summary_row,
    safe_file_name,
)
from .preprocessing import build_training_pipeline, get_sampler_choices
from .settings import TrainingConfig
from .state import TrainingState


def train_late_fusion(
    meta_models_map: dict, late_fusion_sets: dict,
    config: TrainingConfig, state: TrainingState,
):
    output_dir = config.output_dir
    run_external_validation = True
    n_trials = config.n_trials
    top_meta_models_to_tune = config.top_meta_models_to_tune
    for late_dataset_name, late_modalities in late_fusion_sets.items():
        print(f"\n{'=' * 70}\nDataset: {late_dataset_name} {late_modalities}")
        safe_late_dataset = safe_file_name(late_dataset_name)
        late_oof_table = build_late_fusion_table(
            state.base_oof_probability_frames,
            late_modalities,
            require_fold=True,
        )
        late_external_table = None
        if run_external_validation:
            late_external_table = build_late_fusion_table(
                state.base_external_probability_frames,
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
                state.default_cv_rows.append(cv_summary_row(
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
            state.screening_cv_rows.append(cv_summary_row(
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
            state.tuned_cv_rows.append(cv_summary_row(
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
                state.final_cv_rows.append(cv_summary_row(
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
                state.final_cv_rows.append(cv_summary_row(
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
        state.final_results[f"{late_dataset_name}|{late_winner_name}"] = late_result
        state.late_winner_names[late_dataset_name] = late_winner_name

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
                "Base_Model": state.base_winner_names[
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
        state.selected_model_rows.append({
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
        state.selected_oof_rows.extend(late_oof.to_dict("records"))

        if late_external_table is not None:
            X_late_external = late_external_table[late_feature_columns].copy()
            y_late_external = late_external_table["True_Label"].to_numpy(dtype=int)
            late_external_probability = late_result.fitted_model.predict_proba(
                X_late_external
            )[:, 1]
            if late_dataset_name == "Late_Fusion":
                state.full_late_external_inputs = X_late_external.copy()
                state.full_late_external_probability = pd.Series(
                    late_external_probability,
                    index=X_late_external.index.astype(str),
                    name="Probability_Subsidence",
                )
            late_external_metrics = compute_metrics(
                y_late_external,
                late_external_probability,
                threshold=late_result.decision_threshold,
            )
            state.external_summary_rows.append(metric_summary_row(
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
                state.external_prediction_rows.append({
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
                state.late_model_comparison_cv_rows.extend(
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
                        state.late_model_comparison_external_rows.append({
                            "Dataset": "Late_Fusion",
                            "Model": model_name,
                            "Case_ID": str(case_id),
                            "True_Label": int(true_label),
                            "Probability_Subsidence": float(probability),
                            "Decision_Threshold_From_OOF": comparison_threshold,
                        })

                state.late_model_comparison_config_rows.append({
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

        state.late_fusion_run_config[late_dataset_name] = {
            "base_modalities": late_modalities,
            "selected_meta_model": late_winner_name,
            "selected_sampler": late_result.sampler_choice,
            "base_winner_models": {
                modality: state.base_winner_names[modality]
                for modality in late_modalities
            },
        }
