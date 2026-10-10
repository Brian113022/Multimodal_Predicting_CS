"""Train and externally validate the single-domain and early-fusion models."""

import json

import pandas as pd

from .cross_validation import (
    choose_sampler_candidate,
    evaluate_model_cv,
    evaluation_rank_key,
    fit_final_model,
    prepare_cv_folds,
    tune_model_candidate,
)
from .data import DataFusionPipeline, load_external_validation_dataset
from .feature_selection import SharedRadiomicsSelections
from .metrics import compute_metrics, cv_summary_row, metric_summary_row, safe_file_name
from .preprocessing import get_sampler_choices
from .settings import TrainingConfig
from .state import TrainingState


def train_early_fusion(
    fusion: DataFusionPipeline, shared_radiomics_selections: SharedRadiomicsSelections,
    models_map: dict, combinations: dict, config: TrainingConfig, state: TrainingState,
):
    output_dir = config.output_dir
    external_dir = config.data_dir / "external_validation_WF"
    run_external_validation = True
    cv_folds = config.cv_folds
    n_trials = config.n_trials
    top_models_to_tune = config.top_models_to_tune
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
                state.default_cv_rows.append(cv_summary_row(
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
            state.screening_cv_rows.append(cv_summary_row(
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
            state.tuned_cv_rows.append(cv_summary_row(
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
                state.final_cv_rows.append(cv_summary_row(
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
                state.final_cv_rows.append(cv_summary_row(
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
        state.final_results[f"{combo_name}|{winner_name}"] = result
        oof = result.selected_cv_evaluation.oof_predictions.copy()
        if combo_name in {"Clinical", "Muscle", "Radiomics"}:
            state.base_winner_names[combo_name] = winner_name
            state.base_oof_probability_frames[combo_name] = oof[[
                "Fold", "Case_ID", "True_Label", "Probability_Subsidence"
            ]].copy()
        oof_metrics = compute_metrics(
            oof["True_Label"],
            oof["Probability_Subsidence"],
            threshold=result.decision_threshold,
        )
        state.selected_model_rows.append({
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
        state.selected_oof_rows.extend(oof.to_dict("records"))

        if run_external_validation:
            X_external, y_external, _, aligned_audit = (
                load_external_validation_dataset(
                    external_dir,
                    fusion,
                    source_keys,
                )
            )
            if state.external_label_audit is None:
                state.external_label_audit = aligned_audit.copy()
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
                state.base_external_feature_frames[combo_name] = external_features
                state.base_external_probability_frames[combo_name] = pd.DataFrame({
                    "Case_ID": external_case_ids,
                    "True_Label": y_external,
                    "Probability_Subsidence": external_probability,
                })
            external_metrics = compute_metrics(
                y_external,
                external_probability,
                threshold=result.decision_threshold,
            )
            state.external_summary_rows.append(metric_summary_row(
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
                state.external_prediction_rows.append({
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
