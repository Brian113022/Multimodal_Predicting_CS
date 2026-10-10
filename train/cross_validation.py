"""Prepare folds, screen and tune candidates, and fit final models."""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
import numpy as np
import pandas as pd

from .feature_selection import (
    RadiomicsSelection,
    SharedRadiomicsSelections,
    analyze_feature_origin,
    select_features_for_training_split,
)
from .metrics import (
    CVEvaluation,
    MLResults,
    PreparedFold,
    choose_oof_threshold,
    compute_metrics,
    safe_file_name,
)
from .optional_dependencies import OPTUNA_AVAILABLE, optuna
from .preprocessing import SAMPLER_PREFERENCE, build_training_pipeline


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
