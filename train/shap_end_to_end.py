"""Explain held-out late-fusion predictions through the full model graph."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import MLResults
from .optional_dependencies import SHAP_AVAILABLE, shap
from .shap_plots import save_positive_class_shap_outputs
from .shap_utils import (
    SHAP_HIGH_PROBABILITY,
    SHAP_LOW_PROBABILITY,
    SHAP_PAIR_COUNT,
    ShapInputMismatchError,
    select_shap_explanation_rows,
    shap_positive_class_explanation,
    validate_shap_output_probability,
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
