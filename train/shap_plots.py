"""Generate direct model SHAP explanations and save their figures."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import shap_publication_labels as shap_labels
from .metrics import MLResults, safe_file_name
from .optional_dependencies import SHAP_AVAILABLE, shap
from .shap_utils import (
    SHAP_FIGURE_DPI,
    SHAP_FORCE_FIGURE_SIZE,
    SHAP_GLOBAL_TOP_FEATURES,
    SHAP_HIGH_PROBABILITY,
    SHAP_LOCAL_TOP_FEATURES,
    SHAP_LOW_PROBABILITY,
    SHAP_PAIR_COUNT,
    ShapInputMismatchError,
    format_force_plot_axis_two_decimals,
    format_shap_x_axes_two_decimals,
    format_waterfall_axes_two_decimals,
    format_waterfall_bar_labels_two_decimals,
    select_shap_explanation_rows,
    shap_local_top_feature_explanation,
    shap_plot_explanation_with_two_decimals,
    shap_positive_class_explanation,
    validate_shap_output_probability,
)


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
            else:
                # LinearExplainer explains LR's log-odds. All saved SHAP
                # outputs are audited against predict_proba, so explain the
                # positive-class probability directly for non-tree models.
                background_n = min(background_size, len(result.X_train_final))
                X_background = result.X_train_final.sample(
                    n=background_n, random_state=42
                )
                background = preprocessor.transform(X_background)
                explainer = shap.KernelExplainer(
                    lambda values: estimator.predict_proba(values)[:, 1], background
                )
                shap_values = explainer.shap_values(
                    X_encoded, nsamples=100, l1_reg=0, silent=True
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
