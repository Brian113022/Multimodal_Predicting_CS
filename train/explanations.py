"""Generate direct and end-to-end SHAP outputs for the selected late-fusion model."""

from .settings import TrainingConfig
from .shap_end_to_end import generate_late_fusion_end_to_end_shap
from .shap_plots import generate_shap_plots
from .state import TrainingState


def write_training_explanations(late_fusion_sets: dict, config: TrainingConfig, state: TrainingState):
    output_dir = config.output_dir
    shap_output_dir = output_dir / "shap"
    # Manuscript scope: positive-class explanations for the full late-fusion
    # model and its end-to-end original-feature view. The explained algorithm
    # is whichever one development CV actually selected: hard-coding LightGBM
    # here made a run whose winner was RF produce no SHAP output at all, and
    # say nothing about it. The external cohort is the only held-out data, so
    # it is what SHAP sees; development cases would be in-sample for the final
    # fit.
    full_late_winner = state.late_winner_names["Late_Fusion"]
    late_shap_key = f"Late_Fusion|{full_late_winner}"
    if late_shap_key not in state.final_results:
        raise RuntimeError(
            f"The selected Late_Fusion winner {full_late_winner} has no fitted "
            f"result under '{late_shap_key}'; SHAP cannot run."
        )
    if state.full_late_external_inputs is None or state.full_late_external_probability is None:
        raise RuntimeError(
            "External Late Fusion inputs are unavailable for SHAP."
        )
    print("")
    print(f"SHAP explains the selected Late_Fusion winner: {full_late_winner}")
    external_explain_size = len(state.full_late_external_inputs)
    if config.shap_max_cases is not None:
        external_explain_size = min(external_explain_size, config.shap_max_cases)
    generate_shap_plots(
        state.final_results,
        shap_output_dir,
        max_explain_cases=external_explain_size,
        model_keys=[late_shap_key],
        explanation_inputs={
            late_shap_key: state.full_late_external_inputs,
        },
        expected_probabilities={
            late_shap_key: state.full_late_external_probability,
        },
        background_size=config.shap_background_size,
    )
    generate_late_fusion_end_to_end_shap(
        results_dict=state.final_results,
        base_winner_names=state.base_winner_names,
        late_winner_name=full_late_winner,
        modalities=late_fusion_sets["Late_Fusion"],
        holdout_feature_frames=state.base_external_feature_frames,
        holdout_meta_inputs=state.full_late_external_inputs,
        expected_probability=state.full_late_external_probability,
        output_dir=shap_output_dir,
        explain_size=external_explain_size,
        background_size=config.shap_background_size,
        nsamples=config.shap_nsamples,
    )
    print(f"\nDONE. Outputs saved to: {output_dir}")
