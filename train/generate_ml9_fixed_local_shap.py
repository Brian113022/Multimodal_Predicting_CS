"""Generate fixed-case ML9 SHAP Force and Waterfall figures only.

The completed run supplies the selected model configuration, final feature
lists, OOF meta inputs, internal-test meta inputs and audited predictions. This
script does not run CV, Optuna or model selection. It reconstructs the already
selected final pipelines, requires their probabilities to match the saved
internal-test CSV, and explains five fixed mixed-evidence patient pairs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML9_SCRIPT = Path(__file__).resolve().with_name("2026-08-17_ml9.py")
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "2026-08-17_ml9_auc_first_all_meta_external_wf"
)
MODALITIES = ("Clinical", "Muscle", "Radiomics")
META_FEATURES = [f"{modality}_Probability" for modality in MODALITIES]
VISIBLE_SHAP_THRESHOLD = 0.005

# Fixed after auditing all 72 internal-test patients. Each case has at least
# one positive and one negative contribution that remains visible at 2 decimals
# in both the direct LightGBM and end-to-end explanations.
FIXED_PAIRS = (
    (1, "65_18550740", "34_15910916"),
    (2, "54_01545780", "33_06597647"),
    (3, "15_19211295", "63_19920641"),
    (4, "80_16068933", "6_17867728"),
    (5, "46_13277695", "3_09776863"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Completed ML9 experiment output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "SHAP output root. Defaults to <run-dir>/shap; local figures "
            "are written under its local folder."
        ),
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=20,
        help="Number of development patients used as SHAP background.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=500,
        help="Kernel SHAP samples for the end-to-end explanation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ml9_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ml9_fixed_local_shap", ML9_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ML9 module from {ML9_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_files(run_dir: Path, names: list[str]) -> None:
    missing = [name for name in names if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Completed run is missing required files: {missing}")


def clean_previous_local_outputs(output_dir: Path) -> None:
    """Remove only local Late Fusion files that this script regenerates."""
    local_dir = output_dir / "local"
    if not local_dir.is_dir():
        return
    patterns = (
        "shap_force_top6_Late_Fusion_*.tiff",
        "shap_force_top8_Late_Fusion_*.tiff",
        "shap_waterfall_top6_Late_Fusion_*.tiff",
        "shap_waterfall_top8_Late_Fusion_*.tiff",
        "shap_local_case_manifest_Late_Fusion_*.csv",
        "shap_local_top6_contributions_Late_Fusion_*.csv",
        "shap_local_top8_contributions_Late_Fusion_*.csv",
    )
    for pattern in patterns:
        for path in local_dir.glob(pattern):
            path.unlink()


def selected_configuration(run_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(run_dir / "selected_models.csv").set_index("Dataset")
    required = set(MODALITIES) | {"Late_Fusion"}
    missing = required - set(table.index)
    if missing:
        raise ValueError(f"selected_models.csv is missing: {sorted(missing)}")
    if table.loc["Late_Fusion", "Best_Model"] != "LightGBM":
        raise ValueError("The completed Late_Fusion winner is not LightGBM.")
    return table


def read_final_features(run_dir: Path) -> dict[str, list[str]]:
    features = {}
    for modality in MODALITIES:
        table = pd.read_csv(run_dir / f"final_features_{modality}.csv")
        values = table["Feature"].astype(str).tolist()
        if not values:
            raise ValueError(f"No final features were saved for {modality}.")
        features[modality] = values
    return features


def rebuild_final_models(ml9: ModuleType, run_dir: Path) -> dict[str, Any]:
    selected = selected_configuration(run_dir)
    features = read_final_features(run_dir)
    fusion = ml9.DataFusionPipeline({
        "Radiomics": str(ml9.DATA_DIR / "radiomics_features_new.csv"),
        "Clinical": str(ml9.DATA_DIR / "clinical_features_new.csv"),
        "Muscle": str(ml9.DATA_DIR / "CSA_muscles_features_new.csv"),
    })
    fusion.load_and_align()

    all_positions = np.arange(len(fusion.common_index))
    all_labels = fusion.label_col.to_numpy(dtype=int)
    development_positions, test_positions = train_test_split(
        all_positions,
        test_size=0.20,
        stratify=all_labels,
        random_state=42,
    )
    expected_split = pd.read_csv(run_dir / "internal_split_audit.csv")
    rebuilt_split = pd.DataFrame({
        "Case_ID": np.asarray(fusion.common_index).astype(str),
        "Split": np.where(
            np.isin(all_positions, test_positions),
            "Internal_Test",
            "Development",
        ),
    })
    if not rebuilt_split.equals(expected_split[["Case_ID", "Split"]]):
        raise ValueError(
            "Current processed data do not reproduce internal_split_audit.csv."
        )

    base_models = {}
    development_frames = {}
    internal_frames = {}
    rebuilt_meta_inputs = {}
    models_map = ml9.get_models_map()
    for modality in MODALITIES:
        row = selected.loc[modality]
        model_name = str(row["Best_Model"])
        merged = fusion.get_merged_dataset([modality])
        X_all = merged.drop(columns=["Label"])
        y_all = merged["Label"].to_numpy(dtype=int)
        X_development = X_all.iloc[development_positions].loc[
            :, features[modality]
        ].copy()
        X_internal = X_all.iloc[test_positions].loc[:, features[modality]].copy()
        model = ml9.build_training_pipeline(
            models_map[model_name][0],
            features[modality],
            sampler_choice=str(row["Sampler"]),
            y_train=y_all[development_positions],
        )
        model.set_params(**json.loads(str(row["Best_Params"])))
        model.fit(X_development, y_all[development_positions])
        base_models[modality] = model
        development_frames[modality] = X_development
        internal_frames[modality] = X_internal
        rebuilt_meta_inputs[f"{modality}_Probability"] = (
            model.predict_proba(X_internal)[:, 1]
        )

    audited_meta_inputs = pd.read_csv(
        run_dir / "Late_Fusion_internal_test_inputs.csv"
    ).set_index("Case_ID").sort_index()
    rebuilt_meta_inputs = pd.DataFrame(
        rebuilt_meta_inputs,
        index=internal_frames["Clinical"].index.astype(str),
    ).sort_index()
    if not np.allclose(
        rebuilt_meta_inputs[META_FEATURES],
        audited_meta_inputs[META_FEATURES],
        rtol=1e-10,
        atol=1e-12,
    ):
        difference = np.max(np.abs(
            rebuilt_meta_inputs[META_FEATURES].to_numpy()
            - audited_meta_inputs[META_FEATURES].to_numpy()
        ))
        raise ValueError(
            "Rebuilt base probabilities differ from saved internal inputs; "
            f"max difference={difference:.3e}."
        )

    oof_inputs = pd.read_csv(
        run_dir / "Late_Fusion_oof_inputs.csv"
    ).set_index("Case_ID").sort_index()
    late_row = selected.loc["Late_Fusion"]
    y_meta = oof_inputs["True_Label"].to_numpy(dtype=int)
    late_model = ml9.build_training_pipeline(
        ml9.get_meta_models_map()["LightGBM"][0],
        META_FEATURES,
        sampler_choice=str(late_row["Sampler"]),
        y_train=y_meta,
    )
    late_model.set_params(**json.loads(str(late_row["Best_Params"])))
    late_model.fit(oof_inputs[META_FEATURES], y_meta)
    probability = late_model.predict_proba(
        audited_meta_inputs[META_FEATURES]
    )[:, 1]

    audited_predictions = pd.read_csv(
        run_dir / "late_fusion_model_comparison_internal_predictions.csv"
    )
    audited_predictions = audited_predictions.loc[
        (audited_predictions["Dataset"] == "Late_Fusion")
        & (audited_predictions["Model"] == "LightGBM")
    ].copy()
    audited_predictions["Case_ID"] = audited_predictions["Case_ID"].astype(str)
    audited_predictions = audited_predictions.set_index("Case_ID").loc[
        audited_meta_inputs.index
    ]
    audited_probability = audited_predictions[
        "Probability_Subsidence"
    ].to_numpy(dtype=float)
    if not np.allclose(
        probability,
        audited_probability,
        rtol=1e-10,
        atol=1e-12,
    ):
        difference = np.max(np.abs(probability - audited_probability))
        raise ValueError(
            "Rebuilt LightGBM probabilities differ from saved predictions; "
            f"max difference={difference:.3e}."
        )

    return {
        "features": features,
        "base_models": base_models,
        "development_frames": development_frames,
        "internal_frames": internal_frames,
        "audited_meta_inputs": audited_meta_inputs,
        "oof_inputs": oof_inputs,
        "late_model": late_model,
        "probability": pd.Series(
            probability,
            index=audited_meta_inputs.index,
            name="Probability_Subsidence",
        ),
    }


def fixed_case_order(
    probability: pd.Series,
) -> tuple[list[str], list[tuple[int, int, int]]]:
    case_ids = []
    pair_rows = []
    for pair_number, high_case, low_case in FIXED_PAIRS:
        for case_id in (high_case, low_case):
            if case_id not in probability.index:
                raise ValueError(f"Fixed SHAP case is missing: {case_id}")
            case_ids.append(case_id)
        high_probability = float(probability.loc[high_case])
        low_probability = float(probability.loc[low_case])
        if high_probability <= 0.85 or low_probability >= 0.35:
            raise ValueError(
                f"Fixed pair {pair_number} no longer meets probability rules: "
                f"high={high_probability:.4f}, low={low_probability:.4f}."
            )
        pair_rows.append((pair_number, len(case_ids) - 2, len(case_ids) - 1))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("A fixed SHAP Case_ID appears in more than one pair.")
    return case_ids, pair_rows


def require_visible_two_signs(
    explanation: Any,
    case_ids: list[str],
    label: str,
) -> None:
    values = np.asarray(explanation.values, dtype=float)
    failures = []
    for row_index, case_id in enumerate(case_ids):
        has_positive = bool(np.any(values[row_index] > VISIBLE_SHAP_THRESHOLD))
        has_negative = bool(np.any(values[row_index] < -VISIBLE_SHAP_THRESHOLD))
        if not (has_positive and has_negative):
            failures.append(case_id)
    if failures:
        raise ValueError(
            f"{label} no longer has visible positive and negative SHAP "
            f"contributions for: {failures}"
        )


def direct_explanation(
    ml9: ModuleType,
    rebuilt: dict[str, Any],
    case_ids: list[str],
    background_size: int,
):
    late_model = rebuilt["late_model"]
    X_explain = rebuilt["audited_meta_inputs"].loc[
        case_ids, META_FEATURES
    ].copy()
    preprocessor = late_model.named_steps["pre"]
    X_encoded = preprocessor.transform(X_explain)
    background = rebuilt["oof_inputs"][META_FEATURES].sample(
        n=min(background_size, len(rebuilt["oof_inputs"])),
        random_state=42,
    )
    explainer = shap.TreeExplainer(
        late_model.named_steps["lgbm"],
        data=preprocessor.transform(background),
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_values = explainer.shap_values(X_encoded)
    explanation = ml9.shap_positive_class_explanation(
        explainer, shap_values, X_explain
    )
    probability = rebuilt["probability"].loc[case_ids].to_numpy(dtype=float)
    ml9.validate_shap_output_probability(
        explanation, probability, "Late_Fusion|LightGBM"
    )
    require_visible_two_signs(explanation, case_ids, "Direct LightGBM SHAP")
    return explanation, X_explain, probability


def end_to_end_explanation(
    ml9: ModuleType,
    rebuilt: dict[str, Any],
    case_ids: list[str],
    background_size: int,
    nsamples: int,
    seed: int,
):
    features = rebuilt["features"]
    base_models = rebuilt["base_models"]
    development_source = pd.concat([
        rebuilt["development_frames"][modality].loc[:, features[modality]]
        for modality in MODALITIES
    ], axis=1)
    internal_parts = []
    for modality in MODALITIES:
        frame = rebuilt["internal_frames"][modality].copy()
        frame.index = frame.index.astype(str)
        internal_parts.append(frame.loc[case_ids, features[modality]])
    X_explain = pd.concat(internal_parts, axis=1)
    original_columns = development_source.columns.tolist()

    def build_meta_inputs(X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = original_columns
        else:
            frame = pd.DataFrame(np.asarray(X), columns=original_columns)
        return pd.DataFrame({
            f"{modality}_Probability": base_models[modality].predict_proba(
                frame.loc[:, features[modality]]
            )[:, 1]
            for modality in MODALITIES
        }, index=frame.index)

    def predict_end_to_end(X):
        return rebuilt["late_model"].predict_proba(
            build_meta_inputs(X)[META_FEATURES]
        )[:, 1]

    recomputed_meta = build_meta_inputs(X_explain)[META_FEATURES]
    audited_meta = rebuilt["audited_meta_inputs"].loc[case_ids, META_FEATURES]
    if not np.allclose(recomputed_meta, audited_meta, rtol=1e-10, atol=1e-12):
        raise ValueError("Fixed-case end-to-end base probabilities do not match.")

    background = development_source.sample(
        n=min(background_size, len(development_source)),
        random_state=seed,
    )
    explainer = shap.KernelExplainer(predict_end_to_end, background)
    case_values = []
    case_base_values = []
    for row_index in range(len(X_explain)):
        # Reset for each patient so its result cannot depend on batch order.
        np.random.seed(seed)
        row_frame = X_explain.iloc[[row_index]]
        row_shap_values = explainer.shap_values(
            row_frame,
            nsamples=nsamples,
            silent=True,
        )
        row_explanation = ml9.shap_positive_class_explanation(
            explainer, row_shap_values, row_frame
        )
        case_values.append(np.asarray(row_explanation.values[0], dtype=float))
        case_base_values.append(
            float(np.asarray(row_explanation.base_values).ravel()[0])
        )
    explanation = shap.Explanation(
        values=np.vstack(case_values),
        base_values=np.asarray(case_base_values, dtype=float),
        data=X_explain.to_numpy(),
        feature_names=X_explain.columns.tolist(),
    )
    probability = rebuilt["probability"].loc[case_ids].to_numpy(dtype=float)
    ml9.validate_shap_output_probability(
        explanation,
        probability,
        "Late_Fusion|End_to_End",
        tolerance=1e-4,
    )
    require_visible_two_signs(explanation, case_ids, "End-to-end SHAP")
    return explanation, X_explain, probability


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "shap"
    )
    require_files(run_dir, [
        "selected_models.csv",
        "final_features_Clinical.csv",
        "final_features_Muscle.csv",
        "final_features_Radiomics.csv",
        "internal_split_audit.csv",
        "Late_Fusion_internal_test_inputs.csv",
        "Late_Fusion_oof_inputs.csv",
        "late_fusion_model_comparison_internal_predictions.csv",
    ])
    ml9 = load_ml9_module()
    rebuilt = rebuild_final_models(ml9, run_dir)
    case_ids, pair_rows = fixed_case_order(rebuilt["probability"])
    clean_previous_local_outputs(output_dir)

    print("Generating fixed-case direct LightGBM local SHAP...")
    direct, X_direct, probability = direct_explanation(
        ml9, rebuilt, case_ids, args.background_size
    )
    ml9.save_positive_class_shap_outputs(
        "Late_Fusion|LightGBM",
        direct,
        X_direct,
        probability,
        pair_rows,
        output_dir,
        save_global_plots=False,
        save_local_plots=True,
        high_eligibility_rule="fixed mixed-evidence case; probability > 0.85",
        low_eligibility_rule="fixed mixed-evidence case; probability < 0.35",
    )

    print("Generating fixed-case end-to-end local SHAP...")
    end_to_end, X_end_to_end, probability = end_to_end_explanation(
        ml9,
        rebuilt,
        case_ids,
        args.background_size,
        args.nsamples,
        args.seed,
    )
    ml9.save_positive_class_shap_outputs(
        "Late_Fusion|End_to_End",
        end_to_end,
        X_end_to_end,
        probability,
        pair_rows,
        output_dir,
        save_global_plots=False,
        save_local_plots=True,
        high_eligibility_rule="fixed mixed-evidence case; probability > 0.85",
        low_eligibility_rule="fixed mixed-evidence case; probability < 0.35",
    )
    print(f"DONE. Fixed local SHAP outputs: {output_dir / 'local'}")


if __name__ == "__main__":
    main()
