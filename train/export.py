"""Export the trained inference routes and their clinical input schema."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json

import joblib
import numpy as np
import pandas as pd

from .metrics import MLResults
from .preprocessing import CLINICAL_CATEGORICAL_BASE_COLUMNS


def _to_builtin_scalar(value):
    """Convert numpy/pandas scalar values into portable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_clinical_form_schema(clinical_result: MLResults) -> List[Dict[str, Any]]:
    """Describe exactly the selected Clinical inputs required by the web form."""
    schema = []
    for feature in clinical_result.selected_features:
        if not feature.startswith("Clinical___"):
            continue
        source_name = feature.split("___", 1)[1]
        series = clinical_result.X_train_final[feature].dropna()
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        categorical = source_name in CLINICAL_CATEGORICAL_BASE_COLUMNS
        integer_like = bool(
            len(numeric_series)
            and np.allclose(
                numeric_series.to_numpy(dtype=float),
                np.round(numeric_series.to_numpy(dtype=float)),
            )
        )
        field = {
            "name": feature,
            "source_name": source_name,
            "label": source_name.replace("_", " "),
            "input_type": "select" if categorical else "number",
            "value_type": "integer" if integer_like else "number",
            "required": True,
        }
        if categorical:
            field["choices"] = [
                _to_builtin_scalar(value)
                for value in sorted(series.unique().tolist())
            ]
        elif len(numeric_series):
            # These are informational training ranges, not hard validation limits.
            field["observed_min"] = float(numeric_series.min())
            field["observed_max"] = float(numeric_series.max())
            field["step"] = 1 if integer_like else "any"
        schema.append(field)
    return schema


def export_late_fusion_bundle(
        final_results: Dict[str, MLResults],
        base_winner_names: Dict[str, str],
        late_winner_names: Dict[str, str],
        output_dir: Path,
        deployment_dir: Path,
) -> Path:
    """Persist deployable inference routes that always include Clinical.

    Clinical-only uses its strongest base model. Each Clinical-containing pair
    uses whichever selected early- or late-fusion winner has the higher
    development CV AUC. All three inputs use the selected full Late_Fusion
    model. Muscle-only, Radiomics-only, and Muscle+Radiomics are intentionally
    unavailable for deployment.

    ``deployment_dir`` is where the bundle and route artifacts land. The main
    training workflow writes directly to the project's artifacts directory.
    """
    modalities = ["Clinical", "Muscle", "Radiomics"]
    missing_base_winners = [
        modality for modality in modalities if modality not in base_winner_names
    ]
    if missing_base_winners:
        raise RuntimeError(
            f"Cannot export deployment bundle; missing base winners: "
            f"{missing_base_winners}"
        )
    if "Late_Fusion" not in late_winner_names:
        raise RuntimeError(
            "Cannot export deployment bundle; full Late_Fusion winner is missing."
        )

    base_models = {}
    for modality in modalities:
        model_name = base_winner_names[modality]
        result_key = f"{modality}|{model_name}"
        if result_key not in final_results:
            raise RuntimeError(
                f"Cannot export deployment bundle; result {result_key!r} is missing."
            )
        result = final_results[result_key]
        base_models[modality] = {
            "model_name": model_name,
            "pipeline": result.fitted_model,
            "features": list(result.selected_features),
            "decision_threshold": float(result.decision_threshold),
        }

    full_late_model_name = late_winner_names["Late_Fusion"]
    late_result_key = f"Late_Fusion|{full_late_model_name}"
    if late_result_key not in final_results:
        raise RuntimeError(
            f"Cannot export deployment bundle; result {late_result_key!r} is missing."
        )
    late_result = final_results[late_result_key]
    expected_meta_features = [
        "Clinical_Probability",
        "Muscle_Probability",
        "Radiomics_Probability",
    ]
    if list(late_result.selected_features) != expected_meta_features:
        raise RuntimeError(
            "Full Late_Fusion deployment features are not the expected ordered "
            f"three probabilities: {late_result.selected_features}"
        )

    def route_payload(
            dataset_name: str,
            model_name: str,
            result: MLResults,
            route_modalities: Sequence[str],
            strategy: str,
    ) -> Dict[str, Any]:
        return {
            "dataset": dataset_name,
            "modalities": list(route_modalities),
            "strategy": strategy,
            "model_name": model_name,
            "pipeline": result.fitted_model,
            "features": list(result.selected_features),
            "decision_threshold": float(result.decision_threshold),
            "selection_cv_auc": float(
                result.selected_cv_evaluation.metrics_cv_mean["AUC"]
            ),
        }

    clinical_model_name = base_winner_names["Clinical"]
    clinical_result = final_results[f"Clinical|{clinical_model_name}"]
    routing_models = {
        "Clinical": route_payload(
            "Clinical",
            clinical_model_name,
            clinical_result,
            ["Clinical"],
            "direct",
        ),
    }

    pair_datasets = {
        ("Clinical", "Muscle"): (
            "Clinical_Muscle", "Late_Fusion_Clinical_Muscle"
        ),
        ("Clinical", "Radiomics"): (
            "Clinical_Radiomics", "Late_Fusion_Clinical_Radiomics"
        ),
    }
    for pair_modalities, (early_dataset, late_dataset) in pair_datasets.items():
        candidates = []
        early_keys = [
            key for key in final_results if key.startswith(f"{early_dataset}|")
        ]
        if len(early_keys) == 1:
            early_model_name = early_keys[0].split("|", 1)[1]
            early_result = final_results[early_keys[0]]
            candidates.append((
                early_result.selected_cv_evaluation.metrics_cv_mean["AUC"],
                early_dataset,
                early_model_name,
                early_result,
                "direct",
            ))
        late_model_name = late_winner_names.get(late_dataset)
        late_key = (
            f"{late_dataset}|{late_model_name}" if late_model_name else None
        )
        if late_key in final_results:
            pair_late_result = final_results[late_key]
            candidates.append((
                pair_late_result.selected_cv_evaluation.metrics_cv_mean["AUC"],
                late_dataset,
                late_model_name,
                pair_late_result,
                "late_fusion",
            ))
        if not candidates:
            raise RuntimeError(
                f"Cannot export deployment route for pair {pair_modalities}."
            )
        _, dataset_name, model_name, result, strategy = max(
            candidates, key=lambda candidate: candidate[0]
        )
        route_key = "+".join(pair_modalities)
        routing_models[route_key] = route_payload(
            dataset_name, model_name, result, pair_modalities, strategy
        )

    routing_models["Clinical+Muscle+Radiomics"] = route_payload(
        "Late_Fusion",
        full_late_model_name,
        late_result,
        modalities,
        "late_fusion",
    )

    bundle = {
        "bundle_type": "fusion_subsidence_full_late_fusion",
        "bundle_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": 1,
        "base_models": base_models,
        "late_fusion": {
            "model_name": full_late_model_name,
            "pipeline": late_result.fitted_model,
            "features": list(late_result.selected_features),
            "decision_threshold": float(late_result.decision_threshold),
        },
        "routing_models": routing_models,
        "clinical_form_schema": build_clinical_form_schema(
            final_results[f"Clinical|{base_winner_names['Clinical']}"]
        ),
        "metadata": {
            "experiment_output": str(output_dir),
            "training_script": "train/train.py",
            "inference_note": (
                "Clinical is required for every prediction. An optional Muscle "
                "and/or Radiomics input selects the strongest matching "
                "development-validated route; late-fusion routes consume "
                "ordered base probabilities."
            ),
        },
    }

    deployment_dir = Path(deployment_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = deployment_dir / "late_fusion_bundle.pkl"
    temporary_path = deployment_dir / "late_fusion_bundle.pkl.tmp"
    joblib.dump(bundle, temporary_path, compress=3)
    temporary_path.replace(bundle_path)

    route_file_names = {
        "Clinical": "clinical.pkl",
        "Clinical+Muscle": "clinical_muscle.pkl",
        "Clinical+Radiomics": "clinical_radiomics.pkl",
        "Clinical+Muscle+Radiomics": "late_fusion.pkl",
    }
    route_artifact_paths = {}
    for route_key, file_name in route_file_names.items():
        route = routing_models[route_key]
        route_artifact = {
            "bundle_type": "fusion_subsidence_route",
            "bundle_version": 1,
            "created_at_utc": bundle["created_at_utc"],
            "positive_class": 1,
            "route_key": route_key,
            "base_models": {
                modality: base_models[modality]
                for modality in route["modalities"]
            },
            "route_model": route,
            "clinical_form_schema": (
                bundle["clinical_form_schema"]
                if "Clinical" in route["modalities"] else []
            ),
        }
        route_path = deployment_dir / file_name
        route_temporary_path = deployment_dir / f"{file_name}.tmp"
        joblib.dump(route_artifact, route_temporary_path, compress=3)
        route_temporary_path.replace(route_path)
        route_artifact_paths[route_key] = route_path

    # Remove route artifacts that are no longer valid when rerunning into an
    # existing artifacts directory.
    for obsolete_file_name in (
            "muscle.pkl", "radiomics.pkl", "muscle_radiomics.pkl",
    ):
        (deployment_dir / obsolete_file_name).unlink(missing_ok=True)

    manifest = {
        "bundle_type": bundle["bundle_type"],
        "bundle_version": bundle["bundle_version"],
        "created_at_utc": bundle["created_at_utc"],
        "artifact_path": str(bundle_path),
        "positive_class": bundle["positive_class"],
        "base_models": {
            modality: {
                "model_name": payload["model_name"],
                "features": payload["features"],
                "decision_threshold": payload["decision_threshold"],
            }
            for modality, payload in base_models.items()
        },
        "late_fusion": {
            "model_name": full_late_model_name,
            "features": late_result.selected_features,
            "decision_threshold": late_result.decision_threshold,
        },
        "routing_models": {
            route_key: {
                key: value
                for key, value in payload.items()
                if key != "pipeline"
            }
            for route_key, payload in routing_models.items()
        },
        "route_artifacts": {
            route_key: str(path)
            for route_key, path in route_artifact_paths.items()
        },
    }
    with open(
            deployment_dir / "late_fusion_bundle_manifest.json",
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"  Adaptive deployment bundle saved to: {bundle_path}")
    print(
        "  Route artifacts saved: "
        + ", ".join(path.name for path in route_artifact_paths.values())
    )
    return bundle_path
