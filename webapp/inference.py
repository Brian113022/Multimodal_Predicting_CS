"""Load and execute the persisted full Late-Fusion inference graph."""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .clinical_schema import describe_clinical_fields


# The trained artifact references reusable preprocessing classes from
# fusion_subsidence/. PROJECT_ROOT is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import before joblib.load so pickle can resolve the stable module path.
from fusion_subsidence.model_components import SelectiveStandardScaler  # noqa: F401,E402


MODALITIES = ("Clinical", "Muscle", "Radiomics")
EXPECTED_META_FEATURES = (
    "Clinical_Probability",
    "Muscle_Probability",
    "Radiomics_Probability",
)
CLINICAL_REQUIRED_ROUTES = {
    "Clinical",
    "Clinical+Muscle",
    "Clinical+Radiomics",
    "Clinical+Muscle+Radiomics",
}


class BundleValidationError(RuntimeError):
    """Raised when a model bundle is missing or incompatible."""


class PredictionInputError(ValueError):
    """Raised when an inference request does not contain all required features."""

    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class LateFusionPredictor:
    """In-memory predictor backed by one trusted joblib/pickle bundle."""

    def __init__(self, bundle_path: Path):
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = self._load_bundle(self.bundle_path)
        self._validate_bundle(self.bundle)

    @staticmethod
    def _load_bundle(bundle_path: Path) -> Dict[str, Any]:
        if not bundle_path.is_file():
            raise BundleValidationError(
                f"Model bundle not found: {bundle_path}. Run the ML9 training "
                "pipeline once to create artifacts/late_fusion_bundle.pkl."
            )
        try:
            bundle = joblib.load(bundle_path)
        except Exception as exc:
            raise BundleValidationError(
                f"Cannot load model bundle {bundle_path}: {exc}"
            ) from exc
        if not isinstance(bundle, dict):
            raise BundleValidationError("Model bundle must be a dictionary.")
        return bundle

    @staticmethod
    def _validate_bundle(bundle: Dict[str, Any]) -> None:
        if bundle.get("bundle_type") != "fusion_subsidence_full_late_fusion":
            raise BundleValidationError("Unsupported model bundle type.")
        if bundle.get("bundle_version") not in {1, 2}:
            raise BundleValidationError("Unsupported model bundle version.")

        base_models = bundle.get("base_models")
        if not isinstance(base_models, dict):
            raise BundleValidationError("Bundle has no base_models mapping.")
        for modality in MODALITIES:
            payload = base_models.get(modality)
            if not isinstance(payload, dict):
                raise BundleValidationError(
                    f"Bundle is missing the {modality} base model."
                )
            if not payload.get("features"):
                raise BundleValidationError(
                    f"Bundle has no selected features for {modality}."
                )
            if not hasattr(payload.get("pipeline"), "predict_proba"):
                raise BundleValidationError(
                    f"{modality} pipeline does not implement predict_proba."
                )

        late_payload = bundle.get("late_fusion")
        if not isinstance(late_payload, dict):
            raise BundleValidationError("Bundle has no full Late_Fusion model.")
        if tuple(late_payload.get("features", [])) != EXPECTED_META_FEATURES:
            raise BundleValidationError(
                "Late_Fusion features must be the ordered probabilities of "
                "Clinical, Muscle, and Radiomics."
            )
        if not hasattr(late_payload.get("pipeline"), "predict_proba"):
            raise BundleValidationError(
                "Late_Fusion pipeline does not implement predict_proba."
            )
        threshold = late_payload.get("decision_threshold")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise BundleValidationError("Late_Fusion threshold must be within [0, 1].")

        if bundle.get("bundle_version") == 2:
            routing_models = bundle.get("routing_models")
            if not isinstance(routing_models, dict):
                raise BundleValidationError("Bundle has no routing_models mapping.")
            actual_routes = set(routing_models)
            missing_routes = CLINICAL_REQUIRED_ROUTES - actual_routes
            if missing_routes:
                raise BundleValidationError(
                    f"Bundle is missing inference routes: {sorted(missing_routes)}"
                )
            unsupported_routes = actual_routes - CLINICAL_REQUIRED_ROUTES
            if unsupported_routes:
                raise BundleValidationError(
                    "Bundle contains routes without required Clinical input: "
                    f"{sorted(unsupported_routes)}"
                )
            for route_key, route in routing_models.items():
                if "Clinical" not in route.get("modalities", []):
                    raise BundleValidationError(
                        f"Route {route_key} does not include required Clinical input."
                    )
                if route.get("strategy") not in {"direct", "late_fusion"}:
                    raise BundleValidationError(
                        f"Route {route_key} has an unsupported strategy."
                    )
                if not route.get("features"):
                    raise BundleValidationError(
                        f"Route {route_key} has no selected features."
                    )
                if not hasattr(route.get("pipeline"), "predict_proba"):
                    raise BundleValidationError(
                        f"Route {route_key} pipeline does not implement predict_proba."
                    )
                route_threshold = route.get("decision_threshold")
                if (
                        not isinstance(route_threshold, (int, float))
                        or not 0 <= route_threshold <= 1
                ):
                    raise BundleValidationError(
                        f"Route {route_key} threshold must be within [0, 1]."
                    )

    @property
    def clinical_schema(self):
        return describe_clinical_fields(self.bundle.get("clinical_form_schema", []))

    def expected_features(self, modality: str) -> list[str]:
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        features = list(self.bundle["base_models"][modality]["features"])
        prefix = f"{modality}___"
        for route in self._routing_models().values():
            features.extend(
                feature
                for feature in route["features"]
                if feature.startswith(prefix)
            )
        return list(dict.fromkeys(features))

    def _routing_models(self) -> Dict[str, Dict[str, Any]]:
        if self.bundle.get("bundle_version") == 2:
            return self.bundle["routing_models"]
        return {
            "Clinical+Muscle+Radiomics": {
                "dataset": "Late_Fusion",
                "modalities": list(MODALITIES),
                "strategy": "late_fusion",
                **self.bundle["late_fusion"],
            }
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "bundle_type": self.bundle["bundle_type"],
            "bundle_version": self.bundle["bundle_version"],
            "created_at_utc": self.bundle.get("created_at_utc"),
            "positive_class": self.bundle.get("positive_class", 1),
            "base_models": {
                modality: {
                    "model_name": self.bundle["base_models"][modality][
                        "model_name"
                    ],
                    "feature_count": len(self.expected_features(modality)),
                }
                for modality in MODALITIES
            },
            "late_fusion": {
                "model_name": self.bundle["late_fusion"]["model_name"],
                "decision_threshold": self.bundle["late_fusion"][
                    "decision_threshold"
                ],
            },
            "inference_routes": {
                route_key: {
                    "dataset": route.get("dataset"),
                    "modalities": route.get("modalities", []),
                    "strategy": route.get("strategy"),
                    "model_name": route.get("model_name"),
                    "selection_cv_auc": route.get("selection_cv_auc"),
                    "decision_threshold": route.get("decision_threshold"),
                }
                for route_key, route in self._routing_models().items()
            },
        }

    def predict(
            self,
            clinical: Mapping[str, Any] | None = None,
            muscle: Mapping[str, Any] | None = None,
            radiomics: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payloads = {
            "Clinical": clinical,
            "Muscle": muscle,
            "Radiomics": radiomics,
        }
        if not isinstance(clinical, Mapping) or not clinical:
            supplied_optional_modalities = [
                modality for modality in ("Muscle", "Radiomics")
                if isinstance(payloads[modality], Mapping) and payloads[modality]
            ]
            raise PredictionInputError(
                "Complete all Clinical fields before prediction or adding "
                "Muscle/Radiomics inputs.",
                {
                    "required_modality": "Clinical",
                    "supplied_optional_modalities": supplied_optional_modalities,
                },
            )
        available_modalities = [
            modality
            for modality in MODALITIES
            if isinstance(payloads[modality], Mapping) and payloads[modality]
        ]
        route_key = "+".join(available_modalities)
        route = self._routing_models().get(route_key)
        if route is None:
            raise PredictionInputError(
                f"The loaded bundle has no model for {route_key}.",
                {"available_modalities": available_modalities},
            )

        base_probabilities = {}
        for modality in available_modalities:
            model_payload = self.bundle["base_models"][modality]
            values = self._ordered_numeric_features(
                modality,
                payloads[modality],
                model_payload["features"],
            )
            frame = pd.DataFrame([values], columns=model_payload["features"])
            base_probabilities[modality] = self._positive_probability(
                model_payload["pipeline"], frame
            )

        if route["strategy"] == "late_fusion":
            route_values = {
                f"{modality}_Probability": base_probabilities[modality]
                for modality in available_modalities
            }
        else:
            route_values = {}
            for modality in available_modalities:
                modality_features = [
                    feature for feature in route["features"]
                    if feature.startswith(f"{modality}___")
                ]
                route_values.update(self._ordered_numeric_features(
                    modality, payloads[modality], modality_features
                ))
        route_frame = pd.DataFrame(
            [route_values], columns=route["features"]
        )
        final_probability = self._positive_probability(
            route["pipeline"], route_frame
        )
        threshold = float(route["decision_threshold"])
        predicted_class = int(final_probability >= threshold)

        return {
            "model_version": self.bundle.get("created_at_utc"),
            "positive_class": int(self.bundle.get("positive_class", 1)),
            "probability_subsidence": final_probability,
            "decision_threshold": threshold,
            "predicted_class": predicted_class,
            "classification": "Positive" if predicted_class == 1 else "Negative",
            "input_modalities": available_modalities,
            "selected_route": route.get("dataset", route_key),
            "model_name": route.get("model_name"),
            "fusion_strategy": route["strategy"],
            "base_probabilities": {
                modality.lower(): probability
                for modality, probability in base_probabilities.items()
            },
        }

    @staticmethod
    def _ordered_numeric_features(
            modality: str,
            payload: Mapping[str, Any],
            expected_features: Sequence[str],
    ) -> Dict[str, float]:
        if not isinstance(payload, Mapping):
            raise PredictionInputError(
                f"{modality} features must be a JSON object.",
                {"modality": modality},
            )

        normalized = {}
        missing = []
        invalid = []
        prefix = f"{modality}___"
        for feature in expected_features:
            short_name = feature.removeprefix(prefix)
            if feature in payload:
                raw_value = payload[feature]
            elif short_name in payload:
                raw_value = payload[short_name]
            else:
                missing.append(feature)
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid.append(feature)
                continue
            if not math.isfinite(value):
                invalid.append(feature)
                continue
            if (
                modality == "Clinical"
                and short_name == "AI_CCI"
                and (not value.is_integer() or not 0 <= value <= 10)
            ):
                invalid.append(feature)
                continue
            normalized[feature] = value

        if missing or invalid:
            raise PredictionInputError(
                f"{modality} features are incomplete or invalid.",
                {
                    "modality": modality,
                    "missing_features": missing,
                    "invalid_features": invalid,
                },
            )
        return normalized

    @staticmethod
    def _positive_probability(model, frame: pd.DataFrame) -> float:
        probabilities = LateFusionPredictor._positive_probabilities(model, frame)
        if len(probabilities) != 1:
            raise BundleValidationError("Expected one prediction row.")
        return float(probabilities[0])

    @staticmethod
    def _positive_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
        """Use the same positive-class lookup for inference and batched SHAP."""
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        classes = np.asarray(getattr(model, "classes_", []))
        positive_positions = np.flatnonzero(classes == 1)
        if (
            probabilities.ndim != 2 or probabilities.shape[0] != len(frame)
            or probabilities.shape[1] != len(classes) or len(positive_positions) != 1
        ):
            raise BundleValidationError(
                "Stored model does not expose valid probabilities for one unambiguous positive class (1)."
            )
        positive = probabilities[:, positive_positions[0]]
        if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
            raise BundleValidationError(
                "Stored model returned a non-finite probability outside [0, 1]."
            )
        return positive
