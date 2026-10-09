"""One local, end-to-end SHAP explanation of the selected late-fusion route.

The explained function is raw features -> base probabilities -> the stored
late-fusion pipeline -> final subsidence probability. Base models never get
separate plots. Reference patients are joined by Name, not by row position.
No model is fitted here; without usable reference tables SHAP is unavailable.
"""

import base64
from dataclasses import dataclass
import io
import threading
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .clinical_schema import CLINICAL_LABELS
from .inference import MODALITIES, PredictionInputError


BACKGROUND_SOURCES = {
    "Clinical": "clinical_features_new.csv",
    "Muscle": "CSA_muscles_features_new.csv",
    "Radiomics": "radiomics_features_new.csv",
}
BACKGROUND_SAMPLE_SIZE = 50
EXPLANATION_PERMUTATIONS = 5


class ExplanationUnavailableError(RuntimeError):
    """SHAP, the selected route, or its reference tables are not usable."""


class ExplanationError(RuntimeError):
    """The final prediction could not be explained faithfully."""


@dataclass
class PreparedLateFusionExplanation:
    route_key: str
    sample: pd.DataFrame
    prediction: dict


class ShapExplainer:
    """Cache an end-to-end explainer per late-fusion inference route."""

    def __init__(self, predictor, data_directory: Path):
        self.predictor = predictor
        self.data_directory = Path(data_directory)
        self._explainers = {}
        self._lock = threading.Lock()

    def prepare(self, payload: Mapping) -> PreparedLateFusionExplanation:
        """Validate the original prediction inputs; never trust client probabilities."""
        supplied = {}
        for modality in MODALITIES:
            value = payload.get(modality.lower())
            if value in (None, "", {}):
                value = None
            elif not isinstance(value, dict):
                raise PredictionInputError(f"{modality} must be a JSON object of feature values.")
            supplied[modality.lower()] = value
        prediction = self.predictor.predict(**supplied)
        if prediction["fusion_strategy"] != "late_fusion":
            raise ExplanationUnavailableError(
                "SHAP is only available for late-fusion predictions. "
                "The selected prediction does not use late fusion."
            )
        route_key = "+".join(prediction["input_modalities"])
        features, feature_groups = self._feature_layout(route_key)
        values = {}
        for modality, names in feature_groups.items():
            values.update(self.predictor._ordered_numeric_features(
                modality, supplied[modality.lower()], names
            ))
        return PreparedLateFusionExplanation(
            route_key, pd.DataFrame([values], columns=features), prediction
        )

    def explain(self, prepared: PreparedLateFusionExplanation) -> dict:
        """Return one waterfall whose attributions sum to the final probability."""
        route = self.predictor._routing_models()[prepared.route_key]
        sample = prepared.sample
        features = list(sample.columns)
        with self._lock:
            explainer = self._explainer_for(prepared.route_key)
            try:
                explanation = explainer(
                    sample,
                    max_evals=EXPLANATION_PERMUTATIONS * (2 * len(features) + 1),
                    silent=True,
                )
            except Exception as exc:
                raise ExplanationError(f"SHAP could not explain the late-fusion prediction: {exc}") from exc
            contributions = np.asarray(explanation.values, dtype=float).reshape(-1)
            baseline = np.asarray(explanation.base_values, dtype=float).reshape(-1)
            if (
                len(contributions) != len(features) or len(baseline) != 1
                or not np.isfinite(contributions).all() or not np.isfinite(baseline).all()
            ):
                raise ExplanationError("SHAP returned invalid final-probability attributions.")
            base_value = float(baseline[0])
            probability = prepared.prediction["probability_subsidence"]
            reconstructed = float(base_value + contributions.sum())
            if not np.isclose(reconstructed, probability, rtol=1e-5, atol=1e-6):
                raise ExplanationError("SHAP contributions do not reconstruct the final late-fusion probability.")
            labels = [_feature_label(name) for name in features]
            image = _waterfall_png(
                contributions, base_value, sample.iloc[0].to_numpy(dtype=float), labels
            )

        return {
            "modality": "Late Fusion",
            "scope": "end_to_end",
            "fusion_strategy": "late_fusion",
            "selected_route": prepared.prediction["selected_route"],
            "input_modalities": prepared.prediction["input_modalities"],
            "model_name": route["model_name"],
            "base_value": base_value,
            "probability": probability,
            "reconstructed_probability": reconstructed,
            "image_png_base64": image,
            "contributions": [
                {
                    "feature": features[index], "label": labels[index],
                    "value": float(sample.iloc[0, index]),
                    "shap": float(contributions[index]),
                }
                for index in np.argsort(-np.abs(contributions))
            ],
        }

    def _feature_layout(self, route_key):
        route = self.predictor._routing_models()[route_key]
        if route["strategy"] != "late_fusion":
            raise ExplanationUnavailableError("Only late-fusion routes can be explained.")
        groups = {
            modality: list(self.predictor.bundle["base_models"][modality]["features"])
            for modality in route["modalities"]
        }
        return [name for names in groups.values() for name in names], groups

    def _predict_function(self, route_key):
        features, groups = self._feature_layout(route_key)
        route = self.predictor._routing_models()[route_key]

        def predict(data):
            frame = pd.DataFrame(np.asarray(data, dtype=float), columns=features)
            probabilities = {}
            for modality, names in groups.items():
                pipeline = self.predictor.bundle["base_models"][modality]["pipeline"]
                probabilities[f"{modality}_Probability"] = self.predictor._positive_probabilities(
                    pipeline, frame.loc[:, names]
                )
            meta = pd.DataFrame(probabilities, columns=route["features"])
            return self.predictor._positive_probabilities(route["pipeline"], meta)

        return predict

    def _explainer_for(self, route_key):
        if route_key in self._explainers:
            return self._explainers[route_key]
        try:
            import shap
        except ImportError as exc:
            raise ExplanationUnavailableError("SHAP is not installed; add shap from requirements.txt.") from exc
        features, groups = self._feature_layout(route_key)
        reference = self._background(groups)
        reference = reference.sample(n=min(BACKGROUND_SAMPLE_SIZE, len(reference)), random_state=0)
        explainer = shap.PermutationExplainer(
            self._predict_function(route_key), reference.loc[:, features], seed=0
        )
        self._explainers[route_key] = explainer
        return explainer

    def _background(self, groups) -> pd.DataFrame:
        parts = []
        for modality, features in groups.items():
            path = self.data_directory / BACKGROUND_SOURCES[modality]
            if not path.is_file():
                raise ExplanationUnavailableError(f"SHAP needs the development table {path.name} as its reference distribution.")
            try:
                table = pd.read_csv(path, dtype={"Name": "string"})
            except Exception as exc:
                raise ExplanationUnavailableError(f"Cannot read the SHAP background table {path.name}: {exc}") from exc
            short_names = [name.removeprefix(f"{modality}___") for name in features]
            missing = [name for name in ["Name", *short_names] if name not in table.columns]
            if missing:
                raise ExplanationUnavailableError(f"{path.name} is missing reference columns: {', '.join(missing[:4])}")
            names = table["Name"].str.strip()
            if names.isna().any() or (names == "").any() or names.duplicated().any():
                raise ExplanationUnavailableError(f"{path.name} needs unique, nonempty Name values to align reference patients.")
            numeric = table[short_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric.index = names
            numeric.columns = list(features)
            parts.append(numeric)
        background = pd.concat(parts, axis=1, join="inner").dropna()
        if background.empty:
            raise ExplanationUnavailableError("No complete, patient-matched development rows are available for late-fusion SHAP.")
        return background


def _feature_label(feature):
    modality, _, short_name = feature.partition("___")
    name = CLINICAL_LABELS.get(short_name, short_name) if modality == "Clinical" else short_name
    return name


def _waterfall_png(contributions: np.ndarray, base_value: float, data: np.ndarray, feature_names: Sequence[str]) -> str:
    import shap

    explanation = shap.Explanation(
        values=contributions, base_values=base_value, data=data,
        feature_names=[textwrap.fill(name, width=48) for name in feature_names],
    )
    try:
        shap.plots.waterfall(explanation, max_display=11, show=False)
        figure = plt.gcf()
        figure.set_size_inches(10, 0.55 * min(len(feature_names), 11) + 2.0)
        figure.suptitle("Late Fusion: contribution to the final subsidence probability", fontsize=11, y=1.0)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    finally:
        plt.close("all")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
