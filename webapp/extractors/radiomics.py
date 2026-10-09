"""Headless FAE-compatible Radiomics extraction using PyRadiomics.

This module owns the FAE configuration: it reduces the checked-in YAML to the
image types and feature classes the selected model features actually need. The
extraction itself runs in ``fae_worker.py`` under the interpreter that has
PyRadiomics installed, so the web application does not need it.

The VOI comes from one of two places. An uploaded mask is used as-is, which is
how the training features were produced. Otherwise ``vertebral_voi.py`` runs
TotalSegmentator's MR body and level tasks and intersects their masks.
"""

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import yaml
from werkzeug.datastructures import FileStorage

from .errors import (
    ExtractionExecutionError,
    ExtractionInputError,
    ExtractorUnavailableError,
)
from .vertebral_voi import segment_vertebra_voi_events, summarise
from .worker import resolve_python, run_worker

WORKER_SCRIPT = Path(__file__).resolve().parent / "fae_worker.py"
RESULT_PREFIX = "##FAE_WORKER_JSON##"
STAGE_PREFIX = "##FAE_WORKER_STAGE##"


FEATURE_PATTERN = re.compile(
    r"^Radiomics___MRI_(?P<image_type>.+)_"
    r"(?P<feature_class>firstorder|shape|glcm|glrlm|glszm|gldm|ngtdm)_"
    r"(?P<feature_name>.+)$"
)


@dataclass
class PreparedRadiomicsExtraction:
    image_path: Path
    mask_path: Path
    parameters: dict
    python_executable: str
    timeout_seconds: int
    segmentation: dict


def prepare_radiomics_extraction_events(
        image_path, expected_features, *, config_path, python_executable="",
        timeout_seconds=900, levels=(), segmentation_options=None,
):
    """Build the VOI serially, without starting the PyRadiomics worker."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise ExtractorUnavailableError(f"FAE_RADIOMICS_CONFIG does not exist: {config_path}")
    parameters = _selected_fae_params(config_path, expected_features)
    interpreter = resolve_python(python_executable)
    image_path = Path(image_path)
    with image_path.open("rb") as stream:
        upload = FileStorage(stream=stream, filename=image_path.name)
        segmentation = yield from segment_vertebra_voi_events(upload, levels, **(segmentation_options or {}))
    summary = summarise(segmentation)
    yield {"stage": "segmented", "segmentation": summary}
    mask_path = Path(segmentation["mask_path"])
    _ensure_non_empty(mask_path, "Prepared Radiomics VOI")
    return PreparedRadiomicsExtraction(
        Path(segmentation["image_path"]), mask_path, parameters, interpreter, timeout_seconds, summary,
    )


def extract_prepared_radiomics_features_events(prepared, expected_features, *, on_process=None):
    """Run PyRadiomics on the saved image and VOI; never segment here."""
    _ensure_non_empty(prepared.image_path, "Prepared Radiomics MRI")
    _ensure_non_empty(prepared.mask_path, "Prepared Radiomics VOI")
    report = yield from run_worker(
        prepared.python_executable, WORKER_SCRIPT,
        {"image": str(prepared.image_path), "mask": str(prepared.mask_path), "parameters": prepared.parameters},
        stage_prefix=STAGE_PREFIX, result_prefix=RESULT_PREFIX,
        timeout_seconds=prepared.timeout_seconds, label="PyRadiomics", on_process=on_process,
    )
    yield {"stage": "selecting"}
    return _selected_report_features(report, expected_features)


def _selected_report_features(report, expected_features):
    all_features = {
        name: float(value) for name, value in report.get("features", {}).items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    missing = [feature for feature in expected_features if feature not in all_features]
    if missing:
        raise ExtractionExecutionError("FAE/PyRadiomics did not produce selected model features: " + ", ".join(missing))
    return {feature: all_features[feature] for feature in expected_features}


def extract_radiomics_features(
        image_upload: FileStorage,
        mask_upload: FileStorage | None,
        expected_features: Sequence[str],
        **options,
) -> Dict[str, float]:
    """Drain :func:`extract_radiomics_features_events`, dropping stage events."""
    events = extract_radiomics_features_events(
        image_upload, mask_upload, expected_features, **options
    )
    while True:
        try:
            next(events)
        except StopIteration as stop:
            return stop.value


def extract_radiomics_features_events(
        image_upload: FileStorage,
        mask_upload: FileStorage | None,
        expected_features: Sequence[str],
        *,
        config_path: Path,
        python_executable: str = "",
        timeout_seconds: int = 900,
        levels: Sequence[str] = (),
        segmentation_options: Dict | None = None,
):
    """Extract the selected `MRI_*` features with FAE's PyRadiomics settings.

    ``mask_upload`` is the VOI. When it is ``None``, TotalSegmentator segments
    ``image_upload`` and intersects its body mask with ``levels`` (for example
    ``["L3", "L4"]``).

    Yields ``{"stage": ...}`` dicts as each worker reaches each step and returns
    the selected features, so a caller can stream honest progress.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise ExtractorUnavailableError(
            f"FAE_RADIOMICS_CONFIG does not exist: {config_path}"
        )
    parameters = _selected_fae_params(config_path, expected_features)
    interpreter = resolve_python(python_executable)

    with tempfile.TemporaryDirectory(prefix="fae_radiomics_web_") as temp_name:
        temp_dir = Path(temp_name)
        if mask_upload is not None:
            image_path = temp_dir / f"image{_nifti_suffix(image_upload.filename)}"
            mask_path = temp_dir / f"roi{_nifti_suffix(mask_upload.filename)}"
            image_upload.save(image_path)
            mask_upload.save(mask_path)
            _ensure_non_empty(image_path, "Radiomics MRI")
            _ensure_non_empty(mask_path, "Radiomics VOI mask")
        else:
            # The segmentation keeps the image and the VOI it wrote, so both
            # stay on disk for comparison instead of living in this temp dir.
            segmentation = yield from segment_vertebra_voi_events(
                image_upload, levels, **(segmentation_options or {})
            )
            yield {"stage": "segmented", "segmentation": summarise(segmentation)}
            image_path = Path(segmentation["image_path"])
            mask_path = Path(segmentation["mask_path"])

        job = {
            "image": str(image_path),
            "mask": str(mask_path),
            "parameters": parameters,
        }
        report = yield from run_worker(
            interpreter,
            WORKER_SCRIPT,
            job,
            stage_prefix=STAGE_PREFIX,
            result_prefix=RESULT_PREFIX,
            timeout_seconds=timeout_seconds,
            label="PyRadiomics",
        )

    yield {"stage": "selecting"}
    return _selected_report_features(report, expected_features)


def _selected_fae_params(
        config_path: Path, expected_features: Sequence[str]
) -> Dict:
    """Keep FAE settings while calculating only needed filters/features."""
    try:
        with config_path.open("r", encoding="utf-8") as file:
            base_params = yaml.safe_load(file)
    except Exception as exc:
        raise ExtractorUnavailableError(
            f"Cannot read FAE Radiomics YAML {config_path}: {exc}"
        ) from exc
    if not isinstance(base_params, dict):
        raise ExtractorUnavailableError("FAE Radiomics YAML must contain a mapping.")

    base_image_types = base_params.get("imageType", {})
    selected_image_types = {}
    selected_feature_classes: Dict[str, set[str]] = {}
    invalid_features = []
    for full_name in expected_features:
        match = FEATURE_PATTERN.match(full_name)
        if match is None:
            invalid_features.append(full_name)
            continue
        output_image_type = match.group("image_type")
        pyradiomics_image_type = _pyradiomics_image_type(output_image_type)
        if pyradiomics_image_type is None:
            invalid_features.append(full_name)
            continue
        if pyradiomics_image_type not in base_image_types:
            raise ExtractorUnavailableError(
                f"FAE YAML does not enable required image type "
                f"{pyradiomics_image_type}."
            )
        selected_image_types[pyradiomics_image_type] = (
            base_image_types[pyradiomics_image_type] or {}
        )
        selected_feature_classes.setdefault(
            match.group("feature_class"), set()
        ).add(match.group("feature_name"))

    if invalid_features:
        raise ExtractionInputError(
            "Selected Radiomics feature names are not FAE/PyRadiomics-compatible: "
            + ", ".join(invalid_features)
        )
    return {
        "imageType": selected_image_types,
        "featureClass": {
            feature_class: sorted(feature_names)
            for feature_class, feature_names in selected_feature_classes.items()
        },
        "setting": dict(base_params.get("setting", {})),
    }


def _pyradiomics_image_type(output_image_type: str):
    if output_image_type == "original":
        return "Original"
    if output_image_type == "exponential":
        return "Exponential"
    if output_image_type == "gradient":
        return "Gradient"
    if output_image_type.startswith("lbp-3D-"):
        return "LBP3D"
    if output_image_type.startswith("log-sigma-"):
        return "LoG"
    if output_image_type == "logarithm":
        return "Logarithm"
    if output_image_type == "square":
        return "Square"
    if output_image_type == "squareroot":
        return "SquareRoot"
    if output_image_type.startswith("wavelet-"):
        return "Wavelet"
    return None


def _ensure_non_empty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ExtractionInputError(f"{label} upload is empty.")


def _nifti_suffix(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
