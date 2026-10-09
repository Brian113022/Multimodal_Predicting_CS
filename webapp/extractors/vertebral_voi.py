"""Build a level-specific vertebral-body VOI from a sagittal MRI.

TotalSegmentator exposes the two pieces needed by the Radiomics model through
different MR tasks:

* ``total_mr`` provides the combined ``vertebrae`` body mask (posterior arches
  excluded); and
* ``vertebrae_mr`` identifies individual vertebral levels.

The worker intersects those masks and writes label ``1`` on the uploaded MRI
grid, which is the VOI expected by the FAE/PyRadiomics configuration.
"""

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

from werkzeug.datastructures import FileStorage

from .errors import ExtractionInputError, ExtractorUnavailableError
from .worker import resolve_python, run_worker


WORKER_SCRIPT = Path(__file__).resolve().parent / "totalseg_worker.py"
RESULT_PREFIX = "##TOTALSEG_WORKER_JSON##"
STAGE_PREFIX = "##TOTALSEG_WORKER_STAGE##"

# Label indices emitted by TotalSegmentator's ``vertebrae_mr`` task. The web
# app is deliberately limited to the thoracolumbar levels relevant to this
# model; sacrum cannot be intersected with the task's vertebral-body class.
LEVELS = (
    ("T11", 8, "T11 vertebral body"),
    ("T12", 7, "T12 vertebral body"),
    ("L1", 6, "L1 vertebral body"),
    ("L2", 5, "L2 vertebral body"),
    ("L3", 4, "L3 vertebral body"),
    ("L4", 3, "L4 vertebral body"),
    ("L5", 2, "L5 vertebral body"),
)

ALL_LABELS: Dict[str, int] = {code: label for code, label, _ in LEVELS}
VERTEBRA_LABELS = dict(ALL_LABELS)
LEVEL_DESCRIPTIONS = {code: description for code, _, description in LEVELS}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def available_levels():
    """Return selectable vertebrae in cranial-to-caudal order."""
    return [
        {
            "value": code,
            "label": code,
            "kind": "vertebra",
            "description": description,
            "totalsegmentator_label": label,
        }
        for code, label, description in LEVELS
    ]


def normalise_levels(values) -> list:
    """Validate requested vertebrae and return them in anatomical order."""
    if isinstance(values, str):
        values = [values]
    tokens = []
    for value in values or []:
        tokens.extend(re.split(r"[,;\s]+", str(value)))

    requested = []
    unknown = []
    for value in tokens:
        code = str(value).strip().upper()
        if not code:
            continue
        if code not in ALL_LABELS:
            unknown.append(str(value).strip())
        elif code not in requested:
            requested.append(code)

    if unknown:
        raise ExtractionInputError(
            "Unknown or unsupported vertebra level(s): "
            + ", ".join(unknown)
            + ". Choose individual vertebrae ("
            + ", ".join(ALL_LABELS)
            + "). Disc notation such as L4-L5 is not a vertebral-body level."
        )
    if not requested:
        raise ExtractionInputError(
            "Select at least one vertebra (for example L4 and L5) so a "
            "level-specific body VOI can be built."
        )
    order = list(ALL_LABELS)
    return sorted(requested, key=order.index)


def segment_vertebra_voi_events(
        image_upload: FileStorage,
        levels: Sequence[str],
        *,
        output_root: Path,
        python_executable: str = "",
        weights_path: str = "",
        device: str = "",
        require_gpu: bool = True,
        nr_threads_resampling: int = 1,
        nr_threads_saving: int = 2,
        timeout_seconds: int = 3600,
):
    """Segment an MRI and return a binary selected-level body VOI report."""
    codes = normalise_levels(levels)
    if not WORKER_SCRIPT.is_file():
        raise ExtractorUnavailableError(
            f"TotalSegmentator worker script is missing: {WORKER_SCRIPT}"
        )
    interpreter = resolve_python(python_executable)

    output_root = Path(output_root).expanduser()
    case_directory = _case_directory(output_root, image_upload.filename)
    stem = _case_stem(image_upload.filename)
    suffix = _nifti_suffix(image_upload.filename)
    image_path = case_directory / f"{stem}_image{suffix}"
    image_upload.save(image_path)
    if not image_path.is_file() or image_path.stat().st_size == 0:
        shutil.rmtree(case_directory, ignore_errors=True)
        raise ExtractionInputError("Radiomics MRI upload is empty.")

    yield {"stage": "preparing", "levels": codes}
    try:
        with tempfile.TemporaryDirectory(prefix="totalsegmentator_web_") as temp_name:
            job = {
                "image": str(image_path),
                "work_dir": str(Path(temp_name)),
                "output_dir": str(case_directory),
                "case_stem": stem,
                "levels": codes,
                "labels": {code: ALL_LABELS[code] for code in codes},
                "weights_path": str(weights_path or ""),
                "device": str(device or ""),
                "require_gpu": bool(require_gpu),
                "nr_threads_resampling": int(nr_threads_resampling),
                "nr_threads_saving": int(nr_threads_saving),
                "timeout_seconds": int(timeout_seconds),
            }
            report = yield from run_worker(
                interpreter,
                WORKER_SCRIPT,
                job,
                stage_prefix=STAGE_PREFIX,
                result_prefix=RESULT_PREFIX,
                timeout_seconds=timeout_seconds,
                label="TotalSegmentator",
            )
    except Exception:
        # Avoid retaining incomplete patient data when segmentation fails.
        shutil.rmtree(case_directory, ignore_errors=True)
        raise

    report["case_directory"] = str(case_directory)
    return report


def summarise(report: dict) -> dict:
    """Return the non-image portion of a worker report to the browser."""
    directory = Path(report.get("case_directory", ""))
    return {
        "source": "totalsegmentator",
        "levels": report.get("levels", []),
        "labels": report.get("labels", {}),
        "voxel_counts": report.get("voxel_counts", {}),
        "level_voxel_counts": report.get("level_voxel_counts", {}),
        "mask_voxels": report.get("mask_voxels"),
        "body_mask_voxels": report.get("body_mask_voxels"),
        "body_only": True,
        "body_only_voxels": report.get("mask_voxels"),
        "body_only_removed_fraction": report.get("body_only_removed_fraction"),
        "intersection_method": report.get("intersection_method"),
        "resampled_masks": report.get("resampled_masks", []),
        "image_size": report.get("image_size"),
        "image_spacing": report.get("image_spacing"),
        "device": report.get("device"),
        "device_trace": report.get("device_trace") or {},
        "gpu_trace": report.get("gpu_trace"),
        "version": report.get("version"),
        "duration_seconds": report.get("duration_seconds"),
        "directory": str(directory),
        "directory_name": directory.name,
        "files": {**report.get("files", {}), "metadata": "metadata.json"},
    }


def _case_directory(output_root: Path, filename: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{stamp}_{_case_stem(filename)}"
    candidate = base
    index = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{index}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def _case_stem(filename: str) -> str:
    name = Path(filename or "image").name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = _SAFE_NAME.sub("_", name).strip("._-")
    return name[:64] or "image"


def _nifti_suffix(filename: str) -> str:
    return ".nii.gz" if (filename or "").lower().endswith(".nii.gz") else ".nii"
