"""MuscleMap adapter and conversion to the training Muscle feature schema.

Segmentation, the single-slice fixes and the GMM fallback all need nibabel,
monai and MuscleMap's ``scripts`` package, which live in the MuscleMap
environment rather than the web application's. ``mm_worker.py`` runs there and
this module only orchestrates it and aggregates its CSVs.

One case is usually several axial T2 series, so every uploaded series is
segmented and extracted separately and the per-series tables are concatenated
before aggregation, exactly as ``combine_musclemap_case_metrics.py`` does:

    aVol_CODE = sum(Volume (ml)) / sum(Number of Slices)
    tFat_CODE = sum((Fat (%) + Undefined (%)) * Number of Slices) / sum(Number of Slices)
"""

import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
from werkzeug.datastructures import FileStorage

from .errors import (
    ExtractionError,
    ExtractionExecutionError,
    ExtractionInputError,
    ExtractorUnavailableError,
)
from .worker import run_worker
from .muscle_trace import finish_trace, record_metrics, record_segmentation, save_trace, start_trace


WORKER_SCRIPT = Path(__file__).resolve().parent / "mm_worker.py"
RESULT_PREFIX = "##MM_WORKER_JSON##"
STAGE_PREFIX = "##MM_WORKER_STAGE##"

# Label values are fixed by the abdomen model config shipped with MuscleMap
# (contrast_agnostic_abdomen_model.json); matching on them is more robust than
# matching the free-text Anatomy column.
LABEL_TO_CODE = {
    1: "MF_R",
    2: "MF_L",
    3: "ES_R",
    4: "ES_L",
    5: "PM_R",
    6: "PM_L",
    7: "QL_R",
    8: "QL_L",
}

ANATOMY_TO_CODE = {
    "erector spinae left": "ES_L",
    "left erector spinae": "ES_L",
    "erector spinae right": "ES_R",
    "right erector spinae": "ES_R",
    "multifidus left": "MF_L",
    "left multifidus": "MF_L",
    "thoracolumbar multifidus left": "MF_L",
    "left thoracolumbar multifidus": "MF_L",
    "multifidus right": "MF_R",
    "right multifidus": "MF_R",
    "thoracolumbar multifidus right": "MF_R",
    "right thoracolumbar multifidus": "MF_R",
    "psoas major left": "PM_L",
    "left psoas major": "PM_L",
    "psoas major right": "PM_R",
    "right psoas major": "PM_R",
    "quadratus lumborum left": "QL_L",
    "left quadratus lumborum": "QL_L",
    "quadratus lumborum right": "QL_R",
    "right quadratus lumborum": "QL_R",
}

# MuscleMap writes its own NIfTI outputs beside each source series, so selecting
# a whole case folder would otherwise feed those derived volumes back in.
DERIVED_NIFTI_TOKENS = (
    "_dseg",
    "_softseg",
    "_muscle_seg",
    "_fat_seg",
    "_undefined_seg",
    "_gmm_",
    "_kmeans_",
    "_dixon_",
    "_average_",
)

MUSCLE_CODES = ("ES_L", "ES_R", "MF_L", "MF_R", "PM_L", "PM_R", "QL_L", "QL_R")
REQUIRED_METRIC_COLUMNS = {
    "Anatomy",
    "Number of Slices",
    "Fat (%)",
    "Volume (ml)",
}


@dataclass
class PreparedMuscleExtraction:
    jobs: list
    python_executable: str
    timeout_seconds: int
    segmentation: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)


def prepare_muscle_extraction_events(
        image_paths, work_dir, *, command="mm_extract_metrics",
        segmentation_command="mm_segment", segmentation_region="abdomen",
        method="gmm", region="abdomen", components=3, timeout_seconds=900,
        python_executable="", output_root=None, source_manifest=None, workers=1,
):
    """Segment all selected series in one MuscleMap run; no metrics yet."""
    if method not in {"gmm", "kmeans"} or components not in {2, 3}:
        raise ExtractionInputError("Invalid MuscleMap method or component count.")
    if not image_paths:
        raise ExtractionInputError("No Axial T2 VIEW series were supplied.")
    resolved_command = _resolve_command(command, "mm_extract_metrics")
    resolved_segment = _resolve_command(segmentation_command, "mm_segment")
    interpreter = _resolve_python(python_executable, resolved_command)
    trace = start_trace(
        work_dir, output_root=output_root, method=method, region=region,
        components=components, source_manifest=source_manifest,
    )
    series = []
    for index, source in enumerate(image_paths):
        directory = Path(trace["directory"]) / f"series_{index + 1:04d}"
        directory.mkdir(parents=True)
        image = directory / ("image" + _nifti_suffix(Path(source).name))
        shutil.copyfile(source, image)
        series.append((Path(source), image, directory))
    job = {
        "operation": "segment_batch",
        "series": [{"image": str(image), "work_dir": str(directory)} for _, image, directory in series],
        "batch_dir": str(Path(trace["directory"]) / "_batch_segment"),
        "segmentation_command": resolved_segment, "segmentation_region": segmentation_region,
        "timeout_seconds": timeout_seconds,
    }
    report = yield from _muscle_worker_events(interpreter, job, timeout_seconds, {"series_total": len(series)})
    reports = report.get("series") or []
    if len(reports) != len(series):
        raise ExtractionExecutionError("MuscleMap did not segment every selected series.")
    jobs = []
    for index, ((source, image, directory), series_report) in enumerate(zip(series, reports)):
        position = {"series": index + 1, "series_total": len(series), "filename": source.name}
        mask = Path(series_report.get("mask_path", ""))
        if not mask.is_file() or mask.stat().st_size == 0:
            raise ExtractionExecutionError("MuscleMap did not produce a prepared segmentation mask.")
        yield {"stage": "previewing", **position}
        record_segmentation(trace, image, mask, source.name, series_report, source_image=source)
        jobs.append({"image": str(image), "mask": str(mask), "work_dir": str(directory), "position": position})
    trace["status"] = "segmented"
    save_trace(trace)
    yield {"stage": "segmented", "series_total": len(image_paths), "segmentation": trace}
    options = {"method": method, "region": region, "components": components, "workers": workers}
    return PreparedMuscleExtraction(jobs, interpreter, timeout_seconds, trace, options)


def extract_prepared_muscle_features_events(prepared, expected_features, *, on_process=None):
    """Calculate metrics from existing masks. This phase cannot run segmentation."""
    if not prepared.jobs:
        raise ExtractionInputError("No prepared Muscle series are available.")
    if any(not job.get("mask") for job in prepared.jobs):
        raise ExtractionInputError("Muscle feature extraction needs a prepared mask for every series.")
    job = {
        "operation": "extract_batch",
        "series": [{key: job[key] for key in ("image", "mask", "work_dir")} for job in prepared.jobs],
        **prepared.options,
    }
    report = yield from _muscle_worker_events(
        prepared.python_executable, job, prepared.timeout_seconds,
        {"series_total": len(prepared.jobs)}, on_process=on_process,
    )
    reports = report.get("series") or []
    if len(reports) != len(prepared.jobs):
        raise ExtractionExecutionError("MuscleMap did not return metrics for every series.")
    frames = []
    for index, series_report in enumerate(reports):
        results_csv = Path(series_report.get("results_csv", ""))
        if not results_csv.is_file():
            raise ExtractionExecutionError("MuscleMap completed but produced no metrics *_results.csv.")
        frames.append(_read_metrics_csv(results_csv))
        if prepared.segmentation:
            record_metrics(prepared.segmentation, index, str(results_csv), series_report)
    yield {"stage": "aggregating", "series_total": len(prepared.jobs)}
    metrics = pd.concat(frames, ignore_index=True)
    features = _select_expected_features(_metrics_to_features(metrics), expected_features)
    if prepared.segmentation:
        finish_trace(prepared.segmentation, metrics, features)
    return features


def _muscle_worker_events(interpreter, job, timeout_seconds, position, *, on_process=None):
    events = run_worker(
        interpreter, WORKER_SCRIPT, job, stage_prefix=STAGE_PREFIX,
        result_prefix=RESULT_PREFIX, timeout_seconds=timeout_seconds,
        label="MuscleMap", on_process=on_process,
    )
    try:
        while True:
            try:
                stage = next(events)
            except StopIteration as stop:
                return stop.value
            yield {**stage, **position}
    finally:
        events.close()


def extract_muscle_features(
        image_uploads,
        mask_uploads,
        expected_features: Sequence[str],
        **options,
) -> Dict[str, float]:
    """Drain :func:`extract_muscle_features_events`, discarding stage events."""
    events = extract_muscle_features_events(
        image_uploads, mask_uploads, expected_features, **options
    )
    while True:
        try:
            next(events)
        except StopIteration as stop:
            return stop.value


def extract_muscle_features_events(
        image_uploads,
        mask_uploads,
        expected_features: Sequence[str],
        *,
        command: str = "mm_extract_metrics",
        segmentation_command: str = "mm_segment",
        segmentation_region: str = "abdomen",
        method: str = "gmm",
        region: str = "abdomen",
        components: int = 3,
        timeout_seconds: int = 900,
        python_executable: str = "",
):
    """Segment every axial T2w series of one case, then aggregate its features.

    ``image_uploads`` is the case's series; ``mask_uploads`` is either ``None``
    (segment each series with mm_segment) or one mask per series in the same
    order. Yields ``{"stage": ...}`` dicts as each worker reaches each step and
    returns the selected features, so a caller can stream honest progress.
    """
    if method not in {"gmm", "kmeans"}:
        raise ExtractionInputError("MuscleMap method must be gmm or kmeans.")
    if components not in {2, 3}:
        raise ExtractionInputError("MuscleMap components must be 2 or 3.")

    resolved_command = _resolve_command(command, "mm_extract_metrics")
    resolved_segment_command = _resolve_command(segmentation_command, "mm_segment")
    resolved_python = _resolve_python(python_executable, resolved_command)

    series = _source_series(image_uploads)
    masks = list(mask_uploads or [])
    if masks and len(masks) != len(series):
        raise ExtractionInputError(
            f"Received {len(masks)} segmentation masks for {len(series)} series."
        )

    frames = []
    for index, image_upload in enumerate(series):
        position = {
            "series": index + 1,
            "series_total": len(series),
            "filename": image_upload.filename,
        }
        yield {"stage": "series_start", **position}

        with tempfile.TemporaryDirectory(prefix="musclemap_web_") as temp_name:
            temp_dir = Path(temp_name)
            image_path = temp_dir / f"image{_nifti_suffix(image_upload.filename)}"
            image_upload.save(image_path)
            _ensure_non_empty(image_path, f"Muscle MRI {image_upload.filename}")

            mask_path = None
            if masks:
                mask_upload = masks[index]
                mask_path = (
                    temp_dir / f"image_dseg{_nifti_suffix(mask_upload.filename)}"
                )
                mask_upload.save(mask_path)
                _ensure_non_empty(mask_path, "Muscle segmentation")

            job = {
                "work_dir": str(temp_dir),
                "image": str(image_path),
                "mask": str(mask_path) if mask_path is not None else None,
                "command": resolved_command,
                "segmentation_command": resolved_segment_command,
                "segmentation_region": segmentation_region,
                "method": method,
                "region": region,
                "components": components,
                "timeout_seconds": timeout_seconds,
            }
            worker = _run_worker(resolved_python, job, timeout_seconds)
            while True:
                try:
                    stage = next(worker)
                except StopIteration as stop:
                    report = stop.value
                    break
                except ExtractionError as exc:
                    raise type(exc)(f"{image_upload.filename}: {exc}") from exc
                yield {**stage, **position}
            frames.append(_read_metrics_csv(Path(report["results_csv"])))

    yield {"stage": "aggregating", "series_total": len(series)}
    all_features = _metrics_to_features(pd.concat(frames, ignore_index=True))
    return _select_expected_features(all_features, expected_features)


def _source_series(image_uploads):
    """Keep the source series, dropping MuscleMap outputs and duplicates."""
    uploads = [upload for upload in (image_uploads or []) if upload is not None]
    if not uploads:
        raise ExtractionInputError("No Axial T2w MRI series were supplied.")

    series = []
    seen = set()
    skipped = []
    for upload in uploads:
        name = (upload.filename or "").lower()
        if any(token in name for token in DERIVED_NIFTI_TOKENS):
            skipped.append(upload.filename)
            continue
        if name in seen:
            continue
        seen.add(name)
        series.append(upload)

    if not series:
        raise ExtractionInputError(
            "Every selected file is a MuscleMap output, not a source series: "
            + ", ".join(skipped[:4])
        )
    return series


def _run_worker(python_executable: str, job: dict, timeout_seconds: int):
    """Run the worker, yielding its stage events and returning its report."""
    if not WORKER_SCRIPT.is_file():
        raise ExtractorUnavailableError(
            f"MuscleMap worker script is missing: {WORKER_SCRIPT}"
        )
    try:
        process = subprocess.Popen(
            [python_executable, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise ExtractorUnavailableError(
            f"Cannot start the MuscleMap worker with {python_executable}: {exc}"
        ) from exc

    payload = None
    try:
        process.stdin.write(json.dumps(job))
        process.stdin.close()
        # Read as the worker prints so stages surface while it is still running.
        for line in process.stdout:
            line = line.strip()
            if line.startswith(STAGE_PREFIX):
                try:
                    yield json.loads(line[len(STAGE_PREFIX):])
                except ValueError:
                    pass
            elif line.startswith(RESULT_PREFIX):
                payload = line[len(RESULT_PREFIX):]
        # Each MuscleMap call already enforces its own budget; this only bounds
        # the worker's own shutdown.
        process.wait(timeout=120)
        stderr_text = process.stderr.read()
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise ExtractionExecutionError(
            f"MuscleMap extraction exceeded the {timeout_seconds}-second timeout."
        ) from exc
    finally:
        if process.poll() is None:
            process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except Exception:
                pass

    if payload is None:
        diagnostic = _diagnostic_tail(stderr_text)
        if process.returncode != 0:
            raise ExtractorUnavailableError(
                f"The MuscleMap worker exited with code {process.returncode}. "
                "Check that MUSCLEMAP_PYTHON points at the MuscleMap "
                f"environment's interpreter: {diagnostic}"
            )
        raise ExtractionExecutionError(
            f"The MuscleMap worker returned no result: {diagnostic}"
        )

    try:
        report = json.loads(payload)
    except ValueError as exc:
        raise ExtractionExecutionError(
            f"The MuscleMap worker returned malformed output: {exc}"
        ) from exc

    if report.get("status") != "ok":
        raise ExtractionExecutionError(
            report.get("message") or "MuscleMap extraction failed."
        )
    if not report.get("results_csv") or not Path(report["results_csv"]).is_file():
        raise ExtractionExecutionError(
            "MuscleMap completed but produced no metrics *_results.csv."
        )
    return report


def _resolve_command(command: str, command_name: str) -> str:
    command_path = Path(command).expanduser()
    if command_path.parent != Path("."):
        if not command_path.is_file():
            raise ExtractorUnavailableError(
                f"Configured {command_name} command does not exist: {command_path}"
            )
        return str(command_path.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise ExtractorUnavailableError(
            f"{command_name} was not found on PATH, and no MuscleMap conda "
            "environment was auto-detected. Set MUSCLEMAP_ENV to the "
            "environment directory, or set "
            f"{'MUSCLEMAP_COMMAND' if command_name == 'mm_extract_metrics' else 'MUSCLEMAP_SEGMENT_COMMAND'}"
            f" to the full path of {command_name}."
        )
    return resolved


def _resolve_python(configured: str, resolved_command: str) -> str:
    """Find the interpreter that owns MuscleMap's nibabel/monai/scripts stack."""
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise ExtractorUnavailableError(
                f"MUSCLEMAP_PYTHON does not exist: {configured_path}"
            )
        return str(configured_path.resolve())

    # mm_extract_metrics lives in <env>/Scripts on Windows and <env>/bin
    # elsewhere, so the environment's interpreter sits next to or above it.
    command_directory = Path(resolved_command).parent
    candidates = [
        command_directory.parent / "python.exe",
        command_directory / "python",
        command_directory.parent / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return sys.executable


def _read_metrics_csv(csv_path: Path) -> pd.DataFrame:
    try:
        metrics = pd.read_csv(csv_path)
    except Exception as exc:
        raise ExtractionExecutionError(
            f"Cannot read MuscleMap results {csv_path.name}: {exc}"
        ) from exc
    missing_columns = sorted(REQUIRED_METRIC_COLUMNS - set(metrics.columns))
    if missing_columns:
        raise ExtractionExecutionError(
            f"MuscleMap results are missing columns: {missing_columns}"
        )
    return metrics


def _metrics_to_features(metrics: pd.DataFrame) -> Dict[str, float]:
    metrics = metrics.copy()
    if "Undefined (%)" not in metrics.columns:
        # Two-component GMM/KMeans has no middle/undefined tissue component.
        metrics["Undefined (%)"] = 0.0
    for column in (REQUIRED_METRIC_COLUMNS | {"Undefined (%)"}) - {"Anatomy"}:
        metrics[column] = _to_number(metrics[column])
    metrics["Muscle_Code"] = _muscle_codes(metrics)

    features = {}
    missing_anatomies = []
    for code in MUSCLE_CODES:
        rows = metrics.loc[metrics["Muscle_Code"] == code]
        valid = rows[
            rows["Number of Slices"].notna()
            & rows["Volume (ml)"].notna()
            & rows["Fat (%)"].notna()
            & rows["Undefined (%)"].notna()
            & (rows["Number of Slices"] > 0)
        ]
        if valid.empty:
            missing_anatomies.append(code)
            continue
        weights = valid["Number of Slices"]
        average_volume = float(valid["Volume (ml)"].sum() / weights.sum())
        total_fat = valid["Fat (%)"] + valid["Undefined (%)"]
        average_fat = float((total_fat * weights).sum() / weights.sum())
        features[f"Muscle___aVol_{code}"] = average_volume
        features[f"Muscle___tFat_{code}"] = average_fat

    if missing_anatomies:
        raise ExtractionInputError(
            "MuscleMap segmentation lacks required lumbar anatomies: "
            + ", ".join(missing_anatomies)
        )
    if any(not math.isfinite(value) for value in features.values()):
        raise ExtractionExecutionError(
            "MuscleMap produced non-finite aVol/tFat values."
        )
    return features


def _muscle_codes(metrics: pd.DataFrame) -> pd.Series:
    """Prefer the model's fixed label values, fall back to the Anatomy text."""
    anatomy_codes = (
        metrics["Anatomy"].astype(str).str.strip().str.lower().map(ANATOMY_TO_CODE)
    )
    if "Label" not in metrics.columns:
        return anatomy_codes
    label_codes = _to_number(metrics["Label"]).map(
        lambda value: LABEL_TO_CODE.get(int(value)) if pd.notna(value) else None
    )
    return label_codes.fillna(anatomy_codes)


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _select_expected_features(
        features: Dict[str, float], expected_features: Sequence[str]
) -> Dict[str, float]:
    missing = [feature for feature in expected_features if feature not in features]
    if missing:
        raise ExtractionExecutionError(
            f"MuscleMap output does not contain selected model features: {missing}"
        )
    return {feature: float(features[feature]) for feature in expected_features}


def _ensure_non_empty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ExtractionInputError(f"{label} upload is empty.")


def _nifti_suffix(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"


def _diagnostic_tail(value: str, limit: int = 1500) -> str:
    compact = " ".join((value or "").split())
    return compact[-limit:] or "no diagnostic output"
