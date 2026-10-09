"""Batch Radiomics page: every FAE feature for a folder of patients, as CSV.

The page scans a folder of patient folders, lets the user confirm each
patient's image and VOI, and starts one background job that extracts the full
1,688-feature family. The browser polls the job, so a long run survives a page
reload; only one job runs at a time because each already uses several cores.
"""

import math
import os
import threading

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from .extractors import ExtractionInputError, ExtractorUnavailableError
from .extractors.radiomics_batch import (
    DEFAULT_MASK_PATTERN,
    BatchJob,
    default_output_path,
    load_full_parameters,
    resolve_output_path,
    resolve_selection,
    scan_patient_folders,
)
from .extractors.worker import resolve_python


bp = Blueprint("radiomics_batch", __name__)

_jobs: dict[str, BatchJob] = {}
_jobs_lock = threading.Lock()


def _default_workers() -> int:
    # One patient takes 5-10 s on one core, so half the logical CPUs keeps the
    # machine usable while a few hundred patients run in minutes.
    return max(1, min(8, (os.cpu_count() or 2) // 2))


@bp.get("/radiomics-batch")
def page():
    return render_template("radiomics_batch.html")


@bp.get("/api/radiomics-batch/defaults")
def defaults():
    config_path = current_app.config["FAE_RADIOMICS_CONFIG"]
    try:
        parameters = load_full_parameters(config_path)
    except ExtractorUnavailableError as exc:
        return _error("config_unavailable", str(exc), 503)
    active = _active_job()
    return jsonify({
        "config_path": str(config_path),
        "settings": parameters["setting"],
        "image_types": list(parameters["imageType"]),
        "feature_classes": list(parameters["featureClass"]),
        "mask_pattern": DEFAULT_MASK_PATTERN,
        "workers": _default_workers(),
        "max_workers": os.cpu_count() or 1,
        "radiomics_python": current_app.config["RADIOMICS_PYTHON"],
        "active_job": active.id if active else None,
    })


@bp.post("/api/radiomics-batch/scan")
def scan():
    payload = request.get_json(silent=True) or {}
    try:
        root, cases = scan_patient_folders(
            payload.get("folder"), payload.get("mask_pattern") or DEFAULT_MASK_PATTERN
        )
    except ExtractionInputError as exc:
        return _error("invalid_folder", str(exc), 422)
    return jsonify({
        "root": str(root),
        "cases": cases,
        "default_output": str(default_output_path(root)),
    })


@bp.post("/api/radiomics-batch/jobs")
def start_job():
    if _active_job() is not None:
        return _error(
            "job_running", "Another batch is still running; wait for it or cancel it.", 409
        )
    payload = request.get_json(silent=True) or {}
    config = current_app.config
    try:
        # The folder is scanned again so the job only ever reads files the
        # scan found, whatever the request says.
        root, cases = scan_patient_folders(
            payload.get("folder"), payload.get("mask_pattern") or DEFAULT_MASK_PATTERN
        )
        selected = resolve_selection(cases, payload.get("cases"))
        workers = _bounded_int(
            payload.get("workers"), "Parallel workers", 1, os.cpu_count() or 1
        )
        label = _bounded_int(payload.get("label"), "VOI label", 1, 65535)
        normalize_scale = _positive_float(payload.get("normalize_scale"), "normalizeScale")
        parameters = load_full_parameters(
            config["FAE_RADIOMICS_CONFIG"], label=label, normalize_scale=normalize_scale
        )
        python_executable = resolve_python(config["RADIOMICS_PYTHON"])
        output_path = resolve_output_path(payload.get("output_csv"), root)
    except ExtractionInputError as exc:
        return _error("invalid_request", str(exc), 422)
    except ExtractorUnavailableError as exc:
        return _error("extractor_unavailable", str(exc), 503)

    job = BatchJob(
        selected,
        parameters,
        output_path,
        root=root,
        python_executable=python_executable,
        workers=workers or _default_workers(),
        timeout_seconds=config["EXTRACTION_TIMEOUT_SECONDS"],
        config_path=config["FAE_RADIOMICS_CONFIG"],
    )
    with _jobs_lock:
        if any(existing.running for existing in _jobs.values()):
            return _error(
                "job_running", "Another batch is still running; wait for it or cancel it.", 409
            )
        _jobs[job.id] = job.start()
    return jsonify(job.snapshot()), 202


@bp.get("/api/radiomics-batch/jobs/<job_id>")
def job_status(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return _job_not_found()
    return jsonify(job.snapshot())


@bp.post("/api/radiomics-batch/jobs/<job_id>/cancel")
def cancel_job(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return _job_not_found()
    job.cancel()
    return jsonify(job.snapshot())


@bp.get("/api/radiomics-batch/jobs/<job_id>/csv")
def download_csv(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return _job_not_found()
    if not job.rows_written or not job.output_path.is_file():
        return _error("csv_not_ready", "No patient has finished yet.", 404)
    return send_file(
        job.output_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=job.output_path.name,
    )


def _active_job() -> BatchJob | None:
    with _jobs_lock:
        return next((job for job in _jobs.values() if job.running), None)


def _bounded_int(value, label: str, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ExtractionInputError(f"{label} must be a whole number.") from None
    if not minimum <= number <= maximum:
        raise ExtractionInputError(f"{label} must be between {minimum} and {maximum}.")
    return number


def _positive_float(value, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ExtractionInputError(f"{label} must be a number.") from None
    if not (math.isfinite(number) and number > 0):
        raise ExtractionInputError(f"{label} must be a finite number greater than 0.")
    return number


def _job_not_found():
    return _error(
        "job_not_found",
        "This batch job is unknown; the server may have been restarted.",
        404,
    )


def _error(code: str, message: str, status: int):
    return jsonify({"error": code, "message": message}), status
