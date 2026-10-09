"""Extract the full FAE Radiomics feature family for a folder of patients.

The Radiomics upload on the prediction page calculates, for one case, only the
features the loaded model selected. This module walks a folder of patient
folders, each holding one MRI and its VOI mask, runs every image type and
feature class of the FAE YAML on each, and writes one CSV row per patient in
the layout of ``data/processed/radiomics_features_new.csv``: ``Name`` (the
patient folder) followed by the ``MRI_*`` feature columns.

Every patient goes through the same ``fae_worker.py`` as the upload, one worker
process per patient and several at a time.
"""

import csv
import fnmatch
import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import yaml

from .errors import ExtractionError, ExtractionInputError, ExtractorUnavailableError
from .radiomics import RESULT_PREFIX, STAGE_PREFIX, WORKER_SCRIPT
from .worker import run_worker

NIFTI_SUFFIXES = (".nii", ".nii.gz")
DEFAULT_MASK_PATTERN = "*segmentation*"
# root/centre/patient needs two levels; the rest is headroom.
MAX_SEARCH_DEPTH = 4
# fae_worker.py names features for the model bundle. The training CSV has no
# modality prefix, only MRI_.
WORKER_FEATURE_PREFIX = "Radiomics___"
# The CSV is rewritten this often during a run, so stopping the server keeps
# the patients that were already finished.
CHECKPOINT_SECONDS = 30

LOG_COLUMNS = [
    "Name", "Group", "Status", "Image", "VOI", "VOI_voxels", "Features",
    "Seconds", "Message", "Folder",
]


def scan_patient_folders(root, mask_pattern=DEFAULT_MASK_PATTERN):
    """Find every folder below ``root`` that directly holds NIfTI files.

    ``mask_pattern`` holds case-insensitive globs, comma separated, that pick
    the VOI among a folder's files; the other files are image candidates. A
    folder with exactly one of each is ``ready``. Any other folder is
    ``check``, with the best guesses filled in for the user to confirm.

    Returns the resolved root and the cases in natural name order.
    """
    root_path = Path(str(root or "").strip().strip('"')).expanduser()
    if not str(root or "").strip():
        raise ExtractionInputError("Enter the folder that holds the patient folders.")
    if not root_path.is_dir():
        raise ExtractionInputError(f"Folder not found: {root_path}")
    root_path = root_path.resolve()
    patterns = parse_patterns(mask_pattern)

    found = []
    _collect(root_path, 0, found)
    cases = [
        _describe_case(folder, files, root_path, patterns)
        for folder, files in found
    ]
    names = Counter(case["name"] for case in cases)
    for case in cases:
        if names[case["name"]] > 1:
            case["issues"].append(
                f"{names[case['name']]} folders are named {case['name']}; "
                "their CSV rows would share one Name."
            )
            case["status"] = "check"
    return root_path, cases


def parse_patterns(mask_pattern) -> list[str]:
    patterns = [
        part.strip().lower()
        for part in re.split(r"[,;]", str(mask_pattern or ""))
        if part.strip()
    ]
    if not patterns:
        raise ExtractionInputError("Enter a VOI file name pattern, e.g. *segmentation*.")
    return patterns


def resolve_selection(cases, selection) -> list[dict]:
    """Pair the user's per-patient file choices with the scanned cases.

    Only files the scan found in that patient's folder are accepted, so a
    request cannot point the worker anywhere else.
    """
    by_id = {case["id"]: case for case in cases}
    chosen, problems = [], []
    for item in selection if isinstance(selection, list) else []:
        if not isinstance(item, dict):
            continue
        case = by_id.get(str(item.get("id")))
        if case is None:
            problems.append(f"{item.get('id')}: folder not found any more, scan again.")
            continue
        image, mask = item.get("image"), item.get("mask")
        if image not in case["files"] or mask not in case["files"]:
            problems.append(f"{case['id']}: choose an image and a VOI from its folder.")
        elif image == mask:
            problems.append(f"{case['id']}: the image and the VOI are the same file.")
        else:
            chosen.append({**case, "image": image, "mask": mask})
    if problems:
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise ExtractionInputError(" ".join(problems[:5]) + more)
    if not chosen:
        raise ExtractionInputError("Tick at least one patient to extract.")
    return chosen


def load_full_parameters(config_path, *, label=None, normalize_scale=None) -> dict:
    """Every image type and feature class of the FAE YAML, in FAE's order."""
    config_path = Path(config_path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
    except Exception as exc:
        raise ExtractorUnavailableError(
            f"Cannot read FAE Radiomics YAML {config_path}: {exc}"
        ) from exc
    if not isinstance(params, dict) or not params.get("imageType"):
        raise ExtractorUnavailableError(
            f"FAE Radiomics YAML {config_path} enables no image type."
        )
    setting = dict(params.get("setting") or {})
    if label is not None:
        setting["label"] = label
    if normalize_scale is not None:
        setting["normalizeScale"] = normalize_scale
    # FAE writes its YAML with yaml.dump, which sorts keys, and PyRadiomics
    # runs image types and classes in that order: the training CSV's order.
    return {
        "imageType": {
            name: options or {}
            for name, options in sorted(params["imageType"].items())
        },
        "featureClass": dict(sorted((params.get("featureClass") or {}).items())),
        "setting": setting,
    }


def default_output_path(root) -> Path:
    return Path(root) / f"radiomics_features_{datetime.now():%Y%m%d-%H%M%S}.csv"


def resolve_output_path(value, root) -> Path:
    """The CSV to write. Relative paths are taken from ``root``; nothing is
    overwritten."""
    text = str(value or "").strip().strip('"')
    path = Path(text).expanduser() if text else default_output_path(root)
    if not path.is_absolute():
        path = Path(root) / path
    if path.is_dir():
        path = path / default_output_path(root).name
    if path.suffix.lower() != ".csv":
        raise ExtractionInputError("The output file name must end in .csv.")
    existing = [candidate for candidate in output_paths(path) if candidate.exists()]
    if existing:
        raise ExtractionInputError(
            f"{existing[0]} already exists; choose another output file name."
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExtractionInputError(f"Cannot create {path.parent}: {exc}") from exc
    return path.resolve()


def output_paths(features_path: Path) -> tuple[Path, Path, Path]:
    """The features CSV and the log CSV and settings JSON written beside it."""
    stem = features_path.stem
    return (
        features_path,
        features_path.with_name(f"{stem}_log.csv"),
        features_path.with_name(f"{stem}_settings.json"),
    )


def describe_stage(event: dict) -> str:
    stage = event.get("stage")
    if stage == "extracting":
        return (
            f"{event.get('image_type')} "
            f"({event.get('filter_index')}/{event.get('filter_total')})"
        )
    return {
        "reading": "Reading image and VOI",
        "mapping": "Collecting features",
    }.get(stage, str(stage or ""))


class BatchJob:
    """One extraction over the selected patients, on a background thread.

    Up to ``workers`` patients run at once, each in its own ``fae_worker.py``
    process. The page follows the run through :meth:`snapshot`.
    """

    def __init__(
            self,
            cases,
            parameters: dict,
            output_path: Path,
            *,
            root,
            python_executable: str,
            workers: int,
            timeout_seconds: int,
            config_path,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.root = str(root)
        self.parameters = parameters
        self.config_path = str(config_path)
        self.python_executable = python_executable
        self.workers = max(1, int(workers))
        self.timeout_seconds = int(timeout_seconds)
        self.output_path, self.log_path, self.settings_path = output_paths(
            Path(output_path)
        )
        self.cases = [
            {
                "id": case["id"],
                "name": case["name"],
                "group": case["group"],
                "folder": case["folder"],
                "image": case["image"],
                "mask": case["mask"],
                "status": "queued",
                "stage": "",
                "seconds": None,
                "voxels": None,
                "features": None,
                "message": "",
            }
            for case in cases
        ]
        self.state = "running"
        self.message = ""
        self.rows_written = 0
        self.created_at = time.time()
        self.finished_at = None
        self._versions = None
        self._values = {}
        # Insertion-ordered set: the first patient fixes the column order.
        self._columns = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._cancel = threading.Event()
        self._processes = {}
        self._timed_out = set()
        self._last_checkpoint = time.time()
        self._thread = threading.Thread(
            target=self._run, name=f"radiomics-batch-{self.id}", daemon=True
        )

    @property
    def running(self) -> bool:
        return self.state in ("running", "cancelling")

    def start(self) -> "BatchJob":
        self._thread.start()
        return self

    def cancel(self) -> None:
        with self._lock:
            if not self.running:
                return
            self._cancel.set()
            self.state = "cancelling"
            processes = [process for process, _ in self._processes.values()]
        for process in processes:
            _kill(process)

    def wait(self, timeout=None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            cases = [dict(case) for case in self.cases]
            state, message = self.state, self.message
            columns = len(self._columns)
        counts = Counter(case["status"] for case in cases)
        finished = counts["done"] + counts["failed"] + counts["cancelled"]
        elapsed = (self.finished_at or time.time()) - self.created_at
        remaining = len(cases) - finished
        eta = (
            elapsed / finished * remaining
            if state == "running" and finished and remaining
            else None
        )
        return {
            "id": self.id,
            "state": state,
            "message": message,
            "root": self.root,
            "output_csv": str(self.output_path),
            "log_csv": str(self.log_path),
            "settings_json": str(self.settings_path),
            "rows_written": self.rows_written,
            "workers": self.workers,
            "total": len(cases),
            "counts": dict(counts),
            "columns": columns,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta) if eta is not None else None,
            "cases": [
                {key: value for key, value in case.items() if key != "folder"}
                for case in cases
            ],
        }

    def _run(self) -> None:
        try:
            with ThreadPoolExecutor(
                    max_workers=self.workers,
                    thread_name_prefix=f"radiomics-{self.id}",
            ) as pool:
                pending = {
                    pool.submit(self._run_case, index)
                    for index in range(len(self.cases))
                }
                while pending:
                    _, pending = wait(pending, timeout=1)
                    self._stop_overdue_workers()
                    if time.time() - self._last_checkpoint >= CHECKPOINT_SECONDS:
                        self._write_outputs(final=False)
            self._write_outputs(final=True)
            with self._lock:
                self.state = "cancelled" if self._cancel.is_set() else "finished"
        except Exception as exc:
            # A thread has nobody to raise to; the page shows this instead.
            with self._lock:
                self.state = "failed"
                self.message = f"The batch stopped: {exc}"
        finally:
            self.finished_at = time.time()

    def _run_case(self, index: int) -> None:
        case = self.cases[index]
        if self._cancel.is_set():
            self._update(index, status="cancelled", message="Not started.")
            return
        started = time.time()
        self._update(index, status="running", stage="Starting worker")
        folder = Path(case["folder"])
        events = run_worker(
            self.python_executable,
            WORKER_SCRIPT,
            {
                "image": str(folder / case["image"]),
                "mask": str(folder / case["mask"]),
                "parameters": self.parameters,
            },
            stage_prefix=STAGE_PREFIX,
            result_prefix=RESULT_PREFIX,
            timeout_seconds=self.timeout_seconds,
            label="PyRadiomics",
            on_process=lambda process: self._register(index, process),
        )
        try:
            while True:
                try:
                    event = next(events)
                except StopIteration as stop:
                    report = stop.value
                    break
                self._update(index, stage=describe_stage(event))
        except ExtractionError as exc:
            self._fail(index, started, str(exc))
            return
        except Exception as exc:
            self._fail(index, started, f"Unexpected failure: {exc!r}")
            return
        finally:
            with self._lock:
                self._processes.pop(index, None)

        values = {
            (name[len(WORKER_FEATURE_PREFIX):]
             if name.startswith(WORKER_FEATURE_PREFIX) else name): value
            for name, value in report.get("features", {}).items()
        }
        with self._lock:
            self._values[index] = values
            for column in values:
                self._columns.setdefault(column, None)
            self._versions = self._versions or report.get("versions")
            case.update(
                status="done",
                stage="",
                seconds=round(time.time() - started, 1),
                voxels=report.get("voxel_count"),
                features=len(values),
                message="",
            )

    def _register(self, index: int, process) -> None:
        with self._lock:
            self._processes[index] = (process, time.time())
        if self._cancel.is_set():
            _kill(process)

    def _update(self, index: int, **changes) -> None:
        with self._lock:
            self.cases[index].update(changes)

    def _fail(self, index: int, started: float, message: str) -> None:
        with self._lock:
            status = "failed"
            if index in self._timed_out:
                message = f"Stopped after the {self.timeout_seconds}-second limit."
            elif self._cancel.is_set():
                status, message = "cancelled", "Stopped by Cancel."
            self.cases[index].update(
                status=status,
                stage="",
                seconds=round(time.time() - started, 1),
                message=message,
            )

    def _stop_overdue_workers(self) -> None:
        now = time.time()
        with self._lock:
            overdue = [
                (index, process)
                for index, (process, started) in self._processes.items()
                if now - started > self.timeout_seconds
                and index not in self._timed_out
            ]
            self._timed_out.update(index for index, _ in overdue)
        for _, process in overdue:
            _kill(process)

    def _write_outputs(self, final: bool) -> None:
        with self._write_lock:
            with self._lock:
                self._last_checkpoint = time.time()
                columns = list(self._columns)
                rows = [
                    [self.cases[index]["name"],
                     *(self._values[index].get(column) for column in columns)]
                    for index in sorted(self._values)
                ]
                log_rows = [
                    [case["name"], case["group"], case["status"], case["image"],
                     case["mask"], case["voxels"], case["features"],
                     case["seconds"], case["message"], case["folder"]]
                    for case in self.cases
                ]
                settings = {
                    "created": datetime.fromtimestamp(self.created_at).isoformat(
                        timespec="seconds"
                    ),
                    "root": self.root,
                    "config": self.config_path,
                    "python": self.python_executable,
                    "versions": self._versions,
                    "parameters": self.parameters,
                }
            try:
                self._write_files(columns, rows, log_rows, settings)
            except OSError as exc:
                if not final:
                    # Typically the CSV is open in Excel; the next checkpoint
                    # or the final write tries again.
                    self._note(f"Could not update {self.output_path.name}: {exc}")
                    return
                # Keep the results under a fresh name rather than lose them.
                self.output_path, self.log_path, self.settings_path = output_paths(
                    self.output_path.with_name(
                        f"{self.output_path.stem}_{datetime.now():%H%M%S}.csv"
                    )
                )
                self._write_files(columns, rows, log_rows, settings)
                self._note(
                    f"The first output file could not be written ({exc}); "
                    f"saved as {self.output_path.name} instead."
                )
            self.rows_written = len(rows)

    def _write_files(self, columns, rows, log_rows, settings) -> None:
        if rows:
            _write_csv(self.output_path, ["Name", *columns], rows)
        _write_csv(self.log_path, LOG_COLUMNS, log_rows)
        _write_text(
            self.settings_path,
            json.dumps(settings, indent=2, ensure_ascii=False),
        )

    def _note(self, message: str) -> None:
        with self._lock:
            self.message = message


def _collect(directory: Path, depth: int, found: list) -> None:
    try:
        entries = sorted(directory.iterdir(), key=_natural_key)
    except OSError:
        return
    # Hidden files include macOS "._name.nii" resource forks, which are not
    # images.
    files = [
        entry.name
        for entry in entries
        if entry.name.lower().endswith(NIFTI_SUFFIXES)
        and not entry.name.startswith(".")
        and entry.is_file()
    ]
    if files:
        # A patient folder; anything below it is not another patient.
        found.append((directory, files))
        return
    if depth >= MAX_SEARCH_DEPTH:
        return
    for entry in entries:
        if not entry.name.startswith((".", "$")) and entry.is_dir():
            _collect(entry, depth + 1, found)


def _describe_case(folder: Path, files: list, root: Path, patterns: list) -> dict:
    masks = [
        name for name in files
        if any(fnmatch.fnmatchcase(name.lower(), pattern) for pattern in patterns)
    ]
    images = [name for name in files if name not in masks]
    relative = folder.relative_to(root)
    parent = relative.parent.as_posix()
    issues = []
    if not masks:
        issues.append("No file matches the VOI pattern; choose the VOI.")
    if not images:
        issues.append("No image besides the VOI; choose the image.")
    if len(images) > 1:
        issues.append(f"{len(images)} candidate images; confirm which one.")
    if len(masks) > 1:
        issues.append(f"{len(masks)} files match the VOI pattern; confirm which one.")
    return {
        "id": relative.as_posix(),
        "name": folder.name,
        "group": "" if parent == "." else parent,
        "folder": str(folder),
        "files": files,
        "image": _guess_image(images),
        "mask": masks[0] if masks else None,
        "status": "check" if issues else "ready",
        "issues": issues,
    }


def _guess_image(images: list):
    """Prefer a Sagittal T1 series by name; ties keep the natural order."""
    if not images:
        return None
    return max(
        images,
        key=lambda name: ("t1" in name.lower()) + ("sag" in name.lower()),
    )


def _natural_key(path: Path):
    """``2_x`` before ``10_x``: digit runs compare as numbers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _write_csv(path: Path, header: list, rows: list) -> None:
    # Written beside the target and swapped in, so a reader never sees half a
    # file. Plain UTF-8 without a BOM, like radiomics_features_new.csv.
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _kill(process) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass
