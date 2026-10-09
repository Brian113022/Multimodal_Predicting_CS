"""Per-series MuscleMap worker executed by the MuscleMap environment's Python.

The Flask process does not have nibabel, torch, monai or MuscleMap's ``scripts``
package, so every step that needs them runs here. The adapter in ``muscle.py``
feeds one JSON job on stdin and reads one JSON result from stdout.

This mirrors the validated batch pipeline in ``MuscleMap_code``:

* ``run_mm_segment.ipynb``                    -> plain ``mm_segment`` call
* ``fix_single_slice_and_rerun_mm_segment.py`` -> two-slice retry + collapse
* ``extract_metrics_axial_t2.py``             -> squeeze inputs, fixed GMM fallback

The DICOM workflow uses the two batch operations: ``segment_batch`` segments
every series of a case in one mm_segment run, and ``extract_batch`` computes
every series' metrics with the per-label clusterings spread over processes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


RESULT_PREFIX = "##MM_WORKER_JSON##"
STAGE_PREFIX = "##MM_WORKER_STAGE##"
# Each clustering process gets one core; otherwise every process starts its
# own BLAS/OpenMP thread pool and they fight over the same CPUs.
SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def emit(stage: str, **detail) -> None:
    """Announce the step about to run so the caller can show real progress."""
    print(STAGE_PREFIX + json.dumps({"stage": stage, **detail}, ensure_ascii=False), flush=True)


class WorkerError(RuntimeError):
    """A failure that should reach the user as an extraction error."""


def text_tail(value: str, limit: int = 1500) -> str:
    compact = " ".join((value or "").split())
    return compact[-limit:] or "no diagnostic output"


def nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def load_shape(path: Path) -> tuple:
    return tuple(int(v) for v in nib.load(str(path)).shape)


def run_command(arguments, working_directory: Path, timeout_seconds: int):
    try:
        return subprocess.run(
            [str(argument) for argument in arguments],
            cwd=str(working_directory),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerError(
            f"{arguments[0]} exceeded the {timeout_seconds}-second timeout."
        ) from exc
    except OSError as exc:
        raise WorkerError(f"Cannot start {arguments[0]}: {exc}") from exc


def find_dseg(work_dir: Path, stem: str) -> Path | None:
    # MONAI's SaveImaged always writes .nii.gz, even for a .nii input.
    expected = work_dir / f"{stem}_dseg.nii.gz"
    if expected.is_file():
        return expected
    candidates = sorted(work_dir.glob("*_dseg.nii*"))
    return candidates[0] if len(candidates) == 1 else None


def is_single_slice(shape: tuple) -> bool:
    return len(shape) == 2 or (len(shape) >= 3 and shape[2] == 1)


def save_squeezed_single_slice(source: Path, destination: Path) -> bool:
    """Drop a degenerate z axis, keeping the original header's pixdim."""
    image = nib.load(str(source))
    data = np.asanyarray(image.dataobj)
    squeezed = False
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
        squeezed = True
    elif data.ndim == 4 and data.shape[2] == 1 and data.shape[3] == 1:
        data = data[:, :, 0, 0]
        squeezed = True

    destination.parent.mkdir(parents=True, exist_ok=True)
    if squeezed:
        # The copied header keeps pixdim[3] (slice thickness); dropping it would
        # silently change every Volume (ml) value.
        nib.save(
            nib.Nifti1Image(data, image.affine, image.header.copy()),
            str(destination),
        )
    else:
        shutil.copy2(source, destination)
    return squeezed


def make_two_slice_input(source: Path, destination: Path) -> None:
    """Duplicate a single axial slice so the 3D segmentation model accepts it."""
    image = nib.load(str(source))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or data.shape[2] != 1:
        raise WorkerError(
            f"Expected a single-slice image, got shape={tuple(data.shape)}."
        )

    stacked = np.concatenate([data, data], axis=2)
    fixed = nib.Nifti1Image(stacked, image.affine, header=image.header.copy())
    fixed.header.set_data_shape(stacked.shape)
    fixed.header.set_xyzt_units("mm")
    fixed.set_qform(image.affine, code=1)
    fixed.set_sform(image.affine, code=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(fixed, str(destination))


def save_collapsed_dseg(
        temporary_dseg: Path, original_image: Path, final_dseg: Path
) -> None:
    """Keep only the first predicted slice, restoring the original geometry."""
    original = nib.load(str(original_image))
    predicted = nib.load(str(temporary_dseg))
    data = np.asanyarray(predicted.dataobj)
    if data.ndim == 2:
        collapsed = data[:, :, None]
    elif data.ndim == 3:
        collapsed = data[:, :, :1]
    else:
        raise WorkerError(
            f"Unsupported segmentation ndim={data.ndim}, shape={tuple(data.shape)}."
        )

    collapsed = np.ascontiguousarray(collapsed.astype(np.int16))
    output = nib.Nifti1Image(collapsed, original.affine, header=original.header.copy())
    output.header.set_data_shape(collapsed.shape)
    output.header.set_data_dtype(np.int16)
    output.header.set_xyzt_units("mm")
    output.set_qform(original.affine, code=int(original.header["qform_code"]) or 1)
    output.set_sform(original.affine, code=int(original.header["sform_code"]) or 1)
    nib.save(output, str(final_dseg))


def segment(job: dict, image_path: Path, work_dir: Path, report: dict) -> Path:
    stem = nifti_stem(image_path)
    timeout_seconds = int(job["timeout_seconds"])
    arguments = [
        job["segmentation_command"],
        "-i",
        image_path.name,
        "-r",
        job["segmentation_region"],
    ]
    emit("segmenting")
    completed = run_command(arguments, work_dir, timeout_seconds)
    mask_path = find_dseg(work_dir, stem)
    if completed.returncode == 0 and mask_path is not None:
        return mask_path

    diagnostic = text_tail(completed.stderr or completed.stdout)
    if not is_single_slice(load_shape(image_path)):
        raise WorkerError(f"mm_segment failed: {diagnostic}")

    # Single-slice retry, as in fix_single_slice_and_rerun_mm_segment.py.
    emit("segment_retry")
    report["used_two_slice_segmentation_fix"] = True
    fix_dir = work_dir / "_single_slice_fix"
    fix_dir.mkdir(parents=True, exist_ok=True)
    fixed_input = fix_dir / image_path.name
    make_two_slice_input(image_path, fixed_input)

    retry = run_command(
        [
            job["segmentation_command"],
            "-i",
            fixed_input.name,
            "-r",
            job["segmentation_region"],
        ],
        fix_dir,
        timeout_seconds,
    )
    temporary_dseg = find_dseg(fix_dir, stem)
    if retry.returncode != 0 or temporary_dseg is None:
        raise WorkerError(
            "mm_segment failed on the original and on the duplicated-slice "
            f"retry: {text_tail(retry.stderr or retry.stdout)}"
        )

    final_dseg = work_dir / f"{stem}_dseg.nii.gz"
    save_collapsed_dseg(temporary_dseg, image_path, final_dseg)
    return final_dseg


def readable_volume(path: Path) -> bool:
    try:
        np.asanyarray(nib.load(str(path)).dataobj)
        return True
    except Exception:
        return False


def segment_batch(job: dict) -> dict:
    """Segment every series with one ``mm_segment`` run.

    Loading torch and the model takes about 12 s per call, which used to be
    paid once per series. mm_segment accepts a comma-separated image list and
    infers each image independently, so the masks match single-image runs.
    Images are copied under unique names because every ``<stem>_dseg.nii.gz``
    is written to one directory. A series the batch did not segment falls back
    to :func:`segment`, which keeps the two-slice retry.
    """
    items = [
        (Path(item["image"]).resolve(), Path(item["work_dir"]).resolve())
        for item in job["series"]
    ]
    timeout_seconds = int(job["timeout_seconds"])
    batch_dir = Path(job["batch_dir"]).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for index, (image_path, _) in enumerate(items):
        suffix = ".nii.gz" if image_path.name.lower().endswith(".nii.gz") else ".nii"
        name = f"series_{index + 1:04d}{suffix}"
        shutil.copyfile(image_path, batch_dir / name)
        names.append(name)
    # A failure on a single-slice image stops mm_segment, so those go last
    # and cannot cost the multi-slice series their batch result.
    order = sorted(range(len(items)), key=lambda index: is_single_slice(load_shape(items[index][0])))

    emit("segmenting", series_total=len(items))
    completed = run_command(
        [
            job["segmentation_command"],
            "-i",
            ",".join(names[index] for index in order),
            "-r",
            job["segmentation_region"],
        ],
        batch_dir,
        timeout_seconds,
    )
    batch_diagnostic = (
        "" if completed.returncode == 0 else text_tail(completed.stderr or completed.stdout)
    )

    reports = []
    for index, (image_path, work_dir) in enumerate(items):
        report = {
            "used_squeeze": False,
            "used_two_slice_segmentation_fix": False,
            "used_fixed_gmm_fallback": False,
            "skipped_labels": [],
            "batch_segmentation": True,
        }
        batch_dseg = batch_dir / f"{nifti_stem(Path(names[index]))}_dseg.nii.gz"
        if batch_dseg.is_file() and readable_volume(batch_dseg):
            mask_path = work_dir / f"{nifti_stem(image_path)}_dseg.nii.gz"
            shutil.move(str(batch_dseg), str(mask_path))
        else:
            report["batch_segmentation"] = False
            if batch_diagnostic:
                report["batch_error"] = batch_diagnostic
            emit("segmenting", series=index + 1, series_total=len(items))
            mask_path = segment(job, image_path, work_dir, report)
        report.update(mask_path=str(mask_path), status="ok")
        reports.append(report)
    shutil.rmtree(batch_dir, ignore_errors=True)
    return {"status": "ok", "series": reports}


def prepare_metric_inputs(
        image_path: Path, mask_path: Path, work_dir: Path, report: dict
) -> tuple:
    """Return (cwd, image name, mask name, results base name) for extraction."""
    image_shape = load_shape(image_path)
    mask_shape = load_shape(mask_path)
    report["image_shape"] = list(image_shape)
    report["segmentation_shape"] = list(mask_shape)

    needs_squeeze = (
        (len(image_shape) >= 3 and image_shape[2] == 1)
        or (len(mask_shape) >= 3 and mask_shape[2] == 1)
        or len(image_shape) != len(mask_shape)
    )
    if not needs_squeeze:
        return work_dir, image_path.name, mask_path.name, nifti_stem(image_path)

    emit("squeezing", image_shape=list(image_shape), segmentation_shape=list(mask_shape))

    # mm_extract_metrics needs matching ndim; a (X, Y, 1) image next to the
    # (X, Y) mask MONAI writes is the common case in this dataset.
    squeeze_dir = work_dir / "_squeezed"
    squeezed_image = squeeze_dir / "input.nii.gz"
    squeezed_mask = squeeze_dir / "input_dseg.nii.gz"
    image_squeezed = save_squeezed_single_slice(image_path, squeezed_image)
    mask_squeezed = save_squeezed_single_slice(mask_path, squeezed_mask)
    report["used_squeeze"] = bool(image_squeezed or mask_squeezed)
    return squeeze_dir, squeezed_image.name, squeezed_mask.name, "input"


def number_of_slices(mask: np.ndarray) -> int:
    if mask.ndim >= 3:
        return int(np.sum(np.max(np.max(mask, axis=0), axis=0)))
    return int(mask.any())


def mm_util():
    try:
        from scripts import mm_util as module
    except ImportError:
        import mm_util as module  # type: ignore[no-redef]
    return module


def region_config(region: str) -> dict:
    if not region:
        return {"labels": []}
    util = mm_util()
    _, model_config_path = util.get_model_and_config_paths(region, None)
    return util.load_model_config(model_config_path)


def clustering_inputs(segmentations_data, image_array, components: int):
    """Return ``[(label, intensities)]`` to cluster and the labels skipped."""
    inputs = []
    skipped_labels = []
    for value in np.unique(segmentations_data):
        if value <= 0:
            continue
        mask = segmentations_data == value
        voxel_count = int(mask.sum())
        if voxel_count < components:
            skipped_labels.append(
                {
                    "label": float(value),
                    "voxel_count": voxel_count,
                    "reason": "too_few_voxels",
                }
            )
            continue
        inputs.append((value, np.reshape(image_array[mask], (-1, 1))))
    return inputs, skipped_labels


def fit_thresholds(mask_image, method: str, components: int):
    """Cluster one label's intensities into MuscleMap's tissue thresholds.

    Module-level so a process pool can run it; mm_util is imported here so
    the spawned processes load it themselves.
    """
    util = mm_util()
    labels, _ = util.apply_clustering(
        mask_image, method == "kmeans", method == "gmm", components
    )
    muscle_max, undefined_max, _, _ = util.calculate_thresholds(
        labels, mask_image, components
    )
    return muscle_max, undefined_max


def metric_rows(
        image_array, segmentations_data, pixdim, thresholds, model_config, components
) -> list:
    """The ``*_results.csv`` rows mm_extract_metrics writes for these labels."""
    util = mm_util()
    rows = []
    for value, (muscle_max, undefined_max) in thresholds:
        anatomy_side_info = ""
        for label_definition in model_config.get("labels", []):
            if label_definition.get("value") == value:
                anatomy_side_info = (
                    f"{label_definition.get('anatomy', '')} "
                    f"{label_definition.get('side', '')}"
                ).strip()
                break
        (
            muscle_percentage,
            undefined_percentage,
            fat_percentage,
            _muscle_volume,
            _fat_volume,
            total_volume,
            _muscle_mask,
            _undefined_mask,
            _fat_mask,
        ) = util.quantify_muscle_measures(
            image_array,
            segmentations_data,
            muscle_max,
            undefined_max,
            value,
            *pixdim,
        )
        row = {
            "Anatomy": anatomy_side_info,
            "Label": value,
            "Number of Slices": number_of_slices(segmentations_data == value),
            "Muscle (%)": muscle_percentage,
            "Fat (%)": fat_percentage,
        }
        # Like the CLI, two components have no undefined tissue column.
        if components == 3:
            row["Undefined (%)"] = undefined_percentage
        row["Volume (ml)"] = total_volume
        rows.append(row)
    return rows


def fixed_gmm_extract(
        image_path: Path,
        mask_path: Path,
        output_csv: Path,
        region: str,
        components: int,
) -> list:
    """In-process GMM metrics, used when the upstream CLI crashes.

    Ported from ``MuscleMap_code/extract_metrics_axial_t2.py``. It fixes three
    upstream defects: labels with fewer voxels than components abort the whole
    run, ``segmentation_image`` is only initialised when label 1 exists, and
    ``anatomy_side_info`` is undefined when a label is absent from the config.
    The probability maps and helper NIfTIs the CLI writes are byproducts that do
    not affect the CSV, so they are skipped here.
    """
    util = mm_util()
    _, image_array, _, pixdim = util.extract_image_data(str(image_path))
    _, segmentations_data, _, _ = util.extract_image_data(str(mask_path))
    inputs, skipped_labels = clustering_inputs(segmentations_data, image_array, components)
    thresholds = [
        (value, fit_thresholds(mask_image, "gmm", components))
        for value, mask_image in inputs
    ]
    results_list = metric_rows(
        image_array, segmentations_data, pixdim, thresholds,
        region_config(region), components,
    )
    if not results_list:
        raise WorkerError(
            "No segmentation label held enough voxels for GMM metrics; "
            f"skipped_labels={skipped_labels}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    return skipped_labels


def extract_batch(job: dict) -> dict:
    """Metrics for every prepared series, with all clusterings in parallel.

    Each label gets the same clustering and thresholds as mm_extract_metrics,
    which spends nearly all its time fitting a 20-start GMM per label, one
    label after another, and reloads the image for every label. Here each
    image is read once and every (series, label) fit runs in a process pool.
    The probability-map NIfTIs the CLI also writes are not used, so they are
    skipped, as in :func:`fixed_gmm_extract`.
    """
    method = job["method"]
    components = int(job["components"])
    if method not in {"gmm", "kmeans"}:
        raise WorkerError(f"Unsupported MuscleMap method for batch extraction: {method}")
    util = mm_util()
    model_config = region_config(job["region"])

    prepared = []
    tasks = []
    for index, item in enumerate(job["series"]):
        work_dir = Path(item["work_dir"]).resolve()
        report = {
            "used_squeeze": False,
            "used_two_slice_segmentation_fix": False,
            "used_fixed_gmm_fallback": False,
            "skipped_labels": [],
            "engine": "in_process_parallel",
        }
        cwd, image_name, mask_name, base = prepare_metric_inputs(
            Path(item["image"]).resolve(), Path(item["mask"]).resolve(), work_dir, report
        )
        results_csv = cwd / f"{base}_{method}_{components}component_results.csv"
        _, image_array, _, pixdim = util.extract_image_data(str(cwd / image_name))
        _, segmentations_data, _, _ = util.extract_image_data(str(cwd / mask_name))
        inputs, report["skipped_labels"] = clustering_inputs(
            segmentations_data, image_array, components
        )
        prepared.append(
            (report, results_csv, image_array, segmentations_data, pixdim, [value for value, _ in inputs])
        )
        tasks.extend((index, value, mask_image) for value, mask_image in inputs)

    workers = max(1, min(int(job.get("workers") or 1), len(tasks)))
    emit("extracting", method=method, components=components, fits=len(tasks), workers=workers)
    fits = {}
    if workers == 1:
        for index, value, mask_image in tasks:
            fits[(index, value)] = fit_thresholds(mask_image, method, components)
    elif tasks:
        for name in SINGLE_THREAD_ENV:
            os.environ[name] = "1"
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Largest labels first, so the slowest fits do not start last.
            futures = {
                pool.submit(fit_thresholds, mask_image, method, components): (index, value)
                for index, value, mask_image in sorted(tasks, key=lambda task: -len(task[2]))
            }
            for future in as_completed(futures):
                fits[futures[future]] = future.result()

    reports = []
    for index, (report, results_csv, image_array, segmentations_data, pixdim, values) in enumerate(prepared):
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        if not (segmentations_data > 0).any():
            # See extract_metrics: an out-of-anatomy block adds no slices.
            emit("no_muscle_labels", series=index + 1, series_total=len(prepared))
            pd.DataFrame(columns=["Anatomy", "Label", "Number of Slices", "Muscle (%)", "Fat (%)", "Undefined (%)", "Volume (ml)"]).to_csv(results_csv, index=False)
            report["empty_segmentation"] = True
        else:
            rows = metric_rows(
                image_array, segmentations_data, pixdim,
                [(value, fits[(index, value)]) for value in values],
                model_config, components,
            )
            if not rows:
                raise WorkerError(
                    f"Series {index + 1}: no segmentation label held enough voxels "
                    f"for {method} metrics; skipped_labels={report['skipped_labels']}"
                )
            pd.DataFrame(rows).to_csv(results_csv, index=False)
        report.update(results_csv=str(results_csv), status="ok")
        reports.append(report)
    return {"status": "ok", "series": reports, "workers": workers, "fits": len(tasks)}


def extract_metrics(
        job: dict, image_path: Path, mask_path: Path, work_dir: Path, report: dict
) -> Path:
    method = job["method"]
    components = int(job["components"])
    cwd, image_name, mask_name, base = prepare_metric_inputs(
        image_path, mask_path, work_dir, report
    )
    results_csv = cwd / f"{base}_{method}_{components}component_results.csv"

    if not (np.asanyarray(nib.load(str(cwd / mask_name)).dataobj) > 0).any():
        # A block outside the model's anatomy (e.g. the T12-L1 stack of a
        # multi-level axial series) segments to nothing. It adds no slices to
        # any muscle, exactly like the series without a metrics CSV that
        # combine_musclemap_case_metrics.py skipped, so it must not fail the case.
        emit("no_muscle_labels")
        pd.DataFrame(columns=["Anatomy", "Label", "Number of Slices", "Muscle (%)", "Fat (%)", "Undefined (%)", "Volume (ml)"]).to_csv(results_csv, index=False)
        report["empty_segmentation"] = True
        return results_csv

    emit("extracting", method=method, components=components)
    completed = run_command(
        [
            job["command"],
            "-m",
            method,
            "-r",
            job["region"],
            "-i",
            image_name,
            "-s",
            mask_name,
            "-c",
            str(components),
        ],
        cwd,
        int(job["timeout_seconds"]),
    )
    if completed.returncode == 0 and results_csv.is_file():
        return results_csv

    diagnostic = text_tail(completed.stderr or completed.stdout)
    if method != "gmm":
        raise WorkerError(f"mm_extract_metrics failed: {diagnostic}")

    emit("fallback")
    try:
        skipped = fixed_gmm_extract(
            cwd / image_name, cwd / mask_name, results_csv, job["region"], components
        )
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            f"mm_extract_metrics failed ({diagnostic}) and the in-process GMM "
            f"fallback also failed: {exc!r}"
        ) from exc

    report["used_fixed_gmm_fallback"] = True
    report["skipped_labels"] = skipped
    report["cli_error"] = diagnostic
    return results_csv


def run(job: dict) -> dict:
    operation = job.get("operation", "full")
    if operation == "segment_batch":
        emit("reading")
        return segment_batch(job)
    if operation == "extract_batch":
        emit("reading")
        return extract_batch(job)

    work_dir = Path(job["work_dir"]).resolve()
    image_path = Path(job["image"]).resolve()
    report = {
        "used_squeeze": False,
        "used_two_slice_segmentation_fix": False,
        "used_fixed_gmm_fallback": False,
        "skipped_labels": [],
    }

    emit("reading")
    if operation not in {"full", "segment", "extract"}:
        raise WorkerError("Unknown MuscleMap worker operation.")
    mask_argument = job.get("mask")
    if mask_argument:
        mask_path = Path(mask_argument).resolve()
    else:
        if operation == "extract":
            raise WorkerError("Feature-only extraction requires a prepared segmentation mask.")
        mask_path = segment(job, image_path, work_dir, report)

    if operation == "segment":
        report.update(mask_path=str(mask_path), status="ok")
        return report

    results_csv = extract_metrics(job, image_path, mask_path, work_dir, report)
    report["results_csv"] = str(results_csv)
    report["status"] = "ok"
    return report


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
        payload = run(job)
    except WorkerError as exc:
        payload = {"status": "failed", "message": str(exc)}
    except Exception as exc:
        payload = {
            "status": "failed",
            "message": f"{exc!r}",
            "traceback": text_tail(traceback.format_exc(), 2000),
        }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
