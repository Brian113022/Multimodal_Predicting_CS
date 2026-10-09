"""TotalSegmentator worker for level-specific vertebral-body Radiomics VOIs.

The worker runs in the Python environment that owns TotalSegmentator. It runs
``total_mr`` for the combined vertebral-body mask and ``vertebrae_mr`` for
individual levels, then computes ``body AND union(selected levels)`` on the
uploaded MRI grid.

On a GPU both tasks run at the same time; each needs about 2 GB. Both write a
single multilabel file (``--ml``): saving the 25 per-vertebra files of
``vertebrae_mr`` alone took about 5 s, and the masks are the same.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


RESULT_PREFIX = "##TOTALSEG_WORKER_JSON##"
STAGE_PREFIX = "##TOTALSEG_WORKER_STAGE##"
SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
]


class WorkerError(RuntimeError):
    """A failure that should reach the web client as an extraction error."""


_EMIT_LOCK = threading.Lock()


def emit(stage: str, **detail) -> None:
    # Both TotalSegmentator tasks can report at once; keep each line whole.
    with _EMIT_LOCK:
        print(
            STAGE_PREFIX + json.dumps({"stage": stage, **detail}, ensure_ascii=False),
            flush=True,
        )


def _totalsegmentator_version() -> str:
    try:
        from importlib.metadata import version

        return version("TotalSegmentator")
    except Exception:
        return ""


def _resolve_device(job: dict):
    """Resolve the requested device before launching two expensive models."""
    import torch

    requested = (job.get("device") or "").strip().lower()
    if requested.startswith("cuda"):
        requested = "gpu" + requested[4:]
    available = bool(torch.cuda.is_available())
    trace = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
        "cuda_available": available,
    }
    if available:
        try:
            index = 0
            if requested.startswith("gpu:"):
                index = int(requested.split(":", 1)[1])
            properties = torch.cuda.get_device_properties(index)
            trace["gpu_name"] = properties.name
            trace["gpu_total_mb"] = round(properties.total_memory / 2**20)
        except Exception:
            trace["gpu_name"] = ""

    if requested == "cpu":
        return "cpu", "cpu", trace
    if available:
        cli_device = requested if requested.startswith("gpu") else "gpu"
        return "cuda", cli_device, trace
    if job.get("require_gpu"):
        raise WorkerError(
            "GPU required but torch cannot see one. Installed torch is "
            f"{torch.__version__} (CUDA build: {torch.version.cuda or 'none'}). "
            "Install a CUDA-enabled torch build, or set "
            "TOTALSEGMENTATOR_REQUIRE_GPU=0 to allow CPU segmentation."
        )
    return "cpu", "cpu", trace


class _GpuSampler:
    """Poll nvidia-smi without creating another CUDA context."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.baseline_mb = None
        self.peak_mb = 0
        self.peak_util = 0
        self.samples = 0

    def _read(self):
        try:
            output = subprocess.run(
                SMI_QUERY, capture_output=True, text=True, timeout=10
            ).stdout.strip().splitlines()
        except Exception:
            return None
        if not output:
            return None
        try:
            util, used = (part.strip() for part in output[0].split(","))
            return int(util), int(used)
        except ValueError:
            return None

    def _loop(self):
        while not self._stop.is_set():
            reading = self._read()
            if reading is not None:
                util, used = reading
                if self.baseline_mb is None:
                    self.baseline_mb = used
                self.peak_mb = max(self.peak_mb, used)
                self.peak_util = max(self.peak_util, util)
                self.samples += 1
            self._stop.wait(self.interval)

    def start(self):
        first = self._read()
        if first is not None:
            self.baseline_mb = first[1]
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return None
        return {
            "peak_util_percent": self.peak_util,
            "peak_used_mb": self.peak_mb,
            "baseline_used_mb": self.baseline_mb,
            "peak_delta_mb": max(0, self.peak_mb - (self.baseline_mb or 0)),
            "samples": self.samples,
        }


def _base_command(job: dict, cli_device: str) -> list:
    return [
        sys.executable,
        "-m",
        "totalsegmentator.bin.TotalSegmentator",
        "-i",
        job["image"],
        "--device",
        cli_device,
        "--nr_thr_resamp",
        str(max(1, int(job.get("nr_threads_resampling", 1)))),
        "--nr_thr_saving",
        str(max(1, int(job.get("nr_threads_saving", 2)))),
    ]


def _body_command(job: dict, output_path: Path, cli_device: str) -> list:
    return _base_command(job, cli_device) + [
        "-o",
        str(output_path),
        "--ml",
        "--task",
        "total_mr",
        "--roi_subset",
        "vertebrae",
    ]


def _levels_command(job: dict, output_path: Path, cli_device: str) -> list:
    return _base_command(job, cli_device) + [
        "-o",
        str(output_path),
        "--ml",
        "--task",
        "vertebrae_mr",
    ]


def _run_command(command: list, environment: dict, deadline: float, label: str) -> None:
    """Run one TotalSegmentator task while retaining a useful error tail."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            env=environment,
        )
    except OSError as exc:
        raise WorkerError(f"Cannot start TotalSegmentator {label}: {exc}") from exc

    lines: "queue.Queue" = queue.Queue()

    def reader():
        buffer = ""
        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", "replace")
                buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    lines.put(line)
        except Exception:
            pass
        finally:
            if buffer:
                lines.put(buffer)
            lines.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    tail = deque(maxlen=80)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                raise WorkerError(
                    "TotalSegmentator exceeded the configured segmentation timeout "
                    f"while running {label}. Increase SEGMENTATION_TIMEOUT_SECONDS."
                )
            try:
                line = lines.get(timeout=min(remaining, 5))
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()[:800]
            if not line:
                continue
            tail.append(line)
            lowered = line.lower()
            if "download" in lowered or "weights" in lowered:
                emit("weights", task=label, detail=line)
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise WorkerError(f"TotalSegmentator {label} did not shut down.") from exc
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.stdout.close()
        except Exception:
            pass

    if process.returncode != 0:
        detail = " ".join(tail)[-2500:]
        raise WorkerError(
            f"TotalSegmentator {label} exited with code {process.returncode}: {detail}"
        )


def _run_tasks(tasks: list, environment: dict, deadline: float, parallel: bool) -> None:
    """Run ``[(command, label)]`` one after another, or all at once."""
    if not parallel:
        for command, label in tasks:
            _run_command(command, environment, deadline, label)
        return
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [
            pool.submit(_run_command, command, environment, deadline, label)
            for command, label in tasks
        ]
        for future in futures:
            future.result()


def _task_label(task: str, name: str) -> int:
    """Label value of ``name`` in a TotalSegmentator task's multilabel output."""
    from totalsegmentator.map_to_binary import class_map

    for value, class_name in class_map[task].items():
        if class_name == name:
            return int(value)
    raise WorkerError(f"TotalSegmentator {task} has no class named {name}.")


def _load_output(path: Path, task: str):
    import nibabel as nib

    if not path.is_file():
        raise WorkerError(f"TotalSegmentator {task} did not produce {path.name}.")
    return nib.load(str(path), mmap=False)


def _labels_on_source_grid(label_image, source_image):
    """Return a label array on the source grid, using nearest-neighbour if needed."""
    import numpy as np

    target_shape = tuple(source_image.shape[:3])
    same_shape = tuple(label_image.shape[:3]) == target_shape
    same_affine = np.allclose(label_image.affine, source_image.affine, atol=1e-4)
    resampled = not (same_shape and same_affine)
    if resampled:
        from nibabel.processing import resample_from_to

        label_image = resample_from_to(
            label_image,
            (target_shape, source_image.affine),
            order=0,
            mode="constant",
            cval=0,
        )
    data = np.asanyarray(label_image.dataobj)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    return data, resampled


def _intersect_body_and_levels(body_mask, level_masks: dict):
    """Compute body AND union(levels), retaining each level for audit output."""
    import numpy as np

    selected = np.zeros_like(body_mask, dtype=bool)
    body_by_level = {}
    for code, level_mask in level_masks.items():
        selected |= level_mask
        body_by_level[code] = body_mask & level_mask
    return body_mask & selected, selected, body_by_level


def _axis_of(codes, positive: str, negative: str):
    for index, code in enumerate(codes):
        if code == positive:
            return index, True
        if code == negative:
            return index, False
    return None, None


def _render_preview(volume, mask, codes, levels, output_path, title="VOI") -> bool:
    """Draw the mask on the sagittal slice containing the most selected voxels."""
    import numpy as np
    from PIL import Image, ImageDraw

    lateral_axis, _ = _axis_of(codes, "L", "R")
    anterior_axis, anterior_positive = _axis_of(codes, "A", "P")
    superior_axis, superior_positive = _axis_of(codes, "S", "I")
    if None in (lateral_axis, anterior_axis, superior_axis):
        return False

    counts = mask.sum(axis=tuple(axis for axis in range(3) if axis != lateral_axis))
    if not counts.any():
        return False
    index = int(np.argmax(counts))
    plane = np.take(volume, index, axis=lateral_axis).astype(np.float32)
    plane_mask = np.take(mask, index, axis=lateral_axis)
    remaining = [axis for axis in range(3) if axis != lateral_axis]
    if remaining.index(superior_axis) == 1:
        plane, plane_mask = plane.T, plane_mask.T
    if superior_positive:
        plane, plane_mask = plane[::-1], plane_mask[::-1]
    if anterior_positive:
        plane, plane_mask = plane[:, ::-1], plane_mask[:, ::-1]

    low, high = np.percentile(plane, (1, 99))
    grey = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((grey * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    rgb[plane_mask] = (
        0.55 * rgb[plane_mask] + np.array([115, 0, 0])
    ).clip(0, 255)
    edge = plane_mask & ~(
        np.pad(plane_mask, 1)[:-2, 1:-1]
        & np.pad(plane_mask, 1)[2:, 1:-1]
        & np.pad(plane_mask, 1)[1:-1, :-2]
        & np.pad(plane_mask, 1)[1:-1, 2:]
    )
    rgb[edge] = (255, 80, 80)

    picture = Image.fromarray(rgb)
    caption = (
        f"{title}: {', '.join(levels)} | slice {index + 1}/{volume.shape[lateral_axis]}"
    )
    draw = ImageDraw.Draw(picture)
    draw.rectangle([0, 0, picture.width, 34], fill=(0, 0, 0))
    draw.text((8, 9), caption, fill=(255, 190, 190))
    picture.save(output_path, quality=90)
    return True


def _save_mask(nib, np, data, source_image, path: Path, dtype) -> None:
    header = source_image.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(data.astype(dtype), source_image.affine, header),
        str(path),
    )


def run(job: dict) -> dict:
    import nibabel as nib
    import numpy as np

    started = time.monotonic()
    deadline = started + int(job.get("timeout_seconds", 3600))
    work_dir = Path(job["work_dir"])
    output_dir = Path(job["output_dir"])
    image_path = Path(job["image"])
    stem = job["case_stem"]
    levels = list(job["levels"])
    labels = {code: int(value) for code, value in job["labels"].items()}
    if not image_path.is_file():
        raise WorkerError(f"The uploaded MRI is missing: {image_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    body_output = work_dir / "total_mr_vertebrae.nii.gz"
    levels_output = work_dir / "vertebrae_mr_levels.nii.gz"

    browser_device, cli_device, device_trace = _resolve_device(job)
    emit("device", device=browser_device, **device_trace)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if job.get("weights_path"):
        environment["TOTALSEG_WEIGHTS_PATH"] = job["weights_path"]

    tasks = [
        (_body_command(job, body_output, cli_device), "total_mr vertebral body"),
        (_levels_command(job, levels_output, cli_device), "vertebrae_mr levels"),
    ]
    # On a CPU the two tasks would only compete for the same cores.
    parallel = browser_device == "cuda"
    sampler = _GpuSampler().start() if browser_device == "cuda" else None
    try:
        emit("vertebrae_segmentation" if parallel else "body_segmentation")
        if parallel:
            _run_tasks(tasks, environment, deadline, parallel=True)
        else:
            _run_tasks(tasks[:1], environment, deadline, parallel=False)
            emit("level_segmentation")
            _run_tasks(tasks[1:], environment, deadline, parallel=False)
    finally:
        gpu_trace = sampler.stop() if sampler is not None else None

    emit("intersecting")
    source_image = nib.load(str(image_path), mmap=False)
    if len(source_image.shape) != 3:
        raise WorkerError(
            f"Radiomics MRI must be a 3D NIfTI image; received shape {source_image.shape}."
        )

    body_labels, body_resampled = _labels_on_source_grid(
        _load_output(body_output, "total_mr"), source_image
    )
    body_mask = body_labels == _task_label("total_mr", "vertebrae")
    if not body_mask.any():
        raise WorkerError("TotalSegmentator total_mr returned an empty vertebral-body mask.")

    level_labels, levels_resampled = _labels_on_source_grid(
        _load_output(levels_output, "vertebrae_mr"), source_image
    )
    level_masks = {}
    resampled_masks = ["vertebrae"] if body_resampled else []
    level_voxel_counts = {}
    for code in levels:
        name = f"vertebrae_{code}"
        level_mask = level_labels == _task_label("vertebrae_mr", name)
        if not level_mask.any():
            raise WorkerError(
                f"TotalSegmentator vertebrae_mr returned an empty {code} mask. "
                "Check that the selected level is inside the MRI field of view."
            )
        level_masks[code] = level_mask
        level_voxel_counts[code] = int(level_mask.sum())
        if levels_resampled:
            resampled_masks.append(name)

    final_mask, selected_mask, body_by_level = _intersect_body_and_levels(
        body_mask, level_masks
    )
    voxel_counts = {code: int(mask.sum()) for code, mask in body_by_level.items()}
    missing_intersections = [code for code, count in voxel_counts.items() if count == 0]
    if missing_intersections:
        raise WorkerError(
            "The TotalSegmentator body mask did not overlap the level mask(s): "
            + ", ".join(missing_intersections)
            + ". Inspect both model outputs before extracting Radiomics."
        )
    mask_voxels = int(final_mask.sum())
    selected_voxels = int(selected_mask.sum())
    removed_fraction = round(1 - mask_voxels / max(selected_voxels, 1), 3)

    emit("saving")
    level_tag = "-".join(levels)
    files = {
        "image": image_path.name,
        "body": f"{stem}_totalsegmentator_body.nii",
        "segmentation": f"{stem}_totalsegmentator_levels.nii",
        "voi": f"{stem}_voi_{level_tag}_body.nii",
    }
    body_path = output_dir / files["body"]
    segmentation_path = output_dir / files["segmentation"]
    mask_path = output_dir / files["voi"]

    level_segmentation = np.zeros(source_image.shape, dtype=np.uint8)
    for code, level_mask in level_masks.items():
        level_segmentation[level_mask] = labels[code]
    _save_mask(nib, np, body_mask, source_image, body_path, np.uint8)
    _save_mask(nib, np, level_segmentation, source_image, segmentation_path, np.uint8)
    _save_mask(nib, np, final_mask, source_image, mask_path, np.uint8)

    try:
        source = np.asanyarray(source_image.dataobj)
        orientation = nib.aff2axcodes(source_image.affine)
        preview_name = f"{stem}_voi_preview.jpg"
        if _render_preview(
            source, final_mask, orientation, levels,
            output_dir / preview_name, title="Body VOI",
        ):
            files["voi_preview"] = preview_name
        levels_preview_name = f"{stem}_totalsegmentator_levels_preview.jpg"
        if _render_preview(
            source, selected_mask, orientation, levels,
            output_dir / levels_preview_name, title="Level masks",
        ):
            files["level_segmentation_preview"] = levels_preview_name
    except Exception as exc:
        emit("saving", detail=f"Preview skipped: {exc}")

    report = {
        "status": "ok",
        "mask_path": str(mask_path),
        "body_path": str(body_path),
        "segmentation_path": str(segmentation_path),
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "files": files,
        "levels": levels,
        "labels": labels,
        "level_voxel_counts": level_voxel_counts,
        "voxel_counts": voxel_counts,
        "body_mask_voxels": int(body_mask.sum()),
        "mask_voxels": mask_voxels,
        "body_only": True,
        "body_only_voxels": mask_voxels,
        "body_only_removed_fraction": removed_fraction,
        "intersection_method": "total_mr.vertebrae AND union(vertebrae_mr selected levels)",
        "resampled_masks": resampled_masks,
        "image_size": [int(size) for size in source_image.shape],
        "image_spacing": [
            round(float(value), 6) for value in source_image.header.get_zooms()[:3]
        ],
        "device": browser_device,
        "device_trace": device_trace,
        "gpu_trace": gpu_trace,
        "version": _totalsegmentator_version(),
        "duration_seconds": round(time.monotonic() - started, 1),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {key: value for key, value in report.items() if key != "status"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    try:
        payload = run(json.loads(sys.stdin.read()))
    except WorkerError as exc:
        payload = {"status": "failed", "message": str(exc)}
    except ImportError as exc:
        payload = {
            "status": "failed",
            "message": (
                f"TotalSegmentator or one of its dependencies is missing: {exc}. "
                "Install it with 'pip install TotalSegmentator' and point "
                "TOTALSEGMENTATOR_PYTHON at that environment."
            ),
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "message": repr(exc),
            "traceback": traceback.format_exc()[-2500:],
        }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
