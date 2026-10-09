"""Keep the exact MuscleMap inputs, masks and metrics for inspection."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np


LABELS = {
    1: "Multifidus, right", 2: "Multifidus, left",
    3: "Erector spinae, right", 4: "Erector spinae, left",
    5: "Psoas major, right", 6: "Psoas major, left",
    7: "Quadratus lumborum, right", 8: "Quadratus lumborum, left",
}


def start_trace(directory, *, output_root=None, method, region, components, source_manifest=None):
    now = datetime.now(timezone.utc)
    if output_root is not None:
        directory = Path(output_root) / (now.strftime("%Y%m%d-%H%M%S") + "_" + uuid4().hex[:12])
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    trace = {
        "source": "musclemap", "directory": str(directory), "directory_name": directory.name,
        "created_at_utc": now.isoformat(), "method": method, "region": region,
        "components": components, "source_manifest": str(source_manifest) if source_manifest else None,
        "status": "preparing", "labels": LABELS, "series": [], "files": {"metadata": "metadata.json"},
    }
    save_trace(trace)
    return trace


def save_trace(trace):
    path = Path(trace["directory"]) / "metadata.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def record_segmentation(trace, image, mask, source_filename, report, *, source_image=None):
    directory = Path(trace["directory"]).resolve()
    image, mask = Path(image).resolve(), Path(mask).resolve()
    image_volume = nib.load(image)
    mask_data = np.asanyarray(nib.load(mask).dataobj)
    index = len(trace["series"]) + 1
    files = {"image": image.relative_to(directory).as_posix(), "segmentation": mask.relative_to(directory).as_posix()}
    record = {
        "series": index, "source_filename": source_filename,
        "source_image": str(source_image) if source_image else None,
        "image_size": list(image_volume.shape),
        "image_spacing": [float(value) for value in image_volume.header.get_zooms()],
        "mask_voxels": int(np.count_nonzero(mask_data)),
        "label_voxel_counts": {label: int(np.count_nonzero(mask_data == label)) for label in LABELS},
        "segmentation_report": dict(report), "files": files,
    }
    preview = image.parent / "muscle_segmentation_preview.jpg"
    try:
        if render_preview(image, mask, preview):
            files["preview"] = preview.relative_to(directory).as_posix()
    except Exception as exc:
        # The saved mask and quantitative results remain useful without a JPEG.
        record["preview_error"] = str(exc)
    trace["series"].append(record)
    trace["files"].update({f"series_{index:04d}_{key}": value for key, value in files.items()})
    save_trace(trace)
    return record


def record_metrics(trace, index, csv_path, report):
    filename = Path(csv_path).resolve().relative_to(Path(trace["directory"]).resolve()).as_posix()
    record = trace["series"][index]
    record["files"]["metrics"] = filename
    record["extraction_report"] = dict(report)
    trace["files"][f"series_{index + 1:04d}_metrics"] = filename
    save_trace(trace)


def finish_trace(trace, metrics, features):
    directory = Path(trace["directory"])
    metrics.to_csv(directory / "metrics.csv", index=False)
    (directory / "features.json").write_text(
        json.dumps(features, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    trace["files"].update(metrics="metrics.csv", features="features.json")
    trace.update(status="complete", feature_count=len(features))
    save_trace(trace)


def render_preview(image_path, mask_path, output_path):
    """Show representative axial slices; resampling only affects this preview."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from nibabel.processing import resample_from_to

    def spatial_volume(path):
        volume = nib.load(path)
        data = np.asanyarray(volume.dataobj)
        if data.ndim == 2:
            data = data[:, :, None]
        if data.ndim != 3:
            raise ValueError(f"Preview requires a 2D or 3D volume, received {data.shape}.")
        return nib.as_closest_canonical(nib.Nifti1Image(data, volume.affine))

    image, mask = spatial_volume(image_path), spatial_volume(mask_path)
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine):
        mask = resample_from_to(mask, (image.shape, image.affine), order=0)
    pixels, labels = np.asanyarray(image.dataobj), np.asanyarray(mask.dataobj)
    counts = np.count_nonzero(labels, axis=(0, 1))
    slices = np.flatnonzero(counts)
    if not slices.size:
        return False
    slices = sorted(slices[np.argsort(counts[slices])[-3:]].tolist())
    finite = pixels[np.isfinite(pixels)]
    if not finite.size:
        return False
    lower, upper = np.percentile(finite, [1, 99])
    colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#14b8a6", "#3b82f6", "#a855f7", "#ec4899"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 9.5), cmap.N)
    figure = Figure(figsize=(5 * len(slices), 6), facecolor="white")
    for column, index in enumerate(slices, 1):
        axis = figure.add_subplot(1, len(slices), column)
        axis.imshow(np.rot90(pixels[:, :, index]), cmap="gray", vmin=lower, vmax=max(upper, lower + 1))
        overlay = np.rot90(labels[:, :, index])
        axis.imshow(np.ma.masked_where(overlay == 0, overlay), cmap=cmap, norm=norm, alpha=0.5, interpolation="nearest")
        axis.set_title(f"Axial slice {index + 1}")
        axis.set_axis_off()
    figure.suptitle("MuscleMap segmentation", fontsize=14)
    figure.legend(
        handles=[Patch(color=colors[label - 1], label=name) for label, name in LABELS.items()],
        loc="lower center", ncol=2, fontsize=9,
    )
    figure.subplots_adjust(top=0.9, bottom=0.2, wspace=0.05)
    figure.savefig(output_path, format="jpg", dpi=110)
    return True
