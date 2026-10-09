"""PyRadiomics worker executed by the environment that has PyRadiomics.

``radiomics.py`` resolves the FAE configuration down to the image types and
feature classes the selected model features need, then hands the resulting
PyRadiomics parameter dict to this script. Only the extraction itself runs
here, so the web application does not need PyRadiomics or SimpleITK installed.

Each image type is extracted in its own ``execute`` call. PyRadiomics filters
and bins every image type independently, so this yields the same values as one
combined call while letting the caller report which filter is running.
"""

from __future__ import annotations

import json
import sys
import traceback


RESULT_PREFIX = "##FAE_WORKER_JSON##"
STAGE_PREFIX = "##FAE_WORKER_STAGE##"


class WorkerError(RuntimeError):
    """A failure that should reach the user as an extraction error."""


def emit(stage: str, **detail) -> None:
    print(
        STAGE_PREFIX + json.dumps({"stage": stage, **detail}, ensure_ascii=False),
        flush=True,
    )


def _import_pyradiomics():
    """Import the PyRadiomics package, not this package's ``radiomics`` module.

    Python puts this script's own directory first on ``sys.path``, and that
    directory holds ``radiomics.py``, which would shadow the real library.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry and os.path.abspath(entry) != here
    ]
    sys.modules.pop("radiomics", None)

    import SimpleITK as sitk
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    return sitk, RadiomicsFeatureExtractor


def run(job: dict) -> dict:
    sitk, RadiomicsFeatureExtractor = _import_pyradiomics()

    emit("reading")
    try:
        image = sitk.ReadImage(job["image"])
        mask = sitk.ReadImage(job["mask"])
    except Exception as exc:
        raise WorkerError(f"Cannot read the NIfTI image or VOI mask: {exc}") from exc

    if image.GetDimension() != mask.GetDimension():
        raise WorkerError(
            f"Image is {image.GetDimension()}D but the VOI mask is "
            f"{mask.GetDimension()}D."
        )
    if image.GetSize() != mask.GetSize():
        raise WorkerError(
            f"Image size {image.GetSize()} does not match VOI mask size "
            f"{mask.GetSize()}."
        )

    label = int(job["parameters"].get("setting", {}).get("label", 1))
    mask_array = sitk.GetArrayFromImage(mask)
    voxels = int((mask_array == label).sum())
    if voxels == 0:
        present = sorted(int(v) for v in set(mask_array.flatten().tolist()))[:8]
        raise WorkerError(
            f"The VOI mask has no voxels with label {label}; it contains "
            f"{present}."
        )

    # FAE aligns the VOI's metadata with the image once the matrix sizes match.
    mask.CopyInformation(image)

    parameters = job["parameters"]
    image_types = parameters.get("imageType", {})
    extracted = {}
    for index, (image_type, options) in enumerate(image_types.items(), start=1):
        emit(
            "extracting",
            image_type=image_type,
            filter_index=index,
            filter_total=len(image_types),
        )
        single = {
            "imageType": {image_type: options or {}},
            "featureClass": parameters.get("featureClass", {}),
            "setting": parameters.get("setting", {}),
        }
        try:
            extractor = RadiomicsFeatureExtractor(single)
            for key, value in extractor.execute(image, mask).items():
                extracted[str(key)] = value
        except Exception as exc:
            raise WorkerError(
                f"PyRadiomics failed on image type {image_type}: {exc}"
            ) from exc

    emit("mapping")
    features = {}
    for key, raw_value in extracted.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        # NaN and infinity are dropped so the adapter reports them as missing
        # rather than sending them to the model.
        if value == value and value not in (float("inf"), float("-inf")):
            features[f"Radiomics___MRI_{key}"] = value

    import radiomics

    return {
        "status": "ok",
        "features": features,
        "voxel_count": voxels,
        "image_size": list(image.GetSize()),
        "image_spacing": [round(v, 6) for v in image.GetSpacing()],
        "versions": {
            "pyradiomics": radiomics.__version__,
            "SimpleITK": sitk.Version_VersionString(),
            "python": sys.version.split()[0],
        },
    }


def main() -> int:
    try:
        payload = run(json.loads(sys.stdin.read()))
    except WorkerError as exc:
        payload = {"status": "failed", "message": str(exc)}
    except ImportError as exc:
        payload = {
            "status": "failed",
            "message": f"PyRadiomics or SimpleITK is missing: {exc}",
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "message": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
