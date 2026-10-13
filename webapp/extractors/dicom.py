"""Inspect one DICOM case, convert selected series, and run both extractors."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from ..dicom_classification import ClassificationRules, inspect_mri_series
from .errors import ExtractionError, ExtractionExecutionError, ExtractionInputError, ExtractorUnavailableError
from .muscle import prepare_muscle_extraction_events, extract_prepared_muscle_features_events
from .radiomics import prepare_radiomics_extraction_events, extract_prepared_radiomics_features_events
from .parallel import parallel_feature_events


LUMBAR_SEGMENTS = {f"L{i}-L{i + 1}": [f"L{i}", f"L{i + 1}"] for i in range(1, 5)}


def segment_levels(value):
    code = str(value or "").strip().upper().replace("–", "-")
    if code not in LUMBAR_SEGMENTS:
        raise ExtractionInputError("Choose L1-L2, L2-L3, L3-L4, or L4-L5.")
    return list(LUMBAR_SEGMENTS[code])


@dataclass
class DicomSeries:
    uid: str
    description: str
    protocol: str
    files: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    frames: int = 0

    def summary(self, index):
        first = self.reports[0]
        labels = {report["classification"] for report in self.reports}
        planes = {report["plane"] for report in self.reports}
        consistent = len(labels) == 1 and len(planes) == 1
        reasons = list(dict.fromkeys(reason for report in self.reports for reason in report["reasons"]))
        if not consistent:
            reasons.append("inconsistent_series_metadata")
        return {
            "series": index, "description": self.description, "protocol": self.protocol,
            "files": len(self.files), "frames": self.frames,
            "classification": first["classification"] if consistent else None,
            "plane": first["plane"] if len(planes) == 1 else None,
            "tr": first["tr"], "te": first["te"], "ti": first["ti"],
            "normal": first["normal"],
            "reasons": reasons,
        }


def scan_dicom_series(paths, rules=None):
    """Group MR objects by UID, deduplicate SOPs, and check all image headers."""
    grouped, studies, patients, seen = {}, set(), set(), set()
    ignored = duplicates = 0
    for path in paths:
        try:
            try:
                dataset = pydicom.dcmread(path, stop_before_pixels=True)
            except InvalidDicomError:
                # Older exports can omit the preamble. Validate required MR
                # identifiers below before accepting this permissive read.
                dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except (OSError, ValueError, EOFError, TypeError):
            ignored += 1
            continue
        if str(getattr(dataset, "Modality", "")) != "MR" or not getattr(dataset, "Rows", 0) or not getattr(dataset, "Columns", 0):
            ignored += 1
            continue
        study, series, sop = (str(getattr(dataset, name, "")) for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"))
        if not all((study, series, sop)):
            ignored += 1
            continue
        studies.add(study)
        patient = str(getattr(dataset, "PatientID", ""))
        if patient:
            patients.add((patient, str(getattr(dataset, "IssuerOfPatientID", ""))))
        if len(studies) > 1 or len(patients) > 1:
            raise ExtractionInputError("These files contain multiple MRI studies or patients. Select one patient's MRI study.")
        if sop in seen:
            duplicates += 1
            continue
        seen.add(sop)
        item = grouped.setdefault(series, DicomSeries(series, str(getattr(dataset, "SeriesDescription", "")), str(getattr(dataset, "ProtocolName", ""))))
        item.files.append(Path(path))
        item.reports.append(inspect_mri_series(dataset, rules))
        item.frames += int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if not grouped:
        raise ExtractionInputError("No readable MRI DICOM images found. Select the DICOM image files of one patient.")
    return list(grouped.values()), {"ignored_files": ignored, "duplicate_images": duplicates}


def resolve_converter(command):
    found = shutil.which(command) or (str(Path(command).expanduser()) if Path(command).expanduser().is_file() else None)
    if not found:
        raise ExtractorUnavailableError("dcm2niix is unavailable. Install it in the web environment or configure DCM2NIIX_COMMAND.")
    return found


def decode_compressed_copies(source, target):
    """Return ``source``, or ``target`` filled with uncompressed copies when any image is encapsulated.

    dcm2niix's built-in JPEG Lossless decoder rejects restart intervals (DRI),
    which some PACS add when compressing MRI, so pixel data is decoded here
    first. The retained ``source`` copies stay byte-identical to the selection.
    """
    paths = sorted(source.glob("*.dcm"))
    syntaxes = [pydicom.dcmread(path, stop_before_pixels=True, force=True).file_meta.get("TransferSyntaxUID") for path in paths]
    if not any(uid is not None and uid.is_compressed for uid in syntaxes):
        return source
    target.mkdir()
    for path, uid in zip(paths, syntaxes):
        if uid is None or not uid.is_compressed:
            shutil.copyfile(path, target / path.name)
            continue
        try:
            dataset = pydicom.dcmread(path, force=True)
            dataset.decompress(generate_instance_uid=False)
            dataset.save_as(target / path.name, enforce_file_format=True)
        except Exception as exc:
            raise ExtractionExecutionError(
                f"Cannot decode {uid.name} pixel data: {exc}. Install python-gdcm in the web environment."
            ) from exc
    return target


def convert_series(series, directory, command, timeout_seconds=300):
    """Convert a single UID independently; never force incompatible slices to merge."""
    directory = Path(directory)
    source, output, decoded = directory / "dicom", directory / "nifti", directory / "decoded"
    source.mkdir(parents=True)
    output.mkdir()
    for index, path in enumerate(series.files):
        shutil.copyfile(path, source / f"image_{index:06d}.dcm")
    command = resolve_converter(command)
    try:
        completed = subprocess.run(
            [command, "-z", "y", "-b", "y", "-ba", "y", "-m", "n", "-f", directory.name + "_%s", "-o", str(output),
             str(decode_compressed_copies(source, decoded))],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionExecutionError("DICOM conversion timed out.") from exc
    except OSError as exc:
        raise ExtractorUnavailableError(f"Cannot start dcm2niix: {exc}") from exc
    finally:
        # Decoded copies are only converter input; the originals are retained.
        shutil.rmtree(decoded, ignore_errors=True)
    images = sorted(output.glob("*.nii.gz"))
    if completed.returncode != 0 or not images:
        detail = (completed.stderr or completed.stdout)[-800:].strip()
        raise ExtractionExecutionError(f"DICOM conversion failed: {detail}")
    for path in images:
        try:
            image = nib.load(path)
            if len(image.shape) != 3 or not np.isfinite(image.affine).all() or any(value <= 0 for value in image.header.get_zooms()):
                raise ValueError("expected one spatial 3D volume with valid geometry")
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise ExtractionExecutionError(f"Converted MRI has unsupported geometry: {exc}") from exc
    return images


def _save_selection(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def prepare_dicom_events(
        paths, work_dir, *, converter_command, conversion_timeout=300,
        rules=None, archive_root=None, level=None,
):
    rules = rules or ClassificationRules()
    yield {"event": "stage", "stage": "dicom_scanning"}
    series, counts = scan_dicom_series(paths, rules)
    summaries = [item.summary(index + 1) for index, item in enumerate(series)]
    axial = [index for index, item in enumerate(summaries) if item["classification"] == "AXIAL_T2"]
    sagittal = [index for index, item in enumerate(summaries) if item["classification"] == "SAGITTAL_T1"]
    warnings = []
    # VIEW is preferred, not required: when any valid axial T2 carries the VIEW
    # marker, only those series are used; otherwise all valid axial T2 series are.
    view_axial = [index for index in axial if all(report.get("is_view") for report in series[index].reports)]
    if view_axial:
        for index in axial:
            if index not in view_axial:
                summaries[index]["classification"] = None
                summaries[index]["reasons"].append("non_view_axial_t2_not_used_view_present")
        axial = view_axial
    # Conventional sagittal T1 is preferred; T1 FLAIR is used only without it,
    # so a case holding both still yields one unambiguous Radiomics series.
    conventional_sagittal = [index for index in sagittal if not any(report.get("t1_flair") for report in series[index].reports)]
    if conventional_sagittal:
        for index in sagittal:
            if index not in conventional_sagittal:
                summaries[index]["classification"] = None
                summaries[index]["reasons"].append("t1_flair_not_used_conventional_t1_present")
        sagittal = conventional_sagittal
    if not axial:
        warnings.append("No axial T2 series with valid orientation and T2 timing was found.")
    if not sagittal:
        warnings.append("No sagittal T1 or T1 FLAIR series with valid orientation and timing was found; STIR is excluded.")
    elif len(sagittal) > 1:
        warnings.append("Multiple sagittal T1 series match. Radiomics requires one unambiguous series; select only the files of the intended series.")
        sagittal = []
    yield {"event": "discovery", "series": summaries, "warnings": list(warnings), **counts}
    if not axial and not sagittal:
        raise ExtractionInputError(" ".join(warnings))

    root = Path(archive_root or Path(__file__).resolve().parents[1] / "selected_mri_cases").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    archive = root / (now.strftime("%Y%m%d-%H%M%S") + "_" + uuid4().hex[:12])
    archive.mkdir(mode=0o700)
    manifest_path = archive / "selection.json"
    manifest = {
        "created_at_utc": now.isoformat(), "level": level, "classification_rules": asdict(rules),
        "series": summaries, "selected": [], "warnings": list(warnings), **counts,
    }
    _save_selection(manifest_path, manifest)
    converted = {"Muscle": [], "Radiomics": []}
    for modality, indices in (("Muscle", axial), ("Radiomics", sagittal)):
        modality_failed = False
        for index in indices:
            category = "axial_t2_view" if modality == "Muscle" else "sagittal_t1"
            directory = archive / category / f"series_{index + 1:04d}"
            record = {
                **summaries[index], "modality": modality, "series_instance_uid": series[index].uid,
                "directory": directory.relative_to(archive).as_posix(), "status": "selected",
                "nifti_files": [], "dicom_files": [], "sidecars": [],
            }
            manifest["selected"].append(record)
            _save_selection(manifest_path, manifest)
            yield {"event": "stage", "stage": "dicom_converting", "modality": modality, "series": index + 1, "series_total": len(series)}
            try:
                images = convert_series(series[index], directory, converter_command, conversion_timeout)
                if modality == "Radiomics" and len(images) != 1:
                    raise ExtractionInputError("Sagittal T1 conversion produced multiple volumes. Select a single acquisition for Radiomics.")
                converted[modality].extend(images)
                record["status"] = "converted"
            except ExtractionError as exc:
                modality_failed = True
                record["status"] = "conversion_failed"
                record["error"] = str(exc)
                warnings.append(f"{modality}: {exc}")
            record["nifti_files"] = [path.relative_to(archive).as_posix() for path in sorted((directory / "nifti").glob("*.nii.gz"))]
            record["dicom_files"] = [path.relative_to(archive).as_posix() for path in sorted((directory / "dicom").glob("*.dcm"))]
            record["sidecars"] = [path.relative_to(archive).as_posix() for path in sorted((directory / "nifti").glob("*.json"))]
            manifest["warnings"] = list(warnings)
            _save_selection(manifest_path, manifest)
        if modality_failed:
            # Retain every selected series for inspection, but never calculate
            # the trained Muscle aggregates from a partially converted case.
            converted[modality] = []
    saved = {
        "directory": str(archive), "manifest": str(manifest_path),
        "selected": manifest["selected"],
    }
    yield {"event": "saved_series", **saved}
    return converted, warnings, saved


def _validated_features(features, expected, modality):
    if not isinstance(features, dict) or any(name not in features for name in expected):
        raise ExtractionExecutionError(f"{modality} is missing selected model features.")
    try:
        selected = {name: float(features[name]) for name in expected}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtractionExecutionError(f"{modality} returned invalid feature values.") from exc
    if not all(math.isfinite(value) for value in selected.values()):
        raise ExtractionExecutionError(f"{modality} returned nonfinite feature values.")
    return selected


def dicom_features_events(
        paths, level, work_dir, *, expected_features, converter_command,
        muscle_options, radiomics_options, conversion_timeout=300, rules=None,
        archive_root=None,
):
    levels = segment_levels(level)
    converted, warnings, saved = yield from prepare_dicom_events(
        paths, work_dir, converter_command=converter_command,
        conversion_timeout=conversion_timeout, rules=rules,
        archive_root=archive_root, level=level,
    )
    prepared = {}
    # Both segmentation/preparation branches finish serially. Only the
    # extraction-only factories below may enter the two-worker pool.
    for modality in ("Muscle", "Radiomics"):
        images = converted[modality]
        if not images:
            continue
        events = None
        try:
            yield {"event": "stage", "stage": "preparing", "phase": "preparation", "modality": modality}
            if modality == "Muscle":
                events = prepare_muscle_extraction_events(
                    images, Path(saved["directory"]) / "muscle_segmentations",
                    source_manifest=saved["manifest"], **muscle_options,
                )
            else:
                events = prepare_radiomics_extraction_events(
                    images[0], expected_features[modality], levels=levels, **radiomics_options,
                )
            while True:
                try:
                    stage = next(events)
                except StopIteration as stop:
                    prepared[modality] = stop.value
                    segmentation = getattr(stop.value, "segmentation", None)
                    if segmentation and segmentation.get("directory"):
                        manifest_path = Path(saved["manifest"])
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        manifest.setdefault("segmentations", {})[modality] = {
                            "directory": segmentation["directory"],
                            "metadata": str(Path(segmentation["directory"]) / "metadata.json"),
                        }
                        _save_selection(manifest_path, manifest)
                    break
                yield {**stage, "event": "stage", "phase": "preparation", "modality": modality}
        except ExtractionError as exc:
            warnings.append(f"{modality}: {exc}")
            yield {"event": "modality_error", "modality": modality, "phase": "preparation", "message": str(exc)}
        except Exception:
            logging.getLogger(__name__).exception("%s MRI preparation failed", modality)
            message = f"Unexpected {modality} preparation failure. Selected MRI files remain saved for inspection."
            warnings.append(f"{modality}: {message}")
            yield {"event": "modality_error", "modality": modality, "phase": "preparation", "message": message}
        finally:
            if events is not None:
                events.close()

    factories = {}
    for modality, job in prepared.items():
        extractor = extract_prepared_muscle_features_events if modality == "Muscle" else extract_prepared_radiomics_features_events
        factories[modality] = partial(extractor, job, expected_features[modality])
    ready = set()
    if factories:
        yield {"event": "stage", "stage": "features_start", "phase": "feature_extraction", "modalities": list(factories), "parallel": len(factories) == 2}
        events = parallel_feature_events(factories)
        try:
            for event in events:
                modality = event["modality"]
                if event["event"] == "modality_error":
                    warnings.append(f"{modality}: {event['message']}")
                elif event["event"] == "result":
                    try:
                        features = _validated_features(event["features"], expected_features[modality], modality)
                    except ExtractionError as exc:
                        warnings.append(f"{modality}: {exc}")
                        yield {"event": "modality_error", "modality": modality, "phase": "feature_extraction", "message": str(exc)}
                        continue
                    ready.add(modality)
                    images = converted[modality]
                    yield {
                        "event": "result", "modality": modality, "features": features, "features_ready": True,
                        "image_filenames": [path.name for path in images], "image_filename": images[0].name,
                        "series_count": len(images), "segmentation": getattr(prepared[modality], "segmentation", None),
                        "mask_filename": None, "saved_directory": saved["directory"],
                        "message": f"Extracted {len(features)} {modality} features.",
                    }
                    continue
                yield event
        finally:
            events.close()
    if not ready:
        raise ExtractionExecutionError("Neither modality could be extracted. Selected MRI files remain saved for inspection. " + " ".join(warnings))
    yield {
        "event": "complete", "modalities": [name for name in ("Muscle", "Radiomics") if name in ready],
        "warnings": warnings, "level": level, "saved_directory": saved["directory"],
    }
