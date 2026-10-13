"""Geometry-first, conservative identification of conventional spine MR images.

TR/TE/TI limits below are configurable protocol heuristics, not universal MRI
definitions. Free-text names corroborate timing; they never determine plane
or suffice to accept a target. Inversion-prepared images are excluded from
both targets, including STIR regardless of its TR/TE or description, with one
exception: sagittal spin-echo T1 FLAIR (long TI, short TE), which is the
sagittal T1 protocol of most development radiomics patients.
"""

from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class ClassificationRules:
    t2_tr_min: float = 1500.0
    t2_te_min: float = 60.0
    t1_tr_max: float = 1000.0
    t1_te_max: float = 30.0
    require_axial_t2_view: bool = False
    # T1 FLAIR: TI near the CSF null point (STIR uses ~100-250 ms, T2 FLAIR
    # ~2000+ ms) and a TR that is long only because of the inversion.
    t1_flair_tr_max: float = 3500.0
    t1_flair_ti_min: float = 600.0
    t1_flair_ti_max: float = 1200.0


def _value(source, name, default=None):
    return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)


def _numbers(value):
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, str):
        value = value.split("\\")
    elif np.isscalar(value):
        value = [value]
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return []
    return values if values and all(np.isfinite(values)) else []


def _text(value):
    if value is None:
        return ""
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        return " ".join(str(item) for item in value).upper()
    return str(value).upper()


def _functional_value(group, name):
    for sequence in (
        "PlaneOrientationSequence", "MRTimingAndRelatedParametersSequence",
        "MREchoSequence", "MRModifierSequence", "MRImageFrameTypeSequence", "MRDiffusionSequence",
    ):
        for item in _value(group, sequence, []) or []:
            found = _value(item, name)
            if found is not None:
                return found
    return _value(group, name)


def metadata_for_frames(dataset):
    """Read classic MR tags and the corresponding Enhanced MR functional groups."""
    shared = (_value(dataset, "SharedFunctionalGroupsSequence", []) or [None])[0]
    frames = _value(dataset, "PerFrameFunctionalGroupsSequence", []) or [None]
    names = (
        "ImageOrientationPatient", "RepetitionTime", "EchoTime", "InversionTime",
        "ScanningSequence", "SequenceName", "SeriesDescription", "ProtocolName",
        "ImageType", "InversionRecovery", "EchoPulseSequence",
        "DiffusionBValue", "DiffusionDirectionality",
    )
    aliases = {"EchoTime": "EffectiveEchoTime", "InversionTime": "InversionTimes", "ImageType": "FrameType"}
    result = []
    for frame in frames:
        metadata = {}
        for name in names:
            for source in (frame, shared, dataset):
                if source is None:
                    continue
                value = _functional_value(source, name)
                if value is None and name in aliases:
                    value = _functional_value(source, aliases[name])
                if value is not None:
                    metadata[name] = value
                    break
        result.append(metadata)
    return result


def plane_from_orientation(orientation):
    """Return (plane, unit normal) from IOP; oblique planes use the dominant axis."""
    values = _numbers(orientation)
    if len(values) != 6:
        return None, None
    row, column = np.asarray(values[:3]), np.asarray(values[3:])
    row_length, column_length = np.linalg.norm(row), np.linalg.norm(column)
    if row_length < 1e-8 or column_length < 1e-8:
        return None, None
    # Collinear/nonorthogonal vectors are malformed, not an alternative plane.
    if abs(np.dot(row, column) / (row_length * column_length)) > 0.1:
        return None, None
    normal = np.cross(row, column)
    length = np.linalg.norm(normal)
    if length < 1e-8:
        return None, None
    normal = normal / length
    plane = ("SAGITTAL", "CORONAL", "AXIAL")[int(np.argmax(np.abs(normal)))]
    return plane, normal.tolist()


def classify_metadata(metadata, rules=None):
    """Return the label plus auditable plane, timing and exclusion reasons."""
    rules = rules or ClassificationRules()
    plane, normal = plane_from_orientation(_value(metadata, "ImageOrientationPatient"))
    timings = {}
    for short, name in (("tr", "RepetitionTime"), ("te", "EchoTime"), ("ti", "InversionTime")):
        values = _numbers(_value(metadata, name))
        timings[short] = values[0] if len(values) == 1 else None
    report = {"classification": None, "plane": plane, "normal": normal, **timings, "reasons": []}
    reasons = report["reasons"]
    if plane is None:
        reasons.append("missing_or_invalid_image_orientation")
        return report
    text = " ".join(_text(_value(metadata, name)) for name in ("SeriesDescription", "ProtocolName", "SequenceName"))
    scanning = set(re.findall(r"[A-Z]+", _text(_value(metadata, "ScanningSequence"))))
    image_type = _text(_value(metadata, "ImageType"))
    # VIEW is an extra acquisition-selection requirement, never evidence of
    # axial geometry or T2 contrast. Underscores count as token separators.
    is_view = bool(re.search(r"(?<![A-Z0-9])VIEW(?![A-Z0-9])", text + " " + image_type))
    report["is_view"] = is_view
    # Explicit STIR excludes even mislabeled/malformed short-TR short-TE images.
    if "STIR" in text or "SHORT TAU" in text or "SHORT TI INVERSION" in text:
        reasons.append("stir_sequence")
        return report
    inversion_times = _numbers(_value(metadata, "InversionTime"))
    raw_ti = _value(metadata, "InversionTime")
    if raw_ti is not None and str(raw_ti).strip() and (not inversion_times or any(ti < 0 for ti in inversion_times)):
        reasons.append("invalid_inversion_time")
        return report
    ir_name = bool(re.search(r"TIRM|\bTIR\b|INVERSION|\bIR[ _-]*(?:FSE|TSE)\b", text))
    pulse = _text(_value(metadata, "EchoPulseSequence"))
    inversion = "IR" in scanning or ir_name or _text(_value(metadata, "InversionRecovery")) in {"YES", "Y", "TRUE"} or any(ti > 0 for ti in inversion_times)
    # Only a sagittal spin-echo inversion whose TI fits T1 FLAIR continues; its
    # TR/TE are still checked below. STIR and T2 FLAIR fall outside the TI range.
    t1_flair = (
        inversion and plane == "SAGITTAL" and bool(inversion_times)
        and "GR" not in scanning and pulse != "GRADIENT"
        and all(rules.t1_flair_ti_min <= ti <= rules.t1_flair_ti_max for ti in inversion_times)
    )
    report["t1_flair"] = t1_flair
    if inversion and not t1_flair:
        reasons.append("inversion_recovery_excluded")
        return report
    # Some vendors (e.g. Philips with CLEAR/filtering) label ordinary acquired
    # slices DERIVED\PRIMARY, so only secondary captures and reformats/projections
    # are excluded here.
    reformat = re.search(r"(?<![A-Z0-9])(MPR|MIP|MINIP|MNIP|CPR|PROJECTION|REFORMATTED|SUBTRACT|SUBTRACTION)(?![A-Z0-9])", image_type)
    if "SECONDARY" in image_type or ("DERIVED" in image_type and reformat) or re.search(r"LOCALIZ|SCOUT|SURVEY", text):
        reasons.append("derived_or_localizer_image")
        return report
    diffusion = _numbers(_value(metadata, "DiffusionBValue"))
    diffusion_direction = _text(_value(metadata, "DiffusionDirectionality"))
    if "EP" in scanning or any(b > 0 for b in diffusion) or diffusion_direction not in {"", "NONE"} or re.search(r"\b(DWI|ADC|DIFFUSION|TRACEW)\b", text):
        reasons.append("diffusion_or_echo_planar_sequence")
        return report
    tr, te = timings["tr"], timings["te"]
    if tr is None or te is None or tr <= 0 or te <= 0:
        reasons.append("missing_or_invalid_tr_te")
        return report
    if plane == "AXIAL":
        if "GR" in scanning or pulse == "GRADIENT":
            reasons.append("gradient_echo_not_conventional_t2")
        elif tr >= rules.t2_tr_min and te >= rules.t2_te_min:
            if rules.require_axial_t2_view and not is_view:
                reasons.append("axial_t2_without_view_marker")
                return report
            report["classification"] = "AXIAL_T2"
            reasons.extend(["axial_geometry", "long_tr", "long_te"])
            if is_view:
                reasons.append("view_acquisition_marker")
            if re.search(r"T2|FSE|TSE", text):
                reasons.append("supporting_t2_or_spin_echo_name")
            if "SE" in scanning or pulse == "SPIN":
                reasons.append("spin_echo_metadata")
        else:
            reasons.append("timing_not_conventional_t2")
    elif plane == "SAGITTAL":
        if tr <= (rules.t1_flair_tr_max if t1_flair else rules.t1_tr_max) and te <= rules.t1_te_max:
            report["classification"] = "SAGITTAL_T1"
            if t1_flair:
                reasons.extend(["sagittal_geometry", "t1_flair_inversion_time", "t1_flair_tr", "short_te"])
            else:
                reasons.extend(["sagittal_geometry", "short_tr", "short_te", "no_inversion_recovery"])
            if "T1" in text:
                reasons.append("supporting_t1_name")
            if scanning or pulse:
                reasons.append("sequence_metadata_present")
        else:
            reasons.append("timing_not_conventional_t1")
    else:
        reasons.append("coronal_plane_not_requested")
    return report


def inspect_mri_series(dataset, rules=None):
    """Accept an Enhanced MR object only when every frame agrees on its target."""
    reports = [classify_metadata(frame, rules) for frame in metadata_for_frames(dataset)]
    report = reports[0]
    labels = {item["classification"] for item in reports}
    planes = {item["plane"] for item in reports}
    if len(labels) > 1 or len(planes) > 1:
        report = {**report, "classification": None, "reasons": ["inconsistent_frame_metadata"]}
    return report


def classify_mri_series(dataset, rules=None):
    """Return AXIAL_T2, SAGITTAL_T1, or None; accepts pydicom datasets or dicts."""
    return inspect_mri_series(dataset, rules)["classification"]
