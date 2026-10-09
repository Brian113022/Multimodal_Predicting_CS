"""Runtime configuration for the prediction web application."""

import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MuscleMap normally lives in its own conda environment, whose Scripts/bin
# directory is not on PATH for the interpreter running this web application.
MUSCLEMAP_ENV_NAME = os.environ.get("MUSCLEMAP_ENV_NAME", "MuscleMap")


def _musclemap_env_roots():
    home = Path.home()
    roots = []
    configured = os.environ.get("MUSCLEMAP_ENV")
    if configured:
        roots.append(Path(configured).expanduser())
    for distribution in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda"):
        roots.append(home / distribution / "envs" / MUSCLEMAP_ENV_NAME)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix).parent / MUSCLEMAP_ENV_NAME)
    return roots


def _conda_env_roots():
    home = Path.home()
    roots = []
    for distribution in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda"):
        environments = home / distribution / "envs"
        if environments.is_dir():
            try:
                roots.extend(sorted(environments.iterdir()))
            except OSError:
                continue
    return roots


def _find_env_python(*required_packages):
    """Interpreter of the first conda environment providing every package."""
    for root in _conda_env_roots():
        interpreter = next(
            (
                candidate
                for candidate in (root / "python.exe", root / "bin" / "python")
                if candidate.is_file()
            ),
            None,
        )
        if interpreter is None:
            continue
        packages = next(
            (
                candidate
                for candidate in (
                    root / "Lib" / "site-packages",
                    *root.glob("lib/python*/site-packages"),
                )
                if candidate.is_dir()
            ),
            None,
        )
        if packages is not None and all(
            (packages / name).is_dir() for name in required_packages
        ):
            return str(interpreter)
    return ""


def _find_project_venv_python(*required_packages):
    """Return the repository virtualenv interpreter when it owns all packages."""
    root = PROJECT_ROOT / ".venv"
    interpreter = next(
        (
            candidate
            for candidate in (root / "Scripts" / "python.exe", root / "bin" / "python")
            if candidate.is_file()
        ),
        None,
    )
    if interpreter is None:
        return ""
    packages = next(
        (
            candidate
            for candidate in (
                root / "Lib" / "site-packages",
                *root.glob("lib/python*/site-packages"),
            )
            if candidate.is_dir()
        ),
        None,
    )
    if packages is not None and all(
        (packages / name).is_dir() for name in required_packages
    ):
        return str(interpreter)
    return ""


def _find_musclemap_command(name):
    """Locate a MuscleMap executable without requiring it to be on PATH."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    for root in _musclemap_env_roots():
        for candidate in (
            root / "Scripts" / f"{name}.exe",
            root / "bin" / name,
        ):
            if candidate.is_file():
                return str(candidate)
    # Fall back to the bare name so the extractor reports the usual guidance.
    return name


def _find_musclemap_python():
    for root in _musclemap_env_roots():
        for candidate in (root / "python.exe", root / "bin" / "python"):
            if candidate.is_file():
                return str(candidate)
    return ""


class Config:
    # The 2026-09-13 run: development is every internal case (no 20% hold-out)
    # and the external cohort is the only test set, so the routes here were
    # fitted on ~25% more patients than the 2026-08-17 ones. Its Late_Fusion
    # meta-model is RF rather than LightGBM, and its external AUC is 0.858
    # against 0.815.
    #
    # Training exports directly to artifacts/; inference reads the same bundle.
    # MODEL_BUNDLE_PATH can explicitly override this default.
    MODEL_BUNDLE_PATH = Path(
        os.environ.get(
            "MODEL_BUNDLE_PATH",
            PROJECT_ROOT / "artifacts" / "late_fusion_bundle.pkl",
        )
    ).expanduser()
    MAX_CONTENT_LENGTH = int(
        os.environ.get("MAX_UPLOAD_BYTES", 1024 * 1024 * 1024)
    )
    MAX_FORM_PARTS = int(os.environ.get("MAX_DICOM_FILES", 12000)) + 10
    MAX_DICOM_FILES = int(os.environ.get("MAX_DICOM_FILES", 12000))
    DCM2NIIX_COMMAND = os.environ.get("DCM2NIIX_COMMAND") or (
        str(Path(sys.executable).parent / ("dcm2niix.exe" if os.name == "nt" else "dcm2niix"))
        if (Path(sys.executable).parent / ("dcm2niix.exe" if os.name == "nt" else "dcm2niix")).is_file()
        else "dcm2niix"
    )
    DICOM_CONVERSION_TIMEOUT_SECONDS = int(os.environ.get("DICOM_CONVERSION_TIMEOUT_SECONDS", 300))
    DICOM_T2_TR_MIN_MS = float(os.environ.get("DICOM_T2_TR_MIN_MS", 1500))
    DICOM_T2_TE_MIN_MS = float(os.environ.get("DICOM_T2_TE_MIN_MS", 60))
    DICOM_T1_TR_MAX_MS = float(os.environ.get("DICOM_T1_TR_MAX_MS", 1000))
    DICOM_T1_TE_MAX_MS = float(os.environ.get("DICOM_T1_TE_MAX_MS", 30))
    # One retained case contains the selected source DCMs, converted NIfTIs,
    # converter sidecars and a selection manifest. Never served as static files.
    DICOM_SELECTED_SERIES_DIR = Path(
        os.environ.get("DICOM_SELECTED_SERIES_DIR", Path(__file__).resolve().parent / "selected_mri_cases")
    ).expanduser()
    EXTRACTION_TIMEOUT_SECONDS = int(
        os.environ.get("EXTRACTION_TIMEOUT_SECONDS", 900)
    )
    MUSCLEMAP_COMMAND = os.environ.get(
        "MUSCLEMAP_COMMAND"
    ) or _find_musclemap_command("mm_extract_metrics")
    MUSCLEMAP_SEGMENT_COMMAND = os.environ.get(
        "MUSCLEMAP_SEGMENT_COMMAND"
    ) or _find_musclemap_command("mm_segment")
    MUSCLEMAP_SEGMENT_REGION = os.environ.get(
        "MUSCLEMAP_SEGMENT_REGION", "abdomen"
    )
    MUSCLEMAP_METHOD = os.environ.get("MUSCLEMAP_METHOD", "gmm")
    MUSCLEMAP_REGION = os.environ.get("MUSCLEMAP_REGION", "abdomen")
    MUSCLEMAP_COMPONENTS = int(os.environ.get("MUSCLEMAP_COMPONENTS", 3))
    # Processes for the per-muscle GMM fits of one case (8 was fastest on a
    # 20-thread CPU; more only add start-up cost).
    MUSCLEMAP_GMM_WORKERS = int(
        os.environ.get("MUSCLEMAP_GMM_WORKERS", max(1, min(8, (os.cpu_count() or 2) // 2)))
    )
    # Interpreter of the environment that owns MuscleMap's nibabel/monai stack.
    # Empty means "derive it from the mm_extract_metrics path".
    MUSCLEMAP_PYTHON = os.environ.get("MUSCLEMAP_PYTHON") or _find_musclemap_python()
    MUSCLE_SEGMENTATION_DIR = Path(
        os.environ.get("MUSCLE_SEGMENTATION_DIR", Path(__file__).resolve().parent / "segmentations_muscle")
    ).expanduser()
    # PyRadiomics usually lives in its own environment too; empty means "run
    # the extraction in this interpreter".
    RADIOMICS_PYTHON = os.environ.get("RADIOMICS_PYTHON") or _find_env_python(
        "radiomics", "SimpleITK"
    )
    # TotalSegmentator normally lives with torch and nnU-Net in its own
    # environment. Empty means "run it in this interpreter".
    TOTALSEGMENTATOR_PYTHON = os.environ.get(
        "TOTALSEGMENTATOR_PYTHON"
    ) or _find_project_venv_python(
        "totalsegmentator", "nibabel", "torch"
    ) or _find_env_python("totalsegmentator", "nibabel", "torch")
    # Optional nnU-Net weights cache. Empty uses TotalSegmentator's default
    # ~/.totalsegmentator/nnunet/results location.
    TOTALSEGMENTATOR_WEIGHTS_PATH = os.environ.get(
        "TOTALSEGMENTATOR_WEIGHTS_PATH", ""
    )
    # "gpu", "gpu:N", "cuda", or "cpu". Empty selects GPU when available.
    TOTALSEGMENTATOR_DEVICE = os.environ.get("TOTALSEGMENTATOR_DEVICE", "")
    TOTALSEGMENTATOR_REQUIRE_GPU = os.environ.get(
        "TOTALSEGMENTATOR_REQUIRE_GPU", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    TOTALSEGMENTATOR_NR_THREADS_RESAMPLING = int(
        os.environ.get("TOTALSEGMENTATOR_NR_THREADS_RESAMPLING", 1)
    )
    TOTALSEGMENTATOR_NR_THREADS_SAVING = int(
        os.environ.get("TOTALSEGMENTATOR_NR_THREADS_SAVING", 2)
    )
    # Segmentation is far slower than feature extraction, especially on CPU and
    # on the first run that downloads the weights.
    SEGMENTATION_TIMEOUT_SECONDS = int(
        os.environ.get("SEGMENTATION_TIMEOUT_SECONDS", 3600)
    )
    # Individual vertebrae ticked when the page loads. L4 and L5 are the two
    # bodies adjacent to an L4-L5 cage.
    RADIOMIC_DEFAULT_LEVELS = [
        part
        for part in os.environ.get(
            "RADIOMIC_DEFAULT_LEVELS", "L4 L5"
        ).replace(",", " ").split()
    ]
    # Every automatic VOI is kept here so it can be compared against the manual
    # segmentation the training features were extracted from.
    RADIOMIC_SEGMENTATION_DIR = Path(
        os.environ.get(
            "RADIOMIC_SEGMENTATION_DIR",
            Path(__file__).resolve().parent / "segmentations_radiomic",
        )
    ).expanduser()
    # SHAP needs a reference distribution; these are the development tables.
    SHAP_BACKGROUND_DIR = Path(
        os.environ.get(
            "SHAP_BACKGROUND_DIR", PROJECT_ROOT / "data" / "processed",
        )
    ).expanduser()
    FAE_RADIOMICS_CONFIG = Path(
        os.environ.get(
            "FAE_RADIOMICS_CONFIG",
            PROJECT_ROOT / "configs" / "fae_radiomics_all_1688.yaml",
        )
    ).expanduser()
    # Editing a template should not need a server restart, and the browser
    # should not serve a stale app.js against a newer page.
    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 0
    JSON_SORT_KEYS = False


ALLOWED_UPLOAD_SUFFIXES = {
    ".nii",
    ".nii.gz",
}
