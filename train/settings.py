"""Run paths and settings, independent of any particular test dataset."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "experiments" / "2026-09-13_ml9_dev_all_external_test"
)
FEATURE_TABLES = (
    "clinical_features_new.csv",
    "CSA_muscles_features_new.csv",
    "radiomics_features_new.csv",
)
EXTERNAL_FEATURE_TABLES = (
    "clinical_features_EXT_new.csv",
    "CSA_muscles_features_EXT_new.csv",
    "radiomics_features_EXT_new.csv",
)


def resolve_data_dir(path: Path) -> Path:
    """Accept either the processed-table folder or its parent data folder."""
    path = Path(path).expanduser().resolve()
    if not all((path / name).is_file() for name in FEATURE_TABLES):
        if (path / "processed").is_dir():
            path = path / "processed"
    required = [path / name for name in FEATURE_TABLES]
    required.extend(path / "external_validation_WF" / name for name in EXTERNAL_FEATURE_TABLES)
    missing = [str(candidate) for candidate in required if not candidate.is_file()]
    if missing:
        raise FileNotFoundError("Missing required feature tables:\n" + "\n".join(missing))
    return path


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    artifacts_dir: Path
    n_trials: int = 50
    smoke_test: bool = False
    cv_folds: int = 5
    top_models_to_tune: int = 2
    top_meta_models_to_tune: int = 2
    shap_max_cases: Optional[int] = None
    shap_background_size: int = 20
    shap_nsamples: int = 200
    bootstrap_iterations: int = 2000

    @classmethod
    def create(
        cls,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        n_trials: Optional[int] = None,
        smoke_test: bool = False,
    ) -> "TrainingConfig":
        trials = n_trials if n_trials is not None else (1 if smoke_test else 50)
        if trials < 1:
            raise ValueError("--n-trials must be at least 1.")
        processed_dir = resolve_data_dir(data_dir if data_dir is not None else DATA_DIR)
        if output_dir is None:
            output_dir = (
                PROJECT_ROOT / "outputs" / "smoke_tests" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if smoke_test else DEFAULT_OUTPUT_DIR
            )
        output_dir = Path(output_dir).expanduser().resolve()
        if artifacts_dir is None:
            artifacts_dir = output_dir / "artifacts" if smoke_test else PROJECT_ROOT / "artifacts"
        artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        return cls(
            data_dir=processed_dir,
            output_dir=output_dir,
            artifacts_dir=artifacts_dir,
            n_trials=trials,
            smoke_test=smoke_test,
            shap_max_cases=4 if smoke_test else None,
            shap_background_size=5 if smoke_test else 20,
            shap_nsamples=100 if smoke_test else 200,
            bootstrap_iterations=100 if smoke_test else 2000,
        )
