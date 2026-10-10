"""Training-fold scaling and class-imbalance handling."""

from typing import Any, List

from fusion_subsidence.model_components import SelectiveStandardScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SklearnPipeline
import numpy as np

from .optional_dependencies import IMBLEARN_AVAILABLE, ImbPipeline, SMOTE, SMOTENC


CLINICAL_CATEGORICAL_BASE_COLUMNS = {
    "Level",
    "Gender",
    "Tobacco",
    "Drink",
    "Diabetes",
    "Pos_lat",
    "Pos_AP",
    "Cage_type",
    "DBM",
    "Condition",
}


def get_categorical_columns(columns) -> List[str]:
    expected = {f"Clinical___{column}" for column in CLINICAL_CATEGORICAL_BASE_COLUMNS}
    return [column for column in columns if column in expected]


SAMPLER_PREFERENCE = {"none": 2, "class_weight": 1, "smote": 0}


def get_sampler_choices(estimator_template: SklearnPipeline) -> List[str]:
    """Return valid imbalance strategies for an estimator, simplest first."""
    estimator_name, _ = estimator_template.steps[-1]
    choices = ["none"]
    if estimator_name in {"rf", "svm", "lr", "lgbm", "xgb"}:
        choices.append("class_weight")
    if IMBLEARN_AVAILABLE:
        choices.append("smote")
    return choices


def build_training_pipeline(
        estimator_template: SklearnPipeline,
        feature_columns,
        sampler_choice: str = "none",
        y_train=None,
) -> Any:
    """Build a leakage-safe preprocessing/model pipeline for one sampler choice."""
    if sampler_choice not in {"none", "class_weight", "smote"}:
        raise ValueError(f"Unknown sampler choice: {sampler_choice}")
    categorical_columns = get_categorical_columns(feature_columns)
    model_steps = [(name, clone(model)) for name, model in estimator_template.steps]
    preprocessing_steps = [
        ("pre", SelectiveStandardScaler(categorical_columns=categorical_columns))
    ]
    if sampler_choice == "smote":
        if not IMBLEARN_AVAILABLE:
            raise ImportError("sampler='smote' requires imbalanced-learn.")
        if categorical_columns:
            sampler = SMOTENC(
                categorical_features=categorical_columns,
                sampling_strategy="auto",
                random_state=42,
                k_neighbors=5,
            )
        else:
            sampler = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
        pipeline = ImbPipeline([
            *preprocessing_steps,
            ("smote", sampler),
            *model_steps,
        ])
    else:
        pipeline = SklearnPipeline([*preprocessing_steps, *model_steps])

    if sampler_choice == "class_weight":
        if y_train is None:
            raise ValueError("class_weight requires the current training labels.")
        estimator_name = model_steps[-1][0]
        if estimator_name == "xgb":
            counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
            if counts[1] == 0:
                raise ValueError("XGBoost class weighting requires both classes.")
            pipeline.set_params(
                **{"xgb__scale_pos_weight": float(counts[0] / counts[1])}
            )
        elif estimator_name in {"rf", "svm", "lr", "lgbm"}:
            pipeline.set_params(**{f"{estimator_name}__class_weight": "balanced"})
        else:
            raise ValueError(
                f"class_weight is unsupported for estimator step '{estimator_name}'."
            )
    return pipeline
