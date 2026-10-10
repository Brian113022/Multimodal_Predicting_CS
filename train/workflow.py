"""Coordinate the training, external validation, model export and SHAP stages."""

import numpy as np
import pandas as pd

from .data import DataFusionPipeline
from .early_fusion import train_early_fusion
from .explanations import write_training_explanations
from .feature_selection import prepare_shared_radiomics_selections
from .late_fusion import train_late_fusion
from .models import get_meta_models_map, get_models_map
from .outputs import write_training_outputs
from .settings import TrainingConfig
from .state import TrainingState


def run_training(config: TrainingConfig):
    """Run the complete pipeline with explicit input and output locations."""
    data_sources = {
        "Radiomics": str(config.data_dir / "radiomics_features_new.csv"),
        "Clinical": str(config.data_dir / "clinical_features_new.csv"),
        "Muscle": str(config.data_dir / "CSA_muscles_features_new.csv"),
    }
    combinations = {
        "Clinical": ["Clinical"],
        "Muscle": ["Muscle"],
        "Radiomics": ["Radiomics"],
        "Clinical_Radiomics": ["Clinical", "Radiomics"],
        "Clinical_Muscle": ["Clinical", "Muscle"],
        "Radiomics_Muscle": ["Radiomics", "Muscle"],
        "Full_Fusion": ["Radiomics", "Clinical", "Muscle"],
    }
    cv_folds = config.cv_folds
    # External is the only hold-out now, so turning it off would leave the run
    # with no test cohort at all. The flag stays for provenance, not as a knob.
    run_external_validation = True
    external_dir = config.data_dir / "external_validation_WF"
    if not run_external_validation:
        raise RuntimeError(
            "External validation is the only test cohort in this pipeline; "
            "disabling it would leave nothing to evaluate the winners on."
        )
    late_fusion_sets = {
        "Late_Fusion_Clinical_Radiomics": ["Clinical", "Radiomics"],
        "Late_Fusion_Clinical_Muscle": ["Clinical", "Muscle"],
        "Late_Fusion_Radiomics_Muscle": ["Radiomics", "Muscle"],
        "Late_Fusion": ["Clinical", "Muscle", "Radiomics"],
    }
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.smoke_test:
        print("SMOKE TEST: reduced tuning, bootstrap and SHAP; results are for verification only.")
    print(f"Feature tables: {config.data_dir}")
    print(f"Outputs: {output_dir}")
    print(f"Model artifacts: {config.artifacts_dir}")

    fusion = DataFusionPipeline(data_sources)
    fusion.load_and_align()
    models_map = get_models_map()
    meta_models_map = get_meta_models_map()
    if not models_map:
        raise RuntimeError("No model configuration is available.")

    # Every aligned internal case is development data. The external cohort is
    # the only hold-out, so nothing is carved out here.
    all_labels = fusion.label_col.to_numpy(dtype=int)
    cohort_audit = pd.DataFrame({
        "Case_ID": np.asarray(fusion.common_index).astype(str),
        "Label": all_labels,
        "Split": "Development",
        "Positive_Class": "1 = subsidence",
    })
    cohort_audit.to_csv(output_dir / "development_cohort_audit.csv", index=False)

    # Phase 1 is completed once before any Radiomics/Fusion model is built.
    # Fold-specific selections preserve honest CV; the final-development
    # selection is shared by the two final model branches.
    radiomics_merged = fusion.get_merged_dataset(["Radiomics"])
    X_radiomics_all = radiomics_merged.drop(columns=["Label"])
    X_radiomics_development = X_radiomics_all.copy()
    shared_radiomics_selections = prepare_shared_radiomics_selections(
        X_radiomics_development,
        y_development=all_labels,
        cv_folds=cv_folds,
        output_dir=output_dir,
    )

    state = TrainingState()

    print(
        "\nSTARTING ML9 AUC-FIRST: SINGLE/PAIRWISE/FULL EARLY FUSION "
        "+ ALL-STANDARD-MODEL LATE FUSION + AUC OPTUNA"
    )
    print(
        f"Development cohort: all {len(fusion.common_index)} aligned internal "
        "cases (no internal hold-out)."
    )
    print(
        f"External test cohort: {external_dir} "
        "(scoring only; never used for model selection)."
    )
    train_early_fusion(
        fusion, shared_radiomics_selections, models_map, combinations, config, state,
    )
    train_late_fusion(meta_models_map, late_fusion_sets, config, state)
    write_training_outputs(combinations, meta_models_map, config, state)
    write_training_explanations(late_fusion_sets, config, state)
