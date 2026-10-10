"""Results shared by the early-fusion, late-fusion and output stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import MLResults


@dataclass
class TrainingState:
    default_cv_rows: List[Dict[str, Any]] = field(default_factory=list)
    screening_cv_rows: List[Dict[str, Any]] = field(default_factory=list)
    tuned_cv_rows: List[Dict[str, Any]] = field(default_factory=list)
    final_cv_rows: List[Dict[str, Any]] = field(default_factory=list)
    selected_model_rows: List[Dict[str, Any]] = field(default_factory=list)
    external_summary_rows: List[Dict[str, Any]] = field(default_factory=list)
    external_prediction_rows: List[Dict[str, Any]] = field(default_factory=list)
    late_model_comparison_config_rows: List[Dict[str, Any]] = field(default_factory=list)
    late_model_comparison_cv_rows: List[Dict[str, Any]] = field(default_factory=list)
    late_model_comparison_external_rows: List[Dict[str, Any]] = field(default_factory=list)
    selected_oof_rows: List[Dict[str, Any]] = field(default_factory=list)
    external_label_audit: Optional[pd.DataFrame] = None
    final_results: Dict[str, MLResults] = field(default_factory=dict)
    base_oof_probability_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    base_external_probability_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    base_external_feature_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    base_winner_names: Dict[str, str] = field(default_factory=dict)
    late_winner_names: Dict[str, str] = field(default_factory=dict)
    late_fusion_run_config: Dict[str, Any] = field(default_factory=dict)
    full_late_external_inputs: Optional[pd.DataFrame] = None
    full_late_external_probability: Optional[pd.Series] = None
