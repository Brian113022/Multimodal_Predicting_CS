"""Load, align and audit development and external feature tables."""

from pathlib import Path
from typing import Dict, List
import re

import numpy as np
import pandas as pd


class DataFusionPipeline:
    def __init__(self, sources: Dict[str, str]):
        self.sources = sources
        self.dfs: Dict[str, pd.DataFrame] = {}
        self.common_index = None
        self.label_col = None

    def load_and_align(self):
        print("[DataFusion] Loading data...")
        loaded_indices = []
        for name, path in self.sources.items():
            try:
                df = pd.read_csv(path, index_col=0).dropna(axis=1, how="all")
            except Exception as exc:
                raise RuntimeError(f"Cannot load {name}: {path} ({exc})") from exc
            if "Label" not in df.columns:
                raise ValueError(f"Internal {name} data has no Label column: {path}")
            if df.index.has_duplicates:
                raise ValueError(f"Internal {name} data contains duplicate case IDs.")
            self.dfs[name] = df
            loaded_indices.append(df.index)
            print(f"   loaded {name}: {df.shape}")

        self.common_index = loaded_indices[0]
        for index in loaded_indices[1:]:
            self.common_index = self.common_index.intersection(index, sort=False)
        if len(self.common_index) == 0:
            raise ValueError("No common internal cases across data sources.")

        first_source = next(iter(self.sources))
        reference_label = self.dfs[first_source].loc[self.common_index, "Label"].astype(int)
        for name, df in self.dfs.items():
            source_label = df.loc[self.common_index, "Label"].astype(int)
            if not source_label.equals(reference_label):
                mismatch = reference_label.index[source_label.to_numpy() != reference_label.to_numpy()].tolist()
                raise ValueError(f"Label mismatch between {first_source} and {name}: {mismatch[:10]}")
        unexpected_labels = set(reference_label.unique()) - {0, 1}
        if unexpected_labels:
            raise ValueError(f"Internal Label must be binary 0/1; found {unexpected_labels}")

        self.label_col = reference_label
        counts = reference_label.value_counts().sort_index().to_dict()
        print(f"   aligned samples: {len(self.common_index)} | Label distribution: {counts}")

    def get_merged_dataset(self, combination_keys: List[str]) -> pd.DataFrame:
        if self.common_index is None:
            raise RuntimeError("Call load_and_align() before get_merged_dataset().")
        merged = []
        for name in combination_keys:
            df = self.dfs[name].loc[self.common_index].drop(columns=["Label"]).copy()
            df.columns = [f"{name}___{column}" for column in df.columns]
            merged.append(df)
        X = pd.concat(merged, axis=1)
        X["Label"] = self.label_col
        return X


def _normalize_external_case_id(case_id):
    """Normalize external IDs to the shared 8-digit MRN."""
    normalized = str(case_id).strip()
    normalized = re.sub(r"_20\d{6}$", "", normalized)
    prefixed_match = re.fullmatch(r"\d+_(\d{8})", normalized)
    if prefixed_match:
        normalized = prefixed_match.group(1)
    return normalized


def _read_external_source(path: Path, source_name: str):
    if not path.exists():
        raise FileNotFoundError(f"External {source_name} file not found: {path}")
    raw = pd.read_csv(path, index_col=0).dropna(axis=1, how="all")
    original_names = pd.Series(raw.index.astype(str), index=raw.index)
    normalized_index = [_normalize_external_case_id(case_id) for case_id in raw.index]
    raw.index = normalized_index
    original_names.index = normalized_index
    if raw.index.has_duplicates:
        duplicates = raw.index[raw.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate normalized external IDs in {source_name}: {duplicates}")
    return raw, original_names


def load_external_validation_dataset(
        external_dir: Path,
        internal_fusion: DataFusionPipeline,
        combination_keys: List[str],
        subsidence_threshold_mm: float = 3.0,
):
    """
    Load predictors for one modality combination and labels from Clinical.

    When Sub is available, positive class is explicitly verified as:
      Label 1 = subsidence, Sub > 3 mm
      Label 0 = non-subsidence, Sub <= 3 mm
    When only Label is available, use it as supplied and mark it as
    not independently verified in the label audit.
    Clinical is read for the labels even for Radiomics-only or Muscle-only.
    """
    source_files = {
        "Radiomics": external_dir / "radiomics_features_EXT_new.csv",
        "Clinical": external_dir / "clinical_features_EXT_new.csv",
        "Muscle": external_dir / "CSA_muscles_features_EXT_new.csv",
    }

    clinical_raw, clinical_names = _read_external_source(source_files["Clinical"], "Clinical")
    label_from_sub = None
    if "Sub" in clinical_raw.columns:
        sub_values = pd.to_numeric(clinical_raw["Sub"], errors="coerce")
        if sub_values.isna().any():
            bad_ids = sub_values.index[sub_values.isna()].tolist()
            raise ValueError(f"External Sub is missing/non-numeric for: {bad_ids}")
        label_from_sub = (sub_values > subsidence_threshold_mm).astype(int)

    if "Label" in clinical_raw.columns:
        label_series = pd.to_numeric(clinical_raw["Label"], errors="coerce")
        if label_series.isna().any():
            raise ValueError("External Label contains missing/non-numeric values.")
        label_series = label_series.astype(int)
        if label_from_sub is not None and not label_series.equals(label_from_sub):
            mismatches = label_series.index[label_series != label_from_sub].tolist()
            raise ValueError(
                "External Label is inconsistent with positive class Sub > 3 mm for: "
                f"{mismatches[:20]}"
            )
        if label_from_sub is not None:
            label_source = "Label (verified against Sub > 3 mm)"
            label_rule = f"Sub > {subsidence_threshold_mm:g} mm"
        else:
            label_source = "Provided Label (Sub unavailable; not independently verified)"
            label_rule = "Provided binary Label"
    elif label_from_sub is not None:
        label_series = label_from_sub
        label_source = "Derived from Sub > 3 mm"
        label_rule = f"Sub > {subsidence_threshold_mm:g} mm"
    else:
        raise ValueError("External Clinical data must contain Label or Sub for validation.")

    unexpected_labels = set(label_series.unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"External labels must be binary 0/1; found {unexpected_labels}")

    label_audit = pd.DataFrame({
        "Case_ID": clinical_names,
        "Normalized_Case_ID": clinical_raw.index,
        "Sub_mm": sub_values if label_from_sub is not None else np.nan,
        "Derived_Label": label_series,
        "Positive_Class": "1 = subsidence",
        "Negative_Class": "0 = non-subsidence",
        "Label_Rule": label_rule,
        "Label_Source": label_source,
    }, index=clinical_raw.index)

    prepared = {}
    print(f"\n[ExternalValidation] Sources: {combination_keys}")
    for source_name in combination_keys:
        if source_name == "Clinical":
            raw = clinical_raw
        else:
            raw, _ = _read_external_source(source_files[source_name], source_name)
            missing_ids = label_series.index.difference(raw.index, sort=False).tolist()
            extra_ids = raw.index.difference(label_series.index, sort=False).tolist()
            if missing_ids or extra_ids:
                raise ValueError(
                    f"External {source_name} IDs do not match Clinical: "
                    f"missing={missing_ids[:20]}, extra={extra_ids[:20]}"
                )
            if "Label" in raw.columns:
                source_labels = pd.to_numeric(raw["Label"], errors="coerce")
                if source_labels.isna().any():
                    bad_ids = source_labels.index[source_labels.isna()].tolist()
                    raise ValueError(
                        f"External {source_name} Label is missing/non-numeric for: "
                        f"{bad_ids[:20]}"
                    )
                source_labels = source_labels.astype(int)
                label_mismatches = source_labels.index[
                    source_labels != label_series.loc[source_labels.index]
                ].tolist()
                if label_mismatches:
                    raise ValueError(
                        f"External {source_name} Label disagrees with Clinical for: "
                        f"{label_mismatches[:20]}"
                    )

        expected_columns = [
            column for column in internal_fusion.dfs[source_name].columns if column != "Label"
        ]
        missing_columns = [column for column in expected_columns if column not in raw.columns]
        if missing_columns:
            raise ValueError(
                f"External {source_name} is missing {len(missing_columns)} required predictors: "
                f"{missing_columns[:20]}"
            )
        source_predictors = raw[expected_columns].copy()
        source_predictors.columns = [f"{source_name}___{column}" for column in expected_columns]
        prepared[source_name] = source_predictors
        print(f"   prepared {source_name}: {source_predictors.shape}")

    common_index = prepared[combination_keys[0]].index
    for source_name in combination_keys[1:]:
        common_index = common_index.intersection(prepared[source_name].index, sort=False)
    common_index = common_index.intersection(label_series.index, sort=False)
    if len(common_index) == 0:
        raise ValueError("No aligned external cases after ID normalization.")

    X_external = pd.concat(
        [prepared[source_name].loc[common_index] for source_name in combination_keys], axis=1
    )
    y_external = label_series.loc[common_index].astype(int).to_numpy()
    external_names = clinical_names.loc[common_index].to_numpy()
    aligned_audit = label_audit.loc[common_index].reset_index(drop=True)

    counts = pd.Series(y_external).value_counts().sort_index().to_dict()
    print(f"   aligned external samples: {len(common_index)} | Label distribution: {counts}")
    return X_external, y_external, external_names, aligned_audit
