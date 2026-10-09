"""Publication labels for every SHAP figure of the ML9 late-fusion analysis.

The modelling frames carry programming names (``Clinical___Pos_lat``,
``Muscle___aVol_MF_L``). A figure for a clinical journal has to carry clinical
terms instead, so this module holds the one mapping every SHAP figure uses.

Conventions, applied uniformly:

* clinical term first, never the column name;
* sentence case, not Title Case;
* units in parentheses - ``(mm)``, ``(°)``, ``(kg/m²)``, ``(years)``;
* side as ``(L)`` / ``(R)``;
* ``aVol`` is reported as AMV (adjusted muscle volume) and ``tFat`` as FI (fat
  infiltration) - never as CSA;
* binary variables are named after their high value, so a warm point on the
  beeswarm reads directly: ``Pos_lat`` is 0 anterior / 1 posterior, hence
  **Posterior cage position**.

``IH``, ``SA``, ``AMV``, ``FI``, ``L``, ``R`` and ``NGTDM`` are deliberately
left as abbreviations on the axis and are defined once in the figure legend;
see :data:`FIGURE_LEGEND_ABBREVIATIONS`.
"""

from __future__ import annotations


# Clinical and surgical variables. Coding for the categorical ones follows
# MANUSCRIPT_PLAN_v1: Condition = Meyerding grade (0 HIVD/stenosis, 1 grade I,
# 2 grade II), Pos_lat = sagittal (0 anterior, 1 posterior), Pos_AP = coronal
# (0 left, 1 central, 2 right).
CLINICAL_LABELS = {
    "Clinical___Level": "Operated level",
    "Clinical___Age": "Age (years)",
    "Clinical___Gender": "Sex",
    "Clinical___BMI": "BMI (kg/m²)",
    "Clinical___Tobacco": "Smoking",
    "Clinical___Drink": "Alcohol use",
    "Clinical___Diabetes": "Diabetes",
    "Clinical___AI_CCI": "AI-CCI",
    "Clinical___Pre_IH": "Preoperative IH (mm)",
    "Clinical___Pos_lat": "Posterior cage position",
    "Clinical___Pos_AP": "Coronal cage position",
    "Clinical___Post_SA": "Postoperative SA (°)",
    "Clinical___Cage_height_mm": "Cage height (mm)",
    "Clinical___Cage_type": "Cage type",
    "Clinical___DBM": "Demineralized bone matrix",
    "Clinical___Condition": "Meyerding grade",
}

_MUSCLE_NAMES = {
    "ES": "Erector spinae",
    "MF": "Multifidus",
    "PM": "Psoas major",
    "QL": "Quadratus lumborum",
}
_MUSCLE_METRICS = {"aVol": "AMV", "tFat": "FI"}

MUSCLE_LABELS = {
    f"Muscle___{metric}_{muscle}_{side}": f"{name} {short} ({side})"
    for metric, short in _MUSCLE_METRICS.items()
    for muscle, name in _MUSCLE_NAMES.items()
    for side in ("L", "R")
}

# Radiomics names are collapsed to filter + statistic, dropping the modality
# prefix and the PyRadiomics class where the statistic already identifies it.
RADIOMICS_LABELS = {
    "Radiomics___MRI_original_shape_MajorAxisLength": "Shape major axis length",
    "Radiomics___MRI_gradient_firstorder_Kurtosis": "Gradient kurtosis",
    "Radiomics___MRI_gradient_glszm_ZoneEntropy": "Gradient GLSZM zone entropy",
    "Radiomics___MRI_lbp-3D-m2_firstorder_Median": "LBP-3D-m2 median",
    "Radiomics___MRI_logarithm_ngtdm_Strength": "Logarithm NGTDM strength",
    "Radiomics___MRI_squareroot_glcm_Contrast": "Square-root GLCM contrast",
    "Radiomics___MRI_wavelet-LHL_firstorder_Median": "Wavelet-LHL median",
    "Radiomics___MRI_wavelet-HHL_firstorder_Skewness": "Wavelet-HHL skewness",
}

# Meta-features of the direct late-fusion model.
META_LABELS = {
    "Clinical_Probability": "Clinical-model probability",
    "Muscle_Probability": "Muscle-model probability",
    "Radiomics_Probability": "Radiomics-model probability",
}

FEATURE_LABELS = {
    **CLINICAL_LABELS,
    **MUSCLE_LABELS,
    **RADIOMICS_LABELS,
    **META_LABELS,
}

SHAP_VALUE_AXIS_LABEL = "SHAP value (effect on predicted subsidence probability)"
FEATURE_VALUE_BAR_LABEL = "Feature value"

FIGURE_LEGEND_ABBREVIATIONS = (
    "AI-CCI, artificial-intelligence-derived age-adjusted Charlson Comorbidity "
    "Index; AMV, adjusted muscle volume; BMI, body mass index; FI, fat "
    "infiltration; GLCM, gray-level co-occurrence matrix; GLSZM, gray-level "
    "size-zone matrix; IH, intervertebral height; L, left; LBP, local binary "
    "pattern; NGTDM, neighbouring gray-tone difference matrix; R, right; SA, "
    "segmental angle; SHAP, SHapley Additive exPlanations."
)

CAGE_POSITION_CODING_NOTE = (
    "Sagittal cage position was coded as 0, anterior; 1, posterior."
)


def label_for(name: str) -> str:
    """Return the publication label for one feature.

    Anything with no mapping is returned unchanged. That is how the row
    beeswarm writes for the features past the top N ("Sum of 28 other
    features") keeps SHAP's own wording: it is the one row that is not a
    clinical variable, so it is left as the reader expects to see it.
    """
    return FEATURE_LABELS.get(name, name)


def relabel_axes(ax) -> None:
    """Rewrite one beeswarm axis in place: y-tick rows and the SHAP x label."""
    ax.set_yticklabels(
        [label_for(tick.get_text()) for tick in ax.get_yticklabels()]
    )
    ax.set_xlabel(SHAP_VALUE_AXIS_LABEL, fontsize=13)


def unlabelled(names) -> list[str]:
    """Names with no mapping - a guard so a new feature cannot slip through."""
    return [name for name in names if name not in FEATURE_LABELS]
