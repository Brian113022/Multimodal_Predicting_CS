"""Readable clinical form metadata without changing trained feature encodings."""


# Codes follow data/processed/external_validation_WF/README.md and the
# Encoding sheet in data/raw/dictionaries/All_feature_v6.xlsx.
CLINICAL_LABELS = {
    "Level": "Operated level",
    "Age": "Age (years)",
    "Gender": "Sex",
    "BMI": "Body mass index (kg/m²)",
    "Tobacco": "Smoking",
    "Drink": "Alcohol use",
    "Diabetes": "Diabetes mellitus",
    "AI_CCI": "Age-adjusted Charlson Comorbidity Index",
    "Pre_IH": "Preoperative intervertebral height (mm)",
    "Pos_lat": "Cage position (sagittal plane)",
    "Pos_AP": "Cage position (coronal plane)",
    "Post_SA": "Postoperative segmental angle (°)",
    "Cage_height_mm": "Cage height (mm)",
    "Cage_type": "Cage type",
    "DBM": "Demineralized bone matrix use",
    "Condition": "Diagnostic category",
}

CLINICAL_CHOICE_LABELS = {
    "Level": {level: f"L{level}-L{level + 1}" for level in range(1, 5)},
    "Gender": {0: "Female", 1: "Male"},
    "Tobacco": {0: "No", 1: "Yes"},
    "Drink": {0: "No", 1: "Yes"},
    "Diabetes": {0: "No", 1: "Yes"},
    "Pos_lat": {0: "Anterior", 1: "Posterior"},
    "Pos_AP": {0: "Left", 1: "Center", 2: "Right"},
    "Cage_type": {0: "PEEK", 1: "Bullet", 2: "Diamond", 3: "Expandable"},
    "DBM": {0: "No", 1: "Yes"},
    "Condition": {
        0: "Herniation / stenosis",
        1: "Spondylolisthesis I",
        2: "Spondylolisthesis II",
        3: "Spondylolisthesis III",
        4: "Spondylolisthesis IV",
    },
}

CLINICAL_HELP = {
    "BMI": {
        "term": "BMI",
        "description": "Body mass index, measured in kg/m².",
    },
    "AI_CCI": {
        "term": "Age-adjusted Charlson Comorbidity Index",
        "description": "Age-adjusted comorbidity score. Enter a whole number from 0 to 10.",
    },
    "Pre_IH": {
        "term": "Preoperative intervertebral height",
        "description": "Segmental intervertebral height before surgery, measured in millimeters (Pre-IH).",
    },
    "Post_SA": {
        "term": "Postoperative segmental angle",
        "description": "Segmental lordosis angle on immediate postoperative X-rays, measured in degrees (Post-SA).",
    },
    "Pos_lat": {
        "term": "Sagittal cage position",
        "description": "Anterior or posterior on immediate postoperative lateral X-rays.",
    },
    "Pos_AP": {
        "term": "Coronal cage position",
        "description": "Left, center, or right on immediate postoperative anteroposterior X-rays.",
    },
    "Cage_type": {
        "term": "PEEK",
        "description": "Polyetheretherketone. Other cage types are bullet, diamond, and expandable.",
    },
    "DBM": {
        "term": "Demineralized bone matrix",
        "description": "DBM: whether demineralized bone matrix was used during surgery.",
    },
    "Condition": {
        "term": "Diagnostic category",
        "description": "Disc herniation or stenosis without listhesis, or spondylolisthesis. Roman numerals indicate the Meyerding grade of spondylolisthesis.",
    },
}


def describe_clinical_fields(fields):
    """Enrich a copy of the bundle schema; retain its order and numeric codes."""
    described = []
    for original in fields:
        field = dict(original)
        source = field.get("source_name") or field["name"].removeprefix("Clinical___")
        field["label"] = CLINICAL_LABELS.get(source, field["label"])
        if source in CLINICAL_HELP:
            field["help"] = dict(CLINICAL_HELP[source])
        if field["input_type"] == "select" and source in CLINICAL_CHOICE_LABELS:
            labels = CLINICAL_CHOICE_LABELS[source]
            # The bundle lists observed levels (2–4); the form also permits
            # L1-L2, retaining the same cranial-vertebra numeric encoding.
            values = list(labels) if source == "Level" else field.get("choices", [])
            field["choices"] = [
                {"value": value, "label": labels.get(value, str(value))}
                for value in values
            ]
        if source == "AI_CCI":
            # The requested entry range is distinct from the observed training
            # range, which remains part of the original artifact metadata.
            field.update(min=0, max=10, step=1, value_type="integer")
        described.append(field)
    return described
