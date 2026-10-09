# Explainable multimodal machine learning for predicting cage subsidence after lumbar interbody fusion

Source code for the manuscript *"Explainable Multimodal Machine Learning for
Predicting Cage Subsidence after Lumbar Interbody Fusion: Development and
External Validation"* (under review). The repository contains:

- the training and evaluation pipeline;
- the browser-based application that estimates the individual risk of cage
  subsidence and explains each prediction with SHAP.

> **Research use only.** The model has not been prospectively validated and
> must not be used to guide patient care.

## Overview

The model predicts cage subsidence after single-level minimally invasive
transforaminal lumbar interbody fusion (MIS-TLIF) with a single cage.
Subsidence is defined as a loss of more than 3 mm in mean intervertebral
height between the immediate postoperative radiograph and follow-up at 12
months or later. The model combines three predictor domains:

| Domain              | Source                                                                                                                                                                                       | Predictors          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Clinical-surgical   | Medical records, preoperative and immediate postoperative radiographs                                                                                                                        | 16                  |
| Paraspinal muscle   | Axial T2-weighted MRI, segmented with[MuscleMap](https://github.com/MuscleMap/MuscleMap); average muscle volume and fatty infiltration of 8 muscles                                           | 16                  |
| Vertebral radiomics | Sagittal T1-weighted MRI; the two vertebral bodies adjacent to the fused level, segmented with[TotalSegmentator MRI](https://github.com/wasserth/TotalSegmentator); features from PyRadiomics | 1,688 → 8 retained |

The pipeline (`train/train.py`) runs these steps:

1. **Alignment.** It links the three domain tables by case ID and keeps
   patients with complete data in every domain.
2. **Radiomic feature selection.** Inside each training fold it drops
   non-finite and near-zero-variance features, applies a correlation filter
   (|r| > 0.90), then runs L1-logistic regression with its penalty chosen by
   inner cross-validation. The final model keeps features selected in at least
   3 of 5 folds.
3. **Model selection per configuration.** It runs 11 configurations: three
   single-domain models, four early-fusion models and four late-fusion models.
   For each configuration it:
   - screens RF, KNN, SVM, LR, LightGBM and XGBoost;
   - tries three class-imbalance strategies: none, class weights and SMOTE
     inside the training fold;
   - tunes the two best candidates with Optuna;
   - selects by stratified five-fold cross-validated AUC.
4. **Late fusion.** It stacks the out-of-fold probabilities of the three domain
   models into one of six candidate meta-learners. A random forest was
   selected.
5. **External validation.** It refits the domain models on the full
   development cohort and locks the model together with its Youden-index
   threshold. It then scores the external cohort once, without refitting or
   recalibration. It reports:
   - discrimination, as AUC with a bootstrap 95% CI;
   - the Brier score;
   - calibration-in-the-large and calibration slope;
   - threshold metrics;
   - decision curves.
6. **Explanation.** End-to-end Kernel SHAP passes every prediction back
   through the domain models and the meta-learner to the 40 original
   predictors.
7. **Deployment.** It exports the fitted models to `artifacts/` for the
   web application.

## Repository structure

```text
.
├── requirements.txt            Main environment (training + web app)
├── train/
│   ├── train.py                    Full training / evaluation pipeline
│   └── shap_publication_labels.py  Display names for the 40 predictors
├── fusion_subsidence/      Model components imported by the saved models
├── configs/
│   └── fae_radiomics_all_1688.yaml PyRadiomics settings used for training
├── data/                   Input tables (not included, see data/README.md)
├── artifacts/              Fitted models written by train.py (not included)
├── outputs/                Training results (created at run time)
└── webapp/                 Flask application
    ├── app.py, inference.py, explain.py, config.py
    ├── extractors/         DICOM conversion, MuscleMap, TotalSegmentator,
    │                       PyRadiomics workers
    ├── templates/, static/
    ├── selected_mri_cases/     Run-time DICOM copies (git-ignored)
    ├── segmentations_muscle/   Run-time muscle masks (git-ignored)
    └── segmentations_radiomic/ Run-time vertebral VOIs (git-ignored)
```

## Hardware requirements

|      | Minimum                                                                                                      | Tested configuration                        |
| ---- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| OS   | Windows 10/11 or Linux (64-bit)                                                                              | Windows 11 Pro                              |
| CPU  | 4 cores                                                                                                      | Intel Core i5-14500 (14 cores / 20 threads) |
| RAM  | 16 GB                                                                                                        | 32 GB                                       |
| GPU  | Not needed for training. The web app needs an NVIDIA GPU with ≥ 4 GB VRAM and a driver supporting CUDA 11.8 | NVIDIA GeForce RTX 4070, 12 GB              |
| Disk | ~15 GB (Python environments, PyTorch, TotalSegmentator weights)                                              |                                             |

Notes on run time and the GPU:

- **Training** runs on the CPU. A full run takes several hours on the tested
  machine, most of it spent on Optuna tuning and Kernel SHAP.
- **The web app's vertebral segmentation** runs TotalSegmentator `total_mr`
  and `vertebrae_mr` together on the GPU, using about 2 GB of VRAM each. On
  the tested GPU one sagittal T1 series takes about 30 s.
- **Running without a GPU.** Set `TOTALSEGMENTATOR_REQUIRE_GPU=0` to allow CPU
  segmentation. It works, but takes several minutes per case.

## Software environments

Three Python environments are used. Each heavy imaging tool lives in its own
environment, which avoids dependency conflicts. The web app finds each
environment automatically.

| Environment           | Purpose                                    | Python                | Install                                                                         |
| --------------------- | ------------------------------------------ | --------------------- | ------------------------------------------------------------------------------- |
| main (`.venv`)      | Training, web app, TotalSegmentator        | 3.10 (tested 3.10.11) | `pip install -r requirements.txt`                                             |
| `radiomm` (conda)   | PyRadiomics feature extraction             | 3.9                   | `pyradiomics 3.1.0`, `SimpleITK`, `PyYAML`, `numpy<2`                   |
| `MuscleMap` (conda) | Paraspinal muscle segmentation and metrics | per MuscleMap         | follow the[MuscleMap installation guide](https://github.com/MuscleMap/MuscleMap) |

**Main environment.** On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu118 True
```

`requirements.txt` pins the CUDA 11.8 build of PyTorch. On a machine without
an NVIDIA GPU, replace the `torch` and `torchvision` lines with CPU builds.

**Radiomics environment.**

```bash
conda create -n radiomm python=3.9 -y
conda activate radiomm
pip install "numpy<2" PyYAML SimpleITK
pip install pyradiomics==3.1.0
```

PyRadiomics is deliberately kept out of the main environment, because it
requires `numpy<2`.

**Overriding environment detection.** If the environments are not found
automatically, set these variables to their interpreters or executables:

- `RADIOMICS_PYTHON`
- `MUSCLEMAP_PYTHON`, `MUSCLEMAP_COMMAND` and `MUSCLEMAP_SEGMENT_COMMAND`
- `TOTALSEGMENTATOR_PYTHON`

## Data

Patient data are **not** included. Place the six feature tables in
`data/processed/` as described in [`data/README.md`](data/README.md).
The data are available from the corresponding author on reasonable request,
subject to institutional review board approval.

## Running the training pipeline

```powershell
.venv\Scripts\Activate.ps1
python train/train.py
```

The script resolves every path from its own location, so it can be started
from any directory. It writes two sets of files:

- `outputs/experiments/2026-09-13_ml9_dev_all_external_test/` holds:
  - the cross-validation results;
  - the selected models and their thresholds;
  - the external validation metrics;
  - the ROC, calibration and decision curves;
  - the SHAP figures.
- `artifacts/` holds the deployment files: `late_fusion_bundle.pkl`, its
  manifest, and one file per prediction route.

The script uses fixed random seeds (bootstrap seed 20260809). Kernel SHAP
values can differ slightly between runs.

**Treat outputs as confidential.** Prediction and SHAP files are named by case
ID.

## Running the web application

The web app needs a trained model in `artifacts/`. Either run the training
pipeline or copy the authors' artifacts into that folder. For SHAP, the web
app also uses the development tables in `data/processed/` as its
reference distribution.

```powershell
.venv\Scripts\Activate.ps1
python -m webapp.app
```

Then open [http://127.0.0.1:5050](http://127.0.0.1:5050). Set `FLASK_HOST` and `FLASK_PORT` to change
the address or port. Using the app:

1. **Enter the clinical-surgical variables.** These are always required.
2. **Optionally add MRI.** Select the DICOM files of one patient and the
   operated level. The app then:
   - classifies the series, picking axial T2 for the muscles and sagittal T1
     for radiomics;
   - converts them with dcm2niix;
   - runs MuscleMap and TotalSegmentator;
   - extracts muscle and radiomic features.
3. **Run the prediction.** The app returns the predicted probability of
   subsidence. For late-fusion routes, it also returns an end-to-end SHAP
   explanation.

The full model published in the manuscript is the
clinical + muscle + radiomics late-fusion route. The clinical, clinical +
muscle and clinical + radiomics routes are supporting models.

**Run-time files are patient data.** The selected DICOMs, the converted images
and the masks are kept under `webapp/selected_mri_cases/`,
`webapp/segmentations_muscle/` and `webapp/segmentations_radiomic/`.
These folders are git-ignored. Protect and clean them up according to your
institution's policy.

**Optional settings.** Configure these through environment variables, all
read in `webapp/config.py`:

- `MODEL_BUNDLE_PATH`
- `SHAP_BACKGROUND_DIR`
- `FAE_RADIOMICS_CONFIG`
- `TOTALSEGMENTATOR_DEVICE` and `TOTALSEGMENTATOR_REQUIRE_GPU`
- `MAX_UPLOAD_BYTES` and `MAX_DICOM_FILES`
- the DICOM timing thresholds (`DICOM_T2_TR_MIN_MS` and others)

## Data protection

The `.gitignore` excludes every patient-derived file:

- feature tables;
- DICOM and NIfTI images, masks and previews;
- training outputs;
- fitted model files.

The fitted KNN and SVM base learners store training samples. Model artifacts
are therefore shared only on request, not in this repository.

## Citation

Citation details will be added after publication.

## License

To be added.

## Contact

Corresponding authors:

- Meng-Huang Wu, Department of Orthopedics, Taipei Medical University Hospital
- Nguyen-Quoc-Khanh Le, In-Service Master Program in Artificial Intelligence
  in Medicine, Taipei Medical University
