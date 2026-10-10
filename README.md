# Explainable multimodal machine learning for predicting cage subsidence after lumbar interbody fusion

Source code for the manuscript *"Explainable Multimodal Machine Learning for
Predicting Cage Subsidence after Lumbar Interbody Fusion: Development and
External Validation"* (under review). The repository contains:

- the training and evaluation pipeline;
- the browser-based application that estimates the individual risk of cage
  subsidence and explains each prediction with SHAP.

> **Research use only.** The model has not been prospectively validated and
> must not be used to guide patient care.

## Daily use

After completing the one-time setup below and placing the feature tables in
`data/processed/`, run either command from the repository root:

```powershell
# Train and export the models.
.venv\Scripts\python.exe train/train.py

# Start the web app using the exported models.
.venv\Scripts\python.exe -m webapp.app
```

For the web app, open [http://127.0.0.1:5050](http://127.0.0.1:5050).
Training and serving are separate commands; an existing trained model lets you
start the web app without training again.

The web app automatically starts MuscleMap, TotalSegmentator and PyRadiomics
workers when MRI processing is requested and reads their results. You do not
need to start separate services during
daily use. Their dependencies must already be installed and their interpreters
must be discoverable or configured as described below.

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
│   ├── train.py                    CLI entry point
│   ├── workflow.py, settings.py    Workflow and configurable run paths
│   ├── data.py, feature_selection.py, preprocessing.py
│   ├── models.py, cross_validation.py, metrics.py
│   ├── early_fusion.py, late_fusion.py, state.py
│   ├── export.py, outputs.py       Model export and evaluation outputs
│   ├── publication*.py             Evaluation tables and figures
│   └── shap*.py, explanations.py   SHAP explanations and display names
├── fusion_subsidence/      Model components imported by the saved models
├── configs/
│   └── fae_radiomics_all_1688.yaml Web extraction settings matching training CSVs
├── data/                   Input tables (not included, see data/README.md)
├── artifacts/              Fitted models written by train.py (not included)
├── outputs/                Training results (created at run time)
├── deployment/             Launchers, supervisor, Cloudflare setup and .env
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

## One-time setup

Install packages with pip. Conda is not required. If your existing Python
environment already has the dependencies, use it directly; `.venv` below is
an optional way to keep the project's packages separate from other projects.

Training reads the existing feature CSVs and does not run MRI segmentation or
feature extraction. MRI processing in the web app additionally needs
PyRadiomics, MuscleMap and TotalSegmentator installed.

**Install the training and web app dependencies.** On Windows, optionally
create and activate a virtual environment first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu118 True
```

`requirements.txt` pins the CUDA 11.8 build of PyTorch. On a machine without
an NVIDIA GPU, replace the `torch` and `torchvision` lines with CPU builds.

**PyRadiomics for MRI processing.** It is currently excluded from
`requirements.txt`. The previously tested extraction setup used PyRadiomics
3.1.0 with Python 3.9 and `numpy<2`, while the training results were verified
with NumPy 2.2.6. A single installation combining these dependencies has not
been validated here. Installing only `requirements.txt` therefore does not
complete the MRI extraction setup.

The radiomics worker can use the same Python as the web app when that Python
has a compatible PyRadiomics installation. A separate interpreter is a
dependency-compatibility option, not a requirement to use conda or launch
radiomics manually. If using a separate pip/venv installation, set
`RADIOMICS_PYTHON` to that environment's Python executable. Otherwise it falls
back to the web app's interpreter when no alternate environment is detected.

**Install MuscleMap once (for MRI processing).** Follow the
[MuscleMap installation guide](https://github.com/MuscleMap/MuscleMap).
The web app invokes its segmentation and metric
extraction commands automatically when MRI processing is requested.

**Overriding environment detection.** If the environments are not found
automatically, set these variables in the environment used to start the web app
to their interpreters or executables:

- `RADIOMICS_PYTHON`
- `MUSCLEMAP_PYTHON`, `MUSCLEMAP_COMMAND` and `MUSCLEMAP_SEGMENT_COMMAND`
- `TOTALSEGMENTATOR_PYTHON`

## Data

Patient data are **not** included. Place the six feature tables in
`data/processed/` as described in [`data/README.md`](data/README.md).
The data are available from the corresponding author on reasonable request,
subject to institutional review board approval.

## Running the training pipeline

Use the Python environment with the training dependencies installed. This
pipeline reads already-extracted features from CSV and does not call the MRI
tools. If you installed into an existing environment, skip `.venv` activation.

```powershell
.venv\Scripts\Activate.ps1
python train/train.py
```

The script resolves every path from its own location, so it can be started
from any directory when invoked by its path. `train.py` remains the entry point;
the supporting modules separate data preparation, feature selection, model
training, evaluation, model export and SHAP. The default run settings are
unchanged: five-fold CV and 50 Optuna trials per shortlisted model.

To use a different dataset or write results elsewhere, pass paths at launch:

```powershell
python train/train.py --data-dir "D:\my_dataset\data" --output-dir "outputs\my_run" --artifacts-dir "outputs\my_run\artifacts"
```

`--data-dir` accepts either `data/` containing `processed/`, or the processed
folder itself. Both development and external-validation tables are required.
Relative override paths are resolved from the working directory. Without an
`--artifacts-dir` override, a full run still exports to the repository's
`artifacts/` folder.

For an end-to-end verification run with temporary outputs:

```powershell
python train/train.py --data-dir "D:\my_dataset\data" --output-dir "outputs\smoke_review" --smoke-test
```

The smoke test keeps all six available model algorithms, all 11 model
configurations and five-fold CV. It reduces Optuna to one trial per shortlisted
model, bootstrap to 100 iterations, and the SHAP sample/background sizes.
Required high/low-probability pairs can increase the SHAP case count above the
four-case sampling target. Its model artifacts go inside the temporary output
folder by default. Smoke-test results are for checking execution; use a full
run for the final analysis. The smaller tuning budget can select different base
models and change the late-fusion inputs and external AUC. To compare with a
completed reference run, use the same trial budget. `--n-trials` can override
the tuning budget.

**Checking a full run against reference results.** The optional regression tests
compare the model choices, training budget, and every development OOF and
external prediction. Supply the reference and new output folders at launch:

```powershell
$env:TRAIN_REFERENCE_CSV = "D:\reference_run\csv_files"
$env:TRAIN_REPRODUCED_DIR = "outputs\my_full_run"
python -m unittest discover -s tests -p test_training_reference.py -v
```

These tests are skipped when the two paths are not supplied. Reference patient
data and machine-specific paths are not stored in the test source.

The default full run writes two sets of files:

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

Start only the web app from the main environment. It launches the MRI workers
itself when needed, using the interpreters found during environment detection
or supplied through the override variables above.

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

For public deployment with Cloudflare and automatic process recovery, use
`deployment\run.bat` (Windows) or `sh deployment/run.sh` (macOS). Configuration
is in `deployment/.env`; see [deployment/README.md](deployment/README.md) for
status, stop/restart, optional startup registration and recovery limits.
The direct `python -m webapp.app` command above runs without a supervisor.

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
