# Saved MuscleMap segmentations

The main page's DICOM workflow saves every selected axial T2 MRI and the exact
MuscleMap mask used for metrics in one folder per run. Set
`MUSCLE_SEGMENTATION_DIR` to change this location.

```text
<YYYYmmdd-HHMMSS>_<run id>/
  metadata.json                      Source selection, labels, voxel counts and worker reports
  metrics.csv                        All per-series metric rows before aggregation
  features.json                      The selected features sent to the Muscle model
  series_0001/
    image.nii.gz                     Converted source MRI, unchanged
    image_dseg.nii.gz                MuscleMap anatomical label mask
    muscle_segmentation_preview.jpg  Representative axial overlays, when available
    *_results.csv                    Per-series MuscleMap metrics
    _squeezed/                       Metric inputs and outputs for single-slice cases, if needed
  series_0002/
    ...
```

`metadata.json` connects each mask and metrics CSV to the original converted
series and the DICOM `selection.json`. The selection manifest also records this
segmentation directory. Worker reports preserve the two-slice retry, squeezing
and GMM fallback details. Previews do not change the masks or metrics.

Masks remain saved if later feature extraction fails. Preview failures do not
prevent feature extraction. Only a fully extracted case gets `metrics.csv` and
`features.json`; individual series may already have results before a later
series fails.

The page displays the saved paths, downloads and muscle overlays alongside the
extracted features. File links are signed, expire after 24 hours and become
invalid when the server restarts; the files themselves remain on disk. These
patient artifacts are excluded from version control.
