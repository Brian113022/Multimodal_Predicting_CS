# Automatic Radiomics segmentations

Every Radiomics upload that does **not** come with a VOI mask is segmented with
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), and the whole
run is kept here so the automatic VOI can be compared with a manual one.

One folder per run, named `<YYYYmmdd-HHMMSS>_<image name>`:

```text
20260908-143012_case001/
  case001_image.nii            The uploaded MRI, unchanged
  case001_totalsegmentator_body.nii    Body mask from total_mr
  case001_totalsegmentator_levels.nii  Selected vertebrae_mr levels
  case001_voi_L4-L5_body.nii           Intersected VOI for PyRadiomics, label 1
  case001_voi_preview.jpg               Final VOI preview
  metadata.json                         Counts, grids, versions and GPU trace
```

The body file is TotalSegmentator `total_mr` class `vertebrae`. The levels file
contains the selected `vertebrae_mr` masks with their task label indices. The
VOI is `body AND union(levels)`, rewritten to label `1` on the uploaded image
grid because that is what `configs/fae_radiomics_all_1688.yaml` extracts.

## Comparing against a manual VOI

Both files sit on the uploaded image's grid, so a manual mask of the same case
can be loaded next to them without resampling. The count of overlapping voxels
is the quickest check:

```python
import nibabel as nib
import numpy as np

auto = np.asanyarray(nib.load("case001_voi_L4-L5_body.nii", mmap=False).dataobj) > 0
manual = np.asanyarray(nib.load("manual_voi.nii.gz").dataobj) > 0
dice = 2 * (auto & manual).sum() / (auto.sum() + manual.sum())
print(f"Dice {dice:.3f}")
```

Also inspect the body and level masks separately. A plausible final VOI is not
enough if either model assigned the wrong vertebral level or missed a boundary.

## Data protection

These files hold patient images. The folder is listed in `.gitignore` and must
not be committed or shared. Delete the runs you no longer need; nothing in the
web application deletes them automatically. The page reads the VOI preview and
provides downloads through signed extraction links, which expire after 24
hours or when the server restarts.
