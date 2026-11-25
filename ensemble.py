"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.special import expit, logit

from study import WORK, INFERENCE_CUTOFF, check_grid, image_like, examinations, load_predictor, predict_probability

INPUT = WORK / "evaluation" / "images"
ROI = WORK / "evaluation" / "roi"
NATIVE = WORK / "evaluation" / "native"
OUTPUT = WORK / "ensemble"
MODEL_DIRS = [WORK / "models" / name for name in ("M1", "M2", "M3")]
FOLD = 0
WEIGHTS = "checkpoint_best.pth"
MIN_VOLUME_MM3 = 8.0
MIN_ROI_FRACTION = 0.01


def fuse(maps):
    if len(maps) != 3:
        raise ValueError("Expected three probability maps")
    for arr in maps:
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("Expected finite probabilities in [0, 1]")
    odds = [logit(np.clip(arr.astype(np.float64), 1e-7, 1 - 1e-7)) for arr in maps]
    return expit(np.mean(odds, axis=0)).astype(np.float32)


def components(mask, voxel_volume, roi=None):
    regions, _ = label(mask, structure=generate_binary_structure(3, 1))
    counts = np.bincount(regions.ravel())
    keep = counts * voxel_volume >= MIN_VOLUME_MM3
    if roi is not None:
        inside = np.bincount(regions[roi], minlength=len(counts))
        keep &= inside >= counts * MIN_ROI_FRACTION
    keep[0] = False
    return keep[regions].astype(np.uint8)


def main():
    predictors = [load_predictor(model, FOLD, WEIGHTS) for model in MODEL_DIRS]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case, paths in examinations(INPUT).items():
        predictions = [predict_probability(p, paths) for p in predictors]
        reference = predictions[0]
        prob = fuse([sitk.GetArrayFromImage(p) for p in predictions])
        breast = sitk.ReadImage(str(ROI / f"{case}_mask.nii.gz"))
        check_grid(breast, reference)
        roi = binary_dilation(sitk.GetArrayFromImage(breast) > 0, iterations=2)
        mask = components(prob >= INFERENCE_CUTOFF, np.prod(reference.GetSpacing()), roi)
        native = sitk.ReadImage(str(NATIVE / f"{case}.nii.gz"))
        mapped = sitk.Resample(image_like(mask, reference), native, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        final = components(sitk.GetArrayFromImage(mapped), np.prod(native.GetSpacing()))
        sitk.WriteImage(image_like(prob, reference), str(OUTPUT / f"{case}_prob.nii.gz"))
        sitk.WriteImage(image_like(final, native), str(OUTPUT / f"{case}_mask.nii.gz"))


if __name__ == "__main__":
    main()
