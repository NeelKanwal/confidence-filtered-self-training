"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

WORK = Path("work").resolve()
RAW = WORK / "nnUNet_raw"
PLANS = WORK / "nnUNet_preprocessed"
MODELS = WORK / "nnUNet_results"
SOURCE_DATASET = "Dataset501_BreastDCE"
SOURCE_MODEL = MODELS / SOURCE_DATASET / "nnUNetTrainer__nnUNetPlans__3d_fullres"
SOURCE_FOLD = 0
SOURCE_WEIGHTS = "checkpoint_best.pth"
PSEUDO_CUTOFF = 0.70
INFERENCE_CUTOFF = 0.50


def configure_environment():
    for key, folder in (("nnUNet_raw", RAW), ("nnUNet_preprocessed", PLANS), ("nnUNet_results", MODELS)):
        os.environ[key] = str(folder)


def examinations(folder):
    cases = {}
    for first in sorted(folder.rglob("*_0000.nii.gz")):
        case = first.name.removesuffix("_0000.nii.gz")
        paths = [first.with_name(f"{case}_{phase:04d}.nii.gz") for phase in range(4)]
        if case in cases or not all(path.is_file() for path in paths):
            raise ValueError(f"Duplicate case or incomplete four-channel input: {case}")
        cases[case] = paths
    if not cases:
        raise ValueError(f"No four-channel examinations in {folder}")
    return cases


def check_grid(image, reference):
    if not image.IsSameImageGeometryAs(reference):
        raise ValueError("Volume geometry differs from the reference image")


def image_like(array, reference):
    result = sitk.GetImageFromArray(np.asarray(array))
    result.CopyInformation(reference)
    return result


def load_predictor(folder, fold=SOURCE_FOLD, weights=SOURCE_WEIGHTS):
    configure_environment()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(device=device, use_mirroring=False, perform_everything_on_device=device.type == "cuda")
    predictor.initialize_from_trained_model_folder(str(folder), (fold,), weights)
    if predictor.plans_manager.image_reader_writer_class.__name__ != "SimpleITKIO":
        raise ValueError("This release requires plans using SimpleITKIO")
    return predictor


def predict_probability(predictor, paths):
    reader = predictor.plans_manager.image_reader_writer_class()
    image, props = reader.read_images([str(p) for p in paths])
    _, prob = predictor.predict_single_npy_array(image, props, None, None, True)
    if prob.shape[0] != 2:
        raise ValueError("Expected background and lesion channels")
    return image_like(prob[1].astype(np.float32), sitk.ReadImage(str(paths[3])))
