"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import SimpleITK as sitk

from study import WORK, SOURCE_MODEL, SOURCE_FOLD, SOURCE_WEIGHTS, examinations, load_predictor, predict_probability

INPUT = WORK / "evaluation" / "images"
OUTPUT = WORK / "probabilities" / "baseline"
MODEL = SOURCE_MODEL
FOLD = SOURCE_FOLD
WEIGHTS = SOURCE_WEIGHTS


def run(input_dir, output_dir, model, fold, weights):
    predictor = load_predictor(model, fold, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case, paths in examinations(input_dir).items():
        probability = predict_probability(predictor, paths)
        sitk.WriteImage(probability, str(output_dir / f"{case}.nii.gz"))


if __name__ == "__main__":
    run(INPUT, OUTPUT, MODEL, FOLD, WEIGHTS)
