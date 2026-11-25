"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

import math
import random
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

from study import WORK, SOURCE_MODEL, SOURCE_FOLD, SOURCE_WEIGHTS, PSEUDO_CUTOFF
from study import check_grid, examinations, image_like, load_predictor, predict_probability

MODEL_NAME = "M1"
TARGET = WORK / "adaptation" / MODEL_NAME
OUTPUT = WORK / "models" / MODEL_NAME
SEED = 0
EPOCHS = 150
WARMUP = 5
PATIENCE = 15
MIN_IMPROVEMENT = 0.0005
PEAK_LR = 1e-5
WEIGHT_DECAY = 1e-4


def cache_examples(cases, roi_dir, predictor, folder):
    cache = []
    processor = predictor.configuration_manager.preprocessor_class(verbose=False)
    for case, images in cases.items():
        probability = predict_probability(predictor, images)
        breast = sitk.ReadImage(str(roi_dir / f"{case}_mask.nii.gz"))
        check_grid(probability, breast)
        inside = binary_dilation(sitk.GetArrayFromImage(breast) > 0, iterations=2)
        pseudo = ((sitk.GetArrayFromImage(probability) >= PSEUDO_CUTOFF) & inside).astype(np.uint8)
        target = folder / f"{case}.nii.gz"
        sitk.WriteImage(image_like(pseudo, probability), str(target))
        x, y, _ = processor.run_case([str(p) for p in images], str(target), predictor.plans_manager,
                                     predictor.configuration_manager, predictor.dataset_json)
        xp, yp = folder / f"{case}_image.npy", folder / f"{case}_label.npy"
        np.save(xp, x)
        np.save(yp, y[0].clip(0, 1))
        cache.append((xp, yp))
    return cache


def patch(pair, shape, rng, device):
    x, y = (np.load(p, mmap_mode="r") for p in pair)
    starts = []
    for size, extent in zip(y.shape, shape):
        span = max(0, size - extent)
        starts.append(span // 2 if rng is None else rng.randint(0, span))
    slices = tuple(slice(s, s + n) for s, n in zip(starts, shape))
    x, y = x[(slice(None), *slices)], y[slices]
    padding = [(0, n - s) for n, s in zip(shape, y.shape)]
    x = np.pad(x, [(0, 0), *padding])
    y = np.pad(y, padding)
    return torch.from_numpy(x[None]).float().to(device), torch.from_numpy(y[None]).long().to(device)


def objective(logits, target):
    ce = F.cross_entropy(logits, target)
    if not torch.any(target):
        return 0.5 * ce
    p = logits.softmax(1)[:, 1]
    t = target.float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    ti = (tp + 1) / (tp + 0.5 * fp + 0.5 * fn + 1)
    return 0.5 * (1 - ti).clamp_min(1e-6).pow(0.75) + 0.5 * ce


def main():
    training = examinations(TARGET / "train")
    validation = examinations(TARGET / "validation")
    if training.keys() & validation.keys():
        raise ValueError("Training and validation share case identifiers")
    if OUTPUT.exists():
        raise FileExistsError(f"Choose a new output folder: {OUTPUT}")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    predictor = load_predictor(SOURCE_MODEL)
    net = predictor.network.to(predictor.device)
    shape = tuple(predictor.configuration_manager.patch_size)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent) as temp:
        from pathlib import Path
        folder = Path(temp)
        train = cache_examples(training, TARGET / "roi", predictor, folder)
        valid = cache_examples(validation, TARGET / "roi", predictor, folder)
        release = folder / "model"
        fold = release / "fold_0"
        fold.mkdir(parents=True)
        for filename in ("plans.json", "dataset.json"):
            shutil.copyfile(SOURCE_MODEL / filename, release / filename)
        saved = torch.load(SOURCE_MODEL / f"fold_{SOURCE_FOLD}" / SOURCE_WEIGHTS, map_location="cpu", weights_only=False)
        for key in ("optimizer_state", "grad_scaler_state", "logging"):
            saved.pop(key, None)
        opt = torch.optim.AdamW(net.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
        best, stale = math.inf, 0
        for epoch in range(EPOCHS):
            factor = (epoch + 1) / WARMUP if epoch < WARMUP else (1 + math.cos(math.pi * (epoch - WARMUP) / max(1, EPOCHS - WARMUP - 1))) / 2
            for group in opt.param_groups:
                group["lr"] = PEAK_LR * factor
            rng.shuffle(train)
            net.train()
            for pair in train:
                x, y = patch(pair, shape, rng, predictor.device)
                opt.zero_grad(set_to_none=True)
                loss = objective(net(x), y)
                loss.backward()
                opt.step()
            net.eval()
            losses = []
            with torch.no_grad():
                for pair in valid:
                    x, y = patch(pair, shape, None, predictor.device)
                    losses.append(objective(net(x), y).item())
            score = float(np.mean(losses))
            print(f"Epoch {epoch + 1}: validation loss {score:.6f}", flush=True)
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation loss")
            if score < best - MIN_IMPROVEMENT:
                best, stale = score, 0
                saved.update(network_weights=net.state_dict(), current_epoch=epoch + 1)
                torch.save(saved, fold / SOURCE_WEIGHTS)
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        release.rename(OUTPUT)


if __name__ == "__main__":
    main()
