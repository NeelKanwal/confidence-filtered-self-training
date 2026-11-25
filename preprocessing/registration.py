"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import SimpleITK as sitk
from .image_geometry import ants_to_sitk, sitk_to_ants

if TYPE_CHECKING:
    from .mri_preprocessing import Config


def _ants_metric_names(metric: str) -> Tuple[str, str]:
    key = metric.strip().lower()
    if key in {"mi", "mattes", "mutualinformation", "mutual_information"}:
        return "mattes", "mattes"
    if key in {"cc", "crosscorrelation", "cross_correlation"}:

        return "mattes", "CC"
    if key in {"meansquares", "mean_squares", "msq"}:
        return "meansquares", "meansquares"
    raise ValueError(f"Unsupported ANTs metric '{metric}'. Use MI, CC, or MeanSquares.")


def register_ants(
    fixed: sitk.Image,
    moving: sitk.Image,
    cfg: Config,
    mask: Optional[sitk.Image],
    outprefix: str,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    import ants
    fixed_ants = sitk_to_ants(fixed)
    moving_ants = sitk_to_ants(moving)
    ants_mask = (sitk_to_ants(mask) > 0).astype("uint8") if mask is not None else None
    aff_metric, syn_metric = _ants_metric_names(cfg.ants_metric)

    reg_iterations = (30, 15, 5) if cfg.fast_mode else (60, 30, 15, 5)
    tx = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=cfg.ants_transform,
        outprefix=outprefix,
        mask=ants_mask,
        mask_all_stages=ants_mask is not None,
        aff_metric=aff_metric,
        syn_metric=syn_metric,
        reg_iterations=reg_iterations,
        random_seed=int(cfg.ants_random_seed),
        use_legacy_histogram_matching=bool(cfg.ants_histogram_matching),
        verbose=False,
    )
    warped = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=tx["fwdtransforms"],
        interpolator="bSpline",
    )
    return ants_to_sitk(warped), tx
