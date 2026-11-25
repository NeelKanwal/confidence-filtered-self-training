"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .mri_preprocessing import Config


def global_normalize_4d(vol4d: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:


    data = np.asarray(vol4d, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.all():
        raise ValueError("Normalization requires finite input intensities")

    work = data.copy()
    stats: Dict[str, Any] = {}

    kind = cfg.normalize_type.lower()
    if kind == "zscore":
        mean = float(work[finite].mean(dtype=np.float64))
        std = float(work[finite].std(dtype=np.float64))
        if std < 1e-8:
            raise ValueError("Global standard deviation is too small for z-score normalization.")
        out = (work - mean) / std
        stats.update({"mean": mean, "std": std})
    elif kind == "minmax":
        vmin = float(work[finite].min())
        vmax = float(work[finite].max())
        if vmax - vmin < 1e-8:
            raise ValueError("Global intensity range is too small for min-max normalization.")
        out = (work - vmin) / (vmax - vmin)
        stats.update({"min": vmin, "max": vmax})
    else:
        raise ValueError(f"Unsupported normalization type: {cfg.normalize_type}")

    out[~finite] = 0.0
    stats["scope"] = "all_voxels_all_phases_per_patient_sequence"
    stats["normalization"] = kind
    return out.astype(np.float32, copy=False), stats
