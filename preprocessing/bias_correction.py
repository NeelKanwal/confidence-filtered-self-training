"""Author: Neel Kanwal <neel.kanwal0@gmail.com>."""

from __future__ import annotations

import numpy as np
from .image_geometry import sitk_to_ants


def shared_n4_in_memory(arrays, reference, opts, to_sitk):
    import numpy as np
    import ants
    baseline = sitk_to_ants(to_sitk(arrays[0], reference.affine))
    try:
        mask = ants.get_mask(baseline)
    except Exception:
        mask = ants.from_numpy((baseline.numpy() > 0).astype('uint8'),
                              spacing=baseline.spacing, origin=baseline.origin, direction=baseline.direction)
    corrected = ants.n4_bias_field_correction(
        baseline, mask=mask, shrink_factor=opts['n4_shrink_factor'],
        convergence={'iters': list(opts['n4_iterations']), 'tol': 1e-7},
        rescale_intensities=True, verbose=False).numpy().astype(np.float32)
    raw = np.asarray(arrays[0], dtype=np.float32)
    multiplier = np.ones_like(raw)
    valid = np.abs(raw) > max(float(np.percentile(np.abs(raw), 1.)), 1e-6)
    multiplier[valid] = corrected[valid] / raw[valid]
    multiplier = np.clip(multiplier, .1, 10.)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(opts['phase_workers'],len(arrays)-1)) as pool:
        return [corrected] + list(pool.map(lambda a: np.asarray(a*multiplier,dtype=np.float32),arrays[1:]))
